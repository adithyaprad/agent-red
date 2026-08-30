"""A spec, turned into a running agent behind an HTTP endpoint.

This is the stand-in for a merchant agent platform. It takes the same `AgentSpec` every
other package reads, binds the declared tools to the implementations in `tools/`, and serves
the result on two endpoints: one that answers a consent challenge, and one that takes a turn
of conversation and returns the reply along with every tool the agent called while producing
it. Nothing here is part of the product surface. It exists so the harness has something
honest to attack.

Two properties matter and both are enforced rather than documented. Importing this module
asserts test mode, so an agent that can refund money cannot be started against live
credentials. And the tool set is checked against the spec at construction, because a
declared tool with no implementation is an agent that fails halfway through a conversation
and reads on the scorecard as a target that resisted.

Conversation state is held per session id, and the transcript itself is authoritative on the
runner's side. The target keeps only the session's private world and the model session it is
resuming, which is what lets the runner fork a conversation without the target's help.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentred.llm.client import agent_sdk_env, resolve_route
from agentred.spec import AgentSpec
from agentred.targets._guard import TEST_MODE, assert_test_mode
from agentred.targets.tools import TOOLSETS
from agentred.targets.tools.base import ToolSet
from agentred.targets.world import World, fresh_world

assert_test_mode()

MAX_TOOL_TURNS = 12
"""Ceiling on tool calls inside a single reply. A loop is a bug, not an attack result."""


class ChatMessage(BaseModel):
    """One turn of conversation, in the shape the runner sends it.

    Attributes:
        role: `user` or `assistant`.
        content: The text of the turn.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ChatRequest(BaseModel):
    """A request for the agent's next reply.

    Attributes:
        session: Identifies this conversation's private world. A session id the target has
            not seen starts a fresh world, which is how per-conversation isolation is
            obtained without the runner having to ask for it.
        conversation: The conversation so far, ending with the user turn to reply to. The
            runner holds the authoritative transcript; the target replays only the last turn.
    """

    model_config = ConfigDict(extra="forbid")

    session: str = Field(min_length=1)
    conversation: list[ChatMessage] = Field(min_length=1)


class ToolCall(BaseModel):
    """One tool the agent called while producing a reply.

    This is the record the deterministic detectors read. It carries the arguments as the
    model actually sent them, because a bound is checked against what was passed and not
    against what the agent said it passed.

    Attributes:
        name: The declared tool name, without any transport prefix.
        arguments: Arguments as supplied by the model.
        result: What the tool returned.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class ChatResponse(BaseModel):
    """What the target answers a turn with.

    Attributes:
        reply: The agent's text.
        tool_calls: Every tool called during this turn, in call order.
        spec_versions: The four versions this behaviour belongs to, so a transcript can
            never be attributed to a version of the agent that did not produce it.
        session: Echo of the session id.
        usage: What this turn cost the target, as its own model reported it. Empty from a
            backend that does not report it, which is not the same as free.
    """

    model_config = ConfigDict(extra="forbid")

    reply: str
    tool_calls: list[ToolCall]
    spec_versions: dict[str, str]
    session: str
    usage: dict[str, float] = {}


class ForkRequest(BaseModel):
    """A request to branch one conversation into another.

    Attributes:
        source: The session to branch from. Must exist.
        session: The new session id. Must not already exist, because silently reusing one
            would hand the branch somebody else's world.
        at_turn: How many completed exchanges the branch keeps. `None` branches from the
            end of the conversation.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    session: str = Field(min_length=1)
    at_turn: int | None = None


class ChallengeResponse(BaseModel):
    """The consent handshake, from the target's side.

    Attributes:
        challenge: The nonce, echoed unchanged.
        agent_id: Which agent is actually being served here.
        mode: Always `test` for a target that can move money.
    """

    model_config = ConfigDict(extra="forbid")

    challenge: str
    agent_id: str
    mode: str


@dataclass
class Checkpoint:
    """The state of a conversation at the end of one turn.

    Kept so a fork can branch from the middle of a conversation rather than only from its
    end. Without it, a branch taken at turn one would start from a world that turns two and
    three had already spent money in, and two branches would differ by more than the turn
    that was changed.

    Attributes:
        world: The shop as it stood after that turn.
        model_session_id: The model session the turn belonged to.
        message_uuid: The last assistant message of that turn, which is where the model
            session is rewound to when the branch is taken.
    """

    world: World
    model_session_id: str | None
    message_uuid: str | None


@dataclass
class Session:
    """One conversation's private state, held by the target.

    Attributes:
        world: This conversation's copy of the shop.
        model_session_id: The model session being resumed, so that turns after the first
            reuse the cached prefix instead of resending it.
        last_message_uuid: The most recent assistant message, recorded so a later fork can
            rewind the model session to this point.
        fork_pending: Set on a session copied from another. The next turn branches the model
            session rather than continuing it, so the two conversations share a cached
            prefix without sharing a future.
        fork_at: The message the branch rewinds to, when it was taken mid-conversation.
        calls: Tool calls made during the turn currently being served.
        usage: What the turn currently being served cost, as the backend reported it. Reset
            per turn alongside `calls`. Carried because a target's own spend is otherwise
            invisible to the harness: it happens inside the Agent SDK, and without it a run
            can only report half its bill.
        checkpoints: One entry per completed turn, in order.
    """

    world: World
    model_session_id: str | None = None
    last_message_uuid: str | None = None
    fork_pending: bool = False
    fork_at: str | None = None
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def checkpoint(self) -> None:
        """Record the state at the end of a turn."""
        self.checkpoints.append(
            Checkpoint(
                world=copy.deepcopy(self.world),
                model_session_id=self.model_session_id,
                message_uuid=self.last_message_uuid,
            )
        )

    def branch(self, at_turn: int | None = None) -> Session:
        """A branch of this session, taken after `at_turn` exchanges.

        The world is deep copied, so a refund issued in the branch does not appear in the
        conversation it was forked from, and a refund issued after the fork point does not
        appear in the branch. That is what makes two branches comparable: they differ by the
        turn that was changed and by nothing else.

        Args:
            at_turn: How many completed exchanges the branch keeps. `None` means all of
                them, which is the cheapest fork because it resumes the live prefix.

        Returns:
            A new session, not yet registered under any id.

        Raises:
            ValueError: If the session has not completed that many turns.
        """
        if at_turn is None:
            at_turn = len(self.checkpoints)
        if not 1 <= at_turn <= len(self.checkpoints):
            raise ValueError(
                f"cannot fork after {at_turn} turn(s) of a conversation with "
                f"{len(self.checkpoints)}"
            )
        point = self.checkpoints[at_turn - 1]
        return Session(
            world=copy.deepcopy(point.world),
            model_session_id=point.model_session_id,
            last_message_uuid=point.message_uuid,
            fork_pending=point.model_session_id is not None,
            fork_at=point.message_uuid,
            checkpoints=[copy.deepcopy(entry) for entry in self.checkpoints[:at_turn]],
        )


@runtime_checkable
class AgentBackend(Protocol):
    """What turns a conversation into a reply.

    An interface so the HTTP surface, the session isolation and the tool binding can all be
    tested offline against a scripted backend. The real one talks to a model; a fake one in
    `tests/fakes/` replays a recorded conversation.
    """

    async def reply(self, session: Session, conversation: list[ChatMessage]) -> str:
        """Produce the agent's next reply, calling tools against `session.world`.

        Args:
            session: The conversation's private state. Tool calls are appended to
                `session.calls` as they happen.
            conversation: The conversation so far, ending with the user turn to answer.

        Returns:
            The agent's text.
        """
        ...


class TargetAgent:
    """One agent under test: its spec, its tools and its sessions.

    Attributes:
        spec: The agent's config and policy.
        tools: The implementations behind the declared tools.
        backend: What produces replies.
        sessions: Live sessions by id.
    """

    def __init__(self, spec: AgentSpec, tools: ToolSet, backend: AgentBackend) -> None:
        """Bind a spec to its implementations.

        Args:
            spec: The validated spec.
            tools: The toolset for this agent.
            backend: What produces replies.

        Raises:
            ValueError: If the declared tools and the implemented tools are not the same
                set. Named in both directions: a declared tool with no implementation
                fails mid-conversation, and an implemented tool that is not declared is a
                capability no attack will ever aim at and no scorecard will mention.
        """
        declared = {tool.name for tool in spec.config.tools}
        implemented = tools.names
        if missing := sorted(declared - implemented):
            raise ValueError(
                f"{spec.config.agent_id} declares {missing} with no implementation behind them"
            )
        if extra := sorted(implemented - declared):
            raise ValueError(
                f"{spec.config.agent_id} implements {extra}, which its config does not declare, "
                f"so nothing will attack them and the scorecard will not mention them"
            )
        self.spec = spec
        self.tools = tools
        self.backend = backend
        self.sessions: dict[str, Session] = {}

    def session(self, session_id: str) -> Session:
        """The session's state, created with a fresh world the first time it is seen."""
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(world=fresh_world())
        return self.sessions[session_id]

    def call_tool(self, session: Session, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool for a session and record the call.

        The record is written here rather than read back off the model's message stream,
        because this is the only place that sees both the arguments as sent and the result
        as returned.
        """
        result = self.tools.call(name, session.world, arguments)
        session.calls.append(ToolCall(name=name, arguments=arguments, result=result))
        return result

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Answer one turn.

        Args:
            request: The session id and the conversation so far.

        Returns:
            The reply, the tool calls made while producing it, and the spec versions the
            behaviour belongs to.

        Raises:
            ValueError: If the conversation does not end with a user turn, which would mean
                the runner asked the agent to reply to itself.
        """
        if request.conversation[-1].role != "user":
            raise ValueError("the conversation must end with a user turn")

        session = self.session(request.session)
        session.calls = []
        session.usage = {}
        reply = await self.backend.reply(session, request.conversation)
        session.checkpoint()
        versions = self.spec.version_tuple
        return ChatResponse(
            reply=reply,
            tool_calls=list(session.calls),
            usage=dict(session.usage),
            spec_versions={
                "config": versions.config_version,
                "policy": versions.policy_version,
                "model": versions.model_version,
                "tools": versions.tool_version,
            },
            session=request.session,
        )

    def fork(self, source: str, session: str, at_turn: int | None = None) -> None:
        """Branch a session, so two attacks can share a prefix and differ after it.

        The runner uses this when it wants to try several continuations of the same opening:
        the prefix is paid for once, on both the model side and the merchant's side.

        Args:
            source: The session to branch from.
            session: The id for the branch.
            at_turn: How many completed exchanges the branch keeps.

        Raises:
            ValueError: If the source does not exist, the new id is already in use, or the
                source has not run that many turns. All three would silently give a
                conversation a world it did not earn.
        """
        if source not in self.sessions:
            raise ValueError(f"no session {source!r} to fork from")
        if session in self.sessions:
            raise ValueError(f"session {session!r} already exists")
        self.sessions[session] = self.sessions[source].branch(at_turn)

    def challenge(self, nonce: str) -> ChallengeResponse:
        """Answer a consent challenge.

        Echoing the nonce is the target's half of the gate in `runner/consent.py`. The agent
        id is returned with it so a registry entry cannot be quietly repointed at a
        different agent, and the mode is returned so the harness can refuse to attack
        anything that is not a test.
        """
        return ChallengeResponse(
            challenge=nonce, agent_id=self.spec.config.agent_id, mode=TEST_MODE
        )


class ClaudeAgentBackend:
    """The real backend: the Claude Agent SDK, with the spec's tools served in process.

    The agent sees exactly the tools its config declares, with the schemas its config
    declares, and nothing else. Sessions are resumed rather than replayed, so a six-turn
    conversation pays for its prefix once.

    Attributes:
        agent: The target this backend produces replies for. Set by `attach`, because the
            tool handlers need the agent and the agent needs the backend.
    """

    def __init__(self) -> None:
        """Build a backend, resolving the model route now rather than on the first turn.

        Resolving here means a target served against a route the Agent SDK cannot use, or
        with a region missing, fails before the socket opens. Discovering it on the first
        turn instead would put the failure inside a conversation, where it is indexed under
        the attack that happened to be running.

        Raises:
            LLMConfigurationError: If no route resolves, or the resolved route cannot serve
                an Agent SDK target.
        """
        self.agent: TargetAgent | None = None
        self.route = resolve_route()
        self.env = agent_sdk_env(self.route)

    def attach(self, agent: TargetAgent) -> None:
        """Bind this backend to the agent whose tools it should expose."""
        self.agent = agent

    def _server(self, session: Session) -> Any:
        """Build an in-process MCP server exposing this agent's tools for one session."""
        from claude_agent_sdk import create_sdk_mcp_server, tool

        agent = self._require_agent()
        tools = []
        for declaration in agent.spec.config.tools:
            tools.append(self._as_sdk_tool(tool, declaration, session))
        return create_sdk_mcp_server(name="shop", tools=tools)

    def _as_sdk_tool(self, decorator: Any, declaration: Any, session: Session) -> Any:
        """Wrap one declared tool as an SDK tool bound to one session's world."""
        agent = self._require_agent()
        name = declaration.name

        async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            result = agent.call_tool(session, name, dict(arguments))
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        return decorator(name, declaration.description, declaration.parameters)(handler)

    def _require_agent(self) -> TargetAgent:
        """The attached agent.

        Raises:
            RuntimeError: If `attach` was never called, which is a wiring bug rather than
                anything an operator can fix.
        """
        if self.agent is None:
            raise RuntimeError("ClaudeAgentBackend was never attached to a TargetAgent")
        return self.agent

    async def reply(self, session: Session, conversation: list[ChatMessage]) -> str:
        """Send the last user turn to the model and collect the reply."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
        )
        from claude_agent_sdk import TextBlock as SdkTextBlock

        agent = self._require_agent()
        server = self._server(session)
        options = ClaudeAgentOptions(
            system_prompt=agent.spec.config.instructions,
            model=self.route.model_id(agent.spec.config.model),
            env=self.env,
            mcp_servers={"shop": server},
            allowed_tools=[f"mcp__shop__{tool.name}" for tool in agent.spec.config.tools],
            permission_mode="bypassPermissions",
            max_turns=MAX_TOOL_TURNS,
            resume=session.model_session_id,
            fork_session=session.fork_pending,
            resume_session_at=session.fork_at,
            setting_sources=[],
        )
        session.fork_pending = False
        session.fork_at = None

        parts: list[str] = []
        async with ClaudeSDKClient(options=options) as client:
            await client.query(conversation[-1].content)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    parts.extend(
                        block.text for block in message.content if isinstance(block, SdkTextBlock)
                    )
                    if uuid := getattr(message, "uuid", None):
                        session.last_message_uuid = uuid
                if isinstance(message, ResultMessage):
                    session.usage = _usage_of(message)
                session_id = getattr(message, "session_id", None)
                if session_id:
                    session.model_session_id = session_id
        return "".join(parts).strip()


def _usage_of(result: Any) -> dict[str, float]:
    """Flatten what one turn cost out of the SDK's result message.

    An agent turn is not one model call. It is however many the agent needed to read its
    tools and answer, which is exactly why the harness cannot infer this from the outside and
    has to be told.

    Args:
        result: The SDK `ResultMessage` closing a turn.

    Returns:
        Token counts, and `cost_usd` when the SDK priced the turn. It does not always: on a
        route where it does not hold the price list, the cost is absent rather than zero, and
        the difference matters because zero is a claim and absent is not.
    """
    usage = getattr(result, "usage", None) or {}
    if not isinstance(usage, dict):
        usage = getattr(usage, "__dict__", {})
    flat = {
        "input_tokens": float(usage.get("input_tokens", 0) or 0),
        "output_tokens": float(usage.get("output_tokens", 0) or 0),
        "cache_read_tokens": float(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_write_tokens": float(usage.get("cache_creation_input_tokens", 0) or 0),
        "model_turns": float(getattr(result, "num_turns", 0) or 0),
    }
    cost = getattr(result, "total_cost_usd", None)
    if cost is not None:
        flat["cost_usd"] = float(cost)
    return flat


def build_agent(spec: AgentSpec, backend: AgentBackend | None = None) -> TargetAgent:
    """Assemble a target from its spec.

    Args:
        spec: The validated spec.
        backend: What produces replies. Defaults to the real Claude Agent SDK backend.

    Returns:
        A ready `TargetAgent`.

    Raises:
        KeyError: If no toolset is registered for the spec's agent id. A spec without
            implementations is a document, not a target.
    """
    agent_id = spec.config.agent_id
    if agent_id not in TOOLSETS:
        raise KeyError(
            f"no tool implementations registered for agent {agent_id!r}; "
            f"registered: {sorted(TOOLSETS)}"
        )
    backend = ClaudeAgentBackend() if backend is None else backend
    agent = TargetAgent(spec=spec, tools=TOOLSETS[agent_id], backend=backend)
    if isinstance(backend, ClaudeAgentBackend):
        backend.attach(agent)
    return agent


def build_app(agent: TargetAgent) -> Any:
    """Serve one target over HTTP.

    Two endpoints and a health check. `GET /challenge` is the target's half of the consent
    gate; `POST /chat` takes a turn and returns the reply with its tool-call log.

    Args:
        agent: The target to serve.

    Returns:
        A FastAPI application.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title=f"agent-red target: {agent.spec.config.agent_id}")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "agent_id": agent.spec.config.agent_id, "mode": TEST_MODE}

    @app.get("/challenge")
    async def challenge(nonce: str) -> ChallengeResponse:
        return agent.challenge(nonce)

    @app.post("/fork")
    async def fork(request: ForkRequest) -> dict[str, str]:
        try:
            agent.fork(request.source, request.session, request.at_turn)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"session": request.session, "forked_from": request.source}

    @app.post("/chat")
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            return await agent.chat(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app
