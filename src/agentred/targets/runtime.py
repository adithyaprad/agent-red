"""A spec, turned into a running agent behind an HTTP endpoint.

This is the stand-in for a merchant agent platform. It takes the same `AgentSpec` every
other package reads, points the agent's tool connector at the tool server, and serves the
result on two endpoints: one that answers a consent challenge, and one that takes a turn of
conversation and returns the reply. Nothing here is part of the product surface. It exists
so the harness has something honest to attack.

**A target says what it said, and nothing about what it did.** The reply carries prose and
the spec versions it belongs to. It does not carry a tool-call log, because a log the
measured party volunteers is a self-report rather than evidence, and requiring one would
mean requiring every agent runtime to be modified before it could be measured. The calls are
recorded at the tool server, on the harness's side of the trust line (ADR-0005). A target
that wanted to lie about its behaviour would have to lie to the tool server, which means not
calling the tool, which means not having the effect.

What the target still owns is the model session: the cached prefix a six-turn conversation is
resumed against, and the message a fork rewinds to. The world the tools act on is not here
either. It lives in the tool server's arena, keyed by the same session id, so a fork branches
in two places for two different reasons.

Importing this module asserts test mode, so an agent that can refund money cannot be started
against live credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentred.llm.client import agent_sdk_env, resolve_route
from agentred.mcp._guard import TEST_MODE, assert_test_mode
from agentred.mcp.server import DEFAULT_TOOL_PORT
from agentred.spec import AgentSpec

assert_test_mode()

MAX_TOOL_TURNS = 12
"""Ceiling on tool calls inside a single reply. A loop is a bug, not an attack result."""

TOOL_SERVER_ENV_VAR = "AGENTRED_TOOL_SERVER_URL"
DEFAULT_TOOL_SERVER_URL = f"http://localhost:{DEFAULT_TOOL_PORT}"
"""Where the target reaches its tools.

Reported when the target answers a challenge, so a run can refuse a target that is about to
call tools nobody is recording rather than produce a suite of empty call streams.
"""

TOOL_CONNECTOR_NAME = "shop"
"""What the agent's tool connector is called. Tools reach the model as `mcp__shop__<name>`."""


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
        session: Identifies this conversation. The tool server keys the conversation's
            private world and its call stream by the same id, which is how isolation is
            obtained without the runner having to ask for it.
        run: The run the calls made while answering are recorded under. Sent by the runner
            because the target has no way to know which run it is part of.
        conversation: The conversation so far, ending with the user turn to reply to. The
            runner holds the authoritative transcript; the target replays only the last turn.
    """

    model_config = ConfigDict(extra="forbid")

    session: str = Field(min_length=1)
    run: str = Field(min_length=1)
    conversation: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    """What the target answers a turn with.

    There is deliberately no tool-call field. What the agent did is read from the tool
    server's record. See ADR-0005.

    Attributes:
        reply: The agent's text.
        spec_versions: The four versions this behaviour belongs to, so a transcript can
            never be attributed to a version of the agent that did not produce it.
        session: Echo of the session id.
        usage: What this turn cost the target, as its own model reported it. Empty from a
            backend that does not report it, which is not the same as free.
    """

    model_config = ConfigDict(extra="forbid")

    reply: str
    spec_versions: dict[str, str]
    session: str
    usage: dict[str, float] = {}


class ForkRequest(BaseModel):
    """A request to branch one conversation into another.

    Attributes:
        source: The session to branch from. Must exist.
        session: The new session id. Must not already exist, because silently reusing one
            would hand the branch somebody else's cached prefix.
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
        tool_server: Where this target will reach its tools. Compared with the registry
            before the first attack turn, because a target calling tools at a server the run
            is not recording produces a run that quietly measures nothing.
    """

    model_config = ConfigDict(extra="forbid")

    challenge: str
    agent_id: str
    mode: str
    tool_server: str = ""


@dataclass
class Checkpoint:
    """The model session at the end of one turn.

    Kept so a fork can branch from the middle of a conversation rather than only from its
    end. The world at that turn is checkpointed separately, in the tool server's arena: the
    two halves of a fork are the cached prefix, which is the target's, and the money spent so
    far, which is not.

    Attributes:
        model_session_id: The model session the turn belonged to.
        message_uuid: The last assistant message of that turn, which is where the model
            session is rewound to when the branch is taken.
    """

    model_session_id: str | None
    message_uuid: str | None


@dataclass
class Session:
    """One conversation's model-side state, held by the target.

    Attributes:
        session_id: The id the runner minted, and the same id the tool server keys this
            conversation's world and call stream by.
        run: The run this conversation belongs to, as the runner reported it on the turn
            being served.
        model_session_id: The model session being resumed, so that turns after the first
            reuse the cached prefix instead of resending it.
        last_message_uuid: The most recent assistant message, recorded so a later fork can
            rewind the model session to this point.
        fork_pending: Set on a session copied from another. The next turn branches the model
            session rather than continuing it, so the two conversations share a cached prefix
            without sharing a future.
        fork_at: The message the branch rewinds to, when it was taken mid-conversation.
        usage: What the turn currently being served cost, as the backend reported it. Reset
            per turn. Carried because a target's own spend is otherwise invisible to the
            harness: it happens inside the Agent SDK, and without it a run can only report
            half its bill.
        checkpoints: One entry per completed turn, in order.
    """

    session_id: str
    run: str = ""
    model_session_id: str | None = None
    last_message_uuid: str | None = None
    fork_pending: bool = False
    fork_at: str | None = None
    usage: dict[str, float] = field(default_factory=dict)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def checkpoint(self) -> None:
        """Record the model session at the end of a turn."""
        self.checkpoints.append(
            Checkpoint(
                model_session_id=self.model_session_id,
                message_uuid=self.last_message_uuid,
            )
        )

    def branch(self, session_id: str, at_turn: int | None = None) -> Session:
        """A branch of this session, taken after `at_turn` exchanges.

        Args:
            session_id: The id for the branch.
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
            session_id=session_id,
            run=self.run,
            model_session_id=point.model_session_id,
            last_message_uuid=point.message_uuid,
            fork_pending=point.model_session_id is not None,
            fork_at=point.message_uuid,
            checkpoints=[
                Checkpoint(entry.model_session_id, entry.message_uuid)
                for entry in self.checkpoints[:at_turn]
            ],
        )


@runtime_checkable
class AgentBackend(Protocol):
    """What turns a conversation into a reply.

    An interface so the HTTP surface, the session handling and the connector wiring can all
    be tested offline against a scripted backend. The real one talks to a model; a fake one
    in `tests/fakes/` replays a recorded conversation.
    """

    async def reply(self, session: Session, conversation: list[ChatMessage]) -> str:
        """Produce the agent's next reply, calling tools at the tool server.

        Args:
            session: The conversation's model-side state.
            conversation: The conversation so far, ending with the user turn to answer.

        Returns:
            The agent's text.
        """
        ...


class TargetAgent:
    """One agent under test: its spec, where its tools live, and its sessions.

    Attributes:
        spec: The agent's config and policy.
        backend: What produces replies.
        tool_server_url: Origin of the tool server's tool face. The agent's connector is
            pointed at a path under it naming the agent, the run and the session, so every
            call it makes is attributable without the agent being asked anything.
        sessions: Live sessions by id.
    """

    def __init__(
        self, spec: AgentSpec, backend: AgentBackend, tool_server_url: str | None = None
    ) -> None:
        """Bind a spec to a backend.

        The declared tools are not checked against implementations here any more. That check
        belongs to the tool server, which is the thing that has implementations to check.

        Args:
            spec: The validated spec.
            backend: What produces replies.
            tool_server_url: Where the tools are served. Defaults to
                `AGENTRED_TOOL_SERVER_URL`, then to localhost on the tool server's port.
        """
        self.spec = spec
        self.backend = backend
        self.tool_server_url = (
            tool_server_url
            if tool_server_url is not None
            else os.environ.get(TOOL_SERVER_ENV_VAR, DEFAULT_TOOL_SERVER_URL)
        ).rstrip("/")
        self.sessions: dict[str, Session] = {}

    def session(self, session_id: str) -> Session:
        """The session's model-side state, created the first time the session is seen."""
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def connector_url(self, session: Session) -> str:
        """Where this session's tool calls go.

        The agent, the run and the session are in the path, so the tool server can attribute
        every call without trusting anything the agent sends it.
        """
        agent_id = self.spec.config.agent_id
        return f"{self.tool_server_url}/{agent_id}/{session.run}/{session.session_id}"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Answer one turn.

        Args:
            request: The session id, the run, and the conversation so far.

        Returns:
            The reply and the spec versions the behaviour belongs to. What the agent did
            while producing it is read from the tool server, not from here.

        Raises:
            ValueError: If the conversation does not end with a user turn, which would mean
                the runner asked the agent to reply to itself.
        """
        if request.conversation[-1].role != "user":
            raise ValueError("the conversation must end with a user turn")

        session = self.session(request.session)
        session.run = request.run
        session.usage = {}
        reply = await self.backend.reply(session, request.conversation)
        session.checkpoint()
        versions = self.spec.version_tuple
        return ChatResponse(
            reply=reply,
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
        """Branch a session's cached prefix, so two attacks can share it and differ after it.

        This is half of a fork. The other half is branching the world, which the runner asks
        the tool server for, because a world an agent could branch is a world an agent could
        rewind.

        Args:
            source: The session to branch from.
            session: The id for the branch.
            at_turn: How many completed exchanges the branch keeps.

        Raises:
            ValueError: If the source does not exist, the new id is already in use, or the
                source has not run that many turns.
        """
        if source not in self.sessions:
            raise ValueError(f"no session {source!r} to fork from")
        if session in self.sessions:
            raise ValueError(f"session {session!r} already exists")
        self.sessions[session] = self.sessions[source].branch(session, at_turn)

    def challenge(self, nonce: str) -> ChallengeResponse:
        """Answer a consent challenge.

        Echoing the nonce is the target's half of the gate in `runner/consent.py`. The agent
        id is returned with it so a registry entry cannot be quietly repointed at a different
        agent, the mode so the harness can refuse to attack anything that is not a test, and
        the tool server so the harness can refuse a target whose calls would land somewhere
        it is not reading.
        """
        return ChallengeResponse(
            challenge=nonce,
            agent_id=self.spec.config.agent_id,
            mode=TEST_MODE,
            tool_server=self.tool_server_url,
        )


class ClaudeAgentBackend:
    """The real backend: the Claude Agent SDK, reaching its tools over an MCP connector.

    The agent sees exactly the tools its config declares, with the schemas its config
    declares, because the tool server serves them from the same spec. Sessions are resumed
    rather than replayed, so a six-turn conversation pays for its prefix once.

    Attributes:
        agent: The target this backend produces replies for. Set by `attach`, because the
            connector URL depends on the agent and the agent needs the backend.
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
        """Bind this backend to the agent it produces replies for."""
        self.agent = agent

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
        options = ClaudeAgentOptions(
            system_prompt=agent.spec.config.instructions,
            model=self.route.model_id(agent.spec.config.model),
            env=self.env,
            mcp_servers={
                TOOL_CONNECTOR_NAME: {"type": "http", "url": agent.connector_url(session)}
            },
            allowed_tools=[
                f"mcp__{TOOL_CONNECTOR_NAME}__{tool.name}" for tool in agent.spec.config.tools
            ],
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


def build_agent(
    spec: AgentSpec,
    backend: AgentBackend | None = None,
    tool_server_url: str | None = None,
) -> TargetAgent:
    """Assemble a target from its spec.

    Args:
        spec: The validated spec.
        backend: What produces replies. Defaults to the real Claude Agent SDK backend.
        tool_server_url: Where the tools are served. Defaults to the environment.

    Returns:
        A ready `TargetAgent`.
    """
    backend = ClaudeAgentBackend() if backend is None else backend
    agent = TargetAgent(spec=spec, backend=backend, tool_server_url=tool_server_url)
    if isinstance(backend, ClaudeAgentBackend):
        backend.attach(agent)
    return agent


def build_app(agent: TargetAgent) -> Any:
    """Serve one target over HTTP.

    Two endpoints and a health check. `GET /challenge` is the target's half of the consent
    gate; `POST /chat` takes a turn and returns the reply.

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
