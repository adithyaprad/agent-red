"""The tool server: the seam the whole harness meets at.

Every capability an agent under test can reach is served from here, over MCP, and every
invocation is recorded here with its full arguments before the answer goes back. That is the
decision in ADR-0005 made concrete: the agent is never asked what it did, because what it did
was observed on the way through.

**How a call is attributed.** The run and the session are carried in the URL the agent's tool
connector is pointed at, `/{agent_id}/{run}/{session}`, and not in the arguments of a call. An
agent cannot omit them, cannot rename them, and cannot address a session other than the one it
was handed, because the only thing it is given is a URL. The recorder and the arena are then
keyed by what the path said rather than by anything the agent asserted.

**Two faces, on two ports.** The tool face is what agents connect to. The control face is what
the runner uses: read the call stream, checkpoint a world at a turn boundary, branch it for a
fork, restore it to a baseline, plant a payload into it. Splitting them is what keeps an agent
from restoring the world it just spent money in. It is a property of what each process is
handed rather than a secret to keep, so there is no token to leak and no configuration that
can quietly turn it off. The agent is told the tool URL and nothing else.

**No tool here enforces policy.** `apply_discount(percent=35)` succeeds against a ceiling of
10. A tool that refused would answer the question the suite exists to ask, and the run would
prove nothing. What the server does is watch.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentred.mcp._guard import TEST_MODE, assert_test_mode
from agentred.mcp.arena import (
    Arena,
    ArenaError,
    PlantError,
    UnknownSessionError,
    UnknownSourceError,
)
from agentred.mcp.recorder import ToolCallRecorder
from agentred.mcp.tools import TOOLSETS
from agentred.mcp.tools.base import ToolSet
from agentred.mcp.tools.generic import UndeclaredToolError, toolset_for
from agentred.spec import AgentSpec, VersionTuple

assert_test_mode()

DEFAULT_TOOL_PORT = 8090
DEFAULT_CONTROL_PORT = 8091
MCP_PATH_TEMPLATE = "/{agent_id}/{run}/{session}"
"""What an agent's connector is pointed at. Everything about attribution follows from it."""


@dataclass(frozen=True)
class Binding:
    """Who a call belongs to, taken from the URL the connector was pointed at.

    Attributes:
        agent_id: Which agent's declared tool surface is being served.
        run: The run the call is recorded under.
        session: The conversation, or planted attempt, whose world the call acts on.
    """

    agent_id: str
    run: str
    session: str


_BINDING: ContextVar[Binding | None] = ContextVar("agentred_binding", default=None)
"""Set per request by the ASGI wrapper, read by the tool handlers."""


class ToolServerError(RuntimeError):
    """The server was asked for something it cannot honestly serve."""


class BranchRequest(BaseModel):
    """Give a new session the source's world as it stood after `at_turn` turns.

    Attributes:
        source: The conversation to branch from.
        session: The id for the branch. Must not already have a world.
        at_turn: How many completed turns the branch keeps. `None` takes the latest.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    session: str = Field(min_length=1)
    at_turn: int | None = None


class PlantRequest(BaseModel):
    """Write attacker-controlled text into a field a customer genuinely fills in.

    Attributes:
        session: Whose world to write into.
        source: The declared data source to write into. Resolved against the world's own
            map, so a generated shop's own collection names need nothing changed here.
        record_id: The record within it.
        field_name: The field to overwrite. Must already exist.
        payload: The text to write.
    """

    model_config = ConfigDict(extra="forbid")

    session: str = Field(min_length=1)
    source: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    field_name: str = Field(min_length=1)
    payload: str


class ToolServer:
    """The tool surface, the worlds it acts on, and the record of what was called.

    One server can serve several agents, each seeing exactly the tools its own config
    declares with the schemas its own config declares. Serving them together is what makes
    a single recorded stream the whole account of a run.

    Attributes:
        specs: Agent specs by agent id.
        toolsets: Implementations by agent id.
        arena: The worlds, by session.
        recorder: The append-only call stream.
    """

    def __init__(
        self,
        specs: Sequence[AgentSpec],
        *,
        arena: Arena | None = None,
        recorder: ToolCallRecorder | None = None,
    ) -> None:
        """Bind specs to implementations.

        Args:
            specs: The agents to serve tools for. At least one.
            arena: Where worlds live. Defaults to a fresh one.
            recorder: Where calls are written. Defaults to an in-memory stream.

        Raises:
            ToolServerError: If no specs are given, an agent id repeats, or a spec declares
                a tool with no implementation behind it. Named in both directions, as the
                target used to check: a declared tool with no implementation fails halfway
                through a conversation and reads on the scorecard as an agent that resisted,
                and an implemented tool that is not declared is a capability no attack will
                aim at and no scorecard will mention.
        """
        if not specs:
            raise ToolServerError("a tool server with no agents serves nothing")
        self.specs: dict[str, AgentSpec] = {}
        self.toolsets: dict[str, ToolSet] = {}
        for spec in specs:
            agent_id = spec.config.agent_id
            if agent_id in self.specs:
                raise ToolServerError(f"{agent_id!r} was given to the tool server twice")
            # A hand-written surface when one exists, otherwise the surface the agent's own
            # declaration describes. The two shipped agents keep theirs, because they are the
            # fixture a generic one is checked against; an agent nobody wrote code for is
            # served from what its merchant declared and needs nothing registered here.
            toolset = TOOLSETS.get(agent_id)
            if toolset is None:
                try:
                    toolset = toolset_for(spec)
                except UndeclaredToolError as error:
                    raise ToolServerError(
                        f"no tool implementations registered for agent {agent_id!r} and its "
                        f"declaration does not describe them either: {error}"
                    ) from error
            declared = {tool.name for tool in spec.config.tools}
            if missing := sorted(declared - toolset.names):
                raise ToolServerError(
                    f"{agent_id} declares {missing} with no implementation behind them"
                )
            if extra := sorted(toolset.names - declared):
                raise ToolServerError(
                    f"{agent_id} implements {extra}, which its config does not declare, so "
                    f"nothing will attack them and the scorecard will not mention them"
                )
            self.specs[agent_id] = spec
            self.toolsets[agent_id] = toolset
        self.arena = arena if arena is not None else Arena()
        self.recorder = recorder if recorder is not None else ToolCallRecorder()

    @property
    def world_version(self) -> str:
        """A content hash of the shop this process is serving.

        The fifth element of the validity tuple (ADR-0007), and the only one the spec files
        cannot supply, because a world is not a property of a declaration. It comes from here
        because this is the process that holds the world: a scorecard computed against one
        shop says nothing about an agent facing another, and the day the shop was rebuilt
        every earlier scorecard went on citing a tuple that no longer described what the
        agent had faced.
        """
        return self.arena.seed_world().digest

    def versions(self, agent_id: str) -> VersionTuple:
        """The five versions a result against this server is valid for."""
        return self.spec(agent_id).version_tuple.model_copy(
            update={"world_version": self.world_version}
        )

    @property
    def agent_ids(self) -> tuple[str, ...]:
        """Every agent this server serves tools for, in registration order."""
        return tuple(self.specs)

    def spec(self, agent_id: str) -> AgentSpec:
        """The spec for one agent.

        Raises:
            ToolServerError: If this server does not serve that agent. An agent pointed at a
                server that does not know it is a misconfiguration, and answering with an
                empty tool list would look like an agent that chose to do nothing.
        """
        spec = self.specs.get(agent_id)
        if spec is None:
            raise ToolServerError(
                f"this tool server does not serve {agent_id!r}; it serves "
                f"{', '.join(self.agent_ids)}"
            )
        return spec

    def call(self, binding: Binding, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool for one session, and record it.

        The recording happens here, in the same place that produced the result, which is the
        only place that sees both the arguments as they arrived and the result as it was
        returned. A call to a name the agent's config does not declare is recorded too, and
        answered with an error. That attempt is itself a finding, and dropping it would hide
        exactly the call a "reached a tool it was not given" check exists to catch.

        Args:
            binding: Agent, run and session, from the URL.
            name: The declared tool name.
            arguments: Arguments as they arrived.

        Returns:
            The tool result.

        Raises:
            ToolServerError: If the agent is not served here.
        """
        spec = self.spec(binding.agent_id)
        toolset = self.toolsets[spec.config.agent_id]
        world = self.arena.world(binding.session)
        result = toolset.call(name, world, arguments)
        self.recorder.record(
            run=binding.run,
            session=binding.session,
            name=name,
            arguments=arguments,
            result=result,
        )
        return result


def _binding_or_fail() -> Binding:
    """The binding for the request being served.

    Raises:
        ToolServerError: If a tool was reached with no binding set, which can only happen if
            the MCP app was mounted somewhere that does not carry the run and session. That
            is a wiring bug, and serving the call anyway would record it against nothing.
    """
    binding = _BINDING.get()
    if binding is None:
        raise ToolServerError(
            "a tool was called with no run and session bound. The MCP app must be mounted "
            f"at {MCP_PATH_TEMPLATE}, so that every call can be attributed."
        )
    return binding


def build_tool_app(server: ToolServer) -> Any:
    """The face agents connect to: the declared tools, over MCP streamable HTTP.

    Mounted so that the run and the session come from the path. Nothing else is exposed
    here, and in particular nothing here can read the call stream or restore a world.

    Args:
        server: The tool server to serve.

    Returns:
        A Starlette application.
    """
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp_types import CallToolResult, ListToolsResult, TextContent, Tool
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    async def on_list_tools(ctx: Any, params: Any) -> ListToolsResult:
        binding = _binding_or_fail()
        spec = server.spec(binding.agent_id)
        return ListToolsResult(
            tools=[
                Tool(
                    name=declaration.name,
                    description=declaration.description,
                    inputSchema=declaration.parameters,
                )
                for declaration in spec.config.tools
            ]
        )

    async def on_call_tool(ctx: Any, params: Any) -> CallToolResult:
        binding = _binding_or_fail()
        result = server.call(binding, params.name, dict(params.arguments or {}))
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])

    low = Server(
        name="agentred-tools",
        instructions="The merchant's tools. Calls are recorded.",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    manager = StreamableHTTPSessionManager(app=low, stateless=True, json_response=True)

    async def _not_found(scope: Scope, receive: Receive, send: Send) -> None:
        """Answer a URL that does not name an agent, a run and a session."""
        from starlette.responses import JSONResponse

        response = JSONResponse(
            {
                "detail": (
                    f"a tool connector must be pointed at {MCP_PATH_TEMPLATE}, so that every "
                    f"call can be attributed to a run and a session"
                )
            },
            status_code=404,
        )
        await response(scope, receive, send)

    class BoundTransport:
        """Reads the binding out of the path, then serves the MCP request under it.

        The path is parsed here rather than by a router with placeholders, because a router
        answers the path without a trailing slash with a redirect, and an MCP client that
        does not follow one fails with a content-type error that says nothing about why.
        """

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            parts = [part for part in str(scope.get("path", "")).split("/") if part]
            if len(parts) != 3:
                await _not_found(scope, receive, send)
                return
            binding = Binding(agent_id=parts[0], run=parts[1], session=parts[2])
            token = _BINDING.set(binding)
            try:
                await manager.handle_request(scope, receive, send)
            finally:
                _BINDING.reset(token)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(lifespan=lifespan, routes=[Mount("/", app=BoundTransport())])


def build_control_app(server: ToolServer) -> Any:
    """The face the runner uses, on its own port, never handed to an agent.

    Reading the stream, checkpointing a world, branching it, restoring it and planting into
    it are all operations that would let an agent rewrite the evidence about itself. They
    are reachable only from a port the agent is never told about.

    Args:
        server: The tool server to control.

    Returns:
        A FastAPI application.
    """
    from fastapi import FastAPI, HTTPException, Query

    app = FastAPI(title="agent-red tool server (control)")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """What this process is serving, and which spec it is serving it from.

        The versions are here for the same reason the target reports them at the challenge: a
        server reads its specs once and holds them, so a spec edited afterwards leaves this
        process serving the previous tool surface while every check reads the new file. The
        digest is the part that matters, because it is derived from the declared tools and is
        the thing an agent's connector is actually shaped by.
        """
        return {
            "status": "ok",
            "mode": TEST_MODE,
            "agents": list(server.agent_ids),
            "versions": {
                agent_id: server.versions(agent_id).model_dump(mode="json")
                for agent_id in server.agent_ids
            },
        }

    @app.get("/calls/{run}/{session}")
    async def calls(run: str, session: str) -> dict[str, Any]:
        recorded = server.recorder.calls(run, session)
        return {
            "run": run,
            "session": session,
            "calls": [
                {
                    "run": record.run,
                    "session": record.session,
                    "sequence": record.sequence,
                    "name": record.name,
                    "arguments": record.arguments,
                    "result": record.result,
                    "at": record.at,
                }
                for record in recorded
            ],
        }

    @app.post("/sessions/{session}/checkpoint")
    async def checkpoint(session: str) -> dict[str, Any]:
        # A conversation in which the agent called nothing has no world yet, and that is a
        # real outcome rather than a missing session. Seeding it here keeps a fork of such a
        # conversation possible, and keeps the turn count equal to the turns that happened.
        server.arena.world(session)
        turns = server.arena.checkpoint(session)
        return {"session": session, "turns": turns}

    @app.post("/sessions/branch")
    async def branch(request: BranchRequest) -> dict[str, Any]:
        try:
            server.arena.branch(request.source, request.session, request.at_turn)
        except UnknownSessionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ArenaError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"session": request.session, "branched_from": request.source}

    @app.post("/sessions/{session}/restore")
    async def restore(session: str) -> dict[str, Any]:
        server.arena.restore(session)
        return {"session": session, "restored": True}

    @app.post("/sessions/{session}/forget")
    async def forget(session: str) -> dict[str, Any]:
        server.arena.forget(session)
        return {"session": session, "forgotten": True}

    @app.get("/subjects/{session}")
    # `kind` is declared with `Query` as its default rather than inside `Annotated`. This
    # module has postponed annotations on, so an annotation is a string by the time FastAPI
    # reads it, and a nested `Annotated` in a string is a forward reference it cannot
    # resolve. It fails at request time rather than at import, so only a request catches it.
    async def subjects(
        session: str,
        source: str,
        kind: list[str] = Query(default=[]),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            found = server.arena.subjects(session, source=source, kinds=tuple(kind))
        except UnknownSessionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ArenaError, UnknownSourceError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"session": session, "source": source, "subjects": list(found)}

    @app.post("/plant")
    async def plant(request: PlantRequest) -> dict[str, Any]:
        try:
            previous = server.arena.plant(
                request.session,
                source=request.source,
                record_id=request.record_id,
                field_name=request.field_name,
                payload=request.payload,
            )
        except (PlantError, UnknownSourceError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "session": request.session,
            "source": request.source,
            "record_id": request.record_id,
            "field_name": request.field_name,
            "replaced": previous,
        }

    return app


def build_server(spec_dirs: Sequence[Path], *, stream: Path | None = None) -> ToolServer:
    """Load specs from disk and build the server that serves their tools.

    Args:
        spec_dirs: Directories each holding `config.yaml` and `policy.yaml`.
        stream: Where to persist the call stream, as JSON lines. `None` keeps it in memory,
            which is enough when the runner reads it over the control face.

    Returns:
        The server.

    Raises:
        SpecError: If a spec does not load.
        ToolServerError: If a spec has no implementations behind it.
    """
    from agentred.spec import load_spec_dir

    specs = [load_spec_dir(directory) for directory in spec_dirs]
    return ToolServer(specs, recorder=ToolCallRecorder(path=stream))
