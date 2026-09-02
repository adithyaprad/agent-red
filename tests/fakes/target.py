"""A target backend that follows a script instead of calling a model.

Lets the HTTP surface, the session handling, the recorder and the driver all be tested
offline. The script is a list of turns; each turn names the tools to call and the text to
reply with, which is enough to reproduce every shape a real conversation produces without
reproducing the model.

The scripted backend calls tools the way a real one does: at the tool server, which records
them. It skips only the MCP hop, which has its own tests in `tests/mcp/test_server.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentred.mcp.recorder import RecordedCall
from agentred.mcp.server import Binding, ToolServer
from agentred.targets.runtime import ChatMessage, Session, TargetAgent


@dataclass
class ScriptedTurn:
    """One scripted reply.

    Attributes:
        reply: The text to answer with.
        calls: Tools to call first, as `(name, arguments)` pairs.
    """

    reply: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class ScriptedBackend:
    """Replays scripted turns in order, calling real tools at a real tool server.

    Attributes:
        turns: The script. The last turn repeats if the conversation runs past it.
        server: Where the tools live and where the calls are recorded.
        agent: The target this backend replies for. Set by `attach`.
        seen: Every conversation this backend was asked to reply to, for assertions about
            what the runner actually sent.
    """

    def __init__(self, *turns: ScriptedTurn, server: ToolServer) -> None:
        self.turns = list(turns) or [ScriptedTurn(reply="")]
        self.server = server
        self.agent: TargetAgent | None = None
        self.seen: list[list[ChatMessage]] = []

    def attach(self, agent: TargetAgent) -> None:
        self.agent = agent

    async def reply(self, session: Session, conversation: list[ChatMessage]) -> str:
        assert self.agent is not None, "ScriptedBackend was never attached"
        self.seen.append(list(conversation))
        turn = self.turns[min(len(self.seen) - 1, len(self.turns) - 1)]
        binding = Binding(
            agent_id=self.agent.spec.config.agent_id,
            run=session.run,
            session=session.session_id,
        )
        for name, arguments in turn.calls:
            self.server.call(binding, name, arguments)
        return turn.reply


class ScriptedTriggerBackend(ScriptedBackend):
    """A scripted backend that also answers a scheduled firing.

    Separate from `ScriptedBackend` so a test can still build a target with no scheduled
    entry point and assert that firing one is refused rather than silently answered.

    Attributes:
        firings: How many times the entry point was fired.
    """

    def __init__(self, *turns: ScriptedTurn, server: ToolServer) -> None:
        super().__init__(*turns, server=server)
        self.firings = 0

    async def trigger(self, session: Session) -> str:
        assert self.agent is not None, "ScriptedTriggerBackend was never attached"
        self.firings += 1
        turn = self.turns[min(self.firings - 1, len(self.turns) - 1)]
        binding = Binding(
            agent_id=self.agent.spec.config.agent_id,
            run=session.run,
            session=session.session_id,
        )
        for name, arguments in turn.calls:
            self.server.call(binding, name, arguments)
        session.usage = {"input_tokens": 1.0, "output_tokens": 1.0}
        return turn.reply


class InProcessArenaControl:
    """The control face, reached directly rather than over HTTP.

    Same operations the runner performs in production, against the same arena and the same
    recorder, with the socket taken out.
    """

    def __init__(self, server: ToolServer) -> None:
        self.server = server

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "agents": list(self.server.agent_ids)}

    def calls(self, run: str, session: str) -> tuple[RecordedCall, ...]:
        return self.server.recorder.calls(run, session)

    def checkpoint(self, session: str) -> int:
        self.server.arena.world(session)
        return self.server.arena.checkpoint(session)

    def branch(self, source: str, session: str, at_turn: int | None = None) -> None:
        self.server.arena.branch(source, session, at_turn)

    def restore(self, session: str) -> None:
        self.server.arena.restore(session)

    def plant(
        self, session: str, *, collection: str, record_id: str, field_name: str, payload: str
    ) -> str:
        return self.server.arena.plant(
            session,
            collection=collection,
            record_id=record_id,
            field_name=field_name,
            payload=payload,
        )

    def subjects(
        self, session: str, *, collection: str, kinds: tuple[str, ...]
    ) -> tuple[dict[str, str], ...]:
        return self.server.arena.subjects(session, collection=collection, kinds=kinds)


class InProcessTransport:
    """Sends turns straight to a `TargetAgent`, with no socket in between.

    Lets the driver be tested against a real target: real tool implementations, real
    per-session worlds, real recorded calls. Only the HTTP hop and the model are absent, and
    both have their own tests.

    Attributes:
        agent: The target to send to.
        tokens: Every consent token it was handed, so a test can assert that the driver
            never sends a turn without one.
    """

    def __init__(self, agent: TargetAgent) -> None:
        self.agent = agent
        self.tokens: list[object] = []

    def send(
        self, token: Any, session: str, run: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        import asyncio

        from agentred.targets.runtime import ChatRequest

        token.require_live()
        self.tokens.append(token)
        request = ChatRequest(session=session, run=run, conversation=conversation)
        return asyncio.run(self.agent.chat(request)).model_dump()

    def fork(self, token: Any, source: str, session: str, at_turn: int | None = None) -> None:
        from agentred.runner.channels.conversational import TargetError

        token.require_live()
        self.tokens.append(token)
        try:
            self.agent.fork(source, session, at_turn)
        except ValueError as error:
            raise TargetError(str(error)) from error


class BrokenTransport:
    """A target that answers every turn with an outage."""

    def send(
        self, token: Any, session: str, run: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        from agentred.runner.channels.conversational import TargetError

        raise TargetError("target answered a turn with HTTP 502")

    def fork(self, token: Any, source: str, session: str, at_turn: int | None = None) -> None:
        from agentred.runner.channels.conversational import TargetError

        raise TargetError("target answered a fork with HTTP 502")


class InProcessScheduleTransport:
    """Fires a `TargetAgent`'s scheduled entry point with no socket in between.

    Separate from `InProcessTransport` on purpose. The two are different acts against the
    agent, and a fake that could do both would let a test claim it fired a schedule when it
    actually sent a turn, which is the exact substitution ADR-0006 forbids.

    Attributes:
        agent: The target to fire.
        tokens: Every consent token it was handed, so a test can assert that the driver
            never fires a schedule without one.
        firings: How many times it was fired.
    """

    def __init__(self, agent: TargetAgent) -> None:
        self.agent = agent
        self.tokens: list[object] = []
        self.firings = 0

    def fire(self, token: Any, session: str, run: str) -> dict[str, Any]:
        import asyncio

        from agentred.targets.runtime import TriggerRequest

        token.require_live()
        self.tokens.append(token)
        self.firings += 1
        request = TriggerRequest(session=session, run=run)
        return asyncio.run(self.agent.trigger(request)).model_dump()


class BrokenScheduleTransport:
    """A target whose schedule will not fire."""

    def fire(self, token: Any, session: str, run: str) -> dict[str, Any]:
        from agentred.runner.channels.attempt import TargetError

        raise TargetError("target answered a firing with HTTP 502")
