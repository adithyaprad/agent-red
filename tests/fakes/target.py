"""A target backend that follows a script instead of calling a model.

Lets the HTTP surface, the session isolation, the tool binding and the tool-call log all be
tested offline. The script is a list of turns; each turn names the tools to call and the
text to reply with, which is enough to reproduce every shape a real conversation produces
without reproducing the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    """Replays scripted turns in order, calling real tools against the session's world.

    Attributes:
        turns: The script. The last turn repeats if the conversation runs past it.
        agent: The target whose tools are called. Set by `attach`.
        seen: Every conversation this backend was asked to reply to, for assertions about
            what the runner actually sent.
    """

    def __init__(self, *turns: ScriptedTurn) -> None:
        self.turns = list(turns) or [ScriptedTurn(reply="")]
        self.agent: TargetAgent | None = None
        self.seen: list[list[ChatMessage]] = []

    def attach(self, agent: TargetAgent) -> None:
        self.agent = agent

    async def reply(self, session: Session, conversation: list[ChatMessage]) -> str:
        assert self.agent is not None, "ScriptedBackend was never attached"
        self.seen.append(list(conversation))
        turn = self.turns[min(len(self.seen) - 1, len(self.turns) - 1)]
        for name, arguments in turn.calls:
            self.agent.call_tool(session, name, arguments)
        return turn.reply


class InProcessTransport:
    """Sends turns straight to a `TargetAgent`, with no socket in between.

    Lets the driver be tested against a real target: real tool implementations, real
    per-session worlds, real tool-call log. Only the HTTP hop and the model are absent, and
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
        self, token: object, session: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        import asyncio

        from agentred.targets.runtime import ChatRequest

        token.require_live()
        self.tokens.append(token)
        request = ChatRequest(session=session, conversation=conversation)
        return asyncio.run(self.agent.chat(request)).model_dump()

    def fork(self, token: object, source: str, session: str, at_turn: int | None = None) -> None:
        from agentred.runner.conversation import TargetError

        token.require_live()
        self.tokens.append(token)
        try:
            self.agent.fork(source, session, at_turn)
        except ValueError as error:
            raise TargetError(str(error)) from error


class BrokenTransport:
    """A target that answers every turn with an outage."""

    def send(
        self, token: object, session: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        from agentred.runner.conversation import TargetError

        raise TargetError("target answered a turn with HTTP 502")

    def fork(self, token: object, source: str, session: str, at_turn: int | None = None) -> None:
        from agentred.runner.conversation import TargetError

        raise TargetError("target answered a fork with HTTP 502")
