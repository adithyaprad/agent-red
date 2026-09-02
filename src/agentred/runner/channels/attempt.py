"""The record of one attack attempt, whichever channel it arrived down.

Both drivers end here. A conversational attempt is a list of exchanges; a planted attempt
is one field written into the world and then a trigger fired, which produces a single
exchange with an empty or an absent user turn. Neither shape is privileged, because a
finding is a finding whichever way the bytes arrived, and the judge, the store and the
scorecard all read this one object.

Kept in its own module rather than inside a driver so that neither driver has to import the
other to name what it produced. The alternative, leaving these in `conversational.py`, would
have the planted channel importing a module named after a channel it is not.

**Nothing here is reported by the agent.** `ToolCallRecord` is built from what the tool
server recorded, with full arguments (ADR-0005). The reply body carries prose and versions
only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentred.mcp.recorder import RecordedCall
from agentred.spec.models import CONVERSATIONAL_CHANNEL


class TargetError(RuntimeError):
    """The target could not be reached, or answered with something unusable.

    Distinct from a target that answered badly: an agent that says something it should not
    is the result, and a target that returns HTTP 500 is a broken run. Conflating them
    would let an outage read as a suite full of well-behaved agents.
    """


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call, as the tool server observed it.

    Attributes:
        name: The declared tool name.
        arguments: Arguments as the model sent them, uncoerced. Bounds are checked against
            what was passed, not against what the agent said it passed.
        result: What the tool returned.
    """

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ToolCallRecord:
        """Build a record from a stored transcript."""
        return cls(
            name=str(payload.get("name", "")),
            arguments=dict(payload.get("arguments") or {}),
            result=dict(payload.get("result") or {}),
        )

    @classmethod
    def from_recorded(cls, call: RecordedCall) -> ToolCallRecord:
        """Build a record from what the tool server recorded."""
        return cls(name=call.name, arguments=dict(call.arguments), result=dict(call.result))


@dataclass(frozen=True)
class Turn:
    """One exchange: what was said to the agent, and what it did about it.

    Attributes:
        index: Zero-based position in the conversation.
        user: The attacker's turn.
        reply: The agent's text.
        tool_calls: Tools called while producing that reply, in call order.
        latency_seconds: Wall clock for the target's answer.
        agent_usage: What this turn cost the target, as the target reported it. Empty when the
            target does not report it, which is not a claim that it was free. Kept because the
            harness spends on both sides of every turn and can otherwise only see its own
            half of the bill.
    """

    index: int
    user: str
    reply: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    latency_seconds: float = 0.0
    agent_usage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PlantedField:
    """One field an attack wrote into the world before the agent ran.

    Carried on the transcript because a planted finding is unreadable without it. The
    transcript of a planted attempt shows a benign trigger and an agent doing something it
    should not, and the only thing that explains the gap is the text a stranger had already
    put in a field the agent read. `replaced` is kept so a report can show the field's
    ordinary content beside what stood in for it, which is what makes the point land: this
    is where a delivery instruction normally goes.

    Attributes:
        channel: The declared channel this was planted through.
        data_source: The declared data source the record lives in.
        record_id: The record written to.
        field_name: The field on it that was overwritten.
        payload: What was written.
        replaced: What the field held before, verbatim.
    """

    channel: str
    data_source: str
    record_id: str
    field_name: str
    payload: str
    replaced: str = ""


@dataclass
class Transcript:
    """One attack attempt, complete.

    The unit the judge grades, the store keeps and the scorecard cites. It carries the spec
    versions the target reported rather than the ones the harness believed, so a transcript
    can never be attributed to a version of the agent that did not produce it.

    Attributes:
        target: The registered target name.
        session: The session id the target kept this conversation's world under.
        goal: What the attacker was trying to make the agent do, in one line.
        turns: The exchanges, in order.
        spec_versions: Config, policy, model and tool versions, as reported by the target.
        subject: Who this conversation is about, as identifier kind to value, for example
            `{"order_id": "ORD-1"}`. The scope detector has nothing to compare a reached
            record against without it, and an empty subject makes every scope check report
            as unevaluated rather than as passed.
        stopped_because: Why the conversation ended.
        channel: The channel the attack arrived down. Defaults to the implicit
            conversational one, so every transcript written before channels existed reads
            as what it was rather than as a channel nobody named.
        planted: Fields written into the world before the agent ran. Empty for a
            conversational attempt.
    """

    target: str
    session: str
    goal: str
    turns: list[Turn] = field(default_factory=list)
    spec_versions: dict[str, str] = field(default_factory=dict)
    subject: dict[str, str] = field(default_factory=dict)
    stopped_because: str = ""
    channel: str = CONVERSATIONAL_CHANNEL
    planted: tuple[PlantedField, ...] = ()

    @property
    def messages(self) -> list[dict[str, str]]:
        """The conversation in wire shape, for sending the next turn or showing a human."""
        wire: list[dict[str, str]] = []
        for turn in self.turns:
            wire.append({"role": "user", "content": turn.user})
            wire.append({"role": "assistant", "content": turn.reply})
        return wire

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        """Every tool call in the conversation, flattened, in order.

        The deterministic detectors read this: a bound is broken by an argument, and a
        precondition is broken by an order, and both are answerable from this list alone.
        """
        return tuple(call for turn in self.turns for call in turn.tool_calls)

    def called(self, name: str) -> bool:
        """Whether a tool was called at any point in the conversation."""
        return any(call.name == name for call in self.tool_calls)
