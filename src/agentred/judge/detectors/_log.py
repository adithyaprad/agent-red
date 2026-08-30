"""Reading a conversation's action log, shared by the three checks.

The flattened log on a transcript loses which exchange a call belongs to, and a finding a
reader cannot locate in the conversation is not much better than an assertion. So the log is
walked here once, keeping both positions, and every check anchors its evidence with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentred.runner.conversation import ToolCallRecord, Transcript

ERROR_KEY = "error"
"""The result key an implementation uses to report that it did not do the thing.

The weakest possible reading of "succeeded", and the fallback when a declaration does not say
what success looks like. A merchant whose gating step can answer no rather than fail should
declare the stronger condition; the finding says which of the two was applied.
"""


@dataclass(frozen=True)
class LoggedCall:
    """One call, with enough position on it to cite.

    Attributes:
        turn: Which exchange it was made in.
        index: Position in the whole conversation's log, in call sequence.
        record: The call itself, exactly as the target reported it.
    """

    turn: int
    index: int
    record: ToolCallRecord

    @property
    def name(self) -> str:
        """The action's declared name."""
        return self.record.name

    @property
    def failed(self) -> bool:
        """Whether the result reported an outright failure."""
        return isinstance(self.record.result, dict) and ERROR_KEY in self.record.result


def walk(transcript: Transcript) -> tuple[LoggedCall, ...]:
    """Every call in the conversation, in sequence, carrying its position.

    Args:
        transcript: One conversation.

    Returns:
        The calls. Empty for a conversation in which the agent did nothing, which is a real
        result and not an error.
    """
    calls: list[LoggedCall] = []
    for turn in transcript.turns:
        for record in turn.tool_calls:
            calls.append(LoggedCall(turn=turn.index, index=len(calls), record=record))
    return tuple(calls)


def calls_to(log: tuple[LoggedCall, ...], name: str) -> tuple[LoggedCall, ...]:
    """Every call to one action, in sequence."""
    return tuple(call for call in log if call.name == name)


def as_number(value: Any) -> float | None:
    """Read an argument as a number, or `None` if it is not one.

    An argument arrives as whatever the model sent, so a limit meant for a number can be
    handed a string that looks like one. A string that parses is treated as the number it
    spells, because the implementation on the other side will do the same and the limit is
    about the value that reached it.

    A boolean is not a number here. Python disagrees, and treating `True` as 1 would let a
    check pass judgement on an argument it does not understand.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
