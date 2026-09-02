"""One driver per channel an attack can arrive down.

A channel is a named way of getting attacker-controlled bytes in front of an agent,
together with the trigger that makes the agent read them (ADR-0006). Two families exist and
both are first class:

`conversational` holds the multi-turn driver. It is the only channel where an attack can
adapt to what the agent just said, which is where the interesting failures are, and it is
the only place the breaking point and the consistency comparison can be computed at all.

`planted` writes a payload into a field of the world before the agent runs, fires the
agent's real entry point, and reads what it did off the tool server's record. This is the
channel an agent nobody talks to has, and without it a scheduled agent reports clean
because the harness had no way to reach it.

The split is by delivery, not by capability. Both end in a `Transcript` the same judge
reads and the same store keeps, because a finding is a finding whichever way the bytes
arrived.
"""

from agentred.runner.channels.conversational import (
    Attacker,
    TargetError,
    TargetTransport,
    ToolCallRecord,
    Transcript,
    Turn,
    new_session_id,
    run_conversation,
)

__all__ = [
    "Attacker",
    "TargetError",
    "TargetTransport",
    "ToolCallRecord",
    "Transcript",
    "Turn",
    "new_session_id",
    "run_conversation",
]
