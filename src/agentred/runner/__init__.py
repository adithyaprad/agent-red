"""The consent gate and the loop that drives one conversation.

Nothing else in the tree sends a turn to a target. Everything that does takes a
`ConsentToken`, which only `consent.establish_consent` can produce.
"""

from agentred.runner.channels.conversational import (
    Attacker,
    TargetError,
    TargetTransport,
    ToolCallRecord,
    Transcript,
    Turn,
    run_conversation,
)
from agentred.runner.consent import (
    ChallengeFailedError,
    ChallengeTransport,
    ConsentError,
    ConsentLease,
    ConsentToken,
    RegisteredTarget,
    RegistryError,
    TargetNotRegisteredError,
    TargetRegistry,
    establish_consent,
    load_registry,
)
from agentred.runner.fork import Branch, fan_out, fork_conversation, prefix_of

__all__ = [
    "Attacker",
    "Branch",
    "ChallengeFailedError",
    "ChallengeTransport",
    "ConsentError",
    "ConsentLease",
    "ConsentToken",
    "RegisteredTarget",
    "RegistryError",
    "TargetError",
    "TargetNotRegisteredError",
    "TargetRegistry",
    "TargetTransport",
    "ToolCallRecord",
    "Transcript",
    "Turn",
    "establish_consent",
    "fan_out",
    "fork_conversation",
    "load_registry",
    "prefix_of",
    "run_conversation",
]
