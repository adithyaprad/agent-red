"""Checks that are decided by looking, not by asking.

Every check in this package reads two things: a validated spec, and one conversation's
tool-call log. It never reads what the agent said, never calls a model, and never guesses.
If a rule can be settled here it must be, because an assertion carries no error rate and a
model's answer does. Adding a model call where one of these would do is treated as a
regression, not an optimisation.

There are three shapes, and they are the three the stakes derivation aims at:

- a declared limit on an argument was passed
- a required prior step never happened, or happened and did not succeed
- a record outside the session's subject was reached

None of them contains a word about what any agent sells. What is checked comes from the
merchant's own declarations; the code only knows the shapes. A test fails the build if that
stops being true.

**Every check has three answers.** Broken, held, and never in play. The third is not a
failure of the detector, it is a fact about the conversation, and it is kept separate because
an agent that never reached the action scores identically to a careful one otherwise.
"""

from __future__ import annotations

from agentred.judge.detectors.bounds import bound_exceeded
from agentred.judge.detectors.preconditions import precondition_skipped
from agentred.judge.detectors.scope import scope_violation
from agentred.judge.models import Finding
from agentred.runner.conversation import Transcript
from agentred.spec.models import AgentSpec

DETECTORS = (bound_exceeded, precondition_skipped, scope_violation)
"""Every deterministic check, in a fixed sequence."""


def run_detectors(spec: AgentSpec, transcript: Transcript) -> tuple[Finding, ...]:
    """Run every deterministic check against one conversation.

    Args:
        spec: The validated spec the conversation was run against. Its declarations are what
            is being checked; nothing is checked that the merchant did not declare.
        transcript: One conversation.

    Returns:
        Every finding, in detector sequence. Includes the rules that held and the rules that
        were never in play, because a set of findings that only carried violations could not
        be turned into a rate without silently inventing a denominator.
    """
    return tuple(finding for detector in DETECTORS for finding in detector(spec, transcript))


__all__ = [
    "DETECTORS",
    "bound_exceeded",
    "precondition_skipped",
    "run_detectors",
    "scope_violation",
]
