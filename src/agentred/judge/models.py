"""What a check produces, whether a detector asserted it or a model judged it.

One shape for both, deliberately. A scorecard mixes assertions and opinions, a human labels
both from the same screen, and the store keeps both in one table, so a verdict that changed
shape depending on who produced it would push that distinction into every consumer. It is
carried as a field instead: `settled_by` says whether anybody had to form an opinion, and
`confidence` is only ever set by the one who did.

**The third outcome is the point of this module.** A check has three answers, not two: the
rule was broken, the rule held, or the rule was never in play. Collapsing the third into the
second is the single easiest way to publish a safety number that is too good, because an
agent that never took the action at all scores identically to one that took it correctly.
Every detector here has to say which of the three it means, and say why when it means the
third.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agentred.spec.models import Provenance


class Outcome(StrEnum):
    """What a check concluded.

    Attributes:
        VIOLATED: The rule was broken, and the evidence names where.
        HELD: The rule was exercised and not broken.
        NOT_EVALUATED: The rule was never in play, so this conversation says nothing about
            it. Never counted as a pass.
    """

    VIOLATED = "violated"
    HELD = "held"
    NOT_EVALUATED = "not_evaluated"


class Settled(StrEnum):
    """Who decided.

    Attributes:
        DETECTOR: Asserted from the tool-call log. Carries no error rate.
        JUDGE: A model read what was said. Carries the judge's measured error rate.
    """

    DETECTOR = "detector"
    JUDGE = "judge"


class Evidence(BaseModel):
    """Where in the conversation a finding is anchored.

    A verdict a reader cannot check is an assertion, so every violation points at the exact
    call that produced it and renders the comparison that failed.

    Attributes:
        turn: Zero-based index of the exchange the call was made in.
        call: Zero-based index of the call in the conversation's whole tool-call log.
        tool: The action that was called.
        argument: The argument carrying the violation, where the check has one.
        observed: What was actually passed or reached, rendered.
        limit: What the declaration permitted, rendered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int = Field(ge=0)
    call: int = Field(ge=0)
    tool: str = Field(min_length=1)
    argument: str = ""
    observed: str = ""
    limit: str = ""


class Finding(BaseModel):
    """One check against one declared rule, on one conversation.

    Attributes:
        rule: The name of the declaration this checks, as the merchant wrote it. Empty only
            where the check comes from an absence rather than a declaration.
        kind: The shape of the check. Matches the stake kinds, so a finding joins to the
            stake that aimed at it.
        outcome: Broken, held, or never in play.
        summary: One sentence, in the merchant's own vocabulary, saying what happened.
        evidence: Where it happened. Empty for a rule that held or was never in play.
        settled_by: Whether this was asserted or judged.
        provenance: Whether the declaration behind it was written by the merchant or
            inferred from prose, carried so degraded mode stays visible in the output.
        confidence: Only ever set by a judge. A detector leaves it `None`, which is not the
            same as 1.0 and must not be rendered as a percentage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str = ""
    kind: str = Field(min_length=1)
    outcome: Outcome
    summary: str = Field(min_length=1)
    evidence: Evidence | None = None
    settled_by: Settled = Settled.DETECTOR
    provenance: Provenance = Provenance.DECLARED
    confidence: float | None = None

    @property
    def is_violation(self) -> bool:
        """Whether this finding is a broken rule."""
        return self.outcome is Outcome.VIOLATED


def violations(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    """Just the broken rules, in the sequence they were found."""
    return tuple(finding for finding in findings if finding.is_violation)


def evaluated(findings: tuple[Finding, ...]) -> tuple[Finding, ...]:
    """Every finding that actually tested something.

    The denominator for any rate computed off a set of findings. Using the whole set instead
    counts rules that were never in play as rules that held, which is the failure this module
    exists to prevent.
    """
    return tuple(finding for finding in findings if finding.outcome is not Outcome.NOT_EVALUATED)
