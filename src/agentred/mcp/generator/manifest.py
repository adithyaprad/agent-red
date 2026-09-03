"""What a generated world made reachable, and what it could not.

The half of the generator that a person reads. A world that quietly failed to make a rule
breakable produces a run in which that rule holds, and a rule that was tested and held is the
opposite fact from a rule nothing could ever have broken. In a finding count they are
identical: both contribute nothing. So the generator says which it is, per rule, and the
report carries the gaps beside the coverage grid in the same voice the grid uses for a cell
nothing was attempted in.

The number this produces is unflattering by construction on the first run against a new agent.
That is the correct direction for it to be wrong in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentred.mcp.world import World


class Reach(StrEnum):
    """Which half of a rule a fixture exists for.

    Both halves are load-bearing and the second is the one that gets forgotten. An agent that
    refuses everything scores perfectly against a world made entirely of traps, so a world of
    traps cannot tell judgement from compliance and a benign suite run over it measures
    nothing.

    Attributes:
        BREAKABLE: Breaking the rule is one step from a reasonable action.
        HOLDING: Holding is the right answer, and doing the work is not a violation.
    """

    BREAKABLE = "breakable"
    HOLDING = "holding"


@dataclass(frozen=True)
class Fixture:
    """One record, and the rule it exists for.

    The question "why is this record here" always has an answer that is not somebody's
    judgement after the fact, because the answer was written when the record was emitted. That
    is the whole mechanism behind the neutrality property: a fixture belongs only if it would
    still belong had the agent held, and a fixture emitted per rule can be checked against
    that by reading the manifest rather than by re-reading the data.

    Attributes:
        rule: The policy statement this record makes reachable.
        collection: Where the record lives.
        record_id: Its key within that collection.
        reach: Whether it makes the rule breakable or makes holding right.
        why: One line saying what about the record does it.
    """

    rule: str
    collection: str
    record_id: str
    reach: Reach
    why: str


@dataclass(frozen=True)
class Gap:
    """A declared rule the generator could not make reachable.

    Carried into the report rather than dropped. A rule with no reachable fixture and a rule
    that was tested and held look identical in a finding count and are opposite facts about
    the agent.

    Attributes:
        rule: The policy statement.
        why: What the declaration did not say, in the merchant's own terms, so the fix is a
            line they can add rather than a defect they have to report.
    """

    rule: str
    why: str


@dataclass(frozen=True)
class Manifest:
    """Everything the generator did, and everything it could not do.

    Attributes:
        seed: What produced this world. The same seed and the same declaration produce the
            same shop, which is what lets a scorecard be scoped to a world at all.
        digest: A content hash of the emitted world, so two runs can be compared without
            either of them holding the data.
        fixtures: Every record emitted for a rule, in emission sequence.
        gaps: Every rule nothing could be emitted for.
    """

    seed: int
    digest: str
    fixtures: tuple[Fixture, ...] = ()
    gaps: tuple[Gap, ...] = ()

    @property
    def reachable(self) -> tuple[str, ...]:
        """Every rule at least one fixture makes breakable, in declaration sequence."""
        seen: dict[str, None] = {}
        for fixture in self.fixtures:
            if fixture.reach is Reach.BREAKABLE:
                seen.setdefault(fixture.rule, None)
        return tuple(seen)

    @property
    def unreachable(self) -> tuple[str, ...]:
        """Every rule nothing made breakable, in declaration sequence."""
        return tuple(gap.rule for gap in self.gaps)

    def coverage(self) -> float:
        """The share of declared rules this world made breakable, between 0 and 1.

        Returns 0.0 for an agent that declares no rules at all, rather than 1.0. Dividing by
        nothing and calling the answer complete is how a declaration nobody wrote reports as
        fully covered.
        """
        total = len(self.reachable) + len(self.gaps)
        return len(self.reachable) / total if total else 0.0

    def for_rule(self, rule: str) -> tuple[Fixture, ...]:
        """Every fixture emitted for one rule."""
        return tuple(fixture for fixture in self.fixtures if fixture.rule == rule)


def digest_of(collections: dict[str, dict[str, dict[str, Any]]]) -> str:
    """A stable content hash of an emitted world.

    Thin wrapper over `World.digest`, so the manifest and the version tuple can never disagree
    about what world a run was against.
    """
    return World(collections=collections).digest
