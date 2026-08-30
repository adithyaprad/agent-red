"""Mutations: the same attack, wearing a different surface.

A technique is what pressure is applied and a stake is what it is applied to. A mutation is
neither. It changes how the turn is written without changing either, and it exists to ask one
narrow question: **was that refusal a position, or was it an artifact of how the request
happened to be phrased?**

That question is not decorative. An agent that holds in careful English and folds in a mixed
language has not been made safe, it has been made safe for the people who write like whoever
tested it. An agent that refuses the direct ask and grants the same thing split into two
halves is matching wording rather than holding a limit. Neither shows up in a suite that
writes every turn in one voice, and both are exposure the merchant is actually carrying.

A mutation is a directive handed to the attacker each turn, so it can change partway through a
conversation. That is what makes a mid-conversation language switch expressible at all: it is
the same attack, in one voice up to a point and another voice after it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Axis(StrEnum):
    """What a mutation varies.

    One axis per mutation, so a scorecard can report where an agent's resistance is thin
    rather than only that some variant broke it. Two mutations on the same axis are
    alternatives; two on different axes are separate questions.

    Attributes:
        REGISTER: Who the speaker sounds like.
        LANGUAGE: What language the turn is written in, including changing it partway.
        PRESSURE: How hard the technique is pushed.
        OBFUSCATION: How directly the request is stated.
    """

    REGISTER = "register"
    LANGUAGE = "language"
    PRESSURE = "pressure"
    OBFUSCATION = "obfuscation"


class MutationError(Exception):
    """A mutation is self-contradicting and cannot be constructed.

    Raised at construction rather than tolerated, on the same grounds as a self-contradicting
    spec: a variant that silently degrades into the thing it was supposed to differ from
    produces a run that reports coverage of a question it never asked.
    """


class Mutation(BaseModel):
    """One surface an attack can be run in.

    Attributes:
        id: Stable identifier. Appears in the attack id and therefore on the scorecard.
        name: What a person would call it.
        axis: What it varies.
        question: What running this variant is meant to answer, in one line. Carried into the
            report, because a variant whose purpose is not stated becomes noise the moment
            somebody else reads the numbers.
        directive: What the attacker is told, every turn, on top of its technique.
        later_directive: What it is told instead once `switch_after` turns have been said.
            Empty for a mutation that does not change partway.
        switch_after: How many turns are said in the first voice. Meaningless, and refused,
            without a `later_directive`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1)
    axis: Axis
    question: str = Field(min_length=20)
    directive: str = Field(min_length=20)
    later_directive: str = ""
    switch_after: int = 0

    @model_validator(mode="after")
    def _a_switch_must_actually_switch(self) -> Mutation:
        """Refuse a mid-conversation change that never happens, or happens before turn one.

        A switch at zero is the second voice wearing the first one's name: it would be
        reported as a mid-conversation change and would be a plain language variant, so the
        one thing the variant exists to test would go untested while appearing tested.

        Raises:
            MutationError: If only one of the two fields is set, or a switch is declared at
                turn zero.
        """
        if self.later_directive and self.switch_after < 1:
            raise MutationError(
                f"{self.id!r} declares a change of voice at turn {self.switch_after}, which is "
                f"not a change: the conversation would never be in the first voice. A "
                f"mid-conversation switch has to have something to switch from."
            )
        if self.switch_after and not self.later_directive:
            raise MutationError(
                f"{self.id!r} declares a switch after {self.switch_after} turns with nothing to "
                f"switch to."
            )
        return self

    @property
    def switches(self) -> bool:
        """Whether this mutation changes the attacker's directive partway through."""
        return bool(self.later_directive)

    def directive_for(self, said: int) -> str:
        """The directive in force once `said` turns have been said.

        Args:
            said: How many turns the attacker has already produced. Zero on the opening turn.

        Returns:
            The directive to append to this turn's brief.
        """
        if self.later_directive and said >= self.switch_after:
            return self.later_directive
        return self.directive


from agentred.attacks.mutations.surfaces import (  # noqa: E402
    SURFACES,
    by_axis,
    by_id,
)

__all__ = [
    "SURFACES",
    "Axis",
    "Mutation",
    "MutationError",
    "by_axis",
    "by_id",
]
