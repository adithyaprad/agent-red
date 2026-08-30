"""Mutations: the third coordinate, and the one guarantee that is easy to lose.

A mutation only earns its wall clock if it actually varies something. Most of these check that
a variant cannot silently degrade into the thing it was supposed to differ from, because that
failure is invisible in a report: the run says the question was asked.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentred.attacks.mutations import (
    SURFACES,
    Axis,
    Mutation,
    MutationError,
    by_axis,
    by_id,
)


class TestTheSurfaces:
    def test_all_four_axes_are_covered(self):
        assert {mutation.axis for mutation in SURFACES} == set(Axis)

    def test_ids_are_unique(self):
        assert len({mutation.id for mutation in SURFACES}) == len(SURFACES)

    def test_the_sequence_is_fixed(self):
        """A run has to be reproducible, so the sequence cannot come from a set."""
        assert isinstance(SURFACES, tuple)
        assert [m.id for m in SURFACES] == [m.id for m in SURFACES]

    def test_every_mutation_states_what_it_is_asking(self):
        """A variant whose purpose is not stated becomes noise the moment somebody reads it."""
        assert all(mutation.question.endswith("?") for mutation in SURFACES)

    def test_language_is_varied_two_ways(self):
        ids = {mutation.id for mutation in by_axis(Axis.LANGUAGE)}
        assert ids == {"hinglish", "code_switch"}

    def test_by_axis_of_something_unvaried_is_empty(self):
        assert by_axis("handwriting") == ()

    def test_by_id_finds_one(self):
        assert by_id("hinglish").axis is Axis.LANGUAGE

    def test_by_id_of_an_unknown_raises(self):
        with pytest.raises(KeyError):
            by_id("no_such_mutation")


class TestTheMidConversationSwitch:
    def test_exactly_one_mutation_switches(self):
        assert [m.id for m in SURFACES if m.switches] == ["code_switch"]

    def test_it_holds_the_first_voice_until_the_switch(self):
        switch = by_id("code_switch")
        assert switch.directive_for(0) == switch.directive
        assert switch.directive_for(switch.switch_after - 1) == switch.directive

    def test_it_holds_the_second_voice_after(self):
        switch = by_id("code_switch")
        assert switch.directive_for(switch.switch_after) == switch.later_directive
        assert switch.directive_for(switch.switch_after + 3) == switch.later_directive

    def test_an_unswitching_mutation_says_the_same_thing_every_turn(self):
        plain = by_id("hurried_professional")
        assert plain.directive_for(0) == plain.directive_for(9)


class TestASwitchMustActuallySwitch:
    """The failure is invisible in a report, so it is refused at construction."""

    def test_switching_at_turn_zero_is_refused(self):
        with pytest.raises(MutationError, match="something to switch from"):
            Mutation(
                id="fake_switch",
                name="Fake switch",
                axis=Axis.LANGUAGE,
                question="Does this look like a switch without being one?",
                directive="Write in one voice, carefully and plainly.",
                later_directive="Write in a different voice from now on.",
                switch_after=0,
            )

    def test_a_switch_with_nothing_to_switch_to_is_refused(self):
        with pytest.raises(MutationError, match="nothing to switch to"):
            Mutation(
                id="empty_switch",
                name="Empty switch",
                axis=Axis.LANGUAGE,
                question="Does a switch to nothing get caught?",
                directive="Write in one voice, carefully and plainly.",
                switch_after=2,
            )

    def test_a_plain_mutation_needs_neither(self):
        mutation = Mutation(
            id="plain",
            name="Plain",
            axis=Axis.REGISTER,
            question="Does an ordinary variant construct without either field?",
            directive="Write the way anybody would write, with nothing unusual about it.",
        )
        assert not mutation.switches

    def test_an_id_that_is_not_an_identifier_is_refused(self):
        with pytest.raises(ValidationError):
            Mutation(
                id="Not An Id",
                name="Bad",
                axis=Axis.REGISTER,
                question="Does a loose id get through and end up on a scorecard?",
                directive="Write the way anybody would write, with nothing unusual about it.",
            )

    def test_a_mutation_is_frozen(self):
        with pytest.raises(ValidationError):
            by_id("hinglish").directive = "something else"
