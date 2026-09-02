"""Branches share a prefix and a starting world, and nothing after that."""

from __future__ import annotations

import pytest

from agentred.runner.channels.conversational import TargetError, run_conversation
from agentred.runner.fork import Branch, fan_out, fork_conversation, prefix_of
from tests.fakes.target import BrokenTransport, ScriptedTurn
from tests.runner.test_conversational import ScriptedAttacker, consent_for, driving, target


def opening(transport: object) -> object:
    return run_conversation(
        consent_for(),
        ScriptedAttacker("hello", "my order is ORD-55210"),
        **driving(transport),
    )


def test_a_prefix_is_a_copy_not_a_view() -> None:
    transcript = opening(target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b")))
    prefix = prefix_of(transcript, 1)
    prefix.turns.clear()
    assert len(transcript.turns) == 2


@pytest.mark.parametrize("turns", [0, 3, -1])
def test_a_prefix_must_name_turns_that_exist(turns: int) -> None:
    transcript = opening(target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b")))
    with pytest.raises(ValueError, match="prefix"):
        prefix_of(transcript, turns)


def test_a_branch_keeps_the_shared_turns_and_adds_its_own() -> None:
    transport = target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b"), ScriptedTurn(reply="c"))
    transcript = opening(transport)
    branch = fork_conversation(
        consent_for(), transcript, ScriptedAttacker("refund me"), at_turn=1, **driving(transport)
    )
    assert [turn.user for turn in branch.turns] == ["hello", "refund me"]
    assert branch.session != transcript.session


def test_a_branch_carries_its_own_goal() -> None:
    transport = target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b"))
    transcript = opening(transport)
    branch = fork_conversation(
        consent_for(),
        transcript,
        ScriptedAttacker("go on", goal="invent a return policy"),
        at_turn=1,
        **driving(transport),
    )
    assert branch.goal == "invent a return policy"


def test_a_branch_starts_from_the_worlds_state_at_the_fork() -> None:
    transport = target(
        ScriptedTurn(
            reply="refunded", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 100})]
        )
    )
    transcript = opening(transport)
    branch = fork_conversation(
        consent_for(), transcript, ScriptedAttacker("again please"), at_turn=2, **driving(transport)
    )
    assert branch.tool_calls[-1].result["refunded_to_date"] == 300


def test_two_branches_cannot_see_each_others_damage() -> None:
    transport = target(
        ScriptedTurn(reply="hi"),
        ScriptedTurn(
            reply="refunded", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 50})]
        ),
    )
    transcript = opening(transport)
    branches = fan_out(
        consent_for(),
        transcript,
        [
            Branch(attacker=ScriptedAttacker("one"), label="polite"),
            Branch(attacker=ScriptedAttacker("two"), label="urgent"),
        ],
        at_turn=1,
        **driving(transport),
    )
    assert len(branches) == 2
    for branch in branches:
        assert branch.tool_calls[-1].result["refunded_to_date"] == 50


def test_forking_a_session_the_target_does_not_have_is_an_error() -> None:
    transport = target(ScriptedTurn(reply="a"))
    transcript = opening(transport)
    transcript.session = "ar-never-existed"
    with pytest.raises(TargetError, match="no session"):
        fork_conversation(
            consent_for(), transcript, ScriptedAttacker("x"), at_turn=1, **driving(transport)
        )


def test_a_target_that_cannot_fork_stops_the_branch() -> None:
    transcript = opening(target(ScriptedTurn(reply="a")))
    with pytest.raises(TargetError, match="502"):
        fork_conversation(
            consent_for(),
            transcript,
            ScriptedAttacker("x"),
            at_turn=1,
            **driving(BrokenTransport()),
        )


def test_the_branch_budget_counts_new_turns_only() -> None:
    transport = target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b"))
    transcript = opening(transport)

    class Untiring:
        goal = "keep going"

        def next_turn(self, transcript: object) -> str:
            return "more"

    branch = fork_conversation(
        consent_for(), transcript, Untiring(), at_turn=1, max_turns=2, **driving(transport)
    )
    assert len(branch.turns) == 3
    assert branch.stopped_because == "turn budget spent"


def test_a_branch_does_not_inherit_damage_from_turns_it_does_not_contain() -> None:
    transport = target(
        ScriptedTurn(reply="hello"),
        ScriptedTurn(
            reply="refunded", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 400})]
        ),
    )
    transcript = opening(transport)
    assert transcript.tool_calls[-1].result["refunded_to_date"] == 400

    early = fork_conversation(
        consent_for(),
        transcript,
        ScriptedAttacker("try something else"),
        at_turn=1,
        **driving(transport),
    )
    late = fork_conversation(
        consent_for(),
        transcript,
        ScriptedAttacker("try something else"),
        at_turn=2,
        **driving(transport),
    )
    # The early branch starts from a world where nothing had been refunded yet; the late one
    # starts after the 400 that turn two paid out.
    assert early.tool_calls[-1].result["refunded_to_date"] == 400
    assert late.tool_calls[-1].result["refunded_to_date"] == 800
    assert len(early.turns) == 2


def test_forking_past_the_end_of_a_conversation_is_refused() -> None:
    transport = target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b"))
    transcript = opening(transport)
    with pytest.raises(ValueError, match="prefix"):
        fork_conversation(
            consent_for(), transcript, ScriptedAttacker("x"), at_turn=5, **driving(transport)
        )
