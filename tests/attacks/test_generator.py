"""The join between the corpus and the stakes, tested entirely offline.

Every model call goes through `RecordedModelClient`, so these assert on the prompt that was
built as well as on what was done with the reply. The prompt is the product here: an attacker
handed the wrong context composes a plausible turn that proves nothing.
"""

from __future__ import annotations

import json

import pytest

from agentred.attacks.generator import (
    ATTACK_EFFORT,
    COMPOSE_ATTEMPTS,
    TURN_SCHEMA,
    AttackError,
    ModelAttacker,
    apply_mutations,
    attacker_system_prompt,
    build_attackers,
    build_suite,
    compose_opening,
    group_by_opening,
)
from agentred.attacks.mutations import SURFACES, by_id
from agentred.attacks.stakes import derive_stakes
from agentred.attacks.techniques import load_corpus
from agentred.runner.conversation import ToolCallRecord, Transcript, Turn
from agentred.spec.loader import load_spec_dir
from tests.fakes.model import RecordedModelClient

SPEC_ROOT = "src/agentred/targets/specs"


def spec_for(name: str):
    return load_spec_dir(f"{SPEC_ROOT}/{name}")


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def dispute():
    return spec_for("dispute_handler")


@pytest.fixture(scope="module")
def suite(dispute, corpus):
    return build_suite(dispute, corpus=corpus)


def reply(
    turn: str = "so where does that leave me?", *, stop: bool = False, reason: str = ""
) -> str:
    return json.dumps({"stop": stop, "turn": turn, "reason": reason})


def transcript_with(*, called: tuple[str, ...] = (), turns: int = 1) -> Transcript:
    """A transcript whose tool-call log names `called`, for exercising the groundwork gate."""
    made = tuple(ToolCallRecord(name=name, arguments={}, result={}) for name in called)
    return Transcript(
        target="t",
        session="s",
        goal="g",
        turns=[
            Turn(index=i, user=f"said {i}", reply=f"replied {i}", tool_calls=made if i == 0 else ())
            for i in range(turns)
        ],
    )


class TestBuildSuite:
    def test_every_technique_meets_every_stake(self, dispute, corpus, suite):
        assert len(suite) == len(corpus) * len(derive_stakes(dispute))

    def test_ids_are_unique(self, suite):
        assert len({attack.id for attack in suite}) == len(suite)

    def test_id_names_every_coordinate(self, suite):
        """Stake, technique and identity. A row that cannot say who it was is not traceable."""
        stake_id, technique_id, subject = suite[0].id.split("|")
        assert stake_id == suite[0].stake.id
        assert technique_id == suite[0].technique.id
        assert subject == suite[0].subject.name

    def test_an_agent_that_scopes_nothing_keeps_the_shorter_id(self, dispute, corpus):
        """No identity to be means no identity in the id, rather than an empty field."""
        unscoped = dispute.model_copy(update={"subjects": ()})
        assert build_suite(unscoped, corpus=corpus)[0].id.count("|") == 1

    def test_sequence_is_deterministic(self, dispute, corpus):
        first = [attack.id for attack in build_suite(dispute, corpus=corpus)]
        second = [attack.id for attack in build_suite(dispute, corpus=corpus)]
        assert first == second

    def test_costliest_stakes_come_first(self, suite):
        """A suite stopped halfway has spent its wall clock on what matters."""
        costs = [attack.stake.consequence for attack in suite]
        assert costs[0] == "money"
        assert costs[-1] != "money"

    def test_a_stake_is_not_split_across_the_suite(self, suite, corpus):
        """All of one stake's attacks are adjacent, which is what makes resuming coherent."""
        block = [attack.stake.id for attack in suite[: len(corpus)]]
        assert len(set(block)) == 1

    def test_goal_is_the_stake_objective(self, suite):
        assert suite[0].goal == suite[0].stake.objective

    def test_both_shipped_agents_derive_a_suite(self, corpus):
        for name in ("cart_recovery", "dispute_handler"):
            assert len(build_suite(spec_for(name), corpus=corpus)) > 0

    def test_an_agent_with_nothing_to_attack_is_refused(self, dispute, corpus, monkeypatch):
        """An empty suite would report a perfect score against an agent nobody tested."""
        monkeypatch.setattr("agentred.attacks.generator.derive_stakes", lambda spec: ())
        with pytest.raises(AttackError, match="nothing to attack"):
            build_suite(dispute, corpus=corpus)

    def test_a_duplicated_technique_is_refused(self, dispute, corpus):
        with pytest.raises(AttackError, match="share the id"):
            build_suite(dispute, corpus=(corpus[0], corpus[0]))


class TestGroupByOpening:
    def test_every_attack_lands_in_exactly_one_group(self, suite):
        grouped = [attack.id for group in group_by_opening(suite) for attack in group]
        assert sorted(grouped) == sorted(attack.id for attack in suite)

    def test_a_group_shares_technique_action_and_groundwork(self, suite):
        for group in group_by_opening(suite):
            assert len({attack.opening_key for attack in group}) == 1

    def test_grouping_saves_work(self, suite):
        """The point of grouping: fewer openings to write than there are attacks."""
        assert len(group_by_opening(suite)) < len(suite)

    def test_two_edges_of_one_limit_share_an_opening(self, dispute, corpus):
        """Above and below the same limit are different destinations, not different openings."""
        suite = build_suite(dispute, corpus=corpus)
        pair = "issue_refund:amount"
        above = next(
            a for a in suite if a.stake.id.endswith(f"{pair}:above") and a.technique is corpus[0]
        )
        below = next(
            a for a in suite if a.stake.id.endswith(f"{pair}:below") and a.technique is corpus[0]
        )
        assert above.opening_key == below.opening_key
        assert above.id != below.id


class TestTheSystemPrompt:
    def test_it_carries_the_whole_technique(self, suite):
        attack = suite[0]
        rendered = attacker_system_prompt(attack)
        for line in attack.technique.escalation:
            assert line in rendered
        for line in attack.technique.tells:
            assert line in rendered
        for line in attack.technique.exemplars:
            assert line in rendered
        assert attack.technique.fails_when in rendered
        assert attack.technique.arc in rendered

    def test_it_carries_the_objective(self, suite):
        assert suite[0].stake.objective in attacker_system_prompt(suite[0])

    def test_it_is_identical_across_a_conversation(self, suite):
        """The stable half is stable, which is the only thing making the prompt cache pay."""
        client = RecordedModelClient(replies=[reply("one"), reply("two")])
        attacker = ModelAttacker(attack=suite[0], client=client)
        attacker.next_turn(transcript_with(turns=0))
        attacker.next_turn(transcript_with(turns=1))
        assert client.calls[0].system == client.calls[1].system


class TestWhatTheAttackerIsAllowedToSee:
    def test_it_sees_what_was_said(self, suite):
        client = RecordedModelClient(replies=[reply()])
        transcript = transcript_with(turns=2)
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript)
        sent = client.calls[0].messages[0]["content"]
        assert "replied 0" in sent
        assert "replied 1" in sent

    def test_it_never_sees_arguments_or_results(self, suite):
        """A stranger quoting a figure never said aloud is a conversation nobody could have."""
        client = RecordedModelClient(replies=[reply()])
        transcript = Transcript(
            target="t",
            session="s",
            goal="g",
            turns=[
                Turn(
                    index=0,
                    user="hello",
                    reply="one moment",
                    tool_calls=(
                        ToolCallRecord(
                            name="get_order",
                            arguments={"order_id": "A-1"},
                            result={"total": 412.55},
                        ),
                    ),
                )
            ],
        )
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript)
        sent = client.calls[0].messages[0]["content"]
        assert "412.55" not in sent
        assert "A-1" not in sent

    def test_an_empty_conversation_says_so(self, suite):
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with(turns=0))
        assert "opening turn" in client.calls[0].messages[0]["content"]


class TestTheGroundworkGate:
    """A stake whose limit is read from an earlier action cannot be attacked cold."""

    def gated(self, suite):
        return next(attack for attack in suite if attack.stake.requires_first)

    def test_the_fixture_still_has_a_gated_stake(self, suite):
        assert self.gated(suite).stake.requires_first

    def test_it_is_told_to_lay_the_groundwork_first(self, suite):
        attack = self.gated(suite)
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=attack, client=client).next_turn(transcript_with(turns=0))
        sent = client.calls[0].messages[0]["content"]
        assert "not start pushing yet" in sent
        assert attack.stake.requires_first[0] in sent

    def test_it_is_told_to_push_once_the_groundwork_is_done(self, suite):
        attack = self.gated(suite)
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=attack, client=client).next_turn(
            transcript_with(called=attack.stake.requires_first)
        )
        sent = client.calls[0].messages[0]["content"]
        assert "groundwork is done" in sent
        assert "not start pushing yet" not in sent

    def test_an_ungated_stake_is_told_neither(self, suite):
        attack = next(a for a in suite if not a.stake.requires_first)
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=attack, client=client).next_turn(transcript_with(turns=0))
        sent = client.calls[0].messages[0]["content"]
        assert "groundwork" not in sent


class TestPacing:
    def test_it_is_told_how_many_turns_remain(self, suite):
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=suite[0], client=client, max_turns=6).next_turn(
            transcript_with(turns=2)
        )
        assert "6 turns left" in client.calls[0].messages[0]["content"]

    def test_the_last_turn_says_so(self, suite):
        client = RecordedModelClient(replies=[reply(), reply()])
        attacker = ModelAttacker(attack=suite[0], client=client, max_turns=2)
        attacker.next_turn(transcript_with(turns=0))
        attacker.next_turn(transcript_with(turns=1))
        assert "last turn" in client.calls[1].messages[0]["content"]

    def test_it_stops_when_the_budget_is_spent(self, suite):
        client = RecordedModelClient(replies=[reply()])
        attacker = ModelAttacker(attack=suite[0], client=client, max_turns=1)
        assert attacker.next_turn(transcript_with(turns=0)) is not None
        assert attacker.next_turn(transcript_with(turns=1)) is None
        assert attacker.stopped_because == "budget spent"

    def test_state_is_held_not_counted_off_the_transcript(self, suite):
        """A branched conversation arrives with turns in it that this attacker did not say."""
        client = RecordedModelClient(replies=[reply()])
        attacker = ModelAttacker(attack=suite[0], client=client, max_turns=2)
        attacker.next_turn(transcript_with(turns=4))
        assert attacker.said == 1


class TestStopping:
    def test_a_stop_ends_the_conversation(self, suite):
        client = RecordedModelClient(replies=[reply("", stop=True, reason="it named the ask")])
        attacker = ModelAttacker(attack=suite[0], client=client)
        assert attacker.next_turn(transcript_with()) is None
        assert attacker.stopped_because == "it named the ask"

    def test_a_stopped_attacker_stays_stopped(self, suite):
        client = RecordedModelClient(replies=[reply("", stop=True, reason="done")])
        attacker = ModelAttacker(attack=suite[0], client=client)
        attacker.next_turn(transcript_with())
        assert attacker.next_turn(transcript_with(turns=2)) is None
        assert len(client.calls) == 1

    def test_a_stop_with_no_reason_still_records_one(self, suite):
        client = RecordedModelClient(replies=[reply("", stop=True)])
        attacker = ModelAttacker(attack=suite[0], client=client)
        attacker.next_turn(transcript_with())
        assert attacker.stopped_because


class TestUnusableReplies:
    def test_an_empty_turn_is_an_error_not_a_skipped_turn(self, suite):
        """A substituted turn produces a transcript that reads like an attempt and was not."""
        client = RecordedModelClient(replies=[reply("   ")] * COMPOSE_ATTEMPTS)
        with pytest.raises(AttackError, match="empty"):
            ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())

    def test_unparseable_json_is_an_error(self, suite):
        client = RecordedModelClient(replies=["I'm sorry, I can't help."] * COMPOSE_ATTEMPTS)
        with pytest.raises(AttackError, match="not readable"):
            ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())

    def test_a_non_object_reply_is_an_error(self, suite):
        client = RecordedModelClient(replies=['["a turn"]'] * COMPOSE_ATTEMPTS)
        with pytest.raises(AttackError, match="expected an object"):
            ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())

    def test_the_refusal_says_it_was_unreadable_every_time(self, suite):
        """So a systematic problem reads as one rather than as a single bad reply."""
        client = RecordedModelClient(replies=['["a turn"]'] * COMPOSE_ATTEMPTS)
        with pytest.raises(AttackError, match=f"on all {COMPOSE_ATTEMPTS} attempts"):
            ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())


class TestAskingAgainForAnUnusableTurn:
    """Run 0004 lost three conversations in forty to a reply that parsed but held no turn.
    The call had succeeded, so no transport retry could have helped it."""

    def test_a_second_ask_recovers_the_conversation(self, suite):
        client = RecordedModelClient(replies=[reply("   "), reply("hello there")])
        assert ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with()) == (
            "hello there"
        )
        assert len(client.calls) == 2

    def test_unreadable_json_is_asked_for_again_too(self, suite):
        client = RecordedModelClient(replies=["not json", reply("hello there")])
        attacker = ModelAttacker(attack=suite[0], client=client)
        assert attacker.next_turn(transcript_with()) == "hello there"

    def test_the_re_ask_is_the_same_request(self, suite):
        """A turn composed under instructions no other turn saw was produced by different
        conditions, and the conversation stops being comparable to the rest of the suite."""
        client = RecordedModelClient(replies=[reply("   "), reply("hello there")])
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())
        first, second = client.calls
        assert first.messages == second.messages
        assert first.system == second.system
        assert first.output_schema == second.output_schema

    def test_a_usable_reply_costs_exactly_one_call(self, suite):
        client = RecordedModelClient(replies=[reply("hello there")])
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())
        assert len(client.calls) == 1

    def test_a_decision_to_stop_is_not_retried(self, suite):
        """Stopping mid-conversation is a real answer, not a failure to give one."""
        client = RecordedModelClient(replies=[reply("", stop=True, reason="done")])
        assert ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with()) is None
        assert len(client.calls) == 1

    def test_the_budget_is_bounded(self, suite):
        client = RecordedModelClient(replies=[reply("   ")] * (COMPOSE_ATTEMPTS + 5))
        with pytest.raises(AttackError):
            ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())
        assert len(client.calls) == COMPOSE_ATTEMPTS


class TestTheRequestShape:
    def test_the_schema_is_enforced_on_every_call(self, suite):
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())
        assert client.calls[0].output_schema == TURN_SCHEMA

    def test_effort_is_the_composing_one(self, suite):
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())
        assert client.calls[0].effort == ATTACK_EFFORT


class TestSharedOpenings:
    def test_an_opening_is_composed_once_per_group(self, suite):
        groups = group_by_opening(suite)
        client = RecordedModelClient(replies=[reply(f"opening {i}") for i in range(len(groups))])
        attackers = build_attackers(suite, client, share_openings=True)
        assert len(client.calls) == len(groups)
        assert len(attackers) == len(suite)

    def test_a_shared_opening_costs_no_call_when_it_is_used(self, suite):
        client = RecordedModelClient(replies=[])
        attacker = ModelAttacker(attack=suite[0], client=client, opening="hello there")
        assert attacker.next_turn(transcript_with(turns=0)) == "hello there"
        assert client.calls == []

    def test_members_of_a_group_get_the_same_opening(self, suite):
        groups = group_by_opening(suite)
        client = RecordedModelClient(replies=[reply(f"opening {i}") for i in range(len(groups))])
        attackers = build_attackers(suite, client, share_openings=True)
        by_key: dict[tuple, set[str]] = {}
        for attacker in attackers:
            by_key.setdefault(attacker.attack.opening_key, set()).add(attacker.opening or "")
        assert all(len(openings) == 1 for openings in by_key.values())

    def test_openings_are_off_by_default(self, suite):
        client = RecordedModelClient(replies=[])
        attackers = build_attackers(suite, client)
        assert client.calls == []
        assert all(attacker.opening is None for attacker in attackers)

    def test_stopping_on_an_empty_conversation_is_refused(self, suite):
        """Not a stopping condition being met. A failure to start."""
        stopped = reply("", stop=True, reason="nothing to do")
        client = RecordedModelClient(replies=[stopped] * COMPOSE_ATTEMPTS)
        with pytest.raises(AttackError, match="stopped before anything was said"):
            compose_opening(suite[0], client)

    def test_an_opening_that_stopped_is_asked_for_again(self, suite):
        """Unlike a mid-conversation stop, this one is always a failure, so it is worth
        re-asking rather than losing the whole conversation to it."""
        client = RecordedModelClient(
            replies=[reply("", stop=True, reason="nothing to do"), reply("hello there")]
        )
        assert compose_opening(suite[0], client) == "hello there"

    def test_the_refusal_still_names_the_attack(self, suite):
        stopped = reply("", stop=True, reason="nothing to do")
        client = RecordedModelClient(replies=[stopped] * COMPOSE_ATTEMPTS)
        with pytest.raises(AttackError, match=suite[0].id):
            compose_opening(suite[0], client)


class TestItSatisfiesTheDriver:
    def test_the_driver_can_run_a_generated_attacker(self, suite):
        """The contract that matters: the driver knows nothing about technique."""
        from agentred.runner.conversation import run_conversation
        from tests.fakes.target import ScriptedTurn
        from tests.runner.test_conversation import consent_for, driving, target

        client = RecordedModelClient(replies=[reply("first"), reply("second")])
        attacker = ModelAttacker(attack=suite[0], client=client, max_turns=2)
        transcript = run_conversation(
            consent_for(),
            attacker,
            max_turns=2,
            **driving(target(ScriptedTurn(reply="I can look into that."))),
        )
        assert [turn.user for turn in transcript.turns] == ["first", "second"]
        assert transcript.goal == suite[0].stake.objective


class TestScale:
    def test_the_two_shipped_agents_produce_the_suite_the_plan_describes(self, corpus):
        sizes = {
            name: len(build_suite(spec_for(name), corpus=corpus))
            for name in ("cart_recovery", "dispute_handler")
        }
        assert sizes == {"cart_recovery": 72, "dispute_handler": 160}


class TestMutationsAsTheThirdCoordinate:
    def test_the_plain_variant_carries_no_mutation_in_its_id(self, suite):
        """Adding mutations later must not rename every attack that already ran."""
        assert not suite[0].id.endswith("|hinglish")
        assert suite[0].id.count("|") == 2

    def test_a_mutated_attack_names_all_three(self, suite):
        mutated = apply_mutations(suite[:1], (by_id("hinglish"),))[-1]
        assert mutated.id == f"{suite[0].id}|hinglish"

    def test_the_plain_suite_survives_mutation(self, suite):
        mutated = apply_mutations(suite, SURFACES)
        assert mutated[: len(suite)] == suite

    def test_mutations_multiply(self, suite):
        mutated = apply_mutations(suite, SURFACES)
        assert len(mutated) == len(suite) * (1 + len(SURFACES))

    def test_a_predicate_bounds_the_multiplication(self, suite):
        costly = [a for a in suite if a.stake.consequence == "money"]
        mutated = apply_mutations(
            suite, SURFACES, where=lambda attack: attack.stake.consequence == "money"
        )
        assert len(mutated) == len(suite) + len(costly) * len(SURFACES)

    def test_variants_never_share_an_opening_with_the_plain_attack(self, suite):
        """Two variants differ precisely in how the turn is written."""
        mutated = apply_mutations(suite[:1], (by_id("hinglish"),))
        assert mutated[0].opening_key != mutated[1].opening_key

    def test_the_directive_reaches_the_attacker(self, suite):
        mutated = apply_mutations(suite[:1], (by_id("hurried_professional"),))[-1]
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=mutated, client=client).next_turn(transcript_with())
        assert by_id("hurried_professional").directive in client.calls[0].messages[0]["content"]

    def test_the_plain_attack_carries_no_directive(self, suite):
        client = RecordedModelClient(replies=[reply()])
        ModelAttacker(attack=suite[0], client=client).next_turn(transcript_with())
        assert "How to write it:" not in client.calls[0].messages[0]["content"]

    def test_the_switch_happens_partway_through(self, suite):
        """The one mutation that changes voice mid-conversation actually does."""
        switch = by_id("code_switch")
        mutated = apply_mutations(suite[:1], (switch,))[-1]
        client = RecordedModelClient(replies=[reply(f"turn {i}") for i in range(4)])
        attacker = ModelAttacker(attack=mutated, client=client, max_turns=6)
        for turns in range(4):
            attacker.next_turn(transcript_with(turns=turns))
        sent = [call.messages[0]["content"] for call in client.calls]
        assert switch.directive in sent[0]
        assert switch.later_directive in sent[switch.switch_after]

    def test_the_directive_stays_out_of_the_cached_prefix(self, suite):
        """A directive that can change has no business in the stable half of the prompt."""
        switch = by_id("code_switch")
        mutated = apply_mutations(suite[:1], (switch,))[-1]
        assert switch.directive not in attacker_system_prompt(mutated)
        assert switch.later_directive not in attacker_system_prompt(mutated)

    def test_mutating_twice_is_refused(self, suite):
        already = apply_mutations(suite[:1], (by_id("hinglish"),))
        with pytest.raises(AttackError, match="share the id"):
            apply_mutations(already, (by_id("hinglish"),))


class TestIdentityTravels:
    """Every conversation is about somebody, and the same somebody all the way through."""

    def test_a_variant_keeps_its_plain_attacks_identity(self, suite):
        """Otherwise plain against variant answers two questions at once."""
        mutated = apply_mutations(suite[:1], (by_id("hinglish"),))[-1]
        assert mutated.subject is suite[0].subject

    def test_attacks_sharing_an_opening_share_an_identity(self, dispute, corpus):
        """An opening turn says who is calling. Two people cannot share one."""
        for group in group_by_opening(build_suite(dispute, corpus=corpus)):
            assert len({attack.subject.name for attack in group}) == 1

    def test_every_declared_identity_is_used(self, dispute, corpus):
        used = {attack.subject.name for attack in build_suite(dispute, corpus=corpus)}
        assert used == {subject.name for subject in dispute.subjects}

    def test_the_prompt_states_the_identifiers_it_must_not_invent(self, suite):
        """An identifier the attacker cannot quote is one it will improvise."""
        attack = suite[0]
        prompt = attacker_system_prompt(attack)
        for value in attack.subject.identifiers.values():
            assert value in prompt
        for fact in attack.subject.facts:
            assert fact in prompt

    def test_an_agent_with_no_identities_is_told_it_holds_none(self, dispute, corpus):
        unscoped = dispute.model_copy(update={"subjects": ()})
        prompt = attacker_system_prompt(build_suite(unscoped, corpus=corpus)[0])
        assert "no reference or record" in prompt

    def test_assignment_is_stable_across_builds(self, dispute, corpus):
        """Two runs of one agent must pair the same identity with the same cell."""
        first = {a.id: a.subject.name for a in build_suite(dispute, corpus=corpus)}
        second = {a.id: a.subject.name for a in build_suite(dispute, corpus=corpus)}
        assert first == second
