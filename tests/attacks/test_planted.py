"""Hand-written payloads: what the loader accepts, and what it refuses at load."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentred.attacks.generator import Attack, AttackError, PlantedPayload, build_attackers
from agentred.attacks.planted import PayloadError, load_planted
from agentred.attacks.stakes import derive_stakes
from agentred.spec import load_spec_dir

SPEC_ROOT = "src/agentred/targets/specs"
AGENT = "dispute_handler"


@pytest.fixture
def spec():
    return load_spec_dir(f"{SPEC_ROOT}/{AGENT}")


@pytest.fixture
def shipped(tmp_path, spec):
    """The checked-in payload file, copied so a test can edit one entry of it."""
    source = yaml.safe_load(Path(f"data/planted/{AGENT}.yaml").read_text(encoding="utf-8"))

    def write(mutate=None):
        document = yaml.safe_load(yaml.safe_dump(source))
        if mutate is not None:
            mutate(document)
        (tmp_path / f"{AGENT}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        return tmp_path

    return write


class TestTheShippedPayloads:
    def test_all_three_load_against_the_agent_they_were_written_for(self, spec):
        attacks = load_planted(spec)
        assert len(attacks) == 3
        assert {attack.channel for attack in attacks} == {"dispute_reason_text"}

    def test_each_one_is_aimed_at_a_stake_the_policy_derives(self, spec):
        derived = {stake.id for stake in derive_stakes(spec)}
        assert all(attack.stake.id in derived for attack in load_planted(spec))

    def test_each_one_says_in_plain_words_what_the_person_did(self, spec):
        """A merchant page must never show a bare label; the same rule as a technique's."""
        for attack in load_planted(spec):
            assert len(attack.planted.in_plain_words) > 60
            assert attack.planted.label not in attack.planted.in_plain_words

    def test_they_are_aimed_at_three_different_stakes(self, spec):
        """Three payloads on one stake would test one thing and report three cells."""
        assert len({attack.stake.id for attack in load_planted(spec)}) == 3

    def test_an_agent_with_no_payload_file_gets_none(self, tmp_path, spec):
        assert load_planted(spec, tmp_path) == ()


class TestWhatItRefusesAtLoad:
    """Each of these would otherwise plant nowhere and report as a cell that was tested."""

    def test_a_channel_the_agent_does_not_declare(self, shipped, spec):
        directory = shipped(lambda d: d["payloads"][0].update(channel="account_name"))
        with pytest.raises(PayloadError, match="does not declare"):
            load_planted(spec, directory)

    def test_a_stake_nothing_derives(self, shipped, spec):
        directory = shipped(lambda d: d["payloads"][0].update(stake="bound_exceeded:nope:x:above"))
        with pytest.raises(PayloadError, match="derives"):
            load_planted(spec, directory)

    def test_a_subject_nobody_declared(self, shipped, spec):
        directory = shipped(lambda d: d["payloads"][0].update(subject="somebody-else"))
        with pytest.raises(PayloadError, match="declares no subject"):
            load_planted(spec, directory)

    def test_a_payload_with_no_plain_words_line(self, shipped, spec):
        directory = shipped(lambda d: d["payloads"][0].update(in_plain_words=" "))
        with pytest.raises(PayloadError, match="in_plain_words"):
            load_planted(spec, directory)

    def test_a_payload_with_no_text(self, shipped, spec):
        directory = shipped(lambda d: d["payloads"][0].update(text=""))
        with pytest.raises(PayloadError, match="declares no text"):
            load_planted(spec, directory)

    def test_a_file_written_for_a_different_agent(self, shipped, spec):
        directory = shipped(lambda d: d.update(agent_id="somebody_elses_agent"))
        with pytest.raises(PayloadError, match="not about the same agent"):
            load_planted(spec, directory)

    def test_two_payloads_that_would_share_an_id(self, shipped, spec):
        def duplicate(document):
            document["payloads"].append(yaml.safe_load(yaml.safe_dump(document["payloads"][0])))

        with pytest.raises(PayloadError, match="two attacks with id"):
            load_planted(spec, shipped(duplicate))


class TestTheAttackShape:
    """An attack's channel decides which driver runs it, so the two cannot be mixed up."""

    def test_a_planted_attack_says_it_is_planted(self, spec):
        assert all(attack.is_planted for attack in load_planted(spec))

    def test_a_planted_attack_carries_the_channel_in_its_id(self, spec):
        for attack in load_planted(spec):
            assert f"|{attack.channel}:" in attack.id

    def test_a_conversational_attack_id_is_unchanged_by_channels_existing(self, spec):
        """An id already in the frozen corpus must not be renamed by this milestone."""
        from agentred.attacks.generator import build_suite

        for attack in build_suite(spec)[:5]:
            assert "|conversation" not in attack.id

    def test_a_planted_attack_is_refused_by_the_conversational_attacker_builder(self, spec):
        with pytest.raises(AttackError, match="composes no turns"):
            build_attackers(load_planted(spec), client=None)

    def test_a_conversational_attack_with_no_technique_is_refused(self, spec):
        stake = derive_stakes(spec)[0]
        with pytest.raises(AttackError, match="nothing to say"):
            Attack(stake=stake)

    def test_a_planted_attack_with_no_payload_is_refused(self, spec):
        stake = derive_stakes(spec)[0]
        with pytest.raises(AttackError, match="carries no payload"):
            Attack(stake=stake, channel="dispute_reason_text")

    def test_a_conversational_attack_carrying_a_payload_is_refused(self, spec):
        from agentred.attacks.techniques import load_corpus

        stake = derive_stakes(spec)[0]
        with pytest.raises(AttackError, match="which nothing would write"):
            Attack(
                stake=stake,
                technique=load_corpus()[0],
                planted=PlantedPayload(label="l", record_id="r", text="t", in_plain_words="w" * 40),
            )


class TestTheScheduledAgentsPayloads:
    """The agent nobody talks to, which is the whole reason this channel exists."""

    def _spec(self):
        return load_spec_dir(f"{SPEC_ROOT}/cart_recovery")

    def test_they_load_against_the_agent_they_were_written_for(self):
        assert len(load_planted(self._spec())) == 3

    def test_every_one_of_them_fires_on_a_schedule_with_no_user_turn(self):
        from agentred.spec.models import TriggerKind

        channels = self._spec().config.channels_by_name
        for attack in load_planted(self._spec()):
            assert channels[attack.channel].trigger is TriggerKind.SCHEDULE

    def test_they_reach_more_than_one_channel(self):
        """One channel exercised twice would report two cells and test one surface."""
        assert len({attack.channel for attack in load_planted(self._spec())}) == 2
