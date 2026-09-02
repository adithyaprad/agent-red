"""A run keeps what it finished, even when it does not finish.

Run 0006 reached 53 of 88 conversations over ninety minutes and was then interrupted. Every
one of those conversations was lost, because transcripts were written to the store only once
the whole pool had drained, so the work existed nowhere a later command could read. The suite
had to be paid for and run again from nothing.

These tests are about the fix rather than the feature: transcripts land one at a time as they
complete, and an interruption carries back the run so far instead of only an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentred.attacks.generator import Attack, build_suite
from agentred.runner import suite as suite_module
from agentred.runner.channels.conversational import Transcript
from agentred.runner.suite import Outcome, SuiteRun, execute, persist
from agentred.spec import load_spec_dir
from agentred.store.repo import Store

SPEC_DIR = Path("src/agentred/targets/specs/dispute_handler")


class StubLease:
    """Stands in for the consent gate. Reaching a target is not what is under test here."""

    def __init__(self, name: str, **_: Any) -> None:
        self.name = name
        self.nonces = ["stub-nonce"]

    def token(self, now: float | None = None) -> object:
        return object()


def transcript_for(attack: Attack) -> Transcript:
    """A minimal finished conversation for one attack."""
    return Transcript(
        target="dispute_handler",
        session=f"ar-{attack.id[:16]}",
        goal=attack.goal,
        subject=dict(attack.subject.identifiers) if attack.subject is not None else None,
    )


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run `execute` with nothing that reaches a network."""
    monkeypatch.setattr(suite_module, "ConsentLease", StubLease)
    monkeypatch.setattr(suite_module, "AnthropicModelClient", lambda **_: object())
    monkeypatch.setattr(
        suite_module, "build_attackers", lambda attacks, client, max_turns: [None] * len(attacks)
    )


def attacks(count: int) -> tuple[Attack, ...]:
    return build_suite(load_spec_dir(SPEC_DIR))[:count]


def test_every_completed_transcript_is_in_the_store(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        suite_module,
        "run_one",
        lambda attack, attacker, lease, max_turns, run, channels=None, kinds=(): Outcome(
            attack=attack, transcript=transcript_for(attack)
        ),
    )
    store_path = tmp_path / "runs.db"
    run = execute(
        attacks(6),
        target="dispute_handler",
        model="claude-sonnet-5",
        stake="",
        max_turns=2,
        concurrency=3,
        recording=tmp_path / "calls.jsonl",
        store_path=store_path,
        number="0099",
    )
    with Store(store_path) as store:
        assert len(store.conversation_ids(run.run_id)) == 6


def test_an_interrupted_run_keeps_the_conversations_that_finished(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The defect run 0006 exposed. Interrupting used to leave nothing behind."""
    done: list[str] = []

    def stop_after_three(
        attack: Attack,
        attacker: Any,
        lease: Any,
        max_turns: int,
        run: str,
        channels: Any = None,
        kinds: tuple[str, ...] = (),
    ) -> Outcome:
        if len(done) >= 3:
            raise KeyboardInterrupt
        done.append(attack.id)
        return Outcome(attack=attack, transcript=transcript_for(attack))

    monkeypatch.setattr(suite_module, "run_one", stop_after_three)
    store_path = tmp_path / "runs.db"

    with pytest.raises(KeyboardInterrupt) as stopped:
        execute(
            attacks(8),
            target="dispute_handler",
            model="claude-sonnet-5",
            stake="",
            max_turns=2,
            concurrency=1,
            recording=tmp_path / "calls.jsonl",
            store_path=store_path,
            number="0099",
        )

    partial = stopped.value.args[0]
    assert isinstance(partial, SuiteRun)
    assert len(partial.outcomes) == 3
    assert partial.run_id
    with Store(store_path) as store:
        assert len(store.conversation_ids(partial.run_id)) == 3


def test_an_interrupted_run_is_still_marked_finished(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An open run row would make the partial results look like a run still in flight."""

    def stop_immediately(
        attack: Attack,
        attacker: Any,
        lease: Any,
        max_turns: int,
        run: str,
        channels: Any = None,
        kinds: tuple[str, ...] = (),
    ) -> Outcome:
        raise KeyboardInterrupt

    monkeypatch.setattr(suite_module, "run_one", stop_immediately)
    store_path = tmp_path / "runs.db"
    with pytest.raises(KeyboardInterrupt) as stopped:
        execute(
            attacks(4),
            target="dispute_handler",
            model="claude-sonnet-5",
            stake="",
            max_turns=2,
            concurrency=1,
            recording=tmp_path / "calls.jsonl",
            store_path=store_path,
            number="0099",
        )
    run_id = stopped.value.args[0].run_id
    with Store(store_path) as store:
        assert store.load_run(run_id) is not None


def test_outcomes_come_back_in_suite_sequence_not_completion_sequence(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transcripts land as they finish, and a report that reordered them would not be stable."""
    ordered = attacks(6)
    backwards = {attack.id: index for index, attack in enumerate(reversed(ordered))}

    def finish_in_reverse(
        attack: Attack,
        attacker: Any,
        lease: Any,
        max_turns: int,
        run: str,
        channels: Any = None,
        kinds: tuple[str, ...] = (),
    ) -> Outcome:
        import time

        time.sleep(backwards[attack.id] * 0.01)
        return Outcome(attack=attack, transcript=transcript_for(attack))

    monkeypatch.setattr(suite_module, "run_one", finish_in_reverse)
    run = execute(
        ordered,
        target="dispute_handler",
        model="claude-sonnet-5",
        stake="",
        max_turns=2,
        concurrency=6,
        recording=tmp_path / "calls.jsonl",
    )
    assert [o.attack.id for o in run.outcomes] == [a.id for a in ordered]


def test_the_run_number_is_recorded_before_the_first_transcript_is_written(
    offline: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The note beside the transcripts cites the run number, so it cannot arrive afterwards."""
    monkeypatch.setattr(
        suite_module,
        "run_one",
        lambda attack, attacker, lease, max_turns, run, channels=None, kinds=(): Outcome(
            attack=attack, transcript=transcript_for(attack)
        ),
    )
    store_path = tmp_path / "runs.db"
    run = execute(
        attacks(2),
        target="dispute_handler",
        model="claude-sonnet-5",
        stake="",
        max_turns=2,
        concurrency=1,
        recording=tmp_path / "calls.jsonl",
        store_path=store_path,
        number="0099",
    )
    assert run.number == "0099"
    with Store(store_path) as store:
        assert "run 0099" in (store.load_run(run.run_id) or {})["notes"]


def test_persist_does_not_write_a_second_copy_of_an_already_stored_run(tmp_path: Path) -> None:
    """`execute` persists as it goes, so the older whole-run write must stand down."""
    run = SuiteRun(
        target="dispute_handler",
        model="claude-sonnet-5",
        stake="",
        max_turns=2,
        concurrency=1,
        run_id="run-already-written",
    )
    run.outcomes = [
        Outcome(attack=attack, transcript=transcript_for(attack)) for attack in attacks(2)
    ]
    store_path = tmp_path / "runs.db"
    persist(run, store_path)
    with Store(store_path) as store:
        assert store.load_run("run-already-written") is None


class TestChannelDispatch:
    """One attack, one channel, one driver. Getting this wrong is silent both ways."""

    def _spec(self):
        from agentred.spec import load_spec_dir

        return load_spec_dir("src/agentred/targets/specs/dispute_handler")

    def _planted_attack(self):
        from agentred.attacks.planted import load_planted

        return load_planted(self._spec())[0]

    def _lease(self):
        class Lease:
            def token(self):
                return object()

        return Lease()

    def test_a_planted_attack_goes_to_the_planted_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attack = self._planted_attack()
        seen: dict[str, object] = {}

        def fake_planted(token, channel, payload, **kwargs):
            seen["channel"] = channel.name
            seen["payload"] = payload
            seen["record_id"] = kwargs["record_id"]
            return transcript_for(attack)

        def never(*args, **kwargs):
            raise AssertionError("a planted attack was sent down the conversation")

        monkeypatch.setattr(suite_module, "run_planted", fake_planted)
        monkeypatch.setattr(suite_module, "run_conversation", never)

        outcome = suite_module.run_one(
            attack, None, self._lease(), 6, "run-x", self._spec().config.channels_by_name
        )

        assert outcome.error == ""
        assert seen["channel"] == "dispute_reason_text"
        assert seen["record_id"] == attack.planted.record_id
        assert seen["payload"] == attack.planted.text

    def test_a_conversational_attack_goes_to_the_conversation_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attack = attacks(1)[0]

        def never(*args, **kwargs):
            raise AssertionError("a conversational attack was planted")

        monkeypatch.setattr(suite_module, "run_planted", never)
        monkeypatch.setattr(
            suite_module, "run_conversation", lambda *a, **k: transcript_for(attack)
        )

        outcome = suite_module.run_one(attack, None, self._lease(), 6, "run-x", {})
        assert outcome.error == ""

    def test_a_channel_the_agent_does_not_declare_is_an_error_not_a_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a mis-wired channel reads on the scorecard as an attack that held."""
        attack = self._planted_attack()
        monkeypatch.setattr(suite_module, "run_planted", lambda *a, **k: transcript_for(attack))

        outcome = suite_module.run_one(attack, None, self._lease(), 6, "run-x", {})

        assert outcome.transcript is None
        assert "does not declare" in outcome.error
        assert not outcome.ok
