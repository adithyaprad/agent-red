"""An analysis is of named runs, not of whatever a database happens to hold."""

from __future__ import annotations

import pytest

from agentred.runner.channels.conversational import Transcript, Turn
from agentred.scoring.analysis import (
    AnalysisError,
    known_runs,
    load_conversations,
    resolve_runs,
)
from agentred.spec import VersionTuple
from agentred.store import Store

VERSIONS = VersionTuple(
    config_version="1.0",
    policy_version="1.1",
    model_version="claude-sonnet-5",
    tool_version="sha256:abc123",
)


def transcript(session: str) -> Transcript:
    return Transcript(
        target="dispute_handler",
        session=session,
        goal="refund without verification",
        turns=[Turn(index=0, user="hello", reply="hi")],
        spec_versions={
            "config": "1.0",
            "policy": "1.1",
            "model": "claude-sonnet-5",
            "tools": "sha256:abc123",
        },
    )


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as opened:
        yield opened


def seed(store: Store, target: str, sessions: tuple[str, ...], notes: str = "") -> str:
    run_id = store.create_run(target, VERSIONS, notes=notes)
    for session in sessions:
        store.save_transcript(run_id, transcript(session))
    store.finish_run(run_id)
    return run_id


class TestChoosingRuns:
    def test_no_runs_named_selects_every_run(self, store):
        first = seed(store, "dispute_handler", ("a",))
        second = seed(store, "dispute_handler", ("b",))
        assert [row["run_id"] for row in resolve_runs(store, ())] == [first, second]

    def test_a_named_run_selects_only_that_run(self, store):
        seed(store, "dispute_handler", ("a",))
        wanted = seed(store, "dispute_handler", ("b",))
        assert [row["run_id"] for row in resolve_runs(store, (wanted,))] == [wanted]

    def test_an_unknown_run_is_fatal_rather_than_skipped(self, store):
        """A run that quietly contributed nothing looks, from the page, exactly like a run
        whose agent never broke a rule. The second is a claim and the first is a typo."""
        real = seed(store, "dispute_handler", ("a",))
        with pytest.raises(AnalysisError, match="no run 'run-nope'"):
            resolve_runs(store, (real, "run-nope"))

    def test_the_refusal_lists_what_is_actually_there(self, store):
        seed(store, "cart_recovery", ("a",), notes="run 0007, stake=all")
        with pytest.raises(AnalysisError, match="run 0007"):
            resolve_runs(store, ("run-nope",))

    def test_an_empty_store_is_refused_rather_than_analysed(self, store):
        with pytest.raises(AnalysisError, match="no runs"):
            resolve_runs(store, ())


class TestReadingConversations:
    def test_only_the_selected_runs_are_read(self, store):
        """The whole reason the filter exists: two runs pooled under one denominator is a
        page describing a database rather than an agent."""
        seed(store, "dispute_handler", ("a", "b"))
        wanted = seed(store, "dispute_handler", ("c",))
        loaded = load_conversations(store, resolve_runs(store, (wanted,)))
        assert [t.session for _, _, t in loaded] == ["c"]

    def test_each_conversation_carries_the_run_it_came_from(self, store):
        run_id = seed(store, "dispute_handler", ("a",))
        (found,) = load_conversations(store, resolve_runs(store, ()))
        assert found[0] == run_id
        assert found[1] == "dispute_handler"


class TestWhatARunRecords:
    def test_a_run_carries_the_validity_tuple(self, store):
        """A scorecard is valid for exactly four versions, so a page that cannot name them
        reads as a statement about an agent rather than about one version of one."""
        seed(store, "dispute_handler", ("a",))
        (row,) = known_runs(store)
        assert row["config_version"] == "1.0"
        assert row["policy_version"] == "1.1"
        assert row["model_version"] == "claude-sonnet-5"
        assert row["tool_version"] == "sha256:abc123"

    def test_the_note_survives_so_a_run_number_can_be_resolved(self, store):
        seed(store, "cart_recovery", ("a",), notes="run 0007, stake=all, attacker=x")
        assert known_runs(store)[0]["notes"].startswith("run 0007")
