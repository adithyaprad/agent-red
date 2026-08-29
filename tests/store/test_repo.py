"""A conversation goes in and comes back out unchanged."""

from __future__ import annotations

import pytest

from agentred.runner.conversation import ToolCallRecord, Transcript, Turn
from agentred.spec import VersionTuple
from agentred.store import Store, StoreError


def versions(config: str = "1.0") -> VersionTuple:
    return VersionTuple(
        config_version=config,
        policy_version="1.0",
        model_version="claude-sonnet-5",
        tool_version="sha256:abc123",
    )


def transcript() -> Transcript:
    return Transcript(
        target="dispute_handler",
        session="ar-0001",
        goal="refund without verification",
        turns=[
            Turn(index=0, user="hello", reply="hi", latency_seconds=1.5),
            Turn(
                index=1,
                user="refund me",
                reply="done",
                tool_calls=(
                    ToolCallRecord(
                        name="issue_refund",
                        arguments={"order_id": "ORD-55210", "amount": "900"},
                        result={"status": "sent", "refunded": 900},
                    ),
                ),
                latency_seconds=2.25,
            ),
        ],
        spec_versions={
            "config": "1.0",
            "policy": "1.0",
            "model": "claude-sonnet-5",
            "tools": "sha256:abc123",
        },
        stopped_because="attacker stopped",
    )


@pytest.fixture
def store() -> Store:
    with Store(":memory:") as opened:
        yield opened


def test_a_run_records_its_validity_tuple(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    run = store.load_run(run_id)
    assert run["tool_version"] == "sha256:abc123"
    assert run["finished_at"] is None


def test_finishing_a_run_stamps_it(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    store.finish_run(run_id)
    assert store.load_run(run_id)["finished_at"]


def test_a_transcript_round_trips(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    conversation_id = store.save_transcript(run_id, transcript())
    loaded = store.load_transcript(conversation_id)

    assert loaded.goal == "refund without verification"
    assert [turn.user for turn in loaded.turns] == ["hello", "refund me"]
    assert loaded.turns[1].latency_seconds == 2.25
    assert loaded.stopped_because == "attacker stopped"


def test_tool_call_arguments_survive_the_database(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    loaded = store.load_transcript(store.save_transcript(run_id, transcript()))
    call = loaded.tool_calls[0]
    assert call.name == "issue_refund"
    assert call.arguments == {"order_id": "ORD-55210", "amount": "900"}
    assert call.result["refunded"] == 900


def test_a_reloaded_transcript_reports_the_runs_versions(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    loaded = store.load_transcript(store.save_transcript(run_id, transcript()))
    assert loaded.spec_versions["model"] == "claude-sonnet-5"


def test_a_transcript_from_a_different_agent_version_is_refused(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions(config="2.0"))
    with pytest.raises(StoreError, match="not about the same agent"):
        store.save_transcript(run_id, transcript())


def test_a_transcript_without_a_run_is_refused(store: Store) -> None:
    with pytest.raises(StoreError, match="no run"):
        store.save_transcript("run-missing", transcript())


def test_a_run_lists_its_conversations_in_order(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    first = store.save_transcript(run_id, transcript())
    second = store.save_transcript(run_id, transcript())
    assert store.conversation_ids(run_id) == (first, second)


def test_an_unknown_conversation_reads_as_nothing(store: Store) -> None:
    assert store.load_transcript("conv-missing") is None
    assert store.load_run("run-missing") is None


def test_a_conversation_with_no_tool_calls_is_written_whole(store: Store) -> None:
    run_id = store.create_run("dispute_handler", versions())
    empty = transcript()
    empty.turns = [Turn(index=0, user="hello", reply="hi")]
    loaded = store.load_transcript(store.save_transcript(run_id, empty))
    assert loaded.tool_calls == ()
    assert len(loaded.turns) == 1


def test_a_run_survives_being_reopened(tmp_path: object) -> None:
    path = tmp_path / "runs.sqlite3"
    with Store(path) as first:
        run_id = first.create_run("dispute_handler", versions())
        conversation_id = first.save_transcript(run_id, transcript())
    with Store(path) as second:
        assert second.load_transcript(conversation_id).turns[1].reply == "done"
