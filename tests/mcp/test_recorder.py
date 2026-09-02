"""The append-only call stream: sequence, isolation, copying, and reading it back."""

from __future__ import annotations

import pytest

from agentred.mcp.recorder import ToolCallRecorder, read_stream


def test_calls_are_numbered_in_the_order_they_arrive() -> None:
    recorder = ToolCallRecorder()
    for amount in (10, 20, 30):
        recorder.record(
            run="r1", session="s1", name="issue_refund", arguments={"amount": amount}, result={}
        )
    assert [call.sequence for call in recorder.calls("r1", "s1")] == [0, 1, 2]
    assert [call.arguments["amount"] for call in recorder.calls("r1", "s1")] == [10, 20, 30]


def test_two_sessions_do_not_share_a_stream() -> None:
    recorder = ToolCallRecorder()
    recorder.record(run="r1", session="s1", name="a", arguments={}, result={})
    recorder.record(run="r1", session="s2", name="b", arguments={}, result={})
    assert [call.name for call in recorder.calls("r1", "s1")] == ["a"]
    assert [call.name for call in recorder.calls("r1", "s2")] == ["b"]


def test_one_session_id_in_two_runs_is_two_streams() -> None:
    recorder = ToolCallRecorder()
    recorder.record(run="r1", session="s1", name="a", arguments={}, result={})
    recorder.record(run="r2", session="s1", name="b", arguments={}, result={})
    assert [call.name for call in recorder.calls("r1", "s1")] == ["a"]
    assert [call.name for call in recorder.calls("r2", "s1")] == ["b"]


def test_a_session_that_called_nothing_has_an_empty_stream() -> None:
    assert ToolCallRecorder().calls("r1", "s1") == ()


def test_a_record_keeps_what_the_call_saw_when_the_world_moves_on() -> None:
    """The reason arguments and results are copied on the way in.

    A handler returns rows out of a world that later calls mutate. Storing the reference
    would mean a detector reading call three sees the state at call seven, and cumulative
    spend would read as the end total at every position.
    """
    recorder = ToolCallRecorder()
    order = {"order_id": "ORD-1", "refunded_to_date": 0}
    recorder.record(run="r1", session="s1", name="lookup_order", arguments={}, result=order)
    order["refunded_to_date"] = 850

    assert recorder.calls("r1", "s1")[0].result["refunded_to_date"] == 0


def test_arguments_are_copied_too() -> None:
    recorder = ToolCallRecorder()
    arguments = {"lines": [{"sku": "SKU-1"}]}
    recorder.record(run="r1", session="s1", name="x", arguments=arguments, result={})
    arguments["lines"][0]["sku"] = "SKU-CHANGED"

    assert recorder.calls("r1", "s1")[0].arguments["lines"][0]["sku"] == "SKU-1"


def test_the_recorder_offers_no_way_to_edit_or_remove_a_record() -> None:
    """Append-only is a property of the interface, not a convention."""
    surface = {name for name in dir(ToolCallRecorder) if not name.startswith("_")}
    assert not surface & {"delete", "remove", "update", "edit", "clear", "pop", "truncate"}


def test_a_persisted_stream_reads_back_as_it_was_written(tmp_path) -> None:
    path = tmp_path / "calls.jsonl"
    recorder = ToolCallRecorder(path=path)
    recorder.record(
        run="r1",
        session="s1",
        name="issue_refund",
        arguments={"order_id": "ORD-1", "amount": 850.5},
        result={"refunded": 850.5},
    )
    recorder.record(run="r1", session="s2", name="send_message", arguments={}, result={})

    records = read_stream(path)
    assert [(record.session, record.name, record.sequence) for record in records] == [
        ("s1", "issue_refund", 0),
        ("s2", "send_message", 0),
    ]
    assert records[0].arguments == {"order_id": "ORD-1", "amount": 850.5}


def test_a_stream_with_an_unreadable_line_is_refused_rather_than_half_trusted(tmp_path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text('{"run": "r1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a call record"):
        read_stream(path)


def test_sessions_lists_every_session_that_called_something() -> None:
    recorder = ToolCallRecorder()
    recorder.record(run="r1", session="s1", name="a", arguments={}, result={})
    recorder.record(run="r1", session="s2", name="a", arguments={}, result={})
    recorder.record(run="r2", session="s3", name="a", arguments={}, result={})
    assert recorder.sessions("r1") == ("s1", "s2")
    assert recorder.sessions("r2") == ("s3",)
