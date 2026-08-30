"""The call recorder. Offline: the wrapped client is a fake."""

import json
import threading

import pytest

from agentred.llm.client import ModelClient, ModelResponse, Usage
from agentred.llm.recording import CallRecorder, RecordingModelClient, read_records
from tests.fakes.model import RecordedModelClient


class Exploding:
    """A client that always raises, for the failure path."""

    def complete(self, **kwargs):
        raise RuntimeError("the route was unreachable")


@pytest.fixture
def recorder(tmp_path):
    return CallRecorder(tmp_path / "nested" / "calls.jsonl", label="attack-1")


class TestItStaysAModelClient:
    def test_the_wrapper_satisfies_the_protocol(self, recorder):
        """Nothing downstream may need to know it is being recorded."""
        client = RecordingModelClient(RecordedModelClient(replies=["hi"]), recorder)
        assert isinstance(client, ModelClient)

    def test_the_response_is_passed_through_unchanged(self, recorder):
        inner = RecordedModelClient(replies=["the composed turn"])
        client = RecordingModelClient(inner, recorder)
        assert client.complete(system="s", messages=[]).text == "the composed turn"

    def test_the_inner_client_sees_every_argument(self, recorder):
        inner = RecordedModelClient(replies=["x"])
        client = RecordingModelClient(inner, recorder)
        schema = {"type": "object"}
        client.complete(system="s", messages=[], max_tokens=99, effort="low", output_schema=schema)
        call = inner.calls[0]
        assert (call.max_tokens, call.effort, call.output_schema) == (99, "low", schema)


class TestWhatIsRecorded:
    def test_the_whole_prompt_is_kept_verbatim(self, recorder):
        """A summary cannot answer whether a weak attack came from a thin prompt."""
        client = RecordingModelClient(RecordedModelClient(replies=["x"]), recorder)
        client.complete(system="be persuasive", messages=[{"role": "user", "content": "turn 1"}])
        record = read_records(recorder.path)[0]
        assert record["system"] == "be persuasive"
        assert record["messages"] == [{"role": "user", "content": "turn 1"}]

    def test_the_reply_cost_and_label_are_kept(self, recorder):
        client = RecordingModelClient(RecordedModelClient(replies=["answer"]), recorder)
        client.complete(system="s", messages=[])
        record = read_records(recorder.path)[0]
        assert record["ok"] is True
        assert record["text"] == "answer"
        assert record["label"] == "attack-1"
        assert "input_tokens" in record["usage"]
        assert record["seconds"] >= 0

    def test_a_label_can_be_set_per_conversation(self, recorder):
        """One shared file, split by attack afterwards."""
        RecordingModelClient(
            RecordedModelClient(replies=["x"]), recorder, label="attack-2"
        ).complete(system="s", messages=[])
        assert read_records(recorder.path)[0]["label"] == "attack-2"

    def test_the_directory_is_created(self, tmp_path):
        CallRecorder(tmp_path / "a" / "b" / "calls.jsonl")
        assert (tmp_path / "a" / "b").is_dir()


class TestFailures:
    def test_a_failed_call_is_recorded_then_raised(self, recorder):
        """The call that raised is the one someone will want to read about."""
        client = RecordingModelClient(Exploding(), recorder)
        with pytest.raises(RuntimeError):
            client.complete(system="s", messages=[])
        record = read_records(recorder.path)[0]
        assert record["ok"] is False
        assert record["error_type"] == "RuntimeError"
        assert "unreachable" in record["error"]

    def test_the_failing_prompt_is_kept_too(self, recorder):
        client = RecordingModelClient(Exploding(), recorder)
        with pytest.raises(RuntimeError):
            client.complete(system="the prompt that failed", messages=[])
        assert read_records(recorder.path)[0]["system"] == "the prompt that failed"


class TestConcurrency:
    def test_records_from_several_threads_are_whole_and_numbered(self, recorder):
        """Interleaved appends must not produce a half-written line."""

        def run(index):
            inner = RecordedModelClient(replies=["x"] * 10)
            client = RecordingModelClient(inner, recorder, label=f"attack-{index}")
            for _ in range(10):
                client.complete(system="s", messages=[])

        threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = recorder.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 40
        for line in lines:
            json.loads(line)
        assert [r["sequence"] for r in read_records(recorder.path)] == list(range(1, 41))

    def test_reading_sorts_by_sequence_not_file_order(self, recorder):
        """Concurrent conversations append interleaved; file order attributes turns wrongly."""
        recorder.path.write_text(
            "\n".join(json.dumps({"sequence": n, "label": str(n)}) for n in (3, 1, 2)),
            encoding="utf-8",
        )
        assert [r["label"] for r in read_records(recorder.path)] == ["1", "2", "3"]


class TestRetriesSurface:
    def test_a_retry_count_reaches_the_record(self, recorder):
        """Recorded so a slow run reads as throttling rather than as a slow model."""

        class Throttled:
            def complete(self, **kwargs):
                return ModelResponse(
                    text="x", stop_reason="end_turn", model="m", usage=Usage(), retries=2
                )

        RecordingModelClient(Throttled(), recorder).complete(system="s", messages=[])
        assert read_records(recorder.path)[0]["retries"] == 2
