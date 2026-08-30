"""A `ModelClient` that writes down every call it makes.

The transcript a run produces answers what was said and what the agent did about it. It does
not answer why the attacker said it, what prompt it was working from, what the model actually
returned before parsing, or what any of it cost. Those live only inside one `complete()` call
and are discarded the moment it returns.

That is a gap worth closing before the first real run rather than after it. A suite whose
attacks read weakly is a different problem depending on whether the prompt was thin, the model
declined, or the parse threw away half the answer, and none of the three is distinguishable
from the transcript alone.

This wraps any `ModelClient` and appends one JSON object per call to a file. It changes no
behaviour: `ModelClient` is a protocol, the wrapper satisfies it, and everything downstream
keeps calling `complete()`. Failures are recorded too, then re-raised, because a call that
raised is exactly the call someone will want to read about.

Writes are serialised by a lock, so several conversations may share one recorder.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentred.llm.client import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, ModelClient, ModelResponse

SCHEMA_VERSION = 1
"""Bumped when the shape of a record changes, so a reader can refuse what it cannot parse."""


@dataclass
class CallRecorder:
    """Appends one line of JSON per model call to a file.

    Attributes:
        path: Where records are appended. Created with its parents on first write.
        label: Free text identifying what is being recorded, copied onto every record. The
            conversation this call belongs to, in practice, which is what makes a shared
            recorder readable afterwards.
        sequence: How many calls have been recorded. Written onto each record so the file
            can be put back in order even when several threads interleave.
    """

    path: Path
    label: str = ""
    sequence: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the containing directory, so a long run does not fail on its last write."""
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        """Append one record, stamped and numbered.

        Args:
            record: The record's own fields. `schema`, `sequence` and `at` are added here
                and overwrite anything of those names in the record.
        """
        with self._lock:
            self.sequence += 1
            stamped = {
                **record,
                "schema": SCHEMA_VERSION,
                "sequence": self.sequence,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stamped, default=str) + "\n")


class RecordingModelClient:
    """A `ModelClient` that records what it was asked and what came back.

    Attributes:
        inner: The client doing the actual work.
        recorder: Where records go. Shared between conversations when they share a run.
        label: What to tag this client's records with, overriding the recorder's own label.
            One per conversation, so a shared file can be split by attack afterwards.
    """

    def __init__(self, inner: ModelClient, recorder: CallRecorder, *, label: str = "") -> None:
        """Wrap a client.

        Args:
            inner: The client to delegate to.
            recorder: The record sink.
            label: Tag for this client's records. Defaults to the recorder's label.
        """
        self.inner = inner
        self.recorder = recorder
        self.label = label or recorder.label

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        output_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Delegate, record, return. See `ModelClient.complete`.

        The full system prompt and every message are recorded verbatim rather than
        summarised. They are the evidence for whether an attack was weak because the model
        was weak or because the prompt was, and a summary cannot answer that.

        Raises:
            Exception: Whatever the wrapped client raised, after recording it.
        """
        started = time.monotonic()
        request = {
            "label": self.label,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "effort": effort,
            "output_schema": output_schema,
        }
        try:
            response = self.inner.complete(
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                effort=effort,
                output_schema=output_schema,
            )
        except Exception as error:
            self.recorder.write(
                {
                    **request,
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
            raise
        self.recorder.write(
            {
                **request,
                "ok": True,
                "text": response.text,
                "stop_reason": response.stop_reason,
                "model": response.model,
                "retries": response.retries,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": response.usage.cache_read_tokens,
                },
                "seconds": round(time.monotonic() - started, 3),
            }
        )
        return response


def read_records(path: Path | str) -> tuple[dict[str, Any], ...]:
    """Load a recording, in the sequence the calls were made.

    Args:
        path: The JSONL file a `CallRecorder` wrote.

    Returns:
        The records, sorted by `sequence`. Sorting rather than trusting file order is the
        point: concurrent conversations append interleaved, and a report that reads them in
        file order attributes turns to the wrong conversation.

    Raises:
        FileNotFoundError: If nothing was recorded there.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return tuple(sorted(records, key=lambda record: record.get("sequence", 0)))
