"""The append-only record of what agents actually did.

This is the assertion substrate. Every check agent-red makes reduces to a question about
which tool was called, with which arguments, in which sequence, and what came back, and the
answer comes from here rather than from the agent's account of itself. ADR-0005 is the
argument; this module is the thing it decided to build.

Three properties are enforced rather than described, because each one is a way the record
could quietly stop being evidence.

**Append-only.** There is no method that edits or removes a record. A stream that can be
rewritten after the fact is a story about what happened.

**Copied on the way in.** A handler returns rows out of the world it is holding, and a later
call in the same conversation mutates that world. Storing the reference would mean an order
refunded at call seven appears refunded in the record of call three, and a detector reading
cumulative spend would be reading the end state at every position. Arguments and results are
deep copied at the moment they are recorded, so the record of a call is what that call saw.

**Keyed by run and session.** Two conversations against one agent are two separate streams,
because a rate computed across a shared stream measures the suite rather than the agent.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordedCall:
    """One tool invocation, as the tool server saw it.

    Attributes:
        run: The run this call belongs to. Two runs against one agent never share a stream.
        session: The conversation, or the planted attempt, that made the call.
        sequence: Position within `(run, session)`, starting at zero. Preconditions are
            broken by sequence, so this is part of the evidence and not a convenience.
        name: The declared tool name, without any transport prefix.
        arguments: Arguments exactly as they arrived. Full, never a signature: a leak can
            sit inside a call that is correct in every visible respect.
        result: What the tool returned, as it was returned.
        at: Unix timestamp when the call completed.
    """

    run: str
    session: str
    sequence: int
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    at: float

    def as_json(self) -> str:
        """One line of the stream."""
        return json.dumps(
            {
                "run": self.run,
                "session": self.session,
                "sequence": self.sequence,
                "name": self.name,
                "arguments": self.arguments,
                "result": self.result,
                "at": self.at,
            },
            sort_keys=True,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RecordedCall:
        """Rebuild a record from one line of the stream, or from the control API.

        Raises:
            ValueError: If a field the evidence depends on is missing or the wrong shape. A
                half-read record would be a call with no position, and a detector cannot
                tell that from a call that happened first.
        """
        try:
            return cls(
                run=str(payload["run"]),
                session=str(payload["session"]),
                sequence=int(payload["sequence"]),
                name=str(payload["name"]),
                arguments=dict(payload["arguments"]),
                result=dict(payload["result"]),
                at=float(payload["at"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"not a call record: {payload!r}") from error


@dataclass
class ToolCallRecorder:
    """The stream every recorded call is written to.

    Held by the tool server and by nothing else. A caller that could write to it could
    invent a call, which is the one thing the whole arrangement exists to prevent.

    Attributes:
        path: Where the stream is persisted, as JSON lines. `None` keeps it in memory only,
            which is what the tests use and what a run that never leaves one process needs.
    """

    path: Path | None = None
    _calls: dict[tuple[str, str], list[RecordedCall]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record(
        self,
        *,
        run: str,
        session: str,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        at: float | None = None,
    ) -> RecordedCall:
        """Append one call to the stream and return what was written.

        Args:
            run: The run the call belongs to.
            session: The conversation or planted attempt that made it.
            name: The declared tool name.
            arguments: Arguments as they arrived. Deep copied here.
            result: What the tool returned. Deep copied here.
            at: Unix timestamp. Defaults to now.

        Returns:
            The record, with the sequence number it was given.
        """
        with self._lock:
            stream = self._calls.setdefault((run, session), [])
            record = RecordedCall(
                run=run,
                session=session,
                sequence=len(stream),
                name=name,
                arguments=deepcopy(arguments),
                result=deepcopy(result),
                at=time.time() if at is None else at,
            )
            stream.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(record.as_json() + "\n")
            return record

    def calls(self, run: str, session: str) -> tuple[RecordedCall, ...]:
        """Every call one session made, in sequence.

        An empty stream is a real answer: an agent that was asked for money and called
        nothing is the outcome, not a missing record.
        """
        with self._lock:
            return tuple(self._calls.get((run, session), ()))

    def sessions(self, run: str) -> tuple[str, ...]:
        """Every session that has made at least one call in this run, in first-call order."""
        with self._lock:
            return tuple(session for recorded_run, session in self._calls if recorded_run == run)

    def __iter__(self) -> Iterator[RecordedCall]:
        """Every record held, session by session."""
        with self._lock:
            streams = [list(stream) for stream in self._calls.values()]
        for stream in streams:
            yield from stream


def read_stream(path: Path) -> tuple[RecordedCall, ...]:
    """Read a persisted stream back, in file order.

    For a reader that does not share a process with the server that wrote it: the run store,
    an operator inspecting a finished run, or a test asserting that what was served is what
    was written.

    Args:
        path: The JSON lines file.

    Returns:
        The records, in the order they were appended.

    Raises:
        ValueError: If a line is not a call record. A stream with one unreadable line is not
            partially trusted, because the missing line may be the call that mattered.
    """
    records: list[RecordedCall] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{number} is a {type(payload).__name__}, expected an object")
        records.append(RecordedCall.from_payload(payload))
    return tuple(records)
