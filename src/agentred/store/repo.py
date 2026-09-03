"""Reading and writing runs, conversations and their tool calls.

One SQLite file per installation, created on first use. The store is deliberately thin: it
holds what happened, not what it meant. A verdict is a judgement about a stored transcript
and is written by `judge/`, in a later milestone, against tables this module's schema
already declares.

Two invariants are enforced here rather than trusted to callers. A run cannot be created
without its four versions, because a transcript that cannot be attributed to a version of
the agent is not evidence. And a conversation is written in one transaction with its turns
and tool calls, because a half-written conversation reads on the scorecard as an agent that
did less than it did.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentred.runner.channels.conversational import (
    PlantedField,
    ToolCallRecord,
    Transcript,
    Turn,
)
from agentred.spec import VersionTuple
from agentred.spec.models import CONVERSATIONAL_CHANNEL

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path("runs.sqlite3")


class StoreError(RuntimeError):
    """A write was refused, or a read found something that should not exist."""


def _now() -> str:
    """An ISO-8601 UTC timestamp, to the second."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    """A short random identifier, prefixed so it is readable in a shell."""
    return f"{prefix}-{secrets.token_hex(6)}"


class Store:
    """A SQLite database holding runs and the transcripts that make them up.

    Attributes:
        path: The database file. `:memory:` is accepted, for tests.
        connection: The open connection. Rows come back as `sqlite3.Row`.
    """

    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        """Open the database, creating the schema if it is not there yet.

        Args:
            path: Path to the database file, or `:memory:`.
        """
        self.path = path if path == ":memory:" else Path(path)
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._migrate()
        self.connection.commit()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so a
        column added to `schema.sql` never reaches a database created before it. Every
        migration here is an additive column with a default, applied only when absent, so
        opening an existing store is idempotent and opening a new one is a no-op.
        """
        added = {
            ("conversations", "subject_json"): "TEXT NOT NULL DEFAULT '{}'",
            ("conversations", "channel"): "TEXT NOT NULL DEFAULT 'conversation'",
            ("conversations", "planted_json"): "TEXT NOT NULL DEFAULT '[]'",
            ("conversations", "cohort_json"): "TEXT NOT NULL DEFAULT '[]'",
            ("turns", "usage_json"): "TEXT NOT NULL DEFAULT '{}'",
        }
        for (table, column), declaration in added.items():
            present = {
                row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")
            }
            if column not in present:
                self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    def __enter__(self) -> Store:
        """Enter a context that closes the connection on exit."""
        return self

    def __exit__(self, *exception: object) -> None:
        """Close the connection."""
        self.close()

    def create_run(
        self, target: str, versions: VersionTuple, *, run_id: str | None = None, notes: str = ""
    ) -> str:
        """Start a run and return its id.

        Args:
            target: The registered target name.
            versions: The four versions this run's results will be valid for.
            run_id: Force the id. Generated when absent.
            notes: Free text for the operator.

        Returns:
            The run id.
        """
        run_id = new_id("run") if run_id is None else run_id
        self.connection.execute(
            "INSERT INTO runs (run_id, target, started_at, config_version, policy_version, "
            "model_version, tool_version, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                target,
                _now(),
                versions.config_version,
                versions.policy_version,
                versions.model_version,
                versions.tool_version,
                notes,
            ),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str) -> None:
        """Mark a run finished. Idempotent: a run finished twice keeps the later time."""
        self.connection.execute(
            "UPDATE runs SET finished_at = ? WHERE run_id = ?", (_now(), run_id)
        )
        self.connection.commit()

    def save_transcript(self, run_id: str, transcript: Transcript, *, attack_id: str = "") -> str:
        """Write one conversation, its turns and its tool calls, in one transaction.

        Args:
            run_id: The run this conversation belongs to.
            transcript: The conversation, as the driver returned it.
            attack_id: Which attack produced it. Empty until `attacks/` exists.

        Returns:
            The conversation id.

        Raises:
            StoreError: If the run does not exist, or if the transcript reports spec
                versions that disagree with the run's. A conversation filed under a run it
                did not happen in would put a transcript behind a number it does not
                support.
        """
        run = self.load_run(run_id)
        if run is None:
            raise StoreError(f"no run {run_id!r}")
        self._check_versions(run, transcript)

        conversation_id = new_id("conv")
        with self.connection:
            self.connection.execute(
                "INSERT INTO conversations (conversation_id, run_id, target, session, goal, "
                "attack_id, stopped_because, subject_json, channel, planted_json, "
                "cohort_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    run_id,
                    transcript.target,
                    transcript.session,
                    transcript.goal,
                    attack_id,
                    transcript.stopped_because,
                    json.dumps(transcript.subject, sort_keys=True),
                    transcript.channel,
                    json.dumps([asdict(planted) for planted in transcript.planted], sort_keys=True),
                    json.dumps([dict(entry) for entry in transcript.cohort], sort_keys=True),
                    _now(),
                ),
            )
            self.connection.executemany(
                "INSERT INTO turns (conversation_id, turn_index, user_text, reply_text, "
                "latency_seconds, usage_json) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        conversation_id,
                        turn.index,
                        turn.user,
                        turn.reply,
                        turn.latency_seconds,
                        json.dumps(turn.agent_usage, sort_keys=True),
                    )
                    for turn in transcript.turns
                ],
            )
            self.connection.executemany(
                "INSERT INTO tool_calls (conversation_id, turn_index, call_index, name, "
                "arguments_json, result_json) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        conversation_id,
                        turn.index,
                        call_index,
                        call.name,
                        json.dumps(call.arguments, sort_keys=True),
                        json.dumps(call.result, sort_keys=True),
                    )
                    for turn in transcript.turns
                    for call_index, call in enumerate(turn.tool_calls)
                ],
            )
        return conversation_id

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        """One run as a dict, or `None` if there is no such run."""
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def load_transcript(self, conversation_id: str) -> Transcript | None:
        """Rebuild one conversation, or `None` if there is no such conversation.

        The reconstruction is exact for everything the judge reads: turn order, reply text,
        tool names, arguments as sent and results as returned, and the subject the
        conversation was entitled to. The subject is part of that list rather than an extra:
        without it every scope check on a rebuilt transcript reports as never in play, which
        is indistinguishable from a conversation that stayed in bounds.
        """
        row = self.connection.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        run = self.load_run(row["run_id"]) or {}

        calls: dict[int, list[ToolCallRecord]] = {}
        for call in self.connection.execute(
            "SELECT * FROM tool_calls WHERE conversation_id = ? ORDER BY turn_index, call_index",
            (conversation_id,),
        ):
            calls.setdefault(call["turn_index"], []).append(
                ToolCallRecord(
                    name=call["name"],
                    arguments=json.loads(call["arguments_json"]),
                    result=json.loads(call["result_json"]),
                )
            )

        turns = [
            Turn(
                index=turn["turn_index"],
                user=turn["user_text"],
                reply=turn["reply_text"],
                tool_calls=tuple(calls.get(turn["turn_index"], ())),
                latency_seconds=turn["latency_seconds"],
                agent_usage=json.loads(turn["usage_json"]),
            )
            for turn in self.connection.execute(
                "SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_index",
                (conversation_id,),
            )
        ]
        return Transcript(
            target=row["target"],
            session=row["session"],
            goal=row["goal"],
            subject=json.loads(row["subject_json"] or "{}"),
            turns=turns,
            spec_versions={
                "config": run.get("config_version", ""),
                "policy": run.get("policy_version", ""),
                "model": run.get("model_version", ""),
                "tools": run.get("tool_version", ""),
            },
            stopped_because=row["stopped_because"],
            channel=row["channel"] or CONVERSATIONAL_CHANNEL,
            planted=tuple(
                PlantedField(**planted) for planted in json.loads(row["planted_json"] or "[]")
            ),
            cohort=tuple(
                {str(k): str(v) for k, v in entry.items()}
                for entry in json.loads(row["cohort_json"] or "[]")
            ),
        )

    def conversation_ids(self, run_id: str) -> tuple[str, ...]:
        """Every conversation in a run, in the order they were written."""
        return tuple(
            row["conversation_id"]
            for row in self.connection.execute(
                "SELECT conversation_id FROM conversations WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            )
        )

    def _check_versions(self, run: dict[str, Any], transcript: Transcript) -> None:
        """Refuse a transcript whose target reported different versions than the run.

        Raises:
            StoreError: On any disagreement. The transcript is not wrong and the run is not
                wrong; they are about different agents, and only the caller knows which one
                it meant.
        """
        reported = transcript.spec_versions
        if not reported:
            return
        expected = {
            "config": run["config_version"],
            "policy": run["policy_version"],
            "model": run["model_version"],
            "tools": run["tool_version"],
        }
        if reported != expected:
            raise StoreError(
                f"transcript reports {reported} but run {run['run_id']} is for {expected}. "
                f"They are not about the same agent."
            )
