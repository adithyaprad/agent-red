-- agent-red persistence.
--
-- SQLite, one file, no server. Anyone can open a run with the sqlite3 shell and read every
-- transcript that produced a number on the scorecard, which is the point: a metric nobody
-- can trace back to the conversation that produced it is a claim, not a measurement.
--
-- Written by:
--   runs, conversations, turns, tool_calls   the runner
--   verdicts                                  the judge
--   labels                                    the human labelling task, and only that
--
-- The verdicts and labels tables are declared here and filled in later milestones. They
-- live in this file rather than a second one so the schema is readable in one sitting.

PRAGMA foreign_keys = ON;

-- One execution of the suite against one target at one version.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    target          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    -- The validity tuple. A scorecard is valid for exactly these four, and a change in any
    -- of them makes the agent untested again, so none of them is nullable.
    config_version  TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    tool_version    TEXT NOT NULL,
    notes           TEXT NOT NULL DEFAULT ''
);

-- One attack conversation.
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    target          TEXT NOT NULL,
    session         TEXT NOT NULL,
    goal            TEXT NOT NULL,
    attack_id       TEXT NOT NULL DEFAULT '',
    stopped_because TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS conversations_by_run ON conversations(run_id);

-- One exchange within a conversation.
CREATE TABLE IF NOT EXISTS turns (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_index      INTEGER NOT NULL,
    user_text       TEXT NOT NULL,
    reply_text      TEXT NOT NULL,
    latency_seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (conversation_id, turn_index)
);

-- One tool call, with the arguments exactly as the model sent them.
-- The deterministic detectors read this table and nothing else.
CREATE TABLE IF NOT EXISTS tool_calls (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_index      INTEGER NOT NULL,
    call_index      INTEGER NOT NULL,
    name            TEXT NOT NULL,
    arguments_json  TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn_index, call_index)
);

CREATE INDEX IF NOT EXISTS tool_calls_by_name ON tool_calls(name);

-- What the judge concluded about one conversation.
-- `source` separates a detector's assertion from a model's opinion, because they are not
-- the same kind of evidence and the scorecard reports them separately.
CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    violation_type  TEXT NOT NULL,
    violated        INTEGER NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('detector', 'llm')),
    statement_name  TEXT NOT NULL DEFAULT '',
    confidence      REAL,
    rationale       TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS verdicts_by_conversation ON verdicts(conversation_id);

-- A human judgement on a held-out transcript.
-- `round` distinguishes the first pass from the blind re-label that establishes the human
-- ceiling. Nothing outside judge/calibration/ may read this table.
CREATE TABLE IF NOT EXISTS labels (
    label_id        TEXT PRIMARY KEY,
    transcript_ref  TEXT NOT NULL,
    violation_type  TEXT NOT NULL,
    violated        INTEGER NOT NULL,
    labeller        TEXT NOT NULL,
    round           INTEGER NOT NULL DEFAULT 1,
    note            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
