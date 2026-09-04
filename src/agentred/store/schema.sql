-- agent-red persistence.
--
-- SQLite, one file, no server. Anyone can open a run with the sqlite3 shell and read every
-- transcript that produced a number on the scorecard, which is the point: a metric nobody
-- can trace back to the conversation that produced it is a claim, not a measurement.
--
-- Written by the runner, and by nothing else: runs, conversations, turns, tool_calls.
--
-- What a run meant is a separate object. The detectors and the judge read these tables and
-- write an analysis beside the run, so re-deciding a question is free, offline, and cannot
-- edit the record the decision was made from.

PRAGMA foreign_keys = ON;

-- One execution of the suite against one target at one version.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    target          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    -- The validity tuple. A scorecard is valid for exactly these, and a change in any of
    -- them makes the agent untested again, so none of them is nullable. The world is the
    -- fifth (ADR-0007) and defaults to empty rather than being backfilled: a run against a
    -- world nobody generated should read as what it was.
    config_version  TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    tool_version    TEXT NOT NULL,
    world_version   TEXT NOT NULL DEFAULT '',
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
    -- Whose data this conversation is entitled to, as identifier kind to value. Stored
    -- because the scope detector compares every record reached against it, and a transcript
    -- rebuilt without it reports every scope check as never in play rather than as passed.
    subject_json    TEXT NOT NULL DEFAULT '{}',
    -- Which channel the attack arrived down, and what it wrote into the world before the
    -- agent ran. Both are stored because a planted attempt is unreadable without them: the
    -- transcript shows a benign trigger and an agent doing something it should not, and the
    -- only thing that explains the gap is text a stranger had already put in a field the
    -- agent read. Defaults describe a conversational attempt, which is what every row
    -- written before channels existed was.
    channel         TEXT NOT NULL DEFAULT 'conversation',
    planted_json    TEXT NOT NULL DEFAULT '[]',
    -- Every subject a scheduled firing was legitimately woken about, beyond the one the
    -- attempt is named for. Empty for a conversation, which is with one person. Without it
    -- a batch agent's every other record reads back as a stranger it should not have
    -- reached.
    cohort_json     TEXT NOT NULL DEFAULT '[]',
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
    -- What this turn cost the target, as the target's own engine reported it, as a token
    -- name to count map. Stored because the harness spends on both sides of every turn and
    -- the target's half is reported once, in the reply, and is gone when the process is.
    -- An empty object is a target that did not report, which is not a claim that it was
    -- free: the cost report counts those turns and says how many there were.
    usage_json      TEXT NOT NULL DEFAULT '{}',
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


