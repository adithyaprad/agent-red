"""Everything that happens after the conversations, over transcripts already recorded.

No target is contacted and no conversation is created. This reads what is in the store, runs
every check against it, and returns one dictionary. The operator-facing page is generated from
that dictionary and never by calling anything, so a page can be rebuilt a hundred times while
the analysis behind it is paid for once.

Four stages, in the sequence that spends the least:

1. Detectors, over the tool-call log. Free, offline, and they settle everything they can.
2. Rule extraction from each agent's prose, once per agent. Names the rules the policy does
   not carry, which is where both real failures in this project's own runs lived.
3. The judge, on those rules only, and only for conversations where the governing tool was
   actually called.
4. Consistency across conversations, then the breaking point for every confirmed failure.

**An analysis is of named runs, not of a database.** Nothing here defaults to reading
everything, because the two are the same thing only while there is exactly one run on disk.
The moment there are two, an unfiltered analysis silently pools conversations held against
different agent versions, re-pays for judging work already done, and hands the operator a page
whose denominator counts runs they did not ask about. The runs analysed are named in the
result so the page can say which agent at which version the numbers are valid for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentred.attacks.infer_policy import infer_policy
from agentred.judge.detectors import run_detectors
from agentred.judge.llm import judge_conversation
from agentred.judge.models import Finding, Outcome
from agentred.llm.client import ModelClient
from agentred.runner.conversation import Transcript
from agentred.scoring.breaking_point import find_all
from agentred.scoring.consistency import Attempt, compare
from agentred.spec import load_spec_dir
from agentred.store.repo import Store

SPECS = Path("src/agentred/targets/specs")

JUDGE_WORKERS = 4


class AnalysisError(Exception):
    """An analysis was asked for something the store does not hold."""


def known_runs(store: Store) -> tuple[dict[str, Any], ...]:
    """Every run in the store, newest last.

    Args:
        store: The open store.

    Returns:
        One dictionary per run, carrying the validity tuple and the note the runner wrote,
        which is where a run's number is recorded.
    """
    rows = store.connection.execute(
        "SELECT run_id, target, started_at, finished_at, config_version, policy_version, "
        "model_version, tool_version, notes FROM runs ORDER BY started_at"
    )
    return tuple(dict(row) for row in rows)


def resolve_runs(store: Store, runs: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """Turn the run ids a caller named into the run rows they refer to.

    An id that matches nothing is fatal rather than skipped. A run that quietly contributed
    no conversations is indistinguishable, from the page, from a run whose agent never broke
    a rule, and the second is a claim while the first is a typo.

    Args:
        store: The open store.
        runs: Run ids to keep. Empty selects every run in the store.

    Returns:
        The selected run rows, in the sequence the store holds them.

    Raises:
        AnalysisError: If any named id is not in the store, or the store holds no runs.
    """
    available = known_runs(store)
    if not available:
        raise AnalysisError("the store holds no runs")
    if not runs:
        return available
    present = {row["run_id"] for row in available}
    missing = [run for run in runs if run not in present]
    if missing:
        listed = "\n  ".join(f"{r['run_id']}  {r['target']}  {r['notes']}" for r in available)
        raise AnalysisError(
            f"no run {', '.join(repr(run) for run in missing)} in this store. Available:\n  "
            + listed
        )
    return tuple(row for row in available if row["run_id"] in runs)


def load_conversations(
    store: Store, selected: tuple[dict[str, Any], ...]
) -> list[tuple[str, str, Transcript]]:
    """Read every conversation belonging to the selected runs.

    The store holds one connection and is not thread safe, deliberately: persistence is
    microseconds against model calls that take tens of seconds. So everything is read here, on
    one thread, and only the judging fans out.

    Args:
        store: The open store.
        selected: The run rows to read, from `resolve_runs`.

    Returns:
        One entry per conversation: run id, target name, transcript.
    """
    return [
        (row["run_id"], row["target"], store.load_transcript(cid))
        for row in selected
        for cid in store.conversation_ids(row["run_id"])
    ]


def costly_actions(spec: Any) -> set[str]:
    """Which of an agent's actions cost its operator something if taken wrongly."""
    return {
        tool.name for tool in spec.config.tools if tool.consequence.value in {"money", "obligation"}
    }


def damage_turn(finding: Finding) -> int:
    """Which exchange a confirmed failure was recorded in, or -1 if it is not anchored."""
    if finding.evidence is not None:
        return finding.evidence.turn
    if finding.utterance is not None:
        return finding.utterance.turn
    return -1


DECLARED_TOOLS = "declared_tools"
"""The rule that comes from the tool list rather than from anything the operator wrote.

It still needs a sentence, because the page shows a reader their own rules and a row with no
words next to it reads as a bug rather than as a check.
"""


def declared_rules(spec: Any) -> dict[str, str]:
    """Every rule the operator wrote down, as rule name to the sentence they wrote.

    The page shows a person their own words, never our identifier for them. A page that says
    `data_scope.email` has told the reader nothing they can act on and has told them it was
    written for somebody else. Names with no sentence come back empty rather than missing, so
    the caller can see that a declared rule was left undescribed instead of silently rendering
    a blank row that looks like a bug.

    Args:
        spec: A loaded agent spec.

    Returns:
        Rule name to its stated sentence, covering every policy section and one entry per
        identifier kind the data scope binds, which is the form the scope detector reports
        under. Also carries the one rule that is not a sentence anybody wrote: whether every
        action was a declared one, which comes from the tool list rather than from a rule.
    """
    text: dict[str, str] = {}
    for bound in spec.policy.bounds:
        text[bound.name] = bound.description
    for precondition in spec.policy.preconditions:
        text[precondition.name] = precondition.description
    for once in spec.policy.idempotency:
        text[once.name] = once.description
    for rule in spec.policy.outbound:
        text[rule.name] = rule.description
    for cite in spec.policy.citations:
        text[cite.name] = cite.description
    for duty in spec.policy.obligations:
        text[duty.name] = duty.statement or duty.description
    text[DECLARED_TOOLS] = "The agent only used the actions it was set up with, and nothing else."
    kinds = spec.policy.data_scope.subject_identifier_kinds
    for kind in kinds:
        # One scope, one sentence, but the detector reports it once per identifier it binds
        # on. Rendered as written, that is the same sentence twice with no way for a reader
        # to tell the rows apart or to see that they are two different checks. The identifier
        # is appended only when there is more than one, so the ordinary single-identifier
        # agent still reads as the operator wrote it.
        described = spec.policy.data_scope.description
        text[f"data_scope.{kind}"] = (
            f"{described} (checked against the {kind})" if len(kinds) > 1 else described
        )
    return text


def analyse(
    store: Store,
    client: ModelClient,
    *,
    runs: tuple[str, ...] = (),
    specs_root: Path = SPECS,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every post-conversation check over the named runs and return the result.

    Args:
        store: The open store.
        client: The model client, for extraction, judging and the two comparison stages.
        runs: Run ids to analyse. Empty analyses every run in the store, which is only the
            right thing to do while there is one.
        specs_root: Where target spec directories live.
        say: Where progress goes. A long analysis is minutes of model calls with nothing to
            look at otherwise.

    Returns:
        The analysis, ready to serialise and to render a page from.

    Raises:
        AnalysisError: If a named run is not in the store.
    """
    selected = resolve_runs(store, runs)
    conversations = load_conversations(store, selected)
    targets = sorted({target for _, target, _ in conversations})
    say(
        f"{len(conversations)} conversations from {len(selected)} run(s) "
        f"across {len(targets)} agent(s)"
    )

    specs, duties, undeclared = {}, {}, {}
    for name in targets:
        spec = load_spec_dir(specs_root / name)
        specs[name] = spec
        inference = infer_policy(spec.config, client, declared=spec.policy)
        duties[name] = tuple(d for d in inference.obligations if d.name in inference.undeclared)
        undeclared[name] = {
            "read": inference.read,
            "invented_fraction": inference.invented_fraction,
            "statements": [
                {
                    "name": s.name,
                    "description": s.description,
                    "declared": s.name not in inference.undeclared,
                }
                for s in inference.statements
            ],
            "obligations": [
                {
                    "name": o.name,
                    "kind": o.kind.value,
                    "statement": o.statement,
                    "declared": o.name not in inference.undeclared,
                }
                for o in inference.obligations
            ],
        }
        undeclared[name]["declared_rules"] = declared_rules(spec)
        say(f"  {name}: {len(duties[name])} undeclared obligation(s) to judge")

    records: list[dict[str, Any]] = []
    attempts: list[Attempt] = []
    for run_id, target, transcript in conversations:
        spec = specs[target]
        findings = list(run_detectors(spec, transcript))
        findings.extend(
            judge_conversation(
                transcript,
                duties[target],
                client,
                instructions=spec.config.instructions,
                workers=JUDGE_WORKERS,
            )
        )
        called = {call.name for turn in transcript.turns for call in turn.tool_calls}
        for action in costly_actions(spec):
            if action not in transcript.goal and action not in called:
                continue
            attempts.append(
                Attempt(
                    conversation=transcript.session,
                    subject=transcript.subject,
                    action=action,
                    taken=action in called,
                    label=transcript.session,
                    said=tuple(f"person: {t.user}\nagent: {t.reply}" for t in transcript.turns),
                )
            )
        records.append(
            {
                "run": run_id,
                "target": target,
                "session": transcript.session,
                "goal": transcript.goal,
                "subject": transcript.subject,
                "stopped_because": transcript.stopped_because,
                "versions": transcript.spec_versions,
                "turns": [
                    {
                        "index": t.index,
                        "user": t.user,
                        "reply": t.reply,
                        "calls": [
                            {"name": c.name, "arguments": c.arguments, "result": c.result}
                            for c in t.tool_calls
                        ],
                    }
                    for t in transcript.turns
                ],
                "findings": [f.model_dump(mode="json") for f in findings],
            }
        )

    violations = [
        (transcript, record, finding)
        for (_, _, transcript), record in zip(conversations, records, strict=True)
        for finding in [Finding.model_validate(f) for f in record["findings"]]
        if finding.outcome is Outcome.VIOLATED
    ]
    say(f"{len(violations)} confirmed failure(s); locating where each was lost")
    points = find_all(
        tuple((t, damage_turn(f), f.summary) for t, _, f in violations if damage_turn(f) >= 0),
        client,
    )
    anchored = [(r, f) for t, r, f in violations if damage_turn(f) >= 0]
    for (record, finding), point in zip(anchored, points, strict=True):
        if point is None:
            continue
        record.setdefault("breaking_points", []).append(
            {"rule": finding.rule, **asdict(point), "turns_earlier": point.turns_earlier}
        )

    say("comparing conversations against each other")
    comparison = compare(tuple(attempts), client)

    return {
        "runs": [
            {
                "run_id": row["run_id"],
                "target": row["target"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"] or "",
                "notes": row["notes"],
                "versions": {
                    "config": row["config_version"],
                    "policy": row["policy_version"],
                    "model": row["model_version"],
                    "tools": row["tool_version"],
                },
                "conversations": sum(1 for record in records if record["run"] == row["run_id"]),
            }
            for row in selected
        ],
        "conversations": records,
        "policy": undeclared,
        "presentation": {
            name: {
                "unit_symbol": spec.config.unit_symbol,
                "subject_term": spec.config.subject_term,
                "value_fields": list(spec.config.value_fields),
            }
            for name, spec in specs.items()
        },
        "consistency": {
            "groups": comparison.groups,
            "settled": comparison.settled,
            "unknown_subject": comparison.unknown_subject,
            "rate": comparison.divergence_rate,
            "divergences": [
                {
                    "subject": d.subject,
                    "action": d.action,
                    "complied": [a.conversation for a in d.complied],
                    "declined": [a.conversation for a in d.declined],
                    "alike": d.alike,
                    "difference": d.difference,
                    "reasoning": d.reasoning,
                    "summary": d.summary,
                }
                for d in comparison.divergences
            ],
        },
    }
