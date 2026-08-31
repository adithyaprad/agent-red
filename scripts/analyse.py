"""Everything that happens after the conversations, over transcripts already recorded.

No target is contacted and no conversation is created. This reads what is in the store, runs
every check against it, and writes one file. The operator-facing page is generated from that
file and never by calling anything, so a page can be rebuilt a hundred times while the analysis
behind it is paid for once.

    uv run python scripts/analyse.py --out analysis.json

Four stages, in the sequence that spends the least:

1. Detectors, over the tool-call log. Free, offline, and they settle everything they can.
2. Rule extraction from each agent's prose, once per agent. Names the rules the policy does
   not carry, which is where both real failures in this project's own runs lived.
3. The judge, on those rules only, and only for conversations where the governing tool was
   actually called.
4. Consistency across conversations, then the breaking point for every confirmed failure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentred.attacks.infer_policy import infer_policy
from agentred.judge.detectors import run_detectors
from agentred.judge.llm import judge_conversation
from agentred.judge.models import Finding, Outcome
from agentred.llm.client import AnthropicModelClient
from agentred.runner.conversation import Transcript
from agentred.scoring.breaking_point import find_all
from agentred.scoring.consistency import Attempt, compare
from agentred.spec import load_spec_dir
from agentred.store.repo import Store

SPECS = Path("src/agentred/targets/specs")


def load(store: Store) -> list[tuple[str, str, Transcript]]:
    """Read every stored conversation.

    The store holds one connection and is not thread safe, deliberately: persistence is
    microseconds against model calls that take tens of seconds. So everything is read here, on
    one thread, and only the judging fans out.

    Args:
        store: The open store.

    Returns:
        One entry per conversation: run id, target name, transcript.
    """
    rows = list(store.connection.execute("SELECT run_id, target FROM runs"))
    return [
        (row["run_id"], row["target"], store.load_transcript(cid))
        for row in rows
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


def main(argv: list[str] | None = None) -> None:
    """Run every post-conversation check and write the result."""
    parser = argparse.ArgumentParser(description="Analyse recorded conversations.")
    parser.add_argument("--store", type=Path, default=Path("data/agentred.db"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-5")
    arguments = parser.parse_args(argv)

    client = AnthropicModelClient(model=arguments.model)
    store = Store(arguments.store)
    conversations = load(store)
    targets = sorted({target for _, target, _ in conversations})
    print(f"{len(conversations)} conversations across {len(targets)} agent(s)")

    specs, duties, undeclared = {}, {}, {}
    for name in targets:
        spec = load_spec_dir(SPECS / name)
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
        print(f"  {name}: {len(duties[name])} undeclared obligation(s) to judge")

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
                workers=4,
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
    print(f"{len(violations)} confirmed failure(s); locating where each was lost")
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

    print("comparing conversations against each other")
    comparison = compare(tuple(attempts), client)

    arguments.out.write_text(
        json.dumps(
            {
                "conversations": records,
                "policy": undeclared,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
