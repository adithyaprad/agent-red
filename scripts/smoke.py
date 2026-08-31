"""Run a handful of real attacks against a served target, and record everything.

**This is not a measurement.** Nothing here is scored, published, or turned into held-out
data. It settles one assumption that everything after it rests on: whether a model handed a
technique and an objective writes turns as persuasive as the hand-written exemplars in
`data/techniques/`. If it writes politely, a low violation rate would mean the attacker was
weak rather than the agent safe, and every number the suite produces afterwards inherits
that. Cheaper to find out on eight conversations than on four hundred.

Nothing about the attacks is written here. The suite is derived from the target's own spec by
`build_suite`, exactly as a real run would derive it, and this script only filters that suite
down to the stakes named so the run is small. Change `--stake` and a different slice runs;
change nothing and the machinery is identical to the real thing.

    uv run python scripts/smoke.py --target dispute_handler
        --stake precondition_skipped:issue_refund:verify_identity
        --stake bound_exceeded:issue_refund:amount:above

Run `--list-stakes` first to see what the target derives.

This is the conversation half on its own: it holds conversations and stops. `agentred run`
chains this to the analysis and the operator page. The engine both use lives in
`agentred.runner.suite`; what is left here is the command line around it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentred.attacks.generator import build_suite
from agentred.attacks.stakes import derive_stakes
from agentred.runner.consent import load_registry
from agentred.runner.suite import (
    COUNTER_FILENAME,
    DEFAULT_ATTACKER_MODEL,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_TURNS,
    RUNS_ROOT,
    Outcome,
    SuiteRun,
    execute,
    next_run_dir,
    persist,
    select,
    summarise,
    to_json,
)
from agentred.spec import load_spec_dir

SMOKE_MODEL = DEFAULT_ATTACKER_MODEL
"""Kept under its old name because runs already on disk were made with it."""

SmokeRun = SuiteRun

__all__ = [
    "COUNTER_FILENAME",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_TURNS",
    "RUNS_ROOT",
    "SMOKE_MODEL",
    "Outcome",
    "SmokeRun",
    "SuiteRun",
    "execute",
    "main",
    "next_run_dir",
    "persist",
    "select",
    "summarise",
    "to_json",
]


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the slice, write the artefacts.

    Args:
        argv: Command line arguments, defaulting to the process's own.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="dispute_handler", help="Registered target name.")
    parser.add_argument(
        "--stake",
        action="append",
        default=[],
        help="Stake id to run. Repeatable. Omit to run the whole derived suite.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap attacks after filtering.")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=SMOKE_MODEL, help="First-party id for the attacker.")
    parser.add_argument("--out", type=Path, default=None, help="Force this run's directory.")
    parser.add_argument("--label", default="", help="Human tag appended to the run's name.")
    parser.add_argument("--store", type=Path, default=Path("data/agentred.db"))
    parser.add_argument("--list-stakes", action="store_true", help="Print the stakes and exit.")
    arguments = parser.parse_args(argv)

    spec = load_spec_dir(load_registry().resolve(arguments.target).spec_dir)
    if arguments.list_stakes:
        for stake in derive_stakes(spec):
            print(stake.id)
        return

    stakes = tuple(arguments.stake)
    attacks = select(build_suite(spec), stakes, arguments.limit)
    if arguments.out is not None:
        out = arguments.out
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = next_run_dir(arguments.target, stakes, arguments.label)

    print(f"running {len(attacks)} attack(s) against {arguments.target}, writing to {out}")
    run = execute(
        attacks,
        target=arguments.target,
        model=arguments.model,
        stake=", ".join(stakes),
        max_turns=arguments.max_turns,
        concurrency=arguments.concurrency,
        recording=out / "calls.jsonl",
    )
    run.number = out.name[:4] if out.name[:4].isdigit() else ""
    persist(run, arguments.store)
    (out / "run.json").write_text(json.dumps(to_json(run), indent=2), encoding="utf-8")

    print(summarise(run))
    print(f"run.json     {out / 'run.json'}")
    print(f"calls.jsonl  {out / 'calls.jsonl'}")
    print(f"store run id {run.run_id or 'nothing persisted'}")


if __name__ == "__main__":
    sys.exit(main())
