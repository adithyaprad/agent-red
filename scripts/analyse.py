"""Run every post-conversation check over recorded transcripts, and write the result.

No target is contacted and no conversation is created. The work lives in
`agentred.scoring.analysis`; what is here is the command line around it.

    uv run python scripts/analyse.py --list-runs
    uv run python scripts/analyse.py --run run_a1b2c3 --out analysis.json

`--run` is repeatable and, omitted, analyses every run in the store. That default is right
while there is one run and wrong the moment there are two, so `--list-runs` prints what is
there and the analysis names the runs it covered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentred.llm.client import AnthropicModelClient
from agentred.scoring.analysis import AnalysisError, analyse, known_runs
from agentred.store.repo import Store


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, analyse the named runs, write one file."""
    parser = argparse.ArgumentParser(description="Analyse recorded conversations.")
    parser.add_argument("--store", type=Path, default=Path("data/agentred.db"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run id to analyse. Repeatable. Omit to analyse every run in the store.",
    )
    parser.add_argument("--list-runs", action="store_true", help="Print the runs and exit.")
    arguments = parser.parse_args(argv)

    store = Store(arguments.store)
    if arguments.list_runs:
        for row in known_runs(store):
            count = len(store.conversation_ids(row["run_id"]))
            print(f"{row['run_id']}  {row['target']:<18} {count:>3} conversations  {row['notes']}")
        return

    if arguments.out is None:
        parser.error("--out is required unless --list-runs is given")

    try:
        result = analyse(
            store,
            AnthropicModelClient(model=arguments.model),
            runs=tuple(arguments.run),
        )
    except AnalysisError as error:
        raise SystemExit(str(error)) from error

    arguments.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    sys.exit(main())
