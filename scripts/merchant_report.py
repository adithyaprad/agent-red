"""Build the operator-facing page from a finished analysis.

The rendering lives in `agentred.scoring.render`; what is here is the command line around it.
It calls nothing, so a page can be rebuilt as often as anyone wants to argue about the
wording while the analysis behind it is paid for once.

    uv run python scripts/merchant_report.py analysis.json --out what-broke.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentred.scoring.render import build


def main(argv: list[str] | None = None) -> None:
    """Generate the page from a finished analysis."""
    parser = argparse.ArgumentParser(description="Build the operator-facing page.")
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    arguments.out.write_text(build(analysis), encoding="utf-8")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    sys.exit(main())
