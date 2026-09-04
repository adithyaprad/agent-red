"""Documentation may not name a file or directory that does not exist.

A reader's first move on an unfamiliar repository is to follow a path out of the README, and a
path that resolves to nothing is worth more than a missing feature: the feature might be
scheduled, while the dead reference says the documentation was written against an intention and
never checked against the tree. Once one is found, every other claim in the document is read
differently, which is a cost out of all proportion to the mistake.

So it is a build failure rather than something a reader finds. Twelve of these existed when
this file was written, most of them survivors of renames that touched the code and not the prose.

**Architecture decision records are exempt, deliberately.** An ADR describes a decision at the
moment it was taken, and the state it argues against is usually a path that no longer exists.
`ADR-0006` says in as many words that `runner/conversation.py` becomes
`runner/channels/conversational.py`; rewriting the first name to the second would make the
sentence false. A record that is edited to stay current is no longer a record.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CHECKED = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
"""Documents that describe the repository as it stands. `docs/DECISIONS/` is not among them."""

REFERENCE = re.compile(r"`([A-Za-z][A-Za-z0-9_./-]*(?:/|\.py|\.md|\.yaml|\.yml|\.jsonl|\.toml))`")
"""A path in backticks: anything ending in a directory slash or a known source extension.

Deliberately narrow. Prose names plenty of things in backticks that are not paths, and a
pattern that tried to catch every possible spelling of a path would spend its time on false
positives, which is how a guard gets switched off."""

RESOLVED_AGAINST = ("", "src/agentred", "docs")
"""Prefixes a reference may be written relative to, in the order they are tried."""

RUNTIME_ARTIFACTS = frozenset({"data/agentred.db", "out/runs/", "calls.jsonl"})
"""Paths a run creates rather than the repository carrying.

Each one is named here so that adding to this set is a decision rather than an oversight. A
path listed here must still be one the tool genuinely produces."""


def references(document: Path) -> list[tuple[int, str]]:
    """Every backticked path in `document`, as line number and reference.

    Args:
        document: The markdown file to read.

    Returns:
        One entry per occurrence, in document order. Does not deduplicate: a reference repeated
        on three lines is three problems to fix, and reporting one hides the others.
    """
    found = []
    for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
        found.extend((number, match.group(1)) for match in REFERENCE.finditer(line))
    return found


def resolves(reference: str) -> bool:
    """Whether `reference` names something in the tree, or a declared runtime artifact."""
    if reference in RUNTIME_ARTIFACTS:
        return True
    return any((ROOT / prefix / reference).exists() for prefix in RESOLVED_AGAINST)


@pytest.mark.parametrize("document", CHECKED, ids=lambda path: path.name)
def test_every_path_a_document_names_exists(document: Path) -> None:
    """A reader following any path out of this document arrives somewhere."""
    dangling = [
        f"{document.relative_to(ROOT)}:{number} names {reference!r}"
        for number, reference in references(document)
        if not resolves(reference)
    ]
    assert not dangling, "\n".join(["documentation names paths that do not exist:", *dangling])


def test_the_guard_would_notice() -> None:
    """The pattern and the resolver agree on something real and something invented."""
    assert resolves("src/agentred/judge/detectors/")
    assert not resolves("src/agentred/judge/no_such_package/")


ROOTED = re.compile(
    r"\b((?:docs|src/agentred|examples|data|tests|scripts)/[A-Za-z0-9_./-]*[A-Za-z0-9_/])"
)
"""A reference written from the repository root, backticks or not.

The narrow backtick pattern above is right for prose, where most paths are quoted and the
false positives would be many. It is wrong everywhere else: the three references this guard
first missed were in a source docstring, a YAML comment and a test, none of them backticked.
Anything beginning with one of this repository's own top-level directories is a path, in any
file, in any syntax."""

ADR_DIRECTORY = "docs/DECISIONS"
"""Exempt for the reason stated at the top of this file: an ADR is a record, not a description
of the tree as it stands, and the state it argues against is usually a path that is gone."""


def tracked_files() -> list[Path]:
    """Every file git tracks, which is every file a reader of this repository can see."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / name for name in listed.stdout.split("\0") if name]


def rooted_references(path: Path) -> list[tuple[int, str]]:
    """Every root-relative path `path` names, as line number and reference."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    return [
        (number, match.group(1))
        for number, line in enumerate(lines, 1)
        for match in ROOTED.finditer(line)
    ]


def test_no_tracked_file_names_a_path_that_does_not_exist() -> None:
    """A path named anywhere a reader can see resolves, not only one named in prose.

    The narrow guard above covers the documents somebody reads first. This one covers
    everything else they read next: docstrings, comments in a spec, an example invocation in a
    script header. A stale path in a docstring costs exactly what a stale path in the README
    costs, because the reader who follows it has already decided to trust the file.
    """
    dangling = [
        f"{path.relative_to(ROOT)}:{number} names {reference!r}"
        for path in tracked_files()
        if ADR_DIRECTORY not in path.as_posix() and path.name != Path(__file__).name
        for number, reference in rooted_references(path)
        if not (ROOT / reference).exists() and reference not in RUNTIME_ARTIFACTS
    ]
    assert not dangling, "\n".join(["files name paths that do not exist:", *dangling])
