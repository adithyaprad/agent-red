"""There is no way to reach a target without passing through the consent gate.

ADR-0001 makes consent a property of the code rather than a promise in prose. That claim is
only true while it keeps being true, and the way it would stop being true is quiet: someone
adds an HTTP call somewhere convenient, and a year later the README says something the tree
does not do. So the claim is asserted here.

Adding a module to `MAY_SEND` is a deliberate act. It means that module sends turns to a
target, and that every path into it takes a `ConsentToken`.
"""

from __future__ import annotations

import ast
from pathlib import Path

from agentred.runner.consent import load_registry

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "agentred"
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "targets.registry.yaml"

MAY_SEND = {
    "runner/consent.py",  # the gate itself, and the only sender of a challenge
    "runner/conversation.py",  # the driver; every entry point takes a ConsentToken
    "mcp/control.py",  # speaks to our own tool server, never to a target; see below
}
"""Modules permitted to make outbound HTTP calls.

Two kinds, and the distinction is the point. `runner/consent.py` and
`runner/conversation.py` reach the agent under test, and every path into them takes a
`ConsentToken`. `mcp/control.py` reaches the tool server, which is ours: it reads the
recorded call stream and moves worlds, and there is no operation on it that sends anything
to an agent. Its address comes from the registry entry for the target, like every other
address in the harness.

`targets/` is excluded wholesale below: it is the agent under test, not the harness, and it
receives requests rather than sending them.
"""

HTTP_CLIENTS = {"httpx", "requests", "urllib", "urllib3", "http", "aiohttp"}


def harness_modules() -> list[tuple[str, ast.Module]]:
    """Every module in the harness, excluding the stand-in agents in `targets/`."""
    modules = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        if relative.startswith("targets/"):
            continue
        modules.append((relative, ast.parse(path.read_text(encoding="utf-8"))))
    return modules


def imported_roots(tree: ast.Module) -> set[str]:
    """Top-level package names imported anywhere in a module, including inside functions."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_only_the_gate_speaks_http_to_a_target() -> None:
    offenders = {
        relative
        for relative, tree in harness_modules()
        if relative not in MAY_SEND and imported_roots(tree) & HTTP_CLIENTS
    }
    assert not offenders, (
        f"{sorted(offenders)} import an HTTP client but are not in MAY_SEND. A module that "
        f"reaches a target must take a ConsentToken; see ADR-0001."
    )


def test_a_targets_chat_url_is_read_only_through_a_token() -> None:
    offenders = set()
    for relative, tree in harness_modules():
        if relative in MAY_SEND:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "chat_url":
                offenders.add(relative)
    assert not offenders, (
        f"{sorted(offenders)} read chat_url outside the consent gate. Having a target's URL "
        f"and being permitted to use it are meant to be different things."
    )


def test_the_shipped_registry_parses_and_is_test_mode_throughout() -> None:
    registry = load_registry(REGISTRY_PATH)
    assert registry.names
    for target in registry.targets:
        assert target.mode == "test", f"{target.name} is registered in {target.mode!r} mode"
