"""Building a conversation by hand, so a detector can be tested without running anything.

A detector reads a spec and a tool-call log and nothing else, which is what makes it testable
offline and what makes these tests short. `convo` takes the calls each turn made and produces
a transcript shaped exactly like one a real run produces.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentred.runner.conversation import ToolCallRecord, Transcript, Turn
from agentred.spec.loader import load_spec_dir

SPEC_ROOT = "src/agentred/targets/specs"


def spec_for(name: str):
    return load_spec_dir(f"{SPEC_ROOT}/{name}")


@pytest.fixture(scope="session")
def dispute():
    return spec_for("dispute_handler")


@pytest.fixture(scope="session")
def cart():
    return spec_for("cart_recovery")


def call(name: str, arguments: dict[str, Any] | None = None, result: Any = None) -> ToolCallRecord:
    """One tool call, as the target would have reported it."""
    return ToolCallRecord(
        name=name,
        arguments=dict(arguments or {}),
        result=result
        if isinstance(result, dict)
        else ({} if result is None else {"value": result}),
    )


def convo(*turns: list[ToolCallRecord], subject: dict[str, str] | None = None) -> Transcript:
    """A conversation whose turns made the given calls.

    Args:
        *turns: One list of calls per exchange. An empty list is a turn where the agent
            answered without doing anything, which is a real and common shape.
        subject: Who the conversation is about. Omitted, scope checks have nothing to compare
            against and must say so.
    """
    return Transcript(
        target="t",
        session="s",
        goal="g",
        subject=dict(subject or {}),
        turns=[
            Turn(index=i, user=f"turn {i}", reply=f"reply {i}", tool_calls=tuple(calls))
            for i, calls in enumerate(turns)
        ],
    )


def sole(findings, outcome=None):
    """The one finding in a set, or the one violation in it.

    Args:
        findings: What a detector returned.
        outcome: Keep only findings with this outcome first, where a check reports several.

    Returns:
        The single finding left. Raises if there is not exactly one, because a test that
        silently read the first of several is a test that could pass for the wrong reason.
    """
    kept = [f for f in findings if outcome is None or f.outcome is outcome]
    assert len(kept) == 1, f"expected one finding, got {len(kept)}"
    return kept[0]
