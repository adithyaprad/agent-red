"""Recovering a declaration somebody wrote by hand, and saying exactly what was not recovered.

The reader's claim is that a declaration does not have to be authored. `dispute_handler` has
an authored one, so the claim is checkable: read the same agent off its connector, emit, and
put the two side by side. What matters is not how much came back but whether anything came
back differently, because a reader that recovers half a declaration correctly is useful and a
reader that recovers all of it wrongly is worse than none.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentred.ingest.adapters.mcp import read_agent
from agentred.ingest.diff import Verdict, compare
from agentred.ingest.emit import EmitError, to_config
from agentred.mcp.server import ToolServer, build_tool_app
from agentred.spec import load_spec_dir

SPEC_ROOT = "src/agentred/targets/specs"
AGENT = "dispute_handler"


@contextlib.asynccontextmanager
async def served(agent: str) -> AsyncIterator[tuple[str, Any]]:
    """A connector URL for one agent, and an ASGI client that reaches it without a socket."""
    import httpx2

    server = ToolServer([load_spec_dir(f"{SPEC_ROOT}/{agent}")])
    app = build_tool_app(server)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://arena"
        ) as http,
    ):
        yield f"http://arena/{agent}/ingest/s1", http


async def _recovered(agent: str, answers: dict[str, Any]) -> Any:
    """Read an agent off its connector, answer the questions, and emit."""
    authored = load_spec_dir(f"{SPEC_ROOT}/{agent}").config
    async with served(agent) as (url, http):
        package = await read_agent(agent, [url], http_client=http)
    resolved = package.__class__(
        agent_id=package.agent_id,
        tools=tuple(
            tool.__class__(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                consequence=tool.consequence.confirmed(answers[tool.name], by="fde@agent-red"),
                evidence=tool.evidence,
            )
            for tool in package.tools
        ),
        sources=package.sources,
    )
    emission = to_config(resolved, version=authored.version, model=authored.model)
    return authored, emission


async def test_a_declaration_cannot_be_emitted_while_a_question_is_unanswered() -> None:
    """The refusal the whole reader is built around, at the one place it can be enforced."""
    async with served(AGENT) as (url, http):
        package = await read_agent(AGENT, [url], http_client=http)

    with pytest.raises(EmitError) as raised:
        to_config(package, version="1.7", model="claude-sonnet-5")

    assert "issue_refund" in str(raised.value)
    assert "would be a guess reported as a fact" in str(raised.value)


async def test_the_tool_surface_comes_back_identical_to_the_authored_one() -> None:
    """Every tool, every argument schema, every description, off the wire and unedited."""
    authored = load_spec_dir(f"{SPEC_ROOT}/{AGENT}").config
    answers = {tool.name: tool.consequence for tool in authored.tools}

    authored, emission = await _recovered(AGENT, answers)
    recovery = compare(authored, emission.config, uncovered=emission.unreadable)

    assert recovery.faithful, recovery.render()
    assert not recovery.of(Verdict.ADDED)
    assert len(recovery.of(Verdict.MATCHED)) == len(authored.tools) * 3


async def test_what_no_reader_covered_is_named_rather_than_silently_absent() -> None:
    """A thin declaration must not read as a thorough one about a simple agent."""
    authored = load_spec_dir(f"{SPEC_ROOT}/{AGENT}").config
    answers = {tool.name: tool.consequence for tool in authored.tools}

    _authored, emission = await _recovered(AGENT, answers)

    joined = " ".join(emission.unreadable)
    assert "data sources" in joined
    assert "channels" in joined
    assert "policy" in joined


async def test_a_wrong_answer_shows_up_as_a_divergence_rather_than_as_a_match() -> None:
    """The comparison has to be able to fail, or asserting that it passes proves nothing."""
    from agentred.spec.models import Consequence

    authored = load_spec_dir(f"{SPEC_ROOT}/{AGENT}").config
    answers = {tool.name: tool.consequence for tool in authored.tools}
    answers["issue_refund"] = Consequence.INERT

    authored, emission = await _recovered(AGENT, answers)
    recovery = compare(authored, emission.config, uncovered=emission.unreadable)

    assert not recovery.faithful
    diverged = recovery.of(Verdict.DIVERGED)
    assert [field.subject for field in diverged] == ["tool issue_refund consequence"]
    assert diverged[0].authored == "money"
    assert diverged[0].recovered == "inert"
