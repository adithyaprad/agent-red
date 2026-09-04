"""Reading an installed agent's declaration off its own platform, with nothing authored.

`tests/ingest/installed/` holds an agent as a builder installs one: a manifest naming the
connectors it reaches, the limits its operator configured, the prose it runs on, and the
workflow the installer mapped it onto. None of those files were written for agent-red. This
is the test that says what agent-red can recover from them, and, as importantly, what it
cannot and therefore has to ask.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentred.ingest.diff import Verdict, compare
from agentred.ingest.emit import EmitError, to_spec
from agentred.ingest.package import Origin
from agentred.ingest.read import ManifestError, load_manifest, read_agent
from agentred.mcp.server import ToolServer, build_tool_app
from agentred.spec import load_spec_dir

MANIFEST = "tests/ingest/installed/agent.manifest.yaml"
AUTHORED = "src/agentred/targets/specs/dispute_handler"


@contextlib.asynccontextmanager
async def platform() -> AsyncIterator[Any]:
    """The agent's connector, served over the real protocol with no socket."""
    import httpx2

    app = build_tool_app(ToolServer([load_spec_dir(AUTHORED)]))
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://arena"
        ) as http,
    ):
        yield http


async def read_installed() -> Any:
    """Read every source the manifest names."""
    async with platform() as http:
        return await read_agent(load_manifest(MANIFEST), http_client=http)


async def test_every_named_source_contributes() -> None:
    package = await read_installed()

    assert package.sources == ("mcp", "instance", "workflow")
    assert len(package.tools) == 9
    assert package.data_sources
    assert package.instructions


async def test_the_limits_an_operator_configured_come_back_as_bounds() -> None:
    """The money rules, read from where a no-code builder has to store them."""
    package = await read_installed()

    bounds = {facts.rule.name: facts for facts in package.rules if hasattr(facts.rule, "maximum")}
    assert bounds["issue_refund_amount_limit"].rule.maximum == 50000
    assert bounds["apply_discount_percent_limit"].rule.maximum == 10
    assert all(facts.origin is Origin.DECLARED for facts in bounds.values())
    assert "instance.yaml" in bounds["issue_refund_amount_limit"].evidence.locator


async def test_the_workflow_declares_the_ordering_nobody_wrote_down() -> None:
    """A step graph is a policy statement, and this is the half prose is worst at."""
    package = await read_installed()

    named = {facts.rule.name for facts in package.rules}
    assert "issue_refund_follows_get_order" in named
    assert "accept_dispute_follows_get_dispute" in named

    rule = next(f.rule for f in package.rules if f.rule.name == "issue_refund_follows_get_order")
    assert rule.matched_by == ("order_id",)


async def test_two_tools_in_order_that_share_no_argument_are_not_a_rule() -> None:
    """Otherwise the graph emits its cross product and the real rules are lost in it."""
    package = await read_installed()

    named = {facts.rule.name for facts in package.rules}
    assert "send_whatsapp_follows_get_dispute" not in named


async def test_what_a_form_cannot_hold_is_named_on_every_read() -> None:
    """A policy of per-call ceilings is a true statement and a flattering one alone."""
    package = await read_installed()

    joined = " ".join(package.notes)
    assert "added up" in joined
    assert "read from the record being acted on" in joined


async def test_nothing_is_emitted_while_a_question_is_unanswered() -> None:
    package = await read_installed()

    with pytest.raises(EmitError, match="would be a guess reported as a fact"):
        to_spec(package, version="1.7", model="claude-sonnet-5")


async def test_an_answered_package_emits_both_halves_of_a_declaration() -> None:
    """The whole point: a spec the harness will run, from files nobody wrote for it."""
    authored = load_spec_dir(AUTHORED)
    answers = {tool.name: tool.consequence for tool in authored.config.tools}
    package = await read_installed()
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
        engine=package.engine,
        instructions=package.instructions,
        rules=package.rules,
        data_sources=package.data_sources,
        data_scope=package.data_scope,
        sources=package.sources,
        notes=package.notes,
    )

    emission, spec = to_spec(
        resolved, version="1.7", model="claude-sonnet-5", subjects=authored.subjects
    )

    assert spec is not None
    assert spec.config.engine == "workflow"
    assert len(spec.policy.bounds) == 2
    assert len(spec.policy.preconditions) == 10

    recovery = compare(
        authored.config,
        emission.config,
        uncovered=emission.unreadable,
        authored_policy=authored.policy,
        recovered_policy=emission.policy,
    )
    assert recovery.faithful, recovery.render()

    # docs/INTEGRATION.md prints these four counts as what a read of this agent recovers.
    # Pinned rather than bounded, because the section around them explains each number: which
    # rule shapes a form field cannot hold, and which three items are questions rather than
    # rules. A count that drifts turns that explanation into a wrong one, and the drift is
    # invisible from the code.
    assert len(recovery.of(Verdict.MATCHED)) == 32
    assert len(recovery.of(Verdict.DIVERGED)) == 0
    assert len(recovery.of(Verdict.UNCOVERED)) == 10
    assert len(recovery.of(Verdict.ADDED)) == 7


async def test_a_source_the_manifest_names_and_cannot_read_is_refused() -> None:
    """Skipping it would produce a thin declaration indistinguishable from an honest one."""
    import httpx2

    async with httpx2.AsyncClient(base_url="http://arena") as http:
        with pytest.raises(ManifestError, match="could not list tools"):
            await read_agent(load_manifest(MANIFEST), http_client=http)
