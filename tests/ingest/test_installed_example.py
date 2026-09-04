"""Reading the example agent the way somebody pointed at it for the first time would.

`examples/retention_desk/` is an agent as an install wizard leaves it: a manifest, a connector
advertising the merchant's tools, the limits an operator typed into a form, the prose the skill
runs on, and the workflow the installer mapped it onto. Neither half of a declaration is in
that directory, so a spec that loads is evidence the reader produced one rather than that
somebody wrote one.

The connector is served in-process over the real protocol. That matters more than the
convenience: the point of the example is that the tool surface comes from a server which has
never heard of agent-red, and reading it off a server agent-red built from a declaration would
prove nothing.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agentred.ingest.emit import EmitError, to_config, to_spec
from agentred.ingest.read import load_manifest, read_agent
from agentred.spec.models import Consequence

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "retention_desk"
MANIFEST = EXAMPLE / "agent.manifest.yaml"


@contextlib.asynccontextmanager
async def platform() -> AsyncIterator[Any]:
    """The merchant's connector, served over the real protocol with no socket."""
    import httpx2
    from examples.retention_desk.connector import build_app

    app = build_app(EXAMPLE / "tools.registry.yaml")
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://127.0.0.1:8093"
        ) as http,
    ):
        yield http


async def read_installed() -> Any:
    """Read every source the example's manifest names."""
    async with platform() as http:
        return await read_agent(load_manifest(MANIFEST), http_client=http)


async def test_every_named_source_contributes() -> None:
    package = await read_installed()

    assert set(package.sources) == {"mcp", "instance", "workflow"}


async def test_the_connector_supplies_the_whole_tool_surface() -> None:
    package = await read_installed()

    assert len(package.tools) == 8
    assert {item.name for item in package.tools} >= {
        "get_subscription",
        "apply_retention_discount",
        "refund_charge",
        "send_whatsapp",
    }


async def test_the_form_supplies_the_money_limits_and_the_workflow_the_ordering() -> None:
    package = await read_installed()
    names = {facts.rule.name for facts in package.rules}

    assert "refund_charge_amount_limit" in names
    assert "apply_retention_discount_percent_limit" in names
    assert "cancel_subscription_follows_get_subscription" in names


async def test_no_reader_answers_what_a_wrong_call_costs() -> None:
    package = await read_installed()

    assert len(package.unresolved) == len(package.tools)
    assert all(subject.startswith("tool ") for subject, _ in package.unresolved)


async def test_a_declaration_is_refused_while_a_question_is_open() -> None:
    package = await read_installed()

    with pytest.raises(EmitError, match="would be a guess reported as a fact"):
        to_spec(package, version="1.2", model="claude-sonnet-5")


async def test_an_answered_package_emits_a_spec_for_an_agent_nobody_declared() -> None:
    package = await read_installed()
    answered = package.answered(
        {
            "get_subscription": Consequence.DISCLOSURE,
            "get_customer": Consequence.DISCLOSURE,
            "get_billing_history": Consequence.DISCLOSURE,
            "apply_retention_discount": Consequence.MONEY,
            "refund_charge": Consequence.MONEY,
            "pause_subscription": Consequence.OBLIGATION,
            "cancel_subscription": Consequence.OBLIGATION,
            "send_whatsapp": Consequence.OBLIGATION,
        },
        by="fde",
    )

    from agentred.cli import _load_subjects

    subjects = _load_subjects(EXAMPLE / "subjects.yaml")
    _, spec = to_spec(answered, version="1.2", model="claude-sonnet-5", subjects=subjects)

    assert spec is not None
    assert spec.config.engine == "workflow"
    assert len(spec.policy.bounds) == 2
    assert spec.policy.data_scope is not None


async def test_the_read_says_what_no_reader_supplies() -> None:
    package = await read_installed()
    answered = package.answered({item.name: Consequence.INERT for item in package.tools}, by="fde")

    emission = to_config(answered, version="1.2", model="claude-sonnet-5")

    assert any("tool behaviours" in line for line in emission.unreadable)
    assert any("subjects" in line for line in emission.unreadable)
