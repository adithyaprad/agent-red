"""Reading a tool surface off a live connector, and never touching it."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentred.ingest.adapters.mcp import McpReadError, read_agent, read_connector
from agentred.ingest.package import Origin
from agentred.mcp.server import ToolServer, build_tool_app
from agentred.spec import load_spec_dir

SPEC_ROOT = "src/agentred/targets/specs"


class CallSpy:
    """An ASGI app that passes everything through and remembers the JSON-RPC methods sent.

    Wrapping the transport rather than the client is deliberate. A spy on the client session
    proves the reader did not use one particular method on one particular object; a spy on
    the wire proves nothing reached the server, whatever route it took to get there.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.methods: list[str] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async def watched() -> Any:
            message = await receive()
            body = message.get("body", b"") if isinstance(message, dict) else b""
            if body:
                text = body.decode("utf-8", "replace")
                for method in ("tools/call", "tools/list", "initialize"):
                    if f'"{method}"' in text:
                        self.methods.append(method)
            return message

        await self.app(scope, watched, send)


@contextlib.asynccontextmanager
async def connector(*agents: str, agent_id: str) -> AsyncIterator[tuple[str, Any, CallSpy]]:
    """A connector URL and an ASGI client reaching a real tool server over the real protocol.

    Args:
        agents: Spec directories to serve.
        agent_id: Which agent's surface the URL is bound to.

    Yields:
        The connector URL, the HTTP client to read it with, and the wire spy.
    """
    import httpx2

    server = ToolServer([load_spec_dir(f"{SPEC_ROOT}/{name}") for name in agents])
    app = build_tool_app(server)
    spy = CallSpy(app)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=spy), base_url="http://arena"
        ) as http,
    ):
        yield f"http://arena/{agent_id}/ingest/s1", http, spy


async def test_a_connector_yields_the_tools_the_agent_declares() -> None:
    spec = load_spec_dir(f"{SPEC_ROOT}/dispute_handler")
    declared = [tool.name for tool in spec.config.tools]

    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, _spy):
        facts = await read_connector(url, http_client=http)

    assert [fact.name for fact in facts] == declared


async def test_the_schema_read_is_the_schema_served() -> None:
    spec = load_spec_dir(f"{SPEC_ROOT}/dispute_handler")
    declared = {tool.name: tool.parameters for tool in spec.config.tools}

    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, _spy):
        facts = await read_connector(url, http_client=http)

    assert {fact.name: fact.parameters for fact in facts} == declared


async def test_reading_a_connector_never_calls_a_tool() -> None:
    """The guarantee the module is built around, enforced at the wire rather than promised.

    A reader that invokes a tool to confirm it exists has moved somebody's money to answer a
    question about a schema. This is the test that stops that from ever being true.
    """
    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, spy):
        await read_connector(url, http_client=http)

    assert "tools/list" in spy.methods
    assert "tools/call" not in spy.methods


async def test_every_tool_comes_back_with_its_consequence_undetermined() -> None:
    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, _spy):
        facts = await read_connector(url, http_client=http)

    assert facts
    assert all(fact.consequence.origin is Origin.UNDETERMINED for fact in facts)
    assert all(fact.consequence.value is None for fact in facts)


async def test_the_package_reports_one_hole_per_tool() -> None:
    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, _spy):
        package = await read_agent("dispute_handler", [url], http_client=http)

    assert len(package.unresolved) == len(package.tools)
    subjects = {subject for subject, _question in package.unresolved}
    assert subjects == {f"tool {tool.name}" for tool in package.tools}


async def test_the_question_names_the_tool_and_what_guessing_costs() -> None:
    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, _spy):
        package = await read_agent("dispute_handler", [url], http_client=http)

    questions = dict(package.unresolved)
    asked = questions["tool issue_refund"]
    assert "issue_refund" in asked
    assert "drops out of the leak checks" in asked


async def test_two_connectors_serving_the_same_tool_name_are_refused() -> None:
    async with connector("dispute_handler", agent_id="dispute_handler") as (url, http, _spy):
        with pytest.raises(McpReadError, match="two tools named"):
            await read_agent("dispute_handler", [url, url], http_client=http)


async def test_an_unreachable_connector_names_itself() -> None:
    import httpx2

    async with httpx2.AsyncClient(base_url="http://arena") as http:
        with pytest.raises(McpReadError, match="could not list tools at"):
            await read_connector("http://arena/nobody/ingest/s1", http_client=http)
