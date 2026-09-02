"""The workflow engine's tool connector: the server's own tools, as callables.

Every test here runs the real MCP protocol against the real tool server over an ASGI
transport, so what is exercised is the handshake, the tool listing, the path binding and the
recorder. What is absent is the socket and the model, and both have their own tests.

The bridge is opened inside each test rather than in a fixture. The MCP client holds an
anyio task group, and a fixture that yielded across it would tear the group down in a
different task from the one that entered it.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx2

from agentred.mcp.server import ToolServer, build_tool_app
from agentred.spec import load_spec_dir
from agentred.targets.mcp_bridge import mcp_functions

SPEC_ROOT = "src/agentred/targets/specs"
CART = "CART-8891"
RUN = "run-test"


@dataclass
class Bridged:
    """A connected bridge, with the pieces a test asserts against.

    Attributes:
        server: The tool server the calls landed at, holding the recorder.
        functions: The bridged tools, in the order the server advertised them.
        by_name: The same tools, keyed by name.
    """

    server: ToolServer
    functions: list
    by_name: dict


@contextlib.asynccontextmanager
async def bridged(
    name: str = "cart_recovery", session: str = "s1", run: str = RUN
) -> AsyncIterator[Bridged]:
    """A bridge onto one agent's tools, and the server behind it."""
    server = ToolServer([load_spec_dir(f"{SPEC_ROOT}/{name}")])
    app = build_tool_app(server)
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://arena"
        ) as http,
        mcp_functions(f"http://arena/{name}/{run}/{session}", http_client=http) as functions,
    ):
        yield Bridged(
            server=server,
            functions=list(functions),
            by_name={function.name: function for function in functions},
        )


class TestWhatTheAgentIsOffered:
    async def test_every_declared_tool_arrives_in_declaration_order(self) -> None:
        async with bridged() as cart:
            assert [function.name for function in cart.functions] == [
                tool.name for tool in cart.server.specs["cart_recovery"].config.tools
            ]

    async def test_the_schema_is_the_one_the_config_declared(self) -> None:
        """Not a schema this module wrote. A second description would drift from the first."""
        async with bridged() as cart:
            declared = cart.server.specs["cart_recovery"].config.tools_by_name["apply_discount"]
            assert cart.by_name["apply_discount"].parameters == dict(declared.parameters)

    async def test_a_tool_taking_nothing_still_has_an_object_schema(self) -> None:
        async with bridged() as cart:
            assert cart.by_name["list_abandoned_carts"].parameters["type"] == "object"

    async def test_an_agent_is_offered_only_its_own_tools(self) -> None:
        async with bridged(name="dispute_handler") as dispute:
            assert "lookup_cart" not in dispute.by_name
            assert "issue_refund" in dispute.by_name


class TestWhatACallDoes:
    async def test_a_call_returns_what_the_tool_returned(self) -> None:
        async with bridged() as cart:
            result = json.loads(await cart.by_name["lookup_cart"].entrypoint(cart_id=CART))
            assert result["cart_id"] == CART
            assert result["total"] > 0

    async def test_a_call_is_recorded_with_its_arguments(self) -> None:
        async with bridged() as cart:
            await cart.by_name["apply_discount"].entrypoint(cart_id=CART, percent=35)
            recorded = cart.server.recorder.calls(RUN, "s1")
            assert [(call.name, call.arguments) for call in recorded] == [
                ("apply_discount", {"cart_id": CART, "percent": 35})
            ]

    async def test_the_bridge_does_not_enforce_what_the_tools_do_not(self) -> None:
        """A discount of 35 against a declared ceiling of 10 succeeds, and is recorded."""
        async with bridged() as cart:
            result = json.loads(
                await cart.by_name["apply_discount"].entrypoint(cart_id=CART, percent=35)
            )
            assert result["percent_applied"] == 35

    async def test_a_tool_failure_reaches_the_model_rather_than_raising(self) -> None:
        async with bridged() as cart:
            result = json.loads(await cart.by_name["lookup_cart"].entrypoint(cart_id="CART-0000"))
            assert "error" in result

    async def test_calls_are_attributed_to_the_session_in_the_url(self) -> None:
        """The session is in the path, so an agent cannot address another conversation."""
        async with bridged(session="alpha") as cart:
            await cart.by_name["lookup_cart"].entrypoint(cart_id=CART)
            assert len(cart.server.recorder.calls(RUN, "alpha")) == 1
            assert cart.server.recorder.calls(RUN, "beta") == ()


class TestTheScheduledEntryPoint:
    async def test_the_listing_carries_the_field_a_shopper_typed(self) -> None:
        """`note` is what an adversary writes. It is returned verbatim. See ADR-0006."""
        async with bridged() as cart:
            rows = json.loads(await cart.by_name["list_abandoned_carts"].entrypoint())["carts"]
            assert all("note" in row for row in rows)
            assert any(row["note"] for row in rows)
