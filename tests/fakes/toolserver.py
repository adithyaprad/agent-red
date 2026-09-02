"""An MCP client that reaches the tool server without a socket.

The transport is the application itself, so a test exercises the real protocol, the real
path binding and the real recorder, and still makes no network call. What is absent is the
socket and the agent, and both have their own tests.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from agentred.mcp.server import ToolServer, build_tool_app


@contextlib.asynccontextmanager
async def connected(
    server: ToolServer, *, agent_id: str, run: str = "run-test", session: str = "s1"
) -> AsyncIterator[Any]:
    """An initialised MCP session bound to one agent, run and conversation.

    Args:
        server: The tool server to connect to.
        agent_id: Which agent's tool surface to ask for.
        run: The run calls are recorded under.
        session: The conversation whose world the calls act on.

    Yields:
        A connected `mcp.ClientSession`.
    """
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    from mcp import ClientSession

    app = build_tool_app(server)
    url = f"http://arena/{agent_id}/{run}/{session}"
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://arena"
        ) as http,
        streamable_http_client(url, http_client=http) as (read, write),
        ClientSession(read, write) as client,
    ):
        await client.initialize()
        yield client


def text_of(result: Any) -> str:
    """The text content of a tool result."""
    return str(result.content[0].text)
