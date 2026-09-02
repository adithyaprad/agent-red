"""The workflow engine's half of the tool connector: declared MCP tools, as callables.

A target built on a workflow engine still reaches every capability through the tool server
in `mcp/server.py`, because that is where a call is recorded and therefore where a violation
is observed (ADR-0005). What this module does is turn the tools that server advertises into
things a workflow's LLM nodes can call, without inventing a second description of them.

**The schema is the server's, not ours.** Each function is built from what `tools/list`
returned: the name, the description and the input schema the agent's `config.yaml` declared.
Nothing here re-states a parameter, so a workflow-built target and an SDK-built target are
offered the same tool surface, and a spec change reaches both without a code change.

**Why this is not agno's own MCP toolkit.** `agno.tools.mcp` requires the MCP SDK below 2.0
(`mcp<2,>=1.9.2`) and imports `McpError`, a name that version renamed. The tool server is
built on the 2.1 server API, which the Claude Agent SDK also pins. Downgrading to satisfy
agno would break the server every target connects to, so the client side is written here
instead: a dozen lines against the MCP client API, and the pin conflict never arises.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

TOOL_RESULT_ERROR_KEY = "error"
"""What a bridged call returns when the transport itself failed.

A tool that answers `{"error": ...}` is a tool the agent has to cope with, and the targets
are written to be attacked rather than to be robust, so a failure is handed to the model in
the shape a real tool failure arrives in. What is not swallowed is a protocol error: see
`_result_text`.
"""


def _result_text(result: Any) -> str:
    """Flatten an MCP tool result into what the model should see.

    The tool server answers with a single text block holding JSON, because that is what its
    handlers return. Anything else is passed through as text rather than reshaped, so a
    server that later answers with two blocks does not silently lose one.

    Args:
        result: The `CallToolResult` the client session returned.

    Returns:
        The concatenated text of the result's content blocks.
    """
    parts = [
        str(block.text) for block in getattr(result, "content", []) or [] if hasattr(block, "text")
    ]
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structured_content", None)
    return json.dumps(structured) if structured is not None else ""


@contextlib.asynccontextmanager
async def mcp_functions(url: str, *, http_client: Any = None) -> AsyncIterator[list[Any]]:
    """The tools served at `url`, as agno `Function` objects, for as long as the session.

    The session is held open for the duration of the context, so a workflow that makes six
    tool calls pays for one MCP handshake rather than six. Closing the context closes the
    session; a function held past that point will fail, which is correct, because the run it
    belonged to is over.

    Args:
        url: The connector URL for one agent, run and conversation. Everything about
            attribution follows from this path (see `mcp/server.py`), so a caller that
            builds it wrongly gets calls recorded against the wrong conversation rather than
            an error, and `TargetAgent.connector_url` is the only thing that should build it.
        http_client: An `httpx2.AsyncClient` to use instead of opening one. Tests pass an
            ASGI-transport client so the real protocol runs with no socket.

    Yields:
        One `Function` per tool the server advertises, in the order it advertised them.
    """
    from agno.tools.function import Function
    from mcp.client.streamable_http import streamable_http_client

    from mcp import ClientSession

    async with (
        streamable_http_client(url, http_client=http_client) as (read, write, *_rest),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listed = await session.list_tools()

        def bind(name: str) -> Any:
            """One tool's entrypoint, closed over its name.

            Built in a factory rather than in the loop body because a closure over the loop
            variable would give every tool the last name in the list.
            """

            async def call(**arguments: Any) -> str:
                try:
                    return _result_text(await session.call_tool(name, arguments))
                except Exception as error:
                    return json.dumps({TOOL_RESULT_ERROR_KEY: f"{type(error).__name__}: {error}"})

            call.__name__ = name
            return call

        yield [
            Function(
                name=tool.name,
                description=tool.description or "",
                parameters=dict(tool.input_schema or {"type": "object", "properties": {}}),
                entrypoint=bind(tool.name),
                skip_entrypoint_processing=True,
            )
            for tool in listed.tools
        ]
