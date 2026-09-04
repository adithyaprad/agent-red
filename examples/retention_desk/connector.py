"""The merchant's tool connector, serving what it advertises over MCP.

This stands in for the thing an agent platform already runs: the connector an operator wires
their agent to, publishing the operations that platform exposes. It imports nothing from
agent-red, and it holds no policy, no world and no notion of an attack. Its only content is
`tools.registry.yaml`, which is the merchant's, and its only job is to answer `tools/list`
the way any connector does.

Reading a declaration off a server agent-red itself built from a declaration would prove
nothing, so this exists to make the reading real: the tool surface agent-red recovers comes
from a process that has never heard of it.

**Calling is refused rather than implemented.** A connector in front of a real subscription
platform is somebody's production billing API, and the reader is documented never to invoke
one. Under test the agent reaches its capabilities through agent-red's own recorded tool
server, so nothing here is ever called by anything that matters, and a handler that did
something would only be a handler nobody had a reason to trust.

    uv run python examples/retention_desk/connector.py --port 8093
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PORT = 8093
REGISTRY = Path(__file__).parent / "tools.registry.yaml"


def load_registry(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read the tools one connector advertises.

    Args:
        path: The registry file.

    Returns:
        The server name and its tool entries, exactly as written.

    Raises:
        ValueError: If the file is not a mapping or advertises no tools.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    tools = raw.get("tools") or []
    if not tools:
        raise ValueError(f"{path} advertises no tools")
    return str(raw.get("server") or path.stem), list(tools)


def build_app(path: Path = REGISTRY) -> Any:
    """An MCP server advertising one registry and serving no calls.

    Args:
        path: The registry to advertise.

    Returns:
        A Starlette application speaking MCP streamable HTTP at any path.
    """
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp_types import ListToolsResult, Tool
    from starlette.applications import Starlette
    from starlette.routing import Mount

    name, entries = load_registry(path)

    async def on_list_tools(ctx: Any, params: Any) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=str(entry["name"]),
                    description=str(entry.get("description") or ""),
                    inputSchema=dict(entry.get("parameters") or {"type": "object"}),
                )
                for entry in entries
            ]
        )

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        raise RuntimeError(
            "this connector advertises its tools and does not run them. Under test an agent "
            "reaches its capabilities through the recorded tool server instead."
        )

    low = Server(
        name=name,
        instructions="A subscription platform's operations.",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
    manager = StreamableHTTPSessionManager(app=low, stateless=True, json_response=True)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(lifespan=lifespan, routes=[Mount("/", app=manager.handle_request)])


def main(argv: list[str] | None = None) -> None:
    """Serve the registry.

    Args:
        argv: Command line arguments. Defaults to `sys.argv[1:]`.
    """
    parser = argparse.ArgumentParser(description="Serve a merchant's tool connector.")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    arguments = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(build_app(arguments.registry), host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
