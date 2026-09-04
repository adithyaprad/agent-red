"""Run the tool server: the agents' face and the runner's face, on two ports.

    uv run python -m agentred.mcp.serve \
        --spec src/agentred/targets/specs/cart_recovery \
        --spec src/agentred/targets/specs/dispute_handler

The two faces are separate applications on separate ports on purpose. An agent is handed the
tool port and never the control port, so reading the call stream, restoring a world or
planting into one are not operations it can reach. Test mode is asserted by importing
`server`, before either socket opens.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentred.mcp.server import (
    DEFAULT_CONTROL_PORT,
    DEFAULT_TOOL_PORT,
    build_control_app,
    build_server,
    build_tool_app,
)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and serve both faces until interrupted.

    Args:
        argv: Command line arguments. Defaults to `sys.argv[1:]`.
    """
    parser = argparse.ArgumentParser(description="Serve the agent-red tool server.")
    parser.add_argument(
        "--spec",
        dest="specs",
        type=Path,
        action="append",
        required=True,
        help="Directory holding an agent's spec. Repeatable.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_TOOL_PORT, help="Tool port.")
    parser.add_argument(
        "--control-port", type=int, default=DEFAULT_CONTROL_PORT, help="Control port."
    )
    parser.add_argument(
        "--stream",
        type=Path,
        default=None,
        help="Persist the call stream to this JSON lines file as well as in memory.",
    )
    parser.add_argument(
        "--generated",
        action="store_true",
        help=(
            "Serve a shop derived from the agent's own declaration, and its tools from that "
            "declaration too. One agent per server, because one shop is derived from one "
            "declaration."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="What the generated shop derives from. Omit for the generator's fixed default.",
    )
    arguments = parser.parse_args(argv)

    import uvicorn

    server = build_server(
        arguments.specs,
        stream=arguments.stream,
        generated=arguments.generated,
        seed=arguments.seed,
    )
    if arguments.generated:
        print(f"serving a generated shop, {server.world_version}", flush=True)
    tools = uvicorn.Server(
        uvicorn.Config(
            build_tool_app(server), host=arguments.host, port=arguments.port, log_level="info"
        )
    )
    control = uvicorn.Server(
        uvicorn.Config(
            build_control_app(server),
            host=arguments.host,
            port=arguments.control_port,
            log_level="info",
        )
    )

    async def both() -> None:
        await asyncio.gather(tools.serve(), control.serve())

    asyncio.run(both())


if __name__ == "__main__":
    main()
