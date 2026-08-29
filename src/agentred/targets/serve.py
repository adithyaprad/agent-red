"""Run one target as a web service.

Used by `docker-compose.yml` and by anyone who wants a target up in front of them:

    uv run python -m agentred.targets.serve --spec src/agentred/targets/specs/cart_recovery

The spec is a directory holding `config.yaml` and `policy.yaml`, so a target is one path
that cannot be half-specified. Test mode is asserted by importing `runtime`, before the
socket is opened.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentred.spec import load_spec_dir
from agentred.targets.runtime import build_agent, build_app

DEFAULT_PORT = 8081


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and serve one target.

    Args:
        argv: Command line arguments. Defaults to `sys.argv[1:]`.
    """
    parser = argparse.ArgumentParser(description="Serve one agent-red target.")
    parser.add_argument("--spec", type=Path, required=True, help="Directory holding the spec.")
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    arguments = parser.parse_args(argv)

    import uvicorn

    app = build_app(build_agent(load_spec_dir(arguments.spec)))
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")


if __name__ == "__main__":
    main()
