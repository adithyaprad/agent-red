"""The tool server, and the boundary every violation is observed at.

An agent under test reaches the merchant's world only through the tools served here, and
every call is recorded here before its answer goes back. Nothing in the harness asks an agent
what it did. See `docs/DECISIONS/ADR-0005-oracle-at-the-tool-boundary.md`.
"""

from agentred.mcp.arena import Arena, ArenaError, PlantError, UnknownSessionError
from agentred.mcp.recorder import RecordedCall, ToolCallRecorder, read_stream
from agentred.mcp.world import World, fresh_world

__all__ = [
    "Arena",
    "ArenaError",
    "PlantError",
    "RecordedCall",
    "ToolCallRecorder",
    "UnknownSessionError",
    "World",
    "fresh_world",
    "read_stream",
]
