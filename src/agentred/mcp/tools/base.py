"""How a tool implementation is declared and looked up.

A `ToolSet` is the runtime half of what a spec's `config.yaml` declares. The spec is the
contract every other package reads; this is the code behind it. They are checked against
each other when a target starts, because a declared tool with no implementation is an agent
that fails mid-conversation, and an implemented tool that is not declared is a capability
the harness will never attack and the scorecard will never mention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentred.mcp.world import World

ToolHandler = Callable[[World, dict[str, Any]], dict[str, Any]]
"""Takes the conversation's private world and the arguments, returns a JSON-able result."""


@dataclass(frozen=True)
class ToolImplementation:
    """One callable tool.

    Attributes:
        name: Must match a tool name in the agent's `config.yaml`.
        handler: The implementation. Returns a result dict; a failure is a returned
            `{"error": ...}` rather than an exception, because a raised exception ends the
            conversation and a real tool failure does not.
    """

    name: str
    handler: ToolHandler


class ToolSet:
    """The tools one agent can call, keyed by name.

    Attributes:
        implementations: Tools by name, in registration order.
    """

    def __init__(self, *implementations: ToolImplementation) -> None:
        """Build a toolset.

        Args:
            *implementations: The tools, in the order they should appear to the agent.

        Raises:
            ValueError: If a name is registered twice.
        """
        self.implementations: dict[str, ToolImplementation] = {}
        for implementation in implementations:
            if implementation.name in self.implementations:
                raise ValueError(f"duplicate tool implementation {implementation.name!r}")
            self.implementations[implementation.name] = implementation

    @property
    def names(self) -> frozenset[str]:
        """Every implemented tool name."""
        return frozenset(self.implementations)

    def call(self, name: str, world: World, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool against one conversation's world.

        Args:
            name: The tool to call.
            world: That conversation's private world.
            arguments: Arguments as the model supplied them.

        Returns:
            The tool result, or `{"error": ...}` for an unknown tool.
        """
        implementation = self.implementations.get(name)
        if implementation is None:
            return {"error": f"no such tool: {name}"}
        return implementation.handler(world, arguments)


def tool(name: str) -> Callable[[ToolHandler], ToolImplementation]:
    """Decorator turning a handler function into a named `ToolImplementation`."""

    def wrap(handler: ToolHandler) -> ToolImplementation:
        return ToolImplementation(name=name, handler=handler)

    return wrap


def as_number(value: Any) -> float | None:
    """Coerce a tool argument to a number, or `None` if it is not one.

    Arguments arrive as JSON and a model will sometimes send `"35"` or `"35%"`. Coercing
    here keeps that from reading as a tool failure, because a target that rejected a
    malformed argument would look more careful than the agents this stands in for.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().removesuffix("%").replace(",", "").removeprefix("$")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None
