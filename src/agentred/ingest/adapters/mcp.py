"""Reading an agent's tool surface off the MCP servers it is actually wired to.

This is the capability half at its most direct. A platform that puts its APIs behind MCP
connectors has already written down, in a form a machine can read, the one thing the harness
most needs and least wants to guess: the exact tools an agent can reach and the exact
arguments each of them takes. `tools/list` returns it, the agent could not function if it
were wrong, and re-reading it tomorrow gives the same answer. Nothing about this half is
inferred and nothing about it needs an operator to confirm it.

**It lists and never calls, and that is a rule rather than an implementation detail.**
Verifying a declared tool by invoking it is tempting and is exactly right for a tool server
we own, where rule 4 keeps money actions in test mode by construction. It is indefensible
against an integrator's agent: `issue_refund` on the other end of a connector is somebody's
production refund API, the reader has no way to know whether the credentials behind it are
live, and a reader that moves money to confirm a tool exists has caused the harm the harness
was brought in to prevent. So the session opened here is used for `initialize` and
`list_tools`, and `tests/ingest/adapters/test_mcp.py` fails if a call ever reaches the
server, because a promise about what a module does not do is worth what the test that
enforces it is worth.

**What it deliberately leaves unanswered.** `Consequence` is required on every declared tool
and no MCP response carries it. Guessing it from the name is the obvious move and it is the
one that quietly shrinks the suite. Defaulting every tool on `dispute_handler` to `inert`
takes it from twenty-one stakes to fourteen: all three data-leak stakes go, because the set
of tools that can leak is the set declared to disclose, and the four ungated actions go with
them. The run then reports a clean sheet on checks it never made. So every tool comes back
with its consequence undetermined, `AgentPackage.unresolved` says so, and proposing values
for them is a separate step with a person at the end of it.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentred.ingest.package import (
    AgentPackage,
    Evidence,
    Observation,
    Origin,
    ToolFacts,
)
from agentred.spec.models import Consequence

ADAPTER = "mcp"
"""The adapter name recorded on every piece of evidence this module produces."""

EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}
"""What a tool advertising no input schema is recorded as taking.

Not a default standing in for something unknown: a tool that declares no properties takes
no arguments, which is a fact rather than a hole, and the bound validator reads it as the
empty set either way.
"""

CONSEQUENCE_QUESTION = (
    "what does a wrong call to {name} cost: money, an obligation, disclosure, or nothing? "
    "The connector does not say, and a tool recorded as costing nothing drops out of the "
    "leak checks and out of the actions the suite treats as worth guarding."
)
"""What an operator has to answer for each tool, phrased so the cost of guessing is visible.

The reason the consequence is spelled out rather than left as a shrug: whoever is answering
twenty of these is under pressure to click through them, and `inert` is the answer that ends
the conversation fastest. Saying what `inert` does to the suite is the only defence the
wording can offer.
"""


class McpReadError(Exception):
    """An MCP connector could not be read.

    Raised instead of letting a transport or protocol error escape, so a caller reading four
    connectors can say which one failed. A connector that cannot be listed is not a connector
    with no tools, and collapsing the two would silently narrow an agent's declared surface.
    """


def _tool_facts(tool: Any, locator: str) -> ToolFacts:
    """Turn one entry of a `tools/list` response into recovered facts.

    Args:
        tool: An MCP `Tool` as the client returned it.
        locator: The connector URL, recorded as the evidence for every tool on it.

    Returns:
        The tool's name, description and schema as read, with its consequence undetermined.
    """
    name = str(tool.name)
    schema = getattr(tool, "input_schema", None)
    return ToolFacts(
        name=name,
        description=str(getattr(tool, "description", "") or ""),
        parameters=dict(schema) if schema else dict(EMPTY_SCHEMA),
        consequence=Observation[Consequence](
            value=None,
            origin=Origin.UNDETERMINED,
            evidence=Evidence(adapter=ADAPTER, locator=locator),
            question=CONSEQUENCE_QUESTION.format(name=name),
        ),
        evidence=Evidence(adapter=ADAPTER, locator=locator),
    )


@contextlib.asynccontextmanager
async def _listing_session(url: str, http_client: Any) -> AsyncIterator[Any]:
    """An initialised MCP session, opened only to be listed from.

    Args:
        url: The connector URL.
        http_client: An `httpx2.AsyncClient` to use instead of opening one. Tests pass an
            ASGI-transport client so the real protocol runs with no socket.

    Yields:
        A connected `mcp.ClientSession`.
    """
    from mcp.client.streamable_http import streamable_http_client

    from mcp import ClientSession

    async with (
        streamable_http_client(url, http_client=http_client) as (read, write, *_rest),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def read_connector(url: str, *, http_client: Any = None) -> tuple[ToolFacts, ...]:
    """Every tool one MCP connector advertises, with nothing invoked.

    Args:
        url: The connector URL to list.
        http_client: An `httpx2.AsyncClient` to use instead of opening one.

    Returns:
        One `ToolFacts` per advertised tool, in the order the server advertised them, each
        with an undetermined consequence.

    Raises:
        McpReadError: If the connector could not be reached, initialised or listed.
    """
    try:
        async with _listing_session(url, http_client) as session:
            listed = await session.list_tools()
    except Exception as error:
        raise McpReadError(
            f"could not list tools at {url}: {type(error).__name__}: {error}"
        ) from error
    return tuple(_tool_facts(tool, url) for tool in listed.tools)


async def read_agent(
    agent_id: str, connectors: Sequence[str], *, http_client: Any = None
) -> AgentPackage:
    """The tool surface of one agent, across every connector it is wired to.

    Args:
        agent_id: Stable identifier for the agent the connectors belong to.
        connectors: Connector URLs, read in order.
        http_client: An `httpx2.AsyncClient` to use instead of opening one.

    Returns:
        An `AgentPackage` holding the union of the connectors' tools, in connector order and
        within that in the order each server advertised. Its `unresolved` lists one question
        per tool, because no connector carries a consequence.

    Raises:
        McpReadError: If any connector could not be read, or if two connectors advertise the
            same tool name. A duplicate is refused rather than resolved: the agent would see
            one of the two and the reader has no way to know which, so a declaration built
            over it describes a tool surface that does not exist.
    """
    tools: list[ToolFacts] = []
    seen: dict[str, str] = {}
    for url in connectors:
        for tool in await read_connector(url, http_client=http_client):
            if tool.name in seen:
                raise McpReadError(
                    f"{agent_id} reaches two tools named {tool.name!r}, on {seen[tool.name]} "
                    f"and on {url}. The agent sees one of them and this reader cannot tell "
                    f"which."
                )
            seen[tool.name] = url
            tools.append(tool)
    return AgentPackage(agent_id=agent_id, tools=tuple(tools), sources=(ADAPTER,))
