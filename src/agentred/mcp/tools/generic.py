"""Running a declared tool with no code written for it.

The third per-merchant integration, and the one this module removes. ADR-0005 took the oracle
off the agent's own self-report, ADR-0006 took the delivery path off chat, and both were the
same shape: something the harness had quietly assumed it could supply. A tool surface is the
same shape again. `tools/cart.py` and `tools/dispute.py` are four hundred lines of Python that
exist because two particular agents sell two particular things, and nobody outside this
repository can write the third.

So a tool is served from its declaration. The merchant says what shape each tool has, which
of their data it touches and which fields it changes, and the handler is derived. Three shapes
cover it, because three is what the shapes are once the domain is removed: fetch the record
somebody named, fetch the records matching something, or change something.

**Nothing here enforces policy.** A generic write pays what it is asked, sets what it is told
and does not check the record was in a state that permitted it. That is the same posture the
hand-written handlers take and for the same reason: a tool that refused would be answering the
question the run exists to ask. The one thing it does honour is an idempotency key, because a
real payments API honours one, and a synthetic surface that charged twice for a key would make
a correctly written agent look reckless.

**What it deliberately cannot do** is compute a value from the record it is writing to, a
percentage of a total being the case that comes up first. Expressing that needs a small
language in the declaration, every tool then becomes arguable, and the merchant writing the
declaration is the ops team rule 10 of this project describes. Both shipped agents keep their
hand-written handlers, which is what a generic handler is checked against.
"""

from __future__ import annotations

from typing import Any

from agentred.mcp.tools.base import ToolImplementation, ToolSet, as_number
from agentred.mcp.world import Record, UnknownCollectionError, World
from agentred.spec.models import (
    AgentSpec,
    FieldWrite,
    ToolBehaviour,
    ToolDeclaration,
    ToolShape,
    WriteMode,
)


class UndeclaredToolError(ValueError):
    """A tool was asked for generically and its declaration does not say what it does.

    Refused rather than served as a no-op. A tool that returns nothing is a capability the
    agent has and never successfully uses, so every rule that action could break reports as
    never evaluated and the run reads as an agent that had nothing to be talked into.
    """


def _scalar(value: Any) -> str | None:
    """`value` as a string if it is a scalar, `None` otherwise.

    A boolean is not an identifier, and Python would otherwise render one as a number.
    """
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _shown(record: Record, behaviour: ToolBehaviour) -> Record:
    """The record as the tool returns it.

    An undeclared field list returns the whole record, which is what a support tool does and
    is deliberate: a surface that withheld the awkward fields would be hiding exactly the
    internal note whose disclosure the run is trying to observe.
    """
    if not behaviour.result_fields:
        return dict(record)
    return {name: record[name] for name in behaviour.result_fields if name in record}


def _named(
    world: World, behaviour: ToolBehaviour, arguments: dict[str, Any]
) -> tuple[list[Record], str]:
    """Every record the arguments name, and the key that named them.

    The first declared key is the one the collection is keyed by; a later one is matched
    against a field of each record, which is how a reference that names more than one record
    answers with all of them rather than with whichever was found first. Choosing here would
    hide the second record, and one debt filed twice is among the things being measured.
    """
    records = world[behaviour.source]
    for position, key in enumerate(behaviour.keys):
        wanted = _scalar(arguments.get(key))
        if wanted is None or not wanted.strip():
            continue
        wanted = wanted.strip()
        if position == 0:
            found = records.get(wanted)
            return ([found] if found is not None else []), key
        return [row for row in records.values() if _scalar(row.get(key)) == wanted], key
    return [], ""


def _read_one(behaviour: ToolBehaviour, source: str) -> Any:
    """A handler that fetches the record an argument names."""

    def handler(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
        found, key = _named(world, behaviour, arguments)
        if not key:
            named = " or a ".join(behaviour.keys)
            return {"error": f"give a {named}"}
        if not found:
            return {"error": f"no {source} for {key} {_scalar(arguments.get(key))}"}
        if len(found) == 1:
            return _shown(found[0], behaviour)
        return {source: [_shown(row, behaviour) for row in found]}

    return handler


def _list_where(behaviour: ToolBehaviour, source: str) -> Any:
    """A handler that fetches every record matching the arguments given.

    An argument that was not supplied narrows nothing, so a call with no arguments returns
    the collection. That is the shape a scheduled agent's selection call has, and it is why
    the cohort a firing was woken about is read from the world rather than from this result.
    """

    def handler(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for record in world[behaviour.source].values():
            if all(
                _scalar(arguments.get(name)) in (None, _scalar(record.get(name)))
                for name in behaviour.filters
            ):
                rows.append(_shown(record, behaviour))
        return {source: rows, "count": len(rows)}

    return handler


def _apply(record: Record, write: FieldWrite, arguments: dict[str, Any]) -> Any:
    """Put one declared field write onto a record, returning what it wrote."""
    incoming = write.value if write.argument == "" else arguments.get(write.argument)
    if write.mode is WriteMode.SET:
        record[write.field] = incoming
        return incoming
    if write.mode is WriteMode.ADD:
        addend = as_number(incoming)
        if addend is None:
            return None
        record[write.field] = round(float(as_number(record.get(write.field)) or 0.0) + addend, 2)
        return record[write.field]
    existing = record.setdefault(write.field, [])
    if isinstance(existing, list):
        existing.append(incoming)
    return incoming


def _write(behaviour: ToolBehaviour, tool: ToolDeclaration) -> Any:
    """A handler that changes a record, or records an effect that leaves the merchant.

    The result carries what it was asked for, what it wrote, and whether the call was a
    replay. Not the whole record: a write that answered with everything it touched would put
    fields into the log that the agent never asked to see, and the scope check reads that log
    to decide what the agent reached. A write that should answer with the record says so by
    declaring `result_fields`.
    """

    def handler(world: World, arguments: dict[str, Any]) -> dict[str, Any]:
        key = behaviour.idempotency_argument
        supplied = (_scalar(arguments.get(key)) or "").strip() if key else ""
        if supplied and supplied in world.settled_keys:
            settled = dict(world.settled_keys[supplied])
            settled["replayed"] = True
            return settled

        record: Record | None = None
        if behaviour.source and behaviour.keys:
            found, named = _named(world, behaviour, arguments)
            if not named:
                return {"error": f"give a {' or a '.join(behaviour.keys)}"}
            if not found:
                asked = _scalar(arguments.get(named))
                return {"error": f"no {behaviour.source} for {named} {asked}"}
            record = found[0]

        written: dict[str, Any] = {}
        for change in behaviour.writes:
            if record is not None:
                written[change.field] = _apply(record, change, arguments)

        # `action` is the ledger's own column, so an argument of that name would collide with
        # it. Prefixed rather than dropped: the argument is what the check reads.
        detail = {("argument_action" if k == "action" else k): v for k, v in arguments.items()}
        world.record(tool.name, **detail)

        result: dict[str, Any] = {**arguments, **written, "replayed": False}
        if record is not None and behaviour.result_fields:
            result.update(_shown(record, behaviour))
        if supplied:
            world.settled_keys[supplied] = dict(result)
        return result

    return handler


def handler_for(tool: ToolDeclaration) -> Any:
    """Build the implementation a tool's declaration describes.

    Args:
        tool: A declared tool carrying a `behaviour`.

    Returns:
        A `ToolHandler`: takes the conversation's world and the arguments, returns a result.

    Raises:
        UndeclaredToolError: If the tool declares no behaviour.
    """
    behaviour = tool.behaviour
    if behaviour is None:
        raise UndeclaredToolError(
            f"tool {tool.name!r} declares no behaviour, so there is nothing to serve. Either "
            f"declare one or bind a hand-written handler for this agent."
        )
    source = behaviour.source or tool.name
    if behaviour.shape is ToolShape.READ_ONE:
        return _read_one(behaviour, source)
    if behaviour.shape is ToolShape.LIST_WHERE:
        return _list_where(behaviour, source)
    return _write(behaviour, tool)


def toolset_for(spec: AgentSpec) -> ToolSet:
    """Every tool an agent declares, served from the declaration alone.

    Args:
        spec: The validated spec. Its tools carry the behaviours; its data sources are what
            those behaviours may read, already cross-checked at load.

    Returns:
        A `ToolSet` the tool server can serve, with one implementation per declared tool.

    Raises:
        UndeclaredToolError: On the first tool that declares no behaviour. Refused as a set
            rather than per call, so an incomplete declaration stops a run at startup instead
            of halfway through a suite.
    """
    return ToolSet(
        *(
            ToolImplementation(name=tool.name, handler=handler_for(tool))
            for tool in spec.config.tools
        )
    )


__all__ = ["UndeclaredToolError", "UnknownCollectionError", "handler_for", "toolset_for"]
