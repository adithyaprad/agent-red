"""What a collection's records carry, read out of the declaration.

Half of ADR-0007's crossing. The declaration says which collections exist and which of their
fields the rules touch; the checks say what values those fields have to take. This module is
the first half, and it is deliberately dull: it reads names and types out of things the
merchant already wrote and invents nothing.

Four places a field comes from, and between them they cover every field any check reads:

- **The data source** names the identifier kinds its records are keyed and cross-referenced by.
- **A tool's behaviour** names the fields it filters on, writes to, and returns.
- **A rule's result reference** names a field of a result, which is a field of a record.
- **A channel** names the field somebody outside the merchant writes into.

A field nothing names is a field no check reads, so it is not emitted. That is not laziness:
a record padded with plausible-looking fields nothing asks about is a record whose contents
are somebody's invention, and the whole argument for a generated world is that every value in
it is there because a declared rule needed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentred.spec.models import (
    AgentSpec,
    CumulativeBound,
    EnumeratedBound,
    ImputedBound,
    MatchingBound,
    NumericBound,
    RelationalBound,
    ResultReference,
    ToolDeclaration,
    ToolShape,
)


class FieldKind(StrEnum):
    """What sort of value a field holds, as far as any check cares.

    Five, because five is what the checks distinguish. A limit compares numbers, an allowlist
    compares against a set, a scope check compares identifiers, an obligation is about text,
    and a repeated filing is a list that grows. Nothing else about a value changes what a
    check does with it.

    Attributes:
        IDENTIFIER: Names a record, here or in another collection.
        MATCHED: A value the agent has to carry across from one call to another. Not an
            identifier, though it took one to notice: a value compared between two calls is
            shared vocabulary rather than a name for a record, so minting a fresh one per
            record models it as something no two records could ever agree on.
        NUMBER: Compared against a limit.
        ENUM: One of a declared set of values.
        TEXT: Free text a person reads.
        LIST: A list a write appends to.
    """

    IDENTIFIER = "identifier"
    MATCHED = "matched"
    NUMBER = "number"
    ENUM = "enum"
    TEXT = "text"
    LIST = "list"


@dataclass(frozen=True)
class FieldShape:
    """One field of a record.

    Attributes:
        name: What records carry it under.
        kind: What sort of value it holds.
        values: For an `ENUM` or a `MATCHED` field, the values the declaration allows, where
            it says. Empty otherwise, which for a matched field is a stated limitation rather
            than an absence: see `Manifest.notes`.
        written_by: For a field a tool writes, the argument it is filled from. Used to work
            out which limit a field is the reachable side of.
    """

    name: str
    kind: FieldKind
    values: tuple[str, ...] = ()
    written_by: str = ""


@dataclass
class CollectionShape:
    """One collection of a generated world.

    Attributes:
        source: The declared data source name, which is also the collection's name. Identity
            for a generated world, because there is nothing else for it to be: nobody has
            written a shop whose internal names differ from what the agent asked for.
        key: The field records are keyed by, taken from the first key of a tool that fetches
            one record. A collection nothing fetches by name has no key and is reported as a
            gap rather than keyed by a guess.
        fields: Every field any check reads, in the sequence it was discovered.
        identifier_kinds: The declared identifier kinds records here carry.
    """

    source: str
    key: str = ""
    fields: dict[str, FieldShape] = field(default_factory=dict)
    identifier_kinds: tuple[str, ...] = ()

    def add(self, shape: FieldShape) -> None:
        """Record a field, keeping the more specific of two readings of the same name.

        A field named by two declarations is one field. `IDENTIFIER`, `MATCHED` and `ENUM` are
        more specific than `NUMBER`, which is more specific than `TEXT`, so a field a rule
        compares numerically is not downgraded to text by a channel that also writes into it.
        """
        existing = self.fields.get(shape.name)
        more_specific = existing is None or _RANK[shape.kind] > _RANK[existing.kind]
        fills_in_values = (
            existing is not None
            and existing.kind is shape.kind
            and bool(shape.values)
            and not existing.values
        )
        if more_specific or fills_in_values:
            self.fields[shape.name] = shape

    def of_kind(self, kind: FieldKind) -> tuple[FieldShape, ...]:
        """Every field of one kind, in discovery sequence."""
        return tuple(shape for shape in self.fields.values() if shape.kind is kind)


_RANK = {
    FieldKind.TEXT: 0,
    FieldKind.LIST: 1,
    FieldKind.NUMBER: 2,
    FieldKind.ENUM: 3,
    FieldKind.MATCHED: 4,
    FieldKind.IDENTIFIER: 5,
}
"""How specific each reading of a field is, most specific last."""


def _json_kind(tool: ToolDeclaration, argument: str) -> FieldKind:
    """The field kind a tool's own argument schema implies."""
    properties = tool.parameters.get("properties")
    declared = properties.get(argument, {}) if isinstance(properties, dict) else {}
    stated = declared.get("type") if isinstance(declared, dict) else None
    if stated in ("number", "integer"):
        return FieldKind.NUMBER
    if stated == "array":
        return FieldKind.LIST
    return FieldKind.TEXT


def _reads(tool: ToolDeclaration, source: str) -> bool:
    """Whether a tool touches one declared source."""
    return tool.behaviour is not None and tool.behaviour.source == source


def _references(spec: AgentSpec) -> tuple[tuple[ResultReference, FieldKind], ...]:
    """Every result field a rule reads, with what the rule does to it.

    A limit compares it numerically; a matching rule compares it against what the agent passed
    back. Nothing else in the policy vocabulary reads a result field, so those are the two.
    """
    found: list[tuple[ResultReference, FieldKind]] = []
    for bound in spec.policy.bounds:
        if isinstance(bound, RelationalBound):
            for reference in (bound.maximum_from, bound.minimum_from):
                if reference is not None:
                    found.append((reference, FieldKind.NUMBER))
        elif isinstance(bound, CumulativeBound) and bound.maximum_from is not None:
            found.append((bound.maximum_from, FieldKind.NUMBER))
        elif isinstance(bound, ImputedBound):
            found.append((bound.value_from, FieldKind.NUMBER))
        elif isinstance(bound, MatchingBound):
            found.append((bound.matches, FieldKind.MATCHED))
    for precondition in spec.policy.preconditions:
        if precondition.succeeds_when is not None:
            required = next(
                (tool for tool in spec.config.tools if tool.name == precondition.requires), None
            )
            if required is not None:
                found.append(
                    (
                        ResultReference(
                            tool=precondition.requires, field=precondition.succeeds_when.field
                        ),
                        FieldKind.ENUM,
                    )
                )
    return tuple(found)


def shapes_for(spec: AgentSpec) -> dict[str, CollectionShape]:
    """The shape of every collection a generated world for this agent has to hold.

    Args:
        spec: The validated spec. Its data sources are the collections, its tool behaviours
            and its rules are what say which fields those collections carry.

    Returns:
        One shape per declared data source, keyed by source name, in declaration sequence.
        A source no tool reaches still gets a shape, carrying only its identifier kinds: the
        agent declared it reads that source, and a world that omitted it would report every
        rule over it as never evaluated.
    """
    tools = {tool.name: tool for tool in spec.config.tools}
    shapes = {
        source.name: CollectionShape(source=source.name, identifier_kinds=source.identifier_kinds)
        for source in spec.config.data_sources
    }

    for shape in shapes.values():
        for kind in shape.identifier_kinds:
            shape.add(FieldShape(name=kind, kind=FieldKind.IDENTIFIER))

    for tool in spec.config.tools:
        behaviour = tool.behaviour
        if behaviour is None or behaviour.source not in shapes:
            continue
        shape = shapes[behaviour.source]
        if behaviour.shape is ToolShape.READ_ONE and not shape.key:
            shape.key = behaviour.keys[0]
        for key in behaviour.keys:
            shape.add(FieldShape(name=key, kind=FieldKind.IDENTIFIER))
        for name in behaviour.filters:
            shape.add(FieldShape(name=name, kind=_json_kind(tool, name)))
        for write in behaviour.writes:
            kind = (
                FieldKind.LIST
                if write.mode.value == "append"
                else (
                    FieldKind.NUMBER
                    if write.mode.value == "add"
                    else _json_kind(tool, write.argument)
                    if write.argument
                    else FieldKind.TEXT
                )
            )
            shape.add(FieldShape(name=write.field, kind=kind, written_by=write.argument))
        for name in behaviour.result_fields:
            shape.add(FieldShape(name=name, kind=FieldKind.TEXT))

    for reference, kind in _references(spec):
        tool = tools.get(reference.tool)
        if tool is None or tool.behaviour is None or tool.behaviour.source not in shapes:
            continue
        shape = shapes[tool.behaviour.source]
        values: tuple[str, ...] = ()
        if kind is FieldKind.MATCHED:
            # Whatever the declaration says this value may be, if it says anything. A value
            # the agent must carry across is one it also has an opinion about, and the closer
            # the record's value is to something it recognises the less likely it is to
            # substitute one of its own.
            #
            # The two declarations meet at the argument rather than at the field. A matching
            # rule says an argument must agree with a field of a result, and an allowlist
            # constrains that same argument; the field the value is read off is named by
            # whoever wrote the source tool and need not be called the same thing.
            arguments = {
                other.argument
                for other in spec.policy.bounds
                if isinstance(other, MatchingBound) and other.matches == reference
            }
            values = next(
                (
                    tuple(other.allowed_values)
                    for other in spec.policy.bounds
                    if isinstance(other, EnumeratedBound)
                    and other.argument in arguments
                    and other.allowed_values
                ),
                (),
            )
        if kind is FieldKind.ENUM:
            for precondition in spec.policy.preconditions:
                condition = precondition.succeeds_when
                if precondition.requires == reference.tool and condition is not None:
                    values = condition.equals_any or (
                        (condition.equals,) if condition.equals else ()
                    )
        shape.add(FieldShape(name=reference.field, kind=kind, values=values))

    for channel in spec.config.channels:
        if channel.data_source not in shapes:
            continue
        shape = shapes[channel.data_source]
        shape.add(FieldShape(name=channel.record_path, kind=FieldKind.TEXT))
        shape.add(FieldShape(name=channel.record_key, kind=FieldKind.IDENTIFIER))
        # A channel names the records of its own source, and that is a second place a key
        # comes from. A source nothing fetches one record from otherwise has no key, so it
        # holds no records at all, and a channel aimed at it plants into a collection that is
        # empty: the write is refused, every attack down that channel fails before the agent
        # is reached, and what the run reports is a harness that could not deliver rather than
        # an agent that did or did not hold. That is how the first run against a generated
        # shop spent its whole suite.
        if not shape.key:
            shape.key = channel.record_key

    return shapes


def field_named(shapes: dict[str, CollectionShape], name: str) -> tuple[str, FieldShape] | None:
    """The first collection carrying a field of that name, and the field.

    How an argument is connected to the record it is filled from, and it is a heuristic rather
    than a fact: a merchant's tool argument and the record field behind it carry the same name
    far more often than not, and when they do not, the rule is reported as a gap rather than
    quietly left unreachable. Naming them the same thing is then a line the merchant can add,
    which is the shape rule 10 of this project asks a remediation to have.
    """
    for shape in shapes.values():
        found = shape.fields.get(name)
        if found is not None:
            return shape.source, found
    return None


def reached_by(spec: AgentSpec, tool_name: str) -> str:
    """The declared source a tool touches, or an empty string if it touches none."""
    for tool in spec.config.tools:
        if tool.name == tool_name and tool.behaviour is not None:
            return tool.behaviour.source
    return ""


def bound_argument(bound: object) -> str:
    """The argument a bound constrains, or an empty string for one that constrains none."""
    if isinstance(
        bound, (NumericBound, EnumeratedBound, RelationalBound, CumulativeBound, MatchingBound)
    ):
        return bound.argument
    return ""
