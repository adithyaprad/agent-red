"""Reading the limits a no-code builder stored when an operator configured their agent.

This is where the money rules are, and the reason is structural rather than lucky. A
platform that lets somebody configure an agent without writing code has to ask them for the
limits, and has to store the answers somewhere it can enforce them. The skill is generic:
one dispute responder, the same prompt and the same steps for everyone who installs it. What
differs between one operator's instance and another's is the ceiling on a refund, the point
above which a person approves, and which actions are switched off. That difference is exactly
what a per-instance configuration is for, and it is exactly what the policy half needs.

So the rules hardest to recover from a hand-built agent are the ones best recorded by a
no-code one. A wizard asking "what is the largest refund this agent may issue without
approval" has produced a structured policy statement as a side effect of being a wizard, and
the value it stored is `declared` in the strongest sense available: an operator typed it, the
platform enforces it, and re-reading gives the same answer.

**What a wizard cannot have asked, and what this module refuses to invent.** Every limit here
is a ceiling or a floor on one argument of one call, because that is the only shape a form
field has. The rules that catch a patient attacker are not that shape. A limit on everything
paid back against one order added up is not a field on a form, and neither is one whose value
is read from the record being acted on rather than passed in as an argument. Three refunds of
forty thousand each pass a fifty thousand per-call ceiling and together return more than the
customer ever paid. An instance configuration that stops at per-call ceilings is therefore not
a complete policy, and this module says so in `notes` rather than letting the emitted policy
imply that per-call ceilings are all there is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentred.ingest.package import (
    AgentPackage,
    Evidence,
    Observation,
    Origin,
    RuleFacts,
)
from agentred.spec.models import (
    Consequence,
    DataScope,
    DataSource,
    NumericBound,
    Provenance,
)

ADAPTER = "instance"
"""The adapter name recorded on every piece of evidence this module produces."""

SHAPES_NO_FORM_FIELD_HOLDS = (
    "a limit on everything one action does to one record added up, which a per-call ceiling "
    "cannot see: three calls each inside the ceiling can exceed it together",
    "a limit whose value is read from the record being acted on rather than passed as an "
    "argument, which leaves the most expensive action looking free",
    "a step that must run before another step, which an instance configuration has no field "
    "for and a workflow definition states directly",
)
"""Rule shapes an instance configuration structurally cannot hold, named on every read.

Not a list of things this reader failed at. A list of things no form field can express, so
that a policy recovered entirely from a wizard is never mistaken for a complete one. Silence
here is the flattering reading: an agent bounded only by per-call ceilings looks bounded.
"""

SCOPE_QUESTION = (
    "which data sources may one session touch, and which identifier kinds name the person it "
    "is about? The instance configuration lists what the agent can reach, which is a larger "
    "set than what one conversation is allowed to be about."
)
"""What is asked when an instance grants data access but says nothing about session scope.

The distinction is the whole of the scope check. An agent permitted to read every order is
correctly permitted to read every order; it is still leaking when it puts one customer's
order into another customer's reply. A configuration that only lists reachable sources has
not answered the second question, and defaulting it to the first makes every scope violation
unobservable.
"""


class InstanceReadError(Exception):
    """An instance configuration could not be read.

    Raised rather than letting a parse error escape, so that a caller reading a manifest and
    three connectors can say which file was the problem.
    """


def _limit_rule(entry: dict[str, Any], locator: str, index: int) -> RuleFacts:
    """One stored limit, as a bound.

    Args:
        entry: The limit as the instance configuration recorded it.
        locator: The file it was read from.
        index: Position in the list, used to name a limit the operator did not name.

    Returns:
        A `NumericBound` marked declared, carrying the operator's own label as its
        description so the rule reads on a report in the words they configured it in.

    Raises:
        InstanceReadError: If the entry names no action or argument, or sets no limit at
            all. A limit entry with no limit is a form somebody half filled in, and reading
            it as an unbounded argument would turn an operator's abandoned intention into a
            declared permission.
    """
    tool = entry.get("action") or entry.get("tool")
    argument = entry.get("field") or entry.get("argument")
    maximum = entry.get("max", entry.get("maximum"))
    minimum = entry.get("min", entry.get("minimum"))
    if not tool or not argument:
        raise InstanceReadError(f"{locator}: limit {index} names no action and argument")
    if maximum is None and minimum is None:
        raise InstanceReadError(
            f"{locator}: limit {index} on {tool}.{argument} sets neither a maximum nor a "
            f"minimum. An empty limit is an unfinished form, not an unbounded argument."
        )
    name = str(entry.get("name") or f"{tool}_{argument}_limit")
    return RuleFacts(
        rule=NumericBound(
            kind="numeric",
            name=name,
            tool=str(tool),
            argument=str(argument),
            maximum=maximum,
            minimum=minimum,
            provenance=Provenance.DECLARED,
            description=str(entry.get("label") or entry.get("description") or ""),
        ),
        origin=Origin.DECLARED,
        evidence=Evidence(adapter=ADAPTER, locator=f"{locator}#limits[{index}]"),
    )


def _consequence(entry: dict[str, Any], name: str, locator: str) -> Observation[Consequence]:
    """What an action costs, when the instance configuration says.

    Args:
        entry: The action as the instance configuration recorded it.
        name: The action name, for the question if there is no answer.
        locator: The file it was read from.

    Returns:
        A `DECLARED` observation when the configuration records what the action costs, and
        an undetermined one otherwise. A builder that groups actions into ones that move
        money and ones that do not has already answered this; one that does not, has not,
        and the difference is not something to split.
    """
    from agentred.ingest.adapters.mcp import CONSEQUENCE_QUESTION

    stated = entry.get("consequence") or entry.get("cost")
    evidence = Evidence(adapter=ADAPTER, locator=locator)
    if stated is None:
        return Observation[Consequence](
            value=None,
            origin=Origin.UNDETERMINED,
            evidence=evidence,
            question=CONSEQUENCE_QUESTION.format(name=name),
        )
    try:
        return Observation[Consequence](
            value=Consequence(str(stated)), origin=Origin.DECLARED, evidence=evidence
        )
    except ValueError as error:
        raise InstanceReadError(
            f"{locator}: action {name} records its cost as {stated!r}, which is not one of "
            f"{', '.join(c.value for c in Consequence)}"
        ) from error


def _source(entry: Any) -> DataSource:
    """One store the agent was granted, however the builder recorded it.

    A data access screen that only lists names is the common case and is read as a name. One
    that also says what each store is keyed by has answered more, and the difference matters:
    a channel plants into a record, and a source that cannot say which identifier names its
    records is a source nothing can be planted into.

    Args:
        entry: A source as recorded, either a bare name or a mapping carrying one.

    Returns:
        The declared source.

    Raises:
        InstanceReadError: If a mapping names no source.
    """
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("source")
        if not name:
            raise InstanceReadError(f"a granted source names nothing: {entry!r}")
        kinds = tuple(str(kind) for kind in (entry.get("identifiers") or []) if str(kind))
        return DataSource(
            name=str(name),
            description=str(entry.get("label") or entry.get("description") or ""),
            identifier_kinds=kinds,
        )
    return DataSource(name=str(entry))


def read_instance(path: Path | str) -> AgentPackage:
    """Read one agent instance's stored configuration.

    Args:
        path: Path to the instance configuration YAML.

    Returns:
        An `AgentPackage` holding the limits as bounds, the granted data sources, and the
        session scope when the configuration states one. Its `notes` name the rule shapes no
        form field can hold, on every read, whether or not any limit was found.

    Raises:
        InstanceReadError: If the file is missing, is not a YAML mapping, names no agent, or
            holds a limit that sets no limit.
    """
    locator = str(path)
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InstanceReadError(f"no instance configuration at {locator}") from error
    except yaml.YAMLError as error:
        raise InstanceReadError(f"{locator} is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise InstanceReadError(f"{locator} is not a YAML mapping")

    agent_id = raw.get("instance_id") or raw.get("agent_id")
    if not agent_id:
        raise InstanceReadError(f"{locator} names no agent: expected instance_id or agent_id")

    rules = tuple(
        _limit_rule(entry, locator, index)
        for index, entry in enumerate(raw.get("limits") or [])
        if isinstance(entry, dict)
    )
    access = raw.get("data_access") or {}
    sources = tuple(_source(entry) for entry in (access.get("sources") or []) if entry)
    return AgentPackage(
        agent_id=str(agent_id),
        rules=rules,
        data_sources=sources,
        data_scope=_scope(access, locator, sources),
        sources=(ADAPTER,),
        notes=SHAPES_NO_FORM_FIELD_HOLDS,
    )


def _scope(
    access: dict[str, Any], locator: str, sources: tuple[DataSource, ...]
) -> Observation[DataScope]:
    """What one session may touch, when the configuration distinguishes it from what is reachable.

    Args:
        access: The `data_access` block as recorded.
        locator: The file it was read from.
        sources: The sources the instance grants, used only to say whether there is anything
            to scope at all.

    Returns:
        A declared scope when the configuration states the identifier kinds a session is
        about, and an undetermined one when it grants access and stops there.
    """
    evidence = Evidence(adapter=ADAPTER, locator=f"{locator}#data_access")
    kinds = tuple(str(kind) for kind in (access.get("identifiers") or []) if str(kind))
    if not kinds:
        return Observation[DataScope](
            value=None, origin=Origin.UNDETERMINED, evidence=evidence, question=SCOPE_QUESTION
        )
    return Observation[DataScope](
        value=DataScope(
            sources=tuple(source.name for source in sources),
            subject_identifier_kinds=kinds,
            provenance=Provenance.DECLARED,
            description=str(access.get("label") or ""),
        ),
        origin=Origin.DECLARED,
        evidence=evidence,
    )
