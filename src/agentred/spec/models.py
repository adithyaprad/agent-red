"""The two objects agent-red reads before it attacks anything.

`AgentConfig` is capability: what the agent is and what it can do. `AgentPolicy` is
authorisation: what it may and must do. Every system that grants power to an actor
separates the two, and here the split decides what remedies exist. A structured bound can
be tightened, which makes a violation unreachable; a limit written as English inside a
system prompt can only be reworded, which makes a violation less likely.

`AgentSpec` pairs them and is the only object the rest of the tree should accept. Its
constructor rejects a policy that does not describe the config it is paired with: a bound
on a tool that does not exist, a precondition on an argument that was never declared, a
data scope naming a source the agent cannot reach. Those are the errors that would
otherwise surface as a detector that silently never fires, which is worse than a crash.

These models validate shape and internal consistency. They do not fetch anything, do not
read files (see `loader.py`), and do not know what a merchant sells.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Consequence(StrEnum):
    """What calling a tool costs the merchant if the agent is talked into calling it.

    This is the field that turns a tool list into a set of stakes, so it is required on
    every declared tool rather than defaulted. `inert` is a claim, not an absence.

    Attributes:
        MONEY: Moves money directly (refunds, discounts, credits, price overrides).
        OBLIGATION: Commits the merchant to a future cost (orders, delivery promises,
            reservations, warranty extensions).
        DISCLOSURE: Reveals data (order details, customer records, internal notes).
        INERT: No cost if called wrongly. Reads public information or does nothing.
    """

    MONEY = "money"
    OBLIGATION = "obligation"
    DISCLOSURE = "disclosure"
    INERT = "inert"

    @property
    def is_consequential(self) -> bool:
        """Whether a wrong call costs the merchant something."""
        return self is not Consequence.INERT


class Provenance(StrEnum):
    """Where a policy statement came from, and therefore how much to trust it.

    Carried on every bound, precondition and data scope so that degraded mode is visible
    in the output rather than hidden in it. A scorecard mixing the two without saying so
    reports a confidence it has not earned.

    Attributes:
        DECLARED: Read from a structured policy the merchant wrote.
        INFERRED: Extracted from prose by a model, and may be wrong in either direction.
    """

    DECLARED = "declared"
    INFERRED = "inferred"


class ToolDeclaration(BaseModel):
    """One tool the agent can call, with the shape of its arguments.

    `parameters` is a JSON Schema object exactly as the model receives it. It is stored
    verbatim rather than re-modelled, because the bound validator needs to answer only one
    question of it (does this argument exist) and re-modelling would reject schemas the
    agent platform accepts.

    Attributes:
        name: The tool name the model calls.
        description: What the tool does, as the agent sees it.
        parameters: JSON Schema for the arguments, `{"type": "object", ...}`.
        consequence: What a wrong call costs. See `Consequence`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    consequence: Consequence

    @property
    def argument_names(self) -> frozenset[str]:
        """The top-level argument names declared in the JSON Schema.

        Nested properties are not walked: a bound constrains an argument the agent passes,
        and an argument the agent passes is top-level. Returns an empty set for a schema
        that declares no properties.
        """
        properties = self.parameters.get("properties")
        if not isinstance(properties, dict):
            return frozenset()
        return frozenset(str(key) for key in properties)


class DataSource(BaseModel):
    """A store of data the agent can reach during a session.

    Attributes:
        name: Identifier used by `DataScope` and by scope detectors.
        description: What the source holds.
        identifier_kinds: Names of the identifier types records here carry (for example
            `order_id`, `email`). Scope detectors use these to recognise a leaked value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(default="")
    identifier_kinds: tuple[str, ...] = ()


class NumericBound(BaseModel):
    """A numeric ceiling or floor on one argument of one tool.

    At least one of `maximum` and `minimum` must be set. Both may be, which expresses a
    closed range.

    Attributes:
        kind: Discriminator, always `numeric`.
        name: Stable identifier for this bound. Appears in verdicts and on the scorecard.
        tool: The tool this constrains.
        argument: The argument of that tool this constrains.
        maximum: Largest permitted value, inclusive.
        minimum: Smallest permitted value, inclusive.
        provenance: Declared by the merchant, or inferred from prose.
        description: Why the bound exists, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["numeric"] = "numeric"
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    maximum: float | None = None
    minimum: float | None = None
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @model_validator(mode="after")
    def _at_least_one_limit(self) -> NumericBound:
        if self.maximum is None and self.minimum is None:
            raise ValueError(f"numeric bound {self.name!r} sets neither maximum nor minimum")
        if self.maximum is not None and self.minimum is not None and self.maximum < self.minimum:
            raise ValueError(
                f"numeric bound {self.name!r} has maximum {self.maximum} below minimum "
                f"{self.minimum}"
            )
        return self

    def permits(self, value: float) -> bool:
        """Whether `value` is inside this bound."""
        if self.maximum is not None and value > self.maximum:
            return False
        return not (self.minimum is not None and value < self.minimum)


class EnumeratedBound(BaseModel):
    """A closed set of permitted values for one argument of one tool.

    Attributes:
        kind: Discriminator, always `enumerated`.
        name: Stable identifier for this bound.
        tool: The tool this constrains.
        argument: The argument of that tool this constrains.
        allowed_values: The permitted values. Compared as strings, because a tool argument
            arrives as JSON and `"2"` and `2` must not be treated as different answers.
        provenance: Declared by the merchant, or inferred from prose.
        description: Why the bound exists, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["enumerated"] = "enumerated"
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    allowed_values: tuple[str, ...] = Field(min_length=1)
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @field_validator("allowed_values", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return value

    def permits(self, value: object) -> bool:
        """Whether `value` is one of the permitted values, compared as a string."""
        return str(value) in self.allowed_values


class ResultReference(BaseModel):
    """A pointer to one field of one tool's result.

    The field path is dotted and may index a list, so `lines.0.price` reads the first line's
    price. A path is resolved against a decoded JSON result; nothing here does the resolving,
    because these models never touch a transcript.

    Attributes:
        tool: The tool whose result holds the value.
        field: Dotted path into that result, for example `total` or `lines.0.price`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str = Field(min_length=1)
    field: str = Field(min_length=1)

    def __str__(self) -> str:
        """The reference as `tool.field`, for verdicts and error messages."""
        return f"{self.tool}.{self.field}"

    def resolve(self, result: object) -> float | None:
        """Read this reference's value out of a decoded tool result.

        Args:
            result: The decoded result of a call to `self.tool`.

        Returns:
            The value as a float, or `None` if the path does not exist or the value at the
            end of it is not a number. `None` means "this bound has nothing to compare
            against", never "the bound was satisfied".
        """
        current: Any = result
        for step in self.field.split("."):
            if isinstance(current, dict):
                if step not in current:
                    return None
                current = current[step]
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(step)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None
        return float(current)


class RelationalBound(BaseModel):
    """A limit on one argument whose value comes from something the agent already read.

    Some limits are not constants. "Do not give back more than was taken" bounds an argument
    by a figure the agent fetched earlier in the same conversation, so the ceiling is
    different for every conversation and cannot be written into the policy as a number.

    This is deliberately not a judgement call. The figure is in the tool-call log next to the
    argument that exceeded it, so a detector asserts it and no model is asked. Expressing it
    any other way would move a decidable question onto the LLM judge, which the project treats
    as a regression.

    The shape stays domain-independent: an argument, a reference into an earlier result, and a
    comparison. Nothing here knows what is being bounded or why.

    At least one of `maximum_from` and `minimum_from` must be set.

    Attributes:
        kind: Discriminator, always `relational`.
        name: Stable identifier for this bound. Appears in verdicts and on the scorecard.
        tool: The tool this constrains.
        argument: The argument of that tool this constrains.
        maximum_from: Reference whose value is the largest permitted value, inclusive.
        minimum_from: Reference whose value is the smallest permitted value, inclusive.
        provenance: Declared by the merchant, or inferred from prose.
        description: Why the bound exists, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["relational"] = "relational"
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    maximum_from: ResultReference | None = None
    minimum_from: ResultReference | None = None
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @model_validator(mode="after")
    def _at_least_one_limit(self) -> RelationalBound:
        if self.maximum_from is None and self.minimum_from is None:
            raise ValueError(
                f"relational bound {self.name!r} sets neither maximum_from nor minimum_from"
            )
        return self

    @property
    def source_tools(self) -> tuple[str, ...]:
        """Every tool this bound has to read a result from, without duplicates."""
        names: list[str] = []
        for reference in (self.maximum_from, self.minimum_from):
            if reference is not None and reference.tool not in names:
                names.append(reference.tool)
        return tuple(names)

    def permits(self, value: float, *, maximum: float | None, minimum: float | None) -> bool:
        """Whether `value` is inside this bound, given limits already resolved from the log.

        The caller resolves the references, because only the caller has the transcript. An
        unresolved limit is passed as `None` and does not constrain: a bound that could not
        find its figure has not been satisfied, it has not been evaluated, and the detector
        is responsible for reporting that difference rather than hiding it here.
        """
        if maximum is not None and value > maximum:
            return False
        return not (minimum is not None and value < minimum)


Bound = Annotated[NumericBound | EnumeratedBound | RelationalBound, Field(discriminator="kind")]
"""Any kind of limit on a tool argument."""


class Precondition(BaseModel):
    """A tool that must have succeeded earlier in the conversation before another is called.

    This is the shape of most real commerce failures: the refund itself is permitted, and
    the violation is that nothing was verified first. Expressing it as a required prior
    call makes it observable in the tool-call log rather than a matter of judgement.

    Attributes:
        name: Stable identifier for this precondition.
        tool: The consequential tool that is gated.
        requires: The tool that must have returned successfully earlier in the same
            conversation.
        provenance: Declared by the merchant, or inferred from prose.
        description: What the precondition establishes, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    requires: str = Field(min_length=1)
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @model_validator(mode="after")
    def _not_self_gating(self) -> Precondition:
        if self.tool == self.requires:
            raise ValueError(f"precondition {self.name!r} requires {self.tool!r} to precede itself")
        return self


class DataScope(BaseModel):
    """What a single session may reach, and what it may say.

    A session is scoped to a subject (in practice one customer). `sources` names which
    declared data sources are in play at all; `subject_identifier_kinds` names the
    identifiers that bind a record to this session's subject. A record from a permitted
    source that belongs to a different subject is still a violation, which is why both
    fields exist rather than only the first.

    Attributes:
        sources: Names of `DataSource` entries this session may read.
        subject_identifier_kinds: Identifier kinds that scope a record to the session
            subject, for example `customer_id`.
        provenance: Declared by the merchant, or inferred from prose.
        description: What the scope means, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: tuple[str, ...] = ()
    subject_identifier_kinds: tuple[str, ...] = ()
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")


class AgentConfig(BaseModel):
    """What the agent is and can do.

    `tool_version` is deliberately not a field. It is derived from the declared tools, so
    a tool set that changes cannot keep an old version string and quietly invalidate every
    scorecard that cites it.

    Attributes:
        agent_id: Stable identifier for the agent across versions.
        version: Version of this config object. Part of the scorecard validity tuple.
        model: The model the agent runs on, for example `claude-sonnet-5`. Part of the
            validity tuple, because a model upgrade is a reason to re-test.
        instructions: The system prompt, verbatim.
        tools: Every tool the agent can call.
        data_sources: Every store the agent can reach.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    instructions: str = Field(default="")
    tools: tuple[ToolDeclaration, ...] = ()
    data_sources: tuple[DataSource, ...] = ()

    @model_validator(mode="after")
    def _unique_names(self) -> AgentConfig:
        _reject_duplicates((tool.name for tool in self.tools), "tool")
        _reject_duplicates((source.name for source in self.data_sources), "data source")
        return self

    @property
    def tools_by_name(self) -> dict[str, ToolDeclaration]:
        """Declared tools keyed by name."""
        return {tool.name: tool for tool in self.tools}

    @property
    def consequential_tools(self) -> tuple[ToolDeclaration, ...]:
        """Tools whose wrong call costs the merchant something.

        This is the set attack generation aims at, and the set a missing precondition is
        worth reporting on.
        """
        return tuple(tool for tool in self.tools if tool.consequence.is_consequential)

    @property
    def tool_version(self) -> str:
        """A digest of the declared tool set, stable across reorderings.

        Derived rather than declared so that it cannot be stale. Two configs with the same
        tools in a different order produce the same digest; changing a tool's schema or its
        consequence produces a different one.
        """
        canonical = sorted(
            json.dumps(tool.model_dump(mode="json"), sort_keys=True) for tool in self.tools
        )
        digest = hashlib.sha256("\n".join(canonical).encode()).hexdigest()
        return f"sha256:{digest[:12]}"


class AgentPolicy(BaseModel):
    """What the agent may and must do.

    Kept separate from `AgentConfig` because the two have different remedies and different
    owners. Validation against a config happens in `AgentSpec`, not here, so a policy can
    be parsed and inspected on its own.

    Attributes:
        agent_id: Must match the config it is paired with.
        version: Version of this policy object. Part of the scorecard validity tuple.
        bounds: Limits on tool arguments.
        preconditions: Tools that must precede consequential tools.
        data_scope: What one session may reach.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    bounds: tuple[Bound, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    data_scope: DataScope = DataScope()

    @model_validator(mode="after")
    def _unique_names(self) -> AgentPolicy:
        _reject_duplicates((bound.name for bound in self.bounds), "bound")
        _reject_duplicates((pre.name for pre in self.preconditions), "precondition")
        return self

    @property
    def statements(self) -> tuple[NumericBound | EnumeratedBound | Precondition | DataScope, ...]:
        """Every policy statement, of every kind, for provenance reporting."""
        return (*self.bounds, *self.preconditions, self.data_scope)

    @property
    def is_fully_declared(self) -> bool:
        """Whether every statement came from structured policy rather than from prose.

        False means the scorecard must label itself as partly inferred.
        """
        return all(statement.provenance is Provenance.DECLARED for statement in self.statements)


class VersionTuple(BaseModel):
    """The four versions a scorecard is valid for.

    Change any element and the agent is untested again. `store/repo.py` refuses to write a
    scorecard without one.

    Attributes:
        config_version: `AgentConfig.version`.
        policy_version: `AgentPolicy.version`.
        model_version: `AgentConfig.model`.
        tool_version: `AgentConfig.tool_version`, derived from the tool declarations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_version: str
    policy_version: str
    model_version: str
    tool_version: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        """The four versions in declaration order."""
        return (self.config_version, self.policy_version, self.model_version, self.tool_version)

    def __str__(self) -> str:
        """A single-line form for logs and scorecard headers."""
        return (
            f"config={self.config_version} policy={self.policy_version} "
            f"model={self.model_version} tools={self.tool_version}"
        )


class AgentSpec(BaseModel):
    """A config and the policy that authorises it, checked against each other.

    This is the object the rest of the tree accepts. Construction fails if the policy
    describes something the config does not have, because a bound naming a tool that does
    not exist produces a detector that never fires, and a detector that never fires reads
    as a passing agent.

    Attributes:
        config: What the agent is and can do.
        policy: What it may and must do.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: AgentConfig
    policy: AgentPolicy

    @model_validator(mode="after")
    def _policy_describes_config(self) -> AgentSpec:
        if self.policy.agent_id != self.config.agent_id:
            raise ValueError(
                f"policy is for agent {self.policy.agent_id!r} but config is for "
                f"{self.config.agent_id!r}"
            )

        tools = self.config.tools_by_name
        for bound in self.policy.bounds:
            tool = tools.get(bound.tool)
            if tool is None:
                raise ValueError(
                    f"bound {bound.name!r} constrains tool {bound.tool!r}, which is not declared"
                )
            if bound.argument not in tool.argument_names:
                raise ValueError(
                    f"bound {bound.name!r} constrains argument {bound.argument!r} of tool "
                    f"{bound.tool!r}, which declares no such argument"
                )
            if isinstance(bound, RelationalBound):
                for source in bound.source_tools:
                    if source not in tools:
                        raise ValueError(
                            f"bound {bound.name!r} reads its limit from tool {source!r}, "
                            f"which is not declared"
                        )
                    if source == bound.tool:
                        raise ValueError(
                            f"bound {bound.name!r} reads its limit from {source!r}, the tool "
                            f"it constrains. A call cannot be bounded by its own result."
                        )

        for precondition in self.policy.preconditions:
            for field, name in (("tool", precondition.tool), ("requires", precondition.requires)):
                if name not in tools:
                    raise ValueError(
                        f"precondition {precondition.name!r} names {name!r} as its {field}, "
                        f"which is not a declared tool"
                    )

        declared_sources = {source.name for source in self.config.data_sources}
        for source in self.policy.data_scope.sources:
            if source not in declared_sources:
                raise ValueError(
                    f"data scope permits source {source!r}, which the agent cannot reach"
                )

        return self

    @property
    def version_tuple(self) -> VersionTuple:
        """The four versions this spec's results would be valid for."""
        return VersionTuple(
            config_version=self.config.version,
            policy_version=self.policy.version,
            model_version=self.config.model,
            tool_version=self.config.tool_version,
        )

    def bounds_for(self, tool: str) -> tuple[NumericBound | EnumeratedBound | RelationalBound, ...]:
        """Every bound constraining `tool`, in policy order."""
        return tuple(bound for bound in self.policy.bounds if bound.tool == tool)

    def preconditions_for(self, tool: str) -> tuple[Precondition, ...]:
        """Every precondition gating `tool`, in policy order."""
        return tuple(pre for pre in self.policy.preconditions if pre.tool == tool)

    def ungated_consequential_tools(self) -> tuple[ToolDeclaration, ...]:
        """Consequential tools with no declared precondition and no declared bound.

        These are where the only thing standing between a customer and the merchant's money
        is the wording of a system prompt. `patch/` must answer these with a `permission`
        remedy and must not offer an `instruction` remedy as an equivalent.
        """
        return tuple(
            tool
            for tool in self.config.consequential_tools
            if not self.bounds_for(tool.name) and not self.preconditions_for(tool.name)
        )


def _reject_duplicates(names: Iterable[str], label: str) -> None:
    """Raise if any name repeats.

    Args:
        names: An iterable of names.
        label: What the names are, for the error message.

    Raises:
        ValueError: On the first name seen twice.
    """
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate {label} name {name!r}")
        seen.add(name)
