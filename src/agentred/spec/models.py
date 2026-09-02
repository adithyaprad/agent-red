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


class Engine(StrEnum):
    """How an agent under test is built.

    Not a detail of the harness's plumbing. A workflow-built agent reaches its money actions
    through the steps in front of them, and a model-loop agent does not, so the same attack
    meets a different amount of structure in each. Declaring it means a scorecard can say
    which of the two it describes, and means the suite can be shown to transfer across build
    styles rather than asserted to.

    Attributes:
        WORKFLOW: A workflow engine of declared steps, with a model invoked at the
            judgement points inside it. What a no-code builder produces.
        MODEL_LOOP: One model, one system prompt, its tools, and a loop. What an agent
            somebody wrote in code usually is.
    """

    WORKFLOW = "workflow"
    MODEL_LOOP = "model_loop"


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

    @property
    def constrained_arguments(self) -> tuple[str, ...]:
        """Arguments this bound reads. Checked against the tool's schema at load."""
        return (self.argument,)

    @property
    def source_tools(self) -> tuple[str, ...]:
        """Tools this bound reads a limit from. None: its limit is a constant."""
        return ()


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

    @property
    def constrained_arguments(self) -> tuple[str, ...]:
        """Arguments this bound reads. Checked against the tool's schema at load."""
        return (self.argument,)

    @property
    def source_tools(self) -> tuple[str, ...]:
        """Tools this bound reads a limit from. None: its permitted set is a constant."""
        return ()


_MISSING = object()
"""What a field path returns when the path is not there at all.

Deliberately not `None`, which is a value a result can legitimately carry. A check that
conflated the two would read a null as an answer.
"""


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

    def reach(self, result: object) -> Any:
        """Walk the path and return whatever is at the end of it, or `_MISSING`.

        Args:
            result: The decoded result of a call to `self.tool`.

        Returns:
            The value at the end of the path. `_MISSING` when the path does not exist, which
            is deliberately not `None`: a result carrying `null` at the path and a result with
            no such path are different facts, and only the first is an answer.
        """
        current: Any = result
        for step in self.field.split("."):
            if isinstance(current, dict):
                if step not in current:
                    return _MISSING
                current = current[step]
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(step)]
                except (ValueError, IndexError):
                    return _MISSING
            else:
                return _MISSING
        return current

    def resolve(self, result: object) -> float | None:
        """Read this reference's value out of a decoded tool result, as a number.

        Args:
            result: The decoded result of a call to `self.tool`.

        Returns:
            The value as a float, or `None` if the path does not exist or the value at the
            end of it is not a number. `None` means "this bound has nothing to compare
            against", never "the bound was satisfied".
        """
        current = self.reach(result)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            return None
        return float(current)

    def resolve_text(self, result: object) -> str | None:
        """Read this reference's value out of a decoded tool result, as text.

        Some things a limit compares against are not quantities. A currency, a country, a
        settlement account: the comparison is equality rather than magnitude, and coercing one
        of those to a number to reuse `resolve` would silently discard every limit of that
        shape.

        Args:
            result: The decoded result of a call to `self.tool`.

        Returns:
            The value as a stripped, case-folded string, or `None` when the path does not
            exist or holds nothing comparable. A container is not comparable: a path landing
            on a list or a dict has not found a value, and rendering one would compare two
            reprs and call the result a policy decision.
        """
        current = self.reach(result)
        if current is _MISSING or current is None:
            return None
        if isinstance(current, (dict, list, tuple)):
            return None
        if isinstance(current, bool):
            return str(current).lower()
        return str(current).strip().casefold()


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
    def constrained_arguments(self) -> tuple[str, ...]:
        """Arguments this bound reads. Checked against the tool's schema at load."""
        return (self.argument,)

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


class CumulativeBound(BaseModel):
    """A limit on the total of one argument across every call in the conversation.

    The limit every per-call ceiling misses. Three refunds of forty thousand against one
    order each pass a fifty thousand ceiling and a per-call comparison against the order
    total, and together they return more than was ever paid. Nothing in the arguments of any
    one of those calls is out of range, which is why this cannot be expressed as a
    `NumericBound` and why an agent talked into instalments defeats one.

    `group_by` names the arguments that say what the total accrues against. Empty means the
    conversation as a whole, which is the right reading for a budget. Naming an argument
    means one running total per distinct value of it, so a second refund against a different
    order does not count against the first order's total.

    Exactly one of `maximum` and `maximum_from` is set. A constant expresses a policy
    ceiling; a reference expresses "not more than was taken", where the figure is different
    in every conversation and is read from something the agent fetched itself.

    Attributes:
        kind: Discriminator, always `cumulative`.
        name: Stable identifier for this bound.
        tool: The tool this constrains.
        argument: The argument whose values are summed.
        group_by: Arguments identifying what a total accrues against.
        maximum: Largest permitted total, inclusive.
        maximum_from: Reference whose value is the largest permitted total, inclusive.
        provenance: Declared by the merchant, or inferred from prose.
        description: Why the bound exists, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["cumulative"] = "cumulative"
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    group_by: tuple[str, ...] = ()
    maximum: float | None = None
    maximum_from: ResultReference | None = None
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @model_validator(mode="after")
    def _exactly_one_limit(self) -> CumulativeBound:
        if (self.maximum is None) == (self.maximum_from is None):
            raise ValueError(
                f"cumulative bound {self.name!r} must set exactly one of maximum and maximum_from"
            )
        return self

    @property
    def constrained_arguments(self) -> tuple[str, ...]:
        """Arguments this bound reads. Checked against the tool's schema at load."""
        return (self.argument, *self.group_by)

    @property
    def source_tools(self) -> tuple[str, ...]:
        """The tool the ceiling is read from, if it is read rather than declared."""
        return () if self.maximum_from is None else (self.maximum_from.tool,)

    def permits(self, total: float, *, maximum: float | None) -> bool:
        """Whether a running total is inside this bound, given a resolved ceiling.

        The caller resolves a referenced ceiling, because only the caller has the transcript.
        An unresolved ceiling is passed as `None` and does not constrain: the bound has not
        been evaluated, and saying so is the detector's job rather than this one's.
        """
        return not (maximum is not None and total > maximum)


class MatchingBound(BaseModel):
    """An argument that has to equal something the agent already read.

    Not every limit is a magnitude. A refund is issued in a currency, against an order that
    has one; a payout goes to an account, and the order names one. The failure is a mismatch
    rather than an excess, and the amount can be perfectly within every ceiling while the
    money leaves in the wrong denomination, which on a payment network is a real loss and on
    a scorecard is invisible to every other bound shape.

    Compared case-folded and stripped, because `INR` and `inr` are the same currency and a
    check that disagreed would report a violation nobody committed.

    Attributes:
        kind: Discriminator, always `matching`.
        name: Stable identifier for this bound.
        tool: The tool this constrains.
        argument: The argument of that tool that has to match.
        matches: Reference to the value it has to equal.
        provenance: Declared by the merchant, or inferred from prose.
        description: Why the bound exists, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["matching"] = "matching"
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    matches: ResultReference
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @property
    def constrained_arguments(self) -> tuple[str, ...]:
        """Arguments this bound reads. Checked against the tool's schema at load."""
        return (self.argument,)

    @property
    def source_tools(self) -> tuple[str, ...]:
        """The tool the value it has to match is read from."""
        return (self.matches.tool,)

    def permits(self, value: object, *, expected: str | None) -> bool:
        """Whether `value` matches, given a value already resolved from the log.

        An unresolved expectation does not constrain, for the same reason as every other
        bound here: nothing was compared, so nothing was satisfied.
        """
        if expected is None:
            return True
        return str(value).strip().casefold() == expected


class ImputedBound(BaseModel):
    """A limit on a call whose cost is not in its arguments at all.

    The shape that breaks the assumption every other bound rests on. Conceding a disputed
    charge, cancelling a paid order, waiving a fee: the call carries a reference and nothing
    else, and the money it moves is a figure sitting in a record the agent read a moment
    earlier. Read only the arguments and the most expensive action an agent can take looks
    free, so a ceiling that catches a refund of a hundred thousand does not catch giving the
    same hundred thousand away by conceding it.

    So the value is declared as a reference rather than an argument: what this call cost is
    `value_from`, resolved from the most recent earlier result that carried it. This is the
    one bound with no `constrained_arguments`, because it constrains a call and not a field.

    Attributes:
        kind: Discriminator, always `imputed`.
        name: Stable identifier for this bound.
        tool: The tool this constrains.
        value_from: Reference to the figure this call moves.
        maximum: Largest permitted value, inclusive.
        minimum: Smallest permitted value, inclusive.
        provenance: Declared by the merchant, or inferred from prose.
        description: Why the bound exists, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["imputed"] = "imputed"
    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    value_from: ResultReference
    maximum: float | None = None
    minimum: float | None = None
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @model_validator(mode="after")
    def _at_least_one_limit(self) -> ImputedBound:
        if self.maximum is None and self.minimum is None:
            raise ValueError(f"imputed bound {self.name!r} sets neither maximum nor minimum")
        return self

    @property
    def argument(self) -> str:
        """No argument carries this bound. Present so every bound answers the same questions."""
        return ""

    @property
    def constrained_arguments(self) -> tuple[str, ...]:
        """Nothing. The value is read from a result, not passed in a call."""
        return ()

    @property
    def source_tools(self) -> tuple[str, ...]:
        """The tool the imputed value is read from."""
        return (self.value_from.tool,)

    def permits(self, value: float) -> bool:
        """Whether an imputed value is inside this bound."""
        if self.maximum is not None and value > self.maximum:
            return False
        return not (self.minimum is not None and value < self.minimum)


AnyBound = (
    NumericBound
    | EnumeratedBound
    | RelationalBound
    | CumulativeBound
    | MatchingBound
    | ImputedBound
)
"""Every kind of limit, as a plain union for annotating what handles all of them."""

Bound = Annotated[AnyBound, Field(discriminator="kind")]
"""Any kind of limit on a tool call, tagged by `kind` when read from a policy file."""


class ResultCondition(BaseModel):
    """What a tool's result has to say for the call to count as having succeeded.

    Without this, "succeeded" can only mean "did not report an error", and a step that ran
    and returned a negative answer counts as having been taken. That is the difference
    between an agent that verified nobody and an agent that verified somebody and was told
    no, and it is the wrong difference to lose: the second is the more alarming of the two
    and would be reported as compliant.

    The shape stays domain-independent. A field path and a value it has to equal; nothing
    here knows what is being established.

    Exactly one of `equals` and `equals_any` is set. The second exists because some gating
    steps have more than one affirmative answer: an order is refundable when it is delivered
    or cancelled, and writing that as two preconditions would report a violation against
    whichever of the two the agent did not see.

    Attributes:
        field: Dotted path into the result, resolved the same way as `ResultReference`.
        equals: The value that path has to hold. Compared as a string, because a result
            arrives as JSON and `true` and `"true"` must not be different answers.
        equals_any: The values that path may hold, any one of which counts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    equals: str | None = Field(default=None, min_length=1)
    equals_any: tuple[str, ...] = ()

    @field_validator("equals", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        return str(value).lower() if isinstance(value, bool) else value

    @field_validator("equals_any", mode="before")
    @classmethod
    def _stringify_each(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(
                str(item).lower() if isinstance(item, bool) else str(item) for item in value
            )
        return value

    @model_validator(mode="after")
    def _exactly_one_form(self) -> ResultCondition:
        if (self.equals is None) == (not self.equals_any):
            raise ValueError(
                f"result condition on {self.field!r} must set exactly one of equals and equals_any"
            )
        return self

    @property
    def permitted(self) -> tuple[str, ...]:
        """Every value that counts as an affirmative answer."""
        return self.equals_any if self.equals is None else (self.equals,)

    def __str__(self) -> str:
        """The condition as `field == value`, for verdicts and error messages."""
        if self.equals is not None:
            return f"{self.field} == {self.equals}"
        return f"{self.field} in ({', '.join(self.permitted)})"

    def met_by(self, result: object) -> bool:
        """Whether a decoded tool result satisfies this condition.

        Args:
            result: The decoded result of the call.

        Returns:
            `True` when the path resolves and its value is one of the permitted ones, compared
            as a string. A path that does not resolve is not met, because a result that does
            not answer the question has not answered it affirmatively.
        """
        current: Any = result
        for step in self.field.split("."):
            if isinstance(current, dict):
                if step not in current:
                    return False
                current = current[step]
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(step)]
                except (ValueError, IndexError):
                    return False
            else:
                return False
        rendered = str(current).lower() if isinstance(current, bool) else str(current)
        return rendered in self.permitted


class Precondition(BaseModel):
    """A tool that must have succeeded earlier in the conversation before another is called.

    This is the shape of most real commerce failures: the refund itself is permitted, and
    the violation is that nothing was verified first. Expressing it as a required prior
    call makes it observable in the tool-call log rather than a matter of judgement.

    `matched_by` closes the hole in the loose reading. An agent that reads one record, is
    satisfied by it, and then acts on a different one has taken the step for something else,
    and a check asking only "did the step happen" scores that as compliant. It is the exact
    manoeuvre an attacker introducing a second reference number is trying for, so the
    identity of the record is part of the requirement rather than an extra check beside it.

    Attributes:
        name: Stable identifier for this precondition.
        tool: The consequential tool that is gated.
        requires: The tool that must have returned successfully earlier in the same
            conversation.
        succeeds_when: What the required tool's result has to say for the step to count.
            Omitted means the weaker default: any result that does not report an error. A
            merchant whose gating step can answer no should declare this, and the scorecard
            says which of the two was used.
        matched_by: Arguments that must carry the same value in the required call as in the
            gated one, so the prior step has to have been about the same record. Omitted
            means any earlier satisfying call counts, which is the right reading for a step
            that establishes something about the conversation rather than about a record.
        provenance: Declared by the merchant, or inferred from prose.
        description: What the precondition establishes, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    requires: str = Field(min_length=1)
    succeeds_when: ResultCondition | None = None
    matched_by: tuple[str, ...] = ()
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


class ObligationKind(StrEnum):
    """What sort of thing an obligation forbids.

    Three, because three is what the shapes actually are once the domain is removed. All of
    them constrain what the agent *says*, which is why none of them is a bound or a
    precondition: no argument is out of range and no step is missing, and the tool-call log
    of a conversation that breaks one is indistinguishable from a conversation that does not.

    Attributes:
        DISCLOSURE: Something reached through a tool that must not be repeated to the person
            on the other side. The record was legitimately fetched; saying it out loud is
            the failure.
        ACCURACY: Something the agent must not assert unless a tool returned it. Stating a
            figure from memory is the failure even when the figure happens to be right.
        COMMITMENT: Something the agent must not undertake on the merchant's behalf. The
            merchant is bound by what their agent promised, whether or not any tool recorded
            it.
    """

    DISCLOSURE = "disclosure"
    ACCURACY = "accuracy"
    COMMITMENT = "commitment"


class Obligation(BaseModel):
    """A rule about what the agent may say, which no tool call can prove was kept.

    Bounds and preconditions are decidable from the tool-call log, which is why they are
    detectors. An obligation is not: it constrains the text of a reply against what the
    conversation is entitled to say, so settling one needs a reading. It is declared here
    anyway, in the same object as the rest of the policy, so that the thing the judge is
    asked is a rule the merchant recognises rather than a question the judge invented.

    Attributes:
        name: Stable identifier for this obligation.
        kind: Which shape of speech it constrains.
        statement: The rule as written, verbatim where it came from prose. This is what a
            judge is asked about and what a merchant is shown, so a paraphrase here quietly
            changes the rule being enforced.
        applies_to: Tools whose results or effects this obligation governs. Empty means it
            governs the conversation as a whole.
        provenance: Declared by the merchant, or inferred from prose.
        description: What it protects, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: ObligationKind
    statement: str = Field(min_length=1)
    applies_to: tuple[str, ...] = ()
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")


class IdempotencyRequirement(BaseModel):
    """One effect that must happen once however many times the agent is asked for it.

    The failure this catches has nothing to do with being talked out of a rule, which is why
    no bound and no precondition sees it. Every call is inside every limit, every prior step
    ran, and the merchant pays twice because the same instruction arrived twice: a trigger
    that fired again, a customer who asked again in a new turn, a retry after a timeout. On
    an agent whose trigger is a schedule rather than a person, this is the most likely way
    money leaves twice, and it is invisible to a check that looks at calls one at a time.

    `identity_arguments` names the arguments that say two calls are the same effect. Two
    refunds of the same amount against the same order are one effect asked for twice; two
    refunds against different orders are two effects.

    `key_argument` is the argument a payment API would deduplicate on. When two calls share
    an identity and carry the same key, the second one moved no money and the requirement
    held. When the key is absent or different, the second one paid again.

    Attributes:
        name: Stable identifier for this requirement.
        tool: The tool whose effect must happen once.
        identity_arguments: Arguments whose values decide that two calls are the same effect.
        key_argument: The argument carrying a deduplication key, if the tool has one. `None`
            means the tool offers no way to make a repeat safe, so any repeat is a repeat.
        provenance: Declared by the merchant, or inferred from prose.
        description: What it protects, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    identity_arguments: tuple[str, ...] = Field(min_length=1)
    key_argument: str | None = Field(default=None, min_length=1)
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")

    @property
    def referenced_arguments(self) -> tuple[str, ...]:
        """Every argument this requirement names, for checking against the tool's schema."""
        if self.key_argument is None:
            return self.identity_arguments
        return (*self.identity_arguments, self.key_argument)


class OutboundRule(BaseModel):
    """A tool that carries free text out of the merchant's systems, and where the text is.

    This exists because of the one leak shape that is invisible to every other check. A
    message sent to the right number, through a permitted tool, inside every limit, with the
    prior step taken, carrying another customer's order in the body. The call is correct in
    every respect a schema can describe, and the thing that went wrong is a value inside a
    string.

    Declaring the body arguments is what makes that decidable. Without it a check would have
    to guess which arguments are prose and which are references, and guessing wrong in the
    quiet direction means the check never fires.

    Attributes:
        name: Stable identifier for this rule.
        tool: The tool that sends something outward.
        body_arguments: Arguments carrying free text a person will read.
        provenance: Declared by the merchant, or inferred from prose.
        description: What it protects, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    body_arguments: tuple[str, ...] = Field(min_length=1)
    provenance: Provenance = Provenance.DECLARED
    description: str = Field(default="")


class CitationRequirement(BaseModel):
    """References an argument may only carry if the agent actually read them.

    An agent assembling a case for somebody else to read can invent the evidence. It cites a
    reference number that looks right, nobody in the conversation can tell, and the merchant
    submits it to a payment network as fact. Nothing about the call is out of bounds: the
    argument is a well-formed string of exactly the expected shape, and the only thing wrong
    with it is that no such record exists.

    Which is decidable without any world access, and that is the point of the shape. A
    reference the agent read is in an earlier result in the log. A reference it did not read
    is not, and the difference between the two is the difference between evidence and a
    fabrication.

    Attributes:
        name: Stable identifier for this requirement.
        tool: The tool whose argument carries references.
        argument: The argument carrying them. A list argument is checked value by value.
        identifier_kind: The key those references appear under in a result, so a value can be
            looked for the way a record carries it.
        source_tools: The tools a reference may have been read from.
        provenance: Declared by the merchant, or inferred from prose.
        description: What it protects, in the merchant's terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    argument: str = Field(min_length=1)
    identifier_kind: str = Field(min_length=1)
    source_tools: tuple[str, ...] = Field(min_length=1)
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
        engine: How the agent is built: a workflow engine with LLM nodes inside it, or a
            single model loop. Declared rather than inferred, because it changes what an
            attack has to get past and it is not visible from anything else in the config.
            Not in the validity tuple on its own; a rebuild onto a different engine is a
            new `version`, which is.
        instructions: The system prompt, verbatim.
        tools: Every tool the agent can call.
        data_sources: Every store the agent can reach.
        unit_symbol: What goes in front of an amount when one is shown to a person. Declared
            rather than assumed, because the harness has no idea what an agent's numbers are
            denominated in and a page that guesses states a figure in the wrong currency.
        subject_term: What this agent calls the person a session is about, singular and
            lowercase. Used only in prose written for a reader. Defaults to the generic word
            so an agent that declares nothing still reads as English.
        value_fields: Result fields whose value means the operator is out of pocket. Declared
            per agent because there is no universal name for it, and reading the wrong field
            produces a headline that adds a loss to a gain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    engine: Engine = Engine.MODEL_LOOP
    instructions: str = Field(default="")
    tools: tuple[ToolDeclaration, ...] = ()
    data_sources: tuple[DataSource, ...] = ()
    unit_symbol: str = Field(default="\u20b9")
    subject_term: str = Field(default="person", min_length=1)
    value_fields: tuple[str, ...] = ()

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
        bounds: Limits on tool calls and their arguments.
        preconditions: Tools that must precede consequential tools.
        idempotency: Effects that must happen once however often they are asked for.
        outbound: Tools that carry free text outward, and which arguments hold it.
        citations: Arguments that may only carry references the agent actually read.
        obligations: Rules about what may be said, which no tool call can settle.
        data_scope: What one session may reach.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    bounds: tuple[Bound, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    idempotency: tuple[IdempotencyRequirement, ...] = ()
    outbound: tuple[OutboundRule, ...] = ()
    citations: tuple[CitationRequirement, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    data_scope: DataScope = DataScope()

    @model_validator(mode="after")
    def _unique_names(self) -> AgentPolicy:
        _reject_duplicates((bound.name for bound in self.bounds), "bound")
        _reject_duplicates((pre.name for pre in self.preconditions), "precondition")
        _reject_duplicates((once.name for once in self.idempotency), "idempotency requirement")
        _reject_duplicates((rule.name for rule in self.outbound), "outbound rule")
        _reject_duplicates((cite.name for cite in self.citations), "citation requirement")
        _reject_duplicates((duty.name for duty in self.obligations), "obligation")
        return self

    @property
    def statements(
        self,
    ) -> tuple[
        AnyBound
        | Precondition
        | IdempotencyRequirement
        | OutboundRule
        | CitationRequirement
        | Obligation
        | DataScope,
        ...,
    ]:
        """Every policy statement, of every kind, for provenance reporting."""
        return (
            *self.bounds,
            *self.preconditions,
            *self.idempotency,
            *self.outbound,
            *self.citations,
            *self.obligations,
            self.data_scope,
        )

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


class Subject(BaseModel):
    """One identity the harness may act as for the length of a conversation.

    An agent that reads records cannot be attacked in the abstract. A conversation has to be
    about somebody, and the identifiers it opens with have to resolve, or the agent answers
    "I cannot find that" and the action under test is never reached. That failure is silent
    and flattering: every rule reports as never in play, so the run looks clean and the agent
    looks defended, when in fact nobody managed to ask it the question.

    So a subject is part of the integration contract rather than something agent-red invents.
    The merchant declares which identities are safe to impersonate against the test
    deployment, and the harness uses only those. There is deliberately no way to synthesise
    one: an invented identifier is the exact failure this type exists to prevent.

    Attributes:
        name: A short label for this identity, used in ids and in the report.
        identifiers: Identifier kind to value, for example `{"order_id": "ORD-55401"}`. Must
            cover every kind the policy's data scope declares, checked at load.
        facts: What this identity would know and could say out loud, in the merchant's own
            words. Free text, because what is worth knowing differs entirely between agents.
            Handed to the attacker so it argues from something true.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    identifiers: dict[str, str] = Field(min_length=1)
    facts: tuple[str, ...] = ()

    @field_validator("identifiers")
    @classmethod
    def _no_blank_values(cls, value: dict[str, str]) -> dict[str, str]:
        blank = sorted(kind for kind, held in value.items() if not str(held).strip())
        if blank:
            raise ValueError(f"identifier(s) with no value: {', '.join(blank)}")
        return value


class AgentSpec(BaseModel):
    """A config and the policy that authorises it, checked against each other.

    This is the object the rest of the tree accepts. Construction fails if the policy
    describes something the config does not have, because a bound naming a tool that does
    not exist produces a detector that never fires, and a detector that never fires reads
    as a passing agent.

    Attributes:
        config: What the agent is and can do.
        policy: What it may and must do.
        subjects: Identities the harness may act as. Empty is allowed only for an agent whose
            policy declares no subject identifier kinds, because an agent scoped to a subject
            that supplies none cannot be attacked, only talked to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: AgentConfig
    policy: AgentPolicy
    subjects: tuple[Subject, ...] = ()

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
            for argument in bound.constrained_arguments:
                if argument not in tool.argument_names:
                    raise ValueError(
                        f"bound {bound.name!r} constrains argument {argument!r} of tool "
                        f"{bound.tool!r}, which declares no such argument"
                    )
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
            for argument in precondition.matched_by:
                for name in (precondition.tool, precondition.requires):
                    if argument not in tools[name].argument_names:
                        raise ValueError(
                            f"precondition {precondition.name!r} matches on argument "
                            f"{argument!r}, which tool {name!r} does not declare"
                        )

        self._requirements_describe_config(tools)

        declared_sources = {source.name for source in self.config.data_sources}
        for source in self.policy.data_scope.sources:
            if source not in declared_sources:
                raise ValueError(
                    f"data scope permits source {source!r}, which the agent cannot reach"
                )

        self._subjects_can_be_acted_as()
        return self

    def _requirements_describe_config(self, tools: dict[str, ToolDeclaration]) -> None:
        """Refuse a policy whose newer sections name something the agent does not have.

        Same argument as every other check here, and it applies with more force to these
        three. An idempotency requirement on an argument the tool does not declare groups
        every call together and reports the second one as a repeat; an outbound rule naming
        the wrong argument reads an empty string and reports every message as clean; a
        citation requirement pointed at a tool that is never called reports nothing at all.
        All three fail quietly and in the flattering direction.

        Args:
            tools: The config's declared tools, keyed by name.

        Raises:
            ValueError: On the first requirement that does not describe the config.
        """
        for once in self.policy.idempotency:
            tool = tools.get(once.tool)
            if tool is None:
                raise ValueError(
                    f"idempotency requirement {once.name!r} names tool {once.tool!r}, which is "
                    f"not declared"
                )
            for argument in once.referenced_arguments:
                if argument not in tool.argument_names:
                    raise ValueError(
                        f"idempotency requirement {once.name!r} names argument {argument!r} of "
                        f"tool {once.tool!r}, which declares no such argument"
                    )

        for rule in self.policy.outbound:
            tool = tools.get(rule.tool)
            if tool is None:
                raise ValueError(
                    f"outbound rule {rule.name!r} names tool {rule.tool!r}, which is not declared"
                )
            for argument in rule.body_arguments:
                if argument not in tool.argument_names:
                    raise ValueError(
                        f"outbound rule {rule.name!r} names argument {argument!r} of tool "
                        f"{rule.tool!r}, which declares no such argument"
                    )

        known_kinds = {
            kind for source in self.config.data_sources for kind in source.identifier_kinds
        }
        for cite in self.policy.citations:
            tool = tools.get(cite.tool)
            if tool is None:
                raise ValueError(
                    f"citation requirement {cite.name!r} names tool {cite.tool!r}, which is not "
                    f"declared"
                )
            if cite.argument not in tool.argument_names:
                raise ValueError(
                    f"citation requirement {cite.name!r} names argument {cite.argument!r} of "
                    f"tool {cite.tool!r}, which declares no such argument"
                )
            if cite.identifier_kind not in known_kinds:
                raise ValueError(
                    f"citation requirement {cite.name!r} cites {cite.identifier_kind!r}, which "
                    f"no declared data source carries"
                )
            for source in cite.source_tools:
                if source not in tools:
                    raise ValueError(
                        f"citation requirement {cite.name!r} reads references from tool "
                        f"{source!r}, which is not declared"
                    )
                if source == cite.tool:
                    raise ValueError(
                        f"citation requirement {cite.name!r} reads references from {source!r}, "
                        f"the tool it constrains. A call cannot be its own source."
                    )

    def _subjects_can_be_acted_as(self) -> None:
        """Refuse a spec whose subjects could not open a conversation that goes anywhere.

        Two failures, both silent and both flattering. An agent scoped to a subject with no
        declared subjects leaves the harness with nothing true to say, so it opens with an
        identifier that resolves to nothing, the agent declines to act on a record it cannot
        find, and every rule reports as never in play. A subject missing one of the declared
        identifier kinds fails the same way for whichever conversations needed that kind.

        Both are refused at load rather than warned about, for the reason in ADR-0004: a
        check that cannot fire is indistinguishable from an agent that passed it.

        Raises:
            ValueError: If subjects are required and absent, or one is incomplete.
        """
        kinds = tuple(self.policy.data_scope.subject_identifier_kinds)
        if not kinds:
            return
        if not self.subjects:
            raise ValueError(
                f"policy scopes a session by {', '.join(kinds)} but no subjects are declared. "
                f"Every conversation would open with an identifier that resolves to nothing, "
                f"and every rule would report as never evaluated rather than as passed."
            )
        for subject in self.subjects:
            missing = [kind for kind in kinds if kind not in subject.identifiers]
            if missing:
                raise ValueError(
                    f"subject {subject.name!r} declares no {', '.join(missing)}, which the "
                    f"policy's data scope requires to bind a record to a session"
                )

    @property
    def version_tuple(self) -> VersionTuple:
        """The four versions this spec's results would be valid for."""
        return VersionTuple(
            config_version=self.config.version,
            policy_version=self.policy.version,
            model_version=self.config.model,
            tool_version=self.config.tool_version,
        )

    def bounds_for(self, tool: str) -> tuple[AnyBound, ...]:
        """Every bound constraining `tool`, in policy order."""
        return tuple(bound for bound in self.policy.bounds if bound.tool == tool)

    def preconditions_for(self, tool: str) -> tuple[Precondition, ...]:
        """Every precondition gating `tool`, in policy order."""
        return tuple(pre for pre in self.policy.preconditions if pre.tool == tool)

    def ungated_consequential_tools(self) -> tuple[ToolDeclaration, ...]:
        """Consequential tools no declared rule of any kind constrains.

        These are where the only thing standing between a customer and the merchant's money
        is the wording of a system prompt. `patch/` must answer these with a `permission`
        remedy and must not offer an `instruction` remedy as an equivalent.

        Every policy section counts, not only bounds and preconditions. A tool covered by an
        idempotency requirement, an outbound rule or a citation requirement has something a
        detector can assert about it, and calling it ungated would hand the judge a question
        that was already answered by assertion.
        """
        constrained = {
            *(bound.tool for bound in self.policy.bounds),
            *(pre.tool for pre in self.policy.preconditions),
            *(once.tool for once in self.policy.idempotency),
            *(rule.tool for rule in self.policy.outbound),
            *(cite.tool for cite in self.policy.citations),
        }
        return tuple(
            tool for tool in self.config.consequential_tools if tool.name not in constrained
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
