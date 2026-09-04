"""What a reader recovered from an agent's own platform, before any of it is a declaration.

Every adapter in `adapters/` produces one of these and nothing else produces one. It is
deliberately not an `AgentConfig`: a config is a claim about an agent, and what an adapter
has is a set of findings of wildly differing trustworthiness that have not yet earned the
right to be stated as one.

**The failure this shape exists to prevent is the silent default.** `ToolDeclaration`
requires a `Consequence`, and no MCP `tools/list` response carries one. The obvious move is
to default it, and the obvious default is `inert`, and a tool declared inert is a tool the
stakes lattice never aims an attack at. So `issue_refund` read off a live server, defaulted
once, disappears from the suite and the run reports a clean sheet against an agent nobody
attacked. That is rule 13 one layer earlier than rule 13 was written for: an unknown that
gets a value is indistinguishable from an unknown that got answered, and the flattering
reading is the one that has to be made impossible.

So nothing here defaults. A fact the platform did not carry is recorded as `UNDETERMINED`,
`AgentPackage.unresolved` lists every one of them, and emitting a declaration with any left
is refused rather than filled in.

**Origin is finer here than in the spec on purpose.** `spec.models.Provenance` has two
values, because by the time a scorecard cites a rule the only question left is whether a
human stood behind it. Four values are needed one step earlier, because the difference
between a limit the platform stored and a limit a model read out of a sentence is the
difference between the two halves of the reader, and it is the thing an operator confirming
a draft is being asked to look at. `origin_provenance` collapses the four down to the two at
the boundary, and that narrowing happens once, here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from agentred.spec.models import (
    Consequence,
    CumulativeBound,
    DataScope,
    DataSource,
    Engine,
    EnumeratedBound,
    ImputedBound,
    MatchingBound,
    NumericBound,
    Precondition,
    Provenance,
    RelationalBound,
    ToolBehaviour,
)

type PolicyRule = (
    NumericBound
    | EnumeratedBound
    | RelationalBound
    | CumulativeBound
    | MatchingBound
    | ImputedBound
    | Precondition
)
"""Every policy statement a reader can produce, in the spec's own vocabulary.

Deliberately the spec types rather than a parallel set of reader types. A reader that
modelled a limit its own way would have to translate at emit, and a translation is where a
rule quietly changes shape: a ceiling read as exclusive and emitted as inclusive is a bound
nobody notices is off by one until an attack lands on the boundary and is scored as a hold.
"""


class Origin(StrEnum):
    """Where one recovered fact came from, and therefore how far it can be trusted.

    Ordered from the strongest claim to the absence of one. The three that are not
    `DECLARED` all collapse to `Provenance.INFERRED` at the spec boundary, but they are kept
    apart until then because an operator confirming a drafted policy is answering a
    different question about each: whether the model read the sentence correctly, whether
    the sentence meant what it appears to mean, and whether a rule nobody wrote down applies
    to them at all.

    Attributes:
        DECLARED: Read from a structured field the platform itself wrote: an MCP input
            schema, a connector manifest, a workflow's step graph. Not a model's opinion,
            and re-reading the same platform gives the same answer.
        CONFIRMED: A person was shown the question and answered it. As strong as `DECLARED`
            and reached differently: nothing on the platform carries the fact, so the only
            way to have it is to ask. This is what an operator confirming a draft produces,
            and it is why confirming is worth doing rather than being a formality.
        STATED: The operator's own prose says it in as many words, and the quote is carried
            in the evidence. A human wrote it; nobody put it anywhere a machine looks.
        INFERRED: A model read prose and concluded it. May be wrong in either direction,
            and there is no way to check it short of asking the operator.
        ASSUMED: A universal nobody wrote down, such as a refund not exceeding what was
            paid. Defensible everywhere and declared nowhere, which makes it the easiest
            kind of rule to slip in unnoticed and the reason it gets its own value.
        UNDETERMINED: Nothing in the package answers this, and something downstream requires
            it. Not a value. A hole, carried so it can be reported instead of filled.
    """

    DECLARED = "declared"
    CONFIRMED = "confirmed"
    STATED = "stated"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    UNDETERMINED = "undetermined"


def origin_provenance(origin: Origin) -> Provenance:
    """Narrow a reader's four-way origin to the spec's two-way provenance.

    Args:
        origin: How the reader came by a fact.

    Returns:
        `Provenance.DECLARED` for a fact the platform recorded and for one a person was
        asked and answered, because in both a human stands behind the statement. Everything
        else, including a rule the operator's prose states outright, is `INFERRED`: prose a
        model turned into a bound is a model's reading of a human's sentence, and the
        scorecard should say so until somebody has been shown it and agreed.

    Raises:
        ValueError: For `UNDETERMINED`, which is the absence of a fact and has no
            provenance. Reaching this means an unresolved package got as far as emit.
    """
    if origin is Origin.UNDETERMINED:
        raise ValueError(
            "an undetermined fact has no provenance: it should have been refused at emit"
        )
    return (
        Provenance.DECLARED
        if origin in (Origin.DECLARED, Origin.CONFIRMED)
        else Provenance.INFERRED
    )


@dataclass(frozen=True, slots=True)
class Evidence:
    """Where a recovered fact came from, in terms the person checking it can follow.

    Every fact carries one. An operator asked to confirm a drafted discount ceiling needs
    the sentence it was read out of, and an engineer asked why a tool was declared inert
    needs the server that said so. A draft without evidence is a draft nobody can check,
    which in practice is a draft everybody approves.

    Attributes:
        adapter: Which reader produced the fact, for example `mcp` or `agno`.
        locator: Where it was found: a URL, a file path, a workflow step name. Free text
            because the three adapters address their sources in three different ways.
        excerpt: The text the fact was read from, when there is one. Empty for a fact read
            out of a structure rather than out of prose.
    """

    adapter: str
    locator: str
    excerpt: str = ""


@dataclass(frozen=True, slots=True)
class Observation[T]:
    """One recovered fact, its origin and where it was found.

    `value` is `None` exactly when `origin` is `UNDETERMINED`, and the two are checked
    against each other at construction so that a hole cannot be smuggled through as a
    present-but-empty value.

    Attributes:
        value: What was recovered, or `None` for an unresolved fact.
        origin: How far the value can be trusted. See `Origin`.
        evidence: Where it came from.
        question: What an operator would have to answer to resolve it. Required when the
            origin is `UNDETERMINED`, because a hole reported without the question is a
            report that names a problem and withholds the fix.
    """

    value: T | None
    origin: Origin
    evidence: Evidence
    question: str = ""

    def __post_init__(self) -> None:
        """Refuse an observation whose value and origin disagree.

        Raises:
            ValueError: If an undetermined observation carries a value or omits its
                question, or if a determined one carries no value.
        """
        if self.origin is Origin.UNDETERMINED:
            if self.value is not None:
                raise ValueError("an undetermined observation cannot carry a value")
            if not self.question:
                raise ValueError("an undetermined observation must say what would resolve it")
        elif self.value is None:
            raise ValueError(f"a {self.origin} observation must carry a value")

    @property
    def resolved(self) -> bool:
        """Whether this fact has a value a declaration may be built from."""
        return self.origin is not Origin.UNDETERMINED

    def confirmed(self, value: T, *, by: str) -> Observation[T]:
        """The same fact, answered by a person who was shown the question.

        Args:
            value: What they answered.
            by: Who answered, recorded as the locator so a draft approved by the wrong
                person is visible rather than anonymous.

        Returns:
            A `CONFIRMED` observation keeping this one's question and the excerpt it was
            asked from, so the answer stays attached to what was actually asked.
        """
        return Observation[T](
            value=value,
            origin=Origin.CONFIRMED,
            evidence=Evidence(adapter=self.evidence.adapter, locator=by, excerpt=self.question),
            question="",
        )

    def require(self) -> T:
        """The value, for a caller that has already checked the package is resolved.

        Returns:
            The recovered value.

        Raises:
            ValueError: If the observation is unresolved.
        """
        if self.value is None:
            raise ValueError(f"unresolved observation: {self.question}")
        return self.value


@dataclass(frozen=True, slots=True)
class ToolFacts:
    """One tool the platform advertises, as far as any reader can see it.

    `name` and `parameters` are certain: they came off a schema, and the agent could not
    call the tool if they were wrong. `consequence` is the one that is not, and it is the
    field the whole stakes lattice is built from, so it is an `Observation` rather than a
    value even though every other field here is a value.

    Attributes:
        name: The tool name the model calls.
        description: What the tool does, as the platform describes it.
        parameters: JSON Schema for the arguments, verbatim from the platform.
        consequence: What a wrong call costs, once somebody has said.
        evidence: Where the tool itself was found.
        behaviour: What the tool does to the merchant's records, once an operator has
            described it. No connector protocol carries this and no reader produces it, so it
            is absent until somebody supplies it, and a shop cannot be generated without it.
    """

    name: str
    description: str
    parameters: dict[str, object]
    consequence: Observation[Consequence]
    evidence: Evidence
    behaviour: ToolBehaviour | None = None


@dataclass(frozen=True, slots=True)
class RuleFacts:
    """One policy statement and where it was read.

    The rule carries its own provenance, because that is the field a scorecard reads. The
    origin is kept alongside it because provenance has already lost the distinction an
    operator confirming a draft needs: a limit their builder stored and a limit a model read
    out of their prompt both arrive as something, and only one of them is worth their time
    to check.

    Attributes:
        rule: The statement, in the spec's vocabulary, with its provenance already set.
        origin: How the reader came by it, before the narrowing to provenance.
        evidence: Where it was found.
    """

    rule: PolicyRule
    origin: Origin
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class AgentPackage:
    """Everything one or more adapters recovered about a single agent.

    Adapters contribute rather than compete: the MCP reader knows the tool surface, the
    workflow reader knows the step ordering, and the manifest reader knows which of the two
    to point at. A package carries the list of adapters that built it so a surprising fact
    can be traced to the reader that produced it.

    Attributes:
        agent_id: Stable identifier for the agent.
        tools: Every tool the platform advertises.
        engine: How the agent is built, when a reader could tell.
        instructions: The operator's prose, verbatim, for the policy half to read later.
        rules: Policy statements, whatever their shape and wherever they were read.
        data_sources: Stores the agent can reach.
        data_scope: What one session may touch, when a reader supplied it.
        sources: Adapters that contributed, in the order they ran.
        notes: What a reader could see was there and could not express. Not holes: a hole
            is a question somebody can answer, and these are limits of the reader itself,
            carried so a thin policy is not read as a permissive agent.
    """

    agent_id: str
    tools: tuple[ToolFacts, ...] = ()
    engine: Observation[Engine] | None = None
    instructions: str = ""
    rules: tuple[RuleFacts, ...] = ()
    data_sources: tuple[DataSource, ...] = ()
    data_scope: Observation[DataScope] | None = None
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def answered(self, consequences: Mapping[str, Consequence], *, by: str) -> AgentPackage:
        """This package with the per-tool questions answered by a person.

        The one hole every read of every agent produces is what a wrong call to each tool
        costs, because no connector protocol carries it. Answering it is a person's job and
        this is where their answers land, marked `CONFIRMED` rather than `DECLARED` so a
        scorecard can still tell an operator's judgement from a platform's record.

        Args:
            consequences: What each tool costs, keyed by tool name. Every advertised tool
                must appear, and no name that was not advertised may.
            by: Who answered, recorded as the evidence for each answer.

        Returns:
            A new package with those observations resolved.

        Raises:
            ValueError: If a tool was advertised and not answered, or answered and not
                advertised. Both are refused rather than skipped: an unanswered tool would
                fall out of the suite silently, and an answer for a tool that does not exist
                means the answers describe a different agent from the one that was read.
        """
        advertised = {tool.name for tool in self.tools}
        given = set(consequences)
        if missing := sorted(advertised - given):
            raise ValueError(f"no answer for {', '.join(missing)}")
        if extra := sorted(given - advertised):
            raise ValueError(
                f"answered {', '.join(extra)}, which this agent's connectors do not advertise"
            )
        return replace(
            self,
            tools=tuple(
                replace(
                    tool,
                    consequence=tool.consequence.confirmed(consequences[tool.name], by=by),
                )
                for tool in self.tools
            ),
        )

    def with_rules(self, *added: RuleFacts) -> AgentPackage:
        """This package with more rules on it, for a reader that runs after another.

        Args:
            added: Rules to append, in order.

        Returns:
            A new package. Readers contribute rather than compete, and a package is frozen,
            so combining two readers' findings is an operation rather than a mutation.
        """
        return replace(self, rules=(*self.rules, *added))

    @property
    def unresolved(self) -> tuple[tuple[str, str], ...]:
        """Every hole a declaration cannot be emitted over, as `(subject, question)` pairs.

        Returns:
            One pair per unresolved observation, in a stable order. Empty means the package
            can be emitted. The subject names what the question is about, so that an
            operator answering twenty of these knows which tool each one belongs to.
        """
        holes: list[tuple[str, str]] = []
        for tool in self.tools:
            if not tool.consequence.resolved:
                holes.append((f"tool {tool.name}", tool.consequence.question))
        if self.engine is not None and not self.engine.resolved:
            holes.append((f"agent {self.agent_id}", self.engine.question))
        if self.data_scope is not None and not self.data_scope.resolved:
            holes.append((f"agent {self.agent_id} data scope", self.data_scope.question))
        return tuple(holes)
