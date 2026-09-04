"""Turning what a reader recovered into a declaration the harness will run against.

This is the one place a package stops being findings and becomes a claim, so it is the one
place that can refuse. It refuses on exactly one condition: an unresolved fact. Everything
else about a package is somebody's business, but a hole that gets a value here would get it
silently, and the reader's whole shape exists to stop that.

**What it will not invent.** `version` and `model` are part of the tuple a scorecard is valid
for, and neither is discoverable from a tool surface. They are arguments rather than
defaults, because a scorecard citing a guessed model version is a scorecard that cannot be
matched to the agent it describes, and the failure is silent at the moment it is introduced
and load-bearing months later when somebody asks whether a result still holds.

**What it does not emit yet, and says so.** A config recovered from a connector alone carries
tools and nothing else: no data sources, no channels, no policy. That is not a config with
gaps papered over, it is a smaller true statement, and `unreadable` names what a caller has
not supplied a reader for so a thin declaration is never mistaken for a thorough one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from agentred.ingest.package import AgentPackage, PolicyRule
from agentred.spec.models import (
    AgentConfig,
    AgentPolicy,
    AgentSpec,
    Engine,
    Precondition,
    Subject,
    ToolDeclaration,
)


class EmitError(Exception):
    """A package could not be turned into a declaration.

    Carries every reason at once rather than the first, because the caller is a person about
    to answer questions and answering them one run at a time is how a twenty-question list
    becomes twenty runs.
    """


@dataclass(frozen=True, slots=True)
class Emission:
    """A declaration and an honest account of what is not in it.

    Attributes:
        config: The declaration, built only from resolved facts.
        unreadable: What no reader supplied, as one line each. Not errors: an agent read
            from a connector alone genuinely has no known channels, and the difference
            between a channel that does not exist and a channel nobody looked for is the
            difference a coverage grid would otherwise get wrong.
        policy: The bounds and preconditions recovered, when any reader supplied one.
        notes: What a reader could see and could not express, carried from the package. A
            policy holding only per-call ceilings is a true statement and a misleading one
            without the note saying which rule shapes its source could not hold.
    """

    config: AgentConfig
    unreadable: tuple[str, ...] = ()
    policy: AgentPolicy | None = None
    notes: tuple[str, ...] = ()


def _refuse_unresolved(package: AgentPackage) -> None:
    """Raise if any fact in the package is still a hole.

    Args:
        package: What the readers recovered.

    Raises:
        EmitError: Listing every unresolved subject and its question.
    """
    if not package.unresolved:
        return
    lines = "\n".join(f"  {subject}: {question}" for subject, question in package.unresolved)
    raise EmitError(
        f"{len(package.unresolved)} question(s) about {package.agent_id} have no answer, and "
        f"a declaration emitted over them would be a guess reported as a fact:\n{lines}"
    )


def to_config(
    package: AgentPackage, *, version: str, model: str, instructions: str = ""
) -> Emission:
    """Build a declaration from a fully resolved package.

    Args:
        package: What the readers recovered. Every observation in it must be resolved.
        version: Version string for the emitted config. Part of the validity tuple, so it
            is supplied rather than derived: a reader has no way to know whether what it
            just read is a new version of an agent or the same one re-read.
        model: The model the agent runs on. Also part of the validity tuple, and not
            visible from a tool surface.
        instructions: The agent's prose, when a reader supplied it.

    Returns:
        The declaration, and the list of what no reader covered.

    Raises:
        EmitError: If any fact is unresolved, or if the resolved facts do not form a valid
            config. The second is not expected and is surfaced rather than swallowed,
            because it means a reader produced something the spec refuses and the reader is
            what needs fixing.
    """
    _refuse_unresolved(package)
    tools = tuple(
        ToolDeclaration(
            name=facts.name,
            description=facts.description,
            parameters=dict(facts.parameters),
            consequence=facts.consequence.require(),
        )
        for facts in package.tools
    )
    engine = package.engine.require() if package.engine is not None else Engine.MODEL_LOOP
    try:
        config = AgentConfig(
            agent_id=package.agent_id,
            version=version,
            model=model,
            engine=engine,
            instructions=instructions or package.instructions,
            tools=tools,
            data_sources=package.data_sources,
        )
    except Exception as error:
        raise EmitError(
            f"the facts recovered for {package.agent_id} do not form a valid config: {error}"
        ) from error
    return Emission(config=config, unreadable=_unreadable(package))


def _unreadable(package: AgentPackage) -> tuple[str, ...]:
    """What the readers that ran had no way to supply.

    Args:
        package: What the readers recovered.

    Returns:
        One line per part of a declaration no contributing reader covers, phrased as the
        source that would carry it. Silence here would let a config holding tools and
        nothing else read as an agent with no data sources and no channels, which is a
        different and much more flattering statement than an agent nobody asked.
    """
    missing: list[str] = []
    if not package.tools:
        missing.append("tools: no reader supplied a tool surface")
    if package.engine is None:
        missing.append("engine: no reader said how the agent is built, so a model loop is assumed")
    if not package.instructions:
        missing.append("instructions: no reader supplied the agent's prose")
    if not package.data_sources:
        missing.append("data sources: read from a catalogue manifest")
    if package.data_scope is None:
        missing.append("data scope: read from an instance configuration")
    if not package.rules:
        missing.append("policy: read from an instance configuration and a workflow definition")
    missing.append("channels and trigger: read from a manifest, and confirmed by an operator")
    missing.append("subjects: supplied by the harness, never recovered from a platform")
    return tuple(missing)


def _sorted_rules(
    rules: tuple[PolicyRule, ...],
) -> tuple[tuple[PolicyRule, ...], tuple[Precondition, ...]]:
    """Split recovered rules into the buckets a policy keeps them in.

    Args:
        rules: Every rule any reader produced, in the order they were read.

    Returns:
        The bounds and the preconditions, each keeping its reading order so that a policy
        diffed against a previous run does not churn on ordering alone.
    """
    bounds = tuple(rule for rule in rules if not isinstance(rule, Precondition))
    preconditions = tuple(rule for rule in rules if isinstance(rule, Precondition))
    return bounds, preconditions


def to_policy(package: AgentPackage, *, version: str) -> AgentPolicy | None:
    """Build a policy from the rules the readers recovered.

    Args:
        package: What the readers recovered.
        version: Version string for the emitted policy. Part of the validity tuple, and
            supplied for the same reason the config's is.

    Returns:
        The policy, or `None` when no reader produced a rule. `None` rather than an empty
        policy: an agent with no recovered rules and an agent declared to have no rules are
        opposite facts, and an empty policy states the second.

    Raises:
        EmitError: If the recovered rules do not form a valid policy.
    """
    if not package.rules:
        return None
    bounds, preconditions = _sorted_rules(tuple(facts.rule for facts in package.rules))
    scope = package.data_scope.require() if package.data_scope is not None else None
    try:
        return AgentPolicy(
            agent_id=package.agent_id,
            version=version,
            bounds=bounds,
            preconditions=preconditions,
            data_scope=scope,
        )
    except Exception as error:
        raise EmitError(
            f"the rules recovered for {package.agent_id} do not form a valid policy: {error}"
        ) from error


def to_spec(
    package: AgentPackage,
    *,
    version: str,
    model: str,
    policy_version: str | None = None,
    subjects: Sequence[Subject] = (),
) -> tuple[Emission, AgentSpec | None]:
    """Emit both halves, and the spec they form when both are present.

    Args:
        package: What the readers recovered. Every observation in it must be resolved.
        version: Version string for the config.
        model: The model the agent runs on.
        policy_version: Version string for the policy. Defaults to the config's, which is
            right for a first read of an agent and wrong the moment the two are versioned
            apart, which is why it can be given.
        subjects: Identities the harness may act as. Not recovered and not recoverable: a
            subject is a test fixture rather than a rule, which is why the spec loader keeps
            them in their own file, and a platform has no reason to record who a red team
            may impersonate. Supplied here rather than invented, and a policy that scopes a
            session by identifier without them is a policy every rule of which would report
            as never evaluated rather than as passed.

    Returns:
        The emission, and the spec when a policy was recovered and subjects were supplied
        for it. `None` for the spec means one of the two halves is missing, which is a state
        to report rather than to fail on: it is exactly what reading one connector and
        nothing else produces.

    Raises:
        EmitError: If any fact is unresolved, or the recovered facts do not validate.
    """
    emission = to_config(package, version=version, model=model)
    policy = to_policy(package, version=policy_version or version)
    emission = replace(emission, policy=policy, notes=package.notes)
    if policy is None:
        return emission, None
    try:
        return emission, AgentSpec(config=emission.config, policy=policy, subjects=tuple(subjects))
    except Exception as error:
        raise EmitError(
            f"the halves recovered for {package.agent_id} do not agree with each other: {error}"
        ) from error
