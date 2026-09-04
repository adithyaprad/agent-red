"""Reading required ordering off a workflow's own step graph.

A workflow engine exists to make ordering deterministic: a step cannot run before the step in
front of it. That is the property a platform migrates onto one to get, and read from the
outside it is a policy statement. A workflow whose order-reading step precedes its refunding
step has declared that a refund follows a read of the order it refunds. Nobody wrote that
down as a rule. The engine enforces it anyway, which makes it the strongest kind of rule
there is: one that is true of the deployment rather than true of a document about it.

**Why this is the half worth having.** Preconditions are the shape prose is worst at
expressing and attacks are best at breaking, because they are rules about a pair of calls
rather than about the arguments of either one. Every argument is in range and every value is
plausible; what is wrong is that the second call happened without the first. Reading them off
the graph moves them out of the half that is guessed and into the half that is read.

**Two tools in order are not a precondition, and the difference is the argument they share.**
Every tool in an early step precedes every tool in a late one, so ordering alone would emit
the cross product and call it a policy: a hundred rules, most of them meaningless, and the
five real ones lost among them. A precondition is emitted only where the later tool takes an
argument the earlier one also takes, because that shared argument is what makes the rule
checkable at all. `issue_refund(order_id)` after `get_order(order_id)` is a rule a detector
can settle, and it is the same `matched_by` that stops a read of one order from satisfying
the requirement for a refund against a different one.

**A step that hides its tools is reported, not worked around.** A step carrying an agent
declares that agent's tools; a step whose behaviour is closed over inside a function does
not. The ordering is read either way. Where the tools are invisible the note says so, so that
a workflow yielding no preconditions is never read as a workflow that requires no ordering.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agentred.ingest.package import AgentPackage, Evidence, Observation, Origin, RuleFacts
from agentred.spec.models import Engine, Precondition, Provenance

ADAPTER = "workflow"
"""The adapter name recorded on every piece of evidence this module produces."""

OPAQUE_STEP_NOTE = (
    "steps {names} run in a known order and do not declare which tools they may call, so "
    "their ordering yields no rule. A workflow that reports no preconditions has not "
    "declared that it requires no ordering."
)
"""What is said when a step's tools cannot be seen. See the module docstring."""


def _step_tools(step: Any) -> tuple[str, ...]:
    """The tool names one step declares, in the order it declares them.

    Args:
        step: An `agno.workflow.Step`.

    Returns:
        The names of the tools the step's agent carries, or an empty tuple for a step whose
        behaviour is a function rather than an agent. Empty means unknown here, not none:
        the caller separates the two and this function deliberately does not guess.
    """
    agent = getattr(step, "agent", None)
    tools = getattr(agent, "tools", None) if agent is not None else None
    if not tools:
        return ()
    names = []
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name:
            names.append(str(name))
    return tuple(names)


def _shared_arguments(package: AgentPackage, later: str, earlier: str) -> tuple[str, ...]:
    """Argument names both tools take, which is what a precondition would be matched on.

    Args:
        package: The package holding the tool schemas, normally read off a connector first.
        later: The tool that runs second.
        earlier: The tool that runs first.

    Returns:
        The shared top-level argument names, sorted. Empty when the two tools have no
        argument in common, which is the signal that their ordering is incidental rather
        than required.
    """
    schemas = {tool.name: tool.parameters for tool in package.tools}
    if later not in schemas or earlier not in schemas:
        return ()

    def arguments(name: str) -> set[str]:
        properties = schemas[name].get("properties")
        return set(map(str, properties)) if isinstance(properties, dict) else set()

    return tuple(sorted(arguments(later) & arguments(earlier)))


def read_workflow(workflow: Any, package: AgentPackage) -> AgentPackage:
    """Add to a package everything a workflow's step graph declares.

    Args:
        workflow: An `agno.workflow.Workflow`, already constructed. Its steps are read and
            nothing in it is run: a workflow executed to find out what it requires would
            make the calls the requirement is about.
        package: What earlier readers recovered. The tool schemas on it are what turn an
            ordering into a rule, so a package with no tools yields no preconditions and
            says why.

    Returns:
        The package with the engine declared, one precondition per ordered pair of tools
        that share an argument, and a note for every step whose tools could not be seen.
    """
    steps = list(getattr(workflow, "steps", []) or [])
    seen: list[tuple[str, tuple[str, ...]]] = []
    opaque: list[str] = []
    rules: list[RuleFacts] = []
    workflow_name = str(getattr(workflow, "name", "") or "workflow")

    for step in steps:
        name = str(getattr(step, "name", "") or "step")
        tools = _step_tools(step)
        if not tools:
            opaque.append(name)
        for later in tools:
            for earlier_step, earlier_tools in seen:
                for earlier in earlier_tools:
                    matched_by = _shared_arguments(package, later, earlier)
                    if not matched_by:
                        continue
                    rules.append(
                        RuleFacts(
                            rule=Precondition(
                                name=f"{later}_follows_{earlier}",
                                tool=later,
                                requires=earlier,
                                matched_by=matched_by,
                                provenance=Provenance.DECLARED,
                                description=(
                                    f"Step {name!r} runs after step {earlier_step!r}, so "
                                    f"{later} cannot run before {earlier} for the same "
                                    f"{', '.join(matched_by)}."
                                ),
                            ),
                            origin=Origin.DECLARED,
                            evidence=Evidence(
                                adapter=ADAPTER,
                                locator=f"{workflow_name}: {earlier_step} -> {name}",
                            ),
                        )
                    )
        seen.append((name, tools))

    notes = package.notes
    if opaque:
        notes = (*notes, OPAQUE_STEP_NOTE.format(names=", ".join(opaque)))
    return replace(
        package.with_rules(*rules),
        engine=Observation[Engine](
            value=Engine.WORKFLOW,
            origin=Origin.DECLARED,
            evidence=Evidence(adapter=ADAPTER, locator=workflow_name),
        ),
        notes=notes,
        sources=(*package.sources, ADAPTER),
    )
