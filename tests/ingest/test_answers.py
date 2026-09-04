"""Answering the one question every read of every agent raises.

No connector protocol carries what a wrong call to a tool costs, so every tool comes back
undetermined and a declaration is refused until somebody says. `answered` is where their
answers land, and the two ways it refuses matter more than the way it succeeds: a tool nobody
answered for would fall out of the suite in silence, and an answer for a tool that was never
advertised means the answers describe a different agent from the one that was read.
"""

from __future__ import annotations

import pytest

from agentred.ingest.package import AgentPackage, Evidence, Observation, Origin, ToolFacts
from agentred.spec.models import Consequence

QUESTION = "what does a wrong call cost?"


def tool(name: str) -> ToolFacts:
    """One advertised tool with its consequence undetermined, as a connector leaves it."""
    evidence = Evidence(adapter="mcp", locator="http://connector/mcp")
    return ToolFacts(
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        consequence=Observation[Consequence](
            value=None, origin=Origin.UNDETERMINED, evidence=evidence, question=QUESTION
        ),
        evidence=evidence,
    )


def package(*names: str) -> AgentPackage:
    """A package holding nothing but the tools a connector advertised."""
    return AgentPackage(agent_id="an_agent", tools=tuple(tool(name) for name in names))


def test_an_unanswered_package_names_one_question_per_tool() -> None:
    subjects = [subject for subject, _ in package("read_it", "pay_it").unresolved]
    assert subjects == ["tool read_it", "tool pay_it"]


def test_answering_every_tool_resolves_the_package() -> None:
    answered = package("read_it", "pay_it").answered(
        {"read_it": Consequence.DISCLOSURE, "pay_it": Consequence.MONEY}, by="fde"
    )

    assert answered.unresolved == ()
    assert [item.consequence.require() for item in answered.tools] == [
        Consequence.DISCLOSURE,
        Consequence.MONEY,
    ]


def test_an_answer_is_marked_as_a_person_rather_than_as_the_platform() -> None:
    answered = package("pay_it").answered({"pay_it": Consequence.MONEY}, by="fde")

    assert answered.tools[0].consequence.origin is Origin.CONFIRMED
    assert answered.tools[0].consequence.evidence.locator == "fde"


def test_a_tool_left_unanswered_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="no answer for pay_it"):
        package("read_it", "pay_it").answered({"read_it": Consequence.DISCLOSURE}, by="fde")


def test_an_answer_for_a_tool_nobody_advertised_is_refused() -> None:
    with pytest.raises(ValueError, match="do not advertise"):
        package("read_it").answered(
            {"read_it": Consequence.DISCLOSURE, "refund_it": Consequence.MONEY}, by="fde"
        )
