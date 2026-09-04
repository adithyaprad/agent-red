"""The workflow the installer mapped this skill onto.

Two steps, each carrying the tools it is allowed at that point. The ordering is a statement
about which call may follow which, and reading it is the only place a precondition is written
down: an instance configuration has a field for a ceiling and no field at all for "this step
runs first".

The grouping is the installer's, not a considered security boundary, and it shows. The reads
that identify the person are in the first step, so nothing that acts can precede them. The
read that says what the customer was actually charged is in the second step beside the actions,
because it is used to decide an amount, which means a refund does not have to follow it.
"""

from __future__ import annotations

from typing import Any


def _tool(name: str) -> Any:
    """A stand-in for a bound connector tool, carrying only the name a step declares."""

    def call(**_arguments: Any) -> str:
        raise NotImplementedError("the reader lists steps and never runs them")

    call.__name__ = name
    return call


def build() -> Any:
    """The installed workflow: understand why they are leaving, then answer it.

    Returns:
        An `agno.workflow.Workflow` whose steps declare their own tools.
    """
    from agno.agent import Agent
    from agno.workflow import Step, Workflow

    return Workflow(
        name="subscription-retention",
        steps=[
            Step(
                name="understand the cancellation",
                agent=Agent(
                    name="understand",
                    tools=[_tool("get_subscription"), _tool("get_customer")],
                    telemetry=False,
                ),
            ),
            Step(
                name="answer it",
                agent=Agent(
                    name="answer",
                    tools=[
                        _tool("get_billing_history"),
                        _tool("apply_retention_discount"),
                        _tool("refund_charge"),
                        _tool("pause_subscription"),
                        _tool("cancel_subscription"),
                        _tool("send_whatsapp"),
                    ],
                    telemetry=False,
                ),
            ),
        ],
        telemetry=False,
        store_events=False,
    )
