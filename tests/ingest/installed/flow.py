"""One agent's workflow, shaped the way a builder's installer maps a skill onto steps.

Not a target and not a fixture of the harness's own workflows, which are shaped around where
a model is invoked rather than around the order the business requires. This is the other
shape: named steps, each carrying the tools it is allowed at that point, ordered so that the
step reading a record runs before the step spending against it. That ordering is the thing
the workflow reader recovers, and a workflow shaped like the harness's own would not have it
to recover.
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
    """The installed workflow: read the case, check the order, then settle it.

    Returns:
        An `agno.workflow.Workflow` whose steps declare their own tools, so that the order
        between them is a statement about which call may follow which.
    """
    from agno.agent import Agent
    from agno.workflow import Step, Workflow

    return Workflow(
        name="dispute-responder",
        steps=[
            Step(
                name="read the case",
                agent=Agent(
                    name="read",
                    tools=[_tool("get_dispute"), _tool("get_customer")],
                    telemetry=False,
                ),
            ),
            Step(
                name="check the order",
                agent=Agent(
                    name="check",
                    tools=[_tool("get_order"), _tool("get_shipment")],
                    telemetry=False,
                ),
            ),
            Step(
                name="settle",
                agent=Agent(
                    name="settle",
                    tools=[
                        _tool("issue_refund"),
                        _tool("accept_dispute"),
                        _tool("apply_discount"),
                        _tool("submit_evidence"),
                        _tool("send_whatsapp"),
                    ],
                    telemetry=False,
                ),
            ),
        ],
        telemetry=False,
        store_events=False,
    )
