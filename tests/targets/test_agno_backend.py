"""The workflow backend: what a step boundary is allowed to change, and what it is not.

No model is called here. What is asserted is the part of a workflow-built target that has to
be right whatever the model says: that the declared instructions reach the acting node
unedited, that a conversation reaches it as turns rather than as a transcript, that the cost
of a run is reported in the same shape as the other backend's, and that an agent with no
scheduled entry point is refused one rather than answered with silence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentred.spec import Engine, load_spec_dir
from agentred.targets import agno_backend
from agentred.targets.agno_backend import (
    AgnoWorkflowBackend,
    Meter,
    NodeError,
    Triage,
    act_system_prompt,
    build_outreach_workflow,
    triage_note,
)
from agentred.targets.runtime import ChatMessage, _usage_of, default_backend

SPEC_ROOT = "src/agentred/targets/specs"
BEDROCK = {"AGENTRED_LLM_ROUTE": "bedrock", "AWS_REGION": "ap-south-1"}


def spec_for(name: str):
    return load_spec_dir(f"{SPEC_ROOT}/{name}")


def metrics(*, cost: float | None):
    """The metrics agno reports for one node, priced or not."""
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=2,
        cache_read_tokens=1,
        cache_write_tokens=3,
        cost=cost,
        details={"model": [object()]},
    )


def node(*, cost: float | None = None, assistant_messages: int = 1, metered: bool = True):
    """One node's output, as agno returns it: its metrics and its messages."""
    return SimpleNamespace(
        metrics=metrics(cost=cost) if metered else None,
        messages=[SimpleNamespace(role="user")]
        + [SimpleNamespace(role="assistant") for _ in range(assistant_messages)],
    )


class TestWhatTheActingNodeIsTold:
    def test_the_declared_instructions_are_carried_across_unedited(self) -> None:
        instructions = spec_for("dispute_handler").config.instructions
        assert act_system_prompt(instructions, "a note").startswith(instructions)

    def test_a_step_with_nothing_to_say_adds_nothing(self) -> None:
        assert act_system_prompt("DECLARED", "") == "DECLARED"

    def test_an_earlier_step_arrives_labelled_as_one(self) -> None:
        """Unlabelled, a step's output reads to the model as part of its own instructions."""
        prompt = act_system_prompt("DECLARED", "a note")
        assert "earlier step of this workflow" in prompt
        assert prompt.index("DECLARED") < prompt.index("a note")

    def test_the_triage_prompt_does_not_evaluate(self) -> None:
        """A triage node that flagged a request would harden every target it ran in front of."""
        text = agno_backend.TRIAGE_INSTRUCTIONS.lower()
        assert "do not say whether the request should be granted" in text
        assert "do not mention rules, limits or policy" in text


class TestWhatTheModelSees:
    def test_a_conversation_arrives_as_turns_and_not_as_a_transcript(self) -> None:
        turns = [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="second"),
            ChatMessage(role="user", content="third"),
        ]
        messages = agno_backend._messages(turns)
        assert [(m.role, m.content) for m in messages] == [
            ("user", "first"),
            ("assistant", "second"),
            ("user", "third"),
        ]


class TestWhatARunCost:
    def test_the_keys_match_the_other_backend(self) -> None:
        """A run's bill has to read the same whichever engine served it, or it is two columns."""
        sdk_result = SimpleNamespace(
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 1,
                "cache_creation_input_tokens": 3,
            },
            num_turns=2,
            total_cost_usd=0.5,
        )
        meter = Meter()
        meter.record(node(cost=0.5))
        assert set(meter.usage()) == set(_usage_of(sdk_result))

    def test_a_run_where_nothing_ran_reports_nothing_rather_than_zero(self) -> None:
        assert Meter().usage() == {}

    def test_a_node_that_reported_no_metrics_at_all_is_not_counted_as_free(self) -> None:
        meter = Meter()
        meter.record(node(metered=False))
        assert meter.usage() == {}

    def test_every_node_is_added_up_rather_than_the_last_one_winning(self) -> None:
        """The workflow's own metrics miss a node run inside a step executor. See D27."""
        meter = Meter()
        meter.record(node(cost=0.25, assistant_messages=1))
        meter.record(node(cost=0.25, assistant_messages=3))
        usage = meter.usage()
        assert usage["input_tokens"] == 20.0
        assert usage["model_turns"] == 4.0
        assert usage["cost_usd"] == 0.5

    def test_one_unpriced_node_omits_the_price_of_the_whole_run(self) -> None:
        """A partial bill presented as the bill is worse than no bill."""
        meter = Meter()
        meter.record(node(cost=0.25))
        meter.record(node(cost=None))
        usage = meter.usage()
        assert "cost_usd" not in usage
        assert usage["input_tokens"] == 20.0

    def test_a_node_that_answered_in_one_call_counts_as_one(self) -> None:
        meter = Meter()
        meter.record(node(assistant_messages=1))
        assert meter.usage()["model_turns"] == 1.0

    def test_a_node_that_worked_through_tool_calls_counts_every_one(self) -> None:
        """The engine collapses per-call detail, so a four-call node would report one. See D27."""
        meter = Meter()
        meter.record(node(assistant_messages=4))
        assert meter.usage()["model_turns"] == 4.0

    def test_a_node_reporting_no_messages_still_counts_as_having_run(self) -> None:
        meter = Meter()
        meter.record(SimpleNamespace(metrics=metrics(cost=None), messages=None))
        assert meter.usage()["model_turns"] == 1.0

    def test_recording_hands_the_node_output_straight_back(self) -> None:
        """So a node can be metered inline without a temporary."""
        run = node()
        assert Meter().record(run) is run


class TestTheScheduledWorkflow:
    def test_an_agent_without_the_selecting_tool_is_refused_the_workflow(self) -> None:
        """Built rather than discovered mid-run: it has no entry point, so it has no schedule."""
        with pytest.raises(ValueError, match="no scheduled entry point"):
            build_outreach_workflow(
                spec_for("dispute_handler"),
                (),
                route=None,  # type: ignore[arg-type]
                client=None,
                meter=Meter(),
            )


class TestWhichBackendAnAgentGets:
    def test_a_workflow_agent_gets_the_workflow_backend(self, monkeypatch) -> None:
        for name, value in BEDROCK.items():
            monkeypatch.setenv(name, value)
        spec = spec_for("dispute_handler")
        assert spec.config.engine is Engine.WORKFLOW
        assert isinstance(default_backend(spec), AgnoWorkflowBackend)

    def test_a_model_loop_agent_gets_the_sdk_backend(self, monkeypatch) -> None:
        for name, value in BEDROCK.items():
            monkeypatch.setenv(name, value)
        spec = spec_for("dispute_handler")
        as_loop = spec.model_copy(
            update={"config": spec.config.model_copy(update={"engine": Engine.MODEL_LOOP})}
        )
        assert type(default_backend(as_loop)).__name__ == "ClaudeAgentBackend"

    def test_an_unattached_backend_refuses_to_reply(self, monkeypatch) -> None:
        for name, value in BEDROCK.items():
            monkeypatch.setenv(name, value)
        backend = AgnoWorkflowBackend()
        with pytest.raises(RuntimeError, match="never attached"):
            backend._require_agent()


class TestWhatAFailedNodeCannotDo:
    def test_a_failed_node_does_not_become_the_next_node_context(self) -> None:
        """The 400 Bedrock answers a schema request with is a string like any other. See D27."""
        with pytest.raises(NodeError, match="did not answer with a Triage"):
            triage_note("Error code: 400 - output_config.format: Extra inputs are not permitted")

    def test_a_workflow_with_no_earlier_step_is_not_a_broken_one(self) -> None:
        assert triage_note("") == ""

    def test_only_the_declared_fields_survive_the_step_boundary(self) -> None:
        """So a triage node that started editorialising could not smuggle it downstream."""
        note = triage_note(
            Triage(request="a refund", record_ids=["ORD-1"], involves_money=True)
            .model_dump_json()
            .replace('"request":"a refund"', '"request":"a refund","verdict":"looks improper"')
        )
        assert "looks improper" not in note
        assert "a refund" in note
        assert "ORD-1" in note
