"""Spec validation: what it accepts, and what it must refuse."""

import pytest
from pydantic import ValidationError

from agentred.spec import (
    AgentConfig,
    AgentPolicy,
    AgentSpec,
    Consequence,
    DataScope,
    DataSource,
    EnumeratedBound,
    NumericBound,
    Precondition,
    Provenance,
    ToolDeclaration,
)


def tool(name, consequence=Consequence.INERT, arguments=("amount",)):
    return ToolDeclaration(
        name=name,
        consequence=consequence,
        parameters={
            "type": "object",
            "properties": {argument: {"type": "number"} for argument in arguments},
        },
    )


def config(tools=None, data_sources=(), version="1", model="claude-sonnet-5"):
    return AgentConfig(
        agent_id="a",
        version=version,
        model=model,
        instructions="be helpful",
        tools=tools
        if tools is not None
        else (tool("apply_discount", Consequence.MONEY, ("pct",)),),
        data_sources=data_sources,
    )


def policy(**kwargs):
    return AgentPolicy(agent_id="a", version="1", **kwargs)


DISCOUNT_CEILING = NumericBound(
    name="discount_ceiling", tool="apply_discount", argument="pct", maximum=10
)


class TestBounds:
    def test_numeric_bound_needs_a_limit(self):
        with pytest.raises(ValidationError, match="neither maximum nor minimum"):
            NumericBound(name="b", tool="t", argument="a")

    def test_numeric_bound_rejects_inverted_range(self):
        with pytest.raises(ValidationError, match="below minimum"):
            NumericBound(name="b", tool="t", argument="a", maximum=1, minimum=5)

    @pytest.mark.parametrize(
        ("value", "permitted"), [(9, True), (10, True), (10.01, False), (35, False)]
    )
    def test_numeric_bound_is_inclusive(self, value, permitted):
        assert DISCOUNT_CEILING.permits(value) is permitted

    def test_enumerated_bound_compares_as_strings(self):
        bound = EnumeratedBound(
            name="reasons", tool="t", argument="reason", allowed_values=["damaged", 2]
        )
        assert bound.allowed_values == ("damaged", "2")
        assert bound.permits(2)
        assert not bound.permits("late")

    def test_enumerated_bound_needs_values(self):
        with pytest.raises(ValidationError):
            EnumeratedBound(name="b", tool="t", argument="a", allowed_values=[])


class TestPrecondition:
    def test_tool_cannot_gate_itself(self):
        with pytest.raises(ValidationError, match="precede itself"):
            Precondition(name="p", tool="refund", requires="refund")


class TestConfig:
    def test_rejects_duplicate_tool_names(self):
        with pytest.raises(ValidationError, match="duplicate tool name"):
            config(tools=(tool("refund"), tool("refund")))

    def test_rejects_duplicate_data_source_names(self):
        with pytest.raises(ValidationError, match="duplicate data source name"):
            config(data_sources=(DataSource(name="orders"), DataSource(name="orders")))

    def test_consequential_tools_excludes_inert(self):
        spec_config = config(
            tools=(
                tool("lookup"),
                tool("refund", Consequence.MONEY),
                tool("order", Consequence.OBLIGATION),
            )
        )
        assert [t.name for t in spec_config.consequential_tools] == ["refund", "order"]

    def test_tool_version_is_stable_across_reordering(self):
        a, b = tool("lookup"), tool("refund", Consequence.MONEY)
        assert config(tools=(a, b)).tool_version == config(tools=(b, a)).tool_version

    def test_tool_version_changes_with_consequence(self):
        inert = config(tools=(tool("refund", Consequence.INERT),))
        money = config(tools=(tool("refund", Consequence.MONEY),))
        assert inert.tool_version != money.tool_version

    def test_argument_names_tolerates_a_schemaless_tool(self):
        assert (
            ToolDeclaration(name="t", consequence=Consequence.INERT).argument_names == frozenset()
        )


class TestSpecCrossValidation:
    def test_accepts_a_policy_that_describes_its_config(self):
        spec = AgentSpec(config=config(), policy=policy(bounds=(DISCOUNT_CEILING,)))
        assert spec.bounds_for("apply_discount") == (DISCOUNT_CEILING,)

    def test_rejects_mismatched_agent_id(self):
        with pytest.raises(ValidationError, match="policy is for agent"):
            AgentSpec(config=config(), policy=AgentPolicy(agent_id="other", version="1"))

    def test_rejects_a_bound_on_an_undeclared_tool(self):
        bound = NumericBound(name="b", tool="issue_refund", argument="pct", maximum=10)
        with pytest.raises(ValidationError, match="which is not declared"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))

    def test_rejects_a_bound_on_an_undeclared_argument(self):
        bound = NumericBound(name="b", tool="apply_discount", argument="amount", maximum=10)
        with pytest.raises(ValidationError, match="declares no such argument"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))

    def test_rejects_a_precondition_on_an_undeclared_tool(self):
        pre = Precondition(name="p", tool="apply_discount", requires="verify_order")
        with pytest.raises(ValidationError, match="'verify_order' as its requires"):
            AgentSpec(config=config(), policy=policy(preconditions=(pre,)))

    def test_rejects_a_scope_on_an_unreachable_source(self):
        scope = DataScope(sources=("crm",))
        with pytest.raises(ValidationError, match="which the agent cannot reach"):
            AgentSpec(config=config(), policy=policy(data_scope=scope))


class TestProvenance:
    def test_fully_declared_by_default(self):
        assert policy(bounds=(DISCOUNT_CEILING,)).is_fully_declared

    def test_one_inferred_statement_taints_the_policy(self):
        inferred = NumericBound(
            name="b",
            tool="apply_discount",
            argument="pct",
            maximum=10,
            provenance=Provenance.INFERRED,
        )
        assert not policy(bounds=(inferred,)).is_fully_declared

    def test_an_inferred_data_scope_taints_the_policy(self):
        assert not policy(data_scope=DataScope(provenance=Provenance.INFERRED)).is_fully_declared


class TestVersionTuple:
    def test_is_constructible_from_a_loaded_spec(self):
        spec = AgentSpec(config=config(version="7", model="claude-opus-5"), policy=policy())
        versions = spec.version_tuple
        assert versions.config_version == "7"
        assert versions.policy_version == "1"
        assert versions.model_version == "claude-opus-5"
        assert versions.tool_version.startswith("sha256:")
        assert versions.as_tuple() == (
            "7",
            "1",
            "claude-opus-5",
            spec.config.tool_version,
        )

    def test_a_model_upgrade_changes_the_tuple(self):
        first = AgentSpec(config=config(model="claude-sonnet-5"), policy=policy())
        second = AgentSpec(config=config(model="claude-opus-5"), policy=policy())
        assert first.version_tuple.as_tuple() != second.version_tuple.as_tuple()


class TestUngatedConsequentialTools:
    def test_a_bound_or_a_precondition_counts_as_a_gate(self):
        tools = (
            tool("apply_discount", Consequence.MONEY, ("pct",)),
            tool("issue_refund", Consequence.MONEY, ("amount",)),
            tool("verify_order", Consequence.INERT, ("order_id",)),
            tool("promise_delivery", Consequence.OBLIGATION, ("date",)),
        )
        spec = AgentSpec(
            config=config(tools=tools),
            policy=policy(
                bounds=(DISCOUNT_CEILING,),
                preconditions=(
                    Precondition(name="p", tool="issue_refund", requires="verify_order"),
                ),
            ),
        )
        assert [t.name for t in spec.ungated_consequential_tools()] == ["promise_delivery"]
