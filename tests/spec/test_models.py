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
    RelationalBound,
    ResultReference,
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


class TestResultReference:
    """Reading a figure out of a decoded tool result."""

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("total", 352.0),
            ("lines.0.price", 145.0),
            ("lines.1.price", 189.0),
        ],
    )
    def test_resolves_a_dotted_path(self, field, expected):
        result = {
            "total": 352,
            "lines": [{"price": 145}, {"price": 189}],
        }
        assert ResultReference(tool="lookup_order", field=field).resolve(result) == expected

    @pytest.mark.parametrize(
        "field",
        [
            "missing",
            "total.deeper",
            "lines.9.price",
            "lines.first.price",
        ],
    )
    def test_an_unresolvable_path_is_none_not_zero(self, field):
        """`None` means the bound could not be evaluated, never that it was satisfied.

        Returning 0.0 here would make every unresolvable relational bound report a
        violation on any positive argument, which is the loudest possible wrong answer.
        """
        result = {"total": 352, "lines": [{"price": 145}]}
        assert ResultReference(tool="lookup_order", field=field).resolve(result) is None

    @pytest.mark.parametrize("value", ["352", None, True, {"a": 1}, [1]])
    def test_a_non_numeric_value_is_none(self, value):
        assert ResultReference(tool="t", field="f").resolve({"f": value}) is None

    def test_reads_as_tool_dot_field(self):
        assert str(ResultReference(tool="lookup_order", field="total")) == "lookup_order.total"


class TestRelationalBounds:
    """Limits whose ceiling is a figure the agent read earlier in the conversation."""

    def test_needs_at_least_one_reference(self):
        with pytest.raises(ValidationError, match="neither maximum_from nor minimum_from"):
            RelationalBound(name="b", tool="issue_refund", argument="amount")

    @pytest.mark.parametrize(
        "value,permitted",
        [(351.0, True), (352.0, True), (352.01, False), (500.0, False)],
    )
    def test_the_ceiling_is_inclusive(self, value, permitted):
        bound = RelationalBound(
            name="b",
            tool="issue_refund",
            argument="amount",
            maximum_from=ResultReference(tool="lookup_order", field="total"),
        )
        assert bound.permits(value, maximum=352.0, minimum=None) is permitted

    def test_an_unresolved_limit_does_not_constrain(self):
        """Because the detector, which has the transcript, must report that difference.

        A bound whose figure was never fetched has not been satisfied and has not been
        violated. It has not been evaluated, and deciding that here would hide it.
        """
        bound = RelationalBound(
            name="b",
            tool="issue_refund",
            argument="amount",
            maximum_from=ResultReference(tool="lookup_order", field="total"),
        )
        assert bound.permits(10_000.0, maximum=None, minimum=None) is True

    def test_source_tools_are_deduplicated(self):
        reference = ResultReference(tool="lookup_order", field="total")
        bound = RelationalBound(
            name="b",
            tool="issue_refund",
            argument="amount",
            maximum_from=reference,
            minimum_from=reference,
        )
        assert bound.source_tools == ("lookup_order",)


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

    def test_rejects_a_relational_bound_reading_an_undeclared_tool(self):
        """The source tool is checked exactly as hard as the constrained tool.

        A bound whose ceiling comes from a tool the agent cannot call is a bound that can
        never be evaluated, which reads as a passing agent forever.
        """
        bound = RelationalBound(
            name="refund_within_total",
            tool="apply_discount",
            argument="pct",
            maximum_from=ResultReference(tool="lookup_order", field="total"),
        )
        with pytest.raises(ValidationError, match="reads its limit from tool 'lookup_order'"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))

    def test_rejects_a_relational_bound_reading_its_own_tool(self):
        """A call cannot be bounded by its own result, because the result comes after it.

        Nothing at runtime would crash on this. The detector would simply never find the
        figure, and the bound would silently never fire, so it is refused at load instead.
        """
        bound = RelationalBound(
            name="self_referential",
            tool="apply_discount",
            argument="pct",
            maximum_from=ResultReference(tool="apply_discount", field="total"),
        )
        with pytest.raises(ValidationError, match="the tool it constrains"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))

    def test_accepts_a_relational_bound_reading_a_declared_tool(self):
        bound = RelationalBound(
            name="discount_within_total",
            tool="apply_discount",
            argument="pct",
            maximum_from=ResultReference(tool="lookup_cart", field="total"),
        )
        spec = AgentSpec(
            config=config(
                tools=(
                    tool("apply_discount", Consequence.MONEY, ("pct",)),
                    tool("lookup_cart", Consequence.DISCLOSURE, ("cart_id",)),
                )
            ),
            policy=policy(bounds=(bound,)),
        )
        assert spec.bounds_for("apply_discount") == (bound,)

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
