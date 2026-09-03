"""Spec validation: what it accepts, and what it must refuse."""

import pytest
from pydantic import ValidationError

from agentred.spec import (
    AgentConfig,
    AgentPolicy,
    AgentSpec,
    ChannelDeclaration,
    CitationRequirement,
    Consequence,
    CumulativeBound,
    DataScope,
    DataSource,
    EnumeratedBound,
    IdempotencyRequirement,
    ImputedBound,
    MatchingBound,
    NumericBound,
    OutboundRule,
    Precondition,
    Provenance,
    RelationalBound,
    ResultCondition,
    ResultReference,
    ToolDeclaration,
    TriggerKind,
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


def config(tools=None, data_sources=(), version="1", model="claude-sonnet-5", channels=()):
    return AgentConfig(
        agent_id="a",
        version=version,
        model=model,
        instructions="be helpful",
        tools=tools
        if tools is not None
        else (tool("apply_discount", Consequence.MONEY, ("pct",)),),
        data_sources=data_sources,
        channels=channels,
    )


DISPUTES = DataSource(name="disputes", identifier_kinds=("dispute_id", "order_id"))


def channel(**kwargs):
    """A planted channel into the buyer's own words, which is the motivating case."""
    return ChannelDeclaration(
        **{
            "name": "dispute_reason_text",
            "writer": "the buyer, when they raise the dispute",
            "data_source": "disputes",
            "record_path": "reason_text",
            "record_key": "dispute_id",
            "trigger": TriggerKind.SCHEDULE,
            **kwargs,
        }
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
            "",
        )

    def test_a_spec_alone_names_no_world(self):
        """A world is not a property of a declaration. A run against one that was never
        generated reads as what it was rather than as a world nobody named."""
        spec = AgentSpec(config=config(), policy=policy())
        assert spec.version_tuple.world_version == ""
        assert "world=" not in str(spec.version_tuple)

    def test_a_different_world_makes_the_agent_untested_again(self):
        """A scorecard computed against one shop says nothing about an agent facing
        another."""
        spec = AgentSpec(config=config(), policy=policy())
        first = spec.version_tuple.model_copy(update={"world_version": "sha256:aaaaaaaaaaaa"})
        second = spec.version_tuple.model_copy(update={"world_version": "sha256:bbbbbbbbbbbb"})
        assert first.as_tuple() != second.as_tuple()
        assert "world=sha256:aaaaaaaaaaaa" in str(first)

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


def text_tool(name, consequence=Consequence.INERT, arguments=("body",)):
    """A tool whose arguments are strings rather than numbers."""
    return ToolDeclaration(
        name=name,
        consequence=consequence,
        parameters={
            "type": "object",
            "properties": {argument: {"type": "string"} for argument in arguments},
        },
    )


class TestCumulativeBounds:
    def test_needs_exactly_one_ceiling(self):
        with pytest.raises(ValidationError, match="exactly one of maximum and maximum_from"):
            CumulativeBound(name="c", tool="apply_discount", argument="pct")

    def test_two_ceilings_are_also_refused(self):
        with pytest.raises(ValidationError, match="exactly one of maximum and maximum_from"):
            CumulativeBound(
                name="c",
                tool="apply_discount",
                argument="pct",
                maximum=10,
                maximum_from=ResultReference(tool="t", field="f"),
            )

    def test_the_ceiling_is_inclusive(self):
        bound = CumulativeBound(name="c", tool="apply_discount", argument="pct", maximum=10)
        assert bound.permits(10, maximum=10)
        assert not bound.permits(10.01, maximum=10)

    def test_an_unresolved_ceiling_does_not_constrain(self):
        """The detector reports that as never evaluated. This must not call it satisfied."""
        bound = CumulativeBound(
            name="c",
            tool="apply_discount",
            argument="pct",
            maximum_from=ResultReference(tool="t", field="f"),
        )
        assert bound.permits(1_000_000, maximum=None)

    def test_the_grouping_arguments_are_checked_against_the_schema(self):
        bound = CumulativeBound(
            name="c", tool="apply_discount", argument="pct", group_by=("order_id",), maximum=10
        )
        with pytest.raises(ValidationError, match="declares no such argument"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))


class TestMatchingBounds:
    def test_a_match_is_case_folded_and_stripped(self):
        bound = MatchingBound(
            name="m",
            tool="apply_discount",
            argument="pct",
            matches=ResultReference(tool="t", field="f"),
        )
        assert bound.permits(" INR ", expected="inr")
        assert not bound.permits("USD", expected="inr")

    def test_an_unresolved_expectation_does_not_constrain(self):
        bound = MatchingBound(
            name="m",
            tool="apply_discount",
            argument="pct",
            matches=ResultReference(tool="t", field="f"),
        )
        assert bound.permits("anything at all", expected=None)

    def test_the_source_tool_is_checked(self):
        bound = MatchingBound(
            name="m",
            tool="apply_discount",
            argument="pct",
            matches=ResultReference(tool="nowhere", field="f"),
        )
        with pytest.raises(ValidationError, match="reads its limit from tool 'nowhere'"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))


class TestImputedBounds:
    def test_needs_a_limit(self):
        with pytest.raises(ValidationError, match="neither maximum nor minimum"):
            ImputedBound(
                name="i", tool="apply_discount", value_from=ResultReference(tool="t", field="f")
            )

    def test_it_constrains_no_argument(self):
        """The one bound with nothing to check against a schema, which validation must allow."""
        bound = ImputedBound(
            name="i",
            tool="apply_discount",
            value_from=ResultReference(tool="verify_order", field="amount"),
            maximum=50,
        )
        spec = AgentSpec(
            config=config(
                tools=(
                    tool("apply_discount", Consequence.MONEY, ("pct",)),
                    tool("verify_order", Consequence.INERT, ("order_id",)),
                )
            ),
            policy=policy(bounds=(bound,)),
        )
        assert spec.bounds_for("apply_discount") == (bound,)
        assert bound.constrained_arguments == ()
        assert bound.argument == ""

    def test_it_cannot_read_its_own_result(self):
        bound = ImputedBound(
            name="i",
            tool="apply_discount",
            value_from=ResultReference(tool="apply_discount", field="pct"),
            maximum=50,
        )
        with pytest.raises(ValidationError, match="the tool it constrains"):
            AgentSpec(config=config(), policy=policy(bounds=(bound,)))


class TestResultConditions:
    def test_needs_exactly_one_form(self):
        with pytest.raises(ValidationError, match="exactly one of equals and equals_any"):
            ResultCondition(field="status")

    def test_two_forms_are_also_refused(self):
        with pytest.raises(ValidationError, match="exactly one of equals and equals_any"):
            ResultCondition(field="status", equals="delivered", equals_any=("delivered",))

    def test_any_of_several_values_counts(self):
        condition = ResultCondition(field="status", equals_any=("delivered", "cancelled"))
        assert condition.met_by({"status": "cancelled"})
        assert not condition.met_by({"status": "in_transit"})

    def test_it_renders_the_set_for_a_verdict(self):
        condition = ResultCondition(field="status", equals_any=("delivered", "cancelled"))
        assert str(condition) == "status in (delivered, cancelled)"

    def test_a_boolean_is_compared_as_text(self):
        assert ResultCondition(field="ok", equals=True).met_by({"ok": True})


class TestMatchedPreconditions:
    def test_the_matched_argument_must_exist_on_both_tools(self):
        precondition = Precondition(
            name="p", tool="issue_refund", requires="verify_order", matched_by=("order_id",)
        )
        tools = (
            tool("issue_refund", Consequence.MONEY, ("amount",)),
            tool("verify_order", Consequence.INERT, ("order_id",)),
        )
        with pytest.raises(ValidationError, match="matches on argument 'order_id'"):
            AgentSpec(config=config(tools=tools), policy=policy(preconditions=(precondition,)))

    def test_it_is_accepted_when_both_declare_it(self):
        precondition = Precondition(
            name="p", tool="issue_refund", requires="verify_order", matched_by=("order_id",)
        )
        tools = (
            tool("issue_refund", Consequence.MONEY, ("amount", "order_id")),
            tool("verify_order", Consequence.INERT, ("order_id",)),
        )
        spec = AgentSpec(config=config(tools=tools), policy=policy(preconditions=(precondition,)))
        assert spec.preconditions_for("issue_refund") == (precondition,)


class TestTheNewerRequirements:
    def test_an_idempotency_requirement_checks_every_argument_it_names(self):
        requirement = IdempotencyRequirement(
            name="once",
            tool="apply_discount",
            identity_arguments=("pct",),
            key_argument="nowhere",
        )
        with pytest.raises(ValidationError, match="declares no such argument"):
            AgentSpec(config=config(), policy=policy(idempotency=(requirement,)))

    def test_an_idempotency_requirement_on_an_undeclared_tool_is_refused(self):
        requirement = IdempotencyRequirement(
            name="once", tool="nowhere", identity_arguments=("pct",)
        )
        with pytest.raises(ValidationError, match="which is not declared"):
            AgentSpec(config=config(), policy=policy(idempotency=(requirement,)))

    def test_an_outbound_rule_checks_its_body_arguments(self):
        rule = OutboundRule(name="out", tool="send", body_arguments=("nowhere",))
        with pytest.raises(ValidationError, match="declares no such argument"):
            AgentSpec(
                config=config(tools=(text_tool("send", Consequence.OBLIGATION),)),
                policy=policy(outbound=(rule,)),
            )

    def test_a_citation_requirement_needs_a_kind_some_source_carries(self):
        requirement = CitationRequirement(
            name="cite",
            tool="send",
            argument="body",
            identifier_kind="order_id",
            source_tools=("read",),
        )
        tools = (
            text_tool("send", Consequence.OBLIGATION),
            text_tool("read", Consequence.DISCLOSURE, ("order_id",)),
        )
        with pytest.raises(ValidationError, match="no declared data source carries"):
            AgentSpec(config=config(tools=tools), policy=policy(citations=(requirement,)))

    def test_a_citation_requirement_cannot_be_its_own_source(self):
        requirement = CitationRequirement(
            name="cite",
            tool="send",
            argument="body",
            identifier_kind="order_id",
            source_tools=("send",),
        )
        sources = (DataSource(name="orders", identifier_kinds=("order_id",)),)
        with pytest.raises(ValidationError, match="the tool it constrains"):
            AgentSpec(
                config=config(
                    tools=(text_tool("send", Consequence.OBLIGATION),), data_sources=sources
                ),
                policy=policy(citations=(requirement,)),
            )

    def test_every_section_counts_as_a_gate(self):
        """A tool a detector can assert something about is not an ungated one."""
        tools = (
            text_tool("send", Consequence.OBLIGATION),
            tool("issue_refund", Consequence.MONEY, ("amount",)),
        )
        spec = AgentSpec(
            config=config(tools=tools),
            policy=policy(
                outbound=(OutboundRule(name="out", tool="send", body_arguments=("body",)),)
            ),
        )
        assert [t.name for t in spec.ungated_consequential_tools()] == ["issue_refund"]

    def test_a_repeated_name_within_a_section_is_refused(self):
        rule = OutboundRule(name="out", tool="send", body_arguments=("body",))
        with pytest.raises(ValidationError, match="duplicate outbound rule name"):
            AgentPolicy(agent_id="a", version="1", outbound=(rule, rule))

    def test_every_section_is_carried_into_the_provenance_report(self):
        rule = OutboundRule(
            name="out", tool="send", body_arguments=("body",), provenance=Provenance.INFERRED
        )
        assert not AgentPolicy(agent_id="a", version="1", outbound=(rule,)).is_fully_declared


class TestChannels:
    """What an agent declares about the fields an adversary writes (ADR-0006)."""

    def test_a_declared_channel_is_reachable_by_name(self):
        spec = config(data_sources=(DISPUTES,), channels=(channel(),))
        assert spec.channels_by_name["dispute_reason_text"].record_path == "reason_text"

    def test_an_agent_declaring_no_channel_is_accepted(self):
        assert config().channels == ()

    def test_a_channel_into_a_source_the_agent_cannot_reach_is_refused(self):
        with pytest.raises(ValidationError, match="which the agent cannot reach"):
            config(channels=(channel(),))

    def test_a_channel_keyed_by_an_identifier_the_source_lacks_is_refused(self):
        with pytest.raises(ValidationError, match="does not carry"):
            config(data_sources=(DISPUTES,), channels=(channel(record_key="cart_id"),))

    def test_two_channels_of_the_same_name_are_refused(self):
        one = channel()
        with pytest.raises(ValidationError, match="duplicate channel name"):
            config(data_sources=(DISPUTES,), channels=(one, one))

    def test_the_conversational_channel_name_cannot_be_declared(self):
        with pytest.raises(ValidationError, match="implicit conversational channel"):
            channel(name="conversation")

    def test_a_record_path_that_traverses_structure_is_refused(self):
        with pytest.raises(ValidationError, match="traverses structure"):
            channel(record_path="lines[0].note")

    def test_a_request_trigger_without_a_template_is_refused(self):
        with pytest.raises(ValidationError, match="declares no trigger_template"):
            channel(trigger=TriggerKind.REQUEST)

    def test_a_request_template_naming_no_record_is_refused(self):
        with pytest.raises(ValidationError, match=r"no \{record\} in it"):
            channel(trigger=TriggerKind.REQUEST, trigger_template="Deal with this dispute.")

    def test_a_schedule_trigger_carrying_a_template_is_refused(self):
        with pytest.raises(ValidationError, match="which nothing would send"):
            channel(trigger_template="Deal with dispute {record}.")

    def test_a_request_trigger_with_a_usable_template_is_accepted(self):
        declared = channel(
            trigger=TriggerKind.REQUEST, trigger_template="Please deal with dispute {record}."
        )
        assert declared.trigger_template.format(record="DSP-9001").endswith("DSP-9001.")

    def test_declaring_a_channel_does_not_change_the_tool_digest(self):
        """A scorecard cites the tool digest, and channels are not tools.

        The same argument that kept `unit_symbol` off `ToolDeclaration`. A field that
        changed the digest would invalidate every scorecard already produced for an agent
        whose tools had not moved.
        """
        plain = config(data_sources=(DISPUTES,))
        declared = config(data_sources=(DISPUTES,), channels=(channel(),))
        assert plain.tool_version == declared.tool_version
