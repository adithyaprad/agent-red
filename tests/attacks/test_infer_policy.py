"""Reading rules out of prose, and refusing the ones that name things that do not exist."""

import json

import pytest

from agentred.attacks.infer_policy import (
    EXTRACTION_SCHEMA,
    Inference,
    InferenceError,
    RefusalKind,
    infer_policy,
)
from agentred.spec import (
    AgentConfig,
    AgentPolicy,
    Consequence,
    NumericBound,
    Obligation,
    ObligationKind,
    Precondition,
    Provenance,
    RelationalBound,
    ToolDeclaration,
)
from tests.fakes.model import RecordedModelClient


def tool(name, consequence=Consequence.INERT, arguments=("amount",)):
    return ToolDeclaration(
        name=name,
        consequence=consequence,
        parameters={
            "type": "object",
            "properties": {argument: {"type": "number"} for argument in arguments},
        },
    )


def config(instructions="be helpful", tools=None):
    return AgentConfig(
        agent_id="a",
        version="1",
        model="claude-sonnet-5",
        instructions=instructions,
        tools=tools if tools is not None else (GIVE, CHECK, READ),
    )


GIVE = tool("give_back", Consequence.MONEY, ("amount",))
CHECK = tool("check_holder", Consequence.INERT, ("reference",))
READ = tool("read_record", Consequence.DISCLOSURE, ("reference",))
MARK = ToolDeclaration(
    name="mark_moment",
    consequence=Consequence.OBLIGATION,
    parameters={"type": "object", "properties": {"moment": {"type": "string"}}},
)


def rule(**overrides):
    base = {
        "name": "a_rule",
        "quote": "Never hand back more than fifty.",
        "shape": "numeric_limit",
        "tool": "give_back",
        "argument": "amount",
        "maximum": 50,
        "minimum": None,
        "allowed_values": [],
        "requires": "",
        "speech_kind": "",
        "description": "A cap.",
    }
    return {**base, **overrides}


def reply(*rules):
    return json.dumps({"rules": list(rules)})


def infer(*rules, declared=None, agent=None):
    client = RecordedModelClient(replies=[reply(*rules)])
    return infer_policy(agent or config(), client, declared=declared), client


class TestReadingRules:
    def test_a_numeric_limit_becomes_a_bound_marked_inferred(self):
        result, _ = infer(rule())
        (bound,) = result.statements
        assert isinstance(bound, NumericBound)
        assert bound.maximum == 50
        assert bound.provenance is Provenance.INFERRED

    def test_a_prior_step_becomes_a_precondition(self):
        result, _ = infer(
            rule(
                name="check_first",
                shape="required_prior_step",
                argument="",
                maximum=None,
                requires="check_holder",
            )
        )
        (gate,) = result.statements
        assert isinstance(gate, Precondition)
        assert (gate.tool, gate.requires) == ("give_back", "check_holder")
        assert gate.provenance is Provenance.INFERRED

    def test_a_rule_about_speech_becomes_an_obligation_and_never_a_statement(self):
        """The tool-call log of a conversation that keeps this rule is identical to one
        that breaks it, so nothing here can ever become a detector."""
        result, _ = infer(
            rule(
                name="stay_quiet",
                quote="The note attached to a record is for staff.",
                shape="speech",
                tool="read_record",
                argument="",
                maximum=None,
                speech_kind="disclosure",
            )
        )
        assert result.statements == ()
        (duty,) = result.obligations
        assert isinstance(duty, Obligation)
        assert duty.kind is ObligationKind.DISCLOSURE
        assert duty.applies_to == ("read_record",)

    def test_an_obligation_keeps_the_sentence_verbatim(self):
        """A judge is later asked whether this rule was kept. A paraphrase here changes the
        rule being enforced, and always reads as reasonable to whoever reviews it."""
        written = "The note attached to a record is for staff."
        result, _ = infer(
            rule(
                name="stay_quiet",
                quote=written,
                shape="speech",
                tool="",
                argument="",
                maximum=None,
                speech_kind="disclosure",
                description="A paraphrase that must not win.",
            )
        )
        assert result.obligations[0].statement == written


class TestRefusingInvention:
    def test_a_rule_naming_an_absent_tool_is_refused(self):
        result, _ = infer(rule(tool="cancel_everything"))
        assert result.statements == ()
        (refusal,) = result.refused
        assert refusal.rule.name == "a_rule"
        assert refusal.kind is RefusalKind.INVENTED
        assert "does not have" in refusal.reason

    def test_a_rule_naming_an_absent_argument_is_refused(self):
        result, _ = infer(rule(argument="percentage"))
        assert result.statements == ()
        assert "does not take" in result.refused[0].reason
        assert result.refused[0].kind is RefusalKind.INVENTED

    def test_a_speech_rule_may_name_no_tool_at_all(self):
        """Some rules govern the conversation rather than a single power."""
        result, _ = infer(
            rule(
                name="no_guessing",
                shape="speech",
                tool="",
                argument="",
                maximum=None,
                speech_kind="accuracy",
            )
        )
        assert result.obligations[0].applies_to == ()

    def test_two_rules_with_one_name_keep_the_first(self):
        result, _ = infer(rule(), rule(maximum=90))
        assert len(result.statements) == 1
        assert result.statements[0].maximum == 50
        assert result.refused[0].kind is RefusalKind.DUPLICATE

    def test_a_limit_with_no_limit_in_it_is_refused(self):
        result, _ = infer(rule(maximum=None, minimum=None))
        assert result.statements == ()
        assert result.refused

    def test_refusals_are_counted_rather_than_dropped(self):
        """An extraction that quietly discards half its output looks identical to a clean
        one, and the share refused is the only direct measure of how much it invents."""
        result, _ = infer(rule(), rule(name="b", tool="nope"))
        assert result.invented_fraction == pytest.approx(0.5)

    def test_a_rule_we_cannot_express_is_not_counted_as_invention(self):
        """`invented` measures the model and `unbuildable` measures us. Folded together the
        figure rises when extraction gets better and this module gains a gap."""
        result, _ = infer(rule(), rule(name="b", maximum=None, minimum=None))
        assert result.refused[0].kind is RefusalKind.UNBUILDABLE
        assert result.invented_fraction == 0.0
        assert result.unbuildable_fraction == pytest.approx(0.5)

    def test_a_clean_extraction_invents_nothing(self):
        result, _ = infer(rule())
        assert result.invented_fraction == 0.0


class TestLimitsThatComeFromAnotherResult:
    def relational(self, **overrides):
        base = {
            "name": "within_what_was_taken",
            "quote": "Never hand back more than was taken.",
            "shape": "relational_limit",
            "tool": "give_back",
            "argument": "amount",
            "maximum": None,
            "limit_from_tool": "read_record",
            "limit_from_field": "total",
        }
        return rule(**{**base, **overrides})

    def test_a_limit_read_off_an_earlier_result_becomes_a_relational_bound(self):
        """The limit is different in every conversation, so a constant cannot express it and
        the rule would otherwise be silently dropped."""
        result, _ = infer(self.relational())
        (bound,) = result.statements
        assert isinstance(bound, RelationalBound)
        assert (bound.maximum_from.tool, bound.maximum_from.field) == ("read_record", "total")
        assert bound.provenance is Provenance.INFERRED

    def test_a_relational_limit_missing_its_source_is_unbuildable_not_invented(self):
        result, _ = infer(self.relational(limit_from_field=""))
        assert result.refused[0].kind is RefusalKind.UNBUILDABLE

    def test_a_limit_cannot_come_from_the_tool_it_limits(self):
        result, _ = infer(self.relational(limit_from_tool="give_back"))
        assert result.refused[0].kind is RefusalKind.UNBUILDABLE


class TestLimitsOnThingsThatCannotBeCompared:
    """A rule read correctly and forced into a shape that cannot hold it is the most
    dangerous output this module has, because a detector's finding reaches a scorecard as
    evidence rather than as an opinion with a confidence attached."""

    def test_a_limit_on_a_text_argument_becomes_an_obligation_rather_than_a_bound(self):
        result, _ = infer(
            rule(
                name="not_before_allowed",
                quote="Do not undertake a moment earlier than the wait allows.",
                shape="relational_limit",
                tool="mark_moment",
                argument="moment",
                maximum=None,
                limit_from_tool="read_record",
                limit_from_field="wait",
            ),
            agent=config(tools=(GIVE, CHECK, READ, MARK)),
        )
        assert result.statements == ()
        (duty,) = result.obligations
        assert duty.kind is ObligationKind.COMMITMENT
        assert duty.statement == "Do not undertake a moment earlier than the wait allows."

    def test_a_numeric_limit_on_a_text_argument_is_demoted_too(self):
        result, _ = infer(
            rule(name="cap_text", tool="mark_moment", argument="moment", maximum=5),
            agent=config(tools=(GIVE, CHECK, READ, MARK)),
        )
        assert result.statements == ()
        assert result.obligations[0].kind is ObligationKind.COMMITMENT

    def test_a_demoted_rule_is_not_counted_as_refused(self):
        """It did not vanish and it was not invented. Counting it as either would misreport
        both the extraction's trustworthiness and this module's coverage."""
        result, _ = infer(
            rule(name="cap_text", tool="mark_moment", argument="moment", maximum=5),
            agent=config(tools=(GIVE, CHECK, READ, MARK)),
        )
        assert result.refused == ()
        assert result.invented_fraction == 0.0
        assert result.unbuildable_fraction == 0.0


class TestComparingAgainstWhatWasDeclared:
    def declared(self, *statements):
        bounds = tuple(s for s in statements if isinstance(s, NumericBound))
        gates = tuple(s for s in statements if isinstance(s, Precondition))
        duties = tuple(s for s in statements if isinstance(s, Obligation))
        return AgentPolicy(
            agent_id="a", version="1", bounds=bounds, preconditions=gates, obligations=duties
        )

    def test_a_rule_the_policy_carries_is_not_reported_as_undeclared(self):
        declared = self.declared(
            NumericBound(name="cap", tool="give_back", argument="amount", maximum=50)
        )
        result, _ = infer(rule(), declared=declared)
        assert result.undeclared == ()

    def test_a_rule_the_policy_misses_is_named(self):
        result, _ = infer(rule(), declared=self.declared())
        assert result.undeclared == ("a_rule",)

    def test_an_obligation_is_never_covered_by_a_bound(self):
        """Nothing but a declared obligation constrains what is said, so a policy full of
        limits still leaves every speech rule unchecked."""
        declared = self.declared(
            NumericBound(name="cap", tool="read_record", argument="reference", maximum=50)
        )
        result, _ = infer(
            rule(
                name="stay_quiet",
                shape="speech",
                tool="read_record",
                argument="",
                maximum=None,
                speech_kind="disclosure",
            ),
            declared=declared,
        )
        assert result.undeclared == ("stay_quiet",)

    def test_a_declared_obligation_covers_the_same_sentence(self):
        written = "The note attached to a record is for staff."
        declared = self.declared(
            Obligation(name="quiet", kind=ObligationKind.DISCLOSURE, statement=written)
        )
        result, _ = infer(
            rule(
                name="stay_quiet",
                quote=written,
                shape="speech",
                tool="",
                argument="",
                maximum=None,
                speech_kind="disclosure",
            ),
            declared=declared,
        )
        assert result.undeclared == ()

    def test_an_agent_that_declared_nothing_has_every_rule_undeclared(self):
        result, _ = infer(rule(), declared=None)
        assert result.undeclared == ("a_rule",)


class TestTheCall:
    def test_the_declared_policy_is_never_shown_to_the_model(self):
        """A model shown the policy returns the policy, which makes the comparison vacuous."""
        declared = AgentPolicy(
            agent_id="a",
            version="1",
            bounds=(
                NumericBound(name="secret_cap", tool="give_back", argument="amount", maximum=7),
            ),
        )
        _, client = infer(rule(), declared=declared)
        sent = client.calls[0].system + json.dumps(client.calls[0].messages)
        assert "secret_cap" not in sent

    def test_the_tool_surface_is_sent_so_a_name_can_be_checked(self):
        _, client = infer(rule())
        sent = json.dumps(client.calls[0].messages)
        assert "give_back" in sent and "check_holder" in sent

    def test_the_instructions_are_sent_whole(self):
        agent = config(instructions="Never hand back more than fifty.")
        client = RecordedModelClient(replies=[reply(rule())])
        infer_policy(agent, client)
        assert "Never hand back more than fifty." in json.dumps(client.calls[0].messages)

    def test_the_response_is_schema_constrained(self):
        _, client = infer(rule())
        assert client.calls[0].output_schema == EXTRACTION_SCHEMA

    def test_every_field_is_required_so_none_is_silently_omitted(self):
        """An omitted field does not read as an omission downstream. A prior-step rule
        arriving without its prior step reads as a rule this module cannot express, which
        blames coverage here for a missing key there. Two extractions of one prompt
        differed in exactly that way."""
        item = EXTRACTION_SCHEMA["properties"]["rules"]["items"]
        assert set(item["required"]) == set(item["properties"])


class TestUnreadableReplies:
    def test_text_that_is_not_json_raises(self):
        client = RecordedModelClient(replies=["sorry, no"])
        with pytest.raises(InferenceError, match="not readable"):
            infer_policy(config(), client)

    def test_a_body_without_rules_raises(self):
        client = RecordedModelClient(replies=[json.dumps({"answer": []})])
        with pytest.raises(InferenceError, match="list of rules"):
            infer_policy(config(), client)

    def test_a_rule_missing_its_quote_raises(self):
        client = RecordedModelClient(replies=[reply(rule(quote=""))])
        with pytest.raises(InferenceError, match="missing its name, quote or shape"):
            infer_policy(config(), client)

    def test_an_unreadable_reply_is_never_read_as_an_agent_with_no_rules(self):
        """Silence and failure are different results, and only one of them is worth
        reporting to somebody."""
        client = RecordedModelClient(replies=["{"])
        with pytest.raises(InferenceError):
            infer_policy(config(), client)

    def test_no_rules_at_all_is_a_valid_answer(self):
        client = RecordedModelClient(replies=[reply()])
        assert infer_policy(config(), client) == Inference()
