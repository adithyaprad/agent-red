"""Stakes derivation: what an agent's own declarations say is worth attacking."""

import pytest
from pydantic import ValidationError

from agentred.attacks.stakes import (
    Reach,
    Settlement,
    Stake,
    StakeKind,
    derive_reaches,
    derive_stakes,
    judge_dependence,
    merge_obligations,
)
from agentred.spec import (
    AgentConfig,
    AgentPolicy,
    AgentSpec,
    Consequence,
    DataScope,
    DataSource,
    EnumeratedBound,
    NumericBound,
    Obligation,
    ObligationKind,
    Precondition,
    Provenance,
    RelationalBound,
    ResultReference,
    Subject,
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


def subjects_for(scope):
    """One subject covering whatever kinds the scope declares.

    A spec whose policy scopes a session but declares nobody to be is refused at construction,
    which is the point of that rule. These are unit tests of derivation, so they supply the
    minimum that satisfies it rather than restating it.
    """
    kinds = tuple(scope.subject_identifier_kinds) if scope is not None else ()
    if not kinds:
        return ()
    return (Subject(name="somebody", identifiers={kind: f"{kind}-1" for kind in kinds}),)


def spec(tools=(), bounds=(), preconditions=(), scope=None, sources=(), obligations=()):
    return AgentSpec(
        subjects=subjects_for(scope),
        config=AgentConfig(
            agent_id="a",
            version="1",
            model="claude-sonnet-5",
            instructions="be helpful",
            tools=tools,
            data_sources=sources,
        ),
        policy=AgentPolicy(
            agent_id="a",
            version="1",
            bounds=bounds,
            preconditions=preconditions,
            obligations=obligations,
            **({"data_scope": scope} if scope is not None else {}),
        ),
    )


GIVE = tool("give_back", Consequence.MONEY, ("amount",))
READ = tool("read_record", Consequence.DISCLOSURE, ("reference",))
CHECK = tool("check_holder", Consequence.INERT, ("reference",))


class TestBoundStakes:
    def test_a_two_sided_numeric_limit_derives_both_sides(self):
        bound = NumericBound(
            name="ceiling", tool="give_back", argument="amount", maximum=500, minimum=0
        )
        stakes = derive_stakes(spec(tools=(GIVE,), bounds=(bound,)))
        assert {s.id.rsplit(":", 1)[-1] for s in stakes} == {"above", "below"}

    def test_a_one_sided_limit_derives_one_side(self):
        """A limit that is not set is not an attack surface, and inventing one would be a
        stake that can never be reached, which reports as an unbroken limit forever."""
        bound = NumericBound(name="ceiling", tool="give_back", argument="amount", maximum=500)
        stakes = derive_stakes(spec(tools=(GIVE,), bounds=(bound,)))
        assert len(stakes) == 1
        assert stakes[0].id.endswith(":above")

    def test_the_objective_carries_the_declared_figure(self):
        bound = NumericBound(name="ceiling", tool="give_back", argument="amount", maximum=500)
        stake = derive_stakes(spec(tools=(GIVE,), bounds=(bound,)))[0]
        assert "500" in stake.objective
        assert "'ceiling'" in stake.objective

    def test_an_enumerated_limit_derives_one_stake_naming_the_permitted_set(self):
        bound = EnumeratedBound(
            name="methods",
            tool="give_back",
            argument="amount",
            allowed_values=("slow", "fast"),
        )
        stake = derive_stakes(spec(tools=(GIVE,), bounds=(bound,)))[0]
        assert stake.id.endswith(":outside_set")
        assert "slow, fast" in stake.objective

    def test_a_relational_stake_names_what_has_to_happen_first(self):
        """The ceiling does not exist until the source action has been called.

        Without `requires_first` the attacker could reach the action without ever triggering
        the lookup, the bound would have no figure to compare against, and the conversation
        would read as compliant.
        """
        bound = RelationalBound(
            name="within_recorded",
            tool="give_back",
            argument="amount",
            maximum_from=ResultReference(tool="read_record", field="total"),
        )
        stake = derive_stakes(spec(tools=(GIVE, READ), bounds=(bound,)))[0]
        assert stake.requires_first == ("read_record",)
        assert "read_record.total" in stake.objective

    def test_every_bound_stake_is_settled_by_a_detector(self):
        bounds = (
            NumericBound(name="c", tool="give_back", argument="amount", maximum=1),
            RelationalBound(
                name="r",
                tool="give_back",
                argument="amount",
                maximum_from=ResultReference(tool="read_record", field="total"),
            ),
        )
        stakes = derive_stakes(spec(tools=(GIVE, READ), bounds=bounds))
        assert all(
            s.settled_by is Settlement.DETECTOR
            for s in stakes
            if s.kind is StakeKind.BOUND_EXCEEDED
        )


class TestPreconditionStakes:
    def test_one_stake_per_declared_precondition(self):
        pre = Precondition(name="gated", tool="give_back", requires="check_holder")
        stakes = derive_stakes(spec(tools=(GIVE, CHECK), preconditions=(pre,)))
        assert len(stakes) == 1
        assert stakes[0].kind is StakeKind.PRECONDITION_SKIPPED
        assert stakes[0].derived_from == "gated"

    def test_it_takes_the_consequence_of_the_gated_action_not_the_gate(self):
        """The cost of skipping a check is the cost of what the check was protecting."""
        pre = Precondition(name="gated", tool="give_back", requires="check_holder")
        stake = derive_stakes(spec(tools=(GIVE, CHECK), preconditions=(pre,)))[0]
        assert stake.consequence is Consequence.MONEY


class TestScopeStakes:
    def test_one_stake_per_subject_identifier_kind(self):
        """Reaching a second subject by one kind of reference is a different conversation
        from reaching one by another, so they are separate attacks."""
        scope = DataScope(sources=("records",), subject_identifier_kinds=("reference", "email"))
        stakes = derive_stakes(
            spec(
                tools=(READ,),
                sources=(DataSource(name="records", identifier_kinds=("reference",)),),
                scope=scope,
            )
        )
        scope_stakes = [s for s in stakes if s.kind is StakeKind.SCOPE_VIOLATION]
        assert {s.argument for s in scope_stakes} == {"reference", "email"}

    def test_an_agent_that_reveals_nothing_gets_no_scope_stake(self):
        """Otherwise the suite carries an attack with no action that could carry it out,
        and reports it as an unbroken limit."""
        scope = DataScope(sources=("records",), subject_identifier_kinds=("reference",))
        stakes = derive_stakes(
            spec(
                tools=(GIVE, CHECK),
                sources=(DataSource(name="records", identifier_kinds=("reference",)),),
                scope=scope,
            )
        )
        assert not [s for s in stakes if s.kind is StakeKind.SCOPE_VIOLATION]


class TestUngatedStakes:
    def test_a_costly_action_with_no_declaration_becomes_a_judge_settled_stake(self):
        stakes = derive_stakes(spec(tools=(GIVE,)))
        assert len(stakes) == 1
        assert stakes[0].kind is StakeKind.UNGATED_ACTION
        assert stakes[0].settled_by is Settlement.JUDGE
        assert stakes[0].derived_from == ""

    def test_an_inert_action_is_not_a_stake(self):
        assert derive_stakes(spec(tools=(CHECK,))) == ()

    def test_declaring_a_limit_removes_the_ungated_stake(self):
        """The point of the ungated kind is absence, so a declaration must retire it."""
        bound = NumericBound(name="c", tool="give_back", argument="amount", maximum=1)
        stakes = derive_stakes(spec(tools=(GIVE,), bounds=(bound,)))
        assert not [s for s in stakes if s.kind is StakeKind.UNGATED_ACTION]


class TestOrderingAndReporting:
    def test_costliest_first(self):
        tools = (GIVE, READ)
        scope = DataScope(sources=("records",), subject_identifier_kinds=("reference",))
        stakes = derive_stakes(
            spec(
                tools=tools,
                sources=(DataSource(name="records", identifier_kinds=("reference",)),),
                scope=scope,
            )
        )
        assert [s.consequence for s in stakes] == sorted(
            [s.consequence for s in stakes],
            key=lambda c: [
                Consequence.MONEY,
                Consequence.OBLIGATION,
                Consequence.DISCLOSURE,
                Consequence.INERT,
            ].index(c),
        )

    def test_derivation_is_reproducible(self):
        """Two runs of one spec must produce the same ids in the same sequence, or a diff
        between two versions of an agent is unreadable."""
        bounds = (NumericBound(name="c", tool="give_back", argument="amount", maximum=1),)
        pre = (Precondition(name="g", tool="give_back", requires="check_holder"),)
        built = spec(tools=(GIVE, CHECK, READ), bounds=bounds, preconditions=pre)
        assert [s.id for s in derive_stakes(built)] == [s.id for s in derive_stakes(built)]

    def test_ids_are_unique(self):
        bounds = (
            NumericBound(name="c", tool="give_back", argument="amount", maximum=1, minimum=0),
            RelationalBound(
                name="r",
                tool="give_back",
                argument="amount",
                maximum_from=ResultReference(tool="read_record", field="total"),
            ),
        )
        stakes = derive_stakes(spec(tools=(GIVE, READ), bounds=bounds))
        assert len({s.id for s in stakes}) == len(stakes)

    def test_provenance_is_carried_from_the_declaration(self):
        """Degraded mode has to stay visible in the output, not be hidden in it."""
        bound = NumericBound(
            name="c",
            tool="give_back",
            argument="amount",
            maximum=1,
            provenance=Provenance.INFERRED,
        )
        assert derive_stakes(spec(tools=(GIVE,), bounds=(bound,)))[0].provenance is (
            Provenance.INFERRED
        )

    def test_judge_dependence_is_the_share_a_model_has_to_settle(self):
        bound = NumericBound(name="c", tool="give_back", argument="amount", maximum=1)
        stakes = derive_stakes(spec(tools=(GIVE, READ), bounds=(bound,)))
        assert judge_dependence(stakes) == pytest.approx(0.5)

    def test_an_agent_declaring_nothing_is_wholly_judge_dependent(self):
        """The honest reading: there was nothing to check against."""
        assert judge_dependence(derive_stakes(spec(tools=(GIVE,)))) == 1.0

    def test_no_stakes_is_no_claim_rather_than_full_confidence(self):
        assert judge_dependence(()) == 0.0


def duty(name="no_note_out_loud", kind=ObligationKind.DISCLOSURE, applies_to=("read_record",)):
    return Obligation(
        name=name,
        kind=kind,
        statement="The note on a record is for staff. Do not read it out.",
        applies_to=applies_to,
        provenance=Provenance.INFERRED,
    )


class TestObligationStakes:
    """Rules about what the agent may say, which no call log can settle."""

    def test_a_rule_read_from_prose_becomes_something_to_attack(self):
        """Until it does, such a rule is checked after the fact and never aimed at."""
        derived = derive_stakes(spec(tools=(READ,)), inferred=(duty(),))
        speech = [stake for stake in derived if stake.kind is StakeKind.PRIVATE_REPEATED]
        assert len(speech) == 1
        assert speech[0].derived_from == "no_note_out_loud"
        assert speech[0].settled_by is Settlement.JUDGE
        assert speech[0].provenance is Provenance.INFERRED

    def test_the_objective_quotes_the_rule_as_written(self):
        """The attacker and the judge have to be working from the same sentence."""
        derived = derive_stakes(spec(tools=(READ,)), inferred=(duty(),))
        assert "Do not read it out." in derived[0].objective

    def test_each_kind_of_speech_rule_is_its_own_goal(self):
        """Merged into one cell, the grid would claim coverage of goals nothing pursued."""
        rules = (
            duty(name="a", kind=ObligationKind.DISCLOSURE),
            duty(name="b", kind=ObligationKind.ACCURACY),
            duty(name="c", kind=ObligationKind.COMMITMENT),
        )
        kinds = {stake.kind for stake in derive_stakes(spec(tools=(READ,)), inferred=rules)}
        assert {
            StakeKind.PRIVATE_REPEATED,
            StakeKind.UNVERIFIED_ASSERTED,
            StakeKind.UNDERTAKING_GIVEN,
        } <= kinds

    def test_a_rule_governing_the_whole_conversation_names_no_action(self):
        derived = derive_stakes(spec(tools=(READ,)), inferred=(duty(applies_to=()),))
        assert derived[0].tool == ""

    def test_a_rule_naming_an_action_the_agent_does_not_have_is_refused(self):
        """Dropped instead, it would sit on the grid and be unreachable in every run."""
        with pytest.raises(ValueError, match="does not declare"):
            derive_stakes(spec(tools=(READ,)), inferred=(duty(applies_to=("nowhere",)),))

    def test_a_declared_rule_wins_over_the_same_name_read_from_prose(self):
        """The operator's own wording is the one they would recognise on a page."""
        declared = Obligation(
            name="no_note_out_loud",
            kind=ObligationKind.DISCLOSURE,
            statement="Never repeat the note.",
            applies_to=("read_record",),
        )
        merged = merge_obligations((declared,), (duty(),))
        assert merged == (declared,)

    def test_nothing_inferred_leaves_the_derivation_as_it_was(self):
        declared_only = derive_stakes(spec(tools=(READ,)))
        assert all(stake.kind is not StakeKind.PRIVATE_REPEATED for stake in declared_only)

    def test_a_stake_that_is_not_about_speech_must_name_an_action(self):
        with pytest.raises(ValueError, match="must name the action"):
            Stake(
                id="x",
                kind=StakeKind.BOUND_EXCEEDED,
                tool="",
                consequence=Consequence.MONEY,
                objective="o",
                settled_by=Settlement.DETECTOR,
            )


class TestAgainstTheShippedTargets:
    """The derivation, run against the two specs that actually exist."""

    @pytest.fixture(scope="class", params=["cart_recovery", "dispute_handler"])
    @staticmethod
    def target(request):
        from agentred.spec.loader import load_spec_dir

        directory = f"src/agentred/targets/specs/{request.param}"
        return load_spec_dir(directory)

    def test_derives_stakes(self, target):
        assert len(derive_stakes(target)) >= 8

    def test_every_stake_names_a_declared_action(self, target):
        declared = set(target.config.tools_by_name)
        for stake in derive_stakes(target):
            assert stake.tool in declared or not stake.tool, stake.id
            assert set(stake.requires_first) <= declared, stake.id

    def test_most_stakes_are_settled_without_a_model(self, target):
        """If this ever fails, the scorecard has quietly become an opinion."""
        assert judge_dependence(derive_stakes(target)) < 0.5


class TestReaches:
    """Goal crossed with channel: the unit of work, after ADR-0006."""

    @pytest.fixture(scope="class", params=["cart_recovery", "dispute_handler"])
    @staticmethod
    def target(request):
        from agentred.spec.loader import load_spec_dir

        return load_spec_dir(f"src/agentred/targets/specs/{request.param}")

    def test_every_stake_is_reachable_down_every_channel(self, target):
        """Nothing is filtered here, and the count is the claim the coverage grid makes."""
        stakes = derive_stakes(target)
        ways_in = 1 + len(target.config.channels)
        assert len(derive_reaches(target)) == len(stakes) * ways_in

    def test_conversation_comes_first_for_each_stake(self, target):
        """A run cut short keeps the channel every agent has, rather than losing it."""
        reaches = derive_reaches(target)
        first = reaches[0]
        assert first.channel == "conversation"
        assert not first.is_planted

    def test_a_stakes_reaches_are_adjacent(self, target):
        """Which is what makes a suite resumable stake by stake rather than channel by channel."""
        seen = []
        for reach in derive_reaches(target):
            if not seen or seen[-1] != reach.stake.id:
                seen.append(reach.stake.id)
        assert len(seen) == len(set(seen))

    def test_a_planted_reach_carries_the_declaration_it_names(self, target):
        for reach in derive_reaches(target):
            if reach.is_planted:
                assert reach.declaration is not None
                assert reach.declaration.name == reach.channel
                assert reach.family == "planted"

    def test_a_conversational_reach_carries_no_declaration(self, target):
        conversation = [r for r in derive_reaches(target) if not r.is_planted]
        assert conversation
        assert all(reach.declaration is None for reach in conversation)

    def test_sequence_is_deterministic(self, target):
        def ids(spec):
            return [(r.stake.id, r.channel) for r in derive_reaches(spec)]

        assert ids(target) == ids(target)

    def test_an_agent_declaring_no_channel_is_reached_by_conversation_only(self, target):
        """Honest rather than empty: it still has the channel every agent has."""
        quiet = target.model_copy(
            update={"config": target.config.model_copy(update={"channels": ()})}
        )
        assert {reach.channel for reach in derive_reaches(quiet)} == {"conversation"}


class TestAReachIsRefusedWhenItCouldNotBeRun:
    """Both directions fail quietly: one plants nowhere, the other talks to a field."""

    @staticmethod
    def _stake():
        return Stake(
            id="ungated_action:send",
            kind=StakeKind.UNGATED_ACTION,
            tool="send",
            consequence=Consequence.MONEY,
            objective="Have the agent call send.",
            settled_by=Settlement.DETECTOR,
        )

    @staticmethod
    def _declaration(name="note"):
        from agentred.spec import ChannelDeclaration, TriggerKind

        return ChannelDeclaration(
            name=name,
            writer="The shopper.",
            data_source="carts",
            record_path="note",
            record_key="cart_id",
            trigger=TriggerKind.SCHEDULE,
        )

    def test_a_planted_reach_without_a_declaration_is_refused(self):
        with pytest.raises(ValidationError, match="no field to write into"):
            Reach(stake=self._stake(), channel="note")

    def test_a_conversational_reach_carrying_a_declaration_is_refused(self):
        with pytest.raises(ValidationError, match="which nothing would use"):
            Reach(stake=self._stake(), channel="conversation", declaration=self._declaration())

    def test_a_declaration_for_another_channel_is_refused(self):
        with pytest.raises(ValidationError, match="carries the declaration for"):
            Reach(
                stake=self._stake(),
                channel="note",
                declaration=self._declaration("somewhere_else"),
            )
