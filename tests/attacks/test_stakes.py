"""Stakes derivation: what an agent's own declarations say is worth attacking."""

import pytest

from agentred.attacks.stakes import (
    Settlement,
    StakeKind,
    derive_stakes,
    judge_dependence,
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


def spec(tools=(), bounds=(), preconditions=(), scope=None, sources=()):
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
            assert stake.tool in declared, stake.id
            assert set(stake.requires_first) <= declared, stake.id

    def test_most_stakes_are_settled_without_a_model(self, target):
        """If this ever fails, the scorecard has quietly become an opinion."""
        assert judge_dependence(derive_stakes(target)) < 0.5
