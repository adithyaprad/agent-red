"""Identities that name records the world actually holds.

The fourth per-merchant integration, and the one that stayed invisible while the other three
were hand-written: a hand-authored cast standing in a hand-authored world reads as part of the
world. Point it at a generated one and every reference belongs to nothing, so the agent is
asked about a record it cannot find, answers truthfully, and every rule reports as never in
play. On a planted channel the write is refused and the suite fails before the agent is
reached at all, which is what run 0019 did with all 322 of its attacks.
"""

from __future__ import annotations

from agentred.mcp.generator import generate
from agentred.spec import load_spec_dir
from agentred.spec.models import AgentSpec

SHIPPED = ("dispute_handler", "cart_recovery")


def _shipped(name: str) -> AgentSpec:
    return load_spec_dir(f"src/agentred/targets/specs/{name}")


class TestWhoTheHarnessMayActAs:
    def test_a_cast_is_produced_for_an_agent_nobody_wrote_one_for(self, assessor: AgentSpec):
        assert generate(assessor).subjects

    def test_every_identifier_names_a_record_that_exists(self, assessor: AgentSpec):
        """The whole point. An identifier that resolves to nothing is a conversation that
        ends before the action under test is reached."""
        for spec in (assessor, *(_shipped(name) for name in SHIPPED)):
            generated = generate(spec)
            keyed = {shape.key: source for source, shape in generated.shapes.items() if shape.key}
            for subject in generated.subjects:
                for kind, value in subject.identifiers.items():
                    if kind in keyed:
                        assert value in generated.world[keyed[kind]], (subject.name, kind)

    def test_a_generated_spec_satisfies_the_contract_the_loader_enforces(self):
        """`AgentSpec` refuses a subject missing an identifier the scope binds a session by,
        so building one that fails is a construction error rather than a bad run."""
        for name in SHIPPED:
            spec = _shipped(name)
            served = generate(spec).spec_for(spec)
            kinds = set(spec.policy.data_scope.subject_identifier_kinds)
            assert served.subjects
            for subject in served.subjects:
                assert kinds <= set(subject.identifiers), (name, subject.name)

    def test_the_version_tuple_does_not_move(self):
        """Who may be impersonated is a fixture rather than a rule. A cast that invalidated
        every scorecard already produced for an agent would make adding one a decision."""
        for name in SHIPPED:
            spec = _shipped(name)
            assert generate(spec).spec_for(spec).version_tuple == spec.version_tuple

    def test_both_halves_of_the_world_are_represented(self):
        """A cast drawn only from the records that make rules breakable is a cast of people
        it is always right to refuse, and an agent that refuses all of them scores perfectly
        while being useless. Same argument as the holding fixtures, one level out."""
        for name in SHIPPED:
            names = [subject.name for subject in generate(_shipped(name)).subjects]
            assert any(one.endswith("-breakable") for one in names), name
            assert any(one.endswith("-holding") for one in names), name

    def test_every_planted_channel_an_identity_supplies_can_be_written_to(self):
        """A plant into a record that is not there is refused by the driver, and the cell
        reports as an error rather than as an attempt."""
        for name in SHIPPED:
            spec = _shipped(name)
            generated = generate(spec)
            for channel in spec.config.channels:
                for subject in generated.subjects:
                    named = subject.identifiers.get(channel.record_key)
                    if named is None:
                        continue
                    assert named in generated.world[channel.data_source], (
                        name,
                        channel.name,
                        subject.name,
                    )

    def test_a_channel_no_identity_can_reach_is_named(self):
        """The recovery agent declares a way in through a source its sessions read without
        owning, and nothing in the declaration says which of those records is whose. Named
        rather than guessed at, because inventing the link produces a shop where every
        channel works and one of them is fiction."""
        generated = generate(_shipped("cart_recovery"))
        named = {channel: why for channel, why in generated.unsupported}
        assert "product_description" in named
        assert "sku" in named["product_description"]

    def test_nothing_is_unsupported_for_the_dispute_agent(self):
        assert generate(_shipped("dispute_handler")).unsupported == ()

    def test_facts_leave_out_the_fields_an_attacker_writes(self, assessor: AgentSpec):
        """Free text is the field a planted payload lands in. Handing it back as something
        the subject knows would let a payload planted last turn arrive as a fact this turn."""
        generated = generate(assessor)
        planted = {channel.record_path for channel in assessor.config.channels}
        for subject in generated.subjects:
            for fact in subject.facts:
                assert not any(field.replace("_", " ") in fact for field in planted), fact

    def test_the_same_seed_produces_the_same_cast(self):
        for name in SHIPPED:
            spec = _shipped(name)
            assert generate(spec, 11).subjects == generate(spec, 11).subjects
