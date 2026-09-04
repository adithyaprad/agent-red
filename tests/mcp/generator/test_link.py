"""References that resolve, in a shop nobody wrote by hand.

The failure this file guards is the one that is invisible from the manifest. A record emitted
for one rule names another record by an identifier minted from its own counter, so the name
belongs to nothing. Reachability was decided when the record was emitted, so the manifest goes
on saying the rule is reachable; the agent reads the first record, follows the reference, is
truthfully told there is no such record, and stops. The rule reports as never evaluated, which
on a coverage grid is one cell away from a rule that was tested and held.
"""

from __future__ import annotations

from agentred.mcp.generator import generate
from agentred.mcp.generator.link import owners
from agentred.mcp.generator.shape import FieldKind, shapes_for
from agentred.spec import load_spec_dir
from agentred.spec.models import AgentSpec

SHIPPED = ("dispute_handler", "cart_recovery")


def _shipped(name: str) -> AgentSpec:
    return load_spec_dir(f"src/agentred/targets/specs/{name}")


def _dangling(spec: AgentSpec) -> list[tuple[str, str, str, str]]:
    """Every reference in a generated shop that names a record nothing holds."""
    generated = generate(spec)
    held = owners(_ShopLike(shapes_for(spec)))
    found = []
    for source, shape in generated.shapes.items():
        for record_id, record in generated.world[source].items():
            for name, field in shape.fields.items():
                if field.kind is not FieldKind.IDENTIFIER or name == shape.key:
                    continue
                owner = held.get(name)
                if owner is None or owner == source:
                    continue
                if str(record[name]) not in generated.world[owner]:
                    found.append((source, record_id, name, str(record[name])))
    return found


class _ShopLike:
    """The two attributes `owners` reads, so a caller need not build a whole shop."""

    def __init__(self, shapes: dict[str, object]) -> None:
        self.shapes = shapes


class TestReferencesResolve:
    def test_nothing_names_a_record_that_does_not_exist(self, assessor: AgentSpec):
        assert _dangling(assessor) == []

    def test_the_same_holds_for_both_shipped_agents(self):
        for name in SHIPPED:
            assert _dangling(_shipped(name)) == [], name

    def test_a_collection_named_by_another_carries_that_collection_s_keys(self):
        """A source whose own key is a reference to another source. Left unlinked, the tool
        that fetches one of its records answers with an error for every reference there is,
        and a channel that plants into one of its fields writes where no trigger will look.
        """
        generated = generate(_shipped("dispute_handler"))
        assert generated.world["shipments"]
        assert set(generated.world["shipments"]) <= set(generated.world["orders"])

    def test_every_channel_can_reach_the_record_it_plants_into(self):
        """A channel is a field on a record. A channel whose source holds nothing keyed the
        way the trigger names it plants into nothing, and the attack reports as attempted."""
        for name in SHIPPED:
            spec = _shipped(name)
            generated = generate(spec)
            for channel in spec.config.channels:
                records = generated.world[channel.data_source]
                assert records, (name, channel.name)
                for record in records.values():
                    assert channel.record_key in record, (name, channel.name)
                    assert channel.record_path in record, (name, channel.name)


class TestWhatLinkingKeepsIntact:
    def test_records_a_fixture_made_agree_still_agree(self, assessor: AgentSpec):
        """The equivalence classes are the emitters' own. A pass that pointed every reference
        at a real record but pulled apart the pair a rule needs would trade one silent failure
        for another."""
        generated = generate(assessor)
        for rule in generated.manifest.reachable:
            fixtures = generated.manifest.for_rule(rule)
            if len(fixtures) != 2 or fixtures[0].collection != fixtures[1].collection:
                continue
            first, second = (generated.world[f.collection][f.record_id] for f in fixtures)
            assert first != second, rule

    def test_a_record_agrees_with_the_record_it_names_about_whose_it_is(self):
        """A reference resolving is not the same as two records describing one party. Without
        this a check asking whether a message carried this party's details compares two
        spellings of the same person and calls them strangers."""
        generated = generate(_shipped("dispute_handler"))
        for dispute in generated.world["disputes"].values():
            order = generated.world["orders"][dispute["order_id"]]
            assert dispute["customer_id"] == order["customer_id"]
        for order in generated.world["orders"].values():
            customer = generated.world["customers"][order["customer_id"]]
            assert order["email"] == customer["email"]

    def test_the_manifest_still_points_at_the_records_it_names(self):
        """A collection re-keyed to name real records changes the ids fixtures were written
        with, and a fixture pointing at a record that has moved is an account of why a record
        exists attached to nothing."""
        for name in SHIPPED:
            generated = generate(_shipped(name))
            for fixture in generated.manifest.fixtures:
                assert fixture.record_id in generated.world[fixture.collection], (
                    name,
                    fixture.rule,
                    fixture.record_id,
                )

    def test_linking_is_reproducible(self):
        for name in SHIPPED:
            assert generate(_shipped(name)).world.digest == generate(_shipped(name)).world.digest
