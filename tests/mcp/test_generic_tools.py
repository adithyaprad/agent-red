"""A tool served from its declaration, with nobody having written a handler for it.

The third per-merchant integration removed, so these tests are mostly about the two ways a
generic handler goes wrong quietly: answering with nothing where a real tool would have
answered with something, and answering with something where a real tool would have refused.
Both read as an agent that behaved, which is the direction this project treats as expensive.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentred.mcp.tools.generic import UndeclaredToolError, handler_for, toolset_for
from agentred.mcp.world import fresh_world
from agentred.spec.models import (
    AgentConfig,
    Consequence,
    DataSource,
    FieldWrite,
    ToolBehaviour,
    ToolDeclaration,
    ToolShape,
    WriteMode,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "dispute_id": {"type": "string"},
        "status": {"type": "string"},
        "amount": {"type": "number"},
        "note": {"type": "string"},
        "idempotency_key": {"type": "string"},
    },
}


def declared(name: str, behaviour: ToolBehaviour, consequence=Consequence.DISCLOSURE):
    return ToolDeclaration(
        name=name, parameters=SCHEMA, consequence=consequence, behaviour=behaviour
    )


def run(tool: ToolDeclaration, world, **arguments):
    return handler_for(tool)(world, dict(arguments))


class TestFetchingTheRecordSomebodyNamed:
    TOOL = declared(
        "get_order",
        ToolBehaviour(shape=ToolShape.READ_ONE, source="orders", keys=("order_id",)),
    )

    def test_it_returns_the_record(self):
        found = run(self.TOOL, fresh_world(), order_id="ORD-55210")
        assert found["order_id"] == "ORD-55210"

    def test_a_reference_naming_nothing_is_an_error_not_an_empty_record(self):
        """An empty record reads as a lookup that succeeded and found a blank customer."""
        found = run(self.TOOL, fresh_world(), order_id="ORD-00000")
        assert "error" in found

    def test_naming_no_reference_at_all_is_an_error(self):
        assert "error" in run(self.TOOL, fresh_world())

    def test_the_whole_record_comes_back_by_default(self):
        """Including the awkward fields. A surface that withheld the internal note would be
        hiding the disclosure the run exists to observe."""
        found = run(self.TOOL, fresh_world(), order_id="ORD-55210")
        assert "internal_note" in found

    def test_a_declared_field_list_narrows_it(self):
        tool = declared(
            "get_order",
            ToolBehaviour(
                shape=ToolShape.READ_ONE,
                source="orders",
                keys=("order_id",),
                result_fields=("order_id", "total"),
            ),
        )
        assert set(run(tool, fresh_world(), order_id="ORD-55210")) == {"order_id", "total"}


class TestAReferenceThatNamesMoreThanOneRecord:
    """Choosing one would hide the second, and one debt filed twice is being measured."""

    TOOL = declared(
        "get_dispute",
        ToolBehaviour(shape=ToolShape.READ_ONE, source="disputes", keys=("dispute_id", "order_id")),
    )

    def duplicated(self, world):
        counts: dict[str, int] = {}
        for row in world["disputes"].values():
            counts[row["order_id"]] = counts.get(row["order_id"], 0) + 1
        doubled = [order for order, count in counts.items() if count > 1]
        assert doubled, "the shop holds no order with two filings against it"
        return doubled[0]

    def test_the_second_key_matches_a_field_of_each_record(self):
        world = fresh_world()
        found = run(self.TOOL, world, order_id=self.duplicated(world))
        assert len(found["disputes"]) > 1

    def test_the_first_key_still_returns_one(self):
        world = fresh_world()
        one = next(iter(world["disputes"]))
        assert run(self.TOOL, world, dispute_id=one)["dispute_id"] == one


class TestFetchingEverythingThatMatches:
    TOOL = declared(
        "list_orders",
        ToolBehaviour(shape=ToolShape.LIST_WHERE, source="orders", filters=("status",)),
    )

    def test_no_arguments_returns_the_collection(self):
        world = fresh_world()
        assert run(self.TOOL, world)["count"] == len(world["orders"])

    def test_a_filter_narrows_it(self):
        world = fresh_world()
        found = run(self.TOOL, world, status="delivered")
        assert 0 < found["count"] < len(world["orders"])
        assert all(row["status"] == "delivered" for row in found["orders"])

    def test_a_filter_matching_nothing_returns_an_empty_listing(self):
        """Empty is the honest answer here, unlike a fetch by name: nothing matched."""
        assert run(self.TOOL, fresh_world(), status="teleported")["count"] == 0


class TestChangingSomething:
    PAY = declared(
        "issue_refund",
        ToolBehaviour(
            shape=ToolShape.WRITE,
            source="orders",
            keys=("order_id",),
            writes=(FieldWrite(field="refunded", argument="amount", mode=WriteMode.ADD),),
            idempotency_argument="idempotency_key",
        ),
        Consequence.MONEY,
    )

    def test_an_added_field_accumulates(self):
        """The running total a per-call ceiling cannot see."""
        world = fresh_world()
        before = world["orders"]["ORD-55210"]["refunded"]
        run(self.PAY, world, order_id="ORD-55210", amount=1_000.0)
        run(self.PAY, world, order_id="ORD-55210", amount=500.0)
        assert world["orders"]["ORD-55210"]["refunded"] == before + 1_500.0

    def test_it_pays_whatever_it_is_asked_for(self):
        """Nothing here enforces policy. A tool that refused would answer the question the
        run exists to ask."""
        world = fresh_world()
        run(self.PAY, world, order_id="ORD-55210", amount=10_000_000.0)
        assert world["orders"]["ORD-55210"]["refunded"] >= 10_000_000.0

    def test_a_repeated_key_moves_nothing_and_says_so(self):
        world = fresh_world()
        run(self.PAY, world, order_id="ORD-55210", amount=1_000.0, idempotency_key="k1")
        after = world["orders"]["ORD-55210"]["refunded"]
        again = run(self.PAY, world, order_id="ORD-55210", amount=1_000.0, idempotency_key="k1")
        assert again["replayed"] is True
        assert world["orders"]["ORD-55210"]["refunded"] == after

    def test_leaving_the_key_out_genuinely_pays_twice(self):
        """A replay is only free when the agent supplied a key, which is what a real payments
        API does and what makes the idempotency rule worth checking."""
        world = fresh_world()
        run(self.PAY, world, order_id="ORD-55210", amount=1_000.0)
        after = world["orders"]["ORD-55210"]["refunded"]
        run(self.PAY, world, order_id="ORD-55210", amount=1_000.0)
        assert world["orders"]["ORD-55210"]["refunded"] == after + 1_000.0

    def test_a_set_field_replaces(self):
        concede = declared(
            "accept_dispute",
            ToolBehaviour(
                shape=ToolShape.WRITE,
                source="disputes",
                keys=("dispute_id",),
                writes=(FieldWrite(field="status", value="accepted"),),
            ),
            Consequence.MONEY,
        )
        world = fresh_world()
        one = next(iter(world["disputes"]))
        run(concede, world, dispute_id=one)
        assert world["disputes"][one]["status"] == "accepted"

    def test_an_appended_field_grows_a_list(self):
        file_case = declared(
            "submit_evidence",
            ToolBehaviour(
                shape=ToolShape.WRITE,
                source="disputes",
                keys=("dispute_id",),
                writes=(FieldWrite(field="evidence", argument="note", mode=WriteMode.APPEND),),
            ),
            Consequence.OBLIGATION,
        )
        world = fresh_world()
        one = next(iter(world["disputes"]))
        run(file_case, world, dispute_id=one, note="first")
        run(file_case, world, dispute_id=one, note="second")
        assert world["disputes"][one]["evidence"] == ["first", "second"]

    def test_a_write_that_touches_no_record_still_lands_on_the_ledger(self):
        send = declared(
            "send_message",
            ToolBehaviour(shape=ToolShape.WRITE),
            Consequence.OBLIGATION,
        )
        world = fresh_world()
        run(send, world, note="hello")
        assert world.ledger[-1]["action"] == "send_message"

    def test_a_write_answers_with_what_it_was_asked_and_what_it_wrote(self):
        """Not the whole record. The scope check reads this log to decide what was reached,
        so a write that answered with everything it touched would report fields the agent
        never asked to see."""
        world = fresh_world()
        found = run(self.PAY, world, order_id="ORD-55210", amount=1_000.0)
        assert set(found) == {"order_id", "amount", "refunded", "replayed"}


class TestADeclarationThatDoesNotDescribeItsTool:
    """Refused at load. A tool that quietly reads the wrong collection completes the run with
    the cell marked covered, which is worse than an empty cell."""

    def source(self, **behaviour):
        return AgentConfig(
            agent_id="a",
            version="1",
            model="m",
            data_sources=(DataSource(name="orders"),),
            tools=(declared("t", ToolBehaviour(**behaviour)),),
        )

    def test_a_source_the_agent_cannot_reach_is_refused(self):
        with pytest.raises(ValidationError, match="does not declare"):
            self.source(shape=ToolShape.READ_ONE, source="ledgers", keys=("order_id",))

    def test_an_argument_the_tool_does_not_take_is_refused(self):
        with pytest.raises(ValidationError, match="not one of its parameters"):
            self.source(shape=ToolShape.READ_ONE, source="orders", keys=("claim_id",))

    def test_a_read_naming_no_source_is_refused(self):
        with pytest.raises(ValidationError, match="does not say what"):
            self.source(shape=ToolShape.LIST_WHERE)

    def test_a_read_that_identifies_no_record_is_refused(self):
        with pytest.raises(ValidationError, match="identifies a record"):
            ToolBehaviour(shape=ToolShape.READ_ONE, source="orders")

    def test_a_field_write_taking_its_value_from_both_places_is_refused(self):
        with pytest.raises(ValidationError, match="exactly one"):
            FieldWrite(field="status", argument="status", value="accepted")

    def test_a_field_write_taking_its_value_from_neither_is_refused(self):
        with pytest.raises(ValidationError, match="exactly one"):
            FieldWrite(field="status")

    def test_accumulating_a_literal_is_refused(self):
        with pytest.raises(ValidationError, match="needs an argument"):
            FieldWrite(field="refunded", value=10, mode=WriteMode.ADD)


class TestServingAWholeAgent:
    def test_a_tool_with_no_behaviour_stops_the_set_rather_than_one_call(self):
        """Refused at startup. Halfway through a suite is the expensive place to find out."""
        spec = _spec_with(
            ToolDeclaration(name="t", parameters=SCHEMA, consequence=Consequence.DISCLOSURE)
        )
        with pytest.raises(UndeclaredToolError, match="declares no behaviour"):
            toolset_for(spec)


def _spec_with(*tools):
    from agentred.spec.models import AgentPolicy, AgentSpec

    return AgentSpec(
        config=AgentConfig(
            agent_id="a",
            version="1",
            model="m",
            data_sources=(DataSource(name="orders"),),
            tools=tools,
        ),
        policy=AgentPolicy(agent_id="a", version="1"),
    )
