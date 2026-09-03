"""Every conversation gets its own shop, and the shop is named by a declaration."""

from __future__ import annotations

import pytest

from agentred.mcp.world import UnknownCollectionError, fresh_world


def test_the_seed_loads() -> None:
    world = fresh_world()
    assert world["products"] and world["carts"] and world["orders"] and world["customers"]


def test_a_conversation_cannot_see_another_conversations_damage() -> None:
    first = fresh_world()
    first["orders"]["ORD-55210"]["refunded"] = 76_900.0
    first["customers"]["C-1041"]["store_credit"] = 50_000.0
    second = fresh_world()
    assert second["orders"]["ORD-55210"]["refunded"] == 0.0
    assert second["customers"]["C-1041"]["store_credit"] == 0.0


class TestACollectionThisWorldDoesNotHold:
    """Refused rather than answered with an empty collection.

    An empty answer makes every rule over that source report as never evaluated, and never
    evaluated on a coverage grid looks at a glance like a source that was covered.
    """

    def test_reading_one_raises(self) -> None:
        with pytest.raises(UnknownCollectionError):
            fresh_world()["policies"]

    def test_the_error_says_what_the_world_does_hold(self) -> None:
        with pytest.raises(UnknownCollectionError) as raised:
            fresh_world()["policies"]
        assert "disputes" in str(raised.value)

    def test_membership_answers_without_raising(self) -> None:
        world = fresh_world()
        assert "orders" in world
        assert "policies" not in world


def test_the_world_can_say_what_it_holds() -> None:
    assert "shipments" in fresh_world().names
