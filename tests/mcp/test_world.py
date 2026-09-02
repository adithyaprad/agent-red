"""Every conversation gets its own shop."""

from __future__ import annotations

from agentred.mcp.world import fresh_world


def test_the_seed_loads() -> None:
    world = fresh_world()
    assert world.products and world.carts and world.orders and world.customers


def test_a_conversation_cannot_see_another_conversations_damage() -> None:
    first = fresh_world()
    first.orders["ORD-55210"]["refunded"] = 76_900.0
    first.customers["C-1041"]["store_credit"] = 50_000.0
    second = fresh_world()
    assert second.orders["ORD-55210"]["refunded"] == 0.0
    assert second.customers["C-1041"]["store_credit"] == 0.0


def test_cart_total_is_priced_from_the_catalogue() -> None:
    world = fresh_world()
    assert world.cart_total("CART-8891") == 18_900.0 + 4 * 14_500.0


def test_an_unknown_cart_totals_nothing() -> None:
    assert fresh_world().cart_total("CART-0000") == 0.0


def test_customer_lookup_by_email_ignores_case() -> None:
    world = fresh_world()
    assert world.customer_by_email("MAYA.LINDQVIST@example.com")["customer_id"] == "C-1041"
    assert world.customer_by_email("nobody@example.com") is None
