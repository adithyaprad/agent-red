"""The synthetic shop the stand-in agents act on.

Every conversation gets its own copy. That is not tidiness: a refund issued in conversation
14 that is still visible in conversation 15 changes what conversation 15 can be talked into,
and every failure rate the scorecard reports afterwards would be measuring a mixture of the
attack and the damage done by earlier attacks. Isolation is what makes a rate mean anything.

The data is checked in under `data/store/` so a run is reproducible. Nothing here is real,
and the tools in `tools/` are the only things that mutate it.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STORE_DIR = Path(__file__).resolve().parents[3] / "data" / "store"
"""Where the seed files live, relative to the repository root."""


@dataclass
class World:
    """One conversation's private view of the shop.

    Mutating a `World` affects nothing outside the conversation that owns it.

    Attributes:
        products: Catalogue rows keyed by SKU.
        shipping_methods: Shipping options keyed by code.
        discount_codes: Discount codes keyed by code. A code the agent invents is simply
            absent, which is what makes a fabricated code observable.
        customers: Customer records keyed by customer id.
        carts: Abandoned carts keyed by cart id.
        orders: Orders keyed by order id.
        disputes: Chargebacks keyed by dispute id. Each carries the reason the buyer wrote,
            which is text the merchant never authored and an adversary can choose.
        settled_keys: Deduplication keys a payment action has already been given, to the
            result it produced. A real payments API refuses to charge twice for one key, and
            a synthetic one that did not would make a correctly written agent look reckless.
        ledger: Money and obligations the conversation created, in call order. The judge
            reads the tool-call log rather than this, but a target that keeps its own
            record makes an argument about what happened settleable.
    """

    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    shipping_methods: dict[str, dict[str, Any]] = field(default_factory=dict)
    discount_codes: dict[str, dict[str, Any]] = field(default_factory=dict)
    customers: dict[str, dict[str, Any]] = field(default_factory=dict)
    carts: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    disputes: dict[str, dict[str, Any]] = field(default_factory=dict)
    settled_keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def customer_by_email(self, email: str) -> dict[str, Any] | None:
        """Find a customer by email, case-insensitively. `None` if there is no such customer."""
        wanted = email.strip().lower()
        for customer in self.customers.values():
            if customer["email"].lower() == wanted:
                return customer
        return None

    def cart_total(self, cart_id: str) -> float:
        """Sum a cart at catalogue prices.

        Args:
            cart_id: The cart to total.

        Returns:
            The total, rounded to two decimal places. Lines naming a SKU the catalogue does
            not carry contribute nothing.
        """
        cart = self.carts.get(cart_id)
        if cart is None:
            return 0.0
        total = 0.0
        for line in cart["lines"]:
            product = self.products.get(line["sku"])
            if product is not None:
                total += float(product["price"]) * int(line["quantity"])
        return round(total, 2)

    def record(self, action: str, **detail: Any) -> None:
        """Append one consequential action to this conversation's ledger."""
        self.ledger.append({"action": action, **detail})


def _seed() -> World:
    """Read the seed files once, into the template every conversation is copied from.

    Raises:
        FileNotFoundError: If a seed file is missing. A target with no world is not
            something to start and hope about.
    """
    catalog = json.loads((STORE_DIR / "catalog.json").read_text(encoding="utf-8"))
    people = json.loads((STORE_DIR / "customers.json").read_text(encoding="utf-8"))
    orders = json.loads((STORE_DIR / "orders.json").read_text(encoding="utf-8"))
    disputes = json.loads((STORE_DIR / "disputes.json").read_text(encoding="utf-8"))
    return World(
        products={row["sku"]: row for row in catalog["products"]},
        shipping_methods={row["code"]: row for row in catalog["shipping_methods"]},
        discount_codes={row["code"]: row for row in catalog["discount_codes"]},
        customers={row["customer_id"]: row for row in people["customers"]},
        carts={row["cart_id"]: row for row in people["carts"]},
        orders={row["order_id"]: row for row in orders["orders"]},
        disputes={row["dispute_id"]: row for row in disputes["disputes"]},
    )


_TEMPLATE: World | None = None


def fresh_world() -> World:
    """A private copy of the seeded shop, for one conversation.

    The seed files are read once per process; every call after that is a deep copy, so a
    conversation cannot see another conversation's refunds.
    """
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _seed()
    return copy.deepcopy(_TEMPLATE)
