"""The synthetic shop the stand-in agents act on.

Every conversation gets its own copy. That is not tidiness: a refund issued in conversation
14 that is still visible in conversation 15 changes what conversation 15 can be talked into,
and every failure rate the scorecard reports afterwards would be measuring a mixture of the
attack and the damage done by earlier attacks. Isolation is what makes a rate mean anything.

**A world is collections named by the declaration, not fields named by this file.** The
earlier version had eight attributes called `products`, `customers`, `orders` and so on, and
every one of them was a per-merchant integration written into the harness. An agent that
handles insurance claims has no orders, and adding it meant editing this dataclass, the
arena's mapping, the plant path and the tests underneath all three. So the shape moved into
the data: a world holds a mapping of collection name to records, the names come from what the
agent declares it reads, and nothing structural has to change to point the harness at a
different business. See ADR-0007.

The two shipped agents keep a hand-authored world under `data/store/`, because it is the
fixture a generated world is checked against: a generated shop and a hand-written one that
disagree is how the generator is known to be faithful.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentred.paths import repo_path


def store_dir() -> Path:
    """Where the seed files for the hand-authored shop live.

    Looked up rather than computed from this module's location, because the package is not
    always inside the working tree that carries `data/`. See `agentred.paths`.
    """
    return repo_path("data", "store")


Record = dict[str, Any]
"""One row. Whatever the declaration says a record of its kind carries."""

Collection = dict[str, Record]
"""Records of one kind, keyed by whichever identifier the agent reaches them by."""


class UnknownSourceError(KeyError):
    """A declared data source nothing in this world backs.

    Raised rather than guessed at. A channel aimed at a source no collection backs plants a
    payload nothing will ever load: the run completes, no violation is found, and the coverage
    grid records the cell as attempted. A false green is worse than an empty cell.
    """


class UnknownCollectionError(KeyError):
    """A collection this world does not hold was asked for.

    Raised rather than answered with an empty collection. A missing collection read as empty
    makes every rule over it report as never evaluated, and never evaluated on a grid is
    indistinguishable at a glance from a rule that was tested: the run would look like it had
    covered a data source it never touched.
    """


@dataclass
class World:
    """One conversation's private view of the shop.

    Mutating a `World` affects nothing outside the conversation that owns it.

    Attributes:
        collections: Records by collection name, each collection keyed by the identifier the
            agent reaches its records by.
        sources: Declared data source name to the collection backing it. Not the identity
            function for a hand-authored shop, because an agent declares what it reads
            (`catalog`) and the shop stores what is in it (`products`). It lives on the world
            rather than in the runner so that the one thing that knows how a declaration lands
            on a world is the world, which is what lets a generated shop name its own
            collections without anything upstream changing. A generated world's map is the
            identity, because its collections are named by the declaration.
        settled_keys: Deduplication keys a payment action has already been given, to the
            result it produced. A real payments API refuses to charge twice for one key, and
            a synthetic one that did not would make a correctly written agent look reckless.
            Not a collection: nothing declares it and nothing reads it as data.
        ledger: Money and obligations the conversation created, in call sequence. The judge
            reads the tool-call log rather than this, but a target that keeps its own
            record makes an argument about what happened settleable.
    """

    collections: dict[str, Collection] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    settled_keys: dict[str, Record] = field(default_factory=dict)
    ledger: list[Record] = field(default_factory=list)

    def __getitem__(self, name: str) -> Collection:
        """The records of one collection.

        Raises:
            UnknownCollectionError: If this world holds no such collection.
        """
        try:
            return self.collections[name]
        except KeyError:
            held = ", ".join(sorted(self.collections)) or "nothing"
            raise UnknownCollectionError(
                f"this world holds no collection {name!r}. It holds: {held}."
            ) from None

    def __contains__(self, name: str) -> bool:
        """Whether this world holds a collection of that name."""
        return name in self.collections

    @property
    def names(self) -> tuple[str, ...]:
        """Every collection this world holds, in the sequence it was built with."""
        return tuple(self.collections)

    @property
    def digest(self) -> str:
        """A content hash of everything this world holds.

        Stable across dictionary ordering, so a world holding the same records is the same
        world however it was assembled. It exists because a scorecard computed against one
        shop says nothing about an agent facing another (ADR-0007), and that is as true of a
        hand-authored shop as of a generated one: the day `data/store/` was rebuilt, every
        earlier scorecard went on citing a version tuple that no longer described what the
        agent had faced. This is the element that stops that.

        Truncated to the same width as the tool digest, because both are read by people
        comparing two runs rather than by anything guarding against collisions.
        """
        canonical = json.dumps(self.collections, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"

    def collection_for(self, source: str) -> str:
        """The collection backing a declared data source.

        Raises:
            UnknownSourceError: If nothing in this world backs that source.
        """
        collection = self.sources.get(source)
        if collection is None:
            backed = ", ".join(sorted(self.sources)) or "nothing"
            raise UnknownSourceError(
                f"nothing in this world backs data source {source!r}. Backed: {backed}."
            )
        return collection

    def record(self, action: str, **detail: Any) -> None:
        """Append one consequential action to this conversation's ledger."""
        self.ledger.append({"action": action, **detail})


def _seed() -> World:
    """Read the seed files once, into the template every conversation is copied from.

    The collection names and the key each is keyed by are written here rather than inferred,
    because this is the hand-authored shop: it exists to be the thing a generated world is
    compared against, so what is in it is stated rather than derived.

    Raises:
        FileNotFoundError: If a seed file is missing. A target with no world is not
            something to start and hope about.
    """

    def read(name: str) -> dict[str, Any]:
        return json.loads((store_dir() / name).read_text(encoding="utf-8"))

    catalog = read("catalog.json")
    people = read("customers.json")
    orders = read("orders.json")
    disputes = read("disputes.json")
    shipments = read("shipments.json")

    def by(rows: list[Record], key: str) -> Collection:
        return {str(row[key]): row for row in rows}

    return World(
        collections={
            "products": by(catalog["products"], "sku"),
            "shipping_methods": by(catalog["shipping_methods"], "code"),
            "discount_codes": by(catalog["discount_codes"], "code"),
            "customers": by(people["customers"], "customer_id"),
            "carts": by(people["carts"], "cart_id"),
            "orders": by(orders["orders"], "order_id"),
            "disputes": by(disputes["disputes"], "dispute_id"),
            # Keyed by the order rather than by its own reference, because the order is what
            # anyone has when a chargeback arrives, and a record nobody can name from what
            # they were sent is a record nobody reads.
            "shipments": by(shipments["shipments"], "order_id"),
        },
        sources={
            "catalog": "products",
            "customers": "customers",
            "carts": "carts",
            "orders": "orders",
            "disputes": "disputes",
            "shipments": "shipments",
        },
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
