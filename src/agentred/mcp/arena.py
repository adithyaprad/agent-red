"""The worlds the tool server acts on, and the write path an attack plants through.

The world used to live inside the target, one copy per conversation, because the target was
the only thing that touched it. Once tools are served from outside the agent (ADR-0005) the
world has to move with them: a refund is a mutation of the shop, and the shop is on this
side of the trust line now.

Three jobs, and the second and third are new.

**Isolation.** Every session gets its own deep copy of the seeded shop, so a refund granted
in one conversation cannot change what the next conversation can be talked into. Rates
computed across a shared world would measure the suite rather than the agent.

**Snapshot and restore.** A conversation gets its isolation from having its own session. A
planted attack has no conversation and therefore no session of its own to hide behind, so it
gets a known baseline instead: restore, plant, trigger, read (ADR-0006).

**Branching.** A forked conversation needs the world as it stood at the turn it branched
from, not as it stands now. Without that, a branch taken at turn one inherits money spent by
turns two and three, and two branches then differ by more than the turn that was changed
(ADR-0002). The model side of a fork stays with the target; the world side is here.

Nothing in this module enforces policy. A plant writes what it is given and a tool does what
it is asked, because a world that refused would be answering the question the suite exists
to ask.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agentred.mcp.world import World, fresh_world

COLLECTION_FOR_SOURCE = {
    "catalog": "products",
    "customers": "customers",
    "carts": "carts",
    "orders": "orders",
    "disputes": "disputes",
}
"""Declared data source name to the collection of this world that backs it.

Not the identity function, because an agent declares what it reads (`catalog`) and the world
stores what is in it (`products`). The mapping lives here rather than in the runner so that
the one place that knows how a declaration lands on this world is the module that owns the
world. On a real platform this table is whatever the connector does, and the runner is
unchanged either way.
"""

PLANTABLE_COLLECTIONS = tuple(COLLECTION_FOR_SOURCE.values())
"""Collections an attacker's text can reach.

Deliberately not every field of the world. `discount_codes` and `shipping_methods` are
merchant configuration, and a harness that planted into them would be reporting a finding
about an attacker who had already got into the admin panel. Each entry here corresponds to
records a customer or a marketplace writes into: a product title, an account name, a
delivery instruction, the free text on an order, the reason a buyer gives their bank.
"""


class ArenaError(RuntimeError):
    """The world could not do what was asked, and continuing would produce a false result."""


class PlantError(ArenaError):
    """A payload was aimed at a field that does not exist.

    Refused rather than created. A planted field the record never had is a field the agent
    was never going to read, so the attack would report as attempted and land nowhere, which
    reads on a coverage grid as a channel that was tested.
    """


class UnknownSessionError(ArenaError):
    """A session was asked about that has never been seen.

    Distinct from a session with no calls in it. Branching from a conversation that does not
    exist, or restoring one, is a wiring bug, and answering it with a fresh world would hand
    the caller a world it did not earn.
    """


class UnknownSourceError(ArenaError):
    """A declared data source does not correspond to any collection of this world.

    Refused rather than guessed at. A channel that named a source nothing backs would plant
    nowhere and report as attempted, which is the failure this whole module is arranged to
    make impossible.
    """


def collection_for(source: str) -> str:
    """The collection backing a declared data source.

    Args:
        source: The `data_source` name from a `ChannelDeclaration`.

    Returns:
        The collection name `plant` takes.

    Raises:
        UnknownSourceError: If nothing in this world backs that source.
    """
    collection = COLLECTION_FOR_SOURCE.get(source)
    if collection is None:
        raise UnknownSourceError(
            f"no collection of this world backs data source {source!r}. Backed: "
            f"{', '.join(sorted(COLLECTION_FOR_SOURCE))}."
        )
    return collection


@dataclass
class Arena:
    """Every world the tool server is holding, keyed by session.

    Attributes:
        sessions: The live worlds, by session id.
        checkpoints: One saved world per completed turn, by session, in turn order. A fork
            reads from here.
    """

    sessions: dict[str, World] = field(default_factory=dict)
    checkpoints: dict[str, list[World]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def world(self, session: str) -> World:
        """The session's world, seeded fresh the first time the session is seen.

        Creating on first sight is what gives a conversation its isolation without the
        runner having to ask for it, and it is the same rule the target used to apply.
        """
        with self._lock:
            if session not in self.sessions:
                self.sessions[session] = fresh_world()
            return self.sessions[session]

    def knows(self, session: str) -> bool:
        """Whether this session has a world."""
        with self._lock:
            return session in self.sessions

    def snapshot(self, session: str) -> World:
        """A detached copy of the session's world as it stands.

        Raises:
            UnknownSessionError: If the session has never been seen.
        """
        with self._lock:
            world = self.sessions.get(session)
            if world is None:
                raise UnknownSessionError(f"no world for session {session!r}")
            return deepcopy(world)

    def restore(self, session: str, snapshot: World | None = None) -> None:
        """Put a session's world back to a known state.

        Args:
            session: The session to reset. Need not exist yet.
            snapshot: The world to restore. Defaults to a freshly seeded shop, which is the
                baseline a planted attempt starts from. Deep copied on the way in, so the
                caller keeps a snapshot it can restore from again.
        """
        with self._lock:
            self.sessions[session] = fresh_world() if snapshot is None else deepcopy(snapshot)
            self.checkpoints.pop(session, None)

    def checkpoint(self, session: str) -> int:
        """Save the session's world as it stands at the end of a turn.

        Returns:
            How many checkpoints the session now has, which is the turn number a later fork
            can branch after.

        Raises:
            UnknownSessionError: If the session has never been seen.
        """
        with self._lock:
            world = self.sessions.get(session)
            if world is None:
                raise UnknownSessionError(f"no world for session {session!r}")
            saved = self.checkpoints.setdefault(session, [])
            saved.append(deepcopy(world))
            return len(saved)

    def branch(self, source: str, session: str, at_turn: int | None = None) -> None:
        """Give a new session the source's world as it stood after `at_turn` turns.

        Args:
            source: The session to branch from.
            session: The id for the branch.
            at_turn: How many completed turns the branch keeps. `None` takes the latest
                checkpoint.

        Raises:
            UnknownSessionError: If the source has no world.
            ArenaError: If the branch id is already in use, or the source has not completed
                that many turns. Both would silently hand a conversation somebody else's
                money.
        """
        with self._lock:
            if source not in self.sessions:
                raise UnknownSessionError(f"no world for session {source!r} to branch from")
            if session in self.sessions:
                raise ArenaError(f"session {session!r} already has a world")
            saved = self.checkpoints.get(source, [])
            taken = len(saved) if at_turn is None else at_turn
            if not 1 <= taken <= len(saved):
                raise ArenaError(
                    f"cannot branch after {taken} turn(s) of a conversation with {len(saved)}"
                )
            self.sessions[session] = deepcopy(saved[taken - 1])
            self.checkpoints[session] = [deepcopy(world) for world in saved[:taken]]

    def forget(self, session: str) -> None:
        """Drop a session's world once its conversation is over.

        The record of what happened lives in the recorder and is not affected. This frees
        the copy of the shop, which is the large object.
        """
        with self._lock:
            self.sessions.pop(session, None)
            self.checkpoints.pop(session, None)

    def plant(
        self, session: str, *, collection: str, record_id: str, field_name: str, payload: str
    ) -> str:
        """Write attacker-controlled text into a field of the world.

        The planted channel's second step. What makes it an attack rather than a test
        fixture is where it is written: a field a customer genuinely fills in, which the
        agent later reads as context with nothing marking it as hostile.

        Args:
            session: Whose world to write into.
            collection: One of `PLANTABLE_COLLECTIONS`.
            record_id: The key within that collection.
            field_name: The field to overwrite. Must already exist on the record.
            payload: The text to write.

        Returns:
            What the field held before, so a run can show the field's ordinary content
            beside what replaced it.

        Raises:
            PlantError: If the collection is not plantable, the record does not exist, or
                the field does not exist on it.
        """
        if collection not in PLANTABLE_COLLECTIONS:
            raise PlantError(
                f"{collection!r} is not a field an adversary writes into. Plantable: "
                f"{', '.join(PLANTABLE_COLLECTIONS)}."
            )
        world = self.world(session)
        records: dict[str, dict[str, Any]] = getattr(world, collection)
        record = records.get(record_id)
        if record is None:
            raise PlantError(f"{collection} has no record {record_id!r} to plant into")
        if field_name not in record:
            raise PlantError(
                f"{collection}[{record_id!r}] has no field {field_name!r}. A planted field "
                f"the record never had is a field the agent will never read."
            )
        previous = record[field_name]
        record[field_name] = payload
        return "" if previous is None else str(previous)
