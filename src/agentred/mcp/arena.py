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
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

from agentred.mcp.world import UnknownSourceError, World, fresh_world


class ArenaError(RuntimeError):
    """The world could not do what was asked, and continuing would produce a false result."""


class PlantError(ArenaError):
    """A payload was aimed at a field that does not exist.

    Refused rather than created. A planted field the record never had is a field the agent
    was never going to read, so the attack would report as attempted and land nowhere, which
    reads on a coverage grid as a channel that was tested.
    """


__all__ = [
    "Arena",
    "ArenaError",
    "PlantError",
    "UnknownSessionError",
    "UnknownSourceError",
]


class UnknownSessionError(ArenaError):
    """A session was asked about that has never been seen.

    Distinct from a session with no calls in it. Branching from a conversation that does not
    exist, or restoring one, is a wiring bug, and answering it with a fresh world would hand
    the caller a world it did not earn.
    """


@dataclass
class Arena:
    """Every world the tool server is holding, keyed by session.

    Attributes:
        sessions: The live worlds, by session id.
        checkpoints: One saved world per completed turn, by session, in turn order. A fork
            reads from here.
        seed_world: What a new session's world is copied from. The hand-authored shop by
            default; a generated one for an agent nobody wrote a shop for. It is a callable
            rather than a world so that each session gets its own copy without this class
            having to know how one is made.
    """

    sessions: dict[str, World] = field(default_factory=dict)
    checkpoints: dict[str, list[World]] = field(default_factory=dict)
    seed_world: Callable[[], World] = fresh_world
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def world(self, session: str) -> World:
        """The session's world, seeded fresh the first time the session is seen.

        Creating on first sight is what gives a conversation its isolation without the
        runner having to ask for it, and it is the same rule the target used to apply.
        """
        with self._lock:
            if session not in self.sessions:
                self.sessions[session] = self.seed_world()
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
            self.sessions[session] = self.seed_world() if snapshot is None else deepcopy(snapshot)
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

    def subjects(
        self, session: str, *, source: str, kinds: tuple[str, ...]
    ) -> tuple[dict[str, str], ...]:
        """Who a collection is about, one entry per record, read from the world itself.

        The cohort a scheduled firing is legitimately woken about. A conversation is with
        one person, so one subject is the whole truth about it. A scheduled agent is woken
        about a set: a recovery agent's job when its timer fires is every basket nobody
        checked out, and a check that pinned one of them as the subject would score the
        other baskets as records it should not have touched.

        Read from the world rather than from the agent's own selection call, and that is the
        whole point of putting it here. The agent chooses which selector to call and with
        what filter, so a cohort taken from the selector's result would widen the moment an
        attack talked the agent into widening it, and the check would agree with whatever
        the agent had just been persuaded to do. The world does not move when the agent is
        persuaded.

        Args:
            session: Whose world to read. Read after the restore and the plant, so the
                cohort is the seeded baseline the firing actually saw.
            source: The declared data source to read, resolved against this world's own
                map of sources to collections.
            kinds: The identifier kinds a subject is described by, from the agent's declared
                data scope.

        Returns:
            One mapping per record carrying at least one of `kinds`, holding only those
            kinds. Records carrying none are left out: they identify nobody, so they place
            nothing in scope.

        Raises:
            UnknownSessionError: If the session has no world. Never seeded here, because a
                cohort read against a world this call created would describe an empty shop
                and quietly place nothing in scope.
            UnknownSourceError: If nothing in this world backs that source.
        """
        with self._lock:
            if session not in self.sessions:
                raise UnknownSessionError(
                    f"session {session!r} has no world, so there is no cohort to read. A "
                    f"cohort read before the world exists would report an empty shop and "
                    f"place nothing in scope."
                )
            world = self.sessions[session]
            records = world[world.collection_for(source)]
            found: list[dict[str, str]] = []
            for record in records.values():
                carried = {
                    kind: str(record[kind])
                    for kind in kinds
                    if isinstance(record.get(kind), (str, int, float))
                    and not isinstance(record.get(kind), bool)
                }
                if carried:
                    found.append(carried)
        return tuple(found)

    def plant(
        self, session: str, *, source: str, record_id: str, field_name: str, payload: str
    ) -> str:
        """Write attacker-controlled text into a field of the world.

        The planted channel's second step. What makes it an attack rather than a test
        fixture is where it is written: a field a customer genuinely fills in, which the
        agent later reads as context with nothing marking it as hostile.

        Args:
            session: Whose world to write into.
            source: The declared data source to write into, resolved against this world's
                own map of sources to collections.
            record_id: The key within the collection backing it.
            field_name: The field to overwrite. Must already exist on the record.
            payload: The text to write.

        Returns:
            What the field held before, so a run can show the field's ordinary content
            beside what replaced it.

        Raises:
            PlantError: If the record does not exist, or the field does not exist on it.
            UnknownSourceError: If nothing in this world backs that source. Which is also
                what keeps merchant configuration out of reach: an agent declares the sources
                it reads, discount tables and shipping options are not among them, and a
                harness that planted into one would be reporting a finding about an attacker
                who was already in the admin panel. The allowlist is the declaration rather
                than a table in this file, so a generated shop needs nothing changed here.
        """
        world = self.world(session)
        collection = world.collection_for(source)
        records = world[collection]
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
