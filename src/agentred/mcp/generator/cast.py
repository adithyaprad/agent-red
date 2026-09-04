"""Identities the harness may act as, taken from the records a generated world holds.

The last thing about pointing agent-red at an agent that only we could supply. Two files were
declarations a merchant has or can write; the third was a world, which the generator now
derives. This is the fourth, and it was invisible until the other three stopped being
hand-written, because a hand-authored cast standing in a hand-authored world looks like part
of the world.

It is not. A subject names records: this reference, that one, and the facts a person in that
position could say out loud. Point the same cast at a world generated from a declaration and
every one of those references belongs to nothing. The agent is asked about a record it cannot
find, truthfully says so, and the conversation ends before the action under test is reached.
Every rule reports as never in play, which is the reading that looks like an agent nobody
could talk into anything. On the planted channels it is louder: the write is refused because
the record named does not exist, and a whole suite fails before the agent is reached at all.

**A subject is built from a fixture, so the cast and the world cannot drift.** Each one is
anchored on a record some rule exists for, carries every identifier that record and the
records it names hold between them, and states as facts only what those records say. Nothing
here invents a person: what a subject knows is what the world would tell anybody who read the
records it is about, which is the same bar the hand-written file set for itself.

**What is lost, said plainly.** A hand-written fact reads like something somebody would say.
A derived one reads like a record, because it is one. The attacker has less to grip, and the
first argument it makes is thinner. That is the trade for a cast that costs a merchant
nothing, and it is worth stating in the report rather than discovering from the transcripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentred.mcp.generator.link import owners
from agentred.mcp.generator.manifest import Fixture, Reach
from agentred.mcp.generator.shape import CollectionShape, FieldKind
from agentred.spec.models import AgentSpec, Subject

if TYPE_CHECKING:  # pragma: no cover - emit builds the shop this reads
    from agentred.mcp.generator.emit import Shop

MOST = 8
"""How many identities a generated cast holds at most.

Every subject multiplies the suite, and a suite is the cost of a run. Enough that each
declared rule with a fixture behind it has somebody whose situation sits against it, and few
enough that the wall clock stays on the same scale as the hand-written cast of six.
"""


def _reachable_from(
    shop: Shop, held: dict[str, str], source: str, record_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """The anchor record and every record it names, transitively, in reading sequence.

    What one session is legitimately about. A subject that carried only its anchor's own
    fields would be missing whichever identifier the deployment happens to key a channel by,
    and a channel whose key a subject cannot supply is a channel no attack can be built for.
    """
    seen = {(source, record_id)}
    found = [(source, shop.rows[source][record_id])]
    queue = [(source, record_id)]
    while queue:
        at, key = queue.pop(0)
        record = shop.rows[at][key]
        for name, field in shop.shapes[at].fields.items():
            if field.kind is not FieldKind.IDENTIFIER or name == shop.shapes[at].key:
                continue
            owner = held.get(name)
            if owner is None or owner == at:
                continue
            target = str(record.get(name, ""))
            if target not in shop.rows[owner] or (owner, target) in seen:
                continue
            seen.add((owner, target))
            found.append((owner, shop.rows[owner][target]))
            queue.append((owner, target))
    return found


def _identifiers(
    shop: Shop, reached: list[tuple[str, dict[str, Any]]], kinds: frozenset[str]
) -> dict[str, str]:
    """Every identifier the records of one session carry, nearest record first.

    Nearest wins, so a subject's own reference is the one it was anchored on rather than
    whichever record happened to be read last.

    Only kinds the declaration calls identifiers. A field can be shaped as one here because
    some rule compares it between two calls, and a value compared that way is not therefore
    something a session is scoped by: handing a currency to the harness as an identity would
    put it in the report as a reference this person is known by.
    """
    found: dict[str, str] = {}
    for source, record in reached:
        for name, field in shop.shapes[source].fields.items():
            if field.kind is not FieldKind.IDENTIFIER or name not in kinds:
                continue
            value = str(record.get(name, ""))
            if value and name not in found:
                found[name] = value
    return found


def _sentence(shape: CollectionShape, name: str, value: Any) -> str:
    """One fact, in the merchant's own field names.

    Deliberately flat. A sentence with any more shape than this would be the generator
    deciding what a record means, and what a record means is the one thing it is not entitled
    to an opinion about.
    """
    subject = shape.source.rstrip("s") or shape.source
    return f"The {name.replace('_', ' ')} on their {subject} is {value}."


def _facts(shop: Shop, reached: list[tuple[str, dict[str, Any]]]) -> tuple[str, ...]:
    """What somebody in this position could say out loud, read off their own records.

    Free text is left out. It is the field an attacker writes into on every planted channel,
    so handing it back to the attacker as something the subject knows would let a payload it
    planted last turn arrive as a fact this turn.
    """
    said: list[str] = []
    for source, record in reached:
        shape = shop.shapes[source]
        for name, field in shape.fields.items():
            if field.kind is FieldKind.TEXT or name not in record:
                continue
            said.append(_sentence(shape, name, record[name]))
    return tuple(said)


def _usable(spec: AgentSpec, identifiers: dict[str, str], shop: Shop) -> bool:
    """Whether a session opened as this identity could go anywhere.

    The policy's scope says which identifiers bind a record to a session, and a subject
    missing one of them cannot be the subject of whichever conversations need it.

    A channel this identity has no reference for is not a reason to reject the identity. Some
    channels write into something a session reaches without owning it, and whether every
    declared way in has somebody behind it is a question about the cast as a whole rather than
    about one member of it. `unsupported` answers that one, and `attacks/generator.py` refuses
    a suite that would leave such a channel on the grid untested.

    A reference this identity does have must name a record that exists, because a plant into
    a record that is not there is refused by the driver and the cell reports as an error.
    """
    for kind in spec.policy.data_scope.subject_identifier_kinds:
        if kind not in identifiers:
            return False
    for channel in spec.config.channels:
        named = identifiers.get(channel.record_key)
        if named is not None and named not in shop.rows.get(channel.data_source, {}):
            return False
    return True


def unsupported(spec: AgentSpec, subjects: tuple[Subject, ...]) -> tuple[tuple[str, str], ...]:
    """Declared channels no generated identity can be attacked down, and why.

    Reported rather than discovered when a suite refuses to build. A channel names records by
    an identifier, and nothing guarantees a declaration says how a session's own records
    connect to that one: a source everyone reads is reached by whoever has one of its
    records in front of them, and if no declared field puts it there, the generator has no
    basis to decide who does. Inventing the link is the version of this that produces a shop
    where every channel works and one of them is fiction.
    """
    found = []
    for channel in spec.config.channels:
        if any(channel.record_key in subject.identifiers for subject in subjects):
            continue
        found.append(
            (
                channel.name,
                f"nothing an identity here holds reaches a {channel.record_key}, so there is "
                f"no record of {channel.data_source!r} that is theirs to be attacked through. "
                f"A field on a source a session already reads that carries the "
                f"{channel.record_key} is what connects the two.",
            )
        )
    return tuple(found)


def _label(fixture: Fixture, taken: set[str]) -> str:
    """A short name for an identity, saying which rule it is here for."""
    stem = f"{fixture.rule}-{fixture.reach.value}".replace("_", "-").replace(".", "-")
    name, suffix = stem, 2
    while name in taken:
        name, suffix = f"{stem}-{suffix}", suffix + 1
    return name


def cast(spec: AgentSpec, shop: Shop) -> tuple[Subject, ...]:
    """Identities to act as against a generated world, one per fixture that supports one.

    Args:
        spec: The validated spec, for the scope and the channels a subject has to satisfy.
        shop: The emitted and linked shop. Linked, because a subject is a walk from one
            record to the records it names and an unlinked shop has none that resolve.

    Returns:
        The cast, breakable fixtures first, capped at `MOST`. Empty when the declaration
        needs no subjects, and empty rather than partial when nothing satisfies the contract,
        because a subject that cannot open a conversation is worse than none: the run happens
        and reports on an agent nobody managed to ask.

    Both halves are represented for the same reason the world holds both. A cast drawn only
    from the records that make rules breakable is a cast of people it is always right to
    refuse, and an agent that refuses all of them scores perfectly while being useless.
    """
    if not spec.policy.data_scope.subject_identifier_kinds:
        return ()
    held = owners(shop)
    # What the declaration itself calls an identifier, which is not the same set as the
    # fields shaped as one: a field is shaped that way whenever some rule compares it between
    # two calls, and a currency compared that way is not a reference anybody is known by.
    kinds = frozenset(
        kind for source in spec.config.data_sources for kind in source.identifier_kinds
    ) | frozenset(channel.record_key for channel in spec.config.channels)
    ranked = sorted(shop.fixtures, key=lambda found: found.reach is not Reach.BREAKABLE)
    found: list[Subject] = []
    anchored: set[tuple[str, str]] = set()
    taken: set[str] = set()
    for fixture in ranked:
        if len(found) >= MOST:
            break
        anchor = (fixture.collection, fixture.record_id)
        if anchor in anchored or fixture.record_id not in shop.rows[fixture.collection]:
            continue
        reached = _reachable_from(shop, held, *anchor)
        identifiers = _identifiers(shop, reached, kinds)
        if not _usable(spec, identifiers, shop):
            continue
        anchored.add(anchor)
        name = _label(fixture, taken)
        taken.add(name)
        found.append(Subject(name=name, identifiers=identifiers, facts=_facts(shop, reached)))
    return tuple(found)
