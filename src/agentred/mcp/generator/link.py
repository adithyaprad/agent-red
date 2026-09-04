"""References between generated records that resolve to records that exist.

The half of a generated world that is invisible until an agent walks it. Every emitter here
builds one record to answer one rule, and a record's own fields are enough for a limit: a
figure just over a ceiling is checkable without leaving the record it sits on. Almost nothing
else is. An action that has to follow a read of the record it acts on, a message that may
carry only this party's details, a figure read off one record before the call that is bounded
by it. Every one of those needs the agent to start at one record and arrive at another.

Emitted independently, they do not connect. Each identifier field is minted fresh from its own
counter, so a record names a second record nothing holds, that one names a third nothing
holds, and a record whose own name is a reference is filed against something that does not
exist. The agent reads the first record, follows the reference it was given, is truthfully
told there is no such record, and stops. Every rule that needed the second read reports as
never evaluated, and the manifest goes on saying the rule was reachable, because reachability
was decided when the record was emitted and nothing since has tried to walk from one to
another.

**What this pass does not do is invent a relationship.** It rewrites references so that
records which already named the same thing still name the same thing, and what they name now
exists. The equivalence classes are the emitters' own: two records a fixture deliberately
made agree on a reference stay agreeing, and two the fixture deliberately made differ stay
differing. Nothing is merged and nothing is split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentred.mcp.generator.shape import CollectionShape, FieldKind

if TYPE_CHECKING:  # pragma: no cover - import cycle, emit calls link at the end of a shop
    from agentred.mcp.generator.emit import Shop


def owners(shop: Shop) -> dict[str, str]:
    """Identifier kind to the collection whose records are named by it.

    The first declared collection keyed by a kind owns it. A second collection keyed by the
    same kind is a dependent rather than a rival: a record named by another collection's
    reference is a fact about that record, and its key has to be one the owner holds or it
    describes something that is not there.

    The sequence they were declared in decides it, because nothing else can. A generator that
    picked the larger collection, or the one with more fields, would be reading an ownership
    decision out of data it generated itself.
    """
    found: dict[str, str] = {}
    for shape in shop.shapes.values():
        if shape.key and shape.key not in found:
            found[shape.key] = shape.source
    return found


def _references(shape: CollectionShape, held: dict[str, str]) -> list[str]:
    """Every field of this collection that names a record in another one.

    A collection's own key is excluded: it names this record rather than another. So is a
    field whose kind this collection owns, which would be a record referring to its own
    collection and is not something any declaration here describes.
    """
    return [
        name
        for name, field in shape.fields.items()
        if field.kind is FieldKind.IDENTIFIER
        and name != shape.key
        and held.get(name) not in (None, shape.source)
    ]


def derived(shop: Shop) -> dict[str, frozenset[str]]:
    """Per collection, the identifier fields taken from the records it names.

    What an emitter is not free to set. A field copied from a referenced record holds
    whatever that record says whatever the emitter wrote, so a fixture that encodes a
    relationship in one of these encodes nothing: the two records it made agree are pulled
    apart again by the records they point at, and the manifest still says the rule is
    reachable because reachability was decided before anything was linked.
    """
    held = owners(shop)
    found: dict[str, set[str]] = {name: set() for name in shop.shapes}
    for shape in shop.shapes.values():
        for name in _references(shape, held):
            for other, field in shop.shapes[held[name]].fields.items():
                if field.kind is not FieldKind.IDENTIFIER:
                    continue
                if other in (name, shape.key) or other not in shape.fields:
                    continue
                found[shape.source].add(other)
    return {name: frozenset(fields) for name, fields in found.items()}


def settable(shop: Shop, source: str) -> frozenset[str]:
    """The identifier fields of one collection an emitter may use to relate two records.

    Everything it carries, less the one it is named by and the ones it copies from elsewhere.
    """
    shape = shop.shapes[source]
    identifiers = {
        name for name, field in shape.fields.items() if field.kind is FieldKind.IDENTIFIER
    }
    return frozenset(identifiers - {shape.key} - derived(shop)[source])


def _top_up(shop: Shop, source: str, wanted: int) -> None:
    """Give a collection at least `wanted` records, so every dependent can name one.

    Deliberately not fixtures. These exist so a reference resolves, which is the same reason
    the floor of ordinary records exists, and calling them fixtures would put records in the
    manifest with no rule behind them.
    """
    shape = shop.shapes[source]
    while len(shop.rows[source]) < wanted:
        record = shop.blank(source)
        shop.rows[source][str(record[shape.key])] = record


def _rekey_dependents(shop: Shop, held: dict[str, str]) -> None:
    """Re-key a collection whose own key names a record in another collection.

    A collection whose key is another collection's reference has to carry one of that
    collection's keys. Left alone it carries a value minted from the same counter and
    belonging to nothing, so the tool that fetches one of its records answers with an error
    for every reference there is, and a channel that plants into one of its fields writes
    into a record no trigger will ever load.
    """
    for shape in shop.shapes.values():
        owner = held.get(shape.key) if shape.key else None
        if owner is None or owner == shape.source:
            continue
        rows = shop.rows[shape.source]
        if not rows:
            continue
        _top_up(shop, owner, len(rows))
        available = sorted(shop.rows[owner])
        moved: dict[str, str] = {}
        rekeyed: dict[str, dict[str, Any]] = {}
        for index, (old, record) in enumerate(sorted(rows.items())):
            new = available[index]
            record[shape.key] = new
            rekeyed[new] = record
            moved[old] = new
        # One for every record of the collection that names them, not just enough to re-key
        # what was emitted. A dependent that covers only some of the owner's records makes
        # whether a party can be attacked depend on which of the owner's records they were
        # anchored on, which is a property of emission sequence and of nothing anybody
        # declared. Where such a collection is what a channel plants into, that decides
        # silently which identities that channel can reach at all.
        for key in available[len(rekeyed) :]:
            filler = shop.blank(shape.source)
            filler[shape.key] = key
            rekeyed[key] = filler
        shop.rows[shape.source] = rekeyed
        shop.relabel(shape.source, moved)


def _resolve(shop: Shop, held: dict[str, str]) -> None:
    """Point every reference at a record that exists, keeping who agreed with whom.

    One class per distinct value already in the shop, so records the emitters made agree keep
    agreeing and records they made differ keep differing. Classes take distinct targets while
    there are distinct targets to take, because two parties a rule needs to be confusable
    stop being two if they collapse onto one record.
    """
    chosen: dict[tuple[str, str], str] = {}
    taken: dict[str, list[str]] = {}
    for shape in shop.shapes.values():
        naming = _references(shape, held)
        if not naming:
            continue
        for _, record in sorted(shop.rows[shape.source].items()):
            for name in naming:
                owner = held[name]
                pool = sorted(shop.rows[owner])
                if not pool:
                    continue
                seen = (name, str(record.get(name, "")))
                if seen not in chosen:
                    used = taken.setdefault(name, [])
                    free = [key for key in pool if key not in used]
                    picked = free[0] if free else pool[len(used) % len(pool)]
                    used.append(picked)
                    chosen[seen] = picked
                record[name] = chosen[seen]


def _agree(shop: Shop, held: dict[str, str]) -> None:
    """Make a record agree with the record it names about who that is.

    A reference resolving is not the same as the two records describing one party. A record
    naming a second one still carries references of its own minted from a counter, so a check
    asking whether a message carried this party's details compares two spellings of the same
    person and calls them strangers. Whatever a referenced record says about identity, the
    record naming it repeats.

    Run to a fixed point because references chain: the first record names a second, the second
    names a third, and the first one's idea of the third is only right once the second one's
    is. Capped by the number of collections, which is the longest chain a world of that size
    can hold.
    """
    for _ in range(len(shop.shapes) + 1):
        settled = True
        for shape in shop.shapes.values():
            naming = _references(shape, held)
            for record in shop.rows[shape.source].values():
                for name in naming:
                    target = shop.rows[held[name]].get(str(record.get(name, "")))
                    if target is None:
                        continue
                    for other, field in shop.shapes[held[name]].fields.items():
                        if field.kind is not FieldKind.IDENTIFIER:
                            continue
                        if other in (name, shape.key) or other not in record:
                            continue
                        if record[other] != target[other]:
                            record[other] = target[other]
                            settled = False
        if settled:
            return


def link(shop: Shop) -> None:
    """Make every reference in an emitted shop resolve, in place.

    Runs after everything is emitted rather than during it, because a record emitted for the
    first rule cannot name one the eleventh rule has not produced yet, and an emitter that had
    to care about that would be an emitter that knows about rules other than its own.
    """
    held = owners(shop)
    _rekey_dependents(shop, held)
    _resolve(shop, held)
    _agree(shop, held)
