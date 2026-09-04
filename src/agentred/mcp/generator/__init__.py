"""A shop derived from what an agent declares, crossed with what the checks need.

Pointing agent-red at an agent takes three things. Two are declarations a merchant has or can
write: what the agent is and may do, and the bounds it operates under. The third is a world
for it to act on, and until this package existed nobody except us could produce one:
`data/store/` is hand-authored, every record chosen for a reason written down, and none of it
transfers. It was the third per-merchant integration the architecture found and the largest,
because it is the one a person has to sit down and model.

**The input is the declaration crossed with the inventory of checks, and the obvious reading
is wrong.** A world derived from the declaration alone holds every collection the declaration
names, populated with records that are internally consistent and individually plausible, and
none of the properties that make a rule breakable in one step. It is the world eight of nine
checks could not fire against, and it fails by presenting an agent that holds everywhere. See
ADR-0007, and `emit.py` for the two questions asked of every declared rule.

**What comes out is a world and a manifest.** The manifest says which fixture made each rule
reachable and names every rule nothing could. A rule with no reachable fixture and a rule that
was tested and held are opposite facts about an agent and identical in a finding count, so the
gaps travel into the report rather than being dropped.

**It is seeded, and the world joins the version tuple.** The same seed and the same
declaration produce the same shop byte for byte. A world that varied between runs would make
the same attack find something on Tuesday and nothing on Wednesday with nobody able to say
which run was wrong, and a scorecard computed against one shop says nothing about an agent
facing another.

Out of scope, stated rather than assumed: this does not generate the agent, and it does not
infer a policy. The declaration is still supplied and the thesis is unchanged. agent-red
verifies that an agent conforms to its declaration. It does not verify that the declaration is
correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentred.mcp.generator.cast import cast, unsupported
from agentred.mcp.generator.emit import emit
from agentred.mcp.generator.manifest import Fixture, Gap, Manifest, Note, Reach, digest_of
from agentred.mcp.generator.shape import CollectionShape, FieldKind, FieldShape, shapes_for
from agentred.mcp.tools.base import ToolSet
from agentred.mcp.tools.generic import toolset_for
from agentred.mcp.world import World
from agentred.spec.models import AgentSpec, Subject

DEFAULT_SEED = 20260903
"""What a world is generated from when nobody says. Fixed rather than random, because a
default that varied would be a world that varied, and rule 9 of this project is that a verdict
is reproducible byte for byte."""


@dataclass(frozen=True)
class GeneratedWorld:
    """A shop, and the account of what it made reachable.

    Attributes:
        world: The shop itself, ready for the tool server to act on.
        manifest: Which fixture makes each rule reachable, and which rules nothing could.
        shapes: What each collection's records carry, kept so a caller can say why a field is
            there without re-deriving it.
        subjects: Identities the harness may act as in this shop. Derived here rather than
            read from the declaration, because a subject names records and the records are
            these. A cast written against a different shop names nothing that exists.
        unsupported: Declared channels no identity here can be attacked down, with what the
            declaration did not say. The same argument as a gap, one level out: a channel
            nobody can be attacked through and a channel that was attacked and held are
            identical on a coverage grid.
    """

    world: World
    manifest: Manifest
    shapes: dict[str, CollectionShape]
    subjects: tuple[Subject, ...] = ()
    unsupported: tuple[tuple[str, str], ...] = ()

    def spec_for(self, spec: AgentSpec) -> AgentSpec:
        """The declaration as it applies to this shop: its own subjects, everything else kept.

        The version tuple is deliberately untouched. Who the harness may act as is a fixture
        rather than a rule, and a scorecard is valid for a declaration and a world, both of
        which this leaves exactly as they were.
        """
        return spec.model_copy(update={"subjects": self.subjects})

    def tools(self, spec: AgentSpec) -> ToolSet:
        """The tool surface this world is reached through, served from the declaration.

        A generated world served by hand-written handlers is a generated world that only runs
        against the agents whose handlers exist, which is where this started.
        """
        return toolset_for(spec)


def generate(spec: AgentSpec, seed: int = DEFAULT_SEED) -> GeneratedWorld:
    """Build a world for one agent, and say what it made reachable.

    Args:
        spec: The validated spec. Its data sources become the collections, its tool behaviours
            and its rules say what those collections carry, and its rules decide what the
            values are.
        seed: What the emitted values derive from. The same seed and declaration produce the
            same shop.

    Returns:
        The world, its manifest, and the collection shapes behind both.

    The collections are named by the declaration, so the world's own map from a declared data
    source to the collection backing it is the identity. There is nothing else for it to be:
    the alternative is a shop whose internal names differ from what the agent asked for, which
    is a thing a person writes and not a thing a generator has any basis to invent.
    """
    shapes = shapes_for(spec)
    shop = emit(spec, shapes, seed)
    collections = {name: rows for name, rows in shop.rows.items()}
    world = World(
        collections=collections,
        sources={name: name for name in collections},
    )
    manifest = Manifest(
        seed=seed,
        digest=digest_of(collections),
        fixtures=tuple(shop.fixtures),
        gaps=tuple(shop.gaps),
        notes=tuple(shop.notes),
    )
    subjects = cast(spec, shop)
    return GeneratedWorld(
        world=world,
        manifest=manifest,
        shapes=shapes,
        subjects=subjects,
        unsupported=unsupported(spec, subjects),
    )


__all__ = [
    "DEFAULT_SEED",
    "CollectionShape",
    "FieldKind",
    "FieldShape",
    "Fixture",
    "Gap",
    "GeneratedWorld",
    "Manifest",
    "Note",
    "Reach",
    "Subject",
    "generate",
    "shapes_for",
]
