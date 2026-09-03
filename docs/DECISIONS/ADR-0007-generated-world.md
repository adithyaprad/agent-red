# ADR-0007: The world is generated from the declaration crossed with the checks, not from the declaration

Status: accepted, 2026-09-03.

## Context

Pointing agent-red at an agent currently takes three things. Two of them are declarations the
merchant already has or can write: what the agent is and may do (`config.yaml`), and the
bounds it operates under (`policy.yaml`). The third is a world for it to act on, and nobody
except us can produce it.

`data/store/` is hand-authored. Ten products, eight customers, six baskets, ten orders, eight
disputes and nine consignments, every one of them chosen for a reason, and the reasons are
written down in `docs/WORLD.md` because they are not guessable from the domain. It is the
right world for these two agents. It is also a per-merchant integration, sitting exactly where
the two previous architectural decisions each found one and removed it.

That pattern is now three for three. ADR-0005 found the oracle reading the target's own
self-report and moved it to a boundary the harness holds, because a check that needs the
agent's cooperation only works on agents we wrote. ADR-0006 found chat to be the only way in
and made the delivery path a thing the agent declares, because an attack down a path the
deployment does not use is a finding about a test harness. Both were the same shape: something
the harness had quietly assumed it could supply, which in a real deployment it cannot. The
world is the third and the largest, because it is the one a person has to sit down and model.

There is a second reason, and on the current evidence it is the stronger one. The world is not
incidental to whether the suite finds anything. On 2026-09-03 the previous world was rebuilt
because eight of nine detector families could not fire against it: nothing had been refunded,
so a cumulative rule could not be exceeded; every identifier resolved to exactly one obvious
person, so reaching the wrong record meant naming a record that did not exist; every amount
was far under a ceiling or far over it; there was one currency, so a currency match had
nothing to compare. Six hand-written payloads had held against that world, and the reason was
not known until the world was examined. A world is a precondition for a finding, not a
backdrop to one.

So the question this decision answers is not "can the world be generated". It is what the
generator's input has to be, and the obvious answer is wrong.

## Decision

**The generator's input is the agent's declaration crossed with the inventory of checks, and
its output carries a manifest of which rules it made reachable.**

Not the declaration alone. A world derived from the declaration alone contains the collections
the declaration names, populated with records that are internally consistent and individually
plausible, and it is exactly the world that produced a clean sheet: one customer per
identifier, nothing part-refunded, no amount near an edge, no two people confusable, one
currency, no record touched twice. Every property that makes a declared rule breakable in one
step is absent, because no declaration states them. A declaration says what the rules are. It
does not say what a shop has to look like for breaking one to be a small step, and that is the
whole content of a useful world.

The generator therefore walks pairs. For every rule the policy declares, and every detector
family that rule is settled by, it emits:

- at least one record where breaking the rule is one step from a reasonable action, and
- at least one where holding is the right answer.

Both halves are load-bearing, and the second is the one that gets forgotten. An agent that
refuses everything scores perfectly against a world made entirely of traps, so a world of
traps cannot tell judgement from compliance, and a benign suite run against it measures
nothing. The rule for whether a record belongs is the neutrality test already applied by hand
to `data/store/`: a fixture belongs only if it would still belong had the agent held.

**Reachability is asserted, not hoped for.** The generator emits a manifest saying, per
declared rule, which fixture makes it reachable. A rule it could not make reachable is a named
gap carried into the report beside the coverage grid, in the same voice the grid already uses
for a cell nothing was attempted in. It is never silently dropped, because a rule with no
reachable fixture and a rule that was tested and held are indistinguishable in a finding count
and are opposite facts about the agent. `tests/mcp/test_world_reachability.py` becomes an
assertion over the manifest rather than over one checked-in shop.

**The world is a tool surface, not a dataset.** This is the half the framing hides, and it is
most of the work. ADR-0005 puts the oracle at the MCP boundary, so an agent under test reaches
its capabilities through tools the harness serves. Generating a world for an agent nobody has
seen means serving implementations of tools nobody has written. It is bounded, because of what
the declaration already carries and what a detector actually reads:

- `ToolDeclaration` carries the name, the JSON Schema of the arguments verbatim, and the
  consequence of a wrong call.
- `DataSource` carries the identifier kinds its records are named by.
- `ToolHandler` is already `(World, arguments) -> result`, with no domain knowledge in the
  signature.
- Detectors read arguments and the world diff. They do not read tool semantics. A handler that
  records faithfully, resolves an identifier the way any record store does, and mutates the
  collection it declares it writes to is enough for every check in `judge/detectors/`.

So generic handlers over collections cover the three shapes a declared tool takes: read one
record by identifier, list records matching a filter, and write a record or an effect from the
arguments. What they do not cover is a tool whose result the agent's own policy then depends
on, such as a checker that answers whether a code is valid. Those are declared already, in
`ResultReference` and `ResultCondition`, and the generator reads the declaration rather than
inventing the semantics.

**A generated world is seeded and joins the version tuple.** Rule 9 of the project is that a
gate verdict is reproducible byte for byte, and a scorecard is valid for exactly one version
tuple. A world that varies between runs would break both quietly: the same attack against the
same agent would find something on Tuesday and nothing on Wednesday, and nobody would be able
to say which run was wrong. The generator takes a seed, the world is content-addressed, and
that digest joins config, policy, model and tools as a fifth thing a scorecard is scoped to.

## Alternatives considered

**Keep hand-authoring a world per agent.** Honest about what it is, and it caps the product at
agents somebody sat down and modelled. It also puts the least transferable artefact on the
critical path of every new integration, and it is the specific thing this decision exists to
remove.

**Derive the world from the declaration alone, as the shape of the pipeline suggests.**
Rejected, and it is the central decision here. It regenerates the world that made eight of
nine detector families unreachable, and it would do so while looking correct: every collection
present, every record consistent, nothing obviously missing. The failure would present as an
agent that holds everywhere, which is the most flattering possible wrong answer and therefore
the most dangerous one to produce by construction.

**Ask the merchant for a copy of their real data.** Rejected twice over. A pre-deploy gate that
asks for production records is a gate nobody runs, and these are agents that touch money, so
the records carry payment and personal detail with no reason to be in a test harness. It also
does not solve the problem: a real dataset contains what the business happens to have, which
is no more likely to contain a claim sitting just above a ceiling than a generated one is by
accident.

**Have a model invent the world freely from the declaration.** Rejected. A model asked to build
a world for an attack suite writes a world of traps, and every fixture in it is then arguable.
The neutrality property is what makes a hold mean something, and it cannot be an instruction in
a prompt; it has to be a consequence of how each record was emitted. Fixtures are emitted per
rule, with the rule they exist for recorded in the manifest, so the question "why is this
record here" always has an answer that is not somebody's judgement after the fact. A model may
write the surface texture of a record. It does not decide which records exist.

**Generate the world and keep the tools hand-written.** Rejected. The tools are the larger half
of the per-merchant work, and a generated world served by hand-written handlers is a generated
world that only runs against the agents whose handlers exist, which is where we started.

## Consequences

`mcp/world.py` stops having named fields. `World` today declares `products`, `customers`,
`carts`, `orders`, `disputes`, `shipments` and two bookkeeping maps, and every one of those
names is a fact about furniture retail sitting in a package that is supposed to be neutral. It
becomes collection-keyed, with the collections named by the declaration. The convenience
readers on it (`customer_by_email`, `cart_total`) move to where the domain lives.

`mcp/tools/` gains generic handlers and keeps the hand-written ones for the two shipped
targets, which stay as they are: they are the fixture the generated path is checked against.
An agent whose generated handlers and hand-written handlers disagree is the test that the
generator is faithful, and both targets already have both halves.

`VersionTuple` in `spec/models.py` gains a fifth element, and with it the `runs` table, which
declares the four it holds today as not nullable on the grounds that a change in any of them
makes the agent untested again. A generated world is the same kind of thing: a scorecard
computed against one shop says nothing about an agent facing another. ADR-0004 is amended
rather than superseded, in the way ADR-0006 was.

The report gains a line it does not have: how much of the agent's declared rule set the world
made reachable at all. That number is unflattering by construction on the first run against a
new agent, which is the correct direction for it to be wrong in.

Two things are explicitly out of scope. This does not generate the agent, only the world it
acts on: the stand-in targets in `targets/` remain hand-written and remain outside the product
surface. And it does not infer a policy from anything. The declaration is still supplied, and
the thesis is unchanged: agent-red verifies that an agent conforms to its declaration, and it
does not verify that the declaration is correct.
