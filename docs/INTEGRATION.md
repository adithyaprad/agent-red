# Integrating an agent

agent-red verifies that an agent conforms to its declaration. It does not verify that the
declaration is correct. Everything in this document follows from that sentence: the reader
described here is allowed to find a declaration, and is not allowed to invent one and then
report an agent for breaking it.

## What the harness needs

Two objects, and nothing else.

**The config** says what the agent is and can do: its tools and their argument schemas, the
data sources it can reach, the fields an adversary writes and the trigger that reads them.

**The policy** says what it must stay inside: bounds on arguments, steps required before
other steps, the data one session may touch, and the obligations a reply carries.

Everything downstream derives from those two. The shop the agent is tested in is generated
from them crossed with the inventory of checks. The stakes are derived from them. The attack
lattice is goals crossed with channels crossed with techniques, none of which carry domain
vocabulary. The detectors read the tool-call stream and settle a rule the policy stated. So an
integration is finished the moment the declaration is real, and no part of the harness needs
to be told anything about the agent's domain.

## Where a declaration already exists

The declaration is the one artifact agent-red cannot run without, and it is the one artifact
nobody sets out to write. It does not need to be written. An agent that runs at all has
already recorded almost all of it, in five places, for its own reasons.

| Source | Supplies | Origin |
|---|---|---|
| Tool connectors | tools, argument schemas | declared |
| Instance configuration | bounds: limits, thresholds, approval points | declared |
| Workflow definition | preconditions: which step must precede which | declared |
| Catalogue manifest | data sources, data scope, connectors, trigger | declared |
| Instructions | whatever the four above have no field for | inferred |

One part of a spec is not on that list and never will be. **Subjects, the identities the
harness may act as, are test fixtures rather than rules.** A platform has no reason to record
who a red team may impersonate, so they are supplied rather than recovered, and a policy that
scopes a session by identifier is refused without them: every rule would report as never
evaluated rather than as passed.

Four of the five are structured, and are read exactly rather than interpreted. The fifth is
prose, is read by a model, and is drafted for a person to confirm. The difference between the
two is carried on every rule from the moment it is read to the moment it appears on a
scorecard, and it is never allowed to blur.

## Source by source

### Tool connectors

An agent that reaches its capabilities through connectors has published the exact tools it can
call and the exact arguments each one takes. That is the attack surface, stated by the platform
itself, in a form that cannot be wrong: an agent whose connector advertised the wrong schema
could not call the tool.

`src/agentred/ingest/adapters/mcp.py` reads this over the MCP protocol. It lists and never
invokes, which is a rule rather than an implementation detail. Verifying a tool by calling it
is right for a tool server the harness owns, where money actions are held in test mode by
construction. It is indefensible against an integrator's agent, where the tool on the other end
of the connector is a production API and the reader has no way to know whether the credentials
behind it are live. A reader that moves money to confirm a schema has caused the harm the
harness was brought in to prevent, so a test fails if a call ever reaches the server.

### Instance configuration

This is where the money rules live, and the reason is structural rather than lucky.

A platform that lets an operator configure an agent without writing code has to ask them for
the limits, and has to store the answers somewhere it can enforce them. The skill is generic:
one dispute responder, the same prompt and the same workflow for everyone who installs it. What
differs between one merchant's instance and another's is the ceiling on a refund, the threshold
above which a person approves, and which actions are switched off. That difference is precisely
what a per-instance configuration is for, and it is precisely what the policy half needs.

So the rules hardest to recover from a hand-built agent are the ones best recorded by a no-code
one. A wizard that asks "what is the largest refund this agent may issue without approval" has
produced a structured policy statement as a side effect of being a wizard.

### Workflow definition

A workflow engine exists to make ordering deterministic: a step cannot run before the step in
front of it. That property is the reason a platform migrates onto one, and it is also, read
from the outside, a policy statement. A workflow in which the step that reads an order precedes
the step that refunds it has declared that a refund follows a read of the order it refunds.

This matters more than the count of rules it recovers. Preconditions are the shape of rule
prose is worst at expressing and attacks are best at breaking: a rule about a pair of calls
rather than about the arguments of either one. Reading them off the step graph moves them out
of the half that is guessed and into the half that is read.

The recoverable part is the ordering, and which tools each step is given. A step that carries
its tools declares them; a step whose behaviour is closed over inside a function does not, and
the ordering is still read while the tools are not. That limit is reported per step rather than
worked around.

### Catalogue manifest

The manifest an agent is installed from names the connectors it requires, the data it is
allowed to reach, and the entry point that starts it. Those map onto data sources, the data
scope one session may touch, and the trigger: an agent that wakes on a schedule and an agent
that runs when a request names a record are attacked down different paths, and which one it is
comes from the manifest rather than from a guess.

### Instructions

What is left is the residue: the rules an operator stated in prose because the configuration
had no field for them. A model reads them, and every rule it produces names a tool and an
argument, both of which are checked against the agent's declared surface. A rule naming
something that does not exist is refused rather than reported, because a model inventing a rule
and a model then finding it broken is the failure this whole path invites.

The residue is small and it is not incidental. Both agents built here were written by people
who knew exactly what they were building, and both ended up with rules in their instructions
that never reached their policy. A rule stated in prose and never configured is a finding
before any attack runs, because nothing in the deployment enforces it.

## What is not readable, and is asked rather than assumed

Three things are missing from every source above, and each is carried as a question with the
subject named, rather than as a value.

**What a wrong call costs.** No connector protocol carries it. It is the field the entire
stakes lattice is built from, so guessing it is not a small error: a tool recorded as costing
nothing is a tool no attack is aimed at, and the run then reports a clean sheet against an
agent that was never tested.

**Bounds no configuration has a field for.** A per-call ceiling is a wizard question. A limit
on everything paid back against one order added up is not, and neither is a limit whose value
is read from the record being acted on rather than passed as an argument. Those are the rules
an agent talked into instalments defeats, and they are drafted from prose or proposed as a
universal, never read.

**Which declared fields an adversary actually writes.** A data source says what the agent can
read. It does not say which of those fields reached the database from somebody outside the
merchant. That is the difference between a channel and a column, and it is answered by the
operator.

## An agent to try it on

`examples/retention_desk/` holds a subscription retention agent as an install wizard leaves it,
and nothing in that directory was written for agent-red. It is the shortest way to see what
this document describes actually run: serve the connector, read the agent, and watch the read
refuse to write a declaration until the questions it raises are answered.

Its `README.md` carries the commands. The four files an operator supplies alongside it are the
answers themselves, kept in files rather than prompted for, so that a read reproduces and what
a person confirmed sits beside what a reader recovered.

## What this recovers, measured

`dispute_handler` has a declaration a person wrote, so the claim that one need not be written
is checkable rather than arguable. `tests/ingest/installed/` holds the same agent as a builder
installs it: a manifest, the limits an operator configured, the prose it runs on, and the
workflow an installer mapped it onto. None of those four files were written for agent-red.
Reading them and comparing the result against the authored declaration gives this.

```
dispute_handler: 32 matched, 0 diverged, 9 not covered by any reader that ran,
                 7 found that the author did not declare
```

Every tool, every argument schema and every description came back identical. Five of the
fifteen authored rules came back, and none of the fifteen came back differently.

**What did not come back is the interesting half, and it was predicted before it was
measured.** The instance reader names, on every read, the three rule shapes a form field
structurally cannot hold. The seven rules it missed are those shapes and nothing else: a limit
relative to another value, a limit on a running total, a limit whose value is read from the
record rather than passed in, a replay rule about a pair of calls, and the two rules about
what a reply may carry. An agent bounded only by per-call ceilings looks bounded, which is why
that list is printed beside the recovered policy rather than inferred from its absence.

**Seven rules came back that the author never declared.** The workflow requires orderings
nobody wrote down: the settling step runs after the step that reads the consignment, so a
refund cannot precede that read for the same order. Those are not noise. An ordering the
engine enforces is a rule whether or not it reached a document, and reading it is the reason
the workflow half is worth having.

## How a recovered fact is marked

| Origin | Meaning |
|---|---|
| `declared` | Read from a structured field the platform wrote. Re-reading gives the same answer. |
| `stated` | The operator's prose says it outright, and the quote is carried with it. |
| `inferred` | A model read prose and concluded it. May be wrong in either direction. |
| `assumed` | A universal nobody wrote down. Defensible everywhere, configured nowhere. |
| `undetermined` | Nothing answers this, and something downstream requires it. Not a value. |

The last one is the load-bearing one. A declaration is not emitted while any remain, and the
unresolved list names the subject and the question for each. The three in the section above are
the ones that recur.

At the boundary these narrow to the two values a scorecard carries: `declared` stays declared,
and everything else becomes inferred. The narrowing happens once, in
`src/agentred/ingest/package.py`, so that a rule read out of a sentence can never reach a
report looking like a rule an operator configured.

## Fidelity

A run says which of these it was, because the three support different claims and reporting them
as one would overstate the weakest.

| Given | What a report may say |
|---|---|
| Instance configuration and manifest | conformance to a declaration, in full |
| Manifest only | conformance, with the drafted half's provenance shown per rule |
| An endpoint | what broke, and nothing about coverage |

The third is not a lesser version of the first. It is what an attacker has, and running the
same corpus at all three says what internal knowledge is worth in findings rather than
asserting that it is worth something.

## Adding a platform

An adapter produces the intermediate in `src/agentred/ingest/package.py` and nothing else. Its
whole contract is three rules.

**Record what the source says, never what it implies.** An adapter that reads `issue_refund`
and concludes it moves money has formed an opinion, and opinions belong one layer up where they
can be marked as opinions and confirmed. A reader whose adapters editorialise cannot tell an
operator which of its statements came off their platform.

**Leave a hole rather than a default.** Every value that a source did not carry is
`undetermined`, with the question that resolves it and the cost of answering it carelessly
stated in the question itself. Whoever answers twenty of these is under pressure to click
through them, and the answer that ends the conversation fastest is the one that empties the
suite.

**Carry the evidence.** Every fact names the reader that produced it and where it was found: a
connector URL, a file path, a step name, or the sentence a model read it out of. A draft
nobody can check is a draft everybody approves.
