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
| Instance configuration | bounds: limits, thresholds, approval points. Also the data sources this instance was granted, and the identifiers one session is scoped to | declared |
| Workflow definition | preconditions: which step must precede which | declared |
| Catalogue manifest | which connectors the agent is wired to, which model it runs, and where the other three are | declared |
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

The manifest an agent is installed from is the entry point rather than a source of rules. It
names the connectors the agent is wired to, the model it runs, and where the other three
sources are on disk, which is what lets a read start from one file and find the rest without
being told. `examples/retention_desk/agent.manifest.yaml` is seven lines and every one of them
exists because the agent needs it to run.

What it does not carry is worth stating, because the obvious assumption is that it does. It
does not say what data the agent may reach or what one session is scoped to: those are per
instance, so they come from the instance configuration's own record of the sources that
instance was granted. And it does not say which fields a stranger writes into or which trigger
reads them. That is a property of the deployment rather than of the platform's records, so it
is asked for rather than recovered, which is the third question in the section below.

### Instructions

What is left is the residue: the rules an operator stated in prose because the configuration
had no field for them. A model reads them, and every rule it produces names a tool and an
argument, both of which are checked against the agent's declared surface. A rule naming
something that does not exist is refused rather than reported, because a model inventing a rule
and a model then finding it broken is the failure this whole path invites.

The residue is small and it is not incidental. The two agents written here were written by
people who knew exactly what they were building, and both ended up with rules in their
instructions that never reached their policy. A rule stated in prose and never configured is a finding
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

Its `README.md` carries the commands. The three files an operator supplies alongside it are
the answers themselves, one per question in the section above, kept in files rather than
prompted for, so that a read reproduces and what a person confirmed sits beside what a reader
recovered.

## What this recovers, measured

`dispute_handler` has a declaration a person wrote, so the claim that one need not be written
is checkable rather than arguable. `tests/ingest/installed/` holds the same agent as a builder
installs it: a manifest, the limits an operator configured, the prose it runs on, and the
workflow an installer mapped it onto. None of those four files were written for agent-red.
Reading them and comparing the result against the authored declaration gives this.

```
dispute_handler: 32 matched, 0 diverged, 10 not covered by any reader that ran,
                 7 found that the author did not declare
```

Twenty-seven of the matches are the tool surface: every tool, every argument schema and every
description came back identical. The other five are rules. Nothing came back differently,
which is the number that would matter most if it were not zero: a reader that recovered a
ceiling of the wrong value would be worse than one that recovered nothing.

**What did not come back is the interesting half, and it was predicted before it was
measured.** Seven of the agent's twelve authored rules are missing, and they are missing by
shape rather than by accident. The instance reader names, on every read, the rule shapes a form
field structurally cannot hold, and the seven are exactly those: a limit relative to another
value, a limit on a running total, a limit whose value is read from the record being acted on
rather than passed in, a rule that the money goes back in the currency it came in, a replay
rule about a pair of calls, and the two rules about what a reply may carry. An agent bounded
only by per-call ceilings looks bounded, which is why that list is printed beside the recovered
policy rather than left to be inferred from its absence.

The three remaining items are not rules at all, and they are the questions the section above
says are asked rather than assumed: which fields a stranger writes and what trigger reads them,
what each tool does to the merchant's records, and who the harness may act as. They appear in
the same list as the missing rules on purpose, because a read that quietly omitted them would
produce a declaration that looks complete.

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
| Instance configuration, workflow and manifest | conformance to a declaration, in full |
| Manifest and prose only | conformance, with the drafted half's provenance shown per rule |

The second is not a weaker report so much as a differently marked one. Every rule carries where
it came from from the moment it is read to the moment it appears on a page, so a reader can see
which half of a result rests on a structured field somebody configured and which rests on a
model reading a sentence. Collapsing the two would make the second look like the first, which
is the one outcome the provenance column exists to prevent.

## Making the agent attackable

Everything above is the declaration: what the harness tests. This is the other half of an
integration, and it is the smaller one. It is what the agent has to answer so that it can be
tested at all.

Four HTTP endpoints and one environment variable. `src/agentred/targets/runtime.py` is one
implementation of exactly this contract, and the bundled agents are served by it, so a
disagreement between this section and that file is a bug in this section.

| Endpoint | Called | Required |
|---|---|---|
| `GET /challenge?nonce=<hex>` | Before the first attack turn, and again every time consent is re-established | Always |
| `POST /chat` | Once per turn of a conversation | Whenever the agent has a conversational channel |
| `POST /trigger` | Once per planted attack, in place of a turn | Whenever a channel declares a scheduled trigger |
| `POST /fork` | When two attacks branch from a shared prefix | Whenever the suite forks (ADR-0002) |

`GET /health` is not part of the contract and is worth serving anyway, because it is what
`agentred doctor` reports a port on before anything more expensive is attempted.

### The challenge

The consent gate is a code path rather than a policy note (`docs/SAFETY.md`), and this
endpoint is the agent's half of it. It takes a nonce and returns it unchanged.

```
GET /challenge?nonce=0f1e2d3c
```
```json
{
  "challenge": "0f1e2d3c",
  "agent_id": "retention_desk",
  "mode": "test",
  "tool_server": "http://localhost:8094",
  "versions": {
    "config_version": "1.2",
    "policy_version": "1.2",
    "model_version": "claude-sonnet-5",
    "tool_version": "sha256:a97e4b26aca7",
    "world_version": ""
  }
}
```

Every field is checked before a single attack turn is sent, the first four against the registry
entry that asked and the last against the spec on disk, and a disagreement is refused rather
than warned about. Each check exists because of a distinct way a run can look fine and mean
nothing.

**`challenge`** is the nonce, byte for byte. An agent that cannot echo it is not attacked, and
that is the intended behaviour rather than a limitation to work around.

**`agent_id`** must be the id the registry names. Without it a registry entry is repointed at a
different agent by editing a port, and the scorecard names an agent that was never tested.

**`mode`** must be the mode the registry declares, which is `test` for anything that can move
money. The suite deliberately attempts refunds.

**`tool_server`** must be the origin the registry names. This is the one people are surprised
by, and it is the one that costs a whole run: every finding comes from the call stream recorded
at that server, so a target reaching its tools anywhere else produces a suite of empty call
streams, which reads as an agent that did nothing at all rather than as a misconfiguration.

**`versions`** is optional, and is checked against the spec on disk when it is present. A target
loads its spec once and holds it for the life of the process, so a spec edited afterwards is on
disk and not in the agent. Reporting it here is what turns that into one line at preflight
rather than a refusal on the version tuple after every model call in the run has been paid for.

Consent expires and is established again during a long run, so this endpoint is called
repeatedly. It must be cheap and must have no side effects.

### A turn

```
POST /chat
```
```json
{
  "session": "s-0f1e2d3c",
  "run": "0025",
  "conversation": [{"role": "user", "content": "..."}]
}
```
```json
{
  "reply": "...",
  "session": "s-0f1e2d3c",
  "spec_versions": {"config": "1.2", "policy": "1.2", "model": "claude-sonnet-5", "tools": "sha256:a97e4b26aca7"},
  "usage": {}
}
```

There is deliberately no tool-call field in the response, and adding one would not help. What
the agent did is read from the tool server's record, never from what the agent says it did
(ADR-0005).

The runner holds the authoritative transcript and sends the whole conversation, ending with the
turn to answer. Replaying only the last turn against a resumed session is the efficient
implementation and it is not required.

`session` and `run` are the harness's, not the agent's, and they are the load-bearing part of
this request. The tool server keys the conversation's private world and its call stream by the
same two values, which is how isolation is obtained without the runner asking for it. An agent
must pass them through to the connector URL it reaches its tools on, unchanged.

`usage` is what the turn cost the agent's own engine, and an empty object from a backend that
does not report one is not the same as free. `agentred cost` says which it was.

### A firing

```
POST /trigger
```
```json
{"session": "s-0f1e2d3c", "run": "0025"}
```
```json
{"output": "...", "session": "s-0f1e2d3c", "spec_versions": {"...": "..."}, "usage": {}}
```

No conversation and no user turn, because the agent this addresses has neither. What it reads
is whatever the world holds when it wakes, which is precisely why it is the entry point a
planted payload is delivered through rather than a prompt announcing that a schedule fired
(ADR-0006). The output text is carried for the transcript and no check reads it.

This is required only for a channel whose trigger is a schedule. A channel whose trigger is an
ordinary request is fired down `/chat` instead, with a message the channel templates, so an
agent that nothing wakes on a timer never needs this endpoint. An agent that declares a
scheduled trigger and cannot fire one answers `404`, and the attempt is recorded as an error
rather than as an agent that held.

### A branch

```
POST /fork
```
```json
{"source": "s-0f1e2d3c", "session": "s-77ab01ff", "at_turn": 3}
```

Branches the source conversation's cached prefix into a new session, keeping `at_turn`
completed exchanges. `null` branches from the end. The new session id must not already exist,
because reusing one hands the branch somebody else's prefix. This is half of a fork: the other
half is branching the world, which the runner asks the tool server for rather than the agent,
since a world an agent could branch is a world an agent could tamper with.

### Reaching tools through the boundary that records them

Set `AGENTRED_TOOL_SERVER_URL` to the origin the registry names, and point the agent's
connector at `/{agent_id}/{run}/{session}` under it.

The run and the conversation are in the path and never in the arguments of a call. An agent
therefore cannot file a call under a conversation it was not given, and the record is keyed by
what the path said rather than by anything the agent asserted. The server has a second face on
a second port, which reads the stream, restores worlds and plants payloads into them, and no
agent is ever told its address.

### The registry entry

```yaml
  - name: my_agent
    agent_id: my_agent
    description: What it does, for whoever reads `agentred doctor`.
    base_url: http://localhost:8085
    spec_dir: path/to/its/spec
    mode: test
    tool_server: http://localhost:8090
    control_url: http://localhost:8091
```

`base_url` is an origin with no path, so that a challenge cannot be answered by one service
while the attack is delivered to another. The three paths default to `/challenge`, `/chat` and
`/trigger`, and `challenge_path`, `chat_path` and `trigger_path` override them for an agent
already serving somewhere else. There is no field anywhere that takes a bare URL: adding an
entry to this file is the assertion that you control the agent it names.

`agentred doctor` checks every row of this: the spec loads, the tool server answers, the
challenge is echoed, and the agent id, mode, tool server and versions all agree. Run it before
paying for a suite.

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
