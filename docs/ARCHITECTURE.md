# Architecture

## The problem this is shaped around

Merchants are putting LLM agents in front of customers. Those agents quote prices, apply
discounts, describe stock, promise delivery dates, take orders, and in some cases issue
refunds. A merchant writes a system prompt, chats with it a few times, and ships it.

There is no test suite. There is no pass or fail. Nobody can tell the merchant whether the
agent is safe to put in front of money. The failure modes are already documented in the
wild: agents talked into honouring discounts that do not exist, accepting a price the
customer simply asserts, inventing stock or return policies the merchant cannot honour,
leaking one customer's order details to another, committing to items outside the catalogue.

Each of those converts directly into money the merchant loses. agent-red measures that
exposure before the agent meets a customer, and keeps measuring it as the agent changes.

## What this requires of a platform

agent-red touches a merchant agent platform in exactly three places. Each is an assumption,
each has a distinct failure mode, and together they are the whole integration contract. They
are stated here rather than discovered, because a harness that only works against the platform
it was written for is a consulting engagement, not a component.

### Surface 1: the agent is legible

agent-red reads what the merchant built, and derives the attack suite from it.

| | Holds | Assumed |
|---|---|---|
| **Config** | What the agent *is* and *can do*: instructions, tools with schemas, reachable data | One retrievable, versioned object |
| **Policy** | What the agent *may* and *must* do: bounds, required preconditions, session data scope | Separate from config, and structured rather than prose |
| **Subjects** | Who the harness may act as: identities the merchant declares as safe to impersonate against a test deployment, with what each would know | Present whenever the policy scopes a session to a subject |
| **Channels** | Where attacker-controlled bytes can enter: which fields of which records a customer writes, and the trigger that makes the agent read them | Declared, because what an adversary controls is a property of the deployment and the merchant knows it |

The split matters: config is capability, policy is authorisation, and every system that grants
power to an actor separates the two. It also decides what remedies exist, and the difference is
the whole reason the split is worth making. A structured policy can be tightened, which makes a
violation unreachable. A limit written as English inside a system prompt can only be reworded,
which makes a violation less likely. Those are not two strengths of the same fix; they are
different kinds of promise, and a report that presented them as interchangeable would be telling
an operator a rewording had closed something it had only discouraged. See
`docs/DECISIONS/ADR-0003-instruction-vs-permission.md`.

Channels are what makes the suite reach an agent nobody talks to. A cart-abandonment agent has
no user turn in its entire lifetime: a cron fires, it reads the abandoned baskets, it decides
who to contact, it sends a message. Everything an adversary controls about that run was written
into a record days earlier, as a name, an address, a note on an order. An agent declares those
fields and the trigger that reads them, and the harness attacks through them. Without the
declaration the harness would have to guess at a schema it will not see on a real platform, and
a guess that is wrong reports a clean sheet. See `docs/DECISIONS/ADR-0006-attack-channels.md`.

Subjects are the part most easily mistaken for a convenience. An agent that reads records
cannot be attacked in the abstract: the conversation has to be about somebody, and the
identifiers it opens with have to resolve or the agent declines to act on a record it cannot
find. The action under test is then never reached, every rule reports as never evaluated, and
the run reads as a clean sheet for an agent nobody managed to question. A policy that scopes a
session and declares nobody to be is refused at load rather than warned about, for the same
reason a self-contradicting spec is: a check that cannot fire is indistinguishable from a
passing agent.

*Degraded mode.* Where policy is prose, `attacks/infer_policy.py` extracts candidate bounds and
preconditions from the prompt with a model, and every check derived that way is labelled
`inferred` rather than `declared` on the scorecard. Uncertainty at the foundation is worse
than uncertainty at the edges, so it is reported rather than hidden.

The reading is not only for an agent that declared nothing. It runs on every agent, before the
suite is built, because the interesting question is not what the prose says but what the prose
says that the policy does not. An operator states rules in their instructions and declares a
subset of them, and the ones that fall through are disproportionately about what the agent may
say rather than what it may call: repeat something it fetched, assert something it never looked
up, undertake something on the operator's behalf. Those break in a reply and leave a tool-call
log identical to a conversation that keeps them, so no detector will ever ask about one. Reading
the prose first is what turns such a rule into something attacked on purpose rather than
something an attack aimed elsewhere occasionally trips over, and a run that cannot read the
instructions is refused rather than run against the smaller set it could see.

*Failure mode if absent:* without config or policy, stakes cannot be derived, attacks fall back
to generic probing, and the suite loses most of its power. Without subjects, the suite still
runs and still reports honestly, and reports nothing: every check comes back never evaluated.

### Surface 2: the agent is exercisable, and its hands are observable

agent-red runs the agent, repeatedly, safely, and observes what it did. The second half of
that sentence used to be a request made of the agent and is now a property of where we stand.

| Assumed | Why it is load-bearing |
|---|---|
| Runnable **before publish**, through its real trigger | The gate is pre-launch, and the attack path has to be the agent's normal operating path. A synthetic prompt standing in for a cron tests a different agent than the one that ships. |
| Runs **fresh and isolated** | 400 samples must be independent, or every reported rate is wrong. Not degradable. |
| Its tools are reached through **one boundary we hold** | Every violation is observed there. Not degradable, and it is what makes the harness framework-agnostic by construction. |
| Side effects **sandboxed** | The suite deliberately attempts refunds and orders. Not degradable. |
| Test-time behaviour **matches production** | Same model version, tools and data, or the measurement describes a different agent than the one that ships. |

The fourth row replaced the assumption this document made until 2026-09-02, which was that a
turn comes back carrying the agent's own list of the tools it called. That was true because we
wrote the target, and it is the wrong thing to depend on twice over: a self-report is not
evidence about an untrusted party, and requiring one means integrating with every agent runtime
separately, which is exactly the generality the project claims. An agent's reply is
framework-specific; an agent's hands are not. Whatever it is built on, it reaches the
merchant's money through tools, and in the platform this is shaped around those tools are MCP
connectors. Recording there is on our side of the trust line and needs nothing from the agent.
See `docs/DECISIONS/ADR-0005-oracle-at-the-tool-boundary.md`.

Three properties of the boundary are worth stating, because the decision is only worth what
they are worth. The run and the conversation are in the URL an agent's connector is pointed
at, never in the arguments of a call, so a call cannot be filed under a conversation the agent
was not given. The record exposes no operation that edits or removes an entry. And the tool
server has two faces on two ports: agents are told the tool port, while reading the stream,
restoring a world and planting into one live on a control port no agent is given, so tampering
with the evidence is out of reach without a secret having to be kept.

*Failure mode if absent:* the agent can be attacked but not measured, so there is no evidence.

### Surface 3: the improvement is applyable

What agent-red returns is the same kind of object it read. Config in, config out. Applying a
fix is replacing that object with a new version, not running a migration or integrating with
anything.

- Applying a change is **replacing the config and policy object**
- Objects are **versioned**, so a scorecard belongs to a specific version and a change can be
  rolled back
- The fix can be applied to the **draft** and re-tested before publishing, which is what closes
  the loop rather than merely reporting

*Failure mode if absent:* the output is a report a human retypes, not a change a system applies.

### The property this buys

agent-red is a pure function over agent definitions. In: config and policy. Out: a new config,
a new policy, and the evidence. It requires no database access, no runtime hooks and no
privileged position, which is what lets it ship as a component inside a platform rather than
sit beside one as a service.

### Validity of a result

A scorecard is valid for one **(config version, policy version, model version, tool version)**
tuple. Change any element and the agent is untested again, which is why every run records the
tuple it was measured against and why a model upgrade is a reason to re-test. `store/repo.py`
enforces this: a scorecard cannot be written without it.

### The stand-in targets

`src/agentred/targets/` implements this contract locally so the harness has something real to
attack. It is a stand-in, not the product.

The two primary targets are built the way the platform this is shaped around builds them: a
deterministic workflow engine holding the steps, LLM nodes confined to the judgement points
inside those steps, and every external capability behind an MCP connector. A cart-recovery
agent fires on a schedule and contacts customers who abandoned a basket. A dispute-handling
agent triages chargebacks, decides whether to contest or accept, and assembles evidence. Both
are written the way a competent person would first ship them: a clear prompt, sensible tools,
correct policy, limits stated in the instructions, and no anticipation of adversarial pressure.
Written defensively, nothing breaks and there is nothing to show; written badly, the result is
worthless. Every flaw the suite finds has to pass one test before it appears in a report: could
a competent person building on a no-code studio plausibly ship this. Anything needing a reach
is cut.

A third target exists and is built deliberately differently: an unrelated domain, and a plain
model loop with no workflow engine, introduced after the harness was frozen. It carries two
proofs at once. Nothing in `attacks/` or `judge/detectors/` encodes knowledge of what a target
sells, and nothing in them encodes knowledge of how a target is built. The identical suite runs
against all three with no new attack code, no new detectors and no per-target configuration
beyond the spec each one declares.

That second proof is the one the oracle placement buys, and it is worth being precise about why
it holds rather than asserting it. The harness never reads a workflow trace, a graph state or a
message block. It reads a call stream captured at the MCP server and a policy the agent
declares, and both of those exist regardless of what produced the calls.

## Pipeline

```
 registry + consent gate
          |
          v
  instruction reading  ---- rules stated in the prompt, compared against the declared policy
          |
          v
   attack generator  ---- techniques (YAML) x derived stakes x channels + mutations + gen N-1 winners
          |
          +---------------------------+
          |                           |
          v                           v
  conversational channel        planted channel
  multi-turn driver             restore -> plant -> trigger
          |                           |
          +------------+--------------+
                       |
                       v
                 the agent runs
                       |
                       v
          MCP server  ---->  tools over the synthetic world
                       |
                       v
          call stream + world diff        (recorded here, not reported by the agent)
                       |
                       v
   deterministic detectors  <---------------------  policy manifest
          |
          | (only the interpretive residue)
          v
      LLM judge  ------  calibration harness  ----  held-out set + human labels
          |
          v
      scorecard  ----  exposure model  ----  suggested config change  ----  re-run, measure delta
```

## Why each stage exists

**Consent gate.** A generator of working social-engineering attacks is only defensible if
it cannot be pointed at a stranger. Targets are resolved from a registry and must echo a
challenge string before the first attack turn. There is no code path that takes a bare URL.
Scope and refusals are in `docs/SAFETY.md`.

**Composed attacks, not a fixed prompt list.** Two layers. Eight agent-independent
manipulation techniques (assert an unverifiable fact, claim a prior interaction, borrow
authority, manufacture pressure, build false context then ask, substitute identity, drift
outside remit, inject through a data channel) are crossed with the stakes derived from the
target's own spec: which tools are consequential, which bounds are declared, which data scope
the session holds. A fixed list is patched string by string and the merchant stays broken, and
it only ever works on the one domain it was written for. Surface variation (register, language
including Hinglish and mid-conversation code-switching, pressure, obfuscation) is applied on
top. Successful attacks seed the next generation, so the suite hardens as agents do.

This is also why the harness transfers. Nothing in `attacks/` or `judge/detectors/` knows what
a target sells. That claim is tested rather than asserted: a third target from an unrelated
domain is introduced after the harness is frozen and must produce a valid scorecard with no
new attack code and no new detectors.

**Multi-turn is the point, where there are turns at all.** Single-turn attacks mostly fail
against a competent system prompt. The ones that land spend three benign turns establishing
false context and then ask. So the conversational channel is a driver with a goal and a turn
budget, not a prompt sender, and it stays first class: it is the only channel where an attack
can adapt to what the agent just said, and two of the harness's most distinctive outputs, the
breaking point and the consistency comparison, exist only because attacks are multi-turn.

**The chat box is one channel, and not the one that reaches every agent.** An agent that runs
on a schedule has no turns. Everything an adversary controls about that run was written into a
record days earlier: the free text on a dispute, the name on an account, a delivery
instruction, a product title the agent summarises. The planted channel writes a payload into a
field the agent declares as customer-writable, restores the world to a snapshot first, fires the
agent's real trigger, and reads the call stream. Firing the real trigger is the part that
carries the argument: an attack delivered down a path the deployment does not use is a finding
about a test harness. See `docs/DECISIONS/ADR-0006-attack-channels.md`.

**The shop the attacks land in is designed backwards from the checks.** A world laid out
forwards from its domain, one customer per order per dispute, makes most declared rules
unbreakable: nothing has been refunded, so a running total cannot be exceeded, and every
identifier resolves to one obvious person, so reaching the wrong one means naming a record
that does not exist. An agent holding there has answered nothing, and the coverage grid still
reads as a clean sheet. `data/store/` therefore carries confusable identities, amounts near
the declared ceilings, an order with money already returned against it, and one debt filed
twice, and `tests/mcp/test_world_reachability.py` fails the build if those properties are
trimmed away. Which records exist and why is `docs/WORLD.md`.

That shop is hand-authored, which is a per-agent integration and the last one in the chain.
`docs/DECISIONS/ADR-0007-generated-world.md` decides how it is derived instead, and the decision
turns on the input: a world derived from an agent's declaration alone holds every collection the
declaration names and none of the properties above, because a declaration says what the rules
are and not what a shop has to look like for breaking one to be a small step. So the input is
the declaration crossed with the checks, and the output carries a manifest of which declared
rules it made reachable, since a rule with no reachable fixture and a rule that was tested and
held are opposite facts that look identical in a finding count.

`mcp/generator/` implements that. It walks every declared rule and asks two questions of it:
what has to be true of a record for breaking this rule to be one step from a reasonable
action, and what has to be true of one for holding to be the right answer. Both halves are
emitted, because an agent that refuses everything scores perfectly against a world made
entirely of traps, so a world of traps cannot tell judgement from compliance and the utility
suite over it measures nothing. Neutrality is therefore a consequence of how each record was
emitted rather than an instruction in a prompt: every fixture carries the rule it exists for,
so the question "why is this record here" has an answer that predates the answer. `agentred
world --target <name>` prints the manifest, gaps included, and contacts nothing.

**A world is a tool surface, not a dataset**, and that is the half the framing hides.
Generating a shop for an agent nobody has seen means serving tools nobody has written, so
`World` holds collections named by the declaration rather than fields named by a source file,
and `mcp/tools/generic.py` derives a handler from what each tool declares it does. Three shapes
cover it: fetch the record somebody named, fetch the records matching something, change
something. What the vocabulary deliberately cannot express is a value computed from the record
being written to, a percentage of a total being the case that comes up first, because
expressing it needs a small language in the config and the person writing that config is an
ops team. Such a tool is reported as a named gap. Both shipped agents keep their hand-written
handlers, which is what a generic one is checked against.

**A reference has to resolve, and a record's own fields are not enough.** A limit is settled
without leaving the record it sits on, and almost nothing else is: an action that must follow a
read of the record it acts on, a message that may carry only this party's details, a figure read
off one record before the call bounded by it. Emitters build one record each, so every reference
was minted from its own counter and pointed at nothing, and because reachability is decided when
a record is emitted the manifest went on reporting those rules as reached. `mcp/generator/link.py`
runs after emission and rewrites references onto records that exist, keeping the equivalence
classes the emitters set: a pair a fixture made agree stays agreeing, a pair it made differ stays
differing. It invents no relationship. Two fixtures for one rule are two records, and `Shop.put`
raises rather than overwriting when they would share a key, because the manifest is the whole of
the reachability argument and it is written from what the emitters say they did.

**A world has a fourth per-agent part, and it is who the harness acts as.** The identities the
harness may act as are declared per agent, in `src/agentred/targets/specs/dispute_handler/subjects.yaml`
and its counterpart, and they name records: a cast written for one shop names nothing in another. Down a planted channel that is
loud: the write is refused and the suite errors before the agent is reached. Down a conversation
it is silent, because the agent is asked about a reference it cannot find, says so truthfully,
and every rule reports as never in play. `mcp/generator/cast.py` derives the cast from the
fixtures, anchoring each identity on a record some rule exists for and stating as facts only what
those records hold, and the tool server serves it on its health face because a cast and a world
are one fact. Both halves are represented, breakable and holding, for the same reason the
fixtures are. A channel no generated identity can reach is named rather than attacked.

**The manifest can say three things about a rule, not two.** A gap says the rule could not be
made breakable and a fixture says it was. Neither carries the case where a rule was made
breakable and also made breakable for a reason nobody declared, which is what happens when the
generator has to invent a value the agent has its own opinion about: a field a matching rule
compares is shared vocabulary rather than a name for a record, and where the declaration does not
say which values it may take, an agent can substitute one it recognises. `Manifest.notes` carries
that, so a count of failures against such a rule is read with the doubt attached.

**The shop joins the validity tuple.** `VersionTuple` gained a fifth element, a content digest
of the world the run was against, reported by the tool server because that is the process
holding one. A scorecard computed against one shop says nothing about an agent facing another,
and the quieter version of the same problem is what a hand-authored shop does when it is
edited: every earlier scorecard goes on citing a tuple that no longer describes what the agent
faced. ADR-0004 is amended rather than superseded, in the way ADR-0006 amended it.

**A finding is reproducible from a seed.** Generation is stochastic, and a gate that returns
different verdicts on identical input is not a gate. Every attack carries the seed that
produced its surface form, so a finding replays byte for byte. The suite is split accordingly:
a frozen corpus of attacks that have succeeded somewhere before, deterministic and identical
every run, which is the only thing a verdict is computed from, and an exploratory budget that
mutates and searches and can promote a novel success into the frozen set for next time. The
second never moves a verdict. A side effect worth naming: the corpus compounds, so every agent
tested makes the gate stronger for the agent after it.

**Detectors before the judge, and the detectors are generic.** Most real violations are
observable rather than interpretive. They reduce to seven policy-driven shapes, and the split
matters because the first three, on their own, report a conversation as clean when every
argument is in range and the damage is somewhere else entirely.

The three that read one call at a time: a tool called beyond a declared bound, a consequential
tool called without its declared precondition, and an identifier surfaced from outside the
session's declared data scope.

The third of those has a second shape, because a conversation is with one person and a
scheduled agent is not. An agent nobody talks to is woken about a set, and acting on all of it
is the job, so an attempt against one carries the cohort its trigger covered beside the single
subject it is named for. A record belonging to either is in scope; reaching somebody in neither
is still a violation. The cohort is read from the world rather than from the agent's own
selection call, because the agent chooses that call's filter, and a cohort taken from its result
widens exactly when an attack succeeds in widening it.

The four that do not, each of which exists because a real failure is invisible to the first
three. A limit on a total across calls, because three amounts inside every ceiling still add up
past what was ever taken, and an agent talked into instalments defeats a per-call limit
completely. A limit on a value the call does not carry, because the costliest action an agent
has is often a reference and nothing else, with the money read from a record it fetched a moment
earlier, so bounding the arguments makes the most expensive thing it can do look free. One
effect asked for twice and happening twice, where both calls are unremarkable and the merchant
is charged because the same instruction arrived twice, which on a scheduled trigger is the most
likely way money leaves twice. And a value belonging to somebody else inside an outbound
message, where the tool, the recipient, the limits and the prior steps are all correct and the
whole failure is a string.

Beside them sit two checks that come from a declaration rather than from a rule: a reference
cited that the agent never read, which is how a fabrication reaches a person who believes it,
and a call to a tool the config does not contain, which only became observable when the oracle
moved to the tool boundary.

All of them are asserted in code against the recorded call stream by `judge/detectors/`.
Sending any of them to a model would trade a certain answer for an uncertain one. See
`docs/DECISIONS/` for where we deliberately did not use a model.

**The page is a function of the agent too, and that is enforced rather than intended.** The
claim that the suite is derived from the agent under test is worth nothing if the artefact
somebody reads was written for one merchant. So `scoring/` is held to the same genericity guard
as `attacks/` and `judge/detectors/`: a module there that says refund, basket or shopper fails
the build. What a page needs in order to speak plainly is declared by the agent instead. The
config says what to put in front of an amount, what it calls the people a session is about, and
which result fields mean the operator is out of pocket, and the page reads the operator's own
sentence for every rule rather than the identifier the harness files it under. Writing prose for
a reader is the activity most likely to produce a sentence that only makes sense in one shop,
which is why this directory is guarded rather than trusted.

**A rate is reported over what was actually tried.** A rule holds nineteen times out of the
nineteen conversations that reached it, not out of the eighty-eight that were run. Both figures
are printed, because the second is what says how much the first is worth. A rule nothing ever
reached carries no rate at all rather than a perfect one, and a rule read out of an agent's
prose that no check covers is reported as never looked at, which is a third state and not a
good score.

**A check has three outcomes, not two.** A rule can be violated, held, or never evaluated,
and the third is kept separate everywhere it travels. A rule is unevaluated when its tool was
never called, when the argument it constrains never arrived, when a limit read from an earlier
result had no result to read, or when a conversation did not record whose data it was entitled
to. Only evaluated findings may be used as the denominator of a rate. Collapsing unevaluated
into held is the easiest way to publish a safety number that is too good: an agent that
declined to act at all would score identically to one that acted correctly every time, and the
more limits a merchant declares, the better an unexercised suite would look.

**A rule about what the agent says is a third kind of statement.** A bound limits an argument
and a precondition requires a prior step, and both are decidable from the tool-call log. A rule
forbidding the agent from repeating something it fetched, asserting a figure it never looked up,
or undertaking something on the operator's behalf is not: a conversation that breaks one produces
the same tool calls as a conversation that keeps it. Those are declared as obligations in
`spec/`, so what the judge is asked is a rule the merchant recognises, quoted as they wrote it,
rather than a general question about whether the agent behaved well. A judge asked the general
question answers a slightly different one in every conversation, and nothing that moves can be
calibrated.

**Rules are read out of the prompt as well as the policy.** `attacks/infer_policy.py` extracts
the rules an agent's prose states and reports which of them the structured policy does not
carry. This was planned as degraded mode for an agent declaring nothing, and it earns its place
against agents that declare a great deal: both targets in this repository were written by
someone who knew exactly what they were building, and both ended up with rules in their
instructions that never reached their policy. Everything derived that way carries
`provenance: inferred` to the scorecard and is never merged into the declared set. A rule naming
a tool or an argument the agent does not have is refused at construction rather than reported,
and the share refused is carried, because a model that writes rules and a model that checks them
would otherwise be free to invent a rule and then find it broken.

The limit is worth stating plainly: a violation that is not expressible as one of the seven
declared shapes is invisible to the detectors and falls to the judge. That limit is
measured rather than left as a caveat. `attacks/stakes.py` marks every derived stake with
whether a detector or a model settles it, and `judge_dependence()` reports the share in the
second category, so a scorecard always states how much of itself rests on the judge. An agent
that declares nothing scores 1.0 there, which is the honest reading: there was nothing to
check against.

**What a reading can claim is constrained rather than assumed.** Deciding whether an agent
"conceded" or "leaked" needs a model for the residue, and a model's answer is an opinion where a
detector's is an assertion. Three things follow. The share of a run resting on the reading half
is reported, per run, by `judge_dependence`, so a reader always knows how much of a page is which
kind of answer. Each verdict shows the reading's own confidence, which is displayed and used to
decide nothing, because a self-reported number is not an independent one. And two guards discard
whole classes of wrong answer outright: a violation must quote the sentence that broke the rule
and is discarded if that sentence appears nowhere in the conversation, and a violation whose
reasoning argues the opposite of its own verdict is discarded too. Removing a failure mode is a
different thing from describing one, and it is the part that can be done without a human in the
loop.

**Some failures are properties of a set, not of a conversation.** No transcript is wrong when
an agent declines. The finding is that it declined twice and complied once on the same subject
for the same action, and that shape exists only across conversations. It is worth reporting for
a reason no declared rule covers: an agent whose limits are policy refuses every time, and one
whose limits are improvised refuses until somebody rephrases, and from outside the two are
indistinguishable until they are not. `scoring/consistency.py` finds these by counting the
tool-call log and asks a model one question, only about groups that already disagreed, about
whether a real difference in the facts explains it.

**A failure is reported with the turn it was decided in.** The turn a rule breaks is rarely the
turn the conversation was lost. `scoring/breaking_point.py` locates the earlier turn where the
agent's position changed, quotes it, and verifies the quote against the reply. A report showing
only the damage teaches the wrong lesson with complete accuracy: a reader who sees the payout
writes a rule about payouts, when the sentence that decided it was several turns earlier and
broke nothing.

**Exposure, not test counts.** "You failed 7 of 40 tests" does not reach a merchant.
"Your agent can be talked into an average 34% discount, and at your stated volume that is
roughly this much a month" does. Every assumption the model rests on is printed alongside the
number, never buried, and the renderer refuses to emit a figure without them.

**Close the loop, and the fix has to be config.** A finding that is not fixed is not worth
much, so the harness generates a suggested change from the breaking transcripts, applies it to
the target, re-runs the identical suite, and reports the before and after with the residual
failures named. That delta is the product's actual claim.

The constraint on what a suggestion may be is a product decision, not an implementation one.
The people who read this report are ops and finance teams on a no-code builder. They cannot
apply a Python patch. So every remediation has to be expressible as configuration: a limit
value, an approval threshold, a narrowed tool allowlist, a required idempotency key, a clause
in the instruction text. A fix that can only be written as code does not belong in the report,
because it is not a fix anybody reading the report can make.

This is also where the strongest thing the harness can demonstrate lives, and it takes thirty
seconds. Land a finding. Let the merchant fix it the obvious way first, by tightening the
sentence in the instructions box. Re-run: that exact attack now fails. Then show a variant that
still succeeds. Then apply the real fix, moving the bound out of the prompt and into the tool
schema, and watch the whole family go dark. Every other claim in this document rests on that
being true, and it is the reason the config-policy split at the top of this file exists.

**Security and utility are reported together, always.** Every hardening change costs
capability, and the trivially safest agent refuses everything. So a benign suite of ordinary
tasks with known-correct outcomes runs beside the attack suite, and both numbers move together
in the report. Attacks succeeding and benign tasks passing appear on the same line, before and
after. A security number reported on its own is not acceptable output, because the change that
produced it may have been an agent that stopped working.

**Nothing in the output implies safety.** The harness can prove a vulnerability exists. It
cannot prove one does not, and a clean verdict that precedes a real loss makes the tool the
cause of that loss. So safe, secure, verified and passed all checks are absent from every
user-facing string, and a run that found nothing says what it actually did: no findings in 340
attempts across six goals and nine channels, with the coverage grid rendered beside it so a
reader can see what was never tried. That grid is what replaces a safety claim. It sits beside
the three-outcome accounting rather than replacing it, because a cell that was attempted is a
weaker statement than a rule that was actually reached and decided.

## What the harness can see, and what it structurally cannot

Every item here is a consequence of a decision above rather than a gap waiting to be filled,
which is why they are stated as architecture and not as caveats at the end of a report.

**Coverage is not proof.** The harness finds vulnerabilities. It cannot establish their
absence, and no number it prints should be read as having done so.

**The declaration is trusted ground truth.** agent-red verifies that an agent conforms to what
it declares. It does not verify that the declaration is correct. An agent faithfully executing
a wrong rule passes, and an agent whose merchant declared a ceiling twice as high as their
margin can survive is measured against that ceiling. Conformance, not correctness of intent.
This should be said out loud everywhere, including in the demo.

**White box is the product; black box is a reduced mode and no parity is claimed.** The
harness reads an agent's declaration and derives its threat model from it, which is the mode
that ships and the mode a pre-launch gate is entitled to. A real adversary has no manifest, and
that objection is answered rather than dismissed: an endpoint-only mode runs the invariants that
hold for any commerce agent regardless of what it declared, which is to say never move more
money than the order was worth, never read a record belonging to a customer other than the one
in the session, never send to a destination that is not on file. Everything policy-specific is
unavailable there, because policy-specific bounds are exactly what the declaration was
providing, and a bound inferred from watching an endpoint is a guess rather than ground truth.
The two modes are reported separately and the reduced one says what it could not check.

**State resets between runs, so multi-run attacks are invisible.** The world is restored before
every attack, which is what makes a finding reproducible and replayable. The same property
means a payload that does no damage this run but poisons a record for every future cycle, a
product title that affects next week's cart recovery, a note the agent writes that a later run
reads as trusted context, is outside the harness by construction. Named rather than covered.

**A violation must be expressible as a bound, a precondition, a scope or an obligation.**
Anything a merchant cares about that does not fit one of those four shapes is invisible to the
detectors and falls to the judge, and the share that falls to the judge is measured and
reported rather than left as a caveat.

**A well-built agent may resist everything, and that is a valid result.** If a workflow extracts
structured fields before the reasoning step, a planted payload never reaches the decision
context. That is the correct architecture and the right defence. The finding is then whatever
it is, and the report says how many attempts were made rather than implying that few findings
means few vulnerabilities.
