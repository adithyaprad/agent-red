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

The split matters: config is capability, policy is authorisation, and every system that grants
power to an actor separates the two. It also decides what remedies exist. A structured policy
can be tightened, which makes a violation unreachable. A limit written as English inside a
system prompt can only be reworded, which makes a violation less likely. See
`docs/DECISIONS/ADR-0003-instruction-vs-permission.md`.

*Degraded mode.* Where policy is prose, `attacks/stakes.py` extracts candidate bounds and
preconditions from the prompt with a model, and every check derived that way is labelled
`inferred` rather than `declared` on the scorecard. Uncertainty at the foundation is worse
than uncertainty at the edges, so it is reported rather than hidden.

*Failure mode if absent:* stakes cannot be derived, attacks fall back to generic probing, and
the suite loses most of its power.

### Surface 2: the agent is exercisable

agent-red runs the agent, repeatedly, safely, and observes what it did.

| Assumed | Why it is load-bearing |
|---|---|
| Conversable **before publish** | The gate is pre-launch. No draft state, no gate. |
| Conversations **fresh and isolated** | 400 samples must be independent, or every reported rate is wrong. Not degradable. |
| Side effects **sandboxed** | The suite deliberately attempts refunds and orders. Not degradable. |
| A turn returns **tool calls**, not only text | Provable violations instead of judged ones. |
| Test-time behaviour **matches production** | Same model version, tools and data, or the measurement describes a different agent than the one that ships. |

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
attack. It is a stand-in, not the product. Three targets exist: a cart-recovery agent and a
dispute-handling agent, both with declared policy, and a third from an unrelated domain
introduced after the harness was frozen, used to verify that nothing in `attacks/` or
`judge/detectors/` encodes knowledge of what a target sells.

## Pipeline

```
 registry + consent gate
          |
          v
   attack generator  ------ techniques (YAML) x derived stakes + mutations + gen N-1 winners
          |
          v
   conversation runner  ---->  target endpoint  ---->  reply + tool-call log
          |                                                   |
          v                                                   v
   deterministic detectors  <-------------------------  policy manifest
          |
          | (only the interpretive residue)
          v
      LLM judge  ------  calibration harness  ----  held-out set + human labels
          |
          v
      scorecard  ----  exposure model  ----  suggested patch  ----  re-run, measure delta
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

**Multi-turn is the point.** Single-turn attacks mostly fail against a competent system
prompt. The ones that land spend three benign turns establishing false context and then
ask. The runner is therefore a conversation driver with a goal and a turn budget, not a
prompt sender.

**Detectors before the judge, and the detectors are generic.** Most real violations are
observable rather than interpretive, and they reduce to three policy-driven shapes: a tool
called beyond a declared bound, a consequential tool called without its declared precondition,
and an identifier surfaced from outside the session's declared data scope. Those are asserted
in code against the tool-call log by `judge/detectors/`. Sending them to a model would trade a
certain answer for an uncertain one. See `docs/DECISIONS/` for where we deliberately did not
use a model.

**A check has three outcomes, not two.** A rule can be violated, held, or never evaluated,
and the third is kept separate everywhere it travels. A rule is unevaluated when its tool was
never called, when the argument it constrains never arrived, when a limit read from an earlier
result had no result to read, or when a conversation did not record whose data it was entitled
to. Only evaluated findings may be used as the denominator of a rate. Collapsing unevaluated
into held is the easiest way to publish a safety number that is too good: an agent that
declined to act at all would score identically to one that acted correctly every time, and the
more limits a merchant declares, the better an unexercised suite would look.

The limit is worth stating plainly: a violation that is not expressible as a bound, a
precondition or a scope is invisible to the detectors and falls to the judge. That limit is
measured rather than left as a caveat. `attacks/stakes.py` marks every derived stake with
whether a detector or a model settles it, and `judge_dependence()` reports the share in the
second category, so a scorecard always states how much of itself rests on the judge. An agent
that declares nothing scores 1.0 there, which is the honest reading: there was nothing to
check against.

**The judge is measured, not trusted.** Deciding whether an agent "conceded" or "leaked"
does need a model for the residue, and that model is unreliable. `judge/calibration/`
scores the judge against human labels on a held-out set and reports precision, recall,
false-positive cost and the judge's own disagreement rate across repeated samples. Every
scorecard carries those numbers, so a reader can discount its claims correctly.

**Exposure, not test counts.** "You failed 7 of 40 tests" does not reach a merchant.
"Your agent can be talked into an average 34% discount, and at your stated volume that is
roughly this much a month" does. The model, and every assumption it rests on, is in
`docs/EVALUATION.md`. Assumptions are printed alongside the number, never buried.

**Close the loop.** A finding that is not fixed is not worth much, so the harness generates
a suggested policy or prompt patch from the breaking transcripts, applies it to the target,
re-runs the identical suite, and reports the before and after with the residual failures
named. That delta is the product's actual claim.
