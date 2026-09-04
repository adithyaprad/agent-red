# agent-red

A pre-deploy safety gate for agents that touch money.

Point it at an agent before it goes live. It reads the agent's own declaration, derives a
threat model from it, attacks it through the fields a real adversary controls, and reports
what broke with the exact transcript and a configuration change that closes it.

**agent-red verifies that an agent conforms to its declaration. It does not verify that the
declaration is correct.** That distinction is load-bearing and is repeated wherever a number
is reported.

> **Status: in development.** Results tables below are marked `PENDING` until the run that
> produces them is committed alongside the transcripts it came from. No number appears in
> this README that cannot be regenerated from `agentred.db`.

## The problem

Merchants and ops teams build agents on no-code studios. They describe the task in plain
English, pick the systems the agent can touch, and type the guardrails as sentences. Then they
deploy.

Those guardrails are instructions, and instructions are advisory. The tool schema still accepts
any value, so "never refund more than the order value" is a suggestion the model usually
follows and occasionally does not. Nothing in the runtime enforces it.

The people closest to this feel it as a sequencing problem rather than a security one. An agent
can only be exercised once it is deployed, and once it is deployed the exercise is a handful of
happy-path conversations, not the scenarios an adversary would actually try. The failure modes
are already documented in the wild:

- talked into honouring a discount that does not exist
- accepting a price the customer simply asserts
- refunding more than the order was worth, or refunding the same order twice
- inventing stock, delivery dates or return policies the merchant cannot honour
- disclosing one customer's order details to another
- committing to items outside the catalogue

agent-red measures the gap between what a guardrail promises and what the runtime actually
enforces, before the agent meets a customer.

## How it works

```
 registry + consent gate
          |
          v
   attack generator  ---- techniques x derived stakes x channels + mutations + gen N-1 winners
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
          call stream + world diff      (recorded here, not reported by the agent)
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

Six design decisions carry the project. Each is argued in full in `docs/`.

1. **The oracle sits at the tool boundary, never inside the agent**
   (`docs/DECISIONS/ADR-0005-*`). Every violation is observed in the call stream passing
   through the MCP server, plus the resulting world diff. Not by reading a workflow trace, not
   by reading graph state, not by asking the agent what it did. An agent's reply is
   framework-specific; an agent's hands are not. This is what makes the same suite run
   unchanged against agents built three different ways, which is demonstrated rather than
   claimed.

2. **Attacks arrive through the fields an adversary controls, not only a chat box**
   (`docs/DECISIONS/ADR-0006-*`). Payloads are written into dispute reason text, customer
   names, delivery instructions and product titles, and then the agent's real trigger is fired.
   A cron-driven agent has no conversational turns and is still fully attackable. Multi-turn
   conversation remains a first-class channel, because it is the only one where an attack can
   adapt to what the agent just said.

3. **Attacks are composed, not listed.** Agent-independent
   manipulation techniques are crossed with what a given agent actually has to lose, derived
   from its own declared tools, bounds, preconditions and data scope, and with the channels it
   declares. A fixed list of test prompts gets patched string by string and the agent stays
   broken; worse, it only works on the domain it was written for.

4. **Deterministic detection before model judgement** (`docs/DECISIONS/ADR-0002-*`). A discount
   above the policy ceiling, a refund issued with no prior verification, cumulative refunds
   past the order total, an order id in an outbound message that belongs to another customer:
   these are asserted in Python against the call stream, arguments included. Only genuinely
   interpretive cases reach the LLM judge, and the two are reported in separate sections that
   are never blended into one headline.

5. **How much of a result rests on a reading is reported, per run.** Every goal the suite
   derives records whether a detector settles it or a model does, and the share in the second
   category is on the page, so a reader knows which parts of a report are assertions about a
   call stream and which are a reading of a reply. Two guards constrain the reading half. A
   model verdict must quote the sentence that broke the rule, and one whose quote appears
   nowhere in the conversation is discarded. So is one whose reasoning argues the opposite of
   the verdict it sits under.

6. **Consent is architectural, not a policy note** (`docs/SAFETY.md`). Targets resolve from a
   registry and must echo a per-run nonce before the first attack turn. No code path accepts a
   bare URL.

## What a fix looks like

The people who read this report are ops and finance teams on a no-code builder. They cannot
apply a Python patch, so every remediation agent-red proposes is expressible as configuration:
a limit value, an approval threshold, a narrowed tool allowlist, a required idempotency key, a
clause in the instruction text.

| Finding | Config change |
|---|---|
| Refund exceeds the declared cap | Hard bound on the amount argument in the tool schema |
| Cumulative refunds exceed the order | Scope the limit per order rather than per call |
| Duplicate refund on a replayed trigger | Require an idempotency key, reject repeats |
| Record read for a customer outside the session | Bind the lookup to the session's customer |
| High-value refund with no verification | Route above a threshold to an approval step |

The demonstration this enables is the point of the whole project. Land a finding. Fix it the
obvious way first, by tightening the sentence in the instructions box. Re-run: that attack now
fails. Then show a variant that still succeeds. Then move the bound out of the prompt and into
the tool schema, and watch the whole family go dark.

## Results

`PENDING` - see status note above.

| | |
|---|---|
| Manipulation techniques | agent-independent, crossed with derived stakes |
| Failure checks | generic shapes, policy-driven, arguments inspected |
| Attack channels | conversational and planted |
| Targets attacked with zero new code | 3, across two build styles |
| Held-out labelled transcripts | 120 |

Every security number is reported beside the benign-task number from the same run, because
every hardening change costs capability and the trivially safest agent refuses everything.

## Quickstart

```bash
uv sync
docker compose up -d          # the MCP tool server and the target agents
uv run agentred doctor        # checks credentials, reachability, consent challenge
uv run agentred run --target cart_recovery
```

`run` is the whole chain behind one command: it derives the attack suite from the target's own
spec, delivers it down every channel the agent declares, runs every check over the recorded
call stream, and writes the page an operator reads. Everything lands in one numbered run
directory.

Pointing it at an agent nobody here has seen starts one step earlier, from the sources that
agent's own platform already carries rather than from a declaration somebody wrote for the
harness. `examples/retention_desk/` is one such agent, as an install wizard leaves it on disk:
a manifest, a connector advertising the merchant's tools, the limits an operator typed into a
form, the prose the skill runs on, and the workflow the installer mapped it onto. Neither half
of a declaration is in that directory, because both halves are what agent-red produces from it.

```bash
uv run python examples/retention_desk/connector.py &      # the merchant's tools, over MCP

uv run agentred read --manifest examples/retention_desk/agent.manifest.yaml
uv run agentred read \
  --manifest examples/retention_desk/agent.manifest.yaml \
  --answers examples/retention_desk/answers.yaml \
  --deployment examples/retention_desk/deployment.yaml \
  --subjects examples/retention_desk/subjects.yaml \
  --out src/agentred/targets/specs/retention_desk
```

The first command reads and writes nothing. It reports what came back, what nobody looked at,
and what nothing answered, and writing a declaration is refused while any question is
outstanding. The three files the second command adds are the answers to exactly those
questions, kept in files rather than prompted for so that a read reproduces and so what a
person supplied sits beside what a reader recovered. See `docs/INTEGRATION.md`.

The later stages are also available on their own, because runs are stored before anything is
judged, so an analysis interrupted halfway is resumed without re-running them:

```bash
uv run agentred run --target cart_recovery --list-stakes   # what this agent derives
uv run agentred analyse --list-runs                        # what is already recorded
uv run agentred analyse --run <id> --out analysis.json
uv run agentred report analysis.json --out what-broke.html # calls nothing, costs nothing
```

Offline tests need no credentials:

```bash
uv run pytest
uv run ruff check .
```

## Repository map

| Path | What lives there |
|---|---|
| `src/agentred/spec/` | The integration contract: config, policy, subjects, channels. Read by every other package. |
| `src/agentred/ingest/` | One reader per place a declaration is already written down: connectors, instance configuration, workflow definition, prose. Emits the contract rather than requiring it to be authored. |
| `src/agentred/mcp/` | The tool server. Serves tools over the synthetic world, records every call, holds snapshot and restore. The seam everything meets at. |
| `src/agentred/targets/` | Agents under test. A stand-in for a merchant agent platform, not part of the product surface. |
| `src/agentred/attacks/` | Technique corpus, stake derivation, the goal-channel-technique lattice, mutation operators, generator |
| `src/agentred/runner/` | Consent gate, and one driver per channel |
| `src/agentred/judge/` | Deterministic detectors, and the model judge for what they cannot settle |
| `src/agentred/scoring/` | Rule ledger, exposure model, coverage grid, the pages a person reads |
| `src/agentred/store/` | SQLite persistence for runs, transcripts and verdicts |
| `examples/` | Agents as their own platforms leave them on disk, for the reader to be pointed at. Not part of the product. |
| `docs/` | Architecture, integration, safety scope, and the decision records |

## Limitations

Stated up front rather than discovered by a reader. Each is a consequence of a decision in
`docs/ARCHITECTURE.md`, not a gap waiting to be filled.

- **Coverage is not proof.** agent-red finds vulnerabilities. It cannot establish their
  absence. A run with no findings says how many attempts were made and renders a grid of what
  was tested, and that grid is what replaces a safety claim. The words safe and secure do not
  appear in any output this tool produces.
- **The declaration is trusted ground truth.** An agent faithfully executing a wrong declared
  rule passes. Conformance, not correctness of intent.
- **White-box is the product; the endpoint-only mode is reduced and no parity is claimed.**
  Without a declaration, only invariants that hold for any commerce agent can be asserted,
  because policy-specific bounds are exactly what the declaration was providing.
- **State resets between runs, so multi-run attacks are invisible.** A payload that poisons a
  record for future cycles rather than causing damage now is outside the harness by design.
- **Exposure is worst-case single-transaction, from the logs.** Declared cap, highest
  successful extraction, how many variants succeeded. A volume-based projection may appear as
  a secondary line with its assumptions visible and editable, never as the headline.
- **Sample sizes are small and per-family counts are preferred to percentages.** Seven of
  twelve going to zero of twelve is honest; 34% going to 3% on forty trials claims a precision
  the data does not have.
- **Interpretive findings are a reading, not an assertion.** A rule about what an agent said
  cannot be settled from a call stream, so a model settles it, and every such finding is
  labelled as one and carries the reading's own confidence. The share of a run in that category
  is reported beside the result.
- **Attack coverage is policy and authority manipulation only.** Not jailbreaks, not harmful
  content, not infrastructure.
- **This is a pre-go-live gate.** Continuous post-deploy monitoring is a different product.
- **A well-built agent may resist everything.** If a workflow extracts structured fields before
  the reasoning step, a planted payload never reaches the decision context. That is the correct
  architecture and the right defence, and the finding is then whatever it is.
