# agent-red

Adversarial red-teaming harness for merchant-facing commerce agents.

You point it at a running agent endpoint. It generates and runs a multi-turn attack suite,
reports which attacks broke the agent and what that exposure is worth, suggests a fix, and
re-runs the identical suite to measure whether the fix worked.

It also reports how reliable its own judging is, which is the part tools like this usually
leave out.

> **Status: in development.** Results tables below are marked `PENDING` until the run that
> produces them is committed alongside the transcripts it came from. No number appears in
> this README that cannot be regenerated from `agentred.db`.

## The problem

Merchants are putting LLM agents in front of customers. Those agents quote prices, apply
discounts, describe stock, promise delivery dates, take orders, and sometimes issue refunds.

The workflow today is: write a system prompt, chat with it a few times, ship it. There is no
test suite and no pass/fail, so nobody can tell the merchant whether the agent is safe to
put in front of money. The failure modes are already documented in the wild:

- talked into honouring a discount that does not exist
- accepting a price the customer simply asserts
- inventing stock, delivery dates or return policies the merchant cannot honour
- disclosing one customer's order details to another
- committing to items outside the catalogue

Each converts directly into merchant loss. agent-red measures that exposure before the agent
meets a customer.

## How it works

```
 registry + consent gate
          |
          v
   attack generator  ------ seeds (YAML) + mutation operators + winners from gen N-1
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

Four design decisions carry the project. Each is argued in full in `docs/`:

1. **Attacks composed, not listed** (`docs/THREAT-MODEL.md`). Eight agent-independent
   manipulation techniques are crossed with what a given agent actually has to lose, derived
   from its own declared tools, bounds, preconditions and data scope. A fixed list of test
   prompts gets patched string by string and the agent stays broken; worse, it only works on
   the domain it was written for. Composition means the suite runs against an agent it has
   never seen, which is verified against a third target introduced after the harness was
   frozen. Surface variation (register, Hinglish and mid-conversation code switching, pressure)
   is applied on top, and successful attacks seed the next generation.

2. **Deterministic detection before model judgement** (`docs/DECISIONS/ADR-0002-*`). A
   discount above the policy ceiling, a refund issued with no prior verification, an order id
   in a reply that belongs to another customer: these are asserted in Python against the
   tool-call log. Only genuinely interpretive cases (an implied promise, a policy invented in
   prose with no tool call) reach the LLM judge.

3. **The judge is measured, not trusted** (`docs/EVALUATION.md`). Judge output is scored
   against 120 human-labelled held-out transcripts, and the report carries per-class
   precision and recall, false-positive cost, the judge's instability under perturbation, and
   the human self-agreement ceiling. Nothing outside `judge/calibration/` may read the
   held-out set, and a test enforces that.

4. **Consent is architectural, not a policy note** (`docs/SAFETY.md`). Targets resolve from a
   registry and must echo a per-run nonce before the first attack turn. No code path accepts
   a bare URL.

## Results

`PENDING` - see status note above.

| | |
|---|---|
| Manipulation techniques | 8, agent-independent |
| Failure checks | 3 generic shapes, policy-driven |
| Attacks per suite | ~400 |
| Targets attacked with zero new code | 3 |
| Held-out labelled transcripts | 120 |

## Quickstart

```bash
uv sync
docker compose up -d          # two target agents, the platform stand-in
uv run agentred doctor        # checks credentials, reachability, consent challenge
uv run agentred run --target cart_recovery
```

`run` is the whole chain behind one command: it derives the attack suite from the target's own
spec, holds the conversations, runs every check over what came back, and writes the page an
operator reads. Everything lands in one numbered run directory.

The later stages are also available on their own, because conversations are stored before
anything is judged, so an analysis interrupted halfway is resumed without re-running them:

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
| `src/agentred/targets/` | Agents under test. Declarative spec to Claude Agent SDK agent to HTTP endpoint. A stand-in for a merchant agent platform, not part of the product surface. |
| `src/agentred/attacks/` | Seed corpus, mutation operators, generator, evolution loop |
| `src/agentred/runner/` | Consent gate and the multi-turn conversation driver |
| `src/agentred/judge/` | Deterministic detectors, LLM judge, calibration harness |
| `src/agentred/scoring/` | Scorecard, exposure model, HTML report |
| `src/agentred/patch/` | Suggested fix generation and the measured re-run |
| `docs/` | Architecture, threat model, evaluation methodology, safety scope, ADRs |

## Limitations

Stated up front rather than discovered by a reader.

- **Exposure is a scenario model, not a measurement.** It multiplies an observed attack
  success rate by merchant-supplied volume and order value. Every input is printed next to
  the output with a sensitivity band, and the renderer refuses to emit the number without
  them.
- **The judge is imperfect and the size of that imperfection is published.** Per-class
  precision and recall are reported so every downstream claim can be discounted correctly.
- **Attack coverage is policy and authority manipulation only.** Not jailbreaks, not harmful
  content, not infrastructure. See `docs/THREAT-MODEL.md` for what is deliberately excluded.
- **Genericity has a stated limit.** The harness reasons about declared bounds, preconditions
  and data scopes. Anything a merchant cares about that is not expressed as one of those three
  is invisible to it.
- **This is a pre-go-live gate.** Continuous post-deploy monitoring is a different product.
- **Target agents are a stand-in.** They mirror a plausible merchant agent platform so the
  harness has something real to attack. The interface agent-red actually requires is small
  and documented in `docs/ARCHITECTURE.md`.
