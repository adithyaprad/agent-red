# agent-red

A pre-deploy safety gate for agents that touch money.

Point it at an agent before it goes live. It reads the agent's own declaration, derives a
threat model from it, attacks it through the fields a real adversary controls, and reports
what broke with the exact transcript.

**agent-red verifies that an agent conforms to its declaration. It does not verify that the
declaration is correct.** That distinction is load-bearing and is repeated wherever a number
is reported.

New here? [Install](#install), then [give it model access](#model-access), then
[attack a bundled agent](#attack-a-bundled-agent). Pointing it at an agent of your own is
[further down](#point-it-at-your-own-agent).

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
   attack generator  ---- techniques x derived stakes x channels x mutations
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
          MCP server  ---->  tools over the world
                       |
                       v
          call stream + world diff      (recorded here, not reported by the agent)
                       |
                       v
   deterministic detectors  <---------------------  policy manifest
          |
          | (only the interpretive residue)
          v
      LLM judge
          |
          v
   coverage grid + rule ledger  ---->  the page an operator reads
```

Six design decisions carry the project. Each is argued in full in `docs/`.

1. **The oracle sits at the tool boundary, never inside the agent**
   (`docs/DECISIONS/ADR-0005-oracle-at-the-tool-boundary.md`). Every violation is observed in
   the call stream passing through the MCP server, plus the resulting world diff. Not by
   reading a workflow trace, not by reading graph state, not by asking the agent what it did.
   An agent's reply is framework-specific; an agent's hands are not.

2. **Attacks arrive through the fields an adversary controls, not only a chat box**
   (`docs/DECISIONS/ADR-0006-attack-channels.md`). Payloads are written into dispute reason
   text, customer names, delivery instructions and product titles, and then the agent's real
   trigger is fired. A cron-driven agent has no conversational turns and is still fully
   attackable. Multi-turn conversation remains a first-class channel, because it is the only
   one where an attack can adapt to what the agent just said.

3. **Attacks are composed, not listed.** Twelve agent-independent manipulation techniques in
   `data/techniques/` are crossed with what a given agent actually has to lose, derived from
   its own declared tools, bounds, preconditions and data scope, and with the channels it
   declares. Nine surface mutations vary register, language and pressure on top. A fixed list
   of test prompts gets patched string by string and the agent stays broken; worse, it only
   works on the domain it was written for.

4. **Deterministic detection before model judgement.** A discount above the policy ceiling, a
   refund issued with no prior verification, cumulative refunds past the order total, an order
   id in an outbound message that belongs to another customer: these are asserted in Python
   against the call stream, arguments included, by the seven detectors in
   `src/agentred/judge/detectors/`. Only genuinely interpretive cases reach the LLM judge, and
   the two are reported in separate sections that are never blended into one headline.

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

## Install

Requires Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and Docker if you want the
bundled agents brought up for you.

```bash
git clone <this repository>
cd agent-red
uv sync
```

Offline tests need no credentials and no running process:

```bash
uv run pytest
uv run ruff check .
```

## Model access

agent-red calls a model in two places: to compose attack turns, and to judge the interpretive
residue the detectors cannot settle. The bundled agents under test call one too, to answer.

There are three routes to a model and you need exactly one of them. Which one is resolved
from the environment, so nothing is passed on the command line and no key is ever written to
a file the repository tracks.

| Route | What it is | What you set |
|---|---|---|
| `first_party` | The Anthropic API directly | `ANTHROPIC_API_KEY` |
| `bedrock` | Amazon Bedrock, partner-operated | `AWS_REGION`, plus AWS credentials the SDK can find |
| `aws` | Claude Platform on AWS, Anthropic-operated | `AWS_REGION` and `ANTHROPIC_AWS_WORKSPACE_ID` |

**If you have an Anthropic API key and nothing else, that is enough.** Export it and skip the
rest of this section:

```bash
export ANTHROPIC_API_KEY=<your key>
```

With nothing set, the route is detected from whichever credentials are present, preferring
Claude Platform on AWS, then Bedrock, then the first-party API. `AGENTRED_LLM_ROUTE` overrides
that detection and takes `first_party`, `bedrock` or `aws`. Set it whenever you hold more than
one set of credentials and want to be sure which is in use:

```bash
export AGENTRED_LLM_ROUTE=bedrock
export AWS_REGION=ap-south-1
```

Two things about Bedrock are worth knowing before the first request fails. Model access has to
be granted for the models in question in that region, and the ids are served under the
`global.anthropic.` cross-region inference profile prefix, which agent-red applies for you: a
bare `anthropic.claude-opus-5` is listed by the API and cannot be invoked on demand.

Copy `.env.example` to `.env` if you would rather keep these in a file. `.env` is gitignored.
Nothing in agent-red reads it automatically, so load it yourself:

```bash
cp .env.example .env
# edit it, then
set -a; source .env; set +a
```

`uv run agentred doctor` reports which route resolved, and says what is missing if none did.

## Attack a bundled agent

Three stand-in agents live in `src/agentred/targets/`. They are not the product: they exist so
that the harness has something to be pointed at, and so that the claim that it transfers is
demonstrated rather than asserted.

| Target | What it does | Woken by |
|---|---|---|
| `cart_recovery` | Reaches out to shoppers who did not check out. Offers discounts, quotes prices, promises delivery. | A schedule, with no user turn in it |
| `dispute_handler` | Answers card chargebacks from the shop's own records. Issues refunds and store credit. | A request naming a dispute or an order |
| `retention_desk` | Subscription retention. Discounts a recurring price, refunds charges, pauses billing. | A request naming a subscription |

The first two share a hand-authored shop and one tool server. The third has no shop written for
it at all: its world is generated from its own declaration, and its declaration was read off the
files its platform carries rather than written here. See
[Point it at your own agent](#point-it-at-your-own-agent).

### With Docker

```bash
export ANTHROPIC_API_KEY=<your key>    # or the AWS variables above
docker compose up -d
uv run agentred doctor
uv run agentred run --target dispute_handler
```

`docker compose up -d` brings up five containers: one tool server for the two hand-authored
agents, a second for the generated one, and the three agents themselves. Whichever model route
your shell has is passed through to them.

### Without Docker

The tool server comes first, always, because the agents reach every capability through it and
that is also where the call stream is recorded. Three processes bring up the two hand-authored
agents; `retention_desk` adds two more below.

```bash
export AGENTRED_TARGET_MODE=test        # the agents refuse to start without this

# 1. the tool server for the two hand-authored agents: tools on 8090, control on 8091
uv run python -m agentred.mcp.serve \
  --spec src/agentred/targets/specs/cart_recovery \
  --spec src/agentred/targets/specs/dispute_handler &

# 2. the agents
AGENTRED_TOOL_SERVER_URL=http://localhost:8090 \
  uv run python -m agentred.targets.serve \
  --spec src/agentred/targets/specs/cart_recovery --port 8081 &
AGENTRED_TOOL_SERVER_URL=http://localhost:8090 \
  uv run python -m agentred.targets.serve \
  --spec src/agentred/targets/specs/dispute_handler --port 8082 &

uv run agentred doctor
```

The two ports on the tool server are not interchangeable. 8090 serves the tools and is what the
agents are told about. 8091 is the control face: it reads the recorded call stream, restores
worlds and plants payloads into them, and no agent is ever given its address.

`retention_desk` needs its own tool server, because its shop is derived from its own
declaration and one shop comes from one declaration:

```bash
uv run python -m agentred.mcp.serve \
  --spec src/agentred/targets/specs/retention_desk \
  --generated --port 8094 --control-port 8095 &
AGENTRED_TOOL_SERVER_URL=http://localhost:8094 \
  uv run python -m agentred.targets.serve \
  --spec src/agentred/targets/specs/retention_desk --port 8083 &
```

The ports above are the ones in `targets.registry.yaml`, which is the only place agent-red
looks for a target. `doctor` fails loudly if a server is up on the wrong one or serving the
wrong agent.

### What a run does

```bash
uv run agentred run --target dispute_handler
```

One command, four stages: read the rules the agent's instructions state, derive and deliver the
attack suite down every channel the agent declares, run every check over the recorded call
stream, and write the page an operator reads. A full suite is hundreds of model calls and takes
tens of minutes, so start smaller while you are finding your feet:

```bash
uv run agentred run --target dispute_handler --list-stakes      # what this agent has to lose
uv run agentred run --target dispute_handler --list-channels    # what a stranger can write into
uv run agentred run --target dispute_handler --limit 10         # ten attacks, not the suite
```

Everything lands in one numbered directory: `run.json`, `calls.jsonl`, `cost.json`,
`analysis.json` and `what-broke.html`. Runs go to `out/runs/` by default, which is gitignored,
because a run holds complete transcripts of an agent being manipulated and the prompts that did
it. Set `AGENTRED_RUNS_DIR` to put them somewhere else.

**Restart every server after any spec change.** A server loads its spec once and holds it. A
stale one is refused at preflight rather than after a paid-for suite, but the fix is a restart.

## Point it at your own agent

The declaration is the one thing agent-red cannot run without, and it is the one thing nobody
sets out to write. It does not have to be written. An agent that runs at all has already
recorded almost all of it, in the connectors it is wired to, the limits its operator typed into
a form, the workflow it runs on, and its prose.

`examples/retention_desk/` is an agent as an install wizard leaves it on disk. Nothing in that
directory was written for agent-red, and neither half of a declaration is in it.

```bash
uv run python examples/retention_desk/connector.py &      # the merchant's tools, over MCP

uv run agentred read --manifest examples/retention_desk/agent.manifest.yaml
```

That reads and writes nothing. It reports what came back, what nobody looked at, and what
nothing answered. Writing a declaration is refused while any question is outstanding, and the
questions are named with their subject rather than defaulted, because a tool recorded as costing
nothing is a tool no attack is aimed at.

Three files answer exactly those questions, and they are kept in files rather than prompted for
so that a second read reproduces byte for byte and so what a person supplied sits beside what a
reader recovered:

```bash
uv run agentred read \
  --manifest examples/retention_desk/agent.manifest.yaml \
  --answers examples/retention_desk/answers.yaml \
  --deployment examples/retention_desk/deployment.yaml \
  --subjects examples/retention_desk/subjects.yaml \
  --out src/agentred/targets/specs/retention_desk
```

| Flag | Answers |
|---|---|
| `--answers` | What a wrong call to each tool costs. No connector protocol carries it. |
| `--deployment` | What each tool does to the merchant's records, and which fields an adversary writes. |
| `--subjects` | Who the harness may act as. A platform has no reason to record who a red team may impersonate. |

Then add the agent to `targets.registry.yaml`, serve it, and run. Adding an entry to that file
is an assertion that you control the agent it names; there is no command-line argument that
takes a URL. `docs/INTEGRATION.md` is the full account, including what each source recovers and
what it structurally cannot.

## Commands

| Command | What it does | Costs model calls |
|---|---|---|
| `agentred doctor` | Checks the route, the registry, every spec, every tool server and every consent handshake | no |
| `agentred read` | Recovers a declaration from the sources an agent's own platform carries | yes |
| `agentred world --target <name>` | Builds the shop for an agent from its declaration and prints which rules it made reachable, gaps included | no |
| `agentred run --target <name>` | The whole chain: derive, attack, check, render | yes |
| `agentred analyse --run <id> --out <file>` | Re-runs every check over conversations already recorded, contacting no target | yes |
| `agentred report <analysis> --out <page>` | Renders the operator page from a finished analysis | no |
| `agentred cost --run <dir>` | Prices a finished run from its recording and its stored turns | no |

Everything after a conversation runs offline against `data/agentred.db`, so an analysis
interrupted halfway is resumed without paying for the conversations again:

```bash
uv run agentred analyse --list-runs
uv run agentred analyse --run <id> --out analysis.json
uv run agentred report analysis.json --out what-broke.html
```

## What the repository carries

Counts, not results. Every one is checkable from the tree.

| | |
|---|---|
| Manipulation techniques | 12, agent-independent, in `data/techniques/` |
| Surface mutations | 9: register, language including Hinglish and mid-conversation code-switching, pressure, indirection |
| Deterministic detectors | 7, covering nine failure shapes, in `src/agentred/judge/detectors/` |
| Attack channels | 2: conversational and planted |
| Targets attacked with zero new code | 3. The third arrived with no declaration at all: its config and policy were read off the files its own platform carries, and the shop it was attacked in was derived from them. |

**No run output is committed to this repository.** A run holds complete transcripts of an agent
being manipulated and the exact prompts that did it, so results live in the run directory and
travel as artifacts rather than as a table in this file. Every number a run reports can be
traced back to the conversation that produced it: the store is one SQLite file anyone can open
with the `sqlite3` shell.

## Repository map

| Path | What lives there |
|---|---|
| `src/agentred/spec/` | The integration contract: config, policy, subjects, channels. Read by every other package. |
| `src/agentred/llm/` | The model client. One environment switch across the three routes. |
| `src/agentred/ingest/` | One reader per place a declaration is already written down: connectors, instance configuration, workflow definition, prose. Emits the contract rather than requiring it to be authored. |
| `src/agentred/mcp/` | The tool server. Serves tools over the world, records every call, holds snapshot and restore, and generates a world from a declaration. The seam everything meets at. |
| `src/agentred/targets/` | Agents under test. A stand-in for a merchant agent platform, not part of the product surface. |
| `src/agentred/attacks/` | Technique corpus, stake derivation, the goal-channel-technique lattice, mutation operators, generator |
| `src/agentred/runner/` | Consent gate, and one driver per channel |
| `src/agentred/judge/` | Deterministic detectors, and the model judge for what they cannot settle |
| `src/agentred/scoring/` | Rule ledger, coverage grid, consistency, breaking point, cost, and the pages a person reads |
| `src/agentred/store/` | SQLite persistence for runs and transcripts |
| `examples/` | Agents as their own platforms leave them on disk, for the reader to be pointed at. Not part of the product. |
| `scripts/` | Two development entry points: a small live run that checks the composed turns against the hand-written exemplars, and the page that explains one run to whoever is building the harness rather than to a merchant. |
| `docs/` | Architecture, integration, world design, safety scope, and the decision records |

## Limitations

Stated up front rather than discovered by a reader. Each is a consequence of a decision in
`docs/ARCHITECTURE.md`, not a gap waiting to be filled.

- **Coverage is not proof.** agent-red finds vulnerabilities. It cannot establish their
  absence. A run with no findings says how many attempts were made and renders a grid of what
  was tested, and that grid is what replaces a safety claim. The words safe and secure do not
  appear in any output this tool produces.
- **The declaration is trusted ground truth.** An agent faithfully executing a wrong declared
  rule passes. Conformance, not correctness of intent.
- **A reading is reported as a reading.** What an agent said cannot be settled from a call
  stream, so a model settles it, and the share of a run resting on that half is printed beside
  the result. The reading is constrained structurally rather than trusted: a verdict must quote
  the sentence it turns on, and one whose quote appears nowhere in the conversation, or whose
  reasoning argues the opposite of its own outcome, is discarded. Each verdict shows its own
  confidence, which is displayed and decides nothing, because a self-reported number is not an
  independent one.
- **The grid is deterministic and the words in it are generated.** The same declaration derives
  the same stakes, channels, techniques and mutations every time, so two runs of one suite ask
  the same questions. A model composes the turns that ask them, so they are the same experiment
  rather than the same transcript.
- **State resets between runs, so multi-run attacks are invisible.** A payload that poisons a
  record for future cycles rather than causing damage now is outside the harness by design.
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

## Safety

agent-red generates working social-engineering attacks and sends them at a live agent. It is a
gate for an agent you control, before it is in front of customers. It is not a scanner and not
a service for testing somebody else's agent.

Consent is a code path rather than a policy note: a target is resolved by name from
`targets.registry.yaml`, has to echo a fresh nonce before the first attack turn, and the proof
of that exchange is a token that every function sending a turn requires. The bundled agents
assert test mode at import in two independent ways and crash on startup rather than warn.
`docs/SAFETY.md` states the scope and the refusals in full.
