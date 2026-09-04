# Safety scope

agent-red generates working social-engineering attacks and sends them at a live agent. That
is the only way to find out whether an agent can be talked into losing a merchant money. It
is also, pointed somewhere it should not be, an attack tool.

This document states what the harness will and will not do, and where each of those is
enforced. The reasoning behind the design is in
[ADR-0001](DECISIONS/ADR-0001-scope-and-consent.md); this is the operator-facing half.

## What agent-red is for

A pre-go-live gate for an agent you control. You point it at your own agent, before it is in
front of customers, and it tells you which conversations break it and what that would cost.

It is not a production monitor, not a scanner, and not a service that will test somebody
else's agent. Continuous post-deploy monitoring is a genuinely different product, with a
different consent story and a different reader, and it is deliberately outside this scope.

## Consent is a code path

There is no function anywhere in the tree that takes a URL and sends an attack turn to it.

1. A target is resolved **by name** from `targets.registry.yaml`. Adding an entry is an
   assertion that you control the agent it names.
2. Before the first attack turn, the harness sends a fresh random nonce and requires the
   target to echo it back, together with the agent id it believes it is and the mode it is
   running in.
3. The proof of that exchange is a consent token, which cannot be constructed anywhere
   except by the gate. Every function that sends a turn takes one, so a code path that
   reaches a target without passing the gate does not type-check.
4. The token expires. Consent is re-checked before every turn, not once at the start, so a
   long suite cannot outlive the agreement it began under.

An agent that is not configured to echo a challenge cannot be attacked by this harness. That
is the intended behaviour, not a limitation to work around. `runner/consent.py` implements
it and `tests/test_no_unconsented_path.py` asserts that nothing else in the harness speaks
HTTP to a target.

## Money actions are test mode only

The stand-in agents in `targets/` issue refunds, apply discounts and place orders. They
assert test mode at import, in two independent ways: `AGENTRED_TARGET_MODE` must be `test`,
and no credential-shaped environment variable may hold a live-mode key. Failing either one
is a crash on startup rather than a warning in a log.

An operator about to point a refund attack at live credentials gets stopped by the code, not
by remembering.

## The attacks are in the repository

The technique corpus, the mutation operators and the generator are all checked in. What makes
that defensible is that they are attached to an oracle. The output is a page saying which
declared rules an agent broke, quoted from the call stream that broke them, against a shop
that exists to make those rules reachable. It is not a payload list, and the techniques
themselves are documented in public. What the repository adds is the measurement: the checks,
the coverage grid beside them, and the accounting of which findings are assertions and which
are readings.

## Refusals

- The harness will not be adapted to attack an agent the operator does not control.
- No consent bypass will be added for a demo, a benchmark or a deadline. If a target cannot
  echo a challenge, the answer is to configure it.
- Real customer data is never used. Every shop the harness serves is invented: `data/store/`
  is written by hand and a generated one is derived from a declaration, and no record in
  either came from a person.

## What this does not protect against

An operator who owns an agent, registers it and configures it to echo a challenge can attack
it. That is the product. The gate establishes that the far end agreed, not that the person
running the harness is well intentioned, and no amount of code in this repository can
establish the second thing.

The three supported ways to use agent-red are: read a finished scorecard, bring your own key
and run against the bundled targets, or run the whole thing locally against your own agent.
