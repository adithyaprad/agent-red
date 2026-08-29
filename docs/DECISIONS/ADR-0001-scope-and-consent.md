# ADR-0001: Scope, and consent as a code path

Status: accepted, 2026-08-29.

## Context

agent-red generates working social-engineering attacks and fires them at a live agent. That
is the only way to measure whether an agent can be talked into losing a merchant money, and
it is also, pointed at somebody else's endpoint, an attack tool.

The difference between a verifier and an attack tool cannot be a sentence in a README. A
reader who wants the harness to be safe will believe the sentence; nobody else has to. So
the distinction has to be a property of the code: something that is true whether or not the
operator wants it to be.

## Decision

**Scope.** agent-red is a pre-go-live gate for an agent you control. It is not a
production monitor, not a scanner, and not a service that will test somebody else's agent
for them. Continuous post-deploy monitoring is a genuinely different product and is named
as future work rather than half-built here.

**Consent is a code path, not a policy.** There is no function anywhere in the tree that
takes a URL and sends an attack turn to it. A target is resolved from a registry, and
before the first attack turn the runner sends a challenge string and requires the target to
echo it back. An endpoint that cannot be configured to echo the challenge cannot be
attacked by this harness, and that is the intended behaviour rather than a limitation to
work around. Implemented in `runner/consent.py`, and the absence of an unregistered path is
worth a test of its own.

**Money actions in the stand-in targets are test mode only, asserted in code.** Target
agents that call a payments API use test-mode credentials, and the assertion is made by the
target rather than trusted to the operator. An operator who is about to point a refund
attack at live credentials should get a crash, not a warning.

**Attacks are held in the repository, and the repository is a defence artifact.** The seed
corpus, the mutation operators and the generator are all checked in. What makes this
defensible is not that the attacks are hidden but that they are attached to a verifier: the
output is a scorecard with measured precision and recall, not a payload list. The harness
is judged on defence metrics for that reason.

**Refusals.** The harness will not be adapted to attack an agent the operator does not
control, and no consent bypass will be added for a demo, a benchmark or a deadline. If a
target cannot echo a challenge, the answer is to configure it, not to relax the gate.

## Alternatives considered

**A consent flag on the command line.** One `--i-have-permission` argument, defaulted off.
Rejected: a flag is a sentence with a parser attached. It records an assertion rather than
establishing anything, and every operator who should not pass it will pass it.

**An allowlist of hostnames.** Better than a flag, and still one file edit away from
pointing at a stranger. The challenge echo requires cooperation from the far end, which is
what consent actually means.

**Not publishing the attack corpus.** Attractive, and it would make the calibration numbers
unreproducible, which costs more than the corpus is worth to an attacker. The techniques
here are documented in public; the corpus's value is that it is measured.

## Consequences

Every target, including the stand-ins in `targets/` and the unrelated third target used for
the genericity proof, must implement the challenge echo. That is a small tax on writing a
target and it is paid once.

Nobody can point agent-red at an arbitrary hosted agent to try it out. The three supported
modes are reading a finished scorecard, bringing your own key and running against the
bundled targets, or running the whole thing locally.

The gate is only as good as its coverage, so a code path that reaches a target without
passing through consent is a bug of the highest severity in this repository, not a
convenience. `docs/SAFETY.md` states the operator-facing scope and the refusals; this ADR
records why they are enforced where they are.
