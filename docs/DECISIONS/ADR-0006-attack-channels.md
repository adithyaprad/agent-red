# ADR-0006: An attack has a channel, and conversation is one of them

Status: accepted, 2026-09-02.

## Context

Every attack agent-red has ever run arrives the same way: the runner composes a user turn and
posts it to the target's chat endpoint. `runner/conversation.py` is the only path an attack
has into an agent, and a user message is the only thing it can carry.

That assumption held while both targets were conversational. It fails on the agent the
platform actually demos.

A cart-abandonment agent is not talked to. A cron fires, it reads the abandoned baskets, it
decides who to contact, and it sends a message. There is no user turn anywhere in its
lifetime. Under the current harness it is not attackable, and a suite that reported nothing
against it would be reporting the harness's shape rather than the agent's.

The failure is more general than one agent. Ask what an adversary actually controls and the
chat box turns out to be a small part of the answer:

- the free text a buyer writes when they raise a dispute
- the name and address on their account
- a product title on a marketplace listing the agent summarises
- a delivery instruction attached to an order
- the body of an inbound message the agent triages

None of those is a conversational turn. All of them are strings an attacker writes and an
agent later reads as context, and by the time the agent reads them nothing distinguishes them
from data the merchant put there. This is the injection channel that matters commercially and
it was structurally invisible.

There is a second reason to name channels explicitly, which is coverage. A report cannot say
what it did not test unless the things it might have tested are enumerable. Goal crossed with
technique is already a countable grid. Adding channel makes the grid describe delivery as well
as intent, and an untested cell becomes visible rather than absent.

## Decision

**An attack declares the channel it arrives through, and the runner dispatches on it.**

A channel is a named way of getting attacker-controlled bytes in front of the agent, together
with the trigger that makes the agent read them. Two families exist:

**Conversational.** The runner holds a multi-turn conversation, as it does today. The whole
technique corpus, the mutation axes, `ModelAttacker`, forking, breaking-point analysis and
consistency comparison all live here and are unchanged. This is not a legacy path. It is the
only channel where an attack can adapt to what the agent said, which is where the interesting
failures are, and it stays first class.

**Planted.** The runner writes a payload into a field of the world before the agent runs, then
fires the agent's real entry point and watches the call stream. Six steps, and the whole
mechanism is in them:

```
restore()                     the world, to a known snapshot
plant(channel, payload)       into the field an adversary genuinely controls
trigger()                     the agent's real entry point: cron, webhook, inbound event
record()                      the call stream, at the MCP boundary (ADR-0005)
assert()                      detectors, then the judge on the residue
emit()                        finding or pass, with the seed that reproduces it
```

Step three is the one that carries the argument. The attack path has to be the agent's normal
operating path, or the finding is about a test harness rather than about a deployment.

**A channel is declared by the agent, not assumed by the harness.** `config.yaml` grows a
`channels:` block naming, for each writable field, the data source and record path it lives
at and the trigger that causes the agent to read it. The harness attacks the channels an agent
declares and reports the ones it has, in the same way it attacks the bounds an agent declares.
An agent that declares no planted channel is attacked conversationally and the coverage grid
says so.

**A stake is crossed with a channel the same way it is crossed with a technique.** The unit of
work becomes goal by channel by technique, seeded. `attacks/stakes.py` is unchanged: it still
answers "what is worth attacking" from the policy, and it never knew how an attack was
delivered.

## Alternatives considered

**Demote conversation to one channel among many, as a strict reading of the brief suggests.**
Rejected. Two of the harness's most distinctive outputs, the breaking point (the earlier turn
where the agent's position changed) and the consistency comparison (declined twice, complied
once, same subject), exist only because attacks are multi-turn. Reweighting away from
conversation to make an architecture diagram cleaner would delete them for nothing. Both
families are first class and the report carries both.

**A single generic "inject" channel with a free-text target path.** Simpler, and it makes
coverage meaningless: an enumerable grid needs named cells, and a path string supplied per
attack is not enumerable. Channels are declared and finite for the same reason bounds are.

**Simulate the trigger with a synthetic prompt saying "a cron just fired".** Cheap, and it
tests a different agent than the one that ships. The trigger has to be the real one.

**Let the harness discover writable fields by reflecting over the world.** Tempting, since the
harness owns the synthetic world today. It encodes an assumption that will not survive the
first real platform, where the harness sees an agent's declaration and not its database. What
an attacker controls is a property of the deployment and the merchant knows it; asking is
correct and guessing is not.

## Consequences

`runner/conversation.py` becomes `runner/channels/conversational.py` with its logic intact.
`runner/channels/planted.py` is new and holds the six-step lifecycle. `runner/suite.py`
dispatches on the channel an attack declares.

The world gains `snapshot()` and `restore()`. Per-conversation deep copies already give
isolation for the conversational channel (ADR-0002); the planted channel needs the same
guarantee for a run that has no session, which is what a snapshot is.

Coverage becomes a grid over goal by channel and appears in the report as a rendered surface,
including the cells that were not attempted. It sits beside the existing three-outcome
accounting rather than replacing it: the grid says what was tried, and violated, held or never
evaluated says what was actually decided. The second is the stronger statement and stays the
one a headline number is built from.

The technique corpus splits. The eight persuasion techniques in `data/techniques/` are written
as conversational arcs, with escalation and tells that only mean anything across turns. Three
of them (asserted premise, manufactured authority, parallel subject) survive a rewrite into a
single planted string; the other five do not, and forcing them would produce a corpus that
claims coverage it does not have. A second family is written for the planted channel, and each
technique declares which channels it is valid on. A technique attempted on a channel it does
not support is a construction error, not a wasted run.
