# ADR-0002: Per-conversation isolation, and forking from a checkpoint

Status: accepted, 2026-08-29.

## Context

A suite is hundreds of conversations against one target. Two questions follow from that and
they turn out to be the same question.

The first is what world state a conversation acts on. If every conversation shares one copy
of the merchant's data, then a refund extracted in conversation 14 is visible to
conversation 15, and conversation 15 is being run against an agent whose books have already
been damaged. Every rate reported afterwards is then a mixture of the attack under test and
the residue of the attacks before it, in an order nobody recorded.

The second is how to avoid paying for the same opening repeatedly. Many attacks share three
or four turns of setup: establish a plausible customer, get an order looked up, build some
rapport. Running those separately for every variation is slow, and on a metered model it is
also the largest single line in the run's cost.

## Decision

**Every conversation gets a deep copy of the world.** The target holds state per session id
and creates a fresh copy the first time it sees one. The runner does not have to ask for
isolation and cannot forget to.

**A conversation may be branched, and a branch starts from the state that existed at the
branch point.** The target records a checkpoint after every turn: the world as it stood, the
model session, and the last assistant message. Forking at turn `k` copies checkpoint `k`,
and the model session is rewound to that message rather than continued from the present.

The second half of that is not an optimisation, it is the correctness condition. Two
branches of one conversation are only comparable if they differ by the turn that was changed
and by nothing else. A branch that inherited the money spent by later turns of the parent
would differ by that turn plus several hundred rupees of damage, and every comparison built
on it would be quietly wrong in a way the output does not show.

## Alternatives considered

**One world per run, reset between conversations.** Simpler, and it fails the moment
anything runs concurrently. It also makes a conversation's starting state depend on whether
the reset ran, which is exactly the kind of thing that works for months and then does not.

**Branching by replaying the prefix turns against a fresh session.** Correct, and it pays
the full price of the prefix on both sides: the model re-reads it, and the agent re-executes
the tool calls in it, which means the branch's world is rebuilt by re-running actions rather
than copied. Cheap only when the prefix is cheap, which is the case where branching was not
needed anyway.

**Branching only from the live end of a conversation.** This is what the first
implementation did, by copying the session as it currently stood. It is cheaper because the
model session can simply be resumed. It also silently produced the wrong answer for any
branch taken before the last turn, which is the interesting case. Caught by a test rather
than by reading, which is the argument for the test.

## Consequences

Memory grows with the number of turns, because a checkpoint holds a copy of the world. The
world is small (a catalogue, a few customers, a few orders) and a conversation is at most a
handful of turns, so this is a bounded cost paid for a property that is otherwise
untestable.

The target must be told where to branch, so `POST /fork` carries a turn number. A fork
request naming a turn the conversation has not reached is refused rather than clamped: a
caller asking for turn five of a three-turn conversation believes something false about the
transcript, and continuing would hide that.
