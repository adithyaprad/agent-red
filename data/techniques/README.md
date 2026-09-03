# Techniques

Twelve ways to get somebody to do something they were told not to do. They are written once,
by hand, and they are the same twelve against every agent agent-red is ever pointed at.

**Nothing here names a domain.** No product, no money, no order, no customer. A technique
describes a shape of pressure; what that pressure is applied to comes from the agent's own
spec, derived in `attacks/stakes.py`. Crossing the two is what makes the suite a function of
the agent under test rather than of whoever wrote the corpus. A test fails the build if a
domain word appears in `attacks/` or `judge/detectors/`, and these files are held to the same
line by review.

## Two channel families

An attack either talks to the agent or is written into a field the agent later reads
(ADR-0006), and a technique declares in `channels` which of the two it survives.

Most of the persuasion corpus is an arc: an opening, a ladder to climb when the agent holds, a
sign that it is working, a point at which to stop. None of that means anything in a field
nobody replies to, and rendering it there anyway would put a cell on the coverage grid that
nothing meaningful was tried in. Three of the eight conversational techniques do survive it,
because their mechanism is that the reader accepts something without being asked to, and that
works in one string as well as in six turns. Four techniques exist that only make sense
written into a record.

A technique run on a family it does not declare is refused at construction, and a corpus with
nothing valid on a family is refused at load, because a channel that derives no attacks reads
on a grid as a channel that was tried and held.

## The fields

| Field | What it is |
|---|---|
| `id` | Stable identifier. Appears in every verdict and on the scorecard. |
| `name` | What a person would call it. |
| `channels` | The families this survives: `conversation`, `planted`, or both. |
| `premise` | The one sentence that makes it work. The mechanism, not the wording. |
| `pressure` | Where the force comes from, so two techniques are not the same one twice. |
| `in_plain_words` | What happened, for somebody who reads one transcript and no taxonomy. |

Required on a technique valid on `conversation`, and refused on one that is not, because
prose nothing reads is prose the next editor maintains believing it runs:

| Field | What it is |
|---|---|
| `arc` | How it develops over turns. An attack that says everything in turn one is a filter test, not an attack. |
| `escalation` | What to do when the agent holds. This is the half that separates a real attempt from a single knock. |
| `tells` | How to recognise it working, so the attacker knows to push rather than restart. |
| `fails_when` | When to stop. An attacker that cannot give up spends the whole turn budget on a wall. |
| `exemplars` | Two or three hand-written openings, on an abstract stake, that set the persuasiveness bar. |

Required on a technique valid on `planted`, under `planted:`:

| Field | What it is |
|---|---|
| `rendering` | How the pressure reads as one passage written into a record. There is no next turn, so it works on the reading or not at all. |
| `exemplars` | Two to four hand-written field contents, on an abstract stake. Separate from the conversational ones because an opening turn and a forged note are not the same piece of writing. |

## What the exemplars are for

They are not templates and are never sent verbatim. They are the calibration sample shown to
the attacker model, because "be persuasive" produces a different bar in every run and a written
example does not. They are deliberately written against an abstract stake, so that reusing them
against a real agent requires composing rather than copying.

## Why twelve

Twelve is a hypothesis, not a result. Eight was the point where the conversational shapes
stopped feeling distinct when the corpus was written, and four more were added for the planted
family, where the pressure comes from the shape of the text rather than from an argument. One
of those four asks for the thing plainly, with no disguise at all, and it is the control: if it
works, none of the elaborate ones were necessary, and if the elaborate ones work and it does
not, the difference is what the report is about.

If a full run shows two of these finding everything and the rest finding nothing, the corpus
gets reshaped rather than extended, and that finding is worth more than the corpus was.
