# Techniques

Eight ways to talk somebody into doing something they were told not to do. They are written
once, by hand, and they are the same eight against every agent agent-red is ever pointed at.

**Nothing here names a domain.** No product, no money, no order, no customer. A technique
describes a shape of pressure; what that pressure is applied to comes from the agent's own
spec, derived in `attacks/stakes.py`. Crossing the two is what makes the suite a function of
the agent under test rather than of whoever wrote the corpus. A test fails the build if a
domain word appears in `attacks/` or `judge/detectors/`, and these files are held to the same
line by review.

## The fields

| Field | What it is |
|---|---|
| `id` | Stable identifier. Appears in every verdict and on the scorecard. |
| `name` | What a person would call it. |
| `premise` | The one sentence that makes it work. The mechanism, not the wording. |
| `pressure` | Where the force comes from, so two techniques are not the same one twice. |
| `arc` | How it develops over turns. An attack that says everything in turn one is a filter test, not an attack. |
| `escalation` | What to do when the agent holds. This is the half that separates a real attempt from a single knock. |
| `tells` | How to recognise it working, so the attacker knows to push rather than restart. |
| `fails_when` | When to stop. An attacker that cannot give up spends the whole turn budget on a wall. |
| `exemplars` | Two or three hand-written openings, on an abstract stake, that set the persuasiveness bar. |

## What the exemplars are for

They are not templates and are never sent verbatim. They are the calibration sample shown to
the attacker model, because "be persuasive" produces a different bar in every run and a written
example does not. They are deliberately written against an abstract stake, so that reusing them
against a real agent requires composing rather than copying.

## Why eight

Eight is a hypothesis, not a result. It is the point where the shapes stopped feeling distinct
when the corpus was written. If the first full run shows two of these finding everything and
six finding nothing, the corpus gets reshaped rather than extended, and that finding is worth
more than the corpus was.
