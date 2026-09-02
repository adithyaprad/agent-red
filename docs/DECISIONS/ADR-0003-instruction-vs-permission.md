# ADR-0003: An instruction and a permission are different remedies, and are never reported as one

Status: accepted, 2026-09-03.

## Context

Every finding agent-red reports ends in the same question from the person reading it: what do I
change. There are only two places a change can go.

One is the structured policy: a ceiling, an approval threshold, an allowlist, a required prior
step, an idempotency key. The tool surface then refuses the call. A limit expressed there is
enforced by something other than the model's cooperation, and after the change the violation is
not less likely, it is unreachable.

The other is the instruction text: a sentence added to the prompt saying what the agent must not
do. The tool surface is unchanged and still accepts the value. The agent usually complies. The
attack that worked yesterday works less often.

The two feel interchangeable when a report lists them side by side, and they are not. They are
different kinds of promise. Presenting a rewording as though it closed something is the single
most damaging thing this tool could do, because the operator stops looking, and the thing they
stopped looking at is still reachable.

The temptation is real and it is structural rather than careless. Instruction changes are always
available. Every finding can be answered with a sentence, and a report that always has an answer
looks more useful than one that sometimes says the only real fix requires a limit the platform
does not currently offer. A tool optimising for looking useful converges on prompt advice.

This also decides something upstream of the report. `attacks/infer_policy.py` reads the rules an
agent's prose states and compares them to what its policy declares, and the interesting output is
the gap: rules the operator wrote and never declared. That gap is only worth reporting if the two
sides mean different things. If a sentence in a prompt were equivalent to a declared bound, an
undeclared rule would be a formatting preference rather than a finding.

## Decision

**A remediation carries which of the two it is, and the report never blends them.**

A change to the policy is described as making the violation unreachable. A change to the
instruction text is described as making it less likely, in those terms, in the operator's own
words, on the page. Neither phrase is decoration: they are the claim the tool is making, and they
are different claims.

**Where a policy change exists, it is the recommendation, and an instruction change is at most an
addition to it.** A finding whose only available answer is a sentence in a prompt says so
explicitly, and says why: the limit that would have closed it is not something the declaration
can currently express. That sentence is uncomfortable to print and it is the honest output.

**The measured re-run reports the two differently.** Re-running the suite after a policy change
measures whether the violation is now refused, which is a property of the tool surface and does
not depend on which attacks were tried. Re-running after an instruction change measures how often
this particular suite got through, on this run, with this sample size, which is a rate rather
than a guarantee. A residual of zero after a policy change and a residual of zero after a
rewording are not the same result and are not printed as though they were.

**The declaration remains the ground truth for both.** agent-red checks conformance to what was
declared and does not judge whether the declaration is right. An operator who tightens a ceiling
to a wrong value gets a clean run against the wrong value. This ADR governs how a remedy is
described, not whether it is a good idea.

## Alternatives considered

**Rank remediations by expected reduction in exposure and let the number speak.** A single
ordering is easier to read and it collapses exactly the distinction that matters. An instruction
change with a large measured reduction on one run would outrank a policy change with a small one,
and the reader would take the ordering as advice. The reduction is also measured against the
suite that exists, so it rewards a rewording that happens to defeat the attacks that were tried.

**Report only policy changes, and stay silent where none exists.** Attractive for a while: it
makes every recommendation a real one. It fails on the agents that need help most. An agent built
on a platform with no declared limits has no policy to change, and the reader is handed a finding
and nothing at all. Silence is not more honest than a weaker remedy correctly labelled.

**Treat a prompt clause as a declared rule once it is added.** This is what an operator will
assume by default, and it is what the gap report exists to contradict. Adopting it would make
`infer_policy` pointless and would let a run report a rule as enforced on the strength of a
sentence nothing checks.

## Consequences

A remediation is a typed thing rather than a string: it names the kind of change it is, and the
renderer cannot print one without the phrasing that kind carries. Making the two indistinguishable
requires editing the type rather than forgetting a caveat.

Some findings get a weaker remedy than the reader wants, stated plainly, and the report says what
the declaration would need to be able to express for a real fix to exist. That is a finding about
the platform rather than about the agent, and it is worth surfacing rather than smoothing over.

The gap between prose and policy becomes a first-class part of the output rather than a footnote,
because this decision is what makes it mean anything.
