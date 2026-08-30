# ADR-0004: The agent spec format

Status: accepted, 2026-08-29.

## Context

agent-red derives its attack suite from the agent under test rather than from a fixed list
of commerce attacks. That only works if there is something to read. The spec format is
that something: it is what every detector consults, what the generator crosses its
techniques against, and what a patch is expressed in. If the format is wrong, nothing
above it can be right.

It is also a proposal. A merchant agent platform may not attach declared limits to agents
today, in which case this file argues for the smallest declaration an agent needs to carry
in order for testing to become possible at all.

## Decision

Two versioned objects, plus one unversioned set of fixtures, paired at load time into an
`AgentSpec` that is validated against itself.

**Config** is capability. Instructions, tool declarations carrying a JSON schema and a
`consequence` of `money | obligation | disclosure | inert`, and the data sources the agent
can reach.

**Policy** is authorisation. Named `bounds` on tool arguments (a constant ceiling or floor,
a closed set of permitted values, or a limit read from a field of an earlier tool result),
`preconditions` mapping a consequential tool to a tool that must have succeeded earlier in
the same conversation, and a `data_scope` naming the sources one session may read and the
identifier kinds that bind a record to the session subject.

A precondition may also declare **what counts as having succeeded**, as a field path into
the required tool's result and the value it must hold. Without that, "succeeded" can only
mean "did not report an error", and a gating tool that answers rather than fails, which is
how verification tools are usually written, satisfies its own precondition by returning a
refusal. An agent that checked nobody and an agent that checked somebody, was refused, and
proceeded anyway would then be indistinguishable, and the second is the more serious. The
weaker reading remains the default for a policy that declares nothing, and every verdict
states which of the two was applied.

**Subjects** are who the harness may act as. An agent that reads records cannot be attacked in
the abstract: a conversation has to be about somebody, and the identifiers it opens with have to
resolve. Without them the harness improvises a reference, the agent correctly declines to act on
a record it cannot find, and the action under test is never reached. Every rule then reports as
never evaluated, which is honest for one conversation and worthless in aggregate, because it is
indistinguishable at a glance from an agent that held. So a policy declaring
`subject_identifier_kinds` requires subjects supplying every one of those kinds, and both the
absence and an incomplete entry are refused at load.

They sit in a third file rather than inside the policy, and are deliberately not versioned with
it. The policy version is one quarter of the tuple a scorecard is valid for, so folding fixtures
into policy would mean that adding somebody to impersonate silently invalidates every scorecard
already produced for an agent whose rules did not change. There is no way to synthesise a
subject: an invented identifier is the failure this object exists to prevent.

Five consequences of that split are load bearing.

**Capability and authorisation are separate objects.** Every system that grants power to an
actor separates the two, and here the separation decides what remedies exist. A structured
bound can be tightened, which makes a violation unreachable. A limit written as English
inside a system prompt can only be reworded, which makes a violation less likely. Keeping
them in one object would blur a structural fix into a probabilistic one, which is the exact
confusion this project exists to expose. See ADR-0003.

**Every tool declares a consequence.** This is the field that turns a tool list into a set
of stakes. It is required rather than defaulted, because a defaulted `inert` is an
assertion nobody made, and an agent whose refund tool is silently inert gets a clean
scorecard for the wrong reason.

**Every policy statement carries a provenance of `declared` or `inferred`.** Where policy
is prose, limits are extracted from the prompt with a model, and every check derived that
way is labelled `inferred`. Degraded mode is built rather than kept as a contingency, and
uncertainty at the foundation is reported rather than hidden.

**Preconditions are expressed as a required prior tool call, not as a rule about intent.**
Most real commerce failures are not a forbidden action, they are a permitted action taken
without verification. Written this way, the violation is observable in the tool-call log
and belongs to a detector rather than to the judge.

Validation is strict and happens at load. A bound naming a tool that does not exist, a
bound naming an argument that tool does not declare, a precondition naming an undeclared
tool, a data scope naming an unreachable source, and a policy whose `agent_id` does not
match its config are all rejected. Unknown fields are rejected rather than ignored.

`tool_version` is derived from a digest of the declared tools rather than stored as a
field, so a tool set that changes cannot keep an old version string. Together with the
config version, the policy version and the model the agent runs, it forms the tuple a
scorecard is valid for.

## Alternatives considered

**One object.** Simpler to write and to load, and it loses the remedy distinction above.
Rejected.

**Policy as prose only, with everything inferred.** This is what many agents will actually
ship, and it is why degraded mode exists. As the primary format it is unusable: an inferred
bound cannot be tightened, so the loop never closes, and the calibration numbers would be
measuring the extractor rather than the agent.

**Modelling tool parameters instead of storing JSON Schema verbatim.** The only question
the validator asks of a schema is whether an argument exists. Re-modelling would let
agent-red reject a schema the agent platform accepts, which makes the harness the thing
that has to be integrated with. Rejected.

**Putting the spec in `targets/`.** It was written that way first. The spec is the contract
between all three layers: detectors read policy and must never import from `targets/`, and
targets implement the contract rather than owning it. Corrected to a top-level
`spec/` package before any code was written against it.

**Preconditions as free text.** Readable, unenforceable. It would move the most common
class of real violation from a detector to the judge, which trades a certain answer for an
uncertain one. Rejected.

## Consequences

An agent that declares nothing but a prompt and a loose tool list still runs, in degraded
mode, and its scorecard says so. An agent that declares a full policy gets deterministic
detection over most of its failure surface.

Derivation is only as good as declaration. This format is therefore also the ask: if a
platform adopts something like it, adversarial testing becomes a property of the platform
rather than a service someone buys.

A violation that is not expressible as a bound, a precondition or a scope is invisible to
the detectors and falls to the judge. That limit is stated in `docs/ARCHITECTURE.md` rather
than papered over, and it is the main thing to revisit if judged violations turn out to
dominate the results.
