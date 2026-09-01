# ADR-0005: The oracle sits at the tool boundary, not inside the agent

Status: accepted, 2026-09-02. Supersedes the tool-call reporting half of surface 2 in
`docs/ARCHITECTURE.md`.

## Context

Every check agent-red makes reduces to a question about what the agent did: which tool it
called, with which arguments, in which order, and what came back. Until now the answer came
from the agent itself. `targets/runtime.py` recorded each call as it made it and returned the
list in the body of its own HTTP reply, and `runner/conversation.py` read
`body["tool_calls"]` into the transcript the detectors then walked.

That works, and it works for exactly one reason: we wrote the target. The measurement and the
thing being measured are the same process, so the evidence is a self-report.

Two consequences follow, and the second is the one that matters.

**A self-report is not evidence.** A reader who opens `conversation.py` sees that every
finding on the scorecard rests on the target volunteering an accurate account of its own
behaviour. Nothing in the harness can tell a target that omitted a call from one that never
made it. That is a fine assumption for a stand-in and an impossible one for the product,
where the agent under test is the untrusted party by definition.

**A self-report is a framework dependency in disguise.** Requiring the reply body to carry a
tool-call log means requiring every agent runtime to be modified to produce one. An agent
built on a workflow engine, an agent that runs on a cron with no HTTP reply at all, an agent
whose framework we have never seen: each needs its own integration before it can be measured.
The harness would work on whatever it was written against and nothing else, which is the
definition of a consulting engagement rather than a component.

The observation that resolves both: an agent's reply is framework-specific, but an agent's
*hands* are not. Whatever it is built on, it reaches the merchant's money through tools, and
in the platform this is shaped around those tools are MCP connectors. That boundary exists in
every agent worth attacking, and it is on our side of the trust line.

## Decision

**Tool calls are recorded at the MCP server, by the MCP server, and the agent is never asked
what it did.**

`mcp/server.py` serves the tool surface over the synthetic world. Every invocation is written
to an append-only stream keyed by run and session: name, full arguments, return value,
timestamp. `Transcript.tool_calls` is populated from that stream after the turn completes,
not from the reply body.

Three things follow that are worth stating as consequences rather than leaving implicit.

**The agent's reply carries prose and nothing else.** `ChatResponse` keeps `reply` and
`spec_versions`; `tool_calls` is removed from the contract. A target that wants to lie about
its behaviour now has to lie to the tool server, which means not calling the tool, which
means not having the effect.

**Arguments are recorded in full, always.** A signature is not enough. A leak can happen
inside a call that is correct in every visible respect: an outbound message to the right
number carrying another customer's order details in the body. The scope detector already
walks argument payloads for identifiers, and this decision is what guarantees it is walking
the real ones.

**The world diff is available beside the call stream.** The server holds the world, so it can
say what the world looked like before a run and after it. Not every check needs it, and the
ones that do (cumulative spend against a single order, a second money movement from a
replayed trigger) are not expressible over a single call in isolation.

## Alternatives considered

**Keep the self-report and document the assumption.** This is what surface 2 of the
architecture note did, honestly, and it was defensible while the targets were the only agents
in existence. It stops being defensible the moment the pitch is "point it at an agent
somebody else built".

**Read the framework's trace.** Agno emits step traces; the Claude Agent SDK emits message
blocks. Both are richer than a call log and both are wrong to depend on, because depending on
either makes the harness work on that framework only. The generality is the product. See
ADR-0006 for the same argument applied to how attacks arrive.

**Proxy the model API instead.** Sitting between the agent and the model catches the tool
calls in the model's output, which is earlier and therefore tempting. It is the wrong seam:
it sees intentions rather than effects, it breaks on any agent that filters or retries a call
after the model proposes it, and it needs a different integration per model provider. The
tool boundary sees what actually happened.

**Assert against the database only, with no call stream.** A world diff says a refund
happened; it does not say which tool issued it, with what arguments, or in what order
relative to the verification step. Preconditions and argument bounds both need the sequence.
The diff is a complement, not a replacement.

## Consequences

The detectors do not change. `judge/detectors/` reads `LoggedCall` tuples and has no opinion
about where they came from, which is a property worth having kept: the entire oracle moved
and the code that decides what a violation is was untouched.

`targets/runtime.py` loses its recorder and its `tool_calls` field. `targets/tools/` moves out
of the target package entirely, because the tools are no longer something a target owns. A
target is now a prompt, a policy and a list of tool names it is permitted to reach.

Consent grows a second surface. The registry entry already names the agent; it now also names
the MCP server the run will record at, and a target that reaches a tool server the run is not
recording is a configuration error that fails at preflight rather than a run that quietly
measures nothing.

The claim this buys, stated the way it should be said out loud: *a violation agent-red reports
was observed at the boundary the agent reaches the world through, not described by the agent.*
