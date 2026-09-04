"""Turn a finished smoke run into one self-contained HTML page.

Reads `run.json` and `calls.jsonl` from a run directory and writes a single file that opens
with no server and no network. Never hand-edited: the page is regenerated from the run, so
that what it says and what happened cannot drift apart.

The page answers six questions, in the order someone reading it for the first time asks them:
what was run and why, what was actually said, why the attacker said each thing, what the
checks concluded and on what evidence, whether the composed turns hold up against the
hand-written exemplars, and what it cost.

The third of those is the reason this script exists. A transcript shows what was said; it
does not show that the attacker was following an escalation ladder, or that it judged the
agent to be softening, or that it decided to stop. All of that is in its own words in the
recording, and without it a weak-looking run cannot be diagnosed.

    uv run python scripts/report.py out/runs/0001-dispute_handler-full-suite
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentred.attacks.techniques import Technique, load_corpus
from agentred.llm.recording import read_records
from agentred.spec import load_spec_dir
from agentred.spec.models import AgentSpec

REPORT_FILENAME = "report.html"
"""The report lives inside its own run directory.

A run is then one directory holding the transcripts, every model call, and the page that
explains them. Writing the page somewhere central instead would mean the second run silently
overwrites the first one's explanation while both sets of raw material sit there looking
current.
"""

RATES_USD_PER_MTOK = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-5": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_write": 18.75},
}
"""Dollars per million tokens, by model, used to price a run.

**These are list prices and they are not authoritative for your bill.** They do not know about
regional variation, a negotiated rate, credits, or the difference between an on-demand
inference profile and provisioned throughput. The report says which rates it used and shows the
token counts beside the money, so a figure that looks wrong can be recomputed rather than
merely doubted. The bill is the bill.
"""


def price(usage: dict[str, float], model: str) -> float | None:
    """What one set of token counts costs at the named model's rates.

    Args:
        usage: Token counts. Missing keys count as zero.
        model: First-party model id.

    Returns:
        Dollars, or `None` if there is no rate for that model. `None` rather than zero: an
        unpriced call is not a free one, and a report that adds it in as zero understates the
        bill while looking precise.
    """
    rates = RATES_USD_PER_MTOK.get(model)
    if rates is None:
        return None
    return (
        usage.get("input_tokens", 0) * rates["input"]
        + usage.get("output_tokens", 0) * rates["output"]
        + usage.get("cache_read_tokens", 0) * rates["cache_read"]
        + usage.get("cache_write_tokens", 0) * rates["cache_write"]
    ) / 1_000_000


OUTCOME_LABELS = {
    "violated": ("broke", "bad"),
    "held": ("held", "good"),
    "not_evaluated": ("never in play", "idle"),
}
"""How each check outcome is shown, and which colour it carries.

`not_evaluated` is styled as neither good nor bad on purpose. A rule that was never exercised
says nothing about the agent, and rendering it as a pass is the exact mistake the three-outcome
model exists to prevent.
"""


@dataclass
class Composed:
    """One attacker turn, joined back to the call that produced it.

    Attributes:
        turn: What was said, as the model returned it.
        reason: The attacker's own account of why, from the same response.
        stop: Whether this call decided to stop rather than to speak.
        seconds: How long the call took.
        retries: How many times it was throttled first.
        tokens: Input and output tokens.
        system: The full system prompt the turn was composed under.
        state: The rendered conversation state the attacker was shown.
    """

    turn: str = ""
    reason: str = ""
    stop: bool = False
    seconds: float = 0.0
    retries: int = 0
    tokens: tuple[int, int] = (0, 0)
    system: str = ""
    state: str = ""


def compositions(records: tuple[dict[str, Any], ...]) -> dict[str, list[Composed]]:
    """Group every recorded model call by the attack that made it.

    Args:
        records: Everything the recorder wrote, in sequence.

    Returns:
        Attack id to its calls, in the order they were made. A failed call is kept, with its
        error in `reason`, because a conversation that ended early because the composer threw
        must not read as a conversation the agent survived.
    """
    grouped: dict[str, list[Composed]] = defaultdict(list)
    for record in records:
        usage = record.get("usage") or {}
        composed = Composed(
            seconds=record.get("seconds", 0.0),
            retries=record.get("retries", 0),
            tokens=(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
            system=record.get("system", ""),
            state=_last_user_message(record.get("messages") or []),
        )
        if not record.get("ok", False):
            composed.reason = f"the call failed: {record.get('error', '')}"
            composed.stop = True
        else:
            parsed = _parse_turn(record.get("text", ""))
            composed.turn = parsed.get("turn", "")
            composed.reason = parsed.get("reason", "")
            composed.stop = bool(parsed.get("stop", False))
        grouped[record.get("label", "")].append(composed)
    return dict(grouped)


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    """The state block the attacker was shown, which is the only message it receives."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _parse_turn(text: str) -> dict[str, Any]:
    """Read a composed turn out of a response, tolerating a model that wrapped it.

    Args:
        text: The raw response text.

    Returns:
        The parsed object, or an empty one. Tolerant rather than strict on purpose: this is
        a reporting path, and a response the report cannot parse must still be shown rather
        than crash the page that would have explained it.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"turn": text, "reason": "(the response was not valid JSON)"}


def esc(value: object) -> str:
    """HTML-escape any value for interpolation into the page."""
    return html.escape(str(value))


def pretty(value: object) -> str:
    """Render a tool argument or result as compact, readable JSON."""
    return json.dumps(value, indent=2, default=str)


def render_stake_context(spec: AgentSpec, stake_id: str) -> str:
    """Explain, in the merchant's own terms, what this stake is and what breaking it costs.

    Args:
        spec: The target's loaded spec.
        stake_id: The stake the run was filtered to.

    Returns:
        HTML for the opening section. Built from the spec rather than written here, so the
        page cannot describe a rule the agent does not actually carry.
    """
    from agentred.attacks.stakes import derive_stakes

    stake = next((s for s in derive_stakes(spec) if s.id == stake_id), None)
    if stake is None:
        return "<p>The whole derived suite was run.</p>"

    tool = next((t for t in spec.config.tools if t.name == stake.tool), None)
    rows = [
        ("Stake id", f"<code>{esc(stake.id)}</code>"),
        ("Shape of failure", esc(stake.kind)),
        ("Action at risk", f"<code>{esc(stake.tool)}</code>"),
        ("What it costs the merchant", esc(stake.consequence)),
        ("Who settles it", esc(stake.settled_by)),
        ("Where the rule came from", esc(stake.provenance)),
        ("What the attacker was told to achieve", esc(stake.objective)),
    ]
    if tool is not None:
        rows.append(("What the action does", esc(tool.description)))
    body = "".join(f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in rows)
    return f"<table class='kv'>{body}</table>"


def render_tool_call(call: dict[str, Any]) -> str:
    """One tool call, with its arguments and its result both visible.

    Both halves are shown because the detectors read both, and a reader checking a verdict
    needs to see the same thing the detector saw.
    """
    return f"""
    <div class="call">
      <div class="call-name">{esc(call["name"])}</div>
      <div class="call-cols">
        <div><h5>arguments</h5><pre>{esc(pretty(call["arguments"]))}</pre></div>
        <div><h5>result</h5><pre>{esc(pretty(call["result"]))}</pre></div>
      </div>
    </div>"""


def render_turn(turn: dict[str, Any], composed: Composed | None) -> str:
    """One exchange: why it was said, what was said, what the agent did, what it answered."""
    thinking = ""
    if composed is not None and composed.reason:
        meta = (
            f"{composed.seconds}s &middot; {composed.tokens[0]} in / {composed.tokens[1]} out"
            + (f" &middot; {composed.retries} retry" if composed.retries else "")
        )
        thinking = f"""
        <div class="why">
          <div class="why-label">why the attacker said this <span class="meta">{meta}</span></div>
          <div class="why-text">{esc(composed.reason)}</div>
        </div>"""

    calls = "".join(render_tool_call(call) for call in turn["tool_calls"])
    calls_block = (
        f"<div class='calls'><div class='calls-label'>what the agent did</div>{calls}</div>"
        if calls
        else "<div class='calls none'>the agent called no tools on this turn</div>"
    )

    return f"""
    <div class="turn">
      <div class="turn-index">turn {turn["index"] + 1}</div>
      {thinking}
      <div class="bubble attacker"><div class="who">attacker</div>{esc(turn["user"])}</div>
      {calls_block}
      <div class="bubble agent">
        <div class="who">agent <span class="meta">{turn["latency_seconds"]:.1f}s</span></div>
        {esc(turn["reply"])}
      </div>
    </div>"""


def render_persona(subject: dict[str, Any] | None) -> str:
    """Who the attacker was told it was, and what it was allowed to know.

    Shown on every conversation because it is the difference between a conversation that
    could reach the action under test and one that could not. An attacker with no true
    identifier improvises one, the agent declines to act on a record it cannot find, and
    every rule reports as never in play while the run reads as a clean sheet.

    Args:
        subject: The identity, or `None` for an agent that scopes nothing.

    Returns:
        HTML naming the identity, its identifiers and the facts it was given.
    """
    if not subject:
        return "<em>no identity; this agent scopes nothing to a subject</em>"
    identifiers = ", ".join(
        f"<code>{esc(kind)}={esc(value)}</code>"
        for kind, value in sorted(subject["identifiers"].items())
    )
    facts = "".join(f"<li>{esc(fact)}</li>" for fact in subject.get("facts", []))
    return (
        f"<strong>{esc(subject['name'])}</strong> &middot; {identifiers}"
        f"<ul class='exemplars'>{facts}</ul>"
    )


def render_finding(finding: dict[str, Any]) -> str:
    """One check, its outcome, and the evidence that anchors it."""
    label, tone = OUTCOME_LABELS.get(finding["outcome"], (finding["outcome"], "idle"))
    evidence = finding.get("evidence")
    evidence_block = ""
    if evidence:
        rows = [
            ("turn", evidence["turn"] + 1),
            ("call", f"#{evidence['call']} to {evidence['tool']}"),
        ]
        if evidence.get("argument"):
            rows.append(("argument", evidence["argument"]))
        if evidence.get("observed"):
            rows.append(("what was passed", evidence["observed"]))
        if evidence.get("limit"):
            rows.append(("what was permitted", evidence["limit"]))
        cells = "".join(f"<tr><th>{esc(n)}</th><td>{esc(v)}</td></tr>" for n, v in rows)
        evidence_block = f"<table class='kv small'>{cells}</table>"

    rule = (
        f"<code>{esc(finding['rule'])}</code>" if finding["rule"] else "<em>no declared rule</em>"
    )
    return f"""
    <div class="finding {tone}">
      <div class="finding-head">
        <span class="badge {tone}">{esc(label)}</span>
        <span class="finding-kind">{esc(finding["kind"])}</span>
        <span class="finding-rule">{rule}</span>
        <span class="finding-settled">settled by {esc(finding["settled_by"])}</span>
      </div>
      <p>{esc(finding["summary"])}</p>
      {evidence_block}
    </div>"""


def render_conversation(outcome: dict[str, Any], composed: list[Composed], index: int) -> str:
    """One whole attack: its technique, its conversation, and its verdicts."""
    if outcome["transcript"] is None:
        return f"""
        <section class="conversation failed" id="c{index}">
          <h3>{esc(outcome["technique"])}</h3>
          <p class="error">This conversation did not complete: {esc(outcome["error"])}</p>
          <p class="note">Kept as a row rather than dropped. An attack that failed and an
          attack that was never run are different results, and a report showing only the
          successes quietly shrinks its own denominator.</p>
        </section>"""

    transcript = outcome["transcript"]
    persona = render_persona(outcome.get("subject"))
    turns = "".join(
        render_turn(turn, composed[i] if i < len(composed) else None)
        for i, turn in enumerate(transcript["turns"])
    )
    findings = "".join(render_finding(f) for f in outcome["findings"]) or (
        "<p class='note'>No declared rule was in play for this conversation.</p>"
    )
    broke = sum(1 for f in outcome["findings"] if f["outcome"] == "violated")
    evaluated = sum(1 for f in outcome["findings"] if f["outcome"] != "not_evaluated")
    if broke:
        verdict = f"<span class='badge bad'>broke {broke} rule{'s' if broke != 1 else ''}</span>"
    elif evaluated:
        verdict = f"<span class='badge good'>{evaluated} rule(s) held</span>"
    else:
        verdict = "<span class='badge idle'>nothing was tested</span>"

    stop_reason = transcript["stopped_because"] or "the turn budget was spent"
    stopped = next((c for c in composed if c.stop and c.reason), None)
    stop_detail = (
        f"<p class='note'>The attacker's own account of stopping: "
        f"&ldquo;{esc(stopped.reason)}&rdquo;</p>"
        if stopped
        else ""
    )

    return f"""
    <section class="conversation" id="c{index}">
      <div class="conversation-head">
        <h3>{esc(outcome["technique"])}</h3>
        {verdict}
      </div>
      <table class="kv small">
        <tr><th>attack id</th><td><code>{esc(outcome["attack_id"])}</code></td></tr>
        <tr><th>who the attacker was</th><td>{persona}</td></tr>
        <tr><th>goal handed to the attacker</th><td>{esc(outcome["goal"])}</td></tr>
        <tr><th>session</th><td><code>{esc(transcript["session"])}</code></td></tr>
        <tr><th>agent version tested</th>
            <td><code>{esc(transcript["spec_versions"])}</code></td></tr>
        <tr><th>ended because</th><td>{esc(stop_reason)}</td></tr>
        <tr><th>wall clock</th><td>{outcome["seconds"]}s</td></tr>
      </table>
      {stop_detail}
      <h4>What the checks concluded</h4>
      {findings}
      <h4>The conversation</h4>
      {turns}
    </section>"""


def render_exemplars(
    techniques: dict[str, Technique], run: dict[str, Any], composed: dict[str, list[Composed]]
) -> str:
    """The hand-written bar, beside what the model actually wrote.

    This is the section the run exists for. Everything else describes what happened; this one
    is where a person decides whether the attacker is worth believing, by reading the two
    side by side rather than by being told a number.
    """
    blocks = []
    for outcome in run["outcomes"]:
        technique = techniques.get(outcome["technique_id"])
        calls = composed.get(outcome["attack_id"], [])
        opening = next((c.turn for c in calls if c.turn), "")
        if technique is None or not opening:
            continue
        written = "".join(f"<li>{esc(line)}</li>" for line in technique.exemplars)
        blocks.append(f"""
        <div class="compare">
          <h4>{esc(technique.name)}</h4>
          <p class="premise">{esc(technique.premise)}</p>
          <div class="compare-cols">
            <div>
              <h5>hand-written exemplars, the bar</h5>
              <ul class="exemplars">{written}</ul>
            </div>
            <div>
              <h5>what the model composed, unedited</h5>
              <blockquote>{esc(opening)}</blockquote>
            </div>
          </div>
        </div>""")
    return "".join(blocks)


def target_model(run: dict[str, Any], spec: AgentSpec) -> str:
    """Which model the agent under test ran on, for pricing its half of the bill."""
    return spec.config.model


def render_cost(run: dict[str, Any], records: tuple[dict[str, Any], ...], spec: AgentSpec) -> str:
    """Both halves of the bill: what the attacker spent, and what the agent spent answering.

    Reported separately because they are different models on different rates, and because a
    reader deciding whether to run four hundred conversations needs to know which half scales.
    """
    attacker_model = run["model"]
    agent_model = target_model(run, spec)

    by_attack: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "retries": 0,
        }
    )
    for record in records:
        usage = record.get("usage") or {}
        row = by_attack[record.get("label", "")]
        row["calls"] += 1
        row["retries"] += record.get("retries", 0)
        for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
            row[key] += usage.get(key, 0)

    rows = []
    total_attacker = 0.0
    total_agent: float | None = 0.0
    total_reported: float | None = 0.0
    for outcome in run["outcomes"]:
        attacker = by_attack.get(outcome["attack_id"], {})
        attacker_cost = price(attacker, attacker_model) or 0.0
        total_attacker += attacker_cost

        agent: dict[str, float] = defaultdict(float)
        reported: float | None = None
        transcript = outcome["transcript"]
        for turn in (transcript or {}).get("turns", []):
            for key, value in (turn.get("agent_usage") or {}).items():
                agent[key] += value
            if "cost_usd" in (turn.get("agent_usage") or {}):
                reported = (reported or 0.0) + turn["agent_usage"]["cost_usd"]
        agent_cost = price(agent, agent_model)
        if agent_cost is None or not agent:
            total_agent = None
        elif total_agent is not None:
            total_agent += agent_cost
        if reported is None:
            total_reported = None
        elif total_reported is not None:
            total_reported += reported

        rows.append(
            f"<tr><td>{esc(outcome['technique'])}</td>"
            f"<td>{int(attacker.get('calls', 0))}</td>"
            f"<td>{int(attacker.get('input_tokens', 0)):,}</td>"
            f"<td>{int(attacker.get('output_tokens', 0)):,}</td>"
            f"<td>${attacker_cost:.4f}</td>"
            f"<td>{int(agent.get('input_tokens', 0)):,}</td>"
            f"<td>{int(agent.get('output_tokens', 0)):,}</td>"
            f"<td>{'-' if agent_cost is None else f'${agent_cost:.4f}'}</td>"
            f"<td>{int(attacker.get('retries', 0))}</td>"
            f"<td>{outcome['seconds']}s</td></tr>"
        )

    # The SDK priced the calls it made. This repository's rate table only guesses at them, so
    # the headline uses the SDK's figure wherever it exists and falls back otherwise.
    agent_total = total_reported if total_reported else total_agent
    agent_source = "as the SDK priced it" if total_reported else "from the rate table below"
    combined = (
        f"${total_attacker + agent_total:.4f}" if agent_total is not None else "not fully priced"
    )
    reported_line = (
        f"<p class='note'>The agent's half is priced by the Claude Agent SDK itself, which is "
        f"the figure used above. This repository's rate table applied to the same token counts "
        f"gives ${total_agent:.4f} instead. The SDK's is the one to trust: it comes from the "
        f"thing that made the calls. The gap is a reasonable measure of how far the table "
        f"below can be out.</p>"
        if total_reported
        else "<p class='note'>The Claude Agent SDK did not price the agent's half of this run, "
        "which it does not do on every route. The figure above is this repository's rate table "
        "applied to the token counts the SDK did report.</p>"
    )

    return f"""
    <table class="grid">
      <thead>
        <tr><th rowspan="2">technique</th><th colspan="4">attacker ({esc(attacker_model)})</th>
        <th colspan="3">agent under test ({esc(agent_model)})</th>
        <th rowspan="2">retries</th><th rowspan="2">wall clock</th></tr>
        <tr><th>calls</th><th>tokens in</th><th>tokens out</th><th>cost</th>
        <th>tokens in</th><th>tokens out</th><th>cost</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <table class="kv">
      <tr><th>Attacker side</th><td>${total_attacker:.4f}</td></tr>
      <tr><th>Agent under test</th>
          <td>{"not fully priced" if agent_total is None else f"${agent_total:.4f}"}
          <span class="meta">{agent_source}</span></td></tr>
      <tr><th>This run, in total</th><td><strong>{combined}</strong></td></tr>
      <tr><th>Wall clock</th>
          <td>{run["seconds"]}s at {run["concurrency"]} conversations at once</td></tr>
    </table>
    {reported_line}
    <p class="note"><strong>These are list prices, not your bill.</strong> They are applied from
    a rate table in <code>scripts/report.py</code> and know nothing about regional variation, a
    negotiated rate, credits, or provisioned throughput. Token counts are shown beside every
    figure so a number that looks wrong can be recomputed rather than merely doubted.</p>"""


STYLE = """
:root {
  --ink: #14161a; --dim: #5d6570; --line: #e2e5ea; --bg: #fbfbfc; --card: #ffffff;
  --bad: #b0203a; --bad-bg: #fdf1f3; --good: #1c6b45; --good-bg: #f0f8f3;
  --idle: #6a7280; --idle-bg: #f3f4f6; --accent: #2b4a8f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.65 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif; }
.wrap { max-width: 62rem; margin: 0 auto; padding: 3rem 1.5rem 6rem; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 .5rem; letter-spacing: -.02em; }
h2 { font-size: 1.3rem; margin: 3.5rem 0 1rem; padding-bottom: .5rem;
  border-bottom: 2px solid var(--ink); letter-spacing: -.01em; }
h3 { font-size: 1.1rem; margin: 0; }
h4 { font-size: .95rem; margin: 2rem 0 .75rem; color: var(--dim);
  text-transform: uppercase; letter-spacing: .08em; }
h5 { font-size: .75rem; margin: 0 0 .35rem; color: var(--dim);
  text-transform: uppercase; letter-spacing: .08em; }
.sub { color: var(--dim); margin: 0 0 2rem; }
.note { color: var(--dim); font-size: .9rem; }
code { font: .85em ui-monospace, "SF Mono", Menlo, monospace;
  background: #eef0f3; padding: .1em .35em; border-radius: 3px; }
pre { font: .78rem/1.5 ui-monospace, "SF Mono", Menlo, monospace; background: #f6f7f9;
  border: 1px solid var(--line); border-radius: 6px; padding: .6rem .7rem; margin: 0;
  overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
table { border-collapse: collapse; width: 100%; }
.kv th { text-align: left; width: 16rem; padding: .45rem .8rem .45rem 0; color: var(--dim);
  font-weight: 500; vertical-align: top; border-bottom: 1px solid var(--line); }
.kv td { padding: .45rem 0; border-bottom: 1px solid var(--line); vertical-align: top; }
.kv.small th, .kv.small td { font-size: .88rem; padding: .3rem .8rem .3rem 0; }
.grid { font-size: .9rem; margin-top: 1rem; }
.grid th, .grid td { text-align: right; padding: .45rem .6rem;
  border-bottom: 1px solid var(--line); }
.grid th:first-child, .grid td:first-child { text-align: left; }
.grid thead th { color: var(--dim); font-weight: 500; font-size: .8rem;
  text-transform: uppercase; letter-spacing: .05em; }
.grid tfoot th { border-top: 2px solid var(--ink); border-bottom: none; }
.callout { background: var(--card); border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1.5rem 0; }
.callout p:first-child { margin-top: 0; } .callout p:last-child { margin-bottom: 0; }
.badge { display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; padding: .2rem .5rem; border-radius: 4px; white-space: nowrap; }
.badge.bad { background: var(--bad-bg); color: var(--bad); }
.badge.good { background: var(--good-bg); color: var(--good); }
.badge.idle { background: var(--idle-bg); color: var(--idle); }
.conversation { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.5rem 1.75rem; margin: 1.5rem 0; }
.conversation.failed { border-left: 3px solid var(--bad); }
.conversation-head { display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; margin-bottom: 1rem; }
.error { color: var(--bad); font-weight: 500; }
.finding { border: 1px solid var(--line); border-radius: 8px;
  padding: .8rem 1rem; margin: .6rem 0; }
.finding.bad { border-color: #f0c9d1; background: var(--bad-bg); }
.finding.good { background: var(--good-bg); border-color: #cfe6da; }
.finding.idle { background: var(--idle-bg); }
.finding p { margin: .5rem 0; }
.finding-head { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center;
  font-size: .85rem; }
.finding-kind { font-weight: 600; }
.finding-rule, .finding-settled { color: var(--dim); }
.turn { border-top: 1px solid var(--line); padding-top: 1.25rem; margin-top: 1.25rem; }
.turn-index { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em;
  color: var(--dim); margin-bottom: .6rem; }
.why { background: #fffaf0; border: 1px dashed #e3cfa3; border-radius: 8px;
  padding: .6rem .85rem; margin-bottom: .7rem; }
.why-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
  color: #96702a; margin-bottom: .25rem; }
.why-text { font-size: .9rem; font-style: italic; }
.meta { color: var(--dim); font-style: normal; font-weight: 400; text-transform: none;
  letter-spacing: 0; font-size: .8em; }
.bubble { border-radius: 10px; padding: .75rem 1rem; margin: .5rem 0; white-space: pre-wrap;
  word-break: break-word; }
.bubble.attacker { background: #f2f4f8; border: 1px solid #dde2ea; }
.bubble.agent { background: var(--card); border: 1px solid var(--line); }
.who { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--dim); margin-bottom: .35rem; }
.calls { margin: .5rem 0 .5rem 1rem; padding-left: 1rem; border-left: 3px solid #d8cbe8; }
.calls.none { color: var(--dim); font-size: .85rem; font-style: italic;
  margin: .4rem 0 .4rem 1rem; padding-left: 1rem; border-left: 3px solid var(--line); }
.calls-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
  color: #6a4b95; margin-bottom: .4rem; }
.call { margin-bottom: .6rem; }
.call-name { font: .82rem ui-monospace, Menlo, monospace; font-weight: 600; margin-bottom: .3rem; }
.call-cols, .compare-cols { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr);
  gap: .75rem; }
.compare { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 1.25rem 1.5rem; margin: 1rem 0; }
.compare h4 { margin-top: 0; color: var(--ink); text-transform: none; letter-spacing: 0;
  font-size: 1.05rem; }
.premise { color: var(--dim); font-size: .9rem; margin-top: 0; }
.exemplars { margin: 0; padding-left: 1.1rem; font-size: .88rem; color: var(--dim); }
.exemplars li { margin-bottom: .5rem; }
blockquote { margin: 0; padding: .7rem .9rem; background: #f2f4f8;
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0; font-size: .9rem; white-space: pre-wrap; }
@media (max-width: 45rem) { .call-cols, .compare-cols { grid-template-columns: minmax(0,1fr); } }
"""


def build(run: dict[str, Any], records: tuple[dict[str, Any], ...], spec: AgentSpec) -> str:
    """Assemble the whole page.

    Args:
        run: The parsed `run.json`.
        records: The parsed `calls.jsonl`.
        spec: The target's spec, for explaining the stake in the merchant's terms.

    Returns:
        A complete HTML document.
    """
    composed = compositions(records)
    techniques = {t.id: t for t in load_corpus()}
    number = run.get("number", "")
    title = f"Run {number}: {run['target']}" if number else f"Smoke run: {run['target']}"

    ran = len(run["outcomes"])
    completed = [o for o in run["outcomes"] if o["transcript"] is not None]
    broke = [o for o in completed if any(f["outcome"] == "violated" for f in o["findings"])]
    checks = [f for o in completed for f in o["findings"]]
    evaluated = [f for f in checks if f["outcome"] != "not_evaluated"]

    conversations = "".join(
        render_conversation(outcome, composed.get(outcome["attack_id"], []), i)
        for i, outcome in enumerate(run["outcomes"])
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{STYLE}</style></head><body><div class="wrap">

<h1>{esc(title)}</h1>
<p class="sub">{esc(run["started_at"])} &middot; attacker {esc(run["model"])} &middot;
{ran} conversation{"s" if ran != 1 else ""}, {run["concurrency"]} at a time &middot;
{run["seconds"]}s wall clock</p>

<div class="callout">
  <p><strong>This is not a measurement.</strong> Nothing here is scored or reported. It
  exists to settle one question before the real run: does a model,
  handed a persuasion technique and an objective, write turns as convincing as the
  hand-written examples we set as the bar?</p>
  <p>That matters because every number the suite will ever publish depends on it. If the
  attacker writes politely, a low break rate would mean the attacker was weak rather than the
  agent safe, and nothing downstream could tell the two apart.</p>
</div>

<h2>1. What was run, and why this slice</h2>
<p>The suite was not written for this run. It is derived from the agent's own declared
configuration by the same code a full run uses: every technique in the corpus, crossed with
everything the agent's own policy makes worth attacking. This run then filtered that suite
down to a single stake, so that eight conversations answer a question that four hundred would
answer no better.</p>
{render_stake_context(spec, run["stake"])}
<p>Each conversation was handed one of the identities the agent's own spec declares as safe to
impersonate, listed under every conversation below. That is not a convenience. An attacker with
no true reference invents one, the agent correctly declines to act on a record it cannot find,
and the action under test is never reached, so every rule reports as never in play while the run
reads as a clean sheet for an agent nobody managed to question.</p>
<p class="note">Conversations ran {run["concurrency"]} at a time. Each one gets its own copy of
the agent's world under its own session id, so nothing one conversation does can be seen or
undone by another. That isolation is what makes a rate mean anything later.</p>

<h2>2. What happened, at a glance</h2>
<table class="kv">
  <tr><th>Conversations attempted</th><td>{ran}</td></tr>
  <tr><th>Conversations that completed</th><td>{len(completed)}</td></tr>
  <tr><th>Conversations where a rule broke</th><td>{len(broke)}</td></tr>
  <tr><th>Checks run</th><td>{len(checks)}</td></tr>
  <tr><th>Checks that were actually in play</th><td>{len(evaluated)}</td></tr>
</table>
<p class="note">A check that was never in play is reported as such and never counted as a
pass. A rule the conversation never came near says nothing about whether the agent respects
it, and treating silence as compliance is how a suite flatters an agent nobody tested.</p>

<h2>3. The conversations</h2>
<p>Each one shows, per turn: the attacker's own stated reason for saying what it said, the
turn itself, every tool the agent called with the arguments it passed and the result it got
back, and the agent's reply. The reasons come from the attacker's recorded response, not from
a reconstruction.</p>
<p class="note">The attacker never sees the tool calls shown here. It is given only what the
agent said out loud, which is what a real customer would have. Handing it tool results would
let it quote figures nobody said, and any violation found that way would be exposure the
merchant does not actually carry.</p>
{conversations}

<h2>4. The bar, and what the model wrote</h2>
<p>The hand-written exemplars in the technique corpus set the persuasiveness bar. They are
shown here beside the opening the model composed, unedited, so the question can be settled by
reading rather than by assertion.</p>
{render_exemplars(techniques, run, composed)}

<h2>5. What it cost</h2>
{render_cost(run, records, spec)}

<h2>6. Where the raw material is</h2>
<table class="kv">
  <tr><th>Every model call, verbatim</th><td><code>{esc(run["recording"])}</code></td></tr>
  <tr><th>Structured run output</th><td><code>run.json</code>, beside it</td></tr>
  <tr><th>Transcripts in the database</th>
      <td><code>{esc(run["run_id"] or "not persisted")}</code></td></tr>
</table>
<p class="note">The recording holds the full system prompt and state block for every composed
turn, including calls that failed. This page is generated from those files and is never
edited by hand, so it cannot drift from what actually happened.</p>

</div></body></html>"""


def main(argv: list[str] | None = None) -> None:
    """Read a run directory and write the report.

    Args:
        argv: Command line arguments. Defaults to `sys.argv[1:]`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Directory holding run.json and calls.jsonl.")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the HTML.")
    arguments = parser.parse_args(argv)

    run = json.loads((arguments.run_dir / "run.json").read_text(encoding="utf-8"))
    calls = arguments.run_dir / "calls.jsonl"
    records = read_records(calls) if calls.exists() else ()

    from agentred.runner.consent import load_registry

    spec = load_spec_dir(load_registry().resolve(run["target"]).spec_dir)

    out = arguments.out or arguments.run_dir / REPORT_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(run, records, spec), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
