"""The page the person who owns the agent opens, generated from a finished analysis.

The page this replaces opened with how the suite was derived: which stakes the spec produced,
why this slice, how the attacks were crossed. That is an audit trail, and it is the right thing
to show somebody checking whether we are honest. It is not what the owner of the agent opens
the page to find. They already know their policy and their tools. They want to know what broke,
what it cost, and what to change, and they want to see the conversation that did it.

**Order of the page, and it is the argument.**

What it cost, in one line, before anything else. Then the failures as cards, worst first, each
one showing the trick in plain words, the line the attacker said, the agent's reply with the
sentence where it turned, the action that followed, and which rule that broke. Then the rules
that were in their prompt and not in their policy, because that gap is where the failures were.
Then consistency, which no single conversation can show. Then, below all of it, coverage and
sample size and how sure the judge was, which stay in the page and stay out of the way.

**Every card carries the breaking point.** The turn where the agent lost is usually several
turns before the turn where it cost something, and showing only the damage teaches the wrong
lesson with complete accuracy: a reader who sees the payout writes a rule about payouts, when
what would have helped is a rule about what the agent may treat as established four turns
earlier.

Nothing here calls a model. The analysis is paid for once and the page is regenerated as often
as it needs to be.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def escape(value: Any) -> str:
    """Render any value as text safe to place in the page."""
    return html.escape(str(value), quote=True)


GIVEN_AWAY = ("refunded", "reduction")
"""Result fields that mean the operator is out of pocket.

`charged` is deliberately absent. It is what a customer paid, which is revenue, and adding it
to money handed back produces a headline that is the sum of a loss and a gain. The first draft
of this page did exactly that and reported 3,901.90 as money moved, of which most was orders
being placed correctly. A wrong number in the first sentence is worse than no number, because
it is the one thing every reader takes away and the only one most of them check.
"""


def money_out(conversation: dict[str, Any]) -> float:
    """Total what the operator gave away in one conversation.

    Reads amounts out of tool results rather than arguments, so a call the agent adjusted
    counts as what actually happened rather than as what was asked for.

    Args:
        conversation: One conversation record.

    Returns:
        The total handed back or discounted. Zero when nothing was.
    """
    total = 0.0
    for turn in conversation["turns"]:
        for call in turn["calls"]:
            result = call.get("result") or {}
            if not isinstance(result, dict):
                continue
            for key in GIVEN_AWAY:
                value = result.get(key)
                if isinstance(value, int | float):
                    total += float(value)
                    break
    return total


def violations(analysis: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every confirmed failure with the conversation it happened in, worst first.

    Ordered by what a reader should look at first: a failure settled by a detector outranks
    one settled by a judge, because the first carries no error rate, and within each the ones
    the agent was talked furthest into come first.

    Args:
        analysis: The analysis file.

    Returns:
        Pairs of conversation and finding.
    """
    found = [
        (conversation, finding)
        for conversation in analysis["conversations"]
        for finding in conversation["findings"]
        if finding["outcome"] == "violated"
    ]
    found.sort(
        key=lambda pair: (pair[1]["settled_by"] != "detector", -(pair[1].get("confidence") or 0))
    )
    return found


def breaking_point_for(conversation: dict[str, Any], rule: str) -> dict[str, Any] | None:
    """The breaking point recorded for one rule in one conversation, if there is one."""
    for point in conversation.get("breaking_points", ()):
        if point["rule"] == rule:
            return point
    return None


def card(conversation: dict[str, Any], finding: dict[str, Any]) -> str:
    """One failure, as the block a reader forwards to their team.

    Args:
        conversation: The conversation it happened in.
        finding: The confirmed failure.

    Returns:
        The card's markup.
    """
    turns = {turn["index"]: turn for turn in conversation["turns"]}
    point = breaking_point_for(conversation, finding["rule"])
    anchor = finding.get("utterance") or finding.get("evidence") or {}
    damage_turn = anchor.get("turn", 0)

    parts = [
        '<article class="card">',
        '<header class="card-head">',
        f"<h3>{escape(finding['rule'] or finding['kind'])}</h3>",
        f'<p class="claim">{escape(finding["summary"])}</p>',
        '<p class="tags">',
        f'<span class="tag {"detector" if finding["settled_by"] == "detector" else "judge"}">'
        f"settled by {escape(finding['settled_by'])}</span>",
        f'<span class="tag">{escape(conversation["target"])}</span>',
    ]
    if finding.get("provenance") == "inferred":
        parts.append('<span class="tag warn">this rule is in your prompt, not your policy</span>')
    if finding.get("confidence") is not None:
        parts.append(f'<span class="tag">confidence {finding["confidence"]:.2f}</span>')
    parts.append("</p></header>")

    if point is not None:
        lost = turns.get(point["turn"], {})
        parts.append('<section class="turned">')
        parts.append(
            f"<h4>It lost here, turn {point['turn']}"
            + (
                f", {point['turns_earlier']} turn(s) before anything happened"
                if point["turns_earlier"]
                else ", the same turn it acted"
            )
            + "</h4>"
        )
        if lost.get("user"):
            parts.append(f'<p class="said person">{escape(lost["user"])}</p>')
        parts.append(f'<p class="said agent quote">{escape(point["quote"])}</p>')
        if point.get("conceded"):
            parts.append(f'<p class="note">It accepted: {escape(point["conceded"])}</p>')
        parts.append("</section>")

    damage = turns.get(damage_turn, {})
    parts.append('<section class="damage">')
    parts.append(f"<h4>What it did, turn {escape(damage_turn)}</h4>")
    if damage.get("user"):
        parts.append(f'<p class="said person">{escape(damage["user"])}</p>')
    if finding.get("utterance"):
        parts.append(f'<p class="said agent quote">{escape(finding["utterance"]["quote"])}</p>')
        if finding["utterance"].get("source_value"):
            parts.append(
                '<p class="note">What the tool actually returned, which was not for the '
                f"customer: <code>{escape(finding['utterance']['source_value'])}</code></p>"
            )
    elif damage.get("reply"):
        parts.append(f'<p class="said agent">{escape(damage["reply"][:400])}</p>')

    for call in damage.get("calls", ()):
        parts.append(
            f'<p class="call"><code>{escape(call["name"])}'
            f"({escape(json.dumps(call['arguments']))})</code></p>"
        )
    if finding.get("evidence"):
        evidence = finding["evidence"]
        parts.append(
            f'<p class="note">{escape(evidence.get("argument", ""))} was '
            f"<strong>{escape(evidence.get('observed', ''))}</strong>, "
            f"your limit is {escape(evidence.get('limit', ''))}</p>"
        )
    parts.append("</section></article>")
    return "\n".join(parts)


def undeclared_section(analysis: dict[str, Any]) -> str:
    """The rules that are in an agent's prompt and not in its policy."""
    rows = []
    for target, policy in analysis.get("policy", {}).items():
        missing = [
            item
            for group in ("statements", "obligations")
            for item in policy.get(group, ())
            if not item["declared"]
        ]
        if not missing:
            continue
        rows.append(f'<h3>{escape(target)}</h3><ul class="rules">')
        for item in missing:
            text = item.get("statement") or item.get("description") or item["name"]
            checkable = "obligations" if "kind" in item else "statements"
            how = "needs a judge" if checkable == "obligations" else "can be checked automatically"
            rows.append(
                f'<li><span class="quote-inline">{escape(text)}</span>'
                f'<span class="tag">{how}</span></li>'
            )
        rows.append("</ul>")
    if not rows:
        return ""
    return (
        '<section class="block"><h2>Rules you told your agent that your policy does not carry</h2>'
        "<p>Read out of your own instructions and compared against your declared policy. "
        "Nothing was checking these.</p>" + "".join(rows) + "</section>"
    )


def consistency_section(analysis: dict[str, Any]) -> str:
    """Where the same request on the same subject got different answers."""
    consistency = analysis.get("consistency") or {}
    divergences = [d for d in consistency.get("divergences", ()) if d.get("alike") is not False]
    if not consistency.get("groups"):
        return ""
    unexplained = len(divergences)
    body = [
        '<section class="block"><h2>Where it answered the same question two ways</h2>',
        "<p>No single conversation is wrong when an agent refuses. This compares "
        "conversations against each other, on the same customer and the same request. "
        "An agent whose limits are real answers the same every time.</p>",
        f'<p class="stat">{unexplained} unexplained of {consistency["groups"]} comparable '
        f"group(s). {consistency.get('settled', 0)} agreed every time.</p>",
    ]
    for divergence in divergences:
        body.append('<div class="diverge">')
        body.append(f'<p class="claim">{escape(divergence["summary"])}</p>')
        body.append(f'<p class="note">{escape(divergence.get("reasoning", ""))}</p>')
        body.append("</div>")
    if consistency.get("unknown_subject"):
        body.append(
            f'<p class="note">{consistency["unknown_subject"]} conversation(s) could not be '
            "compared because nothing recorded whose they were.</p>"
        )
    body.append("</section>")
    return "".join(body)


def coverage_section(analysis: dict[str, Any]) -> str:
    """Sample size, what was never tested, and how much rests on a judge."""
    tally: defaultdict[str, int] = defaultdict(int)
    for conversation in analysis["conversations"]:
        for finding in conversation["findings"]:
            tally[finding["outcome"]] += 1
            if finding["settled_by"] == "judge":
                tally["by_judge"] += 1
    checks = tally["violated"] + tally["held"] + tally["not_evaluated"]
    in_play = tally["violated"] + tally["held"]
    return (
        '<section class="block quiet"><h2>What this run actually covered</h2>'
        f'<p class="stat">{len(analysis["conversations"])} conversations. {checks} checks run, '
        f"{in_play} of them actually in play.</p>"
        f"<p>A check that was never in play is reported as such and never counted as a pass. "
        f"{tally['not_evaluated']} were never in play. A rule your agent never came near says "
        "nothing about whether it respects it.</p>"
        f"<p>{tally['by_judge']} of {checks} checks were settled by a model reading the "
        "conversation rather than asserted from the tool-call log. Those carry an error rate "
        "that has not yet been measured against human labels, so the confidence shown on each "
        "is displayed and used to decide nothing.</p></section>"
    )


STYLE = """
:root { --ink:#16181d; --muted:#5b6070; --line:#e3e5ea; --bg:#fbfbfc; --card:#fff;
  --bad:#b3261e; --bad-bg:#fdf3f2; --warn:#8a5a00; --warn-bg:#fdf8ee; --ok:#1a6b4a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
.wrap { max-width:53rem; margin:0 auto; padding:2.5rem 1.25rem 5rem; }
h1 { font-size:1.6rem; margin:0 0 .35rem; letter-spacing:-.01em; }
h2 { font-size:1.15rem; margin:0 0 .6rem; letter-spacing:-.01em; }
h3 { font-size:1rem; margin:0 0 .3rem; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
h4 { font-size:.78rem; margin:0 0 .5rem; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); font-weight:600; }
.headline { background:var(--bad-bg); border:1px solid #f0d5d2; border-left:4px solid var(--bad);
  border-radius:10px; padding:1.4rem 1.5rem; margin:0 0 2rem; }
.headline p { margin:0; font-size:1.22rem; line-height:1.45; }
.headline .sub { font-size:.9rem; color:var(--muted); margin-top:.6rem; }
.block { margin:0 0 2.5rem; }
.block.quiet { border-top:1px solid var(--line); padding-top:1.5rem; color:var(--muted);
  font-size:.9rem; }
.block.quiet h2 { color:var(--ink); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:1.1rem 1.25rem; margin:0 0 1rem; }
.card-head { border-bottom:1px solid var(--line); padding-bottom:.7rem; margin-bottom:.9rem; }
.claim { margin:.2rem 0 .5rem; }
.tags { margin:0; display:flex; flex-wrap:wrap; gap:.35rem; }
.tag { font-size:.72rem; padding:.15rem .5rem; border-radius:99px; background:#eef0f4;
  color:var(--muted); white-space:nowrap; }
.tag.detector { background:#e7f2ec; color:var(--ok); }
.tag.judge { background:#eceaf6; color:#4a3f8f; }
.tag.warn { background:var(--warn-bg); color:var(--warn); }
.turned { background:var(--warn-bg); border-radius:8px; padding:.8rem .9rem; margin:0 0 .9rem; }
.damage { padding:0 .1rem; }
.said { margin:.35rem 0; padding-left:.85rem; border-left:3px solid var(--line); }
.said.person { color:var(--muted); font-size:.92rem; }
.said.agent { color:var(--ink); }
.said.quote { border-left-color:var(--bad); font-weight:500; }
.note { font-size:.86rem; color:var(--muted); margin:.45rem 0 0; }
.call { margin:.6rem 0 0; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem;
  background:#f1f2f5; padding:.12rem .35rem; border-radius:4px; overflow-wrap:anywhere; }
.rules { list-style:none; padding:0; margin:0 0 1.2rem; }
.rules li { padding:.6rem .75rem; border:1px solid var(--line); border-radius:8px;
  margin-bottom:.4rem; background:var(--card); display:flex; justify-content:space-between;
  gap:1rem; align-items:flex-start; flex-wrap:wrap; }
.quote-inline { font-style:italic; }
.diverge { border:1px solid var(--line); border-radius:8px; padding:.75rem .9rem;
  margin-bottom:.5rem; background:var(--card); }
.stat { font-size:1rem; font-weight:600; margin:.3rem 0 .8rem; }
"""


def build(analysis: dict[str, Any]) -> str:
    """Render the whole page.

    Args:
        analysis: A finished analysis.

    Returns:
        The HTML.
    """
    found = violations(analysis)
    spent = sum(money_out(conversation) for conversation in analysis["conversations"])
    disclosures = sum(1 for _, f in found if f["kind"].startswith("obligation:disclosure"))
    unchecked = any(f.get("provenance") == "inferred" for _, f in found)
    caveat = (
        " Some of these rules are in your prompt and not in your policy, so nothing was "
        "checking them."
        if unchecked
        else ""
    )

    headline = [
        f"Your agent broke <strong>{len(found)}</strong> of your own rules across "
        f"<strong>{len(analysis['conversations'])}</strong> conversations."
    ]
    if spent:
        headline.append(f"It gave away <strong>{spent:,.2f}</strong> in refunds and discounts.")
    if disclosures:
        headline.append(
            f"It repeated staff-only notes to customers <strong>{disclosures}</strong> times."
        )

    cards = "\n".join(card(conversation, finding) for conversation, finding in found)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>agent-red: what broke</title><style>{STYLE}</style></head>
<body><div class="wrap">
<h1>What broke, and where</h1>
<div class="headline"><p>{" ".join(headline)}</p>
<p class="sub">Every one of these is a rule you wrote. Nothing here is our opinion about how an
agent ought to behave.{caveat}</p></div>
<section class="block"><h2>The failures</h2>{cards or "<p>No rule was broken.</p>"}</section>
{undeclared_section(analysis)}
{consistency_section(analysis)}
{coverage_section(analysis)}
</div></body></html>"""


def main(argv: list[str] | None = None) -> None:
    """Generate the page from a finished analysis."""
    parser = argparse.ArgumentParser(description="Build the operator-facing page.")
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    arguments.out.write_text(build(analysis), encoding="utf-8")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
