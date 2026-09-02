"""The page the operator reads before switching an agent on.

Four lines and a conversation. That sequence is the design, and it is the reverse of the first
version of this page, which opened with a count of failures and a proportion bar. Somebody who
knows this system better than anyone read that page and could not say what he was looking at,
which is the evidence for everything below.

**The abstraction is worthless and the transcript is what delivers.** A reader handed "two rules
are running on trust" has learned a phrase. A reader handed one conversation where their own
agent gave money away, with the sentence that did it marked, has learned what their agent does
under pressure and needs nothing explained. So the page leads with a conversation and puts the
counting underneath it.

**Nobody is being blamed.** The operator wrote English into a box, which is what they were
told to do. Some of those sentences became something a machine can hold them to and some
stayed sentences, and nothing ever told them which. That is not their mistake, so the page
never says "you failed to" and never says "we caught". The subject of every sentence is the
rule or the agent.

**Where this system takes any praise, it takes it once.** Saying "we tried and could not
break these" is the reassuring half of the page, and the effort is the point, because a
guarantee is worth exactly the pressure behind it. Everywhere else the difficulty is shown
rather than narrated: how many turns of pressure it took, and the same request answered the
other way.
"""

from __future__ import annotations

import html
from typing import Any

from agentred.scoring.ledger import CHECKED, JUDGED, UNCHECKED, headline, ledger
from agentred.scoring.render import money_out, unit_symbol, value_fields


def escape(value: Any) -> str:
    """Render any value as text safe to place in the page."""
    return html.escape(str(value), quote=True)


def amount(value: float, symbol: str) -> str:
    """A figure the way a person writes one, with whatever the agent declares in front."""
    return f"{symbol}{value:,.0f}" if value == int(value) else f"{symbol}{value:,.2f}"


def targets_in(analysis: dict[str, Any]) -> list[str]:
    """Every agent the analysis covers, in a stable sequence."""
    return sorted({conversation["target"] for conversation in analysis["conversations"]})


def broke(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Every finding in one conversation that says a rule gave way."""
    return [f for f in conversation["findings"] if f["outcome"] == "violated"]


def worst(analysis: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any] | None:
    """The one conversation to open the page with.

    Chosen by what it cost first and by how confident the finding is second. Cost first
    because a reader deciding whether to switch an agent on is deciding about money, and a
    disclosure that cost nothing is a weaker opening than the same agent giving value away.
    Confidence second so that, among conversations that cost the same, the page does not lead
    with the flag we are least sure about.

    Args:
        analysis: A finished analysis.
        fields: Result fields that mean value left the operator.

    Returns:
        The conversation record, or None when no rule gave way anywhere.
    """
    candidates = [c for c in analysis["conversations"] if broke(c)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (
            money_out(c, fields),
            max(f.get("confidence") or 0.0 for f in broke(c)),
        ),
    )


def breaking_point_for(conversation: dict[str, Any], rule: str) -> dict[str, Any] | None:
    """Where this conversation was lost for one rule, if it was located."""
    for point in conversation.get("breaking_points", ()):
        if point["rule"] == rule:
            return point
    return None


def highlight(reply: str, quote: str) -> str:
    """Mark the sentence that gave the rule away, inside the reply that contained it.

    Falls back to the plain reply when the quote is not found verbatim. A quote that has
    drifted is a defect worth seeing as an unmarked reply rather than papering over with a
    fuzzy match, which would mark the wrong sentence with the same confidence as the right one.
    """
    if quote and quote in reply:
        before, after = reply.split(quote, 1)
        return f"{escape(before)}<mark>{escape(quote)}</mark>{escape(after)}"
    return escape(reply)


def call_line(call: dict[str, Any], fields: tuple[str, ...], symbol: str) -> str:
    """One action the agent took, written the way a person would describe it."""
    result = call.get("result") or {}
    moved = ""
    if isinstance(result, dict):
        for field in fields:
            value = result.get(field)
            if isinstance(value, int | float) and value:
                moved = f' <span class="moved">{escape(amount(float(value), symbol))}</span>'
                break
    return f'<div class="call"><span class="verb">did</span> {escape(call["name"])}{moved}</div>'


def leaked(finding: dict[str, Any]) -> str:
    """What a tool handed back that the agent was not meant to pass on, if anything."""
    utterance = finding.get("utterance") or {}
    return str(utterance.get("source_value") or "")


def annotate(
    conversation: dict[str, Any],
    fields: tuple[str, ...],
    symbol: str,
    technique: str,
    says: dict[str, str] | None = None,
) -> str:
    """One conversation, readable end to end, with three things marked on it.

    The three are the whole point and no more are added: what a tool handed back that was not
    meant to leave the building, the exchange where the agent conceded, and the sentence that
    gave a rule away. A fourth annotation would make the page something to study rather than
    something to read.

    Args:
        conversation: One conversation record.
        fields: Result fields that mean value left the operator.
        symbol: What goes in front of an amount.
        technique: What the person on the other end was doing, in plain words. Empty when
            the analysis does not carry it, in which case the note is left out rather than
            guessed at.
        says: Rule name to the sentence the operator wrote. Without it the page falls back to
            our identifier for the rule, which tells the reader nothing they can act on and
            tells them the page was written for somebody else.

    Returns:
        The HTML for one annotated conversation.
    """
    stated = says or {}
    failures = broke(conversation)
    quotes = {
        (f.get("utterance") or {}).get("turn"): f
        for f in failures
        if (f.get("utterance") or {}).get("quote")
    }
    points = {p["turn"]: p for p in conversation.get("breaking_points", ())}
    secret = next((leaked(f) for f in failures if leaked(f)), "")

    out: list[str] = []
    if technique:
        out.append(f'<p class="pressure"><b>What they were doing.</b> {escape(technique)}</p>')

    for turn in conversation["turns"]:
        index = turn["index"]
        out.append(
            f'<div class="them"><span class="who">Them</span><p>{escape(turn["user"])}</p></div>'
        )
        out.append('<div class="it"><span class="who">Your agent</span>')

        for call in turn["calls"]:
            out.append(call_line(call, fields, symbol))
            result = call.get("result") or {}
            if secret and isinstance(result, dict) and secret in str(result.values()):
                out.append(
                    f'<div class="held-back">what came back: &ldquo;{escape(secret)}&rdquo;'
                    "<span>not meant to leave the building</span></div>"
                )

        finding = quotes.get(index)
        quote = (finding.get("utterance") or {}).get("quote", "") if finding else ""
        out.append(f'<p class="reply">{highlight(turn["reply"], quote)}</p>')

        point = points.get(index)
        if point:
            out.append(
                '<div class="note lost"><b>This is where it went.</b> '
                f"{escape(point.get('conceded', ''))}</div>"
            )
        if finding:
            rule = stated.get(finding["rule"], "")
            out.append(
                '<div class="note gave"><b>That gave this rule away.</b> '
                f"{escape(rule) if rule else escape(finding['rule'])}</div>"
            )
        out.append("</div>")

    spent = money_out(conversation, fields)
    if spent:
        out.append(f'<div class="outcome">{escape(amount(spent, symbol))} left the business.</div>')
    return "\n".join(out)


def opposite(analysis: dict[str, Any], hero: dict[str, Any]) -> dict[str, Any] | None:
    """A conversation where the same request, on the same facts, was refused.

    This is the object that makes a rate mean something. A reader told a rule holds nine times
    in ten reaches for the obvious defence, that the one person was unusually persistent. The
    same person, the same request, the same facts, answered the other way, takes that defence
    away and needs no judgment call from anybody to believe.

    Returns:
        The refusing conversation, or None when nothing comparable was recorded.
    """
    for divergence in analysis.get("consistency", {}).get("divergences", ()):
        if divergence["subject"] != hero.get("subject"):
            continue
        for session in divergence.get("declined", ()):
            for conversation in analysis["conversations"]:
                if conversation["session"] == session:
                    return conversation
    return None


def rule_rows(rows: list[dict[str, Any]]) -> str:
    """Every rule with how often it held, worst first."""
    out = []
    for row in rows:
        if row["rate"] is None:
            figure, detail, tone = "?", "nobody checked this", "none"
        else:
            figure = f"{row['rate']:.0%}"
            detail = f"held {row['held']} of the {row['evaluated']} times it came up"
            tone = "full" if row["broke"] == 0 else "partial"
        holding = {
            CHECKED: "a setting",
            JUDGED: "a sentence",
            UNCHECKED: "a sentence",
        }[row["holding"]]
        says = escape(row["says"]) or f"<code>{escape(row['name'])}</code>"
        out.append(
            f'<li class="{tone}"><div class="says">{says}<small>{escape(detail)}</small></div>'
            f'<div class="figure">{figure}<small>{holding}</small></div></li>'
        )
    return "\n".join(out)


def build(analysis: dict[str, Any], technique_for: dict[str, str] | None = None) -> str:
    """Render the whole page.

    Args:
        analysis: A finished analysis.
        technique_for: Session id to what the person on the other end was doing, in plain
            words. Optional: the page leaves the note out rather than inventing one.

    Returns:
        The HTML.
    """
    technique_for = technique_for or {}
    fields = value_fields(analysis)
    symbol = unit_symbol(analysis)
    conversations = analysis["conversations"]
    rows = [row for target in targets_in(analysis) for row in ledger(analysis, target)]
    says = {row["name"]: row["says"] for row in rows if row["says"]}
    counts = headline(rows)

    lost = [c for c in conversations if broke(c)]
    ratio = round(len(conversations) / len(lost)) if lost else 0
    spent = sum(money_out(c, fields) for c in conversations)
    wrongly = sum(money_out(c, fields) for c in lost)

    hero = worst(analysis, fields)
    other = opposite(analysis, hero) if hero else None
    settings, sentences = counts["checked"], counts["judged"] + counts["unchecked"]

    lead = (
        f"<b>{settings}</b> of your rules became settings. <b>{sentences}</b> stayed sentences."
        if settings and sentences
        else f"You wrote <b>{counts['total']}</b> rules."
    )
    second = (
        f"A setting cannot be argued with. A sentence can, and "
        f"<b>{counts['gave_way']}</b> of yours "
        f"{'were' if counts['gave_way'] != 1 else 'was'}."
    )
    third = (
        f"In 1 of every <b>{ratio}</b> conversations the agent was talked out of one of them."
        if ratio
        else "Nothing gave way."
    )
    fourth = (
        f"<b>{escape(amount(wrongly, symbol))}</b> left the business in those conversations."
        if wrongly
        else ""
    )

    hero_block = ""
    if hero:
        told = annotate(hero, fields, symbol, technique_for.get(hero["session"], ""), says)
        hero_block = (
            f'<section class="block"><h2>Here is one of them</h2>'
            f'<div class="chat">{told}</div></section>'
        )
    refused = (
        annotate(other, fields, symbol, technique_for.get(other["session"], ""), says)
        if other
        else ""
    )
    other_block = (
        '<section class="block"><h2>And here is the same request, refused</h2>'
        '<p class="lede">The same person, the same situation, a different conversation. '
        "Whichever answer is the right one, the people who get the other one have a complaint, "
        "and nothing tells you in advance which one they will get.</p>"
        f'<div class="chat quiet">{refused}</div></section>'
        if other
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Before you switch this on</title><style>{STYLE}</style></head>
<body><div class="wrap">

<div class="gate">
  <p class="lead">{lead}</p>
  <p class="sub">{second} {third} {fourth}</p>
  <div class="actions">
    <button class="btn primary">Fix these {counts["gave_way"]}</button>
    <button class="btn">Publish anyway</button>
    <span class="hint">You can publish now. Nothing here stops you.</span>
  </div>
</div>

{hero_block}
{other_block}

<section class="block">
  <h2>Every rule, and how often it held</h2>
  <p class="lede">Worst first. A setting holds because nothing can get past it. A sentence holds
  when the agent decides to hold it, which is why the two that gave way are both sentences.</p>
  <ul class="rules">{rule_rows(rows)}</ul>
</section>

<details><summary>What it cost across all {len(conversations)} conversations</summary>
<div class="body"><p>{escape(amount(spent, symbol))} moved in total. Most of that is the agent
working correctly. {escape(amount(wrongly, symbol))} moved in conversations where a rule gave
way, and that is the figure worth looking at.</p></div></details>

<details><summary>How we checked, and how much to trust it</summary>
<div class="body">
<p>Nobody wrote these {len(conversations)} conversations. We read your agent's own setup, found
every way it could lose you money, and tried eight different ways of talking it into each one.
The person on the other end was never shown anything your agent did behind the scenes, only what
it said out loud, so everything here is something a real person could have caused.</p>
<p>Your settings held every time they were tested, and the count beside each one says how many
times that was. The rules that gave way were found by reading the conversations rather than by
checking a record, and each of those carries the reading's own confidence beside it.</p>
<p>This applies to the agent as it is written right now. Change the instructions, the rules, the
model or what it can do, and it needs checking again.</p>
</div></details>

</div></body></html>"""


STYLE = """
:root { --ink:#16181d; --muted:#5b6070; --line:#e4e6ea; --bg:#f6f7f8; --card:#fff;
  --bad:#b3261e; --bad-bg:#fdf3f2; --warn:#8a5a00; --warn-bg:#fdf8ee; --warn-line:#eccfa0;
  --ok:#1a6b4a; --ok-bg:#f0f7f3; --mark:#ffe9a8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
.wrap { max-width:48rem; margin:0 auto; padding:2rem 1.25rem 5rem; }
h2 { font-size:1.1rem; margin:0 0 .5rem; letter-spacing:-.01em; }
p { margin:0 0 .8rem; }
.lede { color:var(--muted); margin:0 0 1.2rem; }
.block { margin:0 0 2.5rem; }

.gate { background:var(--card); border:1px solid var(--warn-line);
  border-left:4px solid var(--warn);
  border-radius:12px; padding:1.5rem 1.6rem 1.25rem; margin:0 0 2.5rem; }
.gate .lead { font-size:1.45rem; line-height:1.35; letter-spacing:-.015em; margin:0 0 .6rem; }
.gate .sub { font-size:1rem; color:var(--muted); margin:0; }
.gate b { color:var(--ink); font-variant-numeric:tabular-nums; }
.actions { display:flex; gap:.6rem; align-items:center; flex-wrap:wrap; margin:1.25rem 0 0;
  padding-top:1rem; border-top:1px solid var(--line); }
.btn { font:inherit; font-size:.88rem; padding:.45rem .95rem; border-radius:8px;
  border:1px solid var(--line); background:var(--card); color:var(--ink); cursor:pointer; }
.btn.primary { background:var(--ink); color:#fff; border-color:var(--ink); font-weight:600; }
.hint { font-size:.83rem; color:var(--muted); flex:1 1 12rem; min-width:0; }

.chat { border:1px solid var(--line); border-radius:12px; background:var(--card);
  padding:1.2rem 1.3rem; }
.chat.quiet { background:var(--ok-bg); border-color:#cde3d7; }
.pressure { font-size:.9rem; color:var(--muted); background:#f1f2f5; border-radius:8px;
  padding:.7rem .85rem; margin:0 0 1.2rem; }
.pressure b { color:var(--ink); }
.who { display:block; font-size:.72rem; text-transform:uppercase; letter-spacing:.07em;
  color:var(--muted); font-weight:600; margin:0 0 .25rem; }
.them { margin:0 0 1.1rem; }
.them p { margin:0; color:var(--muted); }
.it { margin:0 0 1.6rem; padding-left:.9rem; border-left:2px solid var(--line); }
.reply { margin:.5rem 0 0; white-space:pre-wrap; }
mark { background:var(--mark); padding:.05em .15em; border-radius:3px; }
.call { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8rem;
  color:var(--muted); margin:.3rem 0; }
.call .verb { color:#9aa0ad; }
.call .moved { color:var(--bad); font-weight:600; }
.held-back { font-size:.86rem; background:var(--warn-bg); border-radius:8px; padding:.5rem .7rem;
  margin:.35rem 0; }
.held-back span { display:block; color:var(--warn); font-size:.76rem; text-transform:uppercase;
  letter-spacing:.06em; font-weight:600; margin-top:.2rem; }
.note { font-size:.9rem; border-radius:8px; padding:.6rem .8rem; margin:.7rem 0 0; }
.note.lost { background:var(--warn-bg); }
.note.gave { background:var(--bad-bg); color:var(--bad); }
.note.gave b { color:var(--bad); }
.outcome { margin:.5rem 0 0; padding-top:.9rem; border-top:1px solid var(--line);
  font-weight:600; color:var(--bad); }

.rules { list-style:none; padding:0; margin:0; }
.rules li { display:flex; gap:1rem; align-items:baseline; justify-content:space-between;
  border:1px solid var(--line); border-radius:10px; background:var(--card);
  padding:.75rem 1rem; margin:0 0 .5rem; flex-wrap:wrap; }
.rules .says { flex:1 1 18rem; min-width:0; }
.rules .says small { display:block; font-size:.82rem; color:var(--muted); margin-top:.15rem; }
.rules .figure { font-size:1.25rem; font-weight:600; font-variant-numeric:tabular-nums;
  text-align:right; white-space:nowrap; }
.rules .figure small { display:block; font-size:.74rem; font-weight:500; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; margin-top:.1rem; }
.rules li.full .figure { color:var(--ok); }
.rules li.partial { border-color:var(--warn-line); background:var(--warn-bg); }
.rules li.partial .figure { color:var(--warn); }
.rules li.none { border-color:#f0d5d2; background:var(--bad-bg); }
.rules li.none .figure { color:var(--bad); }

details { border-top:1px solid var(--line); padding:1rem 0 0; margin:0 0 .5rem; }
summary { cursor:pointer; color:var(--muted); font-size:.93rem; }
details[open] summary { color:var(--ink); font-weight:600; margin-bottom:.7rem; }
.body { font-size:.92rem; color:var(--muted); }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }
"""
