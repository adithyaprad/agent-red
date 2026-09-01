"""Every rule an agent claims, with what happened to it and what was holding it.

The page a person reads is organised around one question: which of the rules I wrote will
actually hold when somebody pushes. That question is answered per rule, not per conversation,
and the answer has three parts.

**How often it held.** Held over held plus broken, counted only across the conversations where
the rule was actually in play. A rule that never came up has no rate, and inventing one by
counting the conversations that never reached it would report a limit as safe on the strength
of the times nobody tried.

**What was holding it.** A rule the operator declared and that a check can settle from the
record of what the agent did is a different kind of promise from a rule that exists only as a
sentence somewhere and has to be read and judged. Both can hold. Only one of them holds for a
reason. The distinction is already carried on every finding, as `settled_by` and `provenance`,
so this module groups rather than decides.

**Whether anybody looked.** A rule found in the agent's own instructions that no check covers
and that no finding names has not survived anything. It has not been tried. That is worse than
a low rate and it is the one state a report of failures alone can never show, because there is
nothing to report.

Nothing here calls a model or reaches a store. It is a pure function of a finished analysis.
"""

from __future__ import annotations

from typing import Any

CHECKED = "checked"
"""Settled from the record of what the agent did. Held means held."""

JUDGED = "judged"
"""Settled by reading what the agent said. Held means nothing was spotted."""

UNCHECKED = "unchecked"
"""Stated somewhere and covered by nothing. Never put to the test."""


def _rule_text(spec_block: dict[str, Any]) -> dict[str, str]:
    """Map every rule name an agent declares to the sentence a person would recognise.

    Args:
        spec_block: One agent's entry from the analysis `policy` section.

    Returns:
        Rule name to its stated sentence. Names with no sentence are absent rather than
        present and empty, so a caller can tell "said nothing" from "said this".
    """
    text: dict[str, str] = dict(spec_block.get("declared_rules", {}))
    for statement in spec_block.get("statements", ()):
        if statement.get("description"):
            text[statement["name"]] = statement["description"]
    for duty in spec_block.get("obligations", ()):
        if duty.get("statement"):
            text[duty["name"]] = duty["statement"]
    return text


def ledger(analysis: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """Every rule this agent claims, with how it fared, worst first.

    Sorting is by how often the rule held, ascending, so a rule that gave way sits above one
    that never did and a rule nobody checked sits above both. That sequence is the whole
    argument of the page: the reader's attention goes to the rules that are not real, and the
    ones that are real take up as little room as they deserve.

    Args:
        analysis: A finished analysis.
        target: Which agent's rules to report on.

    Returns:
        One row per rule: `name`, `says`, `held`, `broke`, `evaluated`, `seen_in`, `rate`
        (None when the rule was never in play), and `holding`, one of `CHECKED`, `JUDGED` or
        `UNCHECKED`.
    """
    conversations = [c for c in analysis["conversations"] if c["target"] == target]
    spec_block = analysis.get("policy", {}).get(target, {})
    text = _rule_text(spec_block)

    rows: dict[str, dict[str, Any]] = {}
    for conversation in conversations:
        for finding in conversation["findings"]:
            row = rows.setdefault(
                finding["rule"],
                {
                    "name": finding["rule"],
                    "says": text.get(finding["rule"], ""),
                    "held": 0,
                    "broke": 0,
                    "seen_in": 0,
                    "holding": CHECKED if finding["settled_by"] == "detector" else JUDGED,
                    "declared": finding.get("provenance") == "declared",
                },
            )
            row["seen_in"] += 1
            if finding["outcome"] == "held":
                row["held"] += 1
            elif finding["outcome"] == "violated":
                row["broke"] += 1

    # A rule read out of the agent's own instructions that no finding names was covered by
    # nothing. It cannot be found by looking at what was checked, only by comparing what the
    # agent says it does against what anybody asked about, which is why it is added here
    # rather than counted above.
    for statement in list(spec_block.get("statements", ())) + list(
        spec_block.get("obligations", ())
    ):
        if statement["name"] in rows or statement.get("declared"):
            continue
        rows[statement["name"]] = {
            "name": statement["name"],
            "says": statement.get("description") or statement.get("statement", ""),
            "held": 0,
            "broke": 0,
            "seen_in": 0,
            "holding": UNCHECKED,
            "declared": False,
        }

    for row in rows.values():
        row["evaluated"] = row["held"] + row["broke"]
        row["rate"] = row["held"] / row["evaluated"] if row["evaluated"] else None

    return sorted(
        rows.values(),
        key=lambda row: (row["rate"] if row["rate"] is not None else -1.0, row["name"]),
    )


def headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The counts the first sentence of the page is built from.

    Returns:
        `total`, `checked`, `judged`, `unchecked`, and `gave_way`, the number of rules that
        broke at least once. `gave_way` is a count of rules and not of failures on purpose:
        one rule that broke seven times is one thing to fix, and reporting seven makes the
        work look larger than it is while telling the reader nothing about its shape.
    """
    return {
        "total": len(rows),
        "checked": sum(1 for row in rows if row["holding"] == CHECKED),
        "judged": sum(1 for row in rows if row["holding"] == JUDGED),
        "unchecked": sum(1 for row in rows if row["holding"] == UNCHECKED),
        "gave_way": sum(1 for row in rows if row["broke"]),
    }
