"""What the page shows, and what it refuses to show.

The failures worth guarding here are the ones that still render. A page that marks the wrong
sentence, that opens with the flag we are least sure about, or that prints a figure with no
unit in front of it all look finished.
"""

from __future__ import annotations

from typing import Any

from agentred.scoring.merchant import amount, annotate, build, highlight, worst


def conversation(
    session: str = "s1",
    *,
    turns: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
    points: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "target": "agent",
        "session": session,
        "subject": {"ref": "R-1"},
        "turns": turns or [{"index": 0, "user": "please", "reply": "no", "calls": []}],
        "findings": findings or [],
        "breaking_points": points or [],
    }


def gave_way(rule: str = "quiet", *, quote: str = "", confidence: float = 0.9, turn: int = 0):
    finding: dict[str, Any] = {
        "rule": rule,
        "outcome": "violated",
        "settled_by": "judge",
        "provenance": "inferred",
        "kind": "obligation:disclosure",
        "confidence": confidence,
    }
    if quote:
        finding["utterance"] = {"turn": turn, "quote": quote, "source_value": "the note"}
    return finding


def analysis(*conversations: dict[str, Any], **extra: Any) -> dict[str, Any]:
    base = {
        "conversations": list(conversations),
        "policy": {"agent": {"declared_rules": {"quiet": "Keep the note to yourself."}}},
        "presentation": {
            "agent": {"unit_symbol": "₹", "subject_term": "person", "value_fields": ["paid"]}
        },
        "consistency": {"divergences": []},
    }
    base.update(extra)
    return base


class TestTheSentenceThatIsMarked:
    def test_the_quote_is_marked_inside_the_reply_that_held_it(self):
        assert highlight("I can do that for you.", "do that") == (
            "I can <mark>do that</mark> for you."
        )

    def test_a_quote_that_is_not_there_leaves_the_reply_unmarked(self):
        # A quote that has drifted is a defect worth seeing. Matching loosely would mark
        # some other sentence with exactly the confidence of a real match.
        assert highlight("I can do that.", "something else") == "I can do that."

    def test_the_reply_is_escaped_on_both_sides_of_the_mark(self):
        marked = highlight("<b>a</b> keep <i>b</i>", "keep")
        assert "&lt;b&gt;a&lt;/b&gt;" in marked and "&lt;i&gt;b&lt;/i&gt;" in marked


class TestWhichConversationOpensThePage:
    def test_the_one_that_cost_the_most_is_chosen(self):
        cheap = conversation("cheap", findings=[gave_way()])
        dear = conversation(
            "dear",
            findings=[gave_way()],
            turns=[
                {
                    "index": 0,
                    "user": "please",
                    "reply": "done",
                    "calls": [{"name": "pay", "arguments": {}, "result": {"paid": 900.0}}],
                }
            ],
        )
        assert worst(analysis(cheap, dear), ("paid",))["session"] == "dear"

    def test_among_equals_the_more_confident_finding_wins(self):
        unsure = conversation("unsure", findings=[gave_way(confidence=0.3)])
        sure = conversation("sure", findings=[gave_way(confidence=0.95)])
        assert worst(analysis(unsure, sure), ("paid",))["session"] == "sure"

    def test_nothing_is_chosen_when_no_rule_gave_way(self):
        assert worst(analysis(conversation()), ("paid",)) is None


class TestWhatTheAnnotationSays:
    def test_it_shows_the_sentence_the_operator_wrote_not_our_name_for_it(self):
        one = conversation(
            findings=[gave_way(quote="do that")],
            turns=[{"index": 0, "user": "go on", "reply": "I can do that.", "calls": []}],
        )
        out = annotate(one, ("paid",), "₹", "", {"quiet": "Keep the note to yourself."})
        assert "Keep the note to yourself." in out
        assert "quiet" not in out

    def test_it_falls_back_to_the_rule_name_rather_than_saying_nothing(self):
        one = conversation(
            findings=[gave_way(quote="do that")],
            turns=[{"index": 0, "user": "go on", "reply": "I can do that.", "calls": []}],
        )
        assert "quiet" in annotate(one, ("paid",), "₹", "", {})

    def test_the_note_about_what_they_were_doing_is_left_out_when_unknown(self):
        out = annotate(conversation(), ("paid",), "₹", "", {})
        assert "What they were doing" not in out


class TestTheFigures:
    def test_an_amount_carries_whatever_the_agent_declares_in_front_of_it(self):
        assert amount(8151.0, "₹") == "₹8,151"
        assert amount(12.5, "$") == "$12.50"

    def test_a_page_reports_the_unit_the_agent_declared(self):
        one = conversation(
            findings=[gave_way()],
            turns=[
                {
                    "index": 0,
                    "user": "please",
                    "reply": "done",
                    "calls": [{"name": "pay", "arguments": {}, "result": {"paid": 500.0}}],
                }
            ],
        )
        assert "₹500" in build(analysis(one))


class TestWhoIsBlamed:
    def test_the_page_never_says_the_operator_failed_and_never_takes_the_praise(self):
        page = build(analysis(conversation(findings=[gave_way()]))).lower()
        for phrase in ("you failed", "your fault", "we caught", "we found", "we detected"):
            assert phrase not in page

    def test_the_gate_advises_and_does_not_block(self):
        page = build(analysis(conversation(findings=[gave_way()])))
        assert "Publish anyway" in page
        assert "Nothing here stops you" in page
