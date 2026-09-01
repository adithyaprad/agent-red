"""What the ledger says about a rule, and what it refuses to say.

Every case here is one the page would get wrong in a way nobody would notice: a rate computed
over conversations that never reached the rule, a rule with nothing checking it reported as
fine because nothing complained, an identifier shown to a reader in place of the sentence they
wrote. All of them read as correct output.
"""

from __future__ import annotations

from typing import Any

from agentred.scoring.ledger import CHECKED, JUDGED, UNCHECKED, headline, ledger


def finding(
    rule: str,
    outcome: str,
    *,
    settled_by: str = "detector",
    provenance: str = "declared",
) -> dict[str, Any]:
    return {
        "rule": rule,
        "outcome": outcome,
        "settled_by": settled_by,
        "provenance": provenance,
        "kind": "bound_exceeded",
    }


def analysis(*conversations: list[dict[str, Any]], policy: dict[str, Any] | None = None):
    return {
        "conversations": [
            {"target": "agent", "session": f"s{i}", "findings": findings}
            for i, findings in enumerate(conversations)
        ],
        "policy": {"agent": policy or {}},
    }


class TestHowOftenItHeld:
    def test_a_rule_that_never_gave_way_is_reported_at_full(self):
        rows = ledger(analysis([finding("cap", "held")], [finding("cap", "held")]), "agent")
        assert rows[0]["rate"] == 1.0
        assert (rows[0]["held"], rows[0]["broke"]) == (2, 0)

    def test_the_denominator_excludes_conversations_the_rule_was_never_in_play_for(self):
        rows = ledger(
            analysis(
                [finding("cap", "held")],
                [finding("cap", "violated")],
                [finding("cap", "not_evaluated")],
                [finding("cap", "not_evaluated")],
            ),
            "agent",
        )
        # Two of the four conversations never reached the rule. Counting them would report
        # it at three quarters rather than a half, on the strength of the times nobody tried.
        assert rows[0]["evaluated"] == 2
        assert rows[0]["rate"] == 0.5
        assert rows[0]["seen_in"] == 4

    def test_a_rule_nothing_ever_reached_has_no_rate_rather_than_a_perfect_one(self):
        rows = ledger(analysis([finding("cap", "not_evaluated")]), "agent")
        assert rows[0]["rate"] is None
        assert rows[0]["evaluated"] == 0


class TestWhatWasHoldingIt:
    def test_a_rule_settled_from_the_record_is_checked(self):
        rows = ledger(analysis([finding("cap", "held", settled_by="detector")]), "agent")
        assert rows[0]["holding"] == CHECKED

    def test_a_rule_settled_by_reading_is_judged(self):
        rows = ledger(analysis([finding("tone", "held", settled_by="judge")]), "agent")
        assert rows[0]["holding"] == JUDGED

    def test_a_rule_in_the_prose_that_nothing_asked_about_is_unchecked(self):
        rows = ledger(
            analysis(
                [finding("cap", "held")],
                policy={
                    "obligations": [
                        {"name": "quiet", "statement": "Say nothing.", "declared": False}
                    ]
                },
            ),
            "agent",
        )
        quiet = next(row for row in rows if row["name"] == "quiet")
        assert quiet["holding"] == UNCHECKED
        assert quiet["rate"] is None

    def test_a_prose_rule_the_policy_also_declares_is_not_reported_as_unchecked(self):
        # The trap: extraction reads a rule out of the prompt that the policy already carries
        # under its own name. It is checked, and listing it again as unchecked would invent a
        # blind spot that does not exist.
        rows = ledger(
            analysis(
                [finding("cap", "held")],
                policy={
                    "statements": [{"name": "cap_rule", "description": "Cap it.", "declared": True}]
                },
            ),
            "agent",
        )
        assert [row["name"] for row in rows] == ["cap"]


class TestWhatTheReaderIsShown:
    def test_a_rule_carries_the_sentence_the_operator_wrote(self):
        rows = ledger(
            analysis(
                [finding("cap", "held")], policy={"declared_rules": {"cap": "Never over 50."}}
            ),
            "agent",
        )
        assert rows[0]["says"] == "Never over 50."

    def test_a_rule_with_no_stated_sentence_comes_back_empty_rather_than_named(self):
        rows = ledger(analysis([finding("cap", "held")]), "agent")
        assert rows[0]["says"] == ""


class TestTheSequence:
    def test_the_worst_rule_is_first_and_the_unchecked_one_is_above_it(self):
        rows = ledger(
            analysis(
                [finding("solid", "held")],
                [finding("weak", "held")],
                [finding("weak", "violated")],
                policy={
                    "obligations": [
                        {"name": "unseen", "statement": "Nobody looked.", "declared": False}
                    ]
                },
            ),
            "agent",
        )
        assert [row["name"] for row in rows] == ["unseen", "weak", "solid"]


class TestTheHeadline:
    def test_it_counts_rules_that_gave_way_rather_than_times_they_did(self):
        rows = ledger(
            analysis(
                [finding("weak", "violated", settled_by="judge")],
                [finding("weak", "violated", settled_by="judge")],
                [finding("weak", "violated", settled_by="judge")],
                [finding("solid", "held")],
            ),
            "agent",
        )
        assert headline(rows) == {
            "total": 2,
            "checked": 1,
            "judged": 1,
            "unchecked": 0,
            "gave_way": 1,
        }

    def test_another_agents_conversations_are_not_counted(self):
        data = analysis([finding("cap", "held")])
        data["conversations"].append(
            {"target": "elsewhere", "session": "x", "findings": [finding("cap", "violated")]}
        )
        assert ledger(data, "agent")[0]["rate"] == 1.0
