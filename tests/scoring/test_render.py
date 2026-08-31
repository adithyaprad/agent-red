"""The page says which runs it is about, and which agent version those numbers are valid for."""

from __future__ import annotations

from agentred.scoring.render import build, provenance_line


def analysis(runs=None, conversations=None) -> dict:
    return {
        "runs": [] if runs is None else runs,
        "conversations": [] if conversations is None else conversations,
        "policy": {},
        "consistency": {
            "groups": 0,
            "settled": 0,
            "unknown_subject": 0,
            "rate": 0.0,
            "divergences": [],
        },
    }


def run(run_id="run-a1", target="dispute_handler", notes="run 0007, stake=all") -> dict:
    return {
        "run_id": run_id,
        "target": target,
        "started_at": "2026-08-31T09:00:00Z",
        "finished_at": "2026-08-31T09:12:00Z",
        "notes": notes,
        "versions": {
            "config": "1.0",
            "policy": "1.1",
            "model": "claude-sonnet-5",
            "tools": "sha256:abc123",
        },
        "conversations": 8,
    }


class TestSayingWhatTheNumbersAreFor:
    def test_the_run_is_named_the_way_the_directory_listing_names_it(self):
        line = provenance_line(analysis(runs=[run()]))
        assert "run 0007" in line

    def test_all_four_versions_appear(self):
        """Change any one of them and the agent is untested again, so a page that omits one
        lets a stale result outlive the thing it measured."""
        line = provenance_line(analysis(runs=[run()]))
        for version in ("1.0", "1.1", "claude-sonnet-5", "sha256:abc123"):
            assert version in line

    def test_two_runs_of_one_agent_state_the_tuple_once(self):
        line = provenance_line(analysis(runs=[run("run-a1"), run("run-a2", notes="run 0008")]))
        assert line.count("sha256:abc123") == 1
        assert "run 0007" in line and "run 0008" in line

    def test_a_run_with_no_note_falls_back_to_its_id(self):
        line = provenance_line(analysis(runs=[run(notes="")]))
        assert "run-a1" in line

    def test_an_analysis_from_before_runs_were_recorded_renders_nothing(self):
        """Analyses already on disk predate this and must still render, without inventing a
        provenance nobody recorded."""
        assert provenance_line({"conversations": []}) == ""


class TestTheWholePage:
    def test_a_run_with_no_failures_still_renders_its_provenance(self):
        page = build(analysis(runs=[run()]))
        assert "run 0007" in page
        assert "No rule was broken" in page

    def test_a_page_built_from_an_older_analysis_does_not_raise(self):
        page = build(analysis())
        assert "<h1>What broke, and where</h1>" in page
