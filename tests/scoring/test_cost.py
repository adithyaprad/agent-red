"""What a run cost, and what the report refuses to guess about.

Offline. The interesting cases are the three the report is built to not smooth over: an
unpriced model, a turn the target reported nothing for, and a call that raised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentred.llm.client import Route
from agentred.llm.rates import UnpricedModelError, canonical_model, cost_usd, rate_source
from agentred.scoring.cost import HARNESS, TARGET, CostReport, add_recording, build_report, render


def write_recording(path: Path, records: list[dict[str, object]]) -> None:
    """Write a recording file the way `CallRecorder` would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            handle.write(json.dumps({"schema": 1, "sequence": index, **record}) + "\n")


def call(
    *,
    model: str = "claude-sonnet-5",
    label: str = "atk-1",
    input_tokens: int = 1_000,
    output_tokens: int = 100,
    ok: bool = True,
) -> dict[str, object]:
    """One recorded call."""
    if not ok:
        return {"label": label, "ok": False, "error_type": "APIStatusError", "error": "boom"}
    return {
        "label": label,
        "ok": True,
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": 0,
        },
    }


def test_sonnet_is_priced_at_two_and_ten_per_million() -> None:
    """The published rate, which is what every figure in a report is built on."""
    usd = cost_usd("claude-sonnet-5", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert usd == pytest.approx(12.00)


def test_a_bedrock_model_id_prices_the_same_as_the_bare_one() -> None:
    """One rate table serves every route, so the prefix is stripped rather than keyed on."""
    assert canonical_model("global.anthropic.claude-sonnet-5") == "claude-sonnet-5"
    usage = {"input_tokens": 1_000_000}
    assert cost_usd("global.anthropic.claude-sonnet-5", usage) == cost_usd("claude-sonnet-5", usage)


def test_an_unknown_model_raises_rather_than_pricing_at_zero() -> None:
    """A bill that reads low because a model was missing looks exactly like a cheap run."""
    with pytest.raises(UnpricedModelError, match="no rate for model"):
        cost_usd("claude-not-a-model", {"input_tokens": 1_000})


def test_bedrock_is_labelled_an_estimate_and_the_first_party_route_is_not() -> None:
    """A dollar figure with no provenance gets quoted as if it were a bill."""
    assert "partner-billed" in rate_source(Route.BEDROCK)
    assert "partner-billed" not in rate_source(Route.FIRST_PARTY)


def test_the_harness_half_is_summed_per_model_and_attributed_per_attack(tmp_path: Path) -> None:
    """Every call carries the attack it belongs to, so the bill can be read per cell."""
    recording = tmp_path / "calls.jsonl"
    write_recording(
        recording,
        [
            call(label="atk-1", input_tokens=1_000_000, output_tokens=0),
            call(label="atk-2", input_tokens=1_000_000, output_tokens=0),
        ],
    )
    report = build_report(recording=recording, route=Route.FIRST_PARTY)
    assert report.total_usd == pytest.approx(4.00)
    assert report.side_usd(HARNESS) == pytest.approx(4.00)
    assert report.per_attack["atk-1"] == pytest.approx(2.00)
    assert report.per_attack["atk-2"] == pytest.approx(2.00)


def test_a_failed_call_is_counted_and_not_priced(tmp_path: Path) -> None:
    """It carries no usage and may still have been billed, so it is reported as its own row."""
    recording = tmp_path / "calls.jsonl"
    write_recording(recording, [call(ok=False), call(input_tokens=1_000_000, output_tokens=0)])
    report = build_report(recording=recording, route=Route.FIRST_PARTY)
    assert report.failed_calls == 1
    assert report.total_usd == pytest.approx(2.00)
    assert "1 harness call(s) raised" in render(report)


def test_an_unpriced_model_leaves_the_total_and_says_so(tmp_path: Path) -> None:
    """Its tokens are still counted, and the total says it does not include them."""
    recording = tmp_path / "calls.jsonl"
    write_recording(recording, [call(model="claude-not-a-model", input_tokens=5_000)])
    report = build_report(recording=recording, route=Route.FIRST_PARTY)
    assert report.total_usd == pytest.approx(0.0)
    assert [tally.model for tally in report.unpriced] == ["claude-not-a-model"]
    rendered = render(report)
    assert "unpriced" in rendered
    assert "not in the total" in rendered


def test_a_missing_recording_reports_nothing_rather_than_raising(tmp_path: Path) -> None:
    """A run cut before its first call has a report and no recording."""
    report = CostReport(rate_source=rate_source(Route.FIRST_PARTY))
    add_recording(report, tmp_path / "absent.jsonl")
    assert report.total_usd == pytest.approx(0.0)
    assert report.tallies == {}


def test_a_turn_the_target_reported_nothing_for_is_unreported_not_free() -> None:
    """A route with no usage accounting would otherwise look cheaper than an honest one."""
    report = CostReport(rate_source=rate_source(Route.FIRST_PARTY))
    report.turns_without_usage = 3
    assert "not as free" in render(report)


def test_both_sides_are_reported_separately(tmp_path: Path) -> None:
    """Reading only the recording sees the attacker and the judge, which is half the bill."""
    report = CostReport(rate_source=rate_source(Route.FIRST_PARTY))
    report.tally(HARNESS, "claude-sonnet-5").add({"input_tokens": 1_000_000})
    report.tally(TARGET, "claude-sonnet-5").add({"input_tokens": 1_000_000})
    assert report.side_usd(HARNESS) == pytest.approx(2.00)
    assert report.side_usd(TARGET) == pytest.approx(2.00)
    assert report.total_usd == pytest.approx(4.00)
    rendered = render(report, attacks=2)
    assert "harness" in rendered
    assert "target" in rendered
    assert "per attack" in rendered
