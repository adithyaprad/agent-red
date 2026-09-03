"""What a run cost, read from the two places tokens are recorded.

A suite spends on both sides of every turn and the two halves are recorded in different
places, for a reason that is not incidental: the harness's own calls go through a client it
owns and are written to the call recording, and the target's calls happen inside the target,
which reports what they cost in its reply and nowhere else. A report that reads only the
recording sees the attacker and the judge and reports roughly half the bill.

So this reads both, keeps them separate in the output, and counts what neither of them
covers. Three things are reported rather than smoothed over:

- Turns where the target reported no usage. Not free, just unreported, and a report that
  silently treated them as zero would make a target on a route with no usage accounting look
  cheaper than one that reports honestly.
- Calls on a model with no rate. Named, with their tokens, rather than dropped or priced at
  nothing. See `agentred.llm.rates`.
- Whether the figure is a bill or an estimate, which is a property of the route.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentred.llm.client import Route
from agentred.llm.rates import UnpricedModelError, cost_usd, rate_source
from agentred.llm.recording import read_records
from agentred.store import Store

HARNESS = "harness"
"""The attacker, the planted payload composer, and the judge. Calls agent-red makes itself."""

TARGET = "target"
"""The agent under test. Calls made inside the target, reported in its replies."""

TOKEN_KINDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
"""The token counts both sides report, in the sequence a report lists them."""


@dataclass
class Tally:
    """Calls and tokens for one side on one model, with what they cost.

    Attributes:
        side: `HARNESS` or `TARGET`.
        model: The model as the response reported it, before any prefix is stripped.
        calls: How many calls landed here.
        tokens: Token counts by kind, summed.
        usd: What they cost, or `None` if the model has no rate.
    """

    side: str
    model: str
    calls: int = 0
    tokens: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TOKEN_KINDS, 0.0))
    usd: float | None = 0.0

    def add(self, usage: dict[str, float]) -> None:
        """Fold one call's usage in, and cost it if the model has a rate.

        Args:
            usage: Token counts as the side reported them. Absent kinds count as zero.
        """
        self.calls += 1
        for kind in TOKEN_KINDS:
            self.tokens[kind] += float(usage.get(kind, 0) or 0)
        if self.usd is None:
            return
        try:
            self.usd += cost_usd(self.model, usage)
        except UnpricedModelError:
            self.usd = None


@dataclass
class CostReport:
    """What one run cost, and what part of it is not known.

    Attributes:
        tallies: One per side and model, in the sequence they were first seen.
        per_attack: USD by attack id, for the harness half only. The target's usage is
            recorded per turn against a conversation, which carries the attack id, so the
            target half is attributable too and is added here when a store is read.
        rate_source: Whether the figure is a bill or an estimate. From the route.
        turns_without_usage: Turns where the target reported nothing. Not free, unreported.
        failed_calls: Harness calls that raised. They may still have been billed and carry
            no usage, so they are counted and not priced.
    """

    tallies: dict[tuple[str, str], Tally] = field(default_factory=dict)
    per_attack: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    rate_source: str = ""
    turns_without_usage: int = 0
    failed_calls: int = 0

    def tally(self, side: str, model: str) -> Tally:
        """The tally for one side and model, created on first use."""
        key = (side, model)
        if key not in self.tallies:
            self.tallies[key] = Tally(side=side, model=model)
        return self.tallies[key]

    @property
    def unpriced(self) -> tuple[Tally, ...]:
        """Tallies whose model has no rate, so whose tokens are counted and not priced."""
        return tuple(tally for tally in self.tallies.values() if tally.usd is None)

    @property
    def total_usd(self) -> float:
        """What the priced part of the run cost.

        Excludes any tally with no rate, which is why `unpriced` has to be read beside it.
        """
        return sum(tally.usd for tally in self.tallies.values() if tally.usd is not None)

    def side_usd(self, side: str) -> float:
        """What the priced part of one side cost."""
        return sum(
            tally.usd
            for tally in self.tallies.values()
            if tally.side == side and tally.usd is not None
        )


def add_recording(report: CostReport, path: Path | str) -> None:
    """Fold the harness's half in, from the call recording.

    Args:
        report: The report to add to.
        path: The JSONL a `CallRecorder` wrote. A missing file adds nothing, because a run
            that was cut before its first call has a report and no recording.
    """
    if not Path(path).exists():
        return
    for record in read_records(path):
        if not record.get("ok"):
            report.failed_calls += 1
            continue
        model = str(record.get("model") or "")
        usage = dict(record.get("usage") or {})
        tally = report.tally(HARNESS, model)
        before = tally.usd
        tally.add(usage)
        label = str(record.get("label") or "")
        if label and before is not None and tally.usd is not None:
            report.per_attack[label] += tally.usd - before


def add_store(report: CostReport, store_path: Path | str, run_id: str, target_model: str) -> None:
    """Fold the target's half in, from the persisted turns.

    The target reports token counts and not which model produced them, because it runs one
    model per version and the version is pinned on the run. So the model is passed in from
    the target's own config rather than guessed from the reply.

    Args:
        report: The report to add to.
        store_path: The SQLite database.
        run_id: Which run to read.
        target_model: The model the target's config declares.
    """
    with Store(store_path) as store:
        rows = store.connection.execute(
            "SELECT c.attack_id AS attack_id, t.usage_json AS usage_json "
            "FROM turns t JOIN conversations c ON c.conversation_id = t.conversation_id "
            "WHERE c.run_id = ?",
            (run_id,),
        ).fetchall()
    tally = report.tally(TARGET, target_model)
    for row in rows:
        usage: dict[str, Any] = json.loads(row["usage_json"] or "{}")
        if not usage:
            report.turns_without_usage += 1
            continue
        before = tally.usd
        tally.add(usage)
        attack_id = str(row["attack_id"] or "")
        if attack_id and before is not None and tally.usd is not None:
            report.per_attack[attack_id] += tally.usd - before


def build_report(
    *,
    recording: Path | str,
    route: Route,
    store_path: Path | str | None = None,
    run_id: str = "",
    target_model: str = "",
) -> CostReport:
    """Read both halves and cost them.

    Args:
        recording: The call recording for the run.
        route: The resolved model route, which decides whether the figure is a bill.
        store_path: The database, when the target's half is wanted. Omit to report the
            harness half alone, which is what a run with no store has.
        run_id: The run whose turns to read. Required with `store_path`.
        target_model: The model the target declares. Required with `store_path`.

    Returns:
        The report.
    """
    report = CostReport(rate_source=rate_source(route))
    add_recording(report, recording)
    if store_path is not None and run_id and target_model:
        add_store(report, store_path, run_id, target_model)
    return report


def render(report: CostReport, *, attacks: int = 0) -> str:
    """The report as lines a person reads in a terminal.

    Args:
        report: The report.
        attacks: How many attacks the run held, for a per-attack average. Omit to skip it.

    Returns:
        The text, without a trailing newline.
    """
    lines = [f"cost        ${report.total_usd:.2f} ({report.rate_source})"]
    for side in (HARNESS, TARGET):
        tallies = [tally for tally in report.tallies.values() if tally.side == side]
        if not tallies:
            continue
        lines.append(f"  {side:9} ${report.side_usd(side):.2f}")
        for tally in tallies:
            money = "unpriced" if tally.usd is None else f"${tally.usd:.2f}"
            lines.append(
                f"    {tally.model:28} {tally.calls:>5} call(s)  "
                f"{tally.tokens['input_tokens']:>10,.0f} in  "
                f"{tally.tokens['output_tokens']:>9,.0f} out  "
                f"{tally.tokens['cache_read_tokens']:>10,.0f} cached  {money}"
            )
    if attacks:
        lines.append(f"  per attack  ${report.total_usd / attacks:.3f} across {attacks} attack(s)")
    if report.unpriced:
        named = ", ".join(sorted(tally.model for tally in report.unpriced))
        lines.append(
            f"  not in the total: {named}. No rate in agentred.llm.rates, so those "
            f"tokens are counted above and priced nowhere."
        )
    if report.turns_without_usage:
        lines.append(
            f"  {report.turns_without_usage} turn(s) where the target reported no usage. "
            f"Counted as unreported, not as free."
        )
    if report.failed_calls:
        lines.append(
            f"  {report.failed_calls} harness call(s) raised and carry no usage. They may "
            f"still have been billed."
        )
    return "\n".join(lines)
