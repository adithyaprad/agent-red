"""Run a handful of real attacks against a served target, and record everything.

**This is not a measurement.** Nothing here is scored, published, or turned into held-out
data. It settles one assumption that everything after it rests on: whether a model handed a
technique and an objective writes turns as persuasive as the hand-written exemplars in
`data/techniques/`. If it writes politely, a low violation rate would mean the attacker was
weak rather than the agent safe, and every number the suite produces afterwards inherits
that. Cheaper to find out on eight conversations than on four hundred.

Nothing about the attacks is written here. The suite is derived from the target's own spec by
`build_suite`, exactly as a real run would derive it, and this script only filters that suite
down to one stake so the run is small. Change `--stake` and a different slice runs; change
nothing and the machinery is identical to the real thing.

Conversations run concurrently because their worlds are isolated: `TargetAgent.session()`
gives each session id a fresh world, and the driver mints a new id per conversation. The
store is not thread-safe (one connection, `check_same_thread`), so transcripts come back to
the main thread to be written. Persistence was never the slow part.

    uv run python scripts/smoke.py --target dispute_handler
        --stake precondition_skipped:issue_refund:verify_identity

Run `--list-stakes` first to see what the target derives.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentred.attacks.generator import Attack, build_attackers, build_suite
from agentred.attacks.stakes import derive_stakes
from agentred.judge.detectors import run_detectors
from agentred.judge.models import Finding
from agentred.judge.models import Outcome as JudgeOutcome
from agentred.llm.client import AnthropicModelClient
from agentred.llm.recording import CallRecorder, RecordingModelClient
from agentred.runner.consent import ConsentToken, establish_consent, load_registry
from agentred.runner.conversation import Transcript, run_conversation
from agentred.spec import load_spec_dir
from agentred.store.repo import Store

SMOKE_MODEL = "claude-sonnet-5"
"""What composes the attack turns here.

Sonnet rather than Opus, matching what the targets themselves run. It makes the result
asymmetric and that is understood: persuasive turns from Sonnet mean Opus would also manage
it, while polite ones leave weak-model and weak-prompt indistinguishable. Worth it for a run
whose whole purpose is to be cheap enough to repeat.
"""

DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_TURNS = 6

RUNS_ROOT = Path.home() / "Desktop" / "agent-red-private" / "runs"
"""Where runs are kept, outside the repository.

A run holds complete transcripts of an agent being manipulated, the full prompts that did it,
and what it cost. None of that belongs in a repository someone else will read, and a timestamped
directory inside `data/` is one `git add .` away from being in one. It lives beside the plan and
the engineering log, which is where the rest of this project's private material already is.
"""


def next_run_dir(target: str, stake: str, label: str = "", root: Path | None = None) -> Path:
    """Allocate the next numbered directory for a run.

    Numbered rather than timestamped, because the question anyone actually asks of a run is
    which one came before it and which one came after. A timestamp answers when, in a format
    nobody can order at a glance, and two runs a minute apart sort by a string of digits that
    means nothing. The name then says what the run was, so a directory listing reads as a
    history rather than as a pile.

    Args:
        target: The registered target name.
        stake: The stake id the run was filtered to, or empty for the whole suite.
        label: Optional human tag appended to the name.
        root: Where runs live. Defaults to `RUNS_ROOT`.

    Returns:
        The created directory, `NNNN-<target>-<stake>[-<label>]`.
    """
    root = RUNS_ROOT if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    parts = [f"{_claim_number(root):04d}", target, _slug(stake) or "full-suite"]
    if label:
        parts.append(_slug(label))
    directory = root / "-".join(parts)
    directory.mkdir(parents=True, exist_ok=False)
    return directory


COUNTER_FILENAME = ".next-run"
"""Where the next run number is kept.

A counter rather than one past the highest directory present. The two agree until somebody
deletes a run, at which point the highest-present reading hands its number to the next run, and
two different runs end up sharing a name in the engineering log, in the store's notes and in
whatever anyone wrote down. A number that has been used once is spent.
"""


def _claim_number(root: Path) -> int:
    """Take the next run number and record that it is taken.

    Args:
        root: The runs directory.

    Returns:
        The number. Falls back to one past the highest directory present when the counter file
        is missing or unreadable, so an existing runs folder adopts the scheme without
        renumbering, and a corrupted counter costs at most a repeated number rather than
        refusing to run anything.
    """
    counter = root / COUNTER_FILENAME
    try:
        claimed = int(counter.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        present = [
            int(entry.name[:4])
            for entry in root.iterdir()
            if entry.is_dir() and len(entry.name) > 4 and entry.name[:4].isdigit()
        ]
        claimed = max(present, default=0) + 1
    counter.write_text(str(claimed + 1), encoding="utf-8")
    return claimed


def _slug(value: str) -> str:
    """Reduce an identifier to something safe and readable in a path.

    Args:
        value: A stake id, a label, or any other identifier.

    Returns:
        Lowercase words joined by hyphens. Stake ids are colon separated, and a colon in a
        path is both a portability problem and unreadable; the leading kind is dropped because
        the action and the argument are what a person recognises a run by, and the kind is
        implied by them. The full id is still in `run.json`, so nothing is lost.
    """
    body = value.split(":", 1)[1] if ":" in value else value
    cleaned = "".join(character if character.isalnum() else "-" for character in body.lower())
    return "-".join(part for part in cleaned.split("-") if part)


@dataclass
class Outcome:
    """What one conversation produced, or why it produced nothing.

    Attributes:
        attack: The attack that was run.
        transcript: The conversation, when it completed.
        findings: What the detectors concluded about it.
        error: The failure, when the conversation did not complete. A conversation that
            raised is kept as a row rather than dropped: an attack that failed and an attack
            that never ran are not the same result, and a report that shows only successes
            silently shrinks its own denominator.
        seconds: Wall clock for the whole conversation.
    """

    attack: Attack
    transcript: Transcript | None = None
    findings: tuple[Finding, ...] = ()
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the conversation ran to a natural end."""
        return self.transcript is not None

    @property
    def violations(self) -> tuple[Finding, ...]:
        """The findings that broke a declared rule."""
        return tuple(finding for finding in self.findings if finding.is_violation)


@dataclass
class SmokeRun:
    """Everything one invocation produced, for the report generator to read.

    Attributes:
        target: The registered target name.
        model: What composed the attack turns.
        stake: The stake the suite was filtered to, or empty for the whole suite.
        max_turns: Per-conversation budget.
        concurrency: How many conversations ran at once.
        started_at: UTC timestamp, for the report's header.
        seconds: Wall clock for the whole run.
        outcomes: One per attack, in suite sequence.
        recording: Path to the JSONL of every model call.
        run_id: The store's id for this run.
        number: The run's sequence number, taken from its directory name, so the report can
            title itself the same way the directory listing does.
    """

    target: str
    model: str
    stake: str
    max_turns: int
    concurrency: int
    started_at: str = ""
    seconds: float = 0.0
    outcomes: list[Outcome] = field(default_factory=list)
    recording: Path | None = None
    run_id: str = ""
    number: str = ""


def select(suite: tuple[Attack, ...], stake: str, limit: int) -> tuple[Attack, ...]:
    """Narrow a derived suite to the slice this run should execute.

    Args:
        suite: Everything the spec derives.
        stake: A stake id to keep, or empty to keep all of them.
        limit: Maximum attacks to keep after filtering. Zero keeps all.

    Returns:
        The attacks to run, in suite sequence.

    Raises:
        SystemExit: If the stake id matches nothing. A typo that silently ran a different
            slice would produce a report about a question nobody asked.
    """
    chosen = suite if not stake else tuple(a for a in suite if a.stake.id == stake)
    if not chosen:
        available = sorted({a.stake.id for a in suite})
        raise SystemExit(
            f"no stake {stake!r} on this target. Available:\n  " + "\n  ".join(available)
        )
    return chosen[:limit] if limit else chosen


def run_one(attack: Attack, attacker: Any, token: ConsentToken, max_turns: int) -> Outcome:
    """Execute one conversation, capturing a failure rather than propagating it.

    One attack failing must not end the run: the others are independent, and the failure is
    itself a result worth reporting.

    The conversation records whose it is, taken from the attack's own identity. Without it
    every scope check reports as never evaluated, which is honest and useless: the most
    sensitive check in the suite would silently never run.

    Args:
        attack: What is being run, carried onto the outcome.
        attacker: The composed attacker for it.
        token: Proof the target consented.
        max_turns: Per-conversation budget.

    Returns:
        The outcome, successful or not.
    """
    started = time.monotonic()
    subject = dict(attack.subject.identifiers) if attack.subject is not None else None
    try:
        transcript = run_conversation(token, attacker, max_turns=max_turns, subject=subject)
    except Exception as error:
        return Outcome(
            attack=attack,
            error=f"{type(error).__name__}: {error}",
            seconds=round(time.monotonic() - started, 2),
        )
    return Outcome(
        attack=attack, transcript=transcript, seconds=round(time.monotonic() - started, 2)
    )


def execute(
    attacks: tuple[Attack, ...],
    *,
    target: str,
    model: str,
    stake: str,
    max_turns: int,
    concurrency: int,
    recording: Path,
) -> SmokeRun:
    """Run every attack, concurrently, and grade each transcript.

    Consent is established once and the token shared: it is frozen, and `require_live()`
    only reads a clock, so every worker re-checks the same expiry independently.

    Detectors run on the main thread after the pool drains. They are pure and would be safe
    concurrently, but running them here keeps the concurrent section to the part that is
    actually slow, which is the two model calls per turn.

    Args:
        attacks: The slice to run.
        target: Registered target name.
        model: First-party model id for the attacker.
        stake: The stake id this run was filtered to, recorded on the result.
        max_turns: Per-conversation budget.
        concurrency: Conversations in flight at once.
        recording: Where every model call is written.

    Returns:
        The completed run, ready to report on.
    """
    spec = load_spec_dir(load_registry().resolve(target).spec_dir)
    token = establish_consent(target)
    inner = AnthropicModelClient(model=model)
    recorder = CallRecorder(recording, label=target)

    run = SmokeRun(
        target=target,
        model=model,
        stake=stake,
        max_turns=max_turns,
        concurrency=concurrency,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        recording=recording,
    )
    started = time.monotonic()

    attackers = [
        build_attackers(
            (attack,),
            RecordingModelClient(inner, recorder, label=attack.id),
            max_turns=max_turns,
        )[0]
        for attack in attacks
    ]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_one, attack, attacker, token, max_turns)
            for attack, attacker in zip(attacks, attackers, strict=True)
        ]
        run.outcomes = [future.result() for future in futures]

    run.seconds = round(time.monotonic() - started, 2)

    for outcome in run.outcomes:
        if outcome.transcript is not None:
            outcome.findings = run_detectors(spec, outcome.transcript)
    return run


def persist(run: SmokeRun, store_path: Path) -> None:
    """Write every completed transcript to SQLite, from this thread only.

    `Store` holds one connection with `check_same_thread` on, so writing from the workers
    would raise. Nothing is lost by waiting: the writes are microseconds next to the model
    calls that produced them.

    Args:
        run: The completed run. Its `run_id` is filled in.
        store_path: Where the database lives.
    """
    completed = [o for o in run.outcomes if o.transcript is not None]
    if not completed:
        return
    spec = load_spec_dir(load_registry().resolve(run.target).spec_dir)
    with Store(store_path) as store:
        run.run_id = store.create_run(
            run.target,
            spec.version_tuple,
            notes=f"smoke run, stake={run.stake or 'all'}, model={run.model}",
        )
        for outcome in completed:
            assert outcome.transcript is not None
            store.save_transcript(run.run_id, outcome.transcript, attack_id=outcome.attack.id)
        store.finish_run(run.run_id)


def to_json(run: SmokeRun) -> dict[str, Any]:
    """Flatten a run into the shape the report generator reads.

    Written to disk rather than passed in memory, so the report can be regenerated from a
    finished run without running it again. That matters more than it sounds: iterating on
    the report is the cheap part and re-running the conversations is not.

    Args:
        run: The completed run.

    Returns:
        A JSON-serialisable dictionary.
    """
    return {
        "target": run.target,
        "model": run.model,
        "stake": run.stake,
        "max_turns": run.max_turns,
        "concurrency": run.concurrency,
        "started_at": run.started_at,
        "seconds": run.seconds,
        "run_id": run.run_id,
        "number": run.number,
        "recording": str(run.recording) if run.recording else "",
        "outcomes": [
            {
                "attack_id": o.attack.id,
                "technique": o.attack.technique.name,
                "technique_id": o.attack.technique.id,
                "stake_id": o.attack.stake.id,
                "stake_kind": str(o.attack.stake.kind),
                "consequence": str(o.attack.stake.consequence),
                "settled_by": str(o.attack.stake.settled_by),
                "goal": o.attack.goal,
                "subject": None
                if o.attack.subject is None
                else {
                    "name": o.attack.subject.name,
                    "identifiers": dict(o.attack.subject.identifiers),
                    "facts": list(o.attack.subject.facts),
                },
                "seconds": o.seconds,
                "error": o.error,
                "transcript": None
                if o.transcript is None
                else {
                    "session": o.transcript.session,
                    "goal": o.transcript.goal,
                    "subject": o.transcript.subject,
                    "stopped_because": o.transcript.stopped_because,
                    "spec_versions": o.transcript.spec_versions,
                    "turns": [
                        {
                            "index": t.index,
                            "user": t.user,
                            "reply": t.reply,
                            "latency_seconds": t.latency_seconds,
                            "agent_usage": dict(t.agent_usage),
                            "tool_calls": [
                                {"name": c.name, "arguments": c.arguments, "result": c.result}
                                for c in t.tool_calls
                            ],
                        }
                        for t in o.transcript.turns
                    ],
                },
                "findings": [f.model_dump(mode="json") for f in o.findings],
            }
            for o in run.outcomes
        ],
    }


def summarise(run: SmokeRun) -> str:
    """One block of text saying what happened, for the terminal."""
    lines = [
        "",
        f"target      {run.target}",
        f"attacker    {run.model}",
        f"stake       {run.stake or 'all'}",
        f"ran         {len(run.outcomes)} conversation(s), {run.concurrency} at a time, "
        f"{run.seconds}s wall clock",
        "",
    ]
    for outcome in run.outcomes:
        if not outcome.ok:
            lines.append(f"  FAILED  {outcome.attack.technique.name}: {outcome.error}")
            continue
        assert outcome.transcript is not None
        broke = len(outcome.violations)
        evaluated = [f for f in outcome.findings if f.outcome != JudgeOutcome.NOT_EVALUATED]
        if broke:
            mark = f"BROKE x{broke}"
        elif evaluated:
            mark = "held    "
        else:
            mark = "UNTESTED"
        lines.append(
            f"  {mark}  {outcome.attack.technique.name}: "
            f"{len(outcome.transcript.turns)} turns, {outcome.seconds}s, "
            f"stopped because {outcome.transcript.stopped_because or 'budget spent'}"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the slice, write the artefacts.

    Args:
        argv: Command line arguments. Defaults to `sys.argv[1:]`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="dispute_handler", help="Registered target name.")
    parser.add_argument("--stake", default="", help="Stake id to run. Empty runs the whole suite.")
    parser.add_argument("--limit", type=int, default=0, help="Cap attacks after filtering.")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=SMOKE_MODEL, help="First-party id for the attacker.")
    parser.add_argument("--out", type=Path, default=None, help="Force this run's directory.")
    parser.add_argument("--label", default="", help="Human tag appended to the run's name.")
    parser.add_argument("--store", type=Path, default=Path("data/agentred.db"))
    parser.add_argument("--list-stakes", action="store_true", help="Print the stakes and exit.")
    arguments = parser.parse_args(argv)

    spec = load_spec_dir(load_registry().resolve(arguments.target).spec_dir)

    if arguments.list_stakes:
        for stake in derive_stakes(spec):
            print(f"{stake.id}\n    {stake.consequence} | settled by {stake.settled_by}")
        return

    attacks = select(build_suite(spec), arguments.stake, arguments.limit)
    if arguments.out is not None:
        out = arguments.out
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = next_run_dir(arguments.target, arguments.stake, arguments.label)

    print(f"running {len(attacks)} attack(s) against {arguments.target}, writing to {out}")
    run = execute(
        attacks,
        target=arguments.target,
        model=arguments.model,
        stake=arguments.stake,
        max_turns=arguments.max_turns,
        concurrency=arguments.concurrency,
        recording=out / "calls.jsonl",
    )
    run.number = out.name[:4] if out.name[:4].isdigit() else ""
    persist(run, arguments.store)
    (out / "run.json").write_text(json.dumps(to_json(run), indent=2), encoding="utf-8")

    print(summarise(run))
    print(f"run.json     {out / 'run.json'}")
    print(f"calls.jsonl  {out / 'calls.jsonl'}")
    print(f"store run id {run.run_id or 'nothing persisted'}")


if __name__ == "__main__":
    sys.exit(main())
