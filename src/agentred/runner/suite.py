"""Running a slice of a derived suite against a served target, and recording everything.

This is the conversation half of a run: consent, the attacks the target's own spec derives,
the turns a model composes against what the agent says back, and the transcripts that come
out. Nothing here judges anything beyond what the deterministic detectors settle from the
tool-call log, and nothing here writes a page.

It lives in the package rather than in `scripts/` because `agentred run` chains it to the
analysis and the report, and a command that shells out to a script in the working directory
is not a command anybody can install. `scripts/smoke.py` is still the way to run this half on
its own.

Conversations run concurrently because their worlds are isolated: `TargetAgent.session()`
gives each session id a fresh world, and the driver mints a new id per conversation. The
store is not thread-safe (one connection, `check_same_thread`), so transcripts come back to
the main thread to be written. Persistence was never the slow part.
"""

from __future__ import annotations

import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentred.attacks.generator import Attack, build_attackers, build_planters
from agentred.judge.detectors import run_detectors
from agentred.judge.models import Finding
from agentred.judge.models import Outcome as JudgeOutcome
from agentred.llm.client import AnthropicModelClient
from agentred.llm.recording import CallRecorder, RecordingModelClient
from agentred.mcp.control import ControlError, HttpxArenaControl
from agentred.runner.channels.conversational import Transcript, run_conversation
from agentred.runner.channels.planted import run_planted
from agentred.runner.consent import ConsentLease, load_registry
from agentred.spec import load_spec_dir
from agentred.spec.models import AgentSpec, ChannelDeclaration, VersionTuple
from agentred.store.repo import Store

DEFAULT_ATTACKER_MODEL = "claude-sonnet-5"
"""What composes the attack turns, unless a caller says otherwise.

Sonnet rather than Opus, matching what the targets themselves run. It makes the result
asymmetric and that is understood: persuasive turns from Sonnet mean Opus would also manage
it, while polite ones leave weak-model and weak-prompt indistinguishable. Worth it for runs
that have to be cheap enough to repeat.
"""

DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_TURNS = 6

RUNS_DIR_ENV_VAR = "AGENTRED_RUNS_DIR"

DEFAULT_RUNS_ROOT = Path("out") / "runs"
"""Where runs are kept unless the environment says otherwise.

A run holds complete transcripts of an agent being manipulated, the full prompts that did it,
and what it cost. None of that belongs in a repository someone else will read, so the default
is a directory the repository ignores rather than one it tracks: a run written into `data/` is
one `git add .` away from being published. `AGENTRED_RUNS_DIR` moves them anywhere, which is
what an operator keeping evidence outside a checkout sets.
"""


def runs_root(env: dict[str, str] | None = None) -> Path:
    """Where this installation keeps its runs.

    Read per call rather than at import, so that a process serving several runs picks up a
    change and so a test can point one somewhere without reloading the module.

    Args:
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        `AGENTRED_RUNS_DIR` if it is set to anything but whitespace, else `DEFAULT_RUNS_ROOT`.
    """
    env = os.environ if env is None else env
    return (
        Path(configured)
        if (configured := env.get(RUNS_DIR_ENV_VAR, "").strip())
        else DEFAULT_RUNS_ROOT
    )


def next_run_dir(
    target: str, stakes: tuple[str, ...] = (), label: str = "", root: Path | None = None
) -> Path:
    """Allocate the next numbered directory for a run.

    Numbered rather than timestamped, because the question anyone actually asks of a run is
    which one came before it and which one came after. A timestamp answers when, in a format
    nobody can order at a glance, and two runs a minute apart sort by a string of digits that
    means nothing. The name then says what the run was, so a directory listing reads as a
    history rather than as a pile.

    Args:
        target: The registered target name.
        stakes: The stake ids the run was filtered to, or empty for the whole suite.
        label: Optional human tag appended to the name.
        root: Where runs live. Defaults to `runs_root()`.

    Returns:
        The created directory, `NNNN-<target>-<what was attacked>[-<label>]`. One stake is
        named; several are counted, because five stake ids joined together make a path no
        one can read and some filesystems will not take. The full list is in `run.json`.
    """
    root = runs_root() if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    if not stakes:
        described = "full-suite"
    elif len(stakes) == 1:
        described = _slug(stakes[0])
    else:
        described = f"{len(stakes)}-stakes"
    parts = [f"{_claim_number(root):04d}", target, described]
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


class SuiteError(RuntimeError):
    """An attack could not be run as specified.

    Distinct from a target that misbehaved and from a target that broke. This is the harness
    being asked to do something incoherent, such as attacking a channel the agent does not
    declare, and it is captured onto the outcome as an error rather than being reported as an
    attack that held.
    """


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
class SuiteRun:
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
        consents: How many separate times the target echoed a challenge during this run. One
            for a short suite, more for a suite that outlasted the consent window.
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
    consents: int = 1


def select(suite: tuple[Attack, ...], stakes: tuple[str, ...], limit: int) -> tuple[Attack, ...]:
    """Narrow a derived suite to the slice this run should execute.

    Args:
        suite: Everything the spec derives.
        stakes: Stake ids to keep, or empty to keep all of them.
        limit: Maximum attacks to keep after filtering. Zero keeps all.

    Returns:
        The attacks to run, in suite sequence.

    Raises:
        SystemExit: If any stake id matches nothing. Every id is checked, not just the first
            to fail, and a typo is fatal rather than dropped: a run that quietly covered four
            of five stakes still divides by a denominator that counts five.
    """
    available = sorted({attack.stake.id for attack in suite})
    unknown = [stake for stake in stakes if stake not in available]
    if unknown:
        raise SystemExit(
            f"no stake {', '.join(repr(stake) for stake in unknown)} on this target. "
            "Available:\n  " + "\n  ".join(available)
        )
    chosen = suite if not stakes else tuple(a for a in suite if a.stake.id in stakes)
    return chosen[:limit] if limit else chosen


def run_one(
    attack: Attack,
    attacker: Any,
    lease: ConsentLease,
    max_turns: int,
    run: str,
    channels: dict[str, ChannelDeclaration] | None = None,
    subject_kinds: tuple[str, ...] = (),
) -> Outcome:
    """Execute one attack down the channel it declares.

    A failure is captured onto the outcome rather than propagated.
    One attack failing must not end the run: the others are independent, and the failure is
    itself a result worth reporting.

    Dispatch is on the attack's channel and on nothing else. A conversational attack is
    driven turn by turn; a planted one restores the world, writes its payload into the field
    the agent declared, and fires the agent's real trigger (ADR-0006).

    The attempt records whose it is, taken from the attack's own identity. Without it every
    scope check reports as never evaluated, which is honest and useless: the most sensitive
    check in the suite would silently never run.

    Args:
        attack: What is being run, carried onto the outcome.
        attacker: What composes this attack's writing: turn by turn against what the agent
            said, or once, into the field it plants. Both are model-backed, and a planted
            attempt costs one call before anything is written.
        lease: Asked for a token as this attempt starts, so a suite longer than the
            consent window renews rather than being refused by its own gate.
        max_turns: Per-conversation budget. Not applicable to a planted attempt, which fires
            once.
        run: The run the tool server records this attempt's calls under, and the runner
            reads them back under.
        channels: The channels the target declares, keyed by name. Required for a planted
            attack and unused otherwise.
        subject_kinds: The identifier kinds the agent's data scope binds a record by. A
            scheduled firing uses them to read the cohort it was woken about, so that the
            other records in the batch are not scored as strangers.

    Returns:
        The outcome, successful or not.
    """
    started = time.monotonic()
    subject = dict(attack.subject.identifiers) if attack.subject is not None else None
    try:
        token = lease.token()
        if attack.is_planted:
            declared = (channels or {}).get(attack.channel)
            if declared is None:
                raise SuiteError(
                    f"attack {attack.id!r} arrives down channel {attack.channel!r}, which "
                    f"this agent does not declare. Attacking a channel the deployment does "
                    f"not have would report a finding about the harness."
                )
            transcript = run_planted(
                token,
                declared,
                attacker.compose(),
                run=run,
                record_id=attack.record_id,
                goal=attack.goal,
                subject=subject,
                subject_kinds=subject_kinds,
            )
        else:
            transcript = run_conversation(
                token, attacker, run=run, max_turns=max_turns, subject=subject
            )
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
    store_path: Path | None = None,
    number: str = "",
) -> SuiteRun:
    """Run every attack, concurrently, and grade each transcript.

    Consent is held as a lease rather than a single token. Each conversation asks for one
    as it starts, and the lease establishes consent again when what it holds is close to
    expiring, so a suite that outlasts the consent window keeps consenting instead of
    losing its tail to its own gate. How many times that happened is reported.

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
        store_path: Where to persist each transcript as it lands. Omit to persist nothing,
            which is what the offline tests do.
        number: The run's sequence number, needed before the first write because the note
            stored beside the transcripts cites it.

    Returns:
        The run. Complete if it was allowed to finish, and holding whatever completed if it
        was not.
    """
    registered = load_registry().resolve(target)
    spec = load_spec_dir(registered.spec_dir)
    lease = ConsentLease(target)
    inner = AnthropicModelClient(model=model)
    recorder = CallRecorder(recording, label=target)

    run = SuiteRun(
        target=target,
        model=model,
        stake=stake,
        max_turns=max_turns,
        concurrency=concurrency,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        recording=recording,
    )
    started = time.monotonic()
    order = {attack.id: index for index, attack in enumerate(attacks)}

    channels = spec.config.channels_by_name
    # One composer per attack, of whichever kind the channel calls for. A planted attack
    # writes one string rather than a conversation, so it is cheaper by roughly the turn
    # count and is not free: the payload is composed by the same model, against the same
    # technique, and each one is labelled with the attack it belongs to so the bill can be
    # read per cell.
    attackers = [
        build_planters((attack,), RecordingModelClient(inner, recorder, label=attack.id), channels)[
            0
        ]
        if attack.is_planted
        else build_attackers(
            (attack,),
            RecordingModelClient(inner, recorder, label=attack.id),
            max_turns=max_turns,
        )[0]
        for attack in attacks
    ]

    run.number = number
    store = Store(store_path) if store_path is not None else None
    if store is not None:
        run.run_id = store.create_run(
            run.target,
            versions_for(spec, registered.control_url),
            notes=describe(run),
        )
    # What the tool server files this run's calls under. The store's id when there is a
    # store, so a transcript and the record behind it carry the same name; a minted one
    # otherwise, because two runs sharing a key would read each other's evidence.
    record_run = run.run_id or f"unstored-{secrets.token_hex(6)}"

    def settle() -> None:
        """Put the run in a reportable state, whether or not every attack ran."""
        run.outcomes.sort(key=lambda o: order.get(o.attack.id, 0))
        run.seconds = round(time.monotonic() - started, 2)
        run.consents = len(lease.nonces)
        for outcome in run.outcomes:
            if outcome.transcript is not None:
                outcome.findings = run_detectors(spec, outcome.transcript)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    run_one,
                    attack,
                    attacker,
                    lease,
                    max_turns,
                    record_run,
                    channels,
                    spec.policy.data_scope.subject_identifier_kinds,
                ): attack
                for attack, attacker in zip(attacks, attackers, strict=True)
            }
            for future in as_completed(futures):
                outcome = future.result()
                run.outcomes.append(outcome)
                if store is not None and outcome.transcript is not None:
                    store.save_transcript(
                        run.run_id, outcome.transcript, attack_id=outcome.attack.id
                    )
    except KeyboardInterrupt as stop:
        settle()
        raise KeyboardInterrupt(run) from stop
    finally:
        if store is not None:
            if run.run_id:
                store.finish_run(run.run_id)
            store.close()

    settle()
    return run


def describe(run: SuiteRun) -> str:
    """One line naming the run, written into the store beside its transcripts.

    A run's number lives in its directory name, and that directory sits outside the
    repository. So a database opened on its own could say which target and which model
    produced a transcript but not which run of them, while everything that cites a run cites
    its number: the engineering log, the report's header, whatever anybody wrote down. The
    number is recorded where the transcripts are so the citation can be resolved.

    Args:
        run: The completed run, with its number already claimed.

    Returns:
        The note stored on the run row.
    """
    number = f"run {run.number}" if run.number else "unnumbered run"
    return f"{number}, stake={run.stake or 'all'}, attacker={run.model}"


def persist(run: SuiteRun, store_path: Path) -> None:
    """Write every completed transcript to SQLite, from this thread only.

    A no-op when `execute` already persisted as it went, which is the normal path. It stays
    here for callers that assembled a run some other way.

    `Store` holds one connection with `check_same_thread` on, so writing from the workers
    would raise. Nothing is lost by waiting: the writes are microseconds next to the model
    calls that produced them.

    Args:
        run: The completed run. Its `run_id` is filled in.
        store_path: Where the database lives.
    """
    if run.run_id:
        return
    completed = [o for o in run.outcomes if o.transcript is not None]
    if not completed:
        return
    spec = load_spec_dir(load_registry().resolve(run.target).spec_dir)
    with Store(store_path) as store:
        registered = load_registry().resolve(run.target)
        run.run_id = store.create_run(
            run.target,
            versions_for(spec, registered.control_url),
            notes=describe(run),
        )
        for outcome in completed:
            assert outcome.transcript is not None
            store.save_transcript(run.run_id, outcome.transcript, attack_id=outcome.attack.id)
        store.finish_run(run.run_id)


def versions_for(spec: AgentSpec, control_url: str) -> VersionTuple:
    """The five versions a run's results will be valid for.

    Four come from the declaration. The fifth is the shop, and it comes from the tool server
    because that is the process holding one: a world is not a property of a declaration, and a
    scorecard computed against one shop says nothing about an agent facing another (ADR-0007).
    It is also what stops the quieter version of that: the day the shop was rebuilt, every
    earlier scorecard went on citing a tuple that no longer described what the agent had faced.

    Args:
        spec: The loaded spec, supplying the four declared versions.
        control_url: Origin of the tool server's control face. Empty skips the read.

    Returns:
        The tuple. A server that cannot be reached, or that reports no world, leaves the
        element empty rather than having a value guessed for it, so a stored result reads as
        what it was rather than as a world nobody named.
    """
    if not control_url:
        return spec.version_tuple
    try:
        reported = HttpxArenaControl(control_url).health().get("versions") or {}
    except ControlError:
        return spec.version_tuple
    world = str((reported.get(spec.config.agent_id) or {}).get("world_version", ""))
    return (
        spec.version_tuple.model_copy(update={"world_version": world})
        if world
        else (spec.version_tuple)
    )


def to_json(run: SuiteRun) -> dict[str, Any]:
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
        "consents": run.consents,
        "recording": str(run.recording) if run.recording else "",
        "outcomes": [
            {
                "attack_id": o.attack.id,
                "technique": o.attack.move_name,
                "technique_id": o.attack.move,
                "channel": o.attack.channel,
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
                    "channel": o.transcript.channel,
                    "planted": [asdict(p) for p in o.transcript.planted],
                    "cohort": [dict(entry) for entry in o.transcript.cohort],
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


def summarise(run: SuiteRun) -> str:
    """One block of text saying what happened, for the terminal."""
    lines = [
        "",
        f"target      {run.target}",
        f"attacker    {run.model}",
        f"stake       {run.stake or 'all'}",
        f"ran         {len(run.outcomes)} conversation(s), {run.concurrency} at a time, "
        f"{run.seconds}s wall clock",
        f"consent     echoed {run.consents} time(s) during the run",
        "",
    ]
    for outcome in run.outcomes:
        if not outcome.ok:
            lines.append(f"  FAILED  {outcome.attack.move_name}: {outcome.error}")
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
            f"  {mark}  {outcome.attack.move_name}: "
            f"{len(outcome.transcript.turns)} turns, {outcome.seconds}s, "
            f"stopped because {outcome.transcript.stopped_because or 'budget spent'}"
        )
    lines.append("")
    return "\n".join(lines)
