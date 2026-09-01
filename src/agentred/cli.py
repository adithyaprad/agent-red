"""The command line.

Four commands, and the order they are listed in is the order an operator meets them.

`doctor` answers "is this installation able to run anything" before an operator spends an
hour and a hundred dollars finding out that it is not. It checks the model route, the
registry, and every registered target's consent handshake, and it reports what it found
rather than raising on the first problem, because an operator fixing three things wants to
see all three.

`run` is the whole thing: conversations against a served target, every check over what they
produced, and the page their operator reads. It is three stages behind one command because
the seams between them are an implementation detail nobody running this should have to know
about, and a chain a person joins up by hand is a chain a person forgets a link of.

`analyse` and `report` are those later stages on their own. They exist because the expensive
stage is the first one and it is already durable: conversations are written to the store
before anything is judged, so an analysis that dies on a rate limit is resumed with
`analyse --run`, and the page is rebuilt from a finished analysis for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from agentred.attacks.generator import build_suite
from agentred.attacks.stakes import derive_stakes
from agentred.llm import LLMConfigurationError, resolve_route
from agentred.llm.client import AnthropicModelClient
from agentred.runner.consent import (
    ConsentError,
    RegisteredTarget,
    TargetRegistry,
    establish_consent,
    load_registry,
)
from agentred.runner.suite import (
    DEFAULT_ATTACKER_MODEL,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_TURNS,
    SuiteRun,
    execute,
    next_run_dir,
    persist,
    select,
    summarise,
    to_json,
)
from agentred.scoring.analysis import AnalysisError, analyse, known_runs, resolve_runs
from agentred.scoring.render import build
from agentred.spec import SpecError, load_spec_dir
from agentred.store.repo import Store

DEFAULT_STORE = Path("data/agentred.db")

app = typer.Typer(help="Adversarial red-teaming harness for merchant-facing commerce agents.")


@dataclass
class Check:
    """One thing that was checked.

    Attributes:
        name: What was checked, in the operator's terms.
        ok: Whether it passed.
        detail: What was found. Present whether or not it passed.
    """

    name: str
    ok: bool
    detail: str

    def render(self) -> str:
        """One line, marked so a failure is visible in a wall of output."""
        mark = "ok  " if self.ok else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail}"


def check_route() -> Check:
    """Whether a model route resolves from the environment."""
    try:
        route = resolve_route()
    except LLMConfigurationError as error:
        return Check("model route", False, str(error))
    return Check("model route", True, f"{route.value}")


def check_registry(path: Path | None) -> tuple[Check, TargetRegistry | None]:
    """Whether the target registry loads, and what it holds."""
    try:
        registry = load_registry(path)
    except ConsentError as error:
        return Check("target registry", False, str(error)), None
    return (
        Check(
            "target registry",
            True,
            f"{len(registry.targets)} target(s): {', '.join(registry.names)}",
        ),
        registry,
    )


def check_spec(target: RegisteredTarget) -> Check:
    """Whether the target's spec loads and its policy describes its config."""
    try:
        spec = load_spec_dir(target.spec_dir)
    except SpecError as error:
        return Check(f"{target.name} spec", False, str(error).splitlines()[0])
    ungated = [tool.name for tool in spec.ungated_consequential_tools()]
    detail = f"{spec.version_tuple}"
    if ungated:
        detail += f"; consequential tools with no declared limit: {', '.join(ungated)}"
    return Check(f"{target.name} spec", True, detail)


def check_consent(name: str, registry: TargetRegistry) -> Check:
    """Whether the target is up and will echo a challenge.

    A failure here is the expected state for a target that is not running, so the message
    says what to start rather than only what went wrong.
    """
    try:
        establish_consent(name, registry=registry)
    except ConsentError as error:
        return Check(f"{name} consent", False, str(error))
    return Check(f"{name} consent", True, "challenge echoed, test mode confirmed")


@app.callback()
def main() -> None:
    """Group the commands.

    Present so `doctor` stays a subcommand rather than becoming the whole program, which is
    what typer does with a single command and would break every documented invocation the
    moment the second command lands.
    """


@app.command()
def doctor(
    registry_path: Annotated[
        Path | None,
        typer.Option("--registry", help="Path to targets.registry.yaml. Defaults to the root."),
    ] = None,
    skip_consent: Annotated[
        bool,
        typer.Option("--skip-consent", help="Skip the live handshake, for checking config only."),
    ] = False,
) -> None:
    """Check that this installation can run a suite, and say what is wrong if it cannot."""
    checks = [check_route()]
    registry_check, registry = check_registry(registry_path)
    checks.append(registry_check)

    if registry is not None:
        for target in registry.targets:
            checks.append(check_spec(target))
            if not skip_consent:
                checks.append(check_consent(target.name, registry))

    typer.echo("agent-red doctor")
    for check in checks:
        typer.echo(check.render())

    failures = [check for check in checks if not check.ok]
    typer.echo("")
    if failures:
        typer.echo(f"{len(failures)} of {len(checks)} checks failed.")
        raise typer.Exit(code=1)
    typer.echo(f"All {len(checks)} checks passed.")


ANALYSIS_FILENAME = "analysis.json"
REPORT_FILENAME = "what-broke.html"


def write_analysis(
    store_path: Path, run_ids: tuple[str, ...], model: str, out: Path
) -> dict[str, object]:
    """Run every post-conversation check and write the result beside the run.

    Args:
        store_path: Where the database lives.
        run_ids: Which runs to analyse. Empty analyses every run in the store.
        model: Model id for extraction, judging and the two comparison stages.
        out: Where the analysis file goes.

    Returns:
        The analysis, so a caller can render it without reading the file back.

    The run selection is resolved before a model client is built, so a mistyped run id costs
    nothing and needs no credentials to find. `analyse` resolves it again, which is a cheap
    query and keeps it correct when called from anywhere else.

    Raises:
        typer.Exit: If a named run is not in the store, or no model route is configured.
    """
    store = Store(store_path)
    try:
        resolve_runs(store, run_ids)
        client = AnthropicModelClient(model=model)
    except (AnalysisError, LLMConfigurationError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    result = analyse(store, client, runs=run_ids, say=typer.echo)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@app.command()
def run(
    target: Annotated[str, typer.Option("--target", help="Registered target name.")],
    stake: Annotated[
        list[str] | None,
        typer.Option("--stake", help="Stake id to attack. Repeatable. Omit for the whole suite."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap attacks after filtering.")] = 0,
    max_turns: Annotated[int, typer.Option("--max-turns")] = DEFAULT_MAX_TURNS,
    concurrency: Annotated[int, typer.Option("--concurrency")] = DEFAULT_CONCURRENCY,
    model: Annotated[
        str, typer.Option("--model", help="Model id for the attacker and the judge.")
    ] = DEFAULT_ATTACKER_MODEL,
    label: Annotated[str, typer.Option("--label", help="Human tag on the run directory.")] = "",
    store_path: Annotated[Path, typer.Option("--store")] = DEFAULT_STORE,
    out: Annotated[Path | None, typer.Option("--out", help="Force the run directory.")] = None,
    list_stakes: Annotated[
        bool, typer.Option("--list-stakes", help="Print what this target derives, and stop.")
    ] = False,
    conversations_only: Annotated[
        bool,
        typer.Option(
            "--conversations-only",
            help="Hold the conversations and stop, leaving the analysis for later.",
        ),
    ] = False,
) -> None:
    """Attack a target, check what came back, and write the page its operator reads.

    Three stages, one command: conversations, analysis, page. Everything lands in one
    numbered run directory outside the repository, and the analysis covers that run alone
    rather than whatever else the store happens to hold.
    """
    spec = load_spec_dir(load_registry().resolve(target).spec_dir)
    if list_stakes:
        for derived in derive_stakes(spec):
            typer.echo(derived.id)
        return

    stakes = tuple(stake or ())
    attacks = select(build_suite(spec), stakes, limit)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        directory = out
    else:
        directory = next_run_dir(target, stakes, label)

    typer.echo(f"running {len(attacks)} attack(s) against {target}, writing to {directory}")
    interrupted: BaseException | None = None
    try:
        completed = execute(
            attacks,
            target=target,
            model=model,
            stake=", ".join(stakes),
            max_turns=max_turns,
            concurrency=concurrency,
            recording=directory / "calls.jsonl",
            store_path=store_path,
            number=directory.name[:4] if directory.name[:4].isdigit() else "",
        )
    except KeyboardInterrupt as stop:
        # Whatever finished is already in the store, transcript by transcript. Rebuild a run
        # around it so the directory describes what happened rather than nothing, then stop.
        # A suite is hours long and the alternative is losing every completed conversation to
        # one interruption, which is how run 0006 was lost.
        completed = stop.args[0] if stop.args and isinstance(stop.args[0], SuiteRun) else None
        if completed is None:
            raise
        interrupted = stop

    persist(completed, store_path)
    (directory / "run.json").write_text(json.dumps(to_json(completed), indent=2), encoding="utf-8")
    typer.echo(summarise(completed))
    if interrupted is not None:
        typer.echo("")
        typer.echo(
            f"INTERRUPTED. {len(completed.outcomes)} of {len(attacks)} conversation(s) are "
            f"recorded and analysable; the rest were never run."
        )
        typer.echo(f"run.json     {directory / 'run.json'}")
        typer.echo(f"store run id {completed.run_id}")
        raise typer.Exit(code=130)

    if conversations_only or not completed.run_id:
        typer.echo(f"run.json     {directory / 'run.json'}")
        typer.echo(f"store run id {completed.run_id or 'nothing persisted'}")
        return

    typer.echo("")
    analysis_path = directory / ANALYSIS_FILENAME
    result = write_analysis(store_path, (completed.run_id,), model, analysis_path)
    report_path = directory / REPORT_FILENAME
    report_path.write_text(build(result), encoding="utf-8")

    typer.echo("")
    typer.echo(f"run.json     {directory / 'run.json'}")
    typer.echo(f"analysis     {analysis_path}")
    typer.echo(f"report       {report_path}")
    typer.echo(f"store run id {completed.run_id}")


@app.command(name="analyse")
def analyse_command(
    run_id: Annotated[
        list[str] | None,
        typer.Option("--run", help="Run id to analyse. Repeatable. Omit for every run."),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Where the analysis goes.")] = None,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_ATTACKER_MODEL,
    store_path: Annotated[Path, typer.Option("--store")] = DEFAULT_STORE,
    report: Annotated[
        Path | None, typer.Option("--report", help="Also render the page here.")
    ] = None,
    list_runs: Annotated[
        bool, typer.Option("--list-runs", help="Print the runs in the store, and stop.")
    ] = False,
) -> None:
    """Run every check over conversations already recorded, contacting no target.

    Omitting `--run` analyses the whole store, which is right while it holds one run and
    wrong the moment it holds two: the page would then pool conversations held against
    different versions of an agent under one denominator.
    """
    store = Store(store_path)
    if list_runs:
        for row in known_runs(store):
            count = len(store.conversation_ids(row["run_id"]))
            typer.echo(
                f"{row['run_id']}  {row['target']:<18} {count:>3} conversations  {row['notes']}"
            )
        return
    if out is None:
        typer.echo("--out is required unless --list-runs is given")
        raise typer.Exit(code=1)

    result = write_analysis(store_path, tuple(run_id or ()), model, out)
    typer.echo(f"analysis  {out}")
    if report is not None:
        report.write_text(build(result), encoding="utf-8")
        typer.echo(f"report    {report}")


@app.command()
def report(
    analysis: Annotated[Path, typer.Argument(help="A finished analysis file.")],
    out: Annotated[Path, typer.Option("--out", help="Where the page goes.")],
) -> None:
    """Render the operator page from a finished analysis. Calls nothing and costs nothing."""
    out.write_text(build(json.loads(analysis.read_text(encoding="utf-8"))), encoding="utf-8")
    typer.echo(f"report  {out}")


if __name__ == "__main__":
    app()
