"""The command line.

`doctor` is the only command that exists yet, and it is the one worth having first: it
answers "is this installation able to run anything" before an operator spends an hour and a
hundred dollars finding out that it is not. It checks the model route, the registry, and
every registered target's consent handshake, and it reports what it found rather than
raising on the first problem, because an operator fixing three things wants to see all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from agentred.llm import LLMConfigurationError, resolve_route
from agentred.runner.consent import (
    ConsentError,
    RegisteredTarget,
    TargetRegistry,
    establish_consent,
    load_registry,
)
from agentred.spec import SpecError, load_spec_dir

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


if __name__ == "__main__":
    app()
