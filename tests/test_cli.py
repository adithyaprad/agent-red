"""`agentred doctor` reports what is wrong instead of stopping at the first thing."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentred.cli import app

runner = CliRunner()
REGISTRY = Path(__file__).resolve().parents[1] / "targets.registry.yaml"


def run(*arguments: str, env: dict[str, str] | None = None) -> object:
    return runner.invoke(app, list(arguments), env=env or {})


def test_doctor_checks_the_shipped_registry_and_specs_offline() -> None:
    result = run(
        "doctor",
        "--skip-consent",
        "--registry",
        str(REGISTRY),
        env={"ANTHROPIC_API_KEY": "not-a-real-key"},
    )
    assert result.exit_code == 0, result.output
    assert "cart_recovery spec" in result.output
    assert "dispute_handler spec" in result.output
    assert "model route: first_party" in result.output


def test_doctor_names_consequential_tools_with_no_declared_limit() -> None:
    result = run(
        "doctor",
        "--skip-consent",
        "--registry",
        str(REGISTRY),
        env={"ANTHROPIC_API_KEY": "not-a-real-key"},
    )
    assert "promise_delivery" in result.output


def test_doctor_fails_when_no_model_route_resolves() -> None:
    result = run("doctor", "--skip-consent", "--registry", str(REGISTRY))
    assert result.exit_code == 1
    assert "model route" in result.output
    assert "checks failed" in result.output


def test_doctor_fails_on_a_registry_that_is_not_there(tmp_path: Path) -> None:
    result = run(
        "doctor",
        "--skip-consent",
        "--registry",
        str(tmp_path / "nothing.yaml"),
        env={"ANTHROPIC_API_KEY": "not-a-real-key"},
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_doctor_reports_every_failure_not_only_the_first(tmp_path: Path) -> None:
    result = run("doctor", "--skip-consent", "--registry", str(tmp_path / "nothing.yaml"))
    assert result.output.count("[FAIL]") == 2
