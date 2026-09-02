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


def test_doctor_refuses_a_tool_server_holding_a_stale_spec() -> None:
    """D24: it loaded the spec once, so what is on disk is not what it is serving."""
    from agentred.cli import _stale_versions
    from agentred.runner.consent import RegisteredTarget

    target = RegisteredTarget(
        name="dispute_handler",
        agent_id="dispute_handler",
        base_url="http://localhost:8082",
        spec_dir=Path("src/agentred/targets/specs/dispute_handler"),
    )
    health = {
        "agents": ["dispute_handler"],
        "versions": {"dispute_handler": {"policy_version": "0.1"}},
    }
    stale = _stale_versions(target, health)
    assert "policy_version: serving '0.1'" in stale
    assert "config_version: serving '(absent)'" in stale


def test_doctor_says_nothing_about_versions_a_server_does_not_report() -> None:
    from agentred.cli import _stale_versions
    from agentred.runner.consent import RegisteredTarget

    target = RegisteredTarget(
        name="dispute_handler",
        agent_id="dispute_handler",
        base_url="http://localhost:8082",
        spec_dir=Path("src/agentred/targets/specs/dispute_handler"),
    )
    assert _stale_versions(target, {"agents": ["dispute_handler"]}) == ""
