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


class TestTheWorldCommand:
    """What a generated shop made reachable, printed for a person. Contacts nothing."""

    def run(self, target: str):
        from typer.testing import CliRunner

        from agentred.cli import app

        result = CliRunner().invoke(app, ["world", "--target", target])
        assert result.exit_code == 0, result.output
        return result.output

    def test_it_names_the_shop_and_the_share_of_rules_it_reached(self):
        output = self.run("dispute_handler")
        assert "sha256:" in output
        assert "declared rules" in output

    def test_a_rule_nothing_could_reach_is_named_rather_than_dropped(self):
        """A rule with no reachable fixture and a rule that was tested and held are opposite
        facts about an agent and identical in a finding count."""
        assert "no fixture" in self.run("cart_recovery")

    def test_a_gap_says_what_the_operator_could_add(self):
        """A remediation has to be config shaped, because the reader is an ops team."""
        assert "Naming the field after the argument" in self.run("cart_recovery")

    def test_both_halves_of_a_rule_are_shown(self):
        output = self.run("dispute_handler")
        assert "breakable" in output
        assert "holding" in output

    def test_it_writes_a_shop_when_asked(self, tmp_path):
        import json

        from typer.testing import CliRunner

        from agentred.cli import app

        out = tmp_path / "shop.json"
        result = CliRunner().invoke(
            app, ["world", "--target", "dispute_handler", "--out", str(out)]
        )
        assert result.exit_code == 0, result.output
        written = json.loads(out.read_text())
        assert written["collections"]["disputes"]
        assert written["fixtures"][0]["rule"]
