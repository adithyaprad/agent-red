"""The consent gate: registry resolution, the challenge echo, and the token."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from agentred.runner.consent import (
    CONSENT_TTL_SECONDS,
    ChallengeFailedError,
    ConsentError,
    ConsentToken,
    RegisteredTarget,
    RegistryError,
    TargetNotRegisteredError,
    TargetRegistry,
    establish_consent,
    load_registry,
)

REGISTRY_YAML = """
version: 1
targets:
  - name: cart_recovery
    agent_id: cart_recovery
    description: Cart recovery.
    base_url: http://localhost:8081
    spec_dir: specs/cart_recovery
    mode: test
  - name: dispute_handler
    agent_id: dispute_handler
    base_url: http://localhost:8082
    spec_dir: specs/dispute_handler
    mode: test
"""


class EchoingTransport:
    """A target that answers the challenge correctly."""

    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.calls: list[tuple[str, str]] = []

    def fetch_challenge(self, url: str, nonce: str) -> dict[str, Any]:
        self.calls.append((url, nonce))
        body: dict[str, Any] = {
            "challenge": nonce,
            "agent_id": "cart_recovery",
            "mode": "test",
        }
        body.update(self.overrides)
        return body


class SilentTransport:
    """A target that cannot be reached."""

    def fetch_challenge(self, url: str, nonce: str) -> dict[str, Any]:
        raise ChallengeFailedError(f"{url} could not be reached")


def write_registry(tmp_path: Path, text: str = REGISTRY_YAML) -> Path:
    path = tmp_path / "targets.registry.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def registry() -> TargetRegistry:
    return TargetRegistry(
        targets=(
            RegisteredTarget(
                name="cart_recovery",
                agent_id="cart_recovery",
                base_url="http://localhost:8081",
                spec_dir=Path("specs/cart_recovery"),
            ),
        )
    )


def test_registry_resolves_by_name(tmp_path: Path) -> None:
    loaded = load_registry(write_registry(tmp_path))
    assert loaded.names == ("cart_recovery", "dispute_handler")
    assert loaded.resolve("dispute_handler").base_url == "http://localhost:8082"


def test_registry_resolves_spec_dir_against_its_own_directory(tmp_path: Path) -> None:
    loaded = load_registry(write_registry(tmp_path))
    assert loaded.resolve("cart_recovery").spec_dir == (tmp_path / "specs/cart_recovery")


def test_unregistered_name_is_refused_and_names_what_exists(tmp_path: Path) -> None:
    loaded = load_registry(write_registry(tmp_path))
    with pytest.raises(TargetNotRegisteredError) as error:
        loaded.resolve("someone_elses_agent")
    assert "dispute_handler" in str(error.value)


def test_a_url_is_not_a_name(tmp_path: Path) -> None:
    loaded = load_registry(write_registry(tmp_path))
    with pytest.raises(TargetNotRegisteredError):
        loaded.resolve("http://localhost:8081")


def test_missing_registry_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(RegistryError) as error:
        load_registry(tmp_path / "targets.registry.yaml")
    assert "does not exist" in str(error.value)


def test_registry_rejects_unparseable_yaml(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not valid YAML"):
        load_registry(write_registry(tmp_path, "targets: [\n"))


def test_registry_rejects_a_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="mapping"):
        load_registry(write_registry(tmp_path, "- cart_recovery\n"))


def test_registry_rejects_a_repeated_name(tmp_path: Path) -> None:
    doubled = (
        REGISTRY_YAML
        + """
  - name: cart_recovery
    agent_id: cart_recovery
    base_url: http://localhost:8083
    spec_dir: specs/other
"""
    )
    with pytest.raises(RegistryError, match="twice"):
        load_registry(write_registry(tmp_path, doubled))


def test_registry_names_the_field_that_failed(tmp_path: Path) -> None:
    text = """
version: 1
targets:
  - name: cart_recovery
    agent_id: cart_recovery
    base_url: localhost:8081
    spec_dir: specs/cart_recovery
"""
    with pytest.raises(RegistryError) as error:
        load_registry(write_registry(tmp_path, text))
    assert "base_url" in str(error.value)


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8081/chat", "ftp://localhost:8081", "http://"],
)
def test_base_url_must_be_a_bare_http_origin(base_url: str) -> None:
    with pytest.raises(ValueError):
        RegisteredTarget(name="t", agent_id="t", base_url=base_url, spec_dir=Path("specs/t"))


def test_base_url_trailing_slash_is_dropped() -> None:
    target = RegisteredTarget(
        name="t", agent_id="t", base_url="http://localhost:8081/", spec_dir=Path("specs/t")
    )
    assert target.challenge_url == "http://localhost:8081/challenge"
    assert target.chat_url == "http://localhost:8081/chat"


def test_consent_is_granted_when_the_target_echoes() -> None:
    transport = EchoingTransport()
    token = establish_consent("cart_recovery", registry=registry(), transport=transport)
    assert token.nonce == transport.calls[0][1]
    assert token.chat_url == "http://localhost:8081/chat"
    assert token.is_live()


def test_the_nonce_is_fresh_on_every_call() -> None:
    transport = EchoingTransport()
    first = establish_consent("cart_recovery", registry=registry(), transport=transport)
    second = establish_consent("cart_recovery", registry=registry(), transport=transport)
    assert first.nonce != second.nonce
    assert len(first.nonce) == 32


def test_the_challenge_goes_to_the_challenge_endpoint() -> None:
    transport = EchoingTransport()
    establish_consent("cart_recovery", registry=registry(), transport=transport)
    assert transport.calls[0][0] == "http://localhost:8081/challenge"


def test_a_wrong_echo_is_refused() -> None:
    transport = EchoingTransport(challenge="not-the-nonce")
    with pytest.raises(ChallengeFailedError, match="did not echo"):
        establish_consent("cart_recovery", registry=registry(), transport=transport)


def test_a_missing_echo_is_refused() -> None:
    transport = EchoingTransport(challenge=None)
    with pytest.raises(ChallengeFailedError, match="did not echo"):
        establish_consent("cart_recovery", registry=registry(), transport=transport)


def test_a_different_agent_id_is_refused() -> None:
    transport = EchoingTransport(agent_id="dispute_handler")
    with pytest.raises(ChallengeFailedError, match="different agent"):
        establish_consent("cart_recovery", registry=registry(), transport=transport)


def test_a_target_not_in_test_mode_is_refused() -> None:
    transport = EchoingTransport(mode="live")
    with pytest.raises(ChallengeFailedError, match="mode"):
        establish_consent("cart_recovery", registry=registry(), transport=transport)


def test_an_unreachable_target_is_refused() -> None:
    with pytest.raises(ChallengeFailedError):
        establish_consent("cart_recovery", registry=registry(), transport=SilentTransport())


def test_an_unregistered_target_is_never_challenged() -> None:
    transport = EchoingTransport()
    with pytest.raises(TargetNotRegisteredError):
        establish_consent("someone_elses_agent", registry=registry(), transport=transport)
    assert transport.calls == []


def test_a_token_cannot_be_constructed_directly() -> None:
    target = registry().resolve("cart_recovery")
    with pytest.raises(ConsentError, match="cannot be constructed directly"):
        ConsentToken(target=target, nonce="deadbeef", granted_at=time.monotonic())


def test_an_expired_token_refuses_to_be_used() -> None:
    token = establish_consent("cart_recovery", registry=registry(), transport=EchoingTransport())
    later = token.granted_at + CONSENT_TTL_SECONDS + 1
    assert not token.is_live(now=later)
    with pytest.raises(ConsentError, match="expired"):
        token.require_live(now=later)


def test_a_live_token_passes_its_own_check() -> None:
    token = establish_consent("cart_recovery", registry=registry(), transport=EchoingTransport())
    token.require_live()
