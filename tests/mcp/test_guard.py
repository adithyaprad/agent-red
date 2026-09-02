"""A target that could move real money refuses to start."""

from __future__ import annotations

import pytest

from agentred.mcp._guard import UnsafeTargetError, assert_test_mode


def test_declared_test_mode_passes() -> None:
    assert_test_mode({"AGENTRED_TARGET_MODE": "test"})


def test_unset_mode_refuses() -> None:
    with pytest.raises(UnsafeTargetError, match="unset"):
        assert_test_mode({})


@pytest.mark.parametrize("mode", ["live", "production", "Test", ""])
def test_any_other_mode_refuses(mode: str) -> None:
    with pytest.raises(UnsafeTargetError):
        assert_test_mode({"AGENTRED_TARGET_MODE": mode})


@pytest.mark.parametrize(
    "name,value",
    [
        ("STRIPE_API_KEY", "sk_live_abcdef"),
        ("PAYMENTS_SECRET_KEY", "rk_live_abcdef"),
        ("BILLING_TOKEN", "live_abcdef"),
        ("PROVIDER_KEY", "pk_live_abcdef"),
    ],
)
def test_a_live_credential_refuses_even_in_declared_test_mode(name: str, value: str) -> None:
    with pytest.raises(UnsafeTargetError, match=name):
        assert_test_mode({"AGENTRED_TARGET_MODE": "test", name: value})


def test_the_error_never_prints_the_credential() -> None:
    with pytest.raises(UnsafeTargetError) as error:
        assert_test_mode({"AGENTRED_TARGET_MODE": "test", "STRIPE_API_KEY": "sk_live_secret"})
    assert "sk_live_secret" not in str(error.value)


def test_a_test_credential_passes() -> None:
    assert_test_mode({"AGENTRED_TARGET_MODE": "test", "STRIPE_API_KEY": "sk_test_abcdef"})
