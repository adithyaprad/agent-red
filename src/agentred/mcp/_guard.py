"""Refuse to start a target that could move real money.

The stand-in agents issue refunds, apply discounts and create orders. Every one of those
is a call that would cost a merchant something if the credentials underneath were real, and
the whole point of an adversarial suite is that it tries hard to make exactly those calls.

ADR-0001 puts the assertion in the target rather than in the operator's habits: an operator
who is about to point a refund attack at live credentials should get a crash on startup, not
a warning in a log they will read afterwards. This module is imported for its side effect at
the top of `runtime.py`, so there is no way to serve a target without passing it.
"""

from __future__ import annotations

import os

MODE_ENV_VAR = "AGENTRED_TARGET_MODE"
TEST_MODE = "test"

LIVE_KEY_PREFIXES = ("sk_live_", "pk_live_", "rk_live_", "live_")
"""Prefixes payment providers use for credentials that move real money."""

CREDENTIAL_ENV_SUFFIXES = ("_API_KEY", "_SECRET_KEY", "_TOKEN", "_KEY")
"""Environment names checked for a live credential. Deliberately broad."""


class UnsafeTargetError(RuntimeError):
    """The environment could move real money, so the target refuses to start.

    Never caught anywhere in the tree. A target that cannot prove it is in test mode is a
    target agent-red does not run, and the operator's fix is to correct the environment.
    """


def assert_test_mode(env: dict[str, str] | None = None) -> None:
    """Refuse to continue unless this process is demonstrably in test mode.

    Two independent checks, because either one alone is escapable. The mode must be
    declared, so a target cannot be started by an operator who never thought about it. And
    no credential-shaped environment variable may hold a live-mode key, so a declared test
    mode cannot sit on top of a production key that was left exported in the shell.

    Args:
        env: Environment to read. Defaults to `os.environ`.

    Raises:
        UnsafeTargetError: If the mode is unset or not `test`, or any credential-shaped
            variable holds a value with a live-mode prefix. The message names the variable
            but never its value.
    """
    env = dict(os.environ) if env is None else env

    mode = env.get(MODE_ENV_VAR, "").strip()
    if mode != TEST_MODE:
        found = "unset" if mode == "" else repr(mode)
        raise UnsafeTargetError(
            f"{MODE_ENV_VAR} is {found}, and a target that cannot prove it is in "
            f"{TEST_MODE!r} mode does not start. These agents issue refunds and discounts, "
            f"and agent-red will spend the next hour trying to make them do it."
        )

    for name, value in sorted(env.items()):
        if not name.endswith(CREDENTIAL_ENV_SUFFIXES):
            continue
        if value.startswith(LIVE_KEY_PREFIXES):
            raise UnsafeTargetError(
                f"{name} holds a live-mode credential. {MODE_ENV_VAR}={TEST_MODE} says this "
                f"process is a test, and the environment says otherwise. Unset it or replace "
                f"it with a test key."
            )
