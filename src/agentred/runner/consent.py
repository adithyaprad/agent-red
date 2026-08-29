"""The gate every attack turn passes through.

agent-red composes working social-engineering attacks and sends them at a live agent.
Pointed at an endpoint the operator does not control, that is an attack tool. The
difference cannot be a paragraph in a README, because a reader who wants the harness to be
safe will believe the paragraph and nobody else has to, so it is enforced here instead.

Two things make it enforceable. A target is resolved by name from a registry file, so no
function in this module accepts a bare URL. And before the first attack turn the harness
sends a fresh nonce and requires the target to echo it back, along with the agent id it
believes it is and the mode it is running in. An endpoint nobody configured to answer that
cannot be attacked by this harness, which is the intended behaviour rather than a
limitation to work around.

The proof that consent was established is a `ConsentToken`, which cannot be constructed
outside this module. Everything that sends a turn takes one, so a code path that reaches a
target without passing through here does not type-check, let alone run. Reasoning in
`docs/DECISIONS/ADR-0001-scope-and-consent.md`.
"""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REGISTRY_FILENAME = "targets.registry.yaml"
"""Name of the registry file, looked for at the repository root."""

NONCE_BYTES = 16
CONSENT_TTL_SECONDS = 900.0
"""How long a token stays valid. Consent is established per run, not once per install."""

DEFAULT_CHALLENGE_PATH = "/challenge"
DEFAULT_CHAT_PATH = "/chat"
CHALLENGE_TIMEOUT_SECONDS = 10.0

_ISSUER = object()
"""Sentinel proving a token came from `establish_consent` and not from a caller."""


class ConsentError(Exception):
    """Consent could not be established, so no attack turn may be sent.

    Every failure here is terminal for the target it concerns. Nothing in the tree may
    catch this and proceed with a degraded or assumed consent.
    """


class RegistryError(ConsentError):
    """The registry file is missing, unparseable, or does not describe a usable target."""


class TargetNotRegisteredError(ConsentError):
    """A target was asked for by a name the registry does not carry.

    Raised rather than falling back to treating the name as a URL, which is the mistake
    this whole module exists to make impossible.
    """


class ChallengeFailedError(ConsentError):
    """The target did not echo the challenge, or answered it with the wrong identity.

    Covers an unreachable target, a non-200 answer, a malformed body, a mismatched nonce,
    an agent id that disagrees with the registry, and a target reporting a mode other than
    the one the registry declares. They are one class because the response to all of them
    is the same: do not attack this endpoint.
    """


class RegisteredTarget(BaseModel):
    """One agent this installation is permitted to attack.

    Attributes:
        name: How the operator refers to this target on the command line. Unique.
        agent_id: The agent id its spec declares. The target must report the same id when
            it answers the challenge, so that a registry entry cannot be pointed at a
            different agent by editing a port.
        base_url: Scheme, host and port. No path, so that a challenge cannot be answered
            by one service and the attack delivered to another.
        spec_dir: Directory holding `config.yaml` and `policy.yaml` for this agent,
            resolved against the registry file's own directory at load time.
        mode: The mode the target must report. `test` for anything that can move money.
        description: What the agent does, for the operator reading `agentred doctor`.
        challenge_path: Path that echoes the nonce.
        chat_path: Path that takes a conversation and returns a reply.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    spec_dir: Path
    mode: str = Field(default="test", min_length=1)
    description: str = Field(default="")
    challenge_path: str = Field(default=DEFAULT_CHALLENGE_PATH)
    chat_path: str = Field(default=DEFAULT_CHAT_PATH)

    @field_validator("base_url")
    @classmethod
    def _bare_origin(cls, value: str) -> str:
        """Require an http origin with no path, and drop a trailing slash."""
        trimmed = value.rstrip("/")
        if not trimmed.startswith(("http://", "https://")):
            raise ValueError(f"base_url {value!r} must start with http:// or https://")
        remainder = trimmed.split("://", 1)[1]
        if not remainder:
            raise ValueError(f"base_url {value!r} names no host")
        if "/" in remainder:
            raise ValueError(f"base_url {value!r} must be an origin with no path")
        return trimmed

    @field_validator("challenge_path", "chat_path")
    @classmethod
    def _absolute_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError(f"path {value!r} must start with /")
        return value

    @property
    def challenge_url(self) -> str:
        """Full URL of the challenge endpoint."""
        return f"{self.base_url}{self.challenge_path}"

    @property
    def chat_url(self) -> str:
        """Full URL of the chat endpoint.

        Read only from a `ConsentToken`, so that having the URL and being allowed to use it
        remain two different things.
        """
        return f"{self.base_url}{self.chat_path}"


class TargetRegistry(BaseModel):
    """Every target this installation may attack, and nothing else.

    Attributes:
        version: Registry format version. Present so the format can change without
            guessing at what an old file meant.
        targets: The registered targets, unique by name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    targets: tuple[RegisteredTarget, ...] = ()

    def resolve(self, name: str) -> RegisteredTarget:
        """Look a target up by name.

        Args:
            name: The registered name. Never a URL, and never a host.

        Returns:
            The registered target.

        Raises:
            TargetNotRegisteredError: If no entry carries that name. The message lists the names
                that do exist, because the fix is always to register the target or correct
                the name, never to relax the lookup.
        """
        for target in self.targets:
            if target.name == name:
                return target
        known = ", ".join(sorted(target.name for target in self.targets)) or "(none)"
        raise TargetNotRegisteredError(
            f"{name!r} is not a registered target. Registered: {known}. "
            f"agent-red only attacks agents listed in {REGISTRY_FILENAME}."
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Registered names, in file order."""
        return tuple(target.name for target in self.targets)


def load_registry(path: Path | str | None = None) -> TargetRegistry:
    """Read the registry file.

    Args:
        path: Path to the registry YAML. Defaults to `targets.registry.yaml` found by
            walking up from this file to the repository root.

    Returns:
        The parsed registry, with every `spec_dir` resolved against the registry's own
        directory so that a registry is portable between checkouts.

    Raises:
        RegistryError: If the file is missing, unparseable, not a mapping, fails
            validation, or repeats a target name.
    """
    path = _default_registry_path() if path is None else Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RegistryError(
            f"{path} does not exist. agent-red attacks only registered targets, so there is "
            f"nothing it can be pointed at until this file lists one."
        ) from error
    except OSError as error:
        raise RegistryError(f"{path} could not be read: {error}") from error

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise RegistryError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict):
        found = "an empty document" if document is None else f"a {type(document).__name__}"
        raise RegistryError(f"{path} must contain a mapping at the top level, found {found}")

    try:
        registry = TargetRegistry.model_validate(document)
    except ValidationError as error:
        details = "\n".join(
            f"  {'.'.join(str(part) for part in detail['loc']) or '(root)'}: {detail['msg']}"
            for detail in error.errors()
        )
        raise RegistryError(f"{path} is not a valid target registry:\n{details}") from error

    seen: set[str] = set()
    for target in registry.targets:
        if target.name in seen:
            raise RegistryError(f"{path} registers {target.name!r} twice")
        seen.add(target.name)

    return TargetRegistry(
        version=registry.version,
        targets=tuple(
            target.model_copy(update={"spec_dir": (path.parent / target.spec_dir).resolve()})
            for target in registry.targets
        ),
    )


def _default_registry_path() -> Path:
    """Find `targets.registry.yaml` by walking up from this module.

    Returns the repository-root path even when the file is absent, so that the error
    message from `load_registry` names where the file was expected.
    """
    here = Path(__file__).resolve()
    for directory in here.parents:
        candidate = directory / REGISTRY_FILENAME
        if candidate.is_file():
            return candidate
    return here.parents[3] / REGISTRY_FILENAME


@runtime_checkable
class ChallengeTransport(Protocol):
    """How the challenge reaches the target.

    An interface rather than a direct httpx call so that the gate itself is testable
    offline. A fake transport in tests can answer a challenge, refuse one, or echo the
    wrong value, which are the three cases worth asserting.
    """

    def fetch_challenge(self, url: str, nonce: str) -> dict[str, Any]:
        """Ask the target to echo `nonce`.

        Args:
            url: The target's challenge URL.
            nonce: The value the target must return.

        Returns:
            The decoded JSON body.

        Raises:
            ChallengeFailedError: If the target is unreachable, answers with a non-200 status,
                or returns a body that is not a JSON object.
        """
        ...


class HttpxChallengeTransport:
    """The real transport, over HTTP.

    Attributes:
        timeout: Seconds to wait. Short on purpose: a target that cannot answer a static
            echo in ten seconds is not in a state to be attacked.
    """

    def __init__(self, timeout: float = CHALLENGE_TIMEOUT_SECONDS) -> None:
        """Build a transport.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout

    def fetch_challenge(self, url: str, nonce: str) -> dict[str, Any]:
        """Send the challenge over HTTP. See `ChallengeTransport.fetch_challenge`."""
        import httpx

        try:
            response = httpx.get(url, params={"nonce": nonce}, timeout=self.timeout)
        except httpx.HTTPError as error:
            raise ChallengeFailedError(f"{url} could not be reached: {error}") from error

        if response.status_code != 200:
            raise ChallengeFailedError(
                f"{url} answered the challenge with HTTP {response.status_code}. A target "
                f"that cannot echo a challenge cannot be attacked by agent-red."
            )
        try:
            body = response.json()
        except ValueError as error:
            raise ChallengeFailedError(
                f"{url} answered the challenge with a non-JSON body"
            ) from error
        if not isinstance(body, dict):
            raise ChallengeFailedError(
                f"{url} answered the challenge with a {type(body).__name__}, expected an object"
            )
        return body


@dataclass(frozen=True)
class ConsentToken:
    """Proof that a registered target echoed a live challenge.

    Held by everything that sends a turn, and constructible only by `establish_consent`.
    That is what makes the gate a property of the code rather than a convention: a driver
    cannot be called without one, and one cannot be manufactured.

    Attributes:
        target: The registered target consent was established with.
        nonce: The value the target echoed. Recorded so a transcript can show what was
            agreed to.
        granted_at: Monotonic clock reading at the moment the echo was verified.
        ttl_seconds: How long the token stays valid.
    """

    target: RegisteredTarget
    nonce: str
    granted_at: float
    ttl_seconds: float = CONSENT_TTL_SECONDS

    def __post_init__(self) -> None:
        """Reject a token that did not come from `establish_consent`.

        Raises:
            ConsentError: Always, unless the issuer sentinel was passed. Frozen dataclasses
                do not offer a private constructor, so the check is made here.
        """
        if getattr(self, "_issuer", None) is not _ISSUER:
            raise ConsentError(
                "A ConsentToken cannot be constructed directly. Call establish_consent(), "
                "which requires the target to echo a live challenge."
            )

    @property
    def chat_url(self) -> str:
        """Where turns for this target go. Reachable only through a token."""
        return self.target.chat_url

    def is_live(self, now: float | None = None) -> bool:
        """Whether the token is still inside its window.

        Args:
            now: Monotonic clock reading. Defaults to `time.monotonic()`.
        """
        now = time.monotonic() if now is None else now
        return now - self.granted_at < self.ttl_seconds

    def require_live(self, now: float | None = None) -> None:
        """Raise unless the token is still inside its window.

        Called by anything about to send a turn, so that a long suite re-establishes
        consent rather than trading on an agreement made an hour ago.

        Raises:
            ConsentError: If the token has expired.
        """
        if not self.is_live(now):
            raise ConsentError(
                f"Consent for {self.target.name!r} expired after {self.ttl_seconds:.0f}s. "
                f"Re-establish it before sending another turn."
            )


def _issue(target: RegisteredTarget, nonce: str) -> ConsentToken:
    """Build a token, bypassing the constructor guard. Private to this module."""
    token = object.__new__(ConsentToken)
    object.__setattr__(token, "_issuer", _ISSUER)
    object.__setattr__(token, "target", target)
    object.__setattr__(token, "nonce", nonce)
    object.__setattr__(token, "granted_at", time.monotonic())
    object.__setattr__(token, "ttl_seconds", CONSENT_TTL_SECONDS)
    return token


def establish_consent(
    name: str,
    *,
    registry: TargetRegistry | None = None,
    transport: ChallengeTransport | None = None,
    nonce: str | None = None,
) -> ConsentToken:
    """Resolve a target by name and require it to echo a fresh challenge.

    This is the only way to obtain a `ConsentToken`, and therefore the only way to reach a
    target at all. It takes a registered name, never a URL: there is deliberately no
    argument by which an operator can point the harness at an arbitrary endpoint.

    The target must return the nonce unchanged, the agent id the registry expects, and the
    mode the registry declares. The identity check matters as much as the echo: without it,
    a registry entry could be repointed at a different agent by changing a port, and the
    scorecard would name an agent that was never tested.

    Args:
        name: A name from the registry.
        registry: The registry to resolve against. Defaults to `load_registry()`.
        transport: How to send the challenge. Defaults to HTTP.
        nonce: Force the nonce. For tests only; production always generates a fresh one.

    Returns:
        A live `ConsentToken`.

    Raises:
        TargetNotRegisteredError: If `name` is not registered.
        RegistryError: If the default registry cannot be read.
        ChallengeFailedError: If the target is unreachable, does not echo the nonce, reports a
            different agent id, or reports a mode other than the declared one.
    """
    registry = load_registry() if registry is None else registry
    target = registry.resolve(name)
    transport = HttpxChallengeTransport() if transport is None else transport
    nonce = secrets.token_hex(NONCE_BYTES) if nonce is None else nonce

    body = transport.fetch_challenge(target.challenge_url, nonce)
    _verify(target, nonce, body)
    return _issue(target, nonce)


def _verify(target: RegisteredTarget, nonce: str, body: dict[str, Any]) -> None:
    """Check a challenge response against the registry entry that asked for it.

    Raises:
        ChallengeFailedError: On a mismatched echo, agent id or mode. The echo is compared with
            `hmac.compare_digest`, which costs nothing and removes the question.
    """
    echoed = body.get("challenge")
    if not isinstance(echoed, str) or not hmac.compare_digest(echoed, nonce):
        raise ChallengeFailedError(
            f"{target.name!r} did not echo the challenge. agent-red attacks only a target "
            f"that proves it consents, so configure the challenge endpoint rather than "
            f"working around this."
        )

    reported_id = body.get("agent_id")
    if reported_id != target.agent_id:
        raise ChallengeFailedError(
            f"{target.name!r} reports agent id {reported_id!r} but the registry expects "
            f"{target.agent_id!r}. The endpoint is serving a different agent than the one "
            f"registered, and a scorecard from it would name the wrong agent."
        )

    reported_mode = body.get("mode")
    if reported_mode != target.mode:
        raise ChallengeFailedError(
            f"{target.name!r} reports mode {reported_mode!r} but the registry declares "
            f"{target.mode!r}. Attacks include refunds and discounts, so a target that "
            f"cannot prove it is in {target.mode!r} mode is not attacked."
        )
