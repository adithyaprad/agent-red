"""Driving one attack conversation from the first turn to the last.

This is the loop that everything else in the harness is arranged around: an attacker
produces a turn, the target answers it, the tool calls it made on the way are recorded, and
the whole thing is handed back as a transcript that the judge can read and the store can
keep.

Every function that reaches a target takes a `ConsentToken`, which only
`runner.consent.establish_consent` can produce. There is no argument here that accepts a
URL, and the token is re-checked before each turn rather than once at the start, so a suite
that outlives its consent stops instead of continuing quietly.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentred.runner.consent import ConsentToken

DEFAULT_MAX_TURNS = 6
TURN_TIMEOUT_SECONDS = 120.0


class TargetError(RuntimeError):
    """The target could not be reached, or answered with something unusable.

    Distinct from a target that answered badly: an agent that says something it should not
    is the result, and a target that returns HTTP 500 is a broken run. Conflating them
    would let an outage read as a suite full of well-behaved agents.
    """


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the target made, as the target reported it.

    Attributes:
        name: The declared tool name.
        arguments: Arguments as the model sent them, uncoerced. Bounds are checked against
            what was passed, not against what the agent said it passed.
        result: What the tool returned.
    """

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ToolCallRecord:
        """Build a record from one entry of a `/chat` response."""
        return cls(
            name=str(payload.get("name", "")),
            arguments=dict(payload.get("arguments") or {}),
            result=dict(payload.get("result") or {}),
        )


@dataclass(frozen=True)
class Turn:
    """One exchange: what was said to the agent, and what it did about it.

    Attributes:
        index: Zero-based position in the conversation.
        user: The attacker's turn.
        reply: The agent's text.
        tool_calls: Tools called while producing that reply, in call order.
        latency_seconds: Wall clock for the target's answer.
    """

    index: int
    user: str
    reply: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    latency_seconds: float = 0.0


@dataclass
class Transcript:
    """One attack conversation, complete.

    The unit the judge grades, the store keeps and the scorecard cites. It carries the spec
    versions the target reported rather than the ones the harness believed, so a transcript
    can never be attributed to a version of the agent that did not produce it.

    Attributes:
        target: The registered target name.
        session: The session id the target kept this conversation's world under.
        goal: What the attacker was trying to make the agent do, in one line.
        turns: The exchanges, in order.
        spec_versions: Config, policy, model and tool versions, as reported by the target.
        stopped_because: Why the conversation ended.
    """

    target: str
    session: str
    goal: str
    turns: list[Turn] = field(default_factory=list)
    spec_versions: dict[str, str] = field(default_factory=dict)
    stopped_because: str = ""

    @property
    def messages(self) -> list[dict[str, str]]:
        """The conversation in wire shape, for sending the next turn or showing a human."""
        wire: list[dict[str, str]] = []
        for turn in self.turns:
            wire.append({"role": "user", "content": turn.user})
            wire.append({"role": "assistant", "content": turn.reply})
        return wire

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        """Every tool call in the conversation, flattened, in order.

        The deterministic detectors read this: a bound is broken by an argument, and a
        precondition is broken by an order, and both are answerable from this list alone.
        """
        return tuple(call for turn in self.turns for call in turn.tool_calls)

    def called(self, name: str) -> bool:
        """Whether a tool was called at any point in the conversation."""
        return any(call.name == name for call in self.tool_calls)


@runtime_checkable
class Attacker(Protocol):
    """Whatever decides what to say next.

    The generator in `attacks/` implements this with a model. Tests implement it with a
    list. The driver does not care which, and deliberately knows nothing about attack
    technique: composing a turn and executing a conversation are separate jobs.
    """

    @property
    def goal(self) -> str:
        """What this attacker is trying to make the agent do, in one line."""
        ...

    def next_turn(self, transcript: Transcript) -> str | None:
        """The next thing to say, or `None` to stop early.

        Args:
            transcript: The conversation so far. Empty on the opening turn.

        Returns:
            The user turn to send, or `None` when there is nothing left worth trying. An
            attacker that stops early saves the turn budget for conversations still moving.
        """
        ...


@runtime_checkable
class TargetTransport(Protocol):
    """How a turn reaches the target.

    Every method takes a `ConsentToken`. That is the enforcement: a transport cannot be
    handed a URL, and a caller without a token cannot construct one.
    """

    def send(
        self, token: ConsentToken, session: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Send a conversation and return the decoded `/chat` response.

        Raises:
            TargetError: If the target is unreachable or answers with a non-200 status or
                an unusable body.
        """
        ...

    def fork(
        self, token: ConsentToken, source: str, session: str, at_turn: int | None = None
    ) -> None:
        """Ask the target to branch `source` into a new session called `session`.

        Args:
            token: Proof that the target consented.
            source: The session to branch from.
            session: The id for the branch.
            at_turn: How many completed exchanges the branch keeps. `None` branches from the
                end of the conversation.

        Raises:
            TargetError: If the target refuses or cannot be reached.
        """
        ...


class HttpxTargetTransport:
    """The real transport, over HTTP.

    Attributes:
        timeout: Seconds to wait for one reply. Generous, because a target that thinks for
            ninety seconds is slow rather than broken, and a timeout that fires early would
            report a resisting agent as a failed run.
    """

    def __init__(self, timeout: float = TURN_TIMEOUT_SECONDS) -> None:
        """Build a transport.

        Args:
            timeout: Per-turn request timeout in seconds.
        """
        self.timeout = timeout

    def send(
        self, token: ConsentToken, session: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Post one turn to the consented target. See `TargetTransport.send`."""
        import httpx

        token.require_live()
        try:
            response = httpx.post(
                token.chat_url,
                json={"session": session, "conversation": conversation},
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise TargetError(f"{token.target.name} could not be reached: {error}") from error

        if response.status_code != 200:
            raise TargetError(
                f"{token.target.name} answered a turn with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise TargetError(
                f"{token.target.name} answered a turn with a non-JSON body"
            ) from error
        if not isinstance(body, dict):
            raise TargetError(
                f"{token.target.name} answered a turn with a {type(body).__name__}, "
                f"expected an object"
            )
        return body

    def fork(
        self, token: ConsentToken, source: str, session: str, at_turn: int | None = None
    ) -> None:
        """Branch a session on the target. See `TargetTransport.fork`."""
        import httpx

        token.require_live()
        try:
            response = httpx.post(
                f"{token.target.base_url}/fork",
                json={"source": source, "session": session, "at_turn": at_turn},
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise TargetError(f"{token.target.name} could not be reached: {error}") from error
        if response.status_code != 200:
            raise TargetError(
                f"{token.target.name} refused to fork {source!r}: {response.text[:200]}"
            )


def new_session_id() -> str:
    """A session id for one conversation.

    Random rather than sequential, so that two runs against the same target cannot collide
    on a session and share a world without anyone noticing.
    """
    return f"ar-{secrets.token_hex(8)}"


def run_conversation(
    token: ConsentToken,
    attacker: Attacker,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    transport: TargetTransport | None = None,
    session: str | None = None,
    resume: Transcript | None = None,
) -> Transcript:
    """Run one attack conversation to its end and return the transcript.

    Consent is re-checked before every turn rather than once at the start. A six-turn
    conversation inside a long suite can outlive the agreement it began under, and the
    correct behaviour then is to stop, not to finish the conversation politely.

    Args:
        token: Proof that the target consented. The only way to name a target.
        attacker: Produces each turn. May stop early by returning `None`.
        max_turns: Ceiling on exchanges. The budget is per conversation, and a conversation
            that has not broken the agent in six turns is a result rather than a reason to
            keep going.
        transport: How turns are sent. Defaults to HTTP.
        session: Force the session id. Used when continuing a forked conversation;
            otherwise a fresh id gives this conversation a private world.
        resume: A conversation to continue rather than start. The turns already in it are
            sent to the target as history and count against nothing; `max_turns` bounds the
            new turns only.

    Returns:
        The transcript, including the reason it stopped.

    Raises:
        ConsentError: If the token expires mid-conversation.
        TargetError: If the target becomes unreachable or answers unusably. A broken target
            is not a well-behaved one, so this is not swallowed into the transcript.
    """
    transport = HttpxTargetTransport() if transport is None else transport
    if resume is not None:
        transcript = resume
        if session is not None:
            transcript.session = session
    else:
        transcript = Transcript(
            target=token.target.name,
            session=new_session_id() if session is None else session,
            goal=attacker.goal,
        )
    already = len(transcript.turns)

    for index in range(already, already + max_turns):
        token.require_live()
        user_turn = attacker.next_turn(transcript)
        if user_turn is None:
            transcript.stopped_because = "attacker stopped"
            return transcript

        conversation = [*transcript.messages, {"role": "user", "content": user_turn}]
        started = time.monotonic()
        body = transport.send(token, transcript.session, conversation)
        elapsed = time.monotonic() - started

        transcript.turns.append(
            Turn(
                index=index,
                user=user_turn,
                reply=str(body.get("reply", "")),
                tool_calls=tuple(
                    ToolCallRecord.from_payload(call) for call in body.get("tool_calls") or []
                ),
                latency_seconds=round(elapsed, 3),
            )
        )
        versions = body.get("spec_versions")
        if isinstance(versions, dict):
            _record_versions(transcript, versions, token)

    transcript.stopped_because = "turn budget spent"
    return transcript


def _record_versions(transcript: Transcript, versions: dict[str, Any], token: ConsentToken) -> None:
    """Record the versions the target reported, and refuse a target that changes them.

    A target whose config version changes mid-conversation has been redeployed underneath
    the run, and the turns before and after are not evidence about the same agent.

    Raises:
        TargetError: If the versions differ from the ones reported earlier in this
            conversation.
    """
    reported = {str(key): str(value) for key, value in versions.items()}
    if not transcript.spec_versions:
        transcript.spec_versions = reported
        return
    if reported != transcript.spec_versions:
        raise TargetError(
            f"{token.target.name} changed spec version mid-conversation, from "
            f"{transcript.spec_versions} to {reported}. The turns before and after are not "
            f"evidence about the same agent."
        )
