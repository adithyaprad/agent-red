"""Driving one attack conversation from the first turn to the last.

This is the loop that everything else in the harness is arranged around: an attacker
produces a turn, the target answers it, what it did on the way is read from the tool
server's record, and the whole thing is handed back as a transcript that the judge can read
and the store can keep.

**The target is never asked what it did.** After each turn the driver reads the calls the
tool server recorded for this session, and the ones that appeared since the previous turn
are that turn's calls. The reply body carries prose and versions only. A driver that read a
tool-call log out of the reply would be taking the measured party's word for it, and would
need a new integration for every agent runtime (ADR-0005).

Every function that reaches a target takes a `ConsentToken`, which only
`runner.consent.establish_consent` can produce. There is no argument here that accepts a
URL, and the token is re-checked before each turn rather than once at the start, so a suite
that outlives its consent stops instead of continuing quietly.
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Protocol, runtime_checkable

from agentred.mcp.control import ArenaControl, HttpxArenaControl
from agentred.runner.channels.attempt import (
    PlantedField,
    TargetError,
    ToolCallRecord,
    Transcript,
    Turn,
)
from agentred.runner.consent import ConsentToken

DEFAULT_MAX_TURNS = 6
TURN_TIMEOUT_SECONDS = 120.0

__all__ = [
    "DEFAULT_MAX_TURNS",
    "TURN_TIMEOUT_SECONDS",
    "Attacker",
    "HttpxTargetTransport",
    "PlantedField",
    "TargetError",
    "TargetTransport",
    "ToolCallRecord",
    "Transcript",
    "Turn",
    "new_session_id",
    "run_conversation",
]
"""Re-exported so that the thirty-odd modules reading a transcript keep one import path.

`attempt.py` owns these; naming them here as well means moving them did not ripple into
`judge/`, `scoring/` and `store/`, none of which care which driver produced the transcript
they are reading.
"""


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
        self, token: ConsentToken, session: str, run: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Send a conversation and return the decoded `/chat` response.

        Args:
            token: Proof that the target consented.
            session: The conversation's session id.
            run: The run its tool calls are recorded under. Passed to the target because the
                target has to point its tool connector somewhere, and this is where.
            conversation: The conversation so far, ending with the turn to answer.

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
        self, token: ConsentToken, session: str, run: str, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Post one turn to the consented target. See `TargetTransport.send`."""
        import httpx

        token.require_live()
        try:
            response = httpx.post(
                token.chat_url,
                json={"session": session, "run": run, "conversation": conversation},
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
    run: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    transport: TargetTransport | None = None,
    control: ArenaControl | None = None,
    session: str | None = None,
    subject: dict[str, str] | None = None,
    resume: Transcript | None = None,
) -> Transcript:
    """Run one attack conversation to its end and return the transcript.

    Consent is re-checked before every turn rather than once at the start. A six-turn
    conversation inside a long suite can outlive the agreement it began under, and the
    correct behaviour then is to stop, not to finish the conversation politely.

    What the agent did is read from the tool server after each turn, never from the reply.
    The world is checkpointed at each turn boundary too, so a fork taken later branches from
    the state that existed at the branch point rather than from the state at the end.

    Args:
        token: Proof that the target consented. The only way to name a target.
        attacker: Produces each turn. May stop early by returning `None`.
        run: The run this conversation belongs to. Its calls are recorded under it, and read
            back under it, so two runs against one agent never read each other's evidence.
        max_turns: Ceiling on exchanges. The budget is per conversation, and a conversation
            that has not broken the agent in six turns is a result rather than a reason to
            keep going.
        transport: How turns are sent. Defaults to HTTP.
        control: How the tool server's record is read. Defaults to HTTP against the control
            face named in the registry entry for this target.
        session: Force the session id. Used when continuing a forked conversation;
            otherwise a fresh id gives this conversation a private world.
        subject: Who the conversation is about, as identifier kind to value. Recorded on the
            transcript so the scope detector has something to compare a reached record
            against. Omitted, scope checks report as unevaluated rather than as passed.
        resume: A conversation to continue rather than start. The turns already in it are
            sent to the target as history and count against nothing; `max_turns` bounds the
            new turns only.

    Returns:
        The transcript, including the reason it stopped.

    Raises:
        ConsentError: If the token expires mid-conversation.
        TargetError: If the target becomes unreachable or answers unusably. A broken target
            is not a well-behaved one, so this is not swallowed into the transcript.
        ControlError: If the tool server cannot be read. A conversation whose record cannot
            be read is not a conversation in which nothing happened.
    """
    transport = HttpxTargetTransport() if transport is None else transport
    control = HttpxArenaControl(token.target.control_url) if control is None else control
    if resume is not None:
        transcript = resume
        if session is not None:
            transcript.session = session
    else:
        transcript = Transcript(
            target=token.target.name,
            session=new_session_id() if session is None else session,
            goal=attacker.goal,
            subject=dict(subject or {}),
        )
    already = len(transcript.turns)
    # How much of this session's stream belongs to turns already in the transcript. Zero for
    # a fresh conversation and for a branch, which has its own stream even though it inherits
    # the world and the turns of the conversation it came from.
    seen_calls = len(control.calls(run, transcript.session))

    for index in range(already, already + max_turns):
        token.require_live()
        user_turn = attacker.next_turn(transcript)
        if user_turn is None:
            transcript.stopped_because = "attacker stopped"
            return transcript

        conversation = [*transcript.messages, {"role": "user", "content": user_turn}]
        started = time.monotonic()
        body = transport.send(token, transcript.session, run, conversation)
        elapsed = time.monotonic() - started

        recorded = control.calls(run, transcript.session)
        this_turn = recorded[seen_calls:]
        seen_calls = len(recorded)
        transcript.turns.append(
            Turn(
                index=index,
                user=user_turn,
                reply=str(body.get("reply", "")),
                tool_calls=tuple(ToolCallRecord.from_recorded(call) for call in this_turn),
                latency_seconds=round(elapsed, 3),
                agent_usage={key: float(value) for key, value in (body.get("usage") or {}).items()},
            )
        )
        control.checkpoint(transcript.session)
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
