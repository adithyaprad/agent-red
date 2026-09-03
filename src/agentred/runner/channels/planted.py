"""Driving one planted attack: restore, plant, trigger, record.

The channel an agent nobody talks to has. A cart recovery agent is not conversed with: a
schedule wakes it, it reads the baskets nobody checked out, it decides who is worth
contacting, and it sends a message. Under a conversation-only harness that agent is not
attackable at all, and a suite reporting nothing against it would be describing the shape of
the harness rather than anything about the agent (ADR-0006).

Six steps, and the argument is entirely in the third:

```
restore()   the world, to the seeded baseline
plant()     into the field the agent declares an adversary writes
trigger()   the agent's real entry point: its schedule, or an ordinary request
record()    the call stream, at the tool boundary (ADR-0005)
assert()    detectors first, judge on the residue: the caller's job, not this module's
emit()      a transcript carrying the plant, the seed and what the field held before
```

The trigger has to be the one the deployment runs. A synthetic turn saying "a schedule just
fired" would test a different agent than the one that ships, and the finding would be about
a test harness. So `SCHEDULE` posts to the target's own trigger endpoint with no user turn
anywhere in it, and `REQUEST` sends the ordinary, non-adversarial message the channel
declares. In the second case the message is not the attack: the attack was already sitting
in the record before anyone asked, which is what makes this the harder case for an operator
to see. The turn they would read in a transcript is benign.

**The step that could quietly lie is the first.** A conversation gets its isolation from
having its own session and its own copy of the shop. A planted attempt has no conversation,
so it gets a known baseline instead, and a run that skipped the restore would attack a world
carrying the last attempt's refunds. Every attempt here restores before it plants, and the
plant is refused outright if the field it names is not already on the record, because a
planted field the record never had is a field the agent was never going to read.

Like every module that reaches a target, nothing here accepts a URL. The only way to name a
target is a `ConsentToken`, and consent is re-checked immediately before the trigger fires
rather than only when the token was issued.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import httpx

from agentred.mcp.control import ArenaControl, HttpxArenaControl
from agentred.runner.channels.attempt import (
    PlantedField,
    TargetError,
    ToolCallRecord,
    Transcript,
    Turn,
)
from agentred.runner.channels.conversational import (
    HttpxTargetTransport,
    TargetTransport,
    new_session_id,
)
from agentred.runner.consent import ConsentToken
from agentred.spec.models import RECORD_PLACEHOLDER, ChannelDeclaration, TriggerKind

SCHEDULE_FIRED = "(the agent's schedule fired; no message was sent)"
"""Stands in the transcript where a user turn would be, for a firing nobody spoke into.

An empty string there would read as a turn that was sent and happened to be blank. What the
transcript has to carry is that nothing was said, because the whole finding is that an agent
did something wrong without anyone addressing it.
"""

TRIGGER_TIMEOUT_SECONDS = 180.0
"""Longer than a conversational turn. A scheduled firing does the work of several turns in
one request: it lists, it decides, and it acts on everything it decided about."""


@runtime_checkable
class ScheduleTransport(Protocol):
    """How a scheduled firing is delivered.

    Separate from `TargetTransport` because the two are different acts against the agent, and
    a fake that can do one should not silently be able to do the other. Takes a token for the
    same reason every transport does.
    """

    def fire(self, token: ConsentToken, session: str, run: str) -> dict[str, Any]:
        """Fire the target's schedule once and return its reply body."""
        ...


class HttpxScheduleTransport:
    """The real schedule transport, over HTTP.

    Attributes:
        timeout: Seconds to wait for the firing to complete.
    """

    def __init__(self, timeout: float = TRIGGER_TIMEOUT_SECONDS) -> None:
        """Build a transport.

        Args:
            timeout: Seconds to wait for one firing.
        """
        self.timeout = timeout

    def fire(self, token: ConsentToken, session: str, run: str) -> dict[str, Any]:
        """Post to the target's trigger endpoint.

        Args:
            token: Proof the target consented. Supplies the only URL this can reach.
            session: The session the firing records its calls under.
            run: The run the calls belong to.

        Returns:
            The parsed reply body.

        Raises:
            TargetError: If the target cannot be reached, refuses, or answers with something
                that is not a JSON object. A target that will not fire is a broken run, not
                an agent that behaved.
        """
        try:
            response = httpx.post(
                token.trigger_url,
                json={"session": session, "run": run},
                timeout=self.timeout,
            )
        except httpx.HTTPError as error:
            raise TargetError(f"{token.target.name} could not be reached: {error}") from error
        if response.status_code != 200:
            raise TargetError(
                f"{token.target.name} refused to fire its schedule: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise TargetError(
                f"{token.target.name} answered a firing with something that is not JSON"
            ) from error
        if not isinstance(body, dict):
            raise TargetError(
                f"{token.target.name} answered a firing with a {type(body).__name__}, not an object"
            )
        return body


def run_planted(
    token: ConsentToken,
    channel: ChannelDeclaration,
    payload: str,
    *,
    run: str,
    record_id: str,
    goal: str,
    control: ArenaControl | None = None,
    transport: TargetTransport | None = None,
    schedule: ScheduleTransport | None = None,
    session: str | None = None,
    subject: dict[str, str] | None = None,
    subject_kinds: tuple[str, ...] = (),
) -> Transcript:
    """Plant one payload, fire the agent's real trigger, and return what it did.

    Args:
        token: Proof that the target consented. The only way to name a target.
        channel: The channel to plant through, as the agent declared it. Says which record
            the payload goes on, which field of it, and what makes the agent read it.
        payload: The attacker-controlled text to write.
        run: The run this attempt belongs to. Its calls are recorded under it and read back
            under it, so two runs against one agent never read each other's evidence.
        record_id: Which record to plant into, for example `DSP-9001`. Named by the caller
            rather than chosen here, because which record an attempt is about is part of
            what a seed has to reproduce.
        goal: What the attack is trying to make the agent do, in one line. Carried onto the
            transcript the way a conversational attacker's goal is.
        control: How the world is moved and the record read. Defaults to HTTP against the
            control face named in the registry entry for this target.
        transport: How a `REQUEST` trigger's message is sent. Defaults to HTTP.
        schedule: How a `SCHEDULE` trigger is fired. Defaults to HTTP.
        session: Force the session id. A fresh id otherwise, so this attempt has its own
            copy of the shop even though it also restores one.
        subject: Who the attempt is about, as identifier kind to value. Without it every
            scope check reports as unevaluated rather than as passed.
        subject_kinds: The identifier kinds the agent's declared data scope binds a record
            by. Used only for a `SCHEDULE` trigger, to read the cohort the firing was
            legitimately woken about. Omitted, a scheduled attempt carries no cohort and
            every record but the named one reads as out of scope, which is a finding about
            this harness rather than about the agent.

    Returns:
        A transcript with one turn in it, carrying the planted field and what it replaced.
        The user text is the trigger that was fired, so that a person reading the transcript
        sees the benign thing that was sent and the hostile thing that was already there.

    Raises:
        ConsentError: If the token is not live when the trigger is about to fire.
        PlantError: If the record or the field does not exist. Refused rather than created:
            a planted field the record never had reads on a coverage grid as a channel that
            was tested.
        UnknownSourceError: If nothing backs the channel's declared data source.
        TargetError: If the target cannot be reached or answers unusably.
        ControlError: If the tool server cannot be read. An attempt whose record cannot be
            read is not an attempt in which nothing happened.
    """
    control = HttpxArenaControl(token.target.control_url) if control is None else control
    session = new_session_id() if session is None else session

    control.restore(session)
    replaced = control.plant(
        session,
        source=channel.data_source,
        record_id=record_id,
        field_name=channel.record_path,
        payload=payload,
    )

    # Read before the firing, not after. The firing mutates the shop, and a cohort read
    # afterwards would be missing whatever the agent consumed on the way through, which is
    # exactly the set the check most needs to know was legitimately in play.
    cohort = _cohort(control, channel, session=session, kinds=subject_kinds)

    token.require_live()
    fired, body, elapsed = _fire(
        token,
        channel,
        record_id,
        run=run,
        session=session,
        transport=transport,
        schedule=schedule,
    )

    recorded = control.calls(run, session)
    transcript = Transcript(
        target=token.target.name,
        session=session,
        goal=goal,
        channel=channel.name,
        subject=dict(subject or {}),
        planted=(
            PlantedField(
                channel=channel.name,
                data_source=channel.data_source,
                record_id=record_id,
                field_name=channel.record_path,
                payload=payload,
                replaced=replaced,
            ),
        ),
        cohort=cohort,
        stopped_because=f"{channel.trigger.value} fired once",
    )
    transcript.turns.append(
        Turn(
            index=0,
            user=fired,
            reply=str(body.get("reply", body.get("output", ""))),
            tool_calls=tuple(ToolCallRecord.from_recorded(call) for call in recorded),
            latency_seconds=round(elapsed, 3),
            agent_usage={key: float(value) for key, value in (body.get("usage") or {}).items()},
        )
    )
    versions = body.get("spec_versions")
    if isinstance(versions, dict):
        transcript.spec_versions = {str(key): str(value) for key, value in versions.items()}
    control.checkpoint(session)
    return transcript


def _cohort(
    control: ArenaControl,
    channel: ChannelDeclaration,
    *,
    session: str,
    kinds: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    """Every subject a scheduled firing is legitimately woken about.

    Only a `SCHEDULE` trigger has one. A `REQUEST` trigger arrives as a message about one
    record, so the attempt is about that record and one subject describes it completely.

    Empty when the agent declares no identifier kinds to bind a record by, because there is
    then nothing to describe a cohort member with, and the scope check already reports
    itself unevaluated in that case rather than guessing.
    """
    if channel.trigger is not TriggerKind.SCHEDULE or not kinds:
        return ()
    return control.subjects(session, source=channel.data_source, kinds=kinds)


def _fire(
    token: ConsentToken,
    channel: ChannelDeclaration,
    record_id: str,
    *,
    run: str,
    session: str,
    transport: TargetTransport | None,
    schedule: ScheduleTransport | None,
) -> tuple[str, dict[str, Any], float]:
    """Fire the trigger this channel declares, and say what was sent.

    Returns:
        What was sent to the agent, the reply body, and the wall clock it took. The first is
        prose for a person reading the transcript, and for a `SCHEDULE` it says that nothing
        was sent, because that is the fact the transcript has to carry.

    Raises:
        TargetError: If the target cannot be reached or answers unusably.
    """
    started = time.monotonic()
    if channel.trigger is TriggerKind.SCHEDULE:
        schedule = HttpxScheduleTransport() if schedule is None else schedule
        body = schedule.fire(token, session, run)
        elapsed = time.monotonic() - started
        return (SCHEDULE_FIRED, body, elapsed)

    message = channel.trigger_template.replace(RECORD_PLACEHOLDER, record_id)
    transport = HttpxTargetTransport() if transport is None else transport
    body = transport.send(token, session, run, [{"role": "user", "content": message}])
    return (message, body, time.monotonic() - started)
