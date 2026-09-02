"""Branching one conversation into several, from a shared prefix.

Attacks share openings. Establishing a plausible customer, getting an order looked up and
building a little rapport takes three turns that are identical across a dozen different
final asks, and paying for them a dozen times is both slow and expensive.

A fork copies the target's session: the same world state, the same cached model prefix, and
a separate future. The branches then differ by the turn that was changed and by nothing
else, which is what makes comparing them worth anything. Prompt caching is the money saving;
comparability is the reason this is in `runner/` rather than an optimisation somewhere.

Forking reaches the target, so every function here takes a `ConsentToken`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from agentred.mcp.control import ArenaControl, HttpxArenaControl
from agentred.runner.consent import ConsentToken
from agentred.runner.conversation import (
    DEFAULT_MAX_TURNS,
    Attacker,
    HttpxTargetTransport,
    TargetTransport,
    Transcript,
    new_session_id,
    run_conversation,
)


@dataclass(frozen=True)
class Branch:
    """One continuation of a shared prefix.

    Attributes:
        attacker: What says the differing turns.
        label: How this branch is named on the scorecard, for example the technique it
            varies. Empty is allowed but makes the results harder to read.
    """

    attacker: Attacker
    label: str = ""


def prefix_of(transcript: Transcript, turns: int) -> Transcript:
    """The first `turns` exchanges of a conversation, as a conversation of its own.

    Args:
        transcript: The conversation to cut.
        turns: How many exchanges to keep.

    Returns:
        A deep copy, so continuing the prefix cannot mutate the transcript it came from.

    Raises:
        ValueError: If `turns` is not between one and the length of the conversation. A
            zero-turn prefix is not a fork, it is a new conversation, and asking for more
            turns than exist means the caller believes something false about the transcript.
    """
    if not 1 <= turns <= len(transcript.turns):
        raise ValueError(
            f"cannot take a {turns}-turn prefix of a {len(transcript.turns)}-turn conversation"
        )
    return Transcript(
        target=transcript.target,
        session=transcript.session,
        goal=transcript.goal,
        turns=copy.deepcopy(transcript.turns[:turns]),
        spec_versions=dict(transcript.spec_versions),
        stopped_because="",
    )


def fork_conversation(
    token: ConsentToken,
    transcript: Transcript,
    attacker: Attacker,
    *,
    run: str,
    at_turn: int,
    max_turns: int = DEFAULT_MAX_TURNS,
    transport: TargetTransport | None = None,
    control: ArenaControl | None = None,
) -> Transcript:
    """Continue a conversation from turn `at_turn` along a different line.

    Args:
        token: Proof that the target consented.
        transcript: The conversation to branch.
        attacker: What says the new turns.
        run: The run both conversations belong to.
        at_turn: How many exchanges of the original to keep. One keeps the opening only.
        max_turns: Ceiling on new turns, not counting the shared prefix.
        transport: How turns are sent. Defaults to HTTP.
        control: How the tool server's record and worlds are reached. Defaults to HTTP.

    Returns:
        The branch, including the shared prefix, with its own session and its own copy of
        the merchant's world.

    Raises:
        ValueError: If `at_turn` does not name a prefix of this conversation.
        ConsentError: If the token has expired.
        TargetError: If the target refuses the fork or cannot be reached.
    """
    transport = HttpxTargetTransport() if transport is None else transport
    control = HttpxArenaControl(token.target.control_url) if control is None else control
    prefix = prefix_of(transcript, at_turn)
    branch_session = new_session_id()

    token.require_live()
    # Two halves, for two reasons. The target branches the cached model prefix, which is
    # what makes the fork cheap. The tool server branches the world, which is what makes the
    # branches comparable: without it the branch would inherit money spent by turns it does
    # not contain (ADR-0002).
    transport.fork(token, transcript.session, branch_session, at_turn)
    control.branch(transcript.session, branch_session, at_turn)
    prefix.goal = attacker.goal

    return run_conversation(
        token,
        attacker,
        run=run,
        max_turns=max_turns,
        transport=transport,
        control=control,
        session=branch_session,
        resume=prefix,
    )


def fan_out(
    token: ConsentToken,
    transcript: Transcript,
    branches: list[Branch],
    *,
    run: str,
    at_turn: int,
    max_turns: int = DEFAULT_MAX_TURNS,
    transport: TargetTransport | None = None,
    control: ArenaControl | None = None,
) -> list[Transcript]:
    """Run several continuations of the same prefix.

    Sequential rather than concurrent. The target holds one world per session and the model
    route is rate limited, so the parallelism that matters is between conversations at the
    suite level, not between branches of one.

    Args:
        token: Proof that the target consented.
        transcript: The conversation to branch.
        branches: The continuations to try.
        run: The run every branch belongs to.
        at_turn: How many exchanges of the original each branch keeps.
        max_turns: Ceiling on new turns per branch.
        transport: How turns are sent. Defaults to HTTP.
        control: How the tool server is reached. Defaults to HTTP.

    Returns:
        One transcript per branch, in the order the branches were given.
    """
    return [
        fork_conversation(
            token,
            transcript,
            branch.attacker,
            run=run,
            at_turn=at_turn,
            max_turns=max_turns,
            transport=transport,
            control=control,
        )
        for branch in branches
    ]
