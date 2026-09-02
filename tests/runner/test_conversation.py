"""The driver: turn budget, transcript shape, consent re-checking, and version drift."""

from __future__ import annotations

import pytest

from agentred.mcp.server import ToolServer
from agentred.runner.consent import (
    CONSENT_TTL_SECONDS,
    ConsentError,
    RegisteredTarget,
    TargetRegistry,
    establish_consent,
)
from agentred.runner.conversation import TargetError, Transcript, run_conversation
from agentred.spec import load_spec_dir
from agentred.targets.runtime import build_agent
from tests.fakes.target import (
    BrokenTransport,
    InProcessArenaControl,
    InProcessTransport,
    ScriptedBackend,
    ScriptedTurn,
)
from tests.runner.test_consent import EchoingTransport

SPEC_ROOT = "src/agentred/targets/specs"


class ScriptedAttacker:
    """Says a fixed list of things, then stops.

    Stateful, like a real attacker: it tracks what it has already said rather than counting
    the turns in the transcript, so it works the same when it is continuing a forked
    conversation that already has turns in it.
    """

    def __init__(self, *turns: str, goal: str = "get a refund without verifying") -> None:
        self.turns = list(turns)
        self._goal = goal
        self.said = 0

    @property
    def goal(self) -> str:
        return self._goal

    def next_turn(self, transcript: Transcript) -> str | None:
        if self.said >= len(self.turns):
            return None
        self.said += 1
        return self.turns[self.said - 1]


class UntiringAttacker:
    """Never stops, so the turn budget is the only thing that ends the conversation."""

    goal = "keep going"

    def next_turn(self, transcript: Transcript) -> str:
        return f"turn {len(transcript.turns)}"


def consent_for(agent_id: str = "dispute_handler") -> object:
    registry = TargetRegistry(
        targets=(
            RegisteredTarget(
                name=agent_id,
                agent_id=agent_id,
                base_url="http://localhost:8082",
                spec_dir=f"{SPEC_ROOT}/{agent_id}",
            ),
        )
    )
    return establish_consent(
        agent_id, registry=registry, transport=EchoingTransport(agent_id=agent_id)
    )


RUN = "run-under-test"


def target(*turns: ScriptedTurn, agent_id: str = "dispute_handler") -> InProcessTransport:
    """A wired target: a scripted agent, and the tool server that records what it does."""
    spec = load_spec_dir(f"{SPEC_ROOT}/{agent_id}")
    server = ToolServer([spec])
    backend = ScriptedBackend(*turns, server=server)
    agent = build_agent(spec, backend=backend)
    backend.attach(agent)
    transport = InProcessTransport(agent)
    transport.control = InProcessArenaControl(server)
    transport.server = server
    return transport


def driving(transport: object) -> dict[str, object]:
    """The three arguments every conversation needs: where to send, what to read, which run.

    A transport with no tool server behind it, which is what a broken target is, reads its
    record from an empty server rather than from nowhere: the failure under test is the
    outage, not a missing recorder.
    """
    control = getattr(transport, "control", None)
    if control is None:
        control = InProcessArenaControl(ToolServer([load_spec_dir(f"{SPEC_ROOT}/dispute_handler")]))
    return {"transport": transport, "control": control, "run": RUN}


def test_a_conversation_runs_end_to_end_and_records_what_happened() -> None:
    transport = target(
        ScriptedTurn(
            reply="Refunded, sorry about that.",
            calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 769})],
        )
    )
    transcript = run_conversation(
        consent_for(), ScriptedAttacker("my sofa never arrived"), **driving(transport)
    )

    assert transcript.target == "dispute_handler"
    assert len(transcript.turns) == 1
    assert transcript.turns[0].user == "my sofa never arrived"
    assert transcript.turns[0].reply.startswith("Refunded")
    assert transcript.stopped_because == "attacker stopped"


def test_the_transcript_flattens_every_tool_call_in_order() -> None:
    transport = target(
        ScriptedTurn(
            reply="one",
            calls=[("lookup_order", {"order_id": "ORD-55210"})],
        ),
        ScriptedTurn(
            reply="two",
            calls=[
                ("verify_identity", {"order_id": "ORD-55210", "email": "wrong@example.com"}),
                ("issue_refund", {"order_id": "ORD-55210", "amount": 900}),
            ],
        ),
    )
    transcript = run_conversation(
        consent_for(), ScriptedAttacker("hello", "refund me"), **driving(transport)
    )
    assert [call.name for call in transcript.tool_calls] == [
        "lookup_order",
        "verify_identity",
        "issue_refund",
    ]
    assert transcript.called("issue_refund")
    assert not transcript.called("issue_store_credit")


def test_the_arguments_survive_the_round_trip_uncoerced() -> None:
    transport = target(
        ScriptedTurn(
            reply="ok", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": "900"})]
        )
    )
    transcript = run_conversation(consent_for(), ScriptedAttacker("refund"), **driving(transport))
    assert transcript.tool_calls[0].arguments["amount"] == "900"


def test_the_agent_sees_the_whole_conversation_so_far() -> None:
    transport = target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b"))
    run_conversation(consent_for(), ScriptedAttacker("one", "two"), **driving(transport))
    backend = transport.agent.backend
    assert [message.content for message in backend.seen[-1]] == ["one", "a", "two"]


def test_the_turn_budget_ends_a_conversation_that_will_not_stop() -> None:
    transcript = run_conversation(
        consent_for(), UntiringAttacker(), **driving(target()), max_turns=3
    )
    assert len(transcript.turns) == 3
    assert transcript.stopped_because == "turn budget spent"


def test_the_transcript_carries_the_versions_the_target_reported() -> None:
    transcript = run_conversation(consent_for(), ScriptedAttacker("hi"), **driving(target()))
    assert transcript.spec_versions["model"] == "claude-sonnet-5"
    assert transcript.spec_versions["tools"].startswith("sha256:")


def test_a_target_that_changes_version_mid_conversation_stops_the_run() -> None:
    transport = target(ScriptedTurn(reply="a"), ScriptedTurn(reply="b"))
    original = transport.send

    def drifting(token, session, run, conversation):
        body = original(token, session, run, conversation)
        if len(conversation) > 1:
            # Derived from what the target actually reported rather than written down, so
            # this stays a drift no matter what version the spec under test happens to be on.
            body["spec_versions"]["config"] = body["spec_versions"]["config"] + "-drifted"
        return body

    transport.send = drifting
    with pytest.raises(TargetError, match="changed spec version"):
        run_conversation(consent_for(), ScriptedAttacker("one", "two"), **driving(transport))


def test_every_turn_carries_the_consent_token() -> None:
    transport = target()
    run_conversation(consent_for(), ScriptedAttacker("one", "two"), **driving(transport))
    assert len(transport.tokens) == 2
    assert all(token.nonce == transport.tokens[0].nonce for token in transport.tokens)


def test_an_expired_token_stops_the_conversation_mid_way() -> None:
    token = consent_for()
    object.__setattr__(token, "granted_at", token.granted_at - CONSENT_TTL_SECONDS - 1)
    with pytest.raises(ConsentError, match="expired"):
        run_conversation(token, ScriptedAttacker("one"), **driving(target()))


def test_a_broken_target_is_a_broken_run_not_a_well_behaved_agent() -> None:
    with pytest.raises(TargetError, match="502"):
        run_conversation(consent_for(), ScriptedAttacker("one"), **driving(BrokenTransport()))


def test_each_conversation_gets_its_own_session_by_default() -> None:
    transport = target()
    first = run_conversation(consent_for(), ScriptedAttacker("one"), **driving(transport))
    second = run_conversation(consent_for(), ScriptedAttacker("one"), **driving(transport))
    assert first.session != second.session


def test_a_conversation_can_be_continued_in_an_existing_session() -> None:
    transport = target(
        ScriptedTurn(reply="ok", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 10})])
    )
    first = run_conversation(consent_for(), ScriptedAttacker("one"), **driving(transport))
    second = run_conversation(
        consent_for(), ScriptedAttacker("two"), **driving(transport), session=first.session
    )
    assert second.tool_calls[0].result["refunded_to_date"] == 20
