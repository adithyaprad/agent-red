"""The HTTP surface: the challenge, a turn, the tool-call log and session isolation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentred.spec import load_spec_dir
from agentred.targets.runtime import TargetAgent, build_agent, build_app
from agentred.targets.tools import CART_TOOLS, DISPUTE_TOOLS
from agentred.targets.tools.base import ToolSet, tool
from tests.fakes.target import ScriptedBackend, ScriptedTurn

SPEC_ROOT = "src/agentred/targets/specs"


def agent_with(*turns: ScriptedTurn, name: str = "dispute_handler") -> TargetAgent:
    backend = ScriptedBackend(*turns)
    agent = build_agent(load_spec_dir(f"{SPEC_ROOT}/{name}"), backend=backend)
    backend.attach(agent)
    return agent


def client_for(*turns: ScriptedTurn, name: str = "dispute_handler") -> TestClient:
    return TestClient(build_app(agent_with(*turns, name=name)))


def test_the_challenge_is_echoed_with_the_identity_and_the_mode() -> None:
    response = client_for().get("/challenge", params={"nonce": "abc123"})
    assert response.status_code == 200
    assert response.json() == {
        "challenge": "abc123",
        "agent_id": "dispute_handler",
        "mode": "test",
    }


def test_a_challenge_without_a_nonce_is_rejected() -> None:
    assert client_for().get("/challenge").status_code == 422


def test_a_turn_returns_the_reply_and_the_tool_calls_it_made() -> None:
    turn = ScriptedTurn(
        reply="That is refunded.",
        calls=[
            ("verify_identity", {"order_id": "ORD-55210", "email": "maya.lindqvist@example.com"}),
            ("issue_refund", {"order_id": "ORD-55210", "amount": 40}),
        ],
    )
    response = client_for(turn).post(
        "/chat",
        json={"session": "s1", "conversation": [{"role": "user", "content": "refund me"}]},
    )
    body = response.json()
    assert body["reply"] == "That is refunded."
    assert [call["name"] for call in body["tool_calls"]] == ["verify_identity", "issue_refund"]
    assert body["tool_calls"][1]["arguments"] == {"order_id": "ORD-55210", "amount": 40}
    assert body["tool_calls"][1]["result"]["refunded"] == 40


def test_the_tool_call_log_carries_the_arguments_as_the_model_sent_them() -> None:
    turn = ScriptedTurn(
        reply="Done.", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": "1,200"})]
    )
    body = (
        client_for(turn)
        .post(
            "/chat",
            json={"session": "s1", "conversation": [{"role": "user", "content": "refund"}]},
        )
        .json()
    )
    assert body["tool_calls"][0]["arguments"]["amount"] == "1,200"


def test_a_reply_reports_the_versions_it_belongs_to() -> None:
    body = (
        client_for()
        .post("/chat", json={"session": "s1", "conversation": [{"role": "user", "content": "hi"}]})
        .json()
    )
    assert set(body["spec_versions"]) == {"config", "policy", "model", "tools"}
    assert body["spec_versions"]["model"] == "claude-sonnet-5"
    assert body["spec_versions"]["tools"].startswith("sha256:")


def test_the_tool_call_log_covers_this_turn_only() -> None:
    client = client_for(
        ScriptedTurn(reply="one", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 5})]),
        ScriptedTurn(reply="two"),
    )
    payload = {"session": "s1", "conversation": [{"role": "user", "content": "a"}]}
    first = client.post("/chat", json=payload).json()
    second = client.post(
        "/chat",
        json={
            "session": "s1",
            "conversation": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "b"},
            ],
        },
    ).json()
    assert len(first["tool_calls"]) == 1
    assert second["tool_calls"] == []


def test_two_sessions_do_not_share_a_world() -> None:
    turn = ScriptedTurn(
        reply="ok", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 100})]
    )
    client = client_for(turn)
    first = client.post(
        "/chat", json={"session": "s1", "conversation": [{"role": "user", "content": "a"}]}
    ).json()
    second = client.post(
        "/chat", json={"session": "s2", "conversation": [{"role": "user", "content": "a"}]}
    ).json()
    assert first["tool_calls"][0]["result"]["refunded_to_date"] == 100
    assert second["tool_calls"][0]["result"]["refunded_to_date"] == 100


def test_one_session_accumulates_its_own_damage() -> None:
    turn = ScriptedTurn(
        reply="ok", calls=[("issue_refund", {"order_id": "ORD-55210", "amount": 100})]
    )
    client = client_for(turn)
    payload = {"session": "s1", "conversation": [{"role": "user", "content": "a"}]}
    client.post("/chat", json=payload)
    second = client.post("/chat", json=payload).json()
    assert second["tool_calls"][0]["result"]["refunded_to_date"] == 200


def test_a_conversation_ending_on_the_agent_is_refused() -> None:
    response = client_for().post(
        "/chat",
        json={"session": "s1", "conversation": [{"role": "assistant", "content": "hello"}]},
    )
    assert response.status_code == 400
    assert "user turn" in response.json()["detail"]


def test_an_empty_conversation_is_refused() -> None:
    response = client_for().post("/chat", json={"session": "s1", "conversation": []})
    assert response.status_code == 422


def test_the_backend_is_given_the_whole_conversation() -> None:
    backend = ScriptedBackend(ScriptedTurn(reply="ok"))
    agent = build_agent(load_spec_dir(f"{SPEC_ROOT}/cart_recovery"), backend=backend)
    backend.attach(agent)
    TestClient(build_app(agent)).post(
        "/chat",
        json={
            "session": "s1",
            "conversation": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        },
    )
    assert [message.content for message in backend.seen[0]] == ["a", "b", "c"]


def test_both_shipped_specs_build() -> None:
    for name in ("cart_recovery", "dispute_handler"):
        agent = agent_with(name=name)
        assert agent.tools.names == {tool.name for tool in agent.spec.config.tools}


def test_a_declared_tool_with_no_implementation_is_refused() -> None:
    spec = load_spec_dir(f"{SPEC_ROOT}/dispute_handler")
    with pytest.raises(ValueError, match="no implementation"):
        TargetAgent(spec=spec, tools=CART_TOOLS, backend=ScriptedBackend())


def test_an_implemented_tool_the_config_does_not_declare_is_refused() -> None:
    @tool("wire_transfer")
    def wire_transfer(world: object, arguments: dict[str, object]) -> dict[str, object]:
        return {}

    spec = load_spec_dir(f"{SPEC_ROOT}/dispute_handler")
    tools = ToolSet(*DISPUTE_TOOLS.implementations.values(), wire_transfer)
    with pytest.raises(ValueError, match="wire_transfer"):
        TargetAgent(spec=spec, tools=tools, backend=ScriptedBackend())


def test_a_spec_with_no_registered_tools_is_refused() -> None:
    spec = load_spec_dir(f"{SPEC_ROOT}/dispute_handler")
    unknown = spec.model_copy(
        update={"config": spec.config.model_copy(update={"agent_id": "warehouse_bot"})}
    )
    with pytest.raises(KeyError, match="warehouse_bot"):
        build_agent(unknown, backend=ScriptedBackend())


def test_health_reports_the_mode() -> None:
    body = client_for().get("/health").json()
    assert body == {"status": "ok", "agent_id": "dispute_handler", "mode": "test"}
