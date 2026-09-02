"""The HTTP surface: the challenge, a turn, the connector it points the agent at."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentred.mcp.server import ToolServer
from agentred.spec import load_spec_dir
from agentred.targets.runtime import TargetAgent, build_agent, build_app
from tests.fakes.target import ScriptedBackend, ScriptedTriggerBackend, ScriptedTurn

SPEC_ROOT = "src/agentred/targets/specs"
TOOL_SERVER_URL = "http://tools.test:8090"
ORDER = "ORD-55210"


def server_for(name: str) -> ToolServer:
    return ToolServer([load_spec_dir(f"{SPEC_ROOT}/{name}")])


def agent_with(
    *turns: ScriptedTurn, name: str = "dispute_handler", server: ToolServer | None = None
) -> TargetAgent:
    server = server_for(name) if server is None else server
    backend = ScriptedBackend(*turns, server=server)
    agent = build_agent(
        load_spec_dir(f"{SPEC_ROOT}/{name}"), backend=backend, tool_server_url=TOOL_SERVER_URL
    )
    backend.attach(agent)
    return agent


def client_for(
    *turns: ScriptedTurn, name: str = "dispute_handler", server: ToolServer | None = None
) -> TestClient:
    return TestClient(build_app(agent_with(*turns, name=name, server=server)))


def turn(session: str = "s1", run: str = "r1", said: str = "refund me") -> dict[str, object]:
    return {"session": session, "run": run, "conversation": [{"role": "user", "content": said}]}


def test_the_challenge_is_echoed_with_the_identity_the_mode_and_the_tool_server() -> None:
    response = client_for().get("/challenge", params={"nonce": "abc123"})
    assert response.status_code == 200
    body = response.json()
    assert body["challenge"] == "abc123"
    assert body["agent_id"] == "dispute_handler"
    assert body["mode"] == "test"
    assert body["tool_server"] == TOOL_SERVER_URL


def test_the_challenge_says_which_spec_this_process_is_actually_holding() -> None:
    """A target loads its spec once, so what is on disk is not what is running. See D24."""
    spec = load_spec_dir(f"{SPEC_ROOT}/dispute_handler")
    body = client_for().get("/challenge", params={"nonce": "abc123"}).json()
    assert body["versions"] == spec.version_tuple.model_dump(mode="json")


def test_a_challenge_without_a_nonce_is_rejected() -> None:
    assert client_for().get("/challenge").status_code == 422


def test_a_reply_carries_prose_and_versions_and_no_account_of_what_it_did() -> None:
    """ADR-0005: what the agent did is observed, never volunteered."""
    server = server_for("dispute_handler")
    scripted = ScriptedTurn(
        reply="That is refunded.",
        calls=[
            ("get_order", {"order_id": ORDER}),
            ("issue_refund", {"order_id": ORDER, "amount": 40}),
        ],
    )
    body = client_for(scripted, server=server).post("/chat", json=turn()).json()

    assert body["reply"] == "That is refunded."
    assert "tool_calls" not in body
    assert [call.name for call in server.recorder.calls("r1", "s1")] == [
        "get_order",
        "issue_refund",
    ]


def test_the_recorded_call_carries_the_arguments_as_the_model_sent_them() -> None:
    server = server_for("dispute_handler")
    scripted = ScriptedTurn(
        reply="Done.", calls=[("issue_refund", {"order_id": ORDER, "amount": "1,200"})]
    )
    client_for(scripted, server=server).post("/chat", json=turn())

    assert server.recorder.calls("r1", "s1")[0].arguments["amount"] == "1,200"


def test_a_reply_reports_the_versions_it_belongs_to() -> None:
    body = client_for().post("/chat", json=turn()).json()
    assert set(body["spec_versions"]) == {"config", "policy", "model", "tools"}
    assert body["spec_versions"]["model"] == "claude-sonnet-5"
    assert body["spec_versions"]["tools"].startswith("sha256:")


def test_two_sessions_do_not_share_a_world() -> None:
    server = server_for("dispute_handler")
    scripted = ScriptedTurn(
        reply="ok", calls=[("issue_refund", {"order_id": ORDER, "amount": 100})]
    )
    client = client_for(scripted, server=server)

    client.post("/chat", json=turn(session="s1"))
    client.post("/chat", json=turn(session="s2"))

    assert server.recorder.calls("r1", "s1")[0].result["refunded_to_date"] == 100
    assert server.recorder.calls("r1", "s2")[0].result["refunded_to_date"] == 100


def test_one_session_accumulates_its_own_damage() -> None:
    server = server_for("dispute_handler")
    scripted = ScriptedTurn(
        reply="ok", calls=[("issue_refund", {"order_id": ORDER, "amount": 100})]
    )
    client = client_for(scripted, server=server)

    client.post("/chat", json=turn(session="s1"))
    client.post("/chat", json=turn(session="s1"))

    assert server.recorder.calls("r1", "s1")[1].result["refunded_to_date"] == 200


def test_the_connector_names_the_agent_the_run_and_the_session() -> None:
    """Attribution comes from the URL, so an agent cannot misfile what it did."""
    agent = agent_with()
    session = agent.session("s9")
    session.run = "r7"

    assert agent.connector_url(session) == f"{TOOL_SERVER_URL}/dispute_handler/r7/s9"


def test_a_conversation_ending_on_the_agent_is_refused() -> None:
    response = client_for().post(
        "/chat",
        json={
            "session": "s1",
            "run": "r1",
            "conversation": [{"role": "assistant", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert "user turn" in response.json()["detail"]


def test_an_empty_conversation_is_refused() -> None:
    response = client_for().post("/chat", json={"session": "s1", "run": "r1", "conversation": []})
    assert response.status_code == 422


def test_a_turn_without_a_run_is_refused() -> None:
    """A call recorded under no run is a call nothing will read."""
    response = client_for().post(
        "/chat", json={"session": "s1", "conversation": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 422


def test_the_backend_is_given_the_whole_conversation() -> None:
    server = server_for("cart_recovery")
    backend = ScriptedBackend(ScriptedTurn(reply="ok"), server=server)
    agent = build_agent(
        load_spec_dir(f"{SPEC_ROOT}/cart_recovery"),
        backend=backend,
        tool_server_url=TOOL_SERVER_URL,
    )
    backend.attach(agent)

    TestClient(build_app(agent)).post(
        "/chat",
        json={
            "session": "s1",
            "run": "r1",
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
        assert agent.spec.config.agent_id == name


def test_health_reports_the_mode() -> None:
    body = client_for().get("/health").json()
    assert body == {"status": "ok", "agent_id": "dispute_handler", "mode": "test"}


def scheduled_client(*turns: ScriptedTurn) -> tuple[TestClient, ScriptedTriggerBackend]:
    """A target with a scheduled entry point, and the backend behind it."""
    server = server_for("cart_recovery")
    backend = ScriptedTriggerBackend(*turns, server=server)
    agent = build_agent(
        load_spec_dir(f"{SPEC_ROOT}/cart_recovery"),
        backend=backend,
        tool_server_url=TOOL_SERVER_URL,
    )
    backend.attach(agent)
    return TestClient(build_app(agent)), backend


def test_firing_a_schedule_needs_no_conversation() -> None:
    """The whole point of the channel: nobody talks to this agent, and nothing pretends to."""
    client, backend = scheduled_client(
        ScriptedTurn(reply="messaged one shopper", calls=[("list_abandoned_carts", {})])
    )
    response = client.post("/trigger", json={"session": "s1", "run": "r1"})
    assert response.status_code == 200
    assert response.json()["output"] == "messaged one shopper"
    assert backend.firings == 1
    assert backend.seen == []


def test_a_firing_carries_the_spec_versions_it_belongs_to() -> None:
    """Read from the spec on disk, not pinned to a literal.

    What this asserts is that a firing reports which spec it loaded, so that a stale server
    is refused at preflight rather than after a paid-for suite. Pinning the version here
    would make every legitimate version bump fail a test about something else, which teaches
    whoever bumps it to edit the assertion.
    """
    spec = load_spec_dir(f"{SPEC_ROOT}/cart_recovery")
    client, _ = scheduled_client(ScriptedTurn(reply="done"))
    body = client.post("/trigger", json={"session": "s1", "run": "r1"}).json()
    assert body["spec_versions"]["config"] == spec.config.version
    assert body["session"] == "s1"


def test_a_firing_reports_what_it_cost() -> None:
    client, _ = scheduled_client(ScriptedTurn(reply="done"))
    body = client.post("/trigger", json={"session": "s1", "run": "r1"}).json()
    assert body["usage"]["input_tokens"] == 1.0


def test_a_firing_leaves_a_checkpoint_a_fork_can_branch_from() -> None:
    client, backend = scheduled_client(ScriptedTurn(reply="done"))
    client.post("/trigger", json={"session": "s1", "run": "r1"})
    assert len(backend.agent.session("s1").checkpoints) == 1


def test_an_agent_with_no_schedule_is_refused_rather_than_answered_with_nothing() -> None:
    """A silent empty firing would report an attack as withstood that never arrived."""
    response = client_for(name="cart_recovery").post(
        "/trigger", json={"session": "s1", "run": "r1"}
    )
    assert response.status_code == 404
    assert "no scheduled entry point" in response.json()["detail"]


def test_a_firing_without_a_run_is_refused() -> None:
    """Unattributed calls are calls nobody is recording."""
    assert scheduled_client()[0].post("/trigger", json={"session": "s1"}).status_code == 422
