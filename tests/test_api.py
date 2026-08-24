"""API contract and browser asset tests."""

from __future__ import annotations

from adaptive_teacher.models import MAX_LLM_CALLS
from adaptive_teacher.state import session_store


def test_root_serves_framework_free_interface(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Adaptive AI Teacher" in response.text
    assert '<script src="/app.js" defer></script>' in response.text

    script = client.get("/app.js")
    assert script.status_code == 200
    assert "LLMOD_API_KEY" not in script.text
    assert 'fetch("/api/execute"' in script.text


def test_required_information_endpoints(client) -> None:
    team = client.get("/api/team_info")
    assert team.status_code == 200
    assert set(team.json()) == {
        "group_batch_order_number",
        "team_name",
        "students",
    }
    assert len(team.json()["students"]) == 3

    agent = client.get("/api/agent_info")
    assert agent.status_code == 200
    assert {
        "description",
        "purpose",
        "prompt_template",
        "prompt_examples",
    } <= set(agent.json())

    architecture = client.get("/api/model_architecture")
    assert architecture.status_code == 200
    assert architecture.headers["content-type"] == "image/png"
    assert architecture.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_execute_has_exact_schema_trace_and_budget_headers(client) -> None:
    response = client.post(
        "/api/execute",
        json={"prompt": "I love basketball. Study material: Water freezes at zero degrees."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"status", "error", "response", "steps"}
    assert payload["status"] == "ok"
    assert payload["error"] is None
    assert isinstance(payload["response"], str)
    assert [step["module"] for step in payload["steps"]] == [
        "LearningSupervisor",
        "AnalyzeMaterial",
    ]
    for step in payload["steps"]:
        assert set(step) == {"module", "prompt", "response"}
        assert set(step["prompt"]) == {"System_prompt", "User_prompt"}

    assert response.headers["X-LLM-Calls-Used"] == "2"
    assert response.headers["X-LLM-Calls-Remaining"] == "14"
    assert response.headers["X-Agent-Session-Id"]
    assert client.cookies.get("adaptive_session_id")


def test_session_is_continuous_and_reset_creates_a_new_id(client) -> None:
    first = client.post(
        "/api/execute",
        json={"prompt": "I love basketball. Material: The Earth orbits the Sun."},
    )
    first_id = first.headers["X-Agent-Session-Id"]
    second = client.post("/api/execute", json={"prompt": "Quiz me"})
    assert second.headers["X-Agent-Session-Id"] == first_id
    assert second.headers["X-LLM-Calls-Used"] == "4"

    reset = client.delete("/api/session")
    assert reset.status_code == 200
    assert reset.json() == {"status": "ok"}

    third = client.post("/api/execute", json={"prompt": "New material: The moon is smaller than the Earth."})
    assert third.headers["X-Agent-Session-Id"] != first_id
    assert third.headers["X-LLM-Calls-Used"] == "2"


def test_hard_call_limit_never_invokes_another_model_call(client) -> None:
    first = client.post("/api/execute", json={"prompt": "Short study material"})
    session_id = first.headers["X-Agent-Session-Id"]
    state = session_store.get(session_id)
    state.llm_calls = MAX_LLM_CALLS
    session_store.save(state)

    limited = client.post("/api/execute", json={"prompt": "Continue"})
    assert limited.status_code == 200
    assert limited.json()["steps"] == []
    assert "16" in limited.json()["response"]
    assert limited.headers["X-LLM-Calls-Used"] == str(MAX_LLM_CALLS)
    assert limited.headers["X-LLM-Calls-Remaining"] == "0"


def test_execute_validation_always_uses_required_error_schema(client) -> None:
    cases = [
        client.post(
            "/api/execute",
            content="not json",
            headers={"Content-Type": "application/json"},
        ),
        client.post("/api/execute", json={}),
        client.post("/api/execute", json={"prompt": "   "}),
        client.post("/api/execute", json={"prompt": "x" * 30_001}),
    ]
    for response in cases:
        assert response.status_code == 400
        assert set(response.json()) == {"status", "error", "response", "steps"}
        assert response.json()["status"] == "error"
        assert isinstance(response.json()["error"], str)
        assert response.json()["response"] is None
        assert response.json()["steps"] == []
