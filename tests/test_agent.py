"""Agent state, failure-budget, and concurrent-session tests."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
 
import adaptive_teacher.agent as agent_module
import app as app_module
from adaptive_teacher.agent import execute_agent
from adaptive_teacher.models import MAX_LLM_CALLS, LearningState
from adaptive_teacher.prompts import supervisor_prompt
from adaptive_teacher.state import session_store


def test_answer_evaluation_persists_score_and_mastery(monkeypatch) -> None:
    replies = iter(
        [
            {
                "action": "AnswerEvaluator",
                "reason": "The student answered the previous question.",
                "tool_instruction": "Evaluate against the material.",
                "direct_response": "",
            },
            {
                "response": "Correct.",
                "score": 94,
                "mastery": True,
                "strong_topics": ["Topic A"],
            },
        ]
    )

    async def fake_call_llm(_messages):
        return next(replies)

    monkeypatch.setattr(agent_module, "call_llm", fake_call_llm)
    state = LearningState(session_id=str(uuid4()), material="Authoritative material")
    response, steps = asyncio.run(execute_agent(state, "My answer"))

    assert response == "Correct."
    assert len(steps) == 2
    assert state.latest_score == 94
    assert state.mastery is True
    _, user_prompt = supervisor_prompt(state, "Continue", 10)
    compact_state = json.loads(user_prompt)["state"]
    assert compact_state["latest_score"] == 94
    assert compact_state["mastery"] is True


def test_failed_attempt_still_consumes_one_call(monkeypatch) -> None:
    async def failing_call_llm(_messages):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(agent_module, "call_llm", failing_call_llm)
    state = LearningState(session_id=str(uuid4()))

    try:
        asyncio.run(execute_agent(state, "Material"))
    except TimeoutError:
        pass
    else:
        raise AssertionError("The simulated provider error should propagate to FastAPI.")
    assert state.llm_calls == 1


def test_concurrent_requests_cannot_exceed_session_limit(monkeypatch) -> None:
    call_count = 0

    async def slow_supervisor(_messages):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.02)
        return {
            "action": "Stop",
            "reason": "Only one call remained.",
            "tool_instruction": "",
            "direct_response": "Done.",
        }

    monkeypatch.setattr(agent_module, "call_llm", slow_supervisor)
    session_id = str(uuid4())
    state = session_store.get(session_id)
    state.llm_calls = MAX_LLM_CALLS - 1
    session_store.save(state)

    async def scenario() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            cookies={"adaptive_session_id": session_id},
        ) as async_client:
            return await asyncio.gather(
                async_client.post(
                    "/api/execute",
                    json={"prompt": "First concurrent turn"},
                ),
                async_client.post(
                    "/api/execute",
                    json={"prompt": "Second concurrent turn"},
                ),
            )

    responses = asyncio.run(scenario())
    assert all(response.status_code == 200 for response in responses)
    assert session_store.get(session_id).llm_calls == MAX_LLM_CALLS
    assert call_count == 1
    assert sorted(len(response.json()["steps"]) for response in responses) == [0, 1]


def test_public_error_does_not_expose_provider_details(client, monkeypatch) -> None:
    async def failing_agent(_state, _message):
        _state.llm_calls += 1
        raise RuntimeError("secret upstream diagnostic")

    monkeypatch.setattr(app_module, "execute_agent", failing_agent)
    response = client.post("/api/execute", json={"prompt": "Material"})

    assert response.status_code == 500
    assert response.json()["error"] == (
        "The learning agent could not complete this request. Please try again."
    )
    assert "secret upstream diagnostic" not in response.text
    assert response.headers["X-LLM-Calls-Used"] == "1"
    assert response.headers["X-Agent-Session-Id"]
    assert client.cookies.get("adaptive_session_id")


def test_reset_does_not_retain_an_orphan_execution_lock(client) -> None:
    attacker_supplied_id = str(uuid4())
    response = client.delete(
        "/api/session",
        headers={"Cookie": f"adaptive_session_id={attacker_supplied_id}"},
    )
    assert response.status_code == 200
    assert attacker_supplied_id not in session_store._executions
