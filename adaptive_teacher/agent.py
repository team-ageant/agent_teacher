"""Dynamic ReAct supervisor execution and learning-state updates."""

from __future__ import annotations

from typing import Any, cast

from .llm import call_llm
from .models import (
    MAX_LLM_CALLS,
    TOOL_NAMES,
    ChatTurn,
    LearningState,
    SupervisorDecision,
    ToolName,
    TraceStep,
) 
from .prompts import supervisor_prompt, tool_prompt
from .retrieval import retrieve_passages

 
def _string_array(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)][:20]


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def _validate_decision(raw: dict[str, Any]) -> SupervisorDecision:
    action = raw.get("action")
    if not isinstance(action, str) or action not in TOOL_NAMES:
        raise ValueError("LearningSupervisor selected an unknown tool.")
    return SupervisorDecision(
        action=cast(ToolName, action),
        reason=raw.get("reason") if isinstance(raw.get("reason"), str) else "",
        tool_instruction=(
            raw.get("tool_instruction") if isinstance(raw.get("tool_instruction"), str) else ""
        ),
        direct_response=(
            raw.get("direct_response") if isinstance(raw.get("direct_response"), str) else ""
        ),
    )


def _update_state(state: LearningState, action: ToolName, result: dict[str, Any]) -> None:
    material = result.get("material")
    if isinstance(material, str) and material.strip():
        state.material = material.strip()[:30_000]
    state.interests = _unique(state.interests + _string_array(result.get("interests")))
    state.topics = _unique(state.topics + _string_array(result.get("topics")))
    state.weak_topics = _unique(state.weak_topics + _string_array(result.get("weak_topics")))
    state.strong_topics = _unique(state.strong_topics + _string_array(result.get("strong_topics")))
    score = result.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool):
        state.latest_score = min(100.0, max(0.0, float(score)))
    mastery = result.get("mastery")
    if isinstance(mastery, bool):
        state.mastery = mastery
    state.last_action = action


async def execute_agent(state: LearningState, message: str) -> tuple[str, list[dict[str, Any]]]:
    steps: list[TraceStep] = []
    if state.llm_calls >= MAX_LLM_CALLS:
        response = (
            "We reached the limit of 16 model calls in this session. You can start a new session to continue learning."
        )
        state.history.extend(
            [
                ChatTurn(role="student", content=message),
                ChatTurn(role="teacher", content=response),
            ]
        )
        state.history = state.history[-20:]
        return response, []

    state.history.append(ChatTurn(role="student", content=message))
    retrieved_passages = await retrieve_passages(message)
    calls_after_supervisor = MAX_LLM_CALLS - state.llm_calls - 1
    supervisor_system, supervisor_user = supervisor_prompt(
        state, message, max(0, calls_after_supervisor), retrieved_passages
    )
    # Reserve each call before dispatch. Even a timeout or invalid provider
    # response consumed an attempted request from the assignment budget.
    state.llm_calls += 1
    supervisor_raw = await call_llm(
        [
            {"role": "system", "content": supervisor_system},
            {"role": "user", "content": supervisor_user},
        ]
    )
    decision = _validate_decision(supervisor_raw)
    steps.append(
        TraceStep(
            module="LearningSupervisor",
            system_prompt=supervisor_system,
            user_prompt=supervisor_user,
            response=supervisor_raw,
        )
    )

    response = decision.direct_response
    if decision.action not in {"RespondDirectly", "Stop"}:
        if state.llm_calls >= MAX_LLM_CALLS:
            response = (
                "I reached the call limit before running the next tool. Open a new session to continue with a full budget."
            )
            state.last_action = "Stop"
        else:
            tool_system, tool_user = tool_prompt(
                decision.action,
                state,
                message,
                decision.tool_instruction,
                retrieved_passages,
            )
            state.llm_calls += 1
            tool_result = await call_llm(
                [
                    {"role": "system", "content": tool_system},
                    {"role": "user", "content": tool_user},
                ]
            )
            steps.append(
                TraceStep(
                    module=decision.action,
                    system_prompt=tool_system,
                    user_prompt=tool_user,
                    response=tool_result,
                )
            )
            _update_state(state, decision.action, tool_result)
            tool_response = tool_result.get("response")
            response = (
                tool_response
                if isinstance(tool_response, str)
                else "The action was completed, but no response was returned to display."
            )
    else:
        state.last_action = decision.action
        if not response:
            response = (
                "We finished the learning process for today." if decision.action == "Stop" else "I am ready to continue."
            )

    state.history.append(ChatTurn(role="teacher", content=response))
    state.history = state.history[-20:]
    return response, [step.as_dict() for step in steps]
