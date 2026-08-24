"""Asynchronous LLMod chat and embedding clients with a free demo fallback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import httpx

from .config import get_settings

LlmMessage = dict[str, str]


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("The model returned invalid JSON.")


def _headers(api_key: str, custom_header: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if custom_header:
        headers[custom_header] = api_key
    return headers


async def call_llm(messages: list[LlmMessage]) -> dict[str, Any]:
    settings = get_settings()
    if settings.demo_mode:
        return _mock_llm(messages)
    if not settings.llmod_api_key:
        if settings.production:
            raise RuntimeError("LLMOD_API_KEY is required in production.")
        return _mock_llm(messages)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            settings.chat_completions_url,
            headers=_headers(settings.llmod_api_key, settings.llmod_api_key_header),
            json={
                "model": settings.llmod_model,
                "messages": messages,
                "temperature": 0.25,
                "response_format": {"type": "json_object"},
            },
        )
    if not response.is_success:
        raise RuntimeError(f"LLMod request failed with HTTP {response.status_code}.")

    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    content: Any = None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLMod returned an empty response.")
    return _extract_json(content)


async def create_embeddings(input_value: str | list[str]) -> list[list[float]]:
    values = input_value if isinstance(input_value, list) else [input_value]
    if not values or not all(isinstance(value, str) for value in values):
        raise ValueError("Embedding input must contain at least one string.")

    settings = get_settings()
    dim = settings.llmod_embedding_dimensions or 1024
    if settings.demo_mode:
        return [_mock_embedding(value, dim) for value in values]
    if not settings.llmod_api_key:
        if settings.production:
            raise RuntimeError("LLMOD_API_KEY is required in production.")
        return [_mock_embedding(value, dim) for value in values]

    payload: dict[str, Any] = {"model": settings.llmod_embedding_model, "input": values}
    if settings.llmod_embedding_dimensions:
        payload["dimensions"] = settings.llmod_embedding_dimensions

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            settings.embeddings_url,
            headers=_headers(settings.llmod_api_key, settings.llmod_api_key_header),
            json=payload,
        )
    if not response.is_success:
        raise RuntimeError(f"LLMod embeddings request failed with HTTP {response.status_code}.")

    res_payload = response.json()
    data = res_payload.get("data") if isinstance(res_payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("LLMod returned an invalid embeddings response.")
    ordered = sorted(
        (item for item in data if isinstance(item, dict)),
        key=lambda item: int(item.get("index", 0)),
    )
    embeddings = [item.get("embedding") for item in ordered]
    if len(embeddings) != len(values) or not all(
        isinstance(embedding, list) for embedding in embeddings
    ):
        raise RuntimeError("LLMod returned an invalid embeddings response.")
    return embeddings  # type: ignore[return-value]


def _mock_embedding(value: str, dimension: int = 1024) -> list[float]:
    vector = [0.0] * dimension
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    for index, byte in enumerate(value.encode("utf-8")):
        vector[(index * 31 + digest[index % len(digest)]) % len(vector)] += (byte % 97) / 97
    magnitude = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / magnitude for item in vector]


def _mock_llm(messages: list[LlmMessage]) -> dict[str, Any]:
    system = messages[0].get("content", "") if messages else ""
    raw_user = messages[-1].get("content", "") if messages else ""
    try:
        user_payload = json.loads(raw_user)
    except json.JSONDecodeError:
        user_payload = {}
    if not isinstance(user_payload, dict):
        user_payload = {}
    student_message = str(user_payload.get("student_message") or raw_user)
    state = user_payload.get("state")
    if not isinstance(state, dict):
        state = {}

    if system.startswith("You are LearningSupervisor,"):
        lowered = student_message.lower()
        remaining = int(user_payload.get("calls_remaining_after_supervisor", 0) or 0)
        if remaining <= 0:
            return {
                "action": "Stop",
                "reason": "The session has no remaining call for another tool.",
                "tool_instruction": "",
                "direct_response": "We have reached the end of the API call budget for this session. You can start a new session to continue.",
            }
        if any(word in lowered for word in ("stop", "end", "halt", "finish")):
            return {
                "action": "Stop",
                "reason": "The student asked to end the learning session.",
                "tool_instruction": "",
                "direct_response": "We are done for today. You can open a new session anytime.",
            }
        if not str(state.get("material") or "").strip():
            action = "AnalyzeMaterial"
        elif state.get("last_action") == "QuestionTool":
            action = "AnswerEvaluator"
        elif not state.get("interests"):
            action = "AskInterests"
        elif any(word in lowered for word in ("story", "tale")):
            action = "StoryTool"
        elif any(word in lowered for word in ("explain", "summary", "summarize", "describe")):
            action = "ExplainMaterial"
        else:
            action = "QuestionTool"
        return {
            "action": action,
            "reason": "Demo mode selected a useful next action from the current state.",
            "tool_instruction": "Respond in the student's language and move the learning conversation forward.",
            "direct_response": "",
        }

    if "AskInterests" in system:
        interests = []
        if student_message.strip() and len(student_message.strip()) < 300:
            interests = [student_message.strip()[:120]]
        return {
            "response": "To tailor the learning to you, tell me about two or three things that interest you—for example sports, music, gaming, or technology.",
            "interests": interests,
        }
    if "AnalyzeMaterial" in system:
        interests = ["basketball"] if "basketball" in student_message else []
        return {
            "response": "I received the study material and saved it for the rest of the conversation. In demo mode I identified the main ideas. Now you can ask for an explanation or start practicing.",
            "material": student_message[:30_000],
            "interests": interests,
            "topics": ["Main topic", "Key concepts"],
        }
    if "ExplainMaterial" in system:
        return {
            "response": "Here is a brief explanation based on the material you provided: let's start with the main idea, connect it to the key concepts, and then check understanding with an example.",
        }
    if "StoryTool" in system:
        return {
            "response": "Let's learn through a short story that connects the main idea to your interests. Every detail in the story represents a concept from the material you provided.",
        }
    if "QuestionTool" in system:
        question = "What is the main idea of the material in your own words?"
        return {"response": f"Let's start with a short question: {question}", "questions": [question]}
    if "AnswerEvaluator" in system:
        return {
            "response": "Good answer. You identified the main direction, but you should be more precise with the key concepts. Try to give one example from the material.",
            "score": 75,
            "weak_topics": ["Key concepts"],
            "strong_topics": ["Main idea"],
            "mastery": False,
        }
    return {"response": "I am ready to continue teaching based on the material you provided."}
