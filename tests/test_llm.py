"""Non-network tests for structured parsing and embedding behavior."""

from __future__ import annotations

import asyncio
import json
import math 

import httpx
import pytest

import adaptive_teacher.llm as llm_module
from adaptive_teacher.config import reset_settings_cache
from adaptive_teacher.llm import _extract_json, call_llm, create_embeddings


def test_extract_json_accepts_plain_fenced_and_surrounded_objects() -> None:
    assert _extract_json('{"ok":true}') == {"ok": True}
    assert _extract_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert _extract_json('Result: {"value": 3}') == {"value": 3}


def test_demo_embedding_matches_production_dimension_and_is_normalized() -> None:
    vectors = asyncio.run(create_embeddings(["photosynthesis", "basketball"]))
    assert len(vectors) == 2
    assert all(len(vector) == 1_024 for vector in vectors)
    assert all(math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0) for vector in vectors)


def test_chat_client_sends_llmod_contract_and_parses_json(monkeypatch) -> None:
    monkeypatch.setenv("ADAPTIVE_TEACHER_DEMO_MODE", "false")
    monkeypatch.setenv("VERCEL_ENV", "development")
    monkeypatch.setenv("LLMOD_API_KEY", "test-key")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai")
    monkeypatch.setenv("LLMOD_MODEL", "test-model")
    reset_settings_cache()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.llmod.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '```json\n{"response":"ok"}\n```'}}]},
        )

    real_async_client = httpx.AsyncClient

    def mock_async_client(*_args, **_kwargs):
        return real_async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", mock_async_client)
    result = asyncio.run(call_llm([{"role": "user", "content": "hello"}]))
    assert result == {"response": "ok"}


def test_production_fails_closed_when_llmod_key_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("ADAPTIVE_TEACHER_DEMO_MODE", "false")
    monkeypatch.setenv("VERCEL_ENV", "production")
    monkeypatch.delenv("LLMOD_API_KEY", raising=False)
    reset_settings_cache()

    with pytest.raises(RuntimeError, match="LLMOD_API_KEY is required"):
        asyncio.run(call_llm([{"role": "user", "content": "hello"}]))
