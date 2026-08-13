"""Non-network tests for Pinecone retrieval and prompt grounding."""

from __future__ import annotations

import asyncio
import json

import adaptive_teacher.retrieval as retrieval_module
from adaptive_teacher.config import reset_settings_cache
from adaptive_teacher.models import LearningState
from adaptive_teacher.prompts import supervisor_prompt


def test_retrieval_is_disabled_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_HOST", raising=False)
    reset_settings_cache()

    assert asyncio.run(retrieval_module.retrieve_passages("Alice")) == []


def test_search_maps_pinecone_hits(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_HOST", "https://test-index.example")
    monkeypatch.setenv("PINECONE_NAMESPACE", "__default__")
    reset_settings_cache()

    calls = {}

    class FakeIndex:
        def search(self, **kwargs):
            calls.update(kwargs)
            return {
                "result": {
                    "hits": [
                        {
                            "_id": "3_chunk_4",
                            "_score": 0.91,
                            "fields": {
                                "text": "Down the rabbit-hole",
                                "title": "Alice's Adventures in Wonderland",
                                "authors": ["Lewis Carroll"],
                                "gutenberg_id": 11,
                                "chunk_index": 4,
                            },
                        }
                    ]
                }
            }

    class FakePinecone:
        def __init__(self, api_key):
            assert api_key == "test-key"

        def Index(self, *, host):
            assert host == "https://test-index.example"
            return FakeIndex()

    monkeypatch.setattr(retrieval_module, "Pinecone", FakePinecone)
    passages = asyncio.run(retrieval_module.retrieve_passages("rabbit hole"))

    assert calls["namespace"] == "__default__"
    assert calls["query"]["inputs"]["text"] == "rabbit hole"
    assert passages[0]["title"] == "Alice's Adventures in Wonderland"
    assert passages[0]["text"] == "Down the rabbit-hole"


def test_retrieved_passages_are_in_supervisor_prompt() -> None:
    _, user_prompt = supervisor_prompt(
        LearningState(session_id="test"),
        "Tell me about Alice",
        10,
        [{"title": "Alice", "text": "A relevant passage"}],
    )

    payload = json.loads(user_prompt)
    assert payload["retrieved_gutenberg_passages"][0]["text"] == "A relevant passage"
