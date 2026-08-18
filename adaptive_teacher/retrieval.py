"""Pinecone retrieval for the pre-indexed Gutenberg corpus."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from pinecone import Pinecone
except ImportError:  # Allows the application to start before optional dependencies are installed.
    Pinecone = None  # type: ignore[assignment,misc]

from .config import get_settings
from .llm import create_embeddings

LOGGER = logging.getLogger(__name__)
MAX_QUERY_CHARS = 4_000
MAX_PASSAGE_CHARS = 3_000


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for method_name in ("to_dict", "model_dump"):
        method = getattr(value, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, dict):
                return converted
    return {}


def _search_sync(query_vector: list[float]) -> list[dict[str, Any]]:
    settings = get_settings()
    if Pinecone is None:
        raise RuntimeError("The pinecone package is not installed.")
    index = Pinecone(api_key=settings.pinecone_api_key).Index(
        host=settings.pinecone_index_host
    )
    response = index.query(
        namespace=settings.pinecone_namespace,
        vector=query_vector,
        top_k=settings.pinecone_top_k,
        include_metadata=True,
    )
    payload = _as_dict(response)
    matches = payload.get("matches", [])
    if not isinstance(matches, list):
        return []

    passages: list[dict[str, Any]] = []
    for raw_match in matches:
        match = _as_dict(raw_match)
        metadata = _as_dict(match.get("metadata"))
        text = metadata.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        passages.append(
            {
                "id": str(match.get("id", match.get("_id", ""))),
                "score": match.get("score", match.get("_score")),
                "text": text.strip()[:MAX_PASSAGE_CHARS],
                "title": metadata.get("title"),
                "authors": metadata.get("authors"),
                "gutenberg_id": metadata.get("gutenberg_id", metadata.get("id")),
                "chunk_index": metadata.get("chunk_index"),
            }
        )
    return passages


async def retrieve_passages(query: str) -> list[dict[str, Any]]:
    """Return relevant book passages, or no passages when retrieval is unavailable."""
    settings = get_settings()
    if not settings.pinecone_api_key or not settings.pinecone_index_host:
        return []
    try:
        query_vectors = await create_embeddings(query[:MAX_QUERY_CHARS])
        if not query_vectors or not query_vectors[0]:
            return []
        return await asyncio.to_thread(_search_sync, query_vectors[0])
    except Exception:
        LOGGER.exception("Pinecone retrieval failed; continuing without external context")
        return []

