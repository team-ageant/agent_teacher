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


def _search_sync(query: str) -> list[dict[str, Any]]:
    settings = get_settings()
    if Pinecone is None:
        raise RuntimeError("The pinecone package is not installed.")
    index = Pinecone(api_key=settings.pinecone_api_key).Index(
        host=settings.pinecone_index_host
    )
    response = index.search(
        namespace=settings.pinecone_namespace,
        query={"inputs": {"text": query[:MAX_QUERY_CHARS]}, "top_k": settings.pinecone_top_k},
        fields=["text", "title", "authors", "gutenberg_id", "chunk_index"],
    )
    payload = _as_dict(response)
    result = _as_dict(payload.get("result"))
    hits = result.get("hits", [])
    if not isinstance(hits, list):
        return []

    passages: list[dict[str, Any]] = []
    for raw_hit in hits:
        hit = _as_dict(raw_hit)
        fields = _as_dict(hit.get("fields"))
        text = fields.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        passages.append(
            {
                "id": str(hit.get("_id", "")),
                "score": hit.get("_score"),
                "text": text.strip()[:MAX_PASSAGE_CHARS],
                "title": fields.get("title"),
                "authors": fields.get("authors"),
                "gutenberg_id": fields.get("gutenberg_id"),
                "chunk_index": fields.get("chunk_index"),
            }
        )
    return passages


async def retrieve_passages(query: str) -> list[dict[str, Any]]:
    """Return relevant book passages, or no passages when retrieval is unavailable."""
    settings = get_settings()
    if not settings.pinecone_api_key or not settings.pinecone_index_host:
        return []
    try:
        return await asyncio.to_thread(_search_sync, query)
    except Exception:
        LOGGER.exception("Pinecone retrieval failed; continuing without external context")
        return []
