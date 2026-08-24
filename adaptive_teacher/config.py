"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load standard .env file followed by .env.local for local overrides.
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env.local", override=True)


def _normalized_base_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("LLMOD_BASE_URL must be an absolute HTTP(S) URL.")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Values used by the server and never exposed to browser JavaScript."""

    llmod_api_key: str
    llmod_base_url: str
    llmod_model: str
    llmod_embedding_model: str
    llmod_embedding_dimensions: int
    llmod_chat_completions_url: str | None
    llmod_embeddings_url: str | None
    llmod_api_key_header: str | None
    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str
    pinecone_top_k: int
    group_batch_order_number: str
    team_name: str
    batel_email: str
    itay_email: str
    boaz_email: str
    demo_mode: bool
    production: bool

    @property
    def chat_completions_url(self) -> str:
        return self.llmod_chat_completions_url or f"{self.llmod_base_url}/chat/completions"

    @property
    def embeddings_url(self) -> str:
        return self.llmod_embeddings_url or f"{self.llmod_base_url}/embeddings"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    environment = os.getenv("VERCEL_ENV") or os.getenv("APP_ENV") or "development"
    raw_dim = os.getenv("LLMOD_EMBEDDING_DIMENSIONS", "1024").strip()
    embedding_dim = int(raw_dim) if raw_dim.isdigit() else 1024
    return Settings(
        llmod_api_key=os.getenv("LLMOD_API_KEY", "").strip(),
        llmod_base_url=_normalized_base_url(os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai")),
        llmod_model=os.getenv("LLMOD_MODEL", "MB5R2CF-azure/gpt-5.4-mini").strip(),
        llmod_embedding_model=os.getenv(
            "LLMOD_EMBEDDING_MODEL",
            "MB5R2CF-azure/text-embedding-3-small",
        ).strip(),
        llmod_embedding_dimensions=embedding_dim,
        llmod_chat_completions_url=os.getenv("LLMOD_CHAT_COMPLETIONS_URL") or None,
        llmod_embeddings_url=os.getenv("LLMOD_EMBEDDINGS_URL") or None,
        llmod_api_key_header=os.getenv("LLMOD_API_KEY_HEADER") or None,
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index_host=os.getenv("PINECONE_INDEX_HOST", "").strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "__default__").strip()
        or "__default__",
        pinecone_top_k=max(1, min(20, int(os.getenv("PINECONE_TOP_K", "5")))),
        group_batch_order_number=os.getenv("GROUP_BATCH_ORDER_NUMBER", "TBD_TBD").strip(),
        team_name=os.getenv("TEAM_NAME", "Adaptive AI Teacher").strip(),
        batel_email=os.getenv("BATEL_EMAIL", "").strip(),
        itay_email=os.getenv("ITAY_EMAIL", "").strip(),
        boaz_email=os.getenv("BOAZ_EMAIL", "").strip(),
        demo_mode=_is_true(os.getenv("ADAPTIVE_TEACHER_DEMO_MODE")),
        production=environment.lower() == "production",
    )


def reset_settings_cache() -> None:
    """Clear cached configuration, primarily for isolated automated tests."""

    get_settings.cache_clear()
