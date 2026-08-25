"""Shared fixtures that keep tests deterministic and free of billable API calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import Any
 
import httpx
import pytest

# This is set before importing the application, so .env.local can never cause a
# live LLMod request during automated tests.
os.environ["ADAPTIVE_TEACHER_DEMO_MODE"] = "true"

from adaptive_teacher.config import reset_settings_cache  # noqa: E402
from adaptive_teacher.state import session_store  # noqa: E402
from app import app  # noqa: E402


class LocalClient:
    """Small synchronous wrapper around HTTPX's in-process ASGI transport."""

    def __init__(self) -> None:
        self._runner = asyncio.Runner()
        self._client = self._runner.run(self._open())

    async def _open(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        )

    @property
    def cookies(self) -> httpx.Cookies:
        return self._client.cookies

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return self._runner.run(self._client.request(method, url, **kwargs))

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        self._runner.run(self._client.aclose())
        self._runner.close()


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Iterator[None]:
    reset_settings_cache()
    session_store.clear()
    yield
    reset_settings_cache()
    session_store.clear()


@pytest.fixture
def client() -> Iterator[LocalClient]:
    test_client = LocalClient()
    try:
        yield test_client
    finally:
        test_client.close()
