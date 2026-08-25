"""Temporary, process-local learning session storage."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import RLock

from .models import LearningState

SESSION_TTL_SECONDS = 60 * 60
MAX_IN_MEMORY_SESSIONS = 1_000

 
@dataclass(slots=True)
class _SessionExecution:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class SessionStore:
    """Thread-safe best-effort memory with one-hour expiry.

    Vercel instances can be recycled or scaled independently, so this store is
    intentionally temporary. Durable state will move to the project database.
    """

    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, LearningState] = {}
        self._executions: dict[str, _SessionExecution] = {}
        self._lock = RLock()

    def _remove_expired(self, now: float) -> None:
        expired: list[str] = []
        for session_id, state in self._sessions.items():
            execution = self._executions.get(session_id)
            if now - state.updated_at > self._ttl_seconds and (
                execution is None or execution.users == 0
            ):
                expired.append(session_id)
        for session_id in expired:
            self._sessions.pop(session_id, None)
            execution = self._executions.get(session_id)
            if execution is not None and execution.users == 0:
                self._executions.pop(session_id, None)

    def _evict_oldest_if_full(self) -> None:
        if len(self._sessions) < MAX_IN_MEMORY_SESSIONS:
            return
        candidates = sorted(self._sessions.values(), key=lambda state: state.updated_at)
        for state in candidates:
            execution = self._executions.get(state.session_id)
            if execution is None or execution.users == 0:
                self._sessions.pop(state.session_id, None)
                self._executions.pop(state.session_id, None)
                return

    def get(self, session_id: str) -> LearningState:
        now = time.time()
        with self._lock:
            self._remove_expired(now)
            state = self._sessions.get(session_id)
            if state is None:
                self._evict_oldest_if_full()
                state = LearningState(session_id=session_id, updated_at=now)
                self._sessions[session_id] = state
            return state

    @asynccontextmanager
    async def execution(self, session_id: str) -> AsyncIterator[None]:
        """Serialize a session and safely account for holders and waiters."""

        with self._lock:
            execution = self._executions.get(session_id)
            if execution is None:
                execution = _SessionExecution()
                self._executions[session_id] = execution
            execution.users += 1
        try:
            async with execution.lock:
                yield
        finally:
            with self._lock:
                execution.users -= 1
                current = self._executions.get(session_id)
                if (
                    current is execution
                    and execution.users == 0
                    and session_id not in self._sessions
                ):
                    self._executions.pop(session_id, None)

    def save(self, state: LearningState) -> None:
        with self._lock:
            state.updated_at = time.time()
            self._sessions[state.session_id] = state

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._executions.clear()


session_store = SessionStore()
