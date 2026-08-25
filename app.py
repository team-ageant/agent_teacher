"""FastAPI entrypoint for local development and Vercel deployment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
 
from adaptive_teacher.agent import execute_agent
from adaptive_teacher.api_info import agent_info, team_info
from adaptive_teacher.config import get_settings
from adaptive_teacher.models import MAX_LLM_CALLS
from adaptive_teacher.state import session_store

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive AI Teacher",
    description="Dynamic ReAct teaching agent for plain-text learning material.",
    version="1.0.0",
)


def _error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "error": message,
            "response": None,
            "steps": [],
        },
        status_code=status_code,
    )


def _session_id(candidate: str | None) -> tuple[str, bool]:
    if candidate:
        try:
            return str(UUID(candidate)), True
        except (ValueError, AttributeError):
            pass
    return str(uuid4()), False


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(PUBLIC / "index.html", media_type="text/html")


@app.get("/styles.css", include_in_schema=False)
async def styles() -> FileResponse:
    return FileResponse(PUBLIC / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def browser_app() -> FileResponse:
    return FileResponse(
        PUBLIC / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"}
    )


@app.get("/favicon.png", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(PUBLIC / "favicon.png", media_type="image/png")


@app.get("/model-architecture.png", include_in_schema=False)
async def architecture_asset() -> FileResponse:
    return FileResponse(PUBLIC / "model-architecture.png", media_type="image/png")


@app.get("/api/team_info")
async def get_team_info() -> dict[str, Any]:
    return team_info()


@app.get("/api/agent_info")
async def get_agent_info() -> dict[str, Any]:
    return agent_info()


@app.get("/api/model_architecture")
async def get_model_architecture() -> FileResponse:
    return FileResponse(
        PUBLIC / "model-architecture.png",
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/execute")
async def execute(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return _error_response("The request body must be valid JSON.")
    if not isinstance(body, dict):
        return _error_response("The request body must be a JSON object.")

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _error_response("The request must include a non-empty string field named 'prompt'.")
    if len(prompt) > 30_000:
        return _error_response("The prompt is limited to 30,000 characters.")

    session_id, reused_cookie = _session_id(request.cookies.get("adaptive_session_id"))
    async with session_store.execution(session_id):
        state = session_store.get(session_id)
        try:
            response_text, steps = await execute_agent(state, prompt.strip())
            session_store.save(state)
        except Exception:
            # Persist any reserved call count, but never expose provider details.
            session_store.save(state)
            LOGGER.exception("Agent execution failed for session %s", session_id)
            response = _error_response(
                "The learning agent could not complete this request. Please try again.",
                500,
            )
            response.headers["X-LLM-Calls-Used"] = str(state.llm_calls)
            response.headers["X-LLM-Calls-Remaining"] = str(max(0, MAX_LLM_CALLS - state.llm_calls))
            response.headers["X-Agent-Session-Id"] = session_id
            if not reused_cookie:
                response.set_cookie(
                    "adaptive_session_id",
                    session_id,
                    max_age=60 * 60,
                    httponly=True,
                    samesite="lax",
                    secure=get_settings().production,
                    path="/",
                )
            return response

    response = JSONResponse(
        {
            "status": "ok",
            "error": None,
            "response": response_text,
            "steps": steps,
        }
    )
    response.headers["X-LLM-Calls-Used"] = str(state.llm_calls)
    response.headers["X-LLM-Calls-Remaining"] = str(max(0, MAX_LLM_CALLS - state.llm_calls))
    response.headers["X-Agent-Session-Id"] = session_id
    if not reused_cookie:
        response.set_cookie(
            "adaptive_session_id",
            session_id,
            max_age=60 * 60,
            httponly=True,
            samesite="lax",
            secure=get_settings().production,
            path="/",
        )
    return response


@app.delete("/api/session")
async def reset_session(request: Request) -> JSONResponse:
    session_id, valid_cookie = _session_id(request.cookies.get("adaptive_session_id"))
    if valid_cookie:
        async with session_store.execution(session_id):
            session_store.delete(session_id)
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("adaptive_session_id", path="/")
    return response
