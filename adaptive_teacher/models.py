"""Core types and constants for the adaptive teaching agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MAX_LLM_CALLS = 16
TOOL_NAMES = (
    "AskInterests",
    "AnalyzeMaterial",
    "ExplainMaterial",
    "StoryTool",
    "QuestionTool",
    "AnswerEvaluator",
    "RespondDirectly",
    "Stop",
)
 
ToolName = Literal[
    "AskInterests",
    "AnalyzeMaterial",
    "ExplainMaterial",
    "StoryTool",
    "QuestionTool",
    "AnswerEvaluator",
    "RespondDirectly",
    "Stop",
]


@dataclass(slots=True)
class ChatTurn:
    role: Literal["student", "teacher"]
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LearningState:
    session_id: str
    material: str = ""
    interests: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    strong_topics: list[str] = field(default_factory=list)
    history: list[ChatTurn] = field(default_factory=list)
    llm_calls: int = 0
    latest_score: float | None = None
    mastery: bool = False
    last_action: ToolName | None = None
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    action: ToolName
    reason: str
    tool_instruction: str
    direct_response: str


@dataclass(frozen=True, slots=True)
class TraceStep:
    module: str
    system_prompt: str
    user_prompt: str
    response: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "prompt": {
                "System_prompt": self.system_prompt,
                "User_prompt": self.user_prompt,
            },
            "response": self.response,
        }
