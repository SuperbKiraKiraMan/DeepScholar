"""
app/llm/ — LLM Client abstraction for Academic Research Copilot.

DeepSeek-compatible integration with deterministic fallback.
"""
from app.llm.client import LLMClient, FakeLLMClient, get_llm_client
from app.llm.schemas import (
    LLMPlannerOutput,
    LLMSearchTask,
    LLMToolSelection,
    LLMFinding,
)

__all__ = [
    "LLMClient",
    "FakeLLMClient",
    "get_llm_client",
    "LLMPlannerOutput",
    "LLMSearchTask",
    "LLMToolSelection",
    "LLMFinding",
]
