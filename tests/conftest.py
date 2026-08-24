"""
tests/conftest.py

Phase 2D.1: 测试隔离配置。

- 强制 AGENT_MODE=rule（不受本地 .env 影响）
- 每个测试后自动重置 FakeLLMClient 和 ToolRegistry
- 所有测试默认离线
"""

import asyncio
import os
import pytest


@pytest.fixture(autouse=True)
def force_rule_mode():
    """Force AGENT_MODE=rule and SEARCH_PROVIDER=mock for offline tests."""
    os.environ["AGENT_MODE"] = "rule"
    os.environ["SEARCH_PROVIDER"] = "mock"
    os.environ["RUN_HISTORY_ENABLED"] = "false"
    os.environ["LLM_ONLY_MODE"] = "false"
    # Keep evaluator tests free to override the class-level harness threshold;
    # production reads the explicit TTL from .env.
    os.environ.pop("RESEARCH_LATENCY_TTL_SECONDS", None)


@pytest.fixture(autouse=True)
def ensure_legacy_event_loop_api():
    """Keep sync tests using get_event_loop portable on Python 3.12+."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield


@pytest.fixture(autouse=True)
def reset_llm_and_registry():
    """每个测试后自动重置全局 LLM client 和 ToolRegistry。"""
    yield
    from app.llm.client import reset_llm_client
    from app.storage import reset_history_repository
    from app.tools.registry import ToolRegistry
    reset_llm_client()
    reset_history_repository()
    ToolRegistry.reset_instance()
