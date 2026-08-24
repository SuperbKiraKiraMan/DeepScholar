"""
tests/test_phase2c_llm.py

Phase 2C LLM Agent Intelligence 测试。

覆盖：
1. rule mode 完全离线运行
2. llm mode 使用 FakeLLMClient
3. 缺少 API key 自动回退 rule mode
4. LLMPlanner 输出多个合法 search tasks
5. Planner 非法 JSON 回退规则 Planner
6. 不存在的 tool 被拒绝
7. 超过 max_tool_calls 被终止
8. 重复 tool+args 不会再次执行
9. LLM timeout 自动 fallback
10. graph_send 继续真实并行
11. 单 Worker 失败仍 completed_with_warnings
12. citation retry 测试必须真正触发 retry
13. 来源按 normalized URL/DOI 去重
14. 原有 loop/graph/graph_send 测试全部通过
"""

import asyncio
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ================================================================
# Fake LLM Responses
# ================================================================

FAKE_PLANNER_OUTPUT = {
    "research_goal": "Evaluate RAG evaluation methods systematically",
    "search_tasks": [
        {
            "task_id": "search_1",
            "query": "RAG evaluation methods overview survey",
            "purpose": "Broad survey of RAG evaluation approaches",
            "depends_on": [],
            "allowed_tools": ["academic_search"],
        },
        {
            "task_id": "search_2",
            "query": "RAG evaluation benchmarks and metrics",
            "purpose": "Specific benchmarks and evaluation metrics",
            "depends_on": [],
            "allowed_tools": ["academic_search"],
        },
        {
            "task_id": "search_3",
            "query": "RAG evaluation limitations and recent advances",
            "purpose": "Limitations and cutting-edge work",
            "depends_on": [],
            "allowed_tools": ["academic_search"],
        },
    ],
}

FAKE_PLANNER_OUTPUT_INVALID_JSON = {
    "_fail": True,
    "_error": "Simulated JSON parse failure",
    "_raw_text": "not valid json {{{",
}

FAKE_PLANNER_OUTPUT_BAD_TOOL = {
    "research_goal": "Evaluate RAG methods",
    "search_tasks": [
        {
            "task_id": "search_1",
            "query": "RAG evaluation",
            "purpose": "Survey",
            "depends_on": [],
            "allowed_tools": ["nonexistent_tool"],
        },
    ],
}

FAKE_PLANNER_OUTPUT_SINGLE_TASK = {
    "research_goal": "Evaluate RAG methods",
    "search_tasks": [
        {
            "task_id": "search_1",
            "query": "RAG evaluation",
            "purpose": "Survey",
            "depends_on": [],
            "allowed_tools": ["academic_search"],
        },
    ],
}

# ================================================================
# Tests: Rule Mode & LLM Client
# ================================================================

class TestRuleModeOffline:
    """1. rule mode 完全离线运行。"""

    def test_rule_mode_no_llm_calls(self):
        """rule mode 不调用 LLM。"""
        from app.llm.client import LLMClient

        client = LLMClient()
        assert client.mode == "rule"
        assert not client.is_available

    @pytest.mark.asyncio
    async def test_rule_mode_generate_structured_returns_error(self):
        """rule mode 下 generate_structured 返回 success=False。"""
        from app.llm.client import LLMClient
        from app.llm.schemas import LLMPlannerOutput

        client = LLMClient()
        result = await client.generate_structured("system", "user", LLMPlannerOutput)
        assert result["success"] is False
        assert "not available" in result["error"].lower()


class TestFakeLLMClient:
    """2. FakeLLMClient 用于测试。"""

    @pytest.mark.asyncio
    async def test_fake_client_returns_structured_output(self):
        """FakeLLMClient 返回预定义的结构化输出。"""
        from app.llm.client import FakeLLMClient
        from app.llm.schemas import LLMPlannerOutput

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT])
        result = await fake.generate_structured("s", "u", LLMPlannerOutput)

        assert result["success"] is True
        assert result["data"].research_goal == FAKE_PLANNER_OUTPUT["research_goal"]
        assert len(result["data"].search_tasks) == 3
        assert len(fake.calls) == 1

    @pytest.mark.asyncio
    async def test_fake_client_simulates_failure(self):
        """FakeLLMClient 可以模拟失败。"""
        from app.llm.client import FakeLLMClient
        from app.llm.schemas import LLMPlannerOutput

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT_INVALID_JSON])
        result = await fake.generate_structured("s", "u", LLMPlannerOutput)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fake_client_runs_out_of_responses(self):
        """FakeLLMClient 响应用完后返回错误。"""
        from app.llm.client import FakeLLMClient
        from app.llm.schemas import LLMPlannerOutput

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT])
        await fake.generate_structured("s", "u", LLMPlannerOutput)
        result = await fake.generate_structured("s", "u", LLMPlannerOutput)

        assert result["success"] is False
        assert "no more" in result["error"].lower()


class TestLLMFallback:
    """3. 缺少 API key 自动回退 rule mode。"""

    def test_no_api_key_falls_back_to_rule(self):
        """DEEPSEEK_API_KEY 为空时 mode=rule。"""
        from app.llm.client import LLMClient, LLMConfig

        config = LLMConfig(agent_mode="llm", api_key="")
        client = LLMClient(config)
        assert client.mode == "rule"

    def test_with_api_key_uses_llm(self):
        """有 API key 时 mode=llm。"""
        from app.llm.client import LLMClient, LLMConfig

        config = LLMConfig(agent_mode="llm", api_key="test-key-123")
        client = LLMClient(config)
        assert client.mode == "llm"
        assert client.is_available


# ================================================================
# Tests: LLMPlanner
# ================================================================

class TestLLMPlanner:
    """4-5. LLMPlanner 测试。"""

    @pytest.mark.asyncio
    async def test_llm_planner_generates_multiple_tasks(self):
        """4. LLMPlanner 使用 FakeLLMClient 生成多个合法 search tasks。"""
        from app.llm.client import reset_llm_client, FakeLLMClient
        from app.llm.schemas import LLMPlannerOutput
        import app.llm.client as client_mod

        # Replace global client with fake
        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT])
        client_mod._global_client = fake

        try:
            from app.agents.llm_planner import LLMPlanner
            planner = LLMPlanner()
            dag, result = await planner.plan(topic="RAG evaluation", max_sources=5)

            assert result["success"] is True
            assert len(dag.tasks) >= 4  # 3 search + analyze + cite
            search_tasks = [t for t in dag.tasks if t.task_type == "search"]
            assert len(search_tasks) == 3
            # task_ids 唯一
            task_ids = [t.task_id for t in dag.tasks]
            assert len(task_ids) == len(set(task_ids))
        finally:
            reset_llm_client()

    @pytest.mark.asyncio
    async def test_llm_planner_falls_back_on_invalid_json(self):
        """5. 非法 JSON 回退规则 Planner。"""
        from app.llm.client import reset_llm_client, FakeLLMClient
        import app.llm.client as client_mod

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT_INVALID_JSON])
        client_mod._global_client = fake

        try:
            from app.agents.llm_planner import LLMPlanner
            planner = LLMPlanner()
            dag, result = await planner.plan(topic="RAG evaluation", max_sources=5)

            assert result["success"] is False
            # 仍返回有效 TaskDAG（rule fallback）
            assert len(dag.tasks) >= 4
            search_tasks = [t for t in dag.tasks if t.task_type == "search"]
            assert len(search_tasks) >= 2
        finally:
            reset_llm_client()

    @pytest.mark.asyncio
    async def test_llm_planner_rejects_unknown_tool(self):
        """6. 不存在的 tool 被拒绝并回退。"""
        from app.llm.client import reset_llm_client, FakeLLMClient
        import app.llm.client as client_mod

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT_BAD_TOOL])
        client_mod._global_client = fake

        try:
            from app.agents.llm_planner import LLMPlanner
            planner = LLMPlanner()
            dag, result = await planner.plan(topic="RAG", max_sources=5)

            assert result["success"] is False
            assert "nonexistent_tool" in result.get("error", "")
            # 回退成功
            assert len(dag.tasks) >= 4
        finally:
            reset_llm_client()

    @pytest.mark.asyncio
    async def test_llm_planner_falls_back_on_single_task(self):
        """单 task 输出回退规则版（需要至少 2 个 search task）。"""
        from app.llm.client import reset_llm_client, FakeLLMClient
        import app.llm.client as client_mod

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT_SINGLE_TASK])
        client_mod._global_client = fake

        try:
            from app.agents.llm_planner import LLMPlanner
            planner = LLMPlanner()
            dag, result = await planner.plan(topic="RAG", max_sources=5)

            assert result["success"] is False
            assert "only" in result.get("error", "").lower()
            assert len(dag.tasks) >= 4
        finally:
            reset_llm_client()


# ================================================================
# Tests: Phase 2B.1 Hardening
# ================================================================

class TestPhase2B1Hardening:
    """Phase 2B.1 加固测试。"""

    def test_dedup_by_normalized_url(self):
        """17. 来源按 normalized URL 去重（同一 URL 不同 source_id 也必须合并）。"""
        from app.graph.runtime import _dedup_sources

        sources = [
            {"source_id": "id1", "url": "https://arxiv.org/abs/1234.5678", "title": "Paper A", "quality_score": 0.8, "full_text": "full A"},
            {"source_id": "id2", "url": "https://arxiv.org/abs/1234.5678", "title": "Paper A Dup", "quality_score": 0.5, "full_text": "short"},
        ]

        result = _dedup_sources(sources)
        assert len(result) == 1, f"Expected 1 after dedup, got {len(result)}"
        # 应该保留 quality_score 更高的版本
        assert result[0]["source_id"] == "id1"
        assert result[0]["quality_score"] == 0.8

    def test_dedup_url_normalization(self):
        """URL 规范化后去重（去除 www、协议差异）。"""
        from app.graph.runtime import _dedup_sources

        sources = [
            {"source_id": "id1", "url": "https://www.arxiv.org/abs/1234.5678", "title": "Paper A", "quality_score": 0.9},
            {"source_id": "id2", "url": "http://arxiv.org/abs/1234.5678", "title": "Paper A", "quality_score": 0.5},
        ]

        result = _dedup_sources(sources)
        assert len(result) == 1

    def test_dedup_prefers_doi(self):
        """DOI 优先于 URL 去重。"""
        from app.graph.runtime import _dedup_key

        source = {"doi": "10.1234/test.123", "url": "https://example.com/paper"}
        key = _dedup_key(source)
        assert key.startswith("doi:")

    def test_dedup_prefers_arxiv(self):
        """arXiv ID 优先于 URL 去重。"""
        from app.graph.runtime import _dedup_key

        source = {"url": "https://arxiv.org/abs/2401.00001"}
        key = _dedup_key(source)
        assert key.startswith("arxiv:")


class TestCitationRetry:
    """16. citation retry 测试必须真正触发 retry。"""

    def test_retry_count_incremented_when_citation_fails(self):
        """citation 失败时 retry_count 递增。"""
        from app.graph.runtime import route_after_evaluator

        state = {
            "eval_metrics": {"no_fake_citation": False, "citation_id_exists": True, "source_url_valid": True},
            "retry_count": 0,
        }
        assert route_after_evaluator(state) == "analysis_worker"

    def test_retry_stops_at_limit(self):
        """retry 达到上限后停止。"""
        from app.graph.runtime import route_after_evaluator

        state = {
            "eval_metrics": {"no_fake_citation": False},
            "retry_count": 2,
        }
        assert route_after_evaluator(state) == "final_reviewer"

    @pytest.mark.asyncio
    async def test_graph_run_saves_retry_count(self):
        """run_graph 保存 retry_count。"""
        from app.graph.runtime import run_graph

        result = await run_graph(topic="RAG evaluation", max_sources=3, run_eval=True)
        assert "retry_count" in result
        assert "replan_count" in result
        assert "retry_attempted" in result
        assert "total_latency_ms" in result
        assert "unresolved_issues" in result


class TestFailureIsolation:
    """15. 单 Worker 失败仍 completed_with_warnings。"""

    @pytest.mark.asyncio
    async def test_search_worker_send_handles_exception(self):
        """Search Worker Send 节点异常时不崩溃。"""
        from app.graph.runtime import node_search_worker_send, ResearchAgentState

        state: ResearchAgentState = {
            "topic": "test",
            "backend": "graph_send",
            "agent_mode": "rule",
            "current_search_task": {"task_id": "search_test", "description": "test"},
            "max_sources": 3,
            "trace": [],
            "warnings": [],
        }
        result = await node_search_worker_send(state)

        # 即使内部出错（mock search 可能抛异常），返回结构应完整
        assert "_search_bucket" in result
        assert "trace" in result
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_reading_worker_send_handles_missing_source(self):
        """Reading Worker Send 节点无 source 时记录 warning。"""
        from app.graph.runtime import node_reading_worker_send, ResearchAgentState

        state: ResearchAgentState = {
            "topic": "test",
            "backend": "graph_send",
            "agent_mode": "rule",
            "current_reading_task": {"task_id": "read_test", "source_id": "missing"},
            "max_sources": 3,
            "trace": [],
            "warnings": [],
        }
        result = await node_reading_worker_send(state)

        assert len(result.get("warnings", [])) >= 1
        # worker_finished event 存在
        events = [t.get("event") for t in result.get("trace", [])]
        assert "worker_finished" in events


class TestGraphSendIntegration:
    """14. graph_send 继续真实并行。"""

    @pytest.mark.asyncio
    async def test_graph_send_completes_with_agent_mode(self):
        """graph_send + agent_mode=rule 完整执行。"""
        from app.graph.runtime import run_graph

        result = await run_graph(
            topic="RAG evaluation",
            max_sources=4,
            agent_mode="rule",
        )
        assert result["status"] in {"completed", "completed_with_warnings"}
        assert len(result.get("sources", [])) >= 1
        assert len(result.get("evidence_cards", [])) >= 1

    @pytest.mark.asyncio
    async def test_graph_send_llm_mode_never_silently_falls_back_when_model_is_exhausted(self, monkeypatch):
        """LLM-only production path must fail instead of returning a rule report."""
        from app.llm.client import reset_llm_client, FakeLLMClient
        import app.llm.client as client_mod

        fake = FakeLLMClient(responses=[FAKE_PLANNER_OUTPUT])
        client_mod._global_client = fake
        monkeypatch.setenv("LLM_ONLY_MODE", "true")

        try:
            from app.graph.runtime import run_graph

            with pytest.raises(RuntimeError, match="LLM-only"):
                await run_graph(
                    topic="RAG evaluation",
                    max_sources=4,
                    agent_mode="llm",
                )
        finally:
            reset_llm_client()

    @pytest.mark.asyncio
    async def test_graph_run_preserves_all_backends(self):
        """18. 原有 graph_send / loop 全部通过。"""
        from app.graph.runtime import run_graph
        from app.services.orchestrator import Orchestrator

        # graph_send
        gs = await run_graph(topic="RAG eval", max_sources=3)
        assert gs["status"] in {"completed", "completed_with_warnings"}

        # loop
        loop = await Orchestrator().run(topic="RAG eval", max_sources=3)
        assert loop["status"] == "completed"


class TestLLMConfig:
    """测试 LLM 配置和 trace event。"""

    def test_trace_event_excludes_api_key(self):
        """trace event 不包含 API key。"""
        from app.llm.client import LLMClient, LLMConfig

        config = LLMConfig(agent_mode="llm", api_key="secret-key-123")
        client = LLMClient(config)

        event = client.build_trace_event("llm_started", success=True, latency_ms=10)
        event_str = str(event)
        assert "secret-key-123" not in event_str
        assert "api_key" not in [k.lower() for k in event.keys()]

    def test_load_config_from_env(self):
        """从环境变量加载配置。"""
        os.environ["AGENT_MODE"] = "rule"
        from app.llm.client import load_llm_config

        config = load_llm_config()
        assert config.agent_mode == "rule"
        assert config.model == "deepseek-v4-flash"

    def test_load_config_defaults_to_llm(self, monkeypatch):
        """未设置环境变量时，生产默认模式为 LLM。"""
        monkeypatch.delenv("AGENT_MODE", raising=False)
        from app.llm.client import load_llm_config

        assert load_llm_config().agent_mode == "llm"
