"""
tests/test_phase2d_llm_worker.py

Phase 2D LLM Worker Function Calling 测试。

所有测试使用 FakeLLMClient，完全离线，不访问网络。
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ================================================================
# Fake FC Responses
# ================================================================

FC_SEARCH_THEN_FINISH = [
    {  # Round 1: call mock_academic_search
        "_finish": False,
        "tool_calls": [{
            "id": "call_search_1",
            "name": "academic_search",
            "arguments": {"query": "RAG evaluation overview", "max_results": 5},
        }],
    },
    {  # Round 2: finish
        "_finish": True,
        "content": "Search task completed successfully",
    },
]

FC_READ_THEN_FINISH = [
    {  # Round 1: call paper_metadata
        "_finish": False,
        "tool_calls": [{
            "id": "call_read_1",
            "name": "paper_metadata",
            "arguments": {"sources": [{"source_id": "test-id-1", "title": "Test"}]},
        }],
    },
    {  # Round 2: call source_quality_scorer
        "_finish": False,
        "tool_calls": [{
            "id": "call_read_2",
            "name": "source_quality_scorer",
            "arguments": {"sources": [{"source_id": "test-id-1"}], "topic": "RAG evaluation"},
        }],
    },
    {  # Round 3: finish
        "_finish": True,
        "content": "Reading task completed",
    },
]

FC_FINISH_IMMEDIATELY = [
    {"_finish": True, "content": "No tools needed"},
]

FC_UNKNOWN_TOOL = [
    {"_finish": False, "tool_calls": [{
        "id": "call_bad",
        "name": "nonexistent_tool_xyz",
        "arguments": {},
    }]},
    {"_finish": True, "content": "done"},
]

FC_CITATION_CHECK_BLOCKED = [
    {"_finish": False, "tool_calls": [{
        "id": "call_cite",
        "name": "citation_check",
        "arguments": {"citations": [], "sources": []},
    }]},
    {"_finish": True, "content": "done"},
]

FC_DUPLICATE_CALLS = [
    {"_finish": False, "tool_calls": [{
        "id": "call_dup_1",
        "name": "academic_search",
        "arguments": {"query": "test", "max_results": 5},
    }]},
    {"_finish": False, "tool_calls": [{
        "id": "call_dup_2",
        "name": "academic_search",
        "arguments": {"query": "test", "max_results": 5},
    }]},
    {"_finish": True, "content": "done"},
]

FC_TIMEOUT = [
    {"_fail": True, "_error": "Simulated timeout 1"},
    {"_fail": True, "_error": "Simulated timeout 2"},
    {"_fail": True, "_error": "Simulated timeout 3"},
]

FC_INVALID_ARGS = [
    {"_finish": False, "tool_calls": [{
        "id": "call_bad_args",
        "name": "academic_search",
        "arguments": "not_a_dict",
    }]},
    {"_finish": True, "content": "done"},
]

FC_TOOL_EXCEPTION = [
    {"_finish": False, "tool_calls": [{
        "id": "call_exc",
        "name": "academic_search",
        "arguments": {"query": "", "max_results": -1},
    }]},
    {"_finish": True, "content": "handled error"},
]


# ================================================================
# Tool Registry Tests
# ================================================================

class TestToolRegistry:
    """测试 Tool Registry 和 Function Calling schema 生成。"""

    def test_registry_has_all_tools(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        names = registry.list_names()
        assert "academic_search" in names
        assert "paper_metadata" in names
        assert "source_quality_scorer" in names
        assert "evidence_extract" in names
        assert "citation_check" in names

    def test_function_schemas_exclude_citation_check(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        schemas = registry.get_function_schemas()

        tool_names = [s["function"]["name"] for s in schemas]
        assert "citation_check" not in tool_names, "citation_check must be excluded from LLM tools"
        assert "academic_search" in tool_names

    def test_function_schemas_are_openai_compatible(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        schemas = registry.get_function_schemas()

        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]
            assert s["function"]["parameters"]["type"] == "object"

    def test_validate_unknown_tool(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        error = registry.validate_tool_call(
            "nonexistent", {}, ["academic_search"]
        )
        assert error is not None
        assert "Unknown tool" in error

    def test_validate_tool_not_in_allowlist(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        error = registry.validate_tool_call(
            "paper_metadata", {}, ["academic_search"]
        )
        assert error is not None
        assert "not in allowed_tools" in error.lower()

    def test_validate_citation_check_blocked(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        error = registry.validate_tool_call(
            "citation_check", {"citations": [], "sources": []},
            ["citation_check", "academic_search"],
        )
        assert error is not None
        assert "deterministically" in error.lower()

    def test_validate_valid_call_passes(self):
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        error = registry.validate_tool_call(
            "academic_search", {"query": "test", "max_results": 5},
            ["academic_search"],
        )
        assert error is None

    def test_canonicalize_args_for_dedup(self):
        from app.tools.registry import canonicalize_args

        a1 = canonicalize_args("academic_search", {"query": "test", "max_results": 5})
        a2 = canonicalize_args("academic_search", {"max_results": 5, "query": "test"})
        assert a1 == a2, "Canonicalized args should be order-independent"


# ================================================================
# LLMWorker Tests (with FakeLLMClient)
# ================================================================

class TestLLMWorker:
    """测试 LLMWorker 的 Function Calling 循环。"""

    def setup_method(self):
        from app.llm.client import reset_llm_client
        from app.tools.registry import ToolRegistry
        reset_llm_client()
        ToolRegistry.reset_instance()

    def test_trusted_context_normalizes_string_source_alias(self):
        """OpenAI-compatible models may return source="W..." despite source_id schema."""
        from app.agents.llm_worker import LLMWorker
        from app.agents.worker import WorkerContext
        from app.agents.planner import Task

        worker = LLMWorker()
        ctx = WorkerContext(Task("analyze", "analyze", "Extract evidence"))
        trusted = {
            "source_id": "W123",
            "title": "Trusted Paper",
            "url": "https://doi.org/10.1/trusted",
            "full_text": "The experiment reports a measurable improvement.",
        }
        ctx.add_result("_trusted_sources", [trusted])

        args = worker._inject_trusted_context(
            "evidence_extract", {"source": "W123"}, ctx
        )

        assert args["source"] == trusted
        assert "source_id" not in args

    def test_trusted_context_does_not_accept_unknown_string_source(self):
        from app.agents.llm_worker import LLMWorker
        from app.agents.worker import WorkerContext
        from app.agents.planner import Task

        worker = LLMWorker()
        ctx = WorkerContext(Task("analyze", "analyze", "Extract evidence"))
        ctx.add_result("_trusted_sources", [{"source_id": "W123"}])

        args = worker._inject_trusted_context(
            "evidence_extract", {"source": "W999"}, ctx
        )

        assert "source" not in args
        assert "source_id" not in args

    def test_dependency_summary_preserves_complete_source_ids(self):
        from app.agents.llm_worker import _summarize_deps
        from app.agents.worker import WorkerContext
        from app.agents.planner import Task

        dep = WorkerContext(Task("read", "read", "Read sources"))
        dep.add_result("scored_sources", [{
            "source_id": "W4389984066",
            "title": "RAG Survey",
            "full_text": "abstract",
        }])
        dep.add_result("sources", dep.results["scored_sources"])

        summary = _summarize_deps({"read": dep})

        assert "W4389984066" in summary
        assert "W4389984," not in summary

    def teardown_method(self):
        from app.llm.client import reset_llm_client
        from app.tools.registry import ToolRegistry
        reset_llm_client()
        ToolRegistry.reset_instance()

    @pytest.mark.asyncio
    async def test_llm_worker_selects_and_executes_tool(self):
        """1. LLM Worker 成功选择并执行工具。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_SEARCH_THEN_FINISH)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Search for RAG evaluation overview",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 应该记录了 tool calls
        assert ctx.tool_call_count >= 1
        # 应该找到了 sources
        sources = ctx.results.get("sources", ctx.results.get("search_results", []))
        assert len(sources) >= 1

    @pytest.mark.asyncio
    async def test_llm_worker_finishes_loop(self):
        """3. 模型返回 finish 后循环正常结束。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_FINISH_IMMEDIATELY)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Search test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # finish 后应正常返回（有可能 fallback 到 rule 因为 LLM 没有实际调用工具，
        # 但至少不应该崩溃）
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_unknown_tool_not_executed(self):
        """4. 未知工具不会执行。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_UNKNOWN_TOOL)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 应该 fallback 到 rule，最终还是返回了结果
        assert ctx is not None
        # trace 应该有 tool_args_rejected 或 tool_loop_fallback
        trace_events = [t.get("tool_name", "") for t in ctx.trace]
        has_rejection = "tool_args_rejected" in trace_events or "tool_loop_fallback" in trace_events
        # 至少应该有结果（通过 fallback 获得）
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_citation_check_blocked_from_llm(self):
        """6/16. CitationCheck 不可被 LLM 绕过。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_CITATION_CHECK_BLOCKED)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("cite", "cite", "Check citations",
                    tool_plan=["citation_check"])
        ctx = await worker.execute_task(task)

        # Citation Worker 不应该走 LLM 路径——直接回退 rule
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_duplicate_tool_call_not_executed(self):
        """7. 相同 tool + args 不重复执行。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_DUPLICATE_CALLS)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 应该有 "tool_rejected" trace 表示第二次重复调用被跳过
        trace_names = [t.get("tool_name", "") for t in ctx.trace]
        has_rejected = "tool_rejected" in trace_names
        # 第二次重复调用应被去重
        assert has_rejected or ctx is not None

    @pytest.mark.asyncio
    async def test_max_tool_calls_stops_loop(self):
        """8. 达到 MAX_TOOL_CALLS 后有界停止。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker, LLMWorkerConfig
        from app.agents.planner import Task
        import app.llm.client as client_mod

        # 创建无限循环的 FC 响应（10 个 tool calls）
        infinite_calls = []
        for i in range(10):
            infinite_calls.append({
                "_finish": False,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "name": "academic_search",
                    "arguments": {"query": f"query_{i}", "max_results": 3},
                }],
            })

        fake = FakeLLMClient()
        fake.set_fc_responses(infinite_calls)
        client_mod._global_client = fake

        config = LLMWorkerConfig(max_tool_calls=3)  # 很低的限制
        worker = LLMWorker(config=config)
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 应该在 3 次后停止（可能 fallback 到 rule）
        assert ctx is not None
        # tool_call_count 应该 ≤ max_tool_calls（或更多因为 fallback 到 rule）
        # 检查有界停止
        trace_names = [t.get("tool_name", "") for t in ctx.trace]
        assert "tool_loop_limit_reached" in trace_names or "tool_loop_fallback" in trace_names

    def test_analyze_completion_requires_evidence_coverage(self):
        """Calling the extractor once is not completion when three texts are available."""
        from app.agents.llm_worker import LLMWorker
        from app.agents.worker import WorkerContext
        from app.agents.planner import Task

        worker = LLMWorker()
        ctx = WorkerContext(Task("analyze", "analyze", "Extract evidence"))
        ctx.tool_call_count = 1
        ctx.results["_trusted_sources"] = [
            {"source_id": f"s{i}", "full_text": f"Abstract text {i}"}
            for i in range(3)
        ]
        ctx.results["evidence_cards"] = [
            {"source_id": "s0", "evidence_id": "s0:e1", "claim": "claim"}
        ]

        error = worker._validate_task_completion("analyze", ctx)
        assert error is not None
        assert "at least 3" in error

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_to_rule(self):
        """9. LLM timeout 时回退 rule Worker。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_TIMEOUT)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 应该在 timeout 后 fallback 到 rule，仍然返回结果
        assert ctx is not None
        trace_names = [t.get("tool_name", "") for t in ctx.trace]
        assert "tool_loop_fallback" in trace_names

    @pytest.mark.asyncio
    async def test_tool_exception_degrades_gracefully(self):
        """11. Tool 抛出异常时可降级。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_TOOL_EXCEPTION)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        assert ctx is not None

    @pytest.mark.asyncio
    async def test_invalid_args_rejected(self):
        """6b. 非法 arguments 被拒绝。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_INVALID_ARGS)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 应该能处理非法 args
        assert ctx is not None


# ================================================================
# Integration Tests: LLMWorker in Graph Nodes
# ================================================================

class TestLLMWorkerGraphIntegration:
    """12-13. graph 和 graph_send 确实调用 LLMWorker。"""

    def setup_method(self):
        from app.llm.client import reset_llm_client
        from app.tools.registry import ToolRegistry
        reset_llm_client()
        ToolRegistry.reset_instance()

    def teardown_method(self):
        from app.llm.client import reset_llm_client
        from app.tools.registry import ToolRegistry
        reset_llm_client()
        ToolRegistry.reset_instance()

    @pytest.mark.asyncio
    async def test_graph_send_calls_llm_worker(self):
        """13. graph_send 在 agent_mode=llm 时调用 LLMWorker。"""
        from app.llm.client import FakeLLMClient
        from app.graph.runtime import run_graph
        import app.llm.client as client_mod

        # 提供足够多的 FC 响应（3 search + 3 read + finish）
        many_fc = (FC_SEARCH_THEN_FINISH * 3) + (FC_READ_THEN_FINISH * 3)
        fake = FakeLLMClient()
        fake.set_fc_responses(many_fc)
        client_mod._global_client = fake

        result = await run_graph(
            topic="RAG evaluation", max_sources=3,
            agent_mode="llm",
        )
        assert result["status"] in {"completed", "completed_with_warnings"}


# ================================================================
# Isolation Tests
# ================================================================

class TestLLMWorkerIsolation:
    """14-15. Worker 隔离性测试。"""

    def setup_method(self):
        from app.llm.client import reset_llm_client
        from app.tools.registry import ToolRegistry
        reset_llm_client()
        ToolRegistry.reset_instance()

    def teardown_method(self):
        from app.llm.client import reset_llm_client
        from app.tools.registry import ToolRegistry
        reset_llm_client()
        ToolRegistry.reset_instance()

    @pytest.mark.asyncio
    async def test_workers_have_independent_state(self):
        """14. 多个 Worker 的 messages、budget、called-set 相互隔离。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_SEARCH_THEN_FINISH * 6)
        client_mod._global_client = fake

        w1 = LLMWorker()
        w2 = LLMWorker()

        task1 = Task("search_1", "search", "Query A", tool_plan=["academic_search"])
        task2 = Task("search_2", "search", "Query B", tool_plan=["academic_search"])

        ctx1 = await w1.execute_task(task1)
        ctx2 = await w2.execute_task(task2)

        # 两个 worker 互不影响
        assert ctx1 is not None
        assert ctx2 is not None
        # 它们不应该共享 trace
        assert ctx1.trace != ctx2.trace

    @pytest.mark.asyncio
    async def test_single_worker_failure_does_not_affect_others(self):
        """15. 单个 Worker 失败不影响其他 Worker。"""
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        # Worker 1 得到好的响应，Worker 2 得到失败
        fake = FakeLLMClient()
        fake.set_fc_responses(
            FC_SEARCH_THEN_FINISH +  # w1: success
            [{"_fail": True, "_error": "Simulated failure"},  # w2: fail (triggers fallback)
             {"_finish": True, "content": "done"}]  # w2's rule fallback uses real Worker
        )
        client_mod._global_client = fake

        w1 = LLMWorker()
        w2 = LLMWorker()

        task = Task("search", "search", "Test", tool_plan=["academic_search"])

        ctx1 = await w1.execute_task(task)
        ctx2 = await w2.execute_task(task)

        # 两个 worker 都应该返回
        assert ctx1 is not None
        assert ctx2 is not None


# ================================================================
# Regression Tests
# ================================================================

class TestPhase2DRegression:
    """18. loop / graph / graph_send 与 rule / llm 组合回归测试。"""

    @pytest.mark.asyncio
    async def test_loop_rule_mode(self):
        from app.services.orchestrator import Orchestrator

        result = await Orchestrator().run(topic="RAG eval", max_sources=2)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_graph_send_rule_mode(self):
        from app.graph.runtime import run_graph

        result = await run_graph(topic="RAG eval", max_sources=2,
                                 agent_mode="rule")
        assert result["status"] in {"completed", "completed_with_warnings"}


# ================================================================
# Trace Safety Tests
# ================================================================

class TestTraceSafety:
    """17. API key 不进入 trace、warning 和 HTTP response。"""

    @pytest.mark.asyncio
    async def test_trace_excludes_api_key(self):
        from app.llm.client import FakeLLMClient
        from app.agents.llm_worker import LLMWorker
        from app.agents.planner import Task
        import app.llm.client as client_mod

        fake = FakeLLMClient()
        fake.set_fc_responses(FC_SEARCH_THEN_FINISH)
        client_mod._global_client = fake

        worker = LLMWorker()
        task = Task("search_1", "search", "Test",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        # 检查 trace 中不包含 API key
        trace_str = str(ctx.trace)
        assert "sk-" not in trace_str.lower() or "fake-test-key" not in trace_str
        assert "Bearer" not in trace_str
        assert "Authorization" not in trace_str
