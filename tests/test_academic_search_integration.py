"""
tests/test_academic_search_integration.py

Phase 4B: Tool Registry / Planner / Worker / FC Schema 集成测试。

验证所有组件统一使用 "academic_search" 工具名。
"""

import pytest

from app.tools.registry import ToolRegistry
from app.agents.planner import Planner
from app.agents.worker import Worker
from app.llm.schemas import LLMPlannerOutput
from app.api.schemas import PaperSource
from app.tools.paper_metadata_tool import PaperMetadataTool


class TestToolRegistry:
    """ToolRegistry 注册 academic_search。"""

    def test_registry_registers_academic_search(self):
        ToolRegistry.reset_instance()
        registry = ToolRegistry.get_instance()
        names = registry.list_names()
        assert "academic_search" in names
        assert "mock_academic_search" not in names

    def test_registry_get_academic_search(self):
        ToolRegistry.reset_instance()
        registry = ToolRegistry.get_instance()
        tool = registry.get("academic_search")
        assert tool is not None
        assert tool.name == "academic_search"

    def test_fc_schema_includes_academic_search(self):
        """FC schema 使用 academic_search。"""
        ToolRegistry.reset_instance()
        registry = ToolRegistry.get_instance()
        schemas = registry.get_function_schemas(["academic_search"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "academic_search"

    def test_fc_schema_excludes_mock_academic_search(self):
        """FC schema 不暴露 mock_academic_search。"""
        ToolRegistry.reset_instance()
        registry = ToolRegistry.get_instance()
        # 即使请求 mock_academic_search，也不应出现
        schemas = registry.get_function_schemas(["mock_academic_search"])
        assert len(schemas) == 0

    def test_citation_check_not_in_fc_schema(self):
        """citation_check 仍然不通过 FC 暴露。"""
        ToolRegistry.reset_instance()
        registry = ToolRegistry.get_instance()
        schemas = registry.get_function_schemas(["citation_check"])
        assert len(schemas) == 0


class TestPlanner:
    """Planner 使用 academic_search。"""

    def test_plan_uses_academic_search(self):
        planner = Planner()
        dag = planner.plan("RAG evaluation", max_sources=5)
        search_task = dag.get_task("search")
        assert set(search_task.tool_plan) == {
            "local_paper_search",
            "academic_search",
            "semantic_scholar_search",
            "semantic_scholar_graph",
            "semantic_scholar_recommendations",
        }

    def test_plan_for_send_uses_academic_search(self):
        planner = Planner()
        dag = planner.plan_for_send("RAG evaluation", max_sources=5)
        search_1 = dag.get_task("search_1")
        expected = {
            "local_paper_search",
            "academic_search",
            "semantic_scholar_search",
            "semantic_scholar_graph",
            "semantic_scholar_recommendations",
        }
        assert set(search_1.tool_plan) == expected
        search_2 = dag.get_task("search_2")
        assert set(search_2.tool_plan) == expected
        search_3 = dag.get_task("search_3")
        assert set(search_3.tool_plan) == expected


class TestWorker:
    """Worker 使用 academic_search 工具。"""

    @pytest.mark.asyncio
    async def test_worker_has_academic_search_tool(self):
        """Worker._tools 包含 academic_search。"""
        worker = Worker()
        assert "academic_search" in worker._tools
        assert "mock_academic_search" not in worker._tools

    @pytest.mark.asyncio
    async def test_worker_execute_search(self):
        """Worker 执行搜索使用 academic_search。"""
        from app.agents.planner import Task
        worker = Worker()
        task = Task("search", "search", "Search for academic sources on: RAG evaluation",
                    tool_plan=["academic_search"])
        ctx = await worker.execute_task(task)

        sources = ctx.results.get("sources", [])
        assert len(sources) >= 1
        # 确认 tool_name 在 trace 中是 academic_search
        tool_names = [t.get("tool_name", "") for t in ctx.trace]
        assert "academic_search" in tool_names


class TestLLMWorkerDefaults:
    """LLMWorker._default_tools 使用 academic_search。"""

    def test_default_search_tool(self):
        from app.agents.llm_worker import LLMWorker
        worker = LLMWorker()
        tools = worker._default_tools("search")
        assert set(tools) == {
            "local_paper_search",
            "academic_search",
            "semantic_scholar_search",
            "semantic_scholar_graph",
            "semantic_scholar_recommendations",
        }

    def test_default_read_tools(self):
        from app.agents.llm_worker import LLMWorker
        worker = LLMWorker()
        tools = worker._default_tools("read")
        assert "paper_metadata" in tools
        assert "source_quality_scorer" in tools

    def test_default_analyze_tools(self):
        from app.agents.llm_worker import LLMWorker
        worker = LLMWorker()
        tools = worker._default_tools("analyze")
        assert tools == ["evidence_extract"]


class TestLLMSchemas:
    """LLM schemas 使用 academic_search。"""

    def test_search_task_default_allowed_tools(self):
        """LLMSearchTask 默认 allowed_tools 包含 academic_search。"""
        task = LLMPlannerOutput(
            research_goal="Test research goal here",
            search_tasks=[{
                "task_id": "search_1",
                "query": "test query",
            }],
        )
        assert task.search_tasks[0].allowed_tools == ["academic_search"]

    def test_validate_tools_accepts_academic_search(self):
        """validate_tools 接受 academic_search。"""
        output = LLMPlannerOutput(
            research_goal="Test research goal here",
            search_tasks=[{
                "task_id": "search_1",
                "query": "test query",
                "allowed_tools": ["academic_search"],
            }],
        )
        errors = output.validate_tools(["academic_search"])
        assert errors == []

    def test_validate_tools_rejects_unknown(self):
        """validate_tools 拒绝不存在的工具。"""
        output = LLMPlannerOutput(
            research_goal="Test research goal here",
            search_tasks=[{
                "task_id": "search_1",
                "query": "test query",
                "allowed_tools": ["nonexistent_tool"],
            }],
        )
        errors = output.validate_tools(["academic_search"])
        assert len(errors) > 0


class TestOpenAlexProvenance:
    @pytest.mark.asyncio
    async def test_metadata_normalization_preserves_provider_fields(self):
        source = {
            "source_id": "W1", "title": "A Book", "url": "https://openalex.org/W1",
            "snippet": "summary", "full_text": "text", "authors": [], "year": 2024,
            "venue": "Venue", "source_type": "book", "provider": "openalex",
            "openalex_id": "W1", "doi": "https://doi.org/10.1/example",
            "cited_by_count": 12, "is_oa": True, "oa_status": "gold",
            "content_url": "https://content.openalex.org/works/W1.grobid-xml",
            "has_content": {"grobid_xml": True}, "content_source": "openalex_tei",
        }
        result = await PaperMetadataTool().run(sources=[source])
        normalized = result.data["sources"][0]

        assert normalized["source_type"] == "book"
        assert normalized["provider"] == "openalex"
        assert normalized["doi"] == source["doi"]
        assert normalized["content_source"] == "openalex_tei"
        assert PaperSource(**normalized).model_dump()["openalex_id"] == "W1"

    @pytest.mark.asyncio
    async def test_rule_worker_surfaces_mock_fallback(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "openalex")
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        monkeypatch.setenv("OPENALEX_FALLBACK_TO_MOCK", "true")
        from app.agents.planner import Task

        ctx = await Worker().execute_task(
            Task("search", "search", "Search for academic sources on: RAG evaluation",
                 tool_plan=["academic_search"])
        )

        assert any("provider fallback" in warning.lower() for warning in ctx.warnings)
        assert any(t.get("tool_name") == "provider_fallback" for t in ctx.trace)

        from app.graph.runtime import _merge_worker_trace
        events = _merge_worker_trace(ctx)
        fallback = next(e for e in events if e["event"] == "provider_fallback")
        assert fallback["provider"] == "mock"
        assert fallback["fallback_reason"] == "no_api_key"
