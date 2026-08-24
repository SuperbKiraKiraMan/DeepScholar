"""
tests/test_worker.py

Worker Tool Loop 测试 —— Phase 1B。
"""

import pytest
from app.agents.planner import Task
from app.agents.worker import Worker, WorkerContext


class TestWorkerToolLoop:
    """测试 Worker Tool Loop。"""

    def setup_method(self):
        self.worker = Worker()

    @pytest.mark.asyncio
    async def test_execute_search_task(self):
        """执行搜索任务。"""
        task = Task(
            task_id="search",
            task_type="search",
            description="Search for academic sources on: RAG evaluation",
            tool_plan=["academic_search"],
        )

        ctx = await self.worker.execute_task(task)

        assert isinstance(ctx, WorkerContext)
        assert len(ctx.trace) >= 1
        assert "search_results" in ctx.results or "sources" in ctx.results
        # 搜索应返回结果
        sources = ctx.results.get("sources", ctx.results.get("search_results", []))
        assert len(sources) >= 1

    @pytest.mark.asyncio
    async def test_execute_search_trace_records_tool_calls(self):
        """搜索任务的 trace 记录了工具调用。"""
        task = Task(
            task_id="search",
            task_type="search",
            description="Search for academic sources on: RAG evaluation",
            tool_plan=["academic_search"],
        )

        ctx = await self.worker.execute_task(task)

        for entry in ctx.trace:
            assert "step" in entry
            assert "task_id" in entry
            assert entry["task_id"] == "search"
            assert "tool_name" in entry
            assert "success" in entry
            assert "latency_ms" in entry

    @pytest.mark.asyncio
    async def test_execute_read_task(self):
        """执行阅读任务（元数据 + 评分）。"""
        # 先搜索获取 sources
        search_task = Task(
            task_id="search",
            task_type="search",
            description="Search for academic sources on: RAG evaluation",
            tool_plan=["academic_search"],
        )
        search_ctx = await self.worker.execute_task(search_task)

        # 再执行 read
        read_task = Task(
            task_id="read",
            task_type="read",
            description="Fetch metadata, score quality, and normalize 5 sources",
            depends_on=["search"],
            tool_plan=["paper_metadata", "source_quality_scorer"],
        )

        read_ctx = await self.worker.execute_task(
            read_task,
            dependency_results={"search": search_ctx},
        )

        assert len(read_ctx.trace) >= 2  # metadata + scorer
        assert "scored_sources" in read_ctx.results
        scored = read_ctx.results["scored_sources"]
        assert len(scored) >= 1
        # 每个 source 应该有 quality_score
        for s in scored:
            assert "quality_score" in s
            assert isinstance(s["quality_score"], (int, float))

    @pytest.mark.asyncio
    async def test_execute_analyze_task(self):
        """执行分析任务（证据抽取）。"""
        # 搜索 + 阅读
        search_task = Task("search", "search", "Search for academic sources on: RAG evaluation")
        search_ctx = await self.worker.execute_task(search_task)

        read_task = Task("read", "read", "Fetch metadata, score quality, and normalize 5 sources",
                         depends_on=["search"])
        read_ctx = await self.worker.execute_task(read_task, {"search": search_ctx})

        # 分析
        analyze_task = Task("analyze", "analyze", "Extract evidence cards",
                            depends_on=["read"])
        analyze_ctx = await self.worker.execute_task(
            analyze_task,
            dependency_results={"read": read_ctx},
        )

        assert "evidence_cards" in analyze_ctx.results
        cards = analyze_ctx.results["evidence_cards"]
        assert len(cards) >= 1
        for card in cards:
            assert "claim" in card
            assert "source_id" in card
            assert "confidence" in card

    @pytest.mark.asyncio
    async def test_execute_cite_task(self):
        """执行引用校验任务。"""
        # 完整前置链路：search → read → analyze
        search_task = Task("search", "search", "Search for academic sources on: RAG evaluation")
        search_ctx = await self.worker.execute_task(search_task)

        read_task = Task("read", "read", "Fetch metadata, score quality, and normalize 5 sources",
                         depends_on=["search"])
        read_ctx = await self.worker.execute_task(read_task, {"search": search_ctx})

        analyze_task = Task("analyze", "analyze", "Extract evidence cards",
                            depends_on=["read"])
        analyze_ctx = await self.worker.execute_task(analyze_task, {"read": read_ctx})

        # 引用校验
        cite_task = Task("cite", "cite", "Check citation validity",
                         depends_on=["analyze"])
        cite_ctx = await self.worker.execute_task(
            cite_task,
            dependency_results={"read": read_ctx, "analyze": analyze_ctx},
        )

        assert "citation_check_results" in cite_ctx.results
        assert "citation_summary" in cite_ctx.results
        summary = cite_ctx.results["citation_summary"]
        assert "total_checked" in summary
        assert "valid_count" in summary

    @pytest.mark.asyncio
    async def test_unknown_task_type(self):
        """未知 task_type 记录 warning，不崩溃。"""
        task = Task("unknown", "unknown_type", "Do something weird")

        ctx = await self.worker.execute_task(task)

        assert len(ctx.warnings) >= 1
        assert "unknown" in str(ctx.warnings).lower()

    @pytest.mark.asyncio
    async def test_worker_context_structure(self):
        """WorkerContext 包含 trace、results、warnings。"""
        ctx = WorkerContext(Task("test", "search", "test desc"))

        assert ctx.task.task_id == "test"
        assert ctx.trace == []
        assert ctx.results == {}
        assert ctx.warnings == []
        assert ctx.tool_call_count == 0

    @pytest.mark.asyncio
    async def test_end_to_end_full_pipeline(self):
        """
        端到端：search → read → analyze → cite 全链路。

        这是 Phase 1B 的核心集成测试。
        """
        task_sequence = [
            Task("search", "search", "Search for academic sources on: RAG evaluation"),
            Task("read", "read", "Fetch metadata, score quality, and normalize 5 sources",
                 depends_on=["search"]),
            Task("analyze", "analyze", "Extract evidence cards",
                 depends_on=["read"]),
            Task("cite", "cite", "Check citation validity",
                 depends_on=["analyze"]),
        ]

        dep_results = {}
        for task in task_sequence:
            ctx = await self.worker.execute_task(task, dep_results)
            dep_results[task.task_id] = ctx

        # 验证各阶段结果
        assert "search" in dep_results
        assert "read" in dep_results
        assert "analyze" in dep_results
        assert "cite" in dep_results

        # 最终要有 trace
        total_trace = sum(len(ctx.trace) for ctx in dep_results.values())
        assert total_trace >= 4  # 至少 4 次工具调用

        # 最终要有 citation check 结果
        cite_ctx = dep_results["cite"]
        assert "citation_check_results" in cite_ctx.results
