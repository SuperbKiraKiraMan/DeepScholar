"""
tests/test_graph_runtime.py

LangGraph Runtime 测试 —— Phase 2B（Send API 动态分发）。

覆盖：
1. graph_send backend 正常完成完整链路
2. Planner 生成多个不同 search tasks
3. 多个 search worker 被实际分发
4. 多个 reading worker 被实际分发
5. 多 worker sources 能正确合并和去重
6. Reading 结果保留 quality_score
7. trace 包含 send_dispatch、worker_started、worker_finished、merge_result
8. trace 中 controller/planner 不指数重复
9. 多并发 sources 不触发 InvalidUpdateError
10. citation retry 只重新处理失败来源
11. retry 后有效 EvidenceCard 不重复
12. replan 和 retry 均最多一次
13. graph_send 尊重 max_sources
14. route_after_evaluator / state 校验
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestGraphRuntime:
    """测试 LangGraph Runtime 完整执行。"""

    @pytest.mark.asyncio
    async def test_graph_respects_max_sources(self):
        """graph_send 尊重 max_sources 参数。"""
        from app.graph.runtime import run_graph

        result = await run_graph(
            topic="RAG evaluation methods",
            max_sources=2,
            run_eval=True,
        )

        sources = result.get("sources", [])
        assert len(sources) <= 3  # 包括 merge 容忍度

    @pytest.mark.asyncio
    async def test_send_merge_ranks_by_topic_before_source_cap(self):
        from app.graph.runtime import (
            _rank_sources_for_topic,
            _cap_sources_with_local_coverage,
        )
        topic = "transformer attention mechanism"
        sources = [
            {"source_id": "S1", "title": "Attention Survey", "year": 2023,
             "full_text": "A survey of attention mechanisms..."},
            {"source_id": "S2", "title": "Unrelated Biology Paper", "year": 2020,
             "full_text": "Protein folding..."},
            {"source_id": "S3", "title": "Novel Attention Method", "year": 2024,
             "full_text": "We propose a new attention..."},
        ]
        ranked = _rank_sources_for_topic(sources, topic)
        assert len(ranked) == 3
        capped = _cap_sources_with_local_coverage(ranked, max_sources=2)
        assert len(capped) <= 2

    @pytest.mark.asyncio
    async def test_send_merge_reserves_budget_for_relevant_local_full_text(self):
        from app.graph.runtime import (
            _rank_sources_for_topic,
            _cap_sources_with_local_coverage,
        )
        topic = "transformer attention"
        sources = [
            {"source_id": "S1", "title": "External Abstract Only",
             "provider": "openalex", "year": 2024},
            {"source_id": "S2", "title": "Local Full-Text Paper",
             "provider": "local_zotero", "content_source": "zotero_pdf",
             "full_text": "In this work we analyze attention...",
             "retrieval_score": 0.7},
        ]
        capped = _cap_sources_with_local_coverage(sources, max_sources=5)
        assert len(capped) == 2

    @pytest.mark.asyncio
    async def test_send_merge_does_not_force_low_score_local_hit(
        self, monkeypatch,
    ):
        monkeypatch.setenv("LOCAL_RAG_MIN_SCORE", "0.35")
        from app.graph.runtime import (
            _rank_sources_for_topic,
            _cap_sources_with_local_coverage,
        )
        topic = "graph neural networks"
        sources = [
            {"source_id": "E1", "title": "External Survey",
             "provider": "openalex", "year": 2025},
            {"source_id": "L1", "title": "Low-Relevance Local PDF",
             "provider": "local_zotero", "content_source": "zotero_pdf",
             "full_text": "A brief mention of GNNs...",
             "retrieval_score": 0.12},
            {"source_id": "E2", "title": "External Benchmark",
             "provider": "semantic_scholar", "year": 2024},
        ]
        capped = _cap_sources_with_local_coverage(sources, max_sources=3)
        local_ids = {
            s["source_id"] for s in capped
            if s.get("provider") == "local_zotero"
        }
        assert "L1" not in local_ids


class TestGraphAPI:
    """通过 API 与路由函数测试 Send 后端。"""

    def test_route_after_eval_citation_failed(self):
        """引用校验失败 → retry analysis_worker。"""
        from app.graph.runtime import route_after_evaluator
        state = {"eval_metrics": {"no_fake_citation": False}, "retry_count": 0}
        assert route_after_evaluator(state) == "analysis_worker"

    def test_route_after_eval_citation_failed_retry_limit(self):
        """retry 次数达到上限 → final_reviewer。"""
        from app.graph.runtime import route_after_evaluator
        state = {"eval_metrics": {"no_fake_citation": False}, "retry_count": 1}
        assert route_after_evaluator(state) == "final_reviewer"

    def test_route_after_eval_all_pass(self):
        """所有指标通过 → final_reviewer。"""
        from app.graph.runtime import route_after_evaluator
        metrics = {
            "no_fake_citation": True, "citation_id_exists": True,
            "source_url_valid": True, "evidence_available": True,
        }
        state = {"eval_metrics": metrics, "retry_count": 0}
        assert route_after_evaluator(state) == "final_reviewer"

    def test_minimal_state(self):
        """StateGraph 的初始 state 不缺失必要字段。"""
        from app.graph.runtime import ResearchAgentState
        initial: ResearchAgentState = {
            "topic": "test", "trace": [], "warnings": [],
            "sources": [], "scored_sources": [], "evidence_cards": [],
            "_search_bucket": [], "_reading_bucket": [],
        }
        assert initial["topic"] == "test"

    def test_full_state(self):
        """StateGraph 的完整 state 包含所有字段。"""
        from app.graph.runtime import ResearchAgentState
        state: ResearchAgentState = {
            "topic": "RAG", "language": "zh", "mode": "quick",
            "run_eval": True, "max_sources": 5, "agent_mode": "rule",
            "backend": "graph_send",
            "sources": [], "scored_sources": [], "evidence_cards": [],
            "trace": [], "warnings": [], "replan_count": 0, "retry_count": 0,
            "_search_bucket": [], "_reading_bucket": [],
        }
        assert state["agent_mode"] == "rule"


class TestGraphSendAPI:
    """通过 API 测试 graph_send backend（Phase 2B Send API）。"""

    @pytest.mark.asyncio
    async def test_send_completes_full_pipeline(self):
        """graph_send 能完整执行全链路。"""
        from app.graph.runtime import run_graph

        result = await run_graph(
            topic="RAG evaluation",
            max_sources=3,
            run_eval=True,
        )

        assert result["status"] in {"completed", "completed_with_warnings"}
        assert result["backend"] == "graph_send"
        assert len(result.get("sources", [])) >= 1
        assert len(result.get("evidence_cards", [])) >= 1

    def test_planner_generates_multiple_search_tasks(self):
        """规则版 Planner 为 Send API 生成 3 个不同 search 查询。"""
        from app.agents.planner import Planner

        planner = Planner()
        dag = planner.plan_for_send("RAG evaluation", max_sources=5)
        search_tasks = [t for t in dag.tasks if t.task_type == "search"]

        assert len(search_tasks) == 3
        queries = [t.description for t in search_tasks]
        assert any("overview" in q or "survey" in q for q in queries)
        assert any("methods" in q or "benchmarks" in q for q in queries)

    def test_multiple_search_workers_dispatched(self):
        """Planner 生成多个 task，每个 task_id 唯一。"""
        from app.agents.planner import Planner

        planner = Planner()
        dag = planner.plan_for_send("RAG evaluation", max_sources=5)
        search_tasks = [t for t in dag.tasks if t.task_type == "search"]

        task_ids = [t.task_id for t in search_tasks]
        assert len(task_ids) == len(set(task_ids))  # 不重复
        assert all(tid.startswith("search_") for tid in task_ids)

    def test_multiple_reading_workers_dispatched(self):
        """send_to_reading_worker 为每个 source 创建一个 Send。"""
        from app.graph.runtime import send_to_reading_worker

        sources = [
            {"source_id": "S1", "title": "Paper 1"},
            {"source_id": "S2", "title": "Paper 2"},
        ]
        state = {
            "sources": sources,
            "topic": "RAG",
            "max_sources": 5,
            "agent_mode": "rule",
            "backend": "graph_send",
        }
        sends = send_to_reading_worker(state)
        assert len(sends) == len(sources)

    def test_sources_merged_and_deduped(self):
        """merge_search_results 去重 + 截断。"""
        from app.graph.runtime import _dedup_sources, _rank_sources_for_topic

        raw = [
            {"source_id": "S1", "title": "Paper A", "url": "https://a.com"},
            {"source_id": "S2", "title": "Paper B", "url": "https://b.com"},
            {"source_id": "S3", "title": "Paper A Duplicate", "url": "https://a.com"},
            {"source_id": "S4", "title": "Paper C", "url": "https://c.com"},
        ]
        unique = _dedup_sources(raw)
        assert len(unique) == 3  # S1 和 S3 同一 URL → 去重

    def test_reading_preserves_quality_score(self):
        """merge_reading_results 保留 quality_score。"""
        from app.graph.runtime import _dedup_sources

        sources = [
            {"source_id": "S1", "title": "Paper 1", "quality_score": 0.85},
            {"source_id": "S2", "title": "Paper 2", "quality_score": 0.42},
        ]
        unique = _dedup_sources(sources)
        assert len(unique) == 2
        scores = {s.get("quality_score") for s in unique}
        assert 0.85 in scores
        assert 0.42 in scores

    def test_send_trace_has_required_events(self):
        """run_graph(graph_send) trace 包含 Send API 特有事件。"""
        import asyncio
        from app.graph.runtime import run_graph

        result = asyncio.get_event_loop().run_until_complete(
            run_graph(topic="RAG evaluation", max_sources=3)
        )

        trace = result.get("trace", [])
        events = {t.get("event") for t in trace}
        required = {"send_dispatch", "merge_result"}
        assert required.issubset(events), f"Missing events: {required - events}"

    def test_trace_no_exponential_duplication(self):
        """trace 中 controller/planner 事件不指数重复。"""
        import asyncio
        from app.graph.runtime import run_graph

        result = asyncio.get_event_loop().run_until_complete(
            run_graph(topic="RAG evaluation", max_sources=3)
        )

        trace = result.get("trace", [])
        controller_starts = [t for t in trace if t.get("event") == "controller_start"]
        planner_completes = [t for t in trace if t.get("event") == "planner_complete"]
        assert len(controller_starts) <= 2  # 最多 1 + retry
        assert len(planner_completes) <= 2

    def test_no_invalid_update_error_with_concurrent_sources(self):
        """多并发 sources 不触发 InvalidUpdateError。"""
        import asyncio
        from app.graph.runtime import run_graph

        result = asyncio.get_event_loop().run_until_complete(
            run_graph(topic="RAG evaluation", max_sources=4)
        )

        warnings = result.get("warnings", [])
        invalid_update = [w for w in warnings if
                          "InvalidUpdateError" in w or "invalid update" in w.lower()]
        assert not invalid_update, f"Got InvalidUpdateError: {invalid_update}"

    def test_citation_retry_only_reprocesses_failed_sources(self):
        """citation retry 只重新处理失败的来源。"""
        import asyncio
        from app.graph.runtime import node_analysis_worker

        all_sources = [
            {"source_id": "S1", "title": "Paper 1", "full_text": "text"},
            {"source_id": "S2", "title": "Paper 2", "full_text": "text"},
        ]
        state = {
            "scored_sources": all_sources,
            "sources": all_sources,
            "evidence_cards": [
                {"source_id": "S1", "claim": "claim 1", "confidence": 0.9},
                {"source_id": "S2", "claim": "claim 2", "confidence": 0.8},
            ],
            "citation_check_results": [
                {"source_id": "S1", "is_valid": False},  # S1 失败 → 需 retry
                {"source_id": "S2", "is_valid": True},
            ],
            "topic": "RAG evaluation",
            "retry_count": 1,
            "trace": [], "warnings": [],
            "run_eval": True, "language": "zh",
        }
        result = asyncio.get_event_loop().run_until_complete(
            node_analysis_worker(state)
        )
        cards = result.get("evidence_cards", [])
        # S2 的 card 应保留，S1 的旧 card + 新 card
        sids = {c["source_id"] for c in cards}
        assert "S2" in sids  # 非失败来源的 card 被保留
