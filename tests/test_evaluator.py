"""
tests/test_evaluator.py

Evaluator 测试 —— Phase 1B。
"""

from app.agents.evaluator import Evaluator


class TestEvaluator:
    """测试 Evaluator。"""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_all_metrics_pass_on_good_input(self):
        """好数据输入 → 所有指标通过。"""
        result = self.evaluator.evaluate(
            topic="RAG evaluation",
            draft_report="# Research Report\n\nThis is a comprehensive research report "
                          "on RAG evaluation methods. It covers multiple sources and "
                          "provides detailed analysis of citation accuracy and evidence quality [1].\n\n"
                          "## Conclusion\n\nThe available evidence supports this bounded conclusion [1].",
            sources=[{"source_id": "s1", "url": "http://a.com"}],
            evidence_cards=[{"claim": "test", "source_id": "s1"}],
            citation_check_results=[
                {
                    "citation_id": 1,
                    "source_id": "s1",
                    "id_exists": True,
                    "url_matches_source": True,
                    "quote_found_in_source": True,
                    "is_valid": True,
                    "issues": [],
                }
            ],
            citation_summary={"total_checked": 1, "valid_count": 1, "invalid_count": 0, "all_valid": True},
            trace=[
                {"event": "tool_finished", "task_id": "search",
                 "tool_name": "academic_search", "success": True},
                {"event": "tool_finished", "task_id": "read",
                 "tool_name": "paper_metadata", "success": True},
            ],
            task_dag={"tasks": [{"task_id": "search"}, {"task_id": "read"}]},
            total_latency_ms=5000,
        )

        assert result["all_passed"] is True
        metrics = result["metrics"]
        assert metrics["no_fake_citation"] is True
        assert metrics["min_sources"] is True
        assert metrics["citation_id_exists"] is True
        assert metrics["source_url_valid"] is True
        assert metrics["evidence_available"] is True
        assert metrics["answer_not_empty"] is True
        assert metrics["task_success_rate"] == 1.0
        assert metrics["tool_error_rate"] is True
        assert metrics["latency_under_threshold"] is True
        assert len(result["feedback"]) == 0

    def test_fake_citation_detected(self):
        """检测到 fake citation → no_fake_citation = False。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            sources=[{"source_id": "s1"}],
            citation_summary={"total_checked": 2, "valid_count": 1, "invalid_count": 1, "all_valid": False},
        )

        assert result["metrics"]["no_fake_citation"] is False
        assert len(result["feedback"]) >= 1

    def test_min_sources_fails_with_empty(self):
        """来源数不够 → min_sources = False。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            sources=[],
        )

        assert result["metrics"]["min_sources"] is False

    def test_answer_not_empty_fails_with_short(self):
        """报告太短 → answer_not_empty = False。"""
        result = self.evaluator.evaluate(
            draft_report="short",
        )

        assert result["metrics"]["answer_not_empty"] is False

    def test_citation_id_exists_fails(self):
        """有 citation 的 id_exists 为 False → 指标失败。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            citation_check_results=[
                {
                    "citation_id": 1,
                    "id_exists": False,
                    "url_matches_source": True,
                    "quote_found_in_source": True,
                    "is_valid": False,
                }
            ],
        )

        assert result["metrics"]["citation_id_exists"] is False

    def test_source_url_valid_fails(self):
        """URL 不匹配 → source_url_valid = False。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            citation_check_results=[
                {
                    "citation_id": 1,
                    "id_exists": True,
                    "url_matches_source": False,
                    "quote_found_in_source": True,
                    "is_valid": False,
                }
            ],
        )

        assert result["metrics"]["source_url_valid"] is False

    def test_tool_error_rate_fails_with_many_errors(self):
        """大量工具错误 → tool_error_rate = False。"""
        trace = [
            {"event": "tool_finished", "task_id": f"task_{i}",
             "tool_name": "academic_search", "success": False}
            for i in range(10)
        ]  # 100% error rate
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            trace=trace,
        )

        assert result["metrics"]["tool_error_rate"] is False

    def test_worker_final_success_overrides_recoverable_internal_tool_error(self):
        """Worker 已恢复并成功完成时，内部提供方错误不能把任务误判为失败。"""
        trace = [
            {"event": "worker_started", "task_id": "search_1", "success": True},
            # Send 节点的汇总事件当前位于详细工具 Trace 之前。
            {"event": "worker_finished", "task_id": "search_1", "success": True},
            {
                "event": "tool_finished",
                "task_id": "search_1",
                "tool_name": "semantic_scholar_graph",
                "success": False,
                "error": "one expansion provider failed",
            },
            {
                "event": "tool_finished",
                "task_id": "search_1",
                "tool_name": "academic_search",
                "success": True,
            },
        ]

        detail = self.evaluator._calc_task_success_rate(
            trace, {"tasks": [{"task_id": "search_1"}]}
        )

        assert detail["rate"] == 1.0
        assert detail["passed"] is True
        assert detail["task_outcomes"]["search_1"] == "success"

    def test_latency_exceeds_threshold(self):
        """延迟超阈值 → latency_under_threshold = False。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            total_latency_ms=240_000,  # 240 秒，超过默认 180 秒 TTL
        )

        assert result["metrics"]["latency_under_threshold"] is False
        assert result["metrics_detail"]["latency"]["threshold_ms"] == 180_000

    def test_extended_latency_ttl_accepts_valid_long_research(self):
        """合法的长 Search + Reviewer 运行不应被旧 60 秒阈值误报。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            total_latency_ms=129_615,
        )

        assert result["metrics"]["latency_under_threshold"] is True
        assert result["metrics_detail"]["latency"]["overage_ms"] == 0

    def test_latency_ttl_can_be_configured(self, monkeypatch):
        """端到端 TTL 可配置，但不影响单工具 timeout。"""
        monkeypatch.setenv("RESEARCH_LATENCY_TTL_SECONDS", "2")
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            total_latency_ms=2_001,
        )

        assert result["metrics"]["latency_under_threshold"] is False
        assert result["metrics_detail"]["latency"]["threshold_ms"] == 2_000

    def test_latency_detail_uses_max_for_parallel_search_workers(self):
        """Send 并行分支取关键路径最大值，不把各 Worker 耗时相加。"""
        trace = [
            {"event": "send_dispatch", "worker_type": "search"},
            {"event": "send_dispatch", "worker_type": "search"},
            {"event": "node_observed", "observed_node": "search_worker_send", "latency_ms": 69_063},
            {"event": "node_observed", "observed_node": "search_worker_send", "latency_ms": 35_053},
            {"event": "node_observed", "observed_node": "reading_worker_send", "latency_ms": 13_068},
            {"event": "tool_finished", "tool_name": "semantic_scholar_graph",
             "success": True, "latency_ms": 29_258},
            {"event": "llm_finished", "agent": "llm_reviewer",
             "success": True, "latency_ms": 31_881},
        ]
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            trace=trace,
            total_latency_ms=129_615,
        )
        detail = result["metrics_detail"]["latency"]

        assert detail["search_stage_max_ms"] == 69_063
        assert detail["reading_stage_max_ms"] == 13_068
        assert detail["slowest_node"]["name"] == "search_worker_send"
        assert detail["slowest_tool"]["name"] == "semantic_scholar_graph"
        assert detail["slowest_llm"]["name"] == "llm_reviewer"
        assert detail["send_dispatch_count"] == 2

    def test_metrics_detail_includes_pass_rate(self):
        """结果包含 metrics_detail。"""
        result = self.evaluator.evaluate(
            draft_report="test report content for evaluation",
            sources=[{"source_id": "s1"}],
        )

        detail = result["metrics_detail"]
        assert "passed_count" in detail
        assert "total_count" in detail
        assert "pass_rate" in detail
        assert detail["total_count"] == 12

    def test_empty_inputs_no_crash(self):
        """空输入不崩溃。"""
        result = self.evaluator.evaluate()

        assert "metrics" in result
        assert "feedback" in result
        assert len(result["metrics"]) == 12

    def test_no_evidence_fails_even_when_sources_and_report_exist(self):
        result = self.evaluator.evaluate(
            draft_report="A polished but unsupported report that is deliberately longer than fifty characters.",
            sources=[{"source_id": "s1"}],
            evidence_cards=[],
        )
        assert result["metrics"]["evidence_available"] is False
        assert result["all_passed"] is False
