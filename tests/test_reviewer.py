"""
tests/test_reviewer.py

Draft Reviewer + Final Reviewer 测试 —— Phase 1B。

FinalReviewer 测试重点：实质修复（删除假内容、诚实记录修不了的），不是追加注解。
"""

from app.agents.draft_reviewer import DraftReviewer
from app.agents.final_reviewer import FinalReviewer


class TestDraftReviewer:
    """Draft Reviewer 测试不变。"""

    def setup_method(self):
        self.reviewer = DraftReviewer()

    def test_chinese_language_builds_chinese_fallback_report(self):
        result = self.reviewer.review(
            topic="多模态实体对齐",
            sources=[{
                "source_id": "s1",
                "title": "Multimodal Entity Alignment",
                "provider": "test",
                "full_text": "Entity alignment matches equivalent entities.",
            }],
            evidence_cards=[{
                "source_id": "s1",
                "claim": "Entity alignment matches equivalent entities.",
                "quote": "Entity alignment matches equivalent entities.",
            }],
            citation_check_results=[{"citation_id": 1, "is_valid": True}],
            citation_summary={
                "total_checked": 1, "valid_count": 1,
                "invalid_count": 0, "all_valid": True,
            },
            language="zh",
        )

        assert "# 调研报告：多模态实体对齐" in result["draft_report"]
        assert "## 执行摘要" in result["draft_report"]
        assert "## 结论" in result["draft_report"]

    def test_generates_report_with_sources(self):
        result = self.reviewer.review(
            topic="RAG evaluation",
            sources=[{
                "source_id": "s1", "title": "RAGAS Paper", "url": "http://a.com",
                "source_type": "paper", "year": 2023, "quality_score": 0.9,
            }],
            evidence_cards=[{
                "claim": "RAGAS proposes faithfulness metrics",
                "quote": "RAGAS proposes...", "source_id": "s1",
                "url": "http://a.com", "confidence": 0.9,
            }],
            citation_check_results=[{
                "citation_id": 1, "source_id": "s1", "id_exists": True,
                "url_matches_source": True, "quote_found_in_source": True,
                "is_valid": True, "issues": [],
            }],
            citation_summary={"total_checked": 1, "valid_count": 1, "invalid_count": 0, "all_valid": True},
        )
        report = result["draft_report"]
        assert "Introduction" in report
        assert "Key Findings" in report
        assert "Citation Validation" in report

    def test_empty_sources_generates_warning(self):
        result = self.reviewer.review(
            topic="RAG evaluation", sources=[], evidence_cards=[], citation_check_results=[],
        )
        assert len(result["warnings"]) >= 1


class TestFinalReviewer:
    """Final Reviewer —— 实质修复测试。"""

    def setup_method(self):
        self.reviewer = FinalReviewer()

    def test_chinese_language_localizes_unresolved_task_message(self):
        result = self.reviewer.review(
            draft_report="# 调研报告\n\n已有正文内容，长度足以通过基础检查。",
            eval_metrics={
                "metrics": {"task_success_rate": 0.9},
                "metrics_detail": {
                    "passed_count": 8,
                    "total_count": 9,
                    "task_success_rate": {
                        "rate": 0.9, "success": 9, "total": 10,
                    },
                },
            },
            language="zh",
        )

        assert "### 未解决的证据与执行问题" in result["final_report"]
        assert "任务成功率低于 100%" in result["final_report"]
        assert "task_success_rate < 100%" not in result["final_report"]

    # ================================================================
    # 无事发生时不变
    # ================================================================

    def test_no_feedback_no_change(self):
        """全部通过时报告不变。"""
        result = self.reviewer.review(
            draft_report="# Original Report\n\nSome content.",
            eval_metrics={"metrics": {
                "no_fake_citation": True, "min_sources": True,
                "answer_not_empty": True, "tool_error_rate": True,
            }},
        )
        assert "Original Report" in result["final_report"]
        assert result["fixes_applied"] == []
        assert result["unresolved_issues"] == []

    # ================================================================
    # 修复 1：假引用被删除（不是标记，是直接删掉）
    # ================================================================

    def test_fake_citations_deleted_from_text(self):
        """无效 [N] 被删除，不是标记为 UNVERIFIED。"""
        draft = (
            "# Report\n\n"
            "Study [1] shows method A works. Study [2] confirms. Both [1][2] agree."
        )

        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"no_fake_citation": False}},
            citation_check_results=[
                {"citation_id": 1, "is_valid": True},
                {"citation_id": 2, "is_valid": False},  # 假引用
            ],
        )

        report = result["final_report"]
        # [2] 被删除
        assert "[2]" not in report or "UNVERIFIED" not in report
        # [1] 保持不变
        assert "[1]" in report
        # 修复记录
        assert len(result["fixes_applied"]) >= 1
        assert any("Deleted" in f and "2" in f for f in result["fixes_applied"])

    def test_sentence_with_only_fake_refs_deleted(self):
        """一句话中所有引用都是假的 → 整句删除。"""
        draft = (
            "# Report\n\n"
            "Real finding [1] is important.\n"
            "Fake finding [3] is wrong.\n"
            "Another real one [2] here."
        )

        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"no_fake_citation": False}},
            citation_check_results=[
                {"citation_id": 1, "is_valid": True},
                {"citation_id": 2, "is_valid": True},
                {"citation_id": 3, "is_valid": False},
            ],
        )

        report = result["final_report"]
        # [3] 的句子被整句删除
        assert "Fake finding" not in report
        assert "[3]" not in report
        # 有效句子保留
        assert "Real finding" in report
        assert "Another real" in report

    def test_valid_citations_untouched(self):
        """全部有效时，不删除任何引用。"""
        draft = "# Report\n\nStudy [1] and [2] show results."
        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"no_fake_citation": True}},
            citation_check_results=[
                {"citation_id": 1, "is_valid": True},
                {"citation_id": 2, "is_valid": True},
            ],
        )
        assert "Study [1] and [2] show results." in result["final_report"]

    # ================================================================
    # 修复 2：清理引用 section 中的无效行
    # ================================================================

    def test_invalid_citation_rows_removed(self):
        """引用 table 中无效引用的行被删除。"""
        draft = (
            "# Report\n\n"
            "## Sources Retrieved\n\n"
            "| [1] | Real Paper |\n"
            "| [2] | Fake Paper |\n"
            "| [3] | Another Real |\n"
        )

        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"no_fake_citation": False}},
            citation_check_results=[
                {"citation_id": 1, "is_valid": True},
                {"citation_id": 2, "is_valid": False},
                {"citation_id": 3, "is_valid": True},
            ],
        )

        report = result["final_report"]
        assert "Real Paper" in report
        assert "Another Real" in report
        assert "Fake Paper" not in report

    # ================================================================
    # 修复 3：警告横幅
    # ================================================================

    def test_banner_says_removed_not_marked(self):
        """假引用横幅说"已删除"，不是说"已标记"。"""
        draft = "# Report\n\nContent [1] here."
        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"no_fake_citation": False}},
            citation_check_results=[{"citation_id": 1, "is_valid": False}],
        )
        report = result["final_report"]
        assert "removed" in report.lower()

    def test_banner_for_few_sources(self):
        """来源不足 → Coverage Warning。"""
        draft = "# Report\n\nContent."
        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"no_fake_citation": True, "min_sources": False}},
        )
        assert "Coverage Warning" in result["final_report"]

    # ================================================================
    # 修复 3：URL 强制覆盖
    # ================================================================

    def test_url_fixed_from_source(self):
        """URL 不匹配 → 用 PaperSource 的权威 URL 替换报告中的错误 URL。"""
        draft = (
            "# Report\n\n"
            "According to the study, RAGAS is effective "
            "(https://wrong-url.com/paper).\n"
        )

        result = self.reviewer.review(
            draft_report=draft,
            eval_metrics={"metrics": {"source_url_valid": False}},
            citation_check_results=[
                {
                    "citation_id": 1,
                    "source_id": "s1",
                    "id_exists": True,
                    "url_matches_source": False,
                    "is_valid": False,
                }
            ],
            sources=[
                {
                    "source_id": "s1",
                    "url": "https://arxiv.org/abs/2309.15217",
                    "title": "RAGAS Paper",
                    "full_text": "RAGAS proposes evaluation metrics...",
                }
            ],
        )

        report = result["final_report"]
        # 错误 URL 被替换为正确的
        assert "wrong-url.com" not in report
        # 正确 URL 出现
        assert "arxiv.org/abs/2309.15217" in report
        # 修复被记录
        assert any("URL" in f for f in result["fixes_applied"])

    # ================================================================
    # 修复 4：UNRESOLVED ISSUES
    # ================================================================

    def test_unresolved_issues_when_min_sources_fails(self):
        """min_sources 失败 → UNRESOLVED ISSUES section 存在。"""
        result = self.reviewer.review(
            draft_report="# Report\n\nContent.",
            eval_metrics={"metrics": {"min_sources": False}},
        )
        report = result["final_report"]
        assert "UNRESOLVED ISSUES" in report
        assert len(result["unresolved_issues"]) >= 1
        assert any("min_sources" in i.lower() for i in result["unresolved_issues"])

    def test_unresolved_issues_when_latency_fails(self):
        """延迟超阈值 → UNRESOLVED ISSUES（列出完整优化策略）。"""
        result = self.reviewer.review(
            draft_report="# Report\n\nContent.",
            eval_metrics={
                "metrics": {"latency_under_threshold": False},
                "metrics_detail": {
                    "latency": {
                        "slowest_node": {
                            "name": "search_worker_send",
                            "latency_ms": 69063,
                        }
                    }
                },
            },
        )
        assert len(result["unresolved_issues"]) >= 1
        issue_text = " ".join(result["unresolved_issues"]).lower()
        assert "latency" in issue_text
        assert "search_worker_send" in issue_text
        assert any(kw in issue_text for kw in ("send api", "parallel", "timeout"))

    def test_unresolved_empty_when_all_pass(self):
        """全部通过时 unresolved 为空。"""
        result = self.reviewer.review(
            draft_report="# Report",
            eval_metrics={"metrics": {
                "no_fake_citation": True, "min_sources": True,
                "citation_id_exists": True, "source_url_valid": True,
                "answer_not_empty": True, "task_success_rate": 1.0,
                "tool_error_rate": True, "latency_under_threshold": True,
            }},
        )
        assert result["unresolved_issues"] == []

    def test_multiple_unresolved_collected(self):
        """多个指标失败 → 多个 unresloved issues。"""
        result = self.reviewer.review(
            draft_report="# Report\n\nContent.",
            eval_metrics={"metrics": {
                "min_sources": False,
                "tool_error_rate": False,
                "latency_under_threshold": False,
            }},
        )
        assert len(result["unresolved_issues"]) >= 3

    # ================================================================
    # 薄报告补全（只使用有效来源的 evidence）
    # ================================================================

    def test_thin_report_expanded_with_valid_only(self):
        """answer_not_empty 失败时，只用有效来源的 evidence 补全。"""
        result = self.reviewer.review(
            draft_report="# Short\n\nToo short.",
            eval_metrics={"metrics": {"answer_not_empty": False}},
            evidence_cards=[
                {"claim": "Valid claim A", "source_id": "s1", "confidence": 0.9},
                {"claim": "Fake-backed claim B", "source_id": "s2", "confidence": 0.5},
            ],
            citation_check_results=[
                {"citation_id": 1, "source_id": "s1", "is_valid": True},
                {"citation_id": 2, "source_id": "s2", "is_valid": False},
            ],
        )
        report = result["final_report"]
        # 有效证据被追加
        assert "Valid claim A" in report
        # 无效来源的证据不应该被追加
        assert "Fake-backed claim B" not in report

    # ================================================================
    # 结果结构
    # ================================================================

    def test_result_contains_unresolved_key(self):
        """返回结果包含 unresolved_issues 键。"""
        result = self.reviewer.review(draft_report="# Report")
        assert "unresolved_issues" in result
        assert isinstance(result["unresolved_issues"], list)

    def test_warnings_include_unresolved_count(self):
        """warnings 包含 unresolved 数量。"""
        result = self.reviewer.review(
            draft_report="# Report",
            eval_metrics={"metrics": {"min_sources": False}},
        )
        assert len(result["warnings"]) >= 1
