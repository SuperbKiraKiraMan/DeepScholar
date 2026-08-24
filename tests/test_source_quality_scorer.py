"""
tests/test_source_quality_scorer.py

SourceQualityScorer 测试 —— Phase 1A。

重点验证 source_id 绑定正确性（Phase 1A 关键修复）。
"""

import pytest
from app.tools.source_quality_scorer import SourceQualityScorer, SourceScoreResult


class TestSourceQualityScorer:
    """测试 SourceQualityScorer。"""

    def setup_method(self):
        self.scorer = SourceQualityScorer()

    def _make_source(self, source_id="src_001", **overrides):
        """辅助方法：创建测试用来源。"""
        base = {
            "source_id": source_id,
            "title": "RAG Evaluation Survey",
            "url": "https://arxiv.org/abs/2405.12345",
            "snippet": "A comprehensive survey of RAG evaluation metrics including faithfulness and relevance.",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "venue": "ACL 2024",
            "source_type": "paper",
        }
        base.update(overrides)
        return base

    # ---- 评分逻辑 ----

    def test_paper_scores_higher_than_blog(self):
        """paper 评分高于 blog。"""
        paper = self._make_source("p1", source_type="paper")
        blog = self._make_source("b1", source_type="blog", title="Some Blog Post")

        paper_score = self.scorer._score_one(paper, "RAG evaluation").total
        blog_score = self.scorer._score_one(blog, "RAG evaluation").total

        assert paper_score > blog_score

    def test_local_zotero_paper_keeps_academic_authority_score(self):
        """local 表示来源位置，不应把 Zotero 学术 PDF 降为 unknown。"""
        local = self._make_source(
            "local-1",
            source_type="local",
            provider="local_zotero",
        )

        result = self.scorer._score_one(local, "RAG evaluation")

        assert result.source_type_score == 1.0

    def test_recent_year_scores_higher(self):
        """越新越高。"""
        new = self._make_source("n1", year=2024)
        old = self._make_source("o1", year=2018)

        new_score = self.scorer._score_one(new, "RAG evaluation").total
        old_score = self.scorer._score_one(old, "RAG evaluation").total

        assert new_score > old_score

    def test_title_match_boosts_score(self):
        """标题匹配提高评分。"""
        matched = self._make_source("m1", title="RAG Evaluation: A Comprehensive Survey")
        unmatched = self._make_source("u1", title="Machine Learning Basics")

        matched_score = self.scorer._score_one(matched, "RAG evaluation").total
        unmatched_score = self.scorer._score_one(unmatched, "RAG evaluation").total

        assert matched_score > unmatched_score

    def test_score_in_range(self):
        """评分在 [0, 1] 范围内。"""
        source = self._make_source("s1")
        result = self.scorer._score_one(source, "RAG evaluation")
        assert 0.0 <= result.total <= 1.0

    def test_missing_year_gets_low_score(self):
        """缺少年份时年份分低。"""
        source = self._make_source("s1")
        del source["year"]
        result = self.scorer._score_one(source, "RAG evaluation")
        assert result.year_score <= 0.3

    def test_empty_title_snippet(self):
        """空标题和摘要不崩溃。"""
        source = {
            "source_id": "s1",
            "title": "",
            "url": "http://x.com",
            "snippet": "",
            "source_type": "unknown",
        }
        result = self.scorer._score_one(source, "RAG evaluation")
        assert 0.0 <= result.total <= 1.0

    # ---- source_id 绑定正确性（Phase 1A 关键修复） ----

    def test_score_batch_returns_dict_keyed_by_source_id(self):
        """score_batch 返回 {source_id: ScoreResult} 字典。"""
        sources = [
            self._make_source("id_a", title="RAG Paper A", year=2024),
            self._make_source("id_b", title="RAG Paper B", year=2023),
            self._make_source("id_c", title="Unrelated", year=2018, source_type="blog"),
        ]

        scores = self.scorer.score_batch(sources, "RAG evaluation")

        assert isinstance(scores, dict)
        assert set(scores.keys()) == {"id_a", "id_b", "id_c"}

    def test_score_batch_each_score_has_correct_source_id(self):
        """每个评分结果内的 source_id 与输入一致。"""
        sources = [
            self._make_source("alpha", title="RAG Stuff", year=2024),
            self._make_source("beta", title="RAG Metrics", year=2024),
        ]

        scores = self.scorer.score_batch(sources, "RAG evaluation")

        assert scores["alpha"].source_id == "alpha"
        assert scores["beta"].source_id == "beta"

    def test_score_batch_no_index_misalignment(self):
        """
        source_id 绑定不受排序影响。

        这是 Phase 1A 的关键测试：即使评分有高低，
        source_id 绑定也必须精确，不依赖数组下标。
        """
        sources = [
            self._make_source("low", title="Unrelated", snippet="Nothing about RAG",
                              year=2018, source_type="blog"),
            self._make_source("high", title="RAG Evaluation Deep Dive",
                              snippet="Comprehensive RAG evaluation framework", year=2024),
        ]

        scores = self.scorer.score_batch(sources, "RAG evaluation")

        # high 评分应该 > low 评分（内容相关）
        assert scores["high"].total > scores["low"].total, (
            f"Expected high ({scores['high'].total:.3f}) > low ({scores['low'].total:.3f})"
        )
        # 但每个 source_id 绑定到正确的结果
        assert scores["low"].source_id == "low"
        assert scores["high"].source_id == "high"

    def test_score_by_id_works_with_missing_ids(self):
        """缺失 source_id 的来源被跳过。"""
        sources = [
            self._make_source("id_1"),
            {"title": "No ID", "url": "http://x.com"},  # 无 source_id
            self._make_source("id_3"),
        ]

        scores = self.scorer.score_batch(sources, "RAG evaluation")

        assert "id_1" in scores
        assert "id_3" in scores
        assert len(scores) == 2  # 跳过无 source_id 的

    # ---- Tool 接口 ----

    @pytest.mark.asyncio
    async def test_tool_arun_returns_scores(self):
        """通过 BaseTool.run() 接口也能正确评分。"""
        sources = [
            self._make_source("t1", title="RAG Paper", year=2024),
            self._make_source("t2", title="Unrelated", year=2018, source_type="blog"),
        ]

        result = await self.scorer.run(sources=sources, topic="RAG evaluation")

        assert result.success is True
        data = result.data
        assert "scores_by_id" in data
        assert "t1" in data["scores_by_id"]
        assert "t2" in data["scores_by_id"]
        assert "sorted_scores" in data
        assert data["scored_count"] == 2
        # sorted 按总分降序
        sorted_scores = data["sorted_scores"]
        assert sorted_scores[0]["total"] >= sorted_scores[1]["total"]

    @pytest.mark.asyncio
    async def test_tool_arun_empty_sources(self):
        """空来源列表返回错误。"""
        result = await self.scorer.run(sources=[], topic="test")
        assert result.success is False

    def test_score_result_to_dict(self):
        """SourceScoreResult.to_dict 返回完整字段。"""
        r = SourceScoreResult(
            source_id="test_id",
            total=0.85,
            title_match=0.9,
            year_score=0.8,
            source_type_score=1.0,
            snippet_relevance=0.7,
        )
        d = r.to_dict()
        assert d["source_id"] == "test_id"
        assert d["total"] == 0.85
        assert d["title_match"] == 0.9
        assert d["year_score"] == 0.8
        assert d["source_type_score"] == 1.0
        assert d["snippet_relevance"] == 0.7
