"""
app/tools/source_quality_scorer.py

SourceQualityScorer —— 来源质量评分工具。

对学术来源从四个维度评分并绑定 source_id。

来源质量评分的稳定性约束：
- Stage2 的 score_batch 排序后返回 List[SourceScoreResult]，
  调用方用 enumerate(sources) 按原数组下标贴回分数，
  这导致排序后的分数贴到了错误的 source 上。
- 本版本修复：score_batch 返回 {source_id: SourceScoreResult} 字典，
  调用方通过 source_id 精确匹配，不依赖数组下标。

在 Agent 调用链中的位置：
Search Worker -> PaperMetadataTool -> SourceQualityScorer -> EvidenceExtractTool
"""

import re
from typing import Any, Dict, List, Optional

from app.tools.base import BaseTool, ToolResult

# 来源类型基础分
SOURCE_TYPE_BASE_SCORE = {
    "paper": 1.0,
    # Local Zotero results are extracted from the user's academic PDFs. "local"
    # describes provenance, not a lower-authority document type.
    "local": 1.0,
    "book": 0.85,
    "benchmark": 0.9,
    "dataset": 0.85,
    "tool": 0.7,
    "blog": 0.5,
    "report": 0.6,
    "other": 0.4,
    "unknown": 0.4,
}

DEFAULT_WEIGHTS = {
    "title_match": 0.25,
    "year": 0.20,
    "source_type": 0.20,
    "snippet_relevance": 0.35,
}


class SourceScoreResult:
    """单条来源的评分结果。"""

    def __init__(
        self,
        source_id: str,
        total: float,
        title_match: float,
        year_score: float,
        source_type_score: float,
        snippet_relevance: float,
    ):
        self.source_id = source_id
        self.total = round(total, 4)
        self.title_match = round(title_match, 4)
        self.year_score = round(year_score, 4)
        self.source_type_score = round(source_type_score, 4)
        self.snippet_relevance = round(snippet_relevance, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "total": self.total,
            "title_match": self.title_match,
            "year_score": self.year_score,
            "source_type_score": self.source_type_score,
            "snippet_relevance": self.snippet_relevance,
        }

    def __repr__(self) -> str:
        return f"SourceScore({self.source_id}, total={self.total:.2f})"


class SourceQualityScorer(BaseTool):
    """
    来源质量评分工具。

    对每条学术来源从四个维度评分：
    1. 标题匹配度：标题与 topic 的关键词重合度
    2. 年份分：越新的来源分越高（当年 = 满分，10 年前 = 0 分）
    3. 来源类型分：paper > benchmark > dataset > tool > blog > unknown
    4. 摘要相关性：snippet 与 topic 的关键词重合度

    核心设计：评分结果通过 source_id 绑定，调用方通过 source_id 精确匹配分数，
    不依赖数组下标。这避免了排序后分数错位的风险。
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        current_year: int = 2026,
        year_decay_years: int = 10,
    ):
        super().__init__()
        self.weights = weights or DEFAULT_WEIGHTS
        self.current_year = current_year
        self.year_decay_years = year_decay_years

    @property
    def name(self) -> str:
        return "source_quality_scorer"

    @property
    def description(self) -> str:
        return (
            "Score academic sources on four dimensions: title match, year recency, "
            "source type authority, and snippet relevance. Scores are bound to "
            "source_id for precise matching."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "description": "List of paper sources to score. Each must have source_id, title, snippet, optional year and source_type.",
                    "items": {"type": "object"},
                },
                "topic": {
                    "type": "string",
                    "description": "Research topic for relevance matching",
                },
            },
            "required": ["sources", "topic"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        sources = kwargs.get("sources", [])
        topic = kwargs.get("topic", "").strip()

        if not sources:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="No sources provided for scoring",
            )

        # 逐条评分，返回 source_id -> score 字典
        scores_by_id: Dict[str, SourceScoreResult] = {}
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_id = source.get("source_id", "")
            if not source_id:
                continue

            score = self._score_one(source, topic)
            scores_by_id[source_id] = score

        # 构建有序结果列表（按总分降序）
        sorted_scores = sorted(
            scores_by_id.values(), key=lambda r: r.total, reverse=True
        )

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "scores_by_id": {sid: s.to_dict() for sid, s in scores_by_id.items()},
                "sorted_scores": [s.to_dict() for s in sorted_scores],
                "avg_score": (
                    round(sum(s.total for s in scores_by_id.values()) / len(scores_by_id), 4)
                    if scores_by_id else 0.0
                ),
                "scored_count": len(scores_by_id),
            },
        )

    def score_batch(
        self,
        sources: List[Dict[str, Any]],
        topic: str,
    ) -> Dict[str, SourceScoreResult]:
        """
        批量评分（同步方法，供 Agent 内部直接调用）。

        返回 {source_id: SourceScoreResult} 字典。
        调用方通过 source_id 精确匹配分数，不依赖数组下标。

        该方法避免依赖 sorted() 后的数组位置，
        本方法返回 source_id-keyed dict，完全消除了下标错位风险。
        """
        result: Dict[str, SourceScoreResult] = {}
        for source in sources:
            source_id = source.get("source_id", "")
            if not source_id:
                continue
            result[source_id] = self._score_one(source, topic)
        return result

    def _score_one(self, source: Dict[str, Any], topic: str) -> SourceScoreResult:
        """对单条来源评分。"""
        source_id = source.get("source_id", "unknown")
        title = source.get("title", "")
        snippet = source.get("snippet", "")
        year = source.get("year")
        source_type = source.get("source_type", "unknown")

        title_match = self._score_title_match(title, topic)
        year_score = self._score_year(year)
        source_type_score = SOURCE_TYPE_BASE_SCORE.get(
            source_type, SOURCE_TYPE_BASE_SCORE["unknown"]
        )
        snippet_rel = self._score_snippet_relevance(snippet, topic)

        total = (
            self.weights["title_match"] * title_match
            + self.weights["year"] * year_score
            + self.weights["source_type"] * source_type_score
            + self.weights["snippet_relevance"] * snippet_rel
        )

        return SourceScoreResult(
            source_id=source_id,
            total=total,
            title_match=title_match,
            year_score=year_score,
            source_type_score=source_type_score,
            snippet_relevance=snippet_rel,
        )

    # ---- 内部评分方法 ----

    def _score_title_match(self, title: str, topic: str) -> float:
        if not title or not topic:
            return 0.0
        topic_kw = self._extract_keywords(topic)
        title_kw = self._extract_keywords(title)
        if not topic_kw:
            return 0.5
        hits = sum(1 for kw in topic_kw if kw in title_kw)
        return min(1.0, hits / max(len(topic_kw), 1))

    def _score_year(self, year: Any) -> float:
        if year is None:
            return 0.3
        try:
            y = int(year)
        except (ValueError, TypeError):
            return 0.3
        if y > self.current_year:
            return 1.0
        age = self.current_year - y
        if age <= 0:
            return 1.0
        if age >= self.year_decay_years:
            return 0.0
        return 1.0 - (age / self.year_decay_years)

    def _score_snippet_relevance(self, snippet: str, topic: str) -> float:
        if not snippet or not topic:
            return 0.0
        topic_kw = self._extract_keywords(topic)
        snippet_lower = snippet.lower()
        if not topic_kw:
            return 0.3
        hits = sum(1 for kw in topic_kw if kw in snippet_lower)
        return min(1.0, hits / max(len(topic_kw), 1))

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        stopwords = {
            "the", "and", "for", "are", "was", "has", "its", "not",
            "this", "that", "with", "from", "have", "been", "can",
            "will", "they", "their", "about", "which", "when", "what",
            "how", "into", "more", "some", "such", "than", "then",
            "also", "very", "just", "over", "our", "after", "before",
        }
        return [w for w in words if w not in stopwords][:20]
