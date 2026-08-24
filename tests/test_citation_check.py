"""
tests/test_citation_check.py

CitationCheckTool 测试 —— Phase 1A。

覆盖：正常引用、fake citation（不存在的 ID）、URL 不匹配、
quote 不在 source 中、空引用列表、空来源列表。
"""

import pytest
from app.tools.citation_check_tool import CitationCheckTool


# 测试用 sources
_SOURCES = [
    {
        "source_id": "src_A",
        "url": "https://arxiv.org/abs/2309.15217",
        "title": "RAGAS Paper",
        "full_text": "RAGAS proposes faithfulness, answer relevancy, context recall, "
                     "and context precision for RAG evaluation.",
    },
    {
        "source_id": "src_B",
        "url": "https://docs.confident-ai.com/",
        "title": "DeepEval Docs",
        "full_text": "DeepEval provides groundedness, answer relevancy, contextual recall, "
                     "contextual precision, and hallucination detection metrics.",
    },
    {
        "source_id": "src_C",
        "url": "https://arxiv.org/abs/2405.12345",
        "title": "RAG Evaluation Survey",
        "full_text": "This paper surveys evaluation methods for RAG systems. "
                     "Retrieval quality is the dominant factor for faithful generation.",
    },
]


class TestCitationCheckTool:
    """测试 CitationCheckTool。"""

    def setup_method(self):
        self.tool = CitationCheckTool()

    # ---- 正常引用 ----

    @pytest.mark.asyncio
    async def test_valid_citation_passes_all_checks(self):
        """正确的引用通过所有校验。"""
        citations = [
            {
                "id": 1,
                "source_id": "src_A",
                "url": "https://arxiv.org/abs/2309.15217",
                "quote": "RAGAS proposes faithfulness, answer relevancy",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        assert result.success is True
        data = result.data
        assert data["total_checked"] == 1
        assert data["all_valid"] is True
        assert data["valid_count"] == 1

        check = data["results"][0]
        assert check["citation_id"] == 1
        assert check["id_exists"] is True
        assert check["url_matches_source"] is True
        assert check["quote_found_in_source"] is True
        assert check["is_valid"] is True
        assert check["issues"] == []

    @pytest.mark.asyncio
    async def test_multiple_valid_citations(self):
        """多条正确引用全部通过。"""
        citations = [
            {
                "id": 1,
                "source_id": "src_A",
                "url": "https://arxiv.org/abs/2309.15217",
                "quote": "RAGAS proposes faithfulness",
            },
            {
                "id": 2,
                "source_id": "src_B",
                "url": "https://docs.confident-ai.com/",
                "quote": "DeepEval provides groundedness",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        assert result.data["total_checked"] == 2
        assert result.data["all_valid"] is True
        assert result.data["valid_count"] == 2

    # ---- Fake citation: 不存在的 ID ----

    @pytest.mark.asyncio
    async def test_fake_id_detected(self):
        """检测到不存在的引用编号。"""
        citations = [
            {
                "id": 1,
                "source_id": "src_NONEXISTENT",
                "url": "https://fake-url.com/paper",
                "quote": "Some made up quote",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        data = result.data
        assert data["all_valid"] is False
        assert data["invalid_count"] == 1

        check = data["results"][0]
        assert check["id_exists"] is False
        assert check["url_matches_source"] is False
        assert check["quote_found_in_source"] is False
        assert check["is_valid"] is False
        assert len(check["issues"]) >= 1
        assert "not found in source list" in check["issues"][0].lower()

    # ---- Fake citation: URL 不匹配 ----

    @pytest.mark.asyncio
    async def test_url_mismatch_detected(self):
        """检测到 URL 不匹配。"""
        citations = [
            {
                "id": 1,
                "source_id": "src_A",
                "url": "https://wrong-url.com/paper",  # 不属于 src_A
                "quote": "RAGAS proposes faithfulness",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        check = result.data["results"][0]
        assert check["url_matches_source"] is False
        assert check["is_valid"] is False
        # 应该包含 URL 不匹配的问题
        url_issues = [i for i in check["issues"] if "url" in i.lower()]
        assert len(url_issues) >= 1

    # ---- Fake citation: quote 不在 source 中 ----

    @pytest.mark.asyncio
    async def test_quote_not_found_detected(self):
        """检测到 quote 不在 source full_text 中。"""
        citations = [
            {
                "id": 1,
                "source_id": "src_A",
                "url": "https://arxiv.org/abs/2309.15217",
                "quote": "This text does NOT appear anywhere in the original source material.",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        check = result.data["results"][0]
        assert check["id_exists"] is True  # ID 是存在的
        assert check["url_matches_source"] is True  # URL 正确
        assert check["quote_found_in_source"] is False  # 但 quote 不对
        assert check["is_valid"] is False
        assert len(check["issues"]) >= 1

    # ---- 边界情况 ----

    @pytest.mark.asyncio
    async def test_empty_citations_returns_empty(self):
        """空引用列表返回空结果。"""
        result = await self.tool.run(citations=[], sources=_SOURCES)

        assert result.success is True
        assert result.data["total_checked"] == 0
        assert result.data["all_valid"] is True

    @pytest.mark.asyncio
    async def test_no_sources_all_citations_invalid(self):
        """没有来源时所有引用都无效。"""
        citations = [
            {"id": 1, "source_id": "src_A", "url": "http://x.com", "quote": "test"},
        ]

        result = await self.tool.run(citations=citations, sources=[])

        assert result.data["all_valid"] is False
        assert result.data["invalid_count"] == 1
        check = result.data["results"][0]
        assert "no sources" in check["issues"][0].lower()

    @pytest.mark.asyncio
    async def test_fragment_quote_matching(self):
        """较长的 quote 通过片段匹配也能找到（宽松匹配）。"""
        citations = [
            {
                "id": 1,
                "source_id": "src_C",
                "url": "https://arxiv.org/abs/2405.12345",
                # 这段 quote 的部分内容在 full_text 中
                "quote": "This paper surveys evaluation methods for RAG systems. "
                         "Retrieval quality is the dominant factor for faithful generation.",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        check = result.data["results"][0]
        assert check["quote_found_in_source"] is True
        assert check["is_valid"] is True

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid(self):
        """混合有效和无效引用时统计正确。"""
        citations = [
            {  # 有效
                "id": 1,
                "source_id": "src_A",
                "url": "https://arxiv.org/abs/2309.15217",
                "quote": "RAGAS proposes faithfulness",
            },
            {  # 无效：不存在的 ID
                "id": 2,
                "source_id": "src_FAKE",
                "url": "https://fake-url.com/paper",
                "quote": "Made up content",
            },
            {  # 有效
                "id": 3,
                "source_id": "src_B",
                "url": "https://docs.confident-ai.com/",
                "quote": "DeepEval provides groundedness",
            },
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        assert result.data["total_checked"] == 3
        assert result.data["valid_count"] == 2
        assert result.data["invalid_count"] == 1
        assert result.data["all_valid"] is False

    @pytest.mark.asyncio
    async def test_all_results_have_required_fields(self):
        """每条检查结果包含所有必填字段。"""
        citations = [
            {"id": 1, "source_id": "src_A", "url": "http://x.com", "quote": "test"},
        ]

        result = await self.tool.run(citations=citations, sources=_SOURCES)

        for check in result.data["results"]:
            assert "citation_id" in check
            assert "source_id" in check
            assert "id_exists" in check
            assert isinstance(check["id_exists"], bool)
            assert "url_matches_source" in check
            assert isinstance(check["url_matches_source"], bool)
            assert "quote_found_in_source" in check
            assert isinstance(check["quote_found_in_source"], bool)
            assert "is_valid" in check
            assert isinstance(check["is_valid"], bool)
            assert "issues" in check
            assert isinstance(check["issues"], list)
