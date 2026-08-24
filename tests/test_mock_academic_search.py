"""
tests/test_mock_academic_search.py

MockAcademicSearchTool 测试 —— Phase 1A。
"""

import pytest
from app.tools.mock_academic_search_tool import MockAcademicSearchTool


class TestMockAcademicSearchTool:
    """测试 MockAcademicSearchTool 的搜索能力。"""

    def setup_method(self):
        self.tool = MockAcademicSearchTool()

    @pytest.mark.asyncio
    async def test_tool_has_correct_metadata(self):
        """工具元数据正确。"""
        assert self.tool.name == "mock_academic_search"
        assert "academic" in self.tool.description.lower()
        assert "query" in self.tool.input_schema["required"]

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """正常搜索返回结果。"""
        result = await self.tool.run(query="RAG evaluation", max_results=3)

        assert result.success is True
        data = result.data
        assert "results" in data
        assert len(data["results"]) >= 1
        assert len(data["results"]) <= 3

    @pytest.mark.asyncio
    async def test_results_have_required_fields(self):
        """每条结果包含必填字段。"""
        result = await self.tool.run(query="RAG evaluation", max_results=3)

        for source in result.data["results"]:
            assert "source_id" in source, f"Missing source_id in {source.get('title', '?')}"
            assert len(source["source_id"]) == 8, f"source_id should be 8 chars, got {source['source_id']}"
            assert "title" in source
            assert "url" in source
            assert "snippet" in source
            assert "full_text" in source
            assert len(source["full_text"]) > 100, f"full_text too short for {source['title']}"
            assert "authors" in source
            assert isinstance(source["authors"], list)
            assert "year" in source
            assert "venue" in source
            assert "source_type" in source

    @pytest.mark.asyncio
    async def test_results_have_scholarly_fields(self):
        """学术字段存在且合理。"""
        result = await self.tool.run(query="RAG evaluation", max_results=3)

        source_types = set()
        for source in result.data["results"]:
            source_types.add(source["source_type"])
            assert source["source_type"] in ("paper", "benchmark", "tool", "blog", "report", "unknown")
            assert isinstance(source["year"], int) or source["year"] is None
            if source["authors"]:
                assert all(isinstance(a, str) for a in source["authors"])

        # 至少包含一种学术类型
        assert len(source_types) >= 1

    @pytest.mark.asyncio
    async def test_source_ids_are_unique(self):
        """每个 source_id 唯一（不重复）。"""
        result = await self.tool.run(query="RAG evaluation", max_results=5)

        ids = [s["source_id"] for s in result.data["results"]]
        assert len(ids) == len(set(ids)), f"Duplicate source_ids found: {ids}"

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        """空查询返回错误。"""
        result = await self.tool.run(query="", max_results=3)

        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_keyword_filtering_works(self):
        """关键词过滤返回相关结果。"""
        result = await self.tool.run(query="faithfulness factuality", max_results=5)

        results = result.data["results"]
        assert len(results) >= 1
        for r in results:
            text = (r["title"] + " " + r["snippet"]).lower()
            assert "faithfulness" in text or "factuality" in text

    @pytest.mark.asyncio
    async def test_no_match_falls_back_to_all(self):
        """无匹配关键词时兜底返回全部数据。"""
        result = await self.tool.run(query="zzznotexistzzz", max_results=7)

        assert result.success is True
        assert len(result.data["results"]) >= 1, "Should fall back to all results"

    @pytest.mark.asyncio
    async def test_max_results_respected(self):
        """max_results 上限被遵守。"""
        result = await self.tool.run(query="RAG", max_results=2)

        assert len(result.data["results"]) <= 2

    @pytest.mark.asyncio
    async def test_full_text_contains_abstract(self):
        """full_text 包含 Abstract 内容（供 EvidenceExtractTool 使用）。"""
        result = await self.tool.run(query="RAG", max_results=1)

        source = result.data["results"][0]
        assert "abstract" in source["full_text"].lower() or \
               "introduction" in source["full_text"].lower() or \
               "method" in source["full_text"].lower(), \
               f"full_text should contain structured content, got: {source['full_text'][:100]}"
