"""
tests/test_openalex_search.py

Phase 4B: OpenAlex Academic Search 完整测试。

所有测试默认零网络（httpx.MockTransport）。
包含：200 映射、空结果、缺字段、abstract 重建、source_id 稳定、
DOI/URL 选择、去重、400/401/403/429/5xx 重试、timeout、fallback、
API key 脱敏、TEI 解析等。
"""

import json
import os
import time
from unittest import mock

import httpx
import pytest

from app.tools.academic_search_provider import (
    MockAcademicSearchProvider,
    OpenAlexSearchProvider,
    _reconstruct_abstract,
    _select_canonical_url,
    _select_source_type,
    _TYPE_MAPPING,
    openalex_work_to_paper_source,
)
from app.tools.academic_search_tool import AcademicSearchTool
from app.tools.base import ToolResult

# 保存原始 httpx.AsyncClient 引用（在 mock 内部使用，避免递归）
_RealAsyncClient = httpx.AsyncClient


def _make_mock_client(transport, *a, **kw):
    """创建带 MockTransport 的真实 AsyncClient（不触发 mock 递归）。"""
    return _RealAsyncClient(transport=transport, *a, **kw)


# ================================================================
# OpenAlex mock JSON fixtures
# ================================================================

FULL_WORK = {
    "id": "https://openalex.org/works/W2741809807",
    "doi": "https://doi.org/10.48550/arxiv.2309.15217",
    "display_name": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
    "publication_year": 2023,
    "type": "article",
    "cited_by_count": 245,
    "authorships": [
        {
            "author": {"display_name": "Shahul Es", "id": "https://openalex.org/A5100000001"},
        },
        {
            "author": {"display_name": "Jithin James", "id": "https://openalex.org/A5100000002"},
        },
    ],
    "primary_location": {
        "source": {"display_name": "arXiv preprint (arXiv:2309.15217)", "id": "..."},
        "landing_page_url": "https://arxiv.org/abs/2309.15217",
        "pdf_url": "https://arxiv.org/pdf/2309.15217.pdf",
    },
    "abstract_inverted_index": {
        "We": [0], "introduce": [1], "RAGAS": [2], "a": [3], "framework": [4],
        "for": [5], "automated": [6], "evaluation": [7], "of": [8], "RAG": [9],
        "systems": [10],
    },
    "open_access": {
        "is_oa": True,
        "oa_status": "gold",
    },
    "best_oa_location": {
        "landing_page_url": "https://arxiv.org/pdf/2309.15217.pdf",
    },
    "content_url": "https://content.openalex.org/works/W2741809807.grobid-xml",
    "has_content": {"pdf": True, "grobid_xml": True},
}

MINIMAL_WORK = {
    "id": "https://openalex.org/works/W12345678",
    "display_name": "Minimal Paper",
    # No authorships, no publication_year, no abstract, minimal primary_location
    "primary_location": None,
    "type": None,
}

NO_ABSTRACT_WORK = {
    "id": "https://openalex.org/works/W11111111",
    "doi": "https://doi.org/10.1000/test.111",
    "display_name": "No Abstract Paper",
    "publication_year": 2024,
    "type": "preprint",
    "authorships": [],
    "primary_location": {"source": {"display_name": "Test Journal"}, "landing_page_url": ""},
    "abstract_inverted_index": None,
    "open_access": {"is_oa": False, "oa_status": None},
    "cited_by_count": 0,
}


def _make_search_response(works, meta_count=None):
    """Helper to build an OpenAlex /works response."""
    return httpx.Response(
        200,
        json={
            "meta": {
                "count": meta_count if meta_count is not None else len(works),
                "db_response_time_ms": 42,
                "page": 1,
                "per_page": 25,
            },
            "results": works,
        },
    )


def _make_error_response(status_code, body_text="error"):
    return httpx.Response(status_code, json={"error": body_text})


# ================================================================
# Unit tests: Adapter functions
# ================================================================


class TestAbstractReconstruction:
    """Abstract 从 inverted_index 重建。"""

    def test_reconstruct_normal(self):
        abstract = _reconstruct_abstract(FULL_WORK["abstract_inverted_index"])
        expected = "We introduce RAGAS a framework for automated evaluation of RAG systems"
        assert abstract == expected

    def test_reconstruct_none(self):
        assert _reconstruct_abstract(None) == ""

    def test_reconstruct_empty_dict(self):
        assert _reconstruct_abstract({}) == ""

    def test_reconstruct_no_int_positions(self):
        inv = {"word": ["not_an_int"]}
        assert _reconstruct_abstract(inv) == ""


class TestSourceTypeMapping:
    """OpenAlex type → 项目 source_type 映射。"""

    def test_article_to_paper(self):
        assert _select_source_type("article") == "paper"

    def test_review_to_paper(self):
        assert _select_source_type("review") == "paper"

    def test_preprint_to_paper(self):
        assert _select_source_type("preprint") == "paper"

    def test_book_chapter(self):
        assert _select_source_type("book-chapter") == "book"

    def test_book(self):
        assert _select_source_type("book") == "book"

    def test_dissertation(self):
        assert _select_source_type("dissertation") == "paper"

    def test_unknown_type(self):
        assert _select_source_type("unknown-type") == "other"

    def test_none_type(self):
        assert _select_source_type(None) == "unknown"


class TestCanonicalURL:
    """Canonical URL 优先级测试。"""

    def test_doi_first(self):
        url = _select_canonical_url(FULL_WORK)
        assert url == "https://doi.org/10.48550/arxiv.2309.15217"

    def test_landing_page_when_no_doi(self):
        work = {
            "doi": "",
            "primary_location": {"landing_page_url": "https://example.com/paper"},
        }
        assert _select_canonical_url(work) == "https://example.com/paper"

    def test_openalex_id_fallback(self):
        work = {
            "doi": None,
            "id": "https://openalex.org/works/W999",
            "primary_location": None,
        }
        assert _select_canonical_url(work) == "https://openalex.org/works/W999"

    def test_empty_all(self):
        work = {"doi": "", "primary_location": None, "id": ""}
        assert _select_canonical_url(work) == ""


# ================================================================
# Unit tests: openalex_work_to_paper_source
# ================================================================


class TestWorkToPaperSource:
    """OpenAlex Work → PaperSource 映射。"""

    def test_full_work_mapping(self):
        source = openalex_work_to_paper_source(FULL_WORK)
        assert source["source_id"] == "W2741809807"
        assert source["title"] == "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
        assert source["authors"] == ["Shahul Es", "Jithin James"]
        assert source["year"] == 2023
        assert source["venue"] == "arXiv preprint (arXiv:2309.15217)"
        assert source["source_type"] == "paper"
        assert source["provider"] == "openalex"
        assert source["openalex_id"] == "W2741809807"
        assert source["doi"] == "https://doi.org/10.48550/arxiv.2309.15217"
        assert source["cited_by_count"] == 245
        assert source["is_oa"] is True
        assert source["oa_status"] == "gold"
        assert source["content_url"] == "https://content.openalex.org/works/W2741809807.grobid-xml"
        assert source["url"] == "https://doi.org/10.48550/arxiv.2309.15217"
        # abstract 应该被重建
        assert "introduce" in source["snippet"].lower()
        assert "RAGAS" in source["full_text"]

    def test_minimal_work_safe_degradation(self):
        """缺失字段不 KeyError。"""
        source = openalex_work_to_paper_source(MINIMAL_WORK)
        assert source["source_id"] == "W12345678"
        assert source["title"] == "Minimal Paper"
        assert source["authors"] == []
        assert source["year"] is None
        assert source["venue"] == ""
        assert source["source_type"] == "unknown"
        assert source["full_text"] == ""

    def test_no_abstract_work(self):
        """无 abstract 的 Work 正确处理。"""
        source = openalex_work_to_paper_source(NO_ABSTRACT_WORK)
        assert source["full_text"] == ""
        assert source["snippet"] == "No Abstract Paper"  # fallback to title

    def test_dedup_keys_present(self):
        """source_id 和 doi 可用于去重。"""
        source = openalex_work_to_paper_source(FULL_WORK)
        assert source["source_id"]  # non-empty for dedup
        assert source["doi"]  # non-empty for dedup


# ================================================================
# MockAcademicSearchProvider tests
# ================================================================


class TestMockAcademicSearchProvider:
    """Mock Provider 离线搜索。"""

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        provider = MockAcademicSearchProvider()
        result = await provider.search("RAG evaluation", 3)
        assert result.success
        assert len(result.data["results"]) >= 1
        assert result.data["results"][0]["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        provider = MockAcademicSearchProvider()
        result = await provider.search("", 3)
        assert not result.success

    @pytest.mark.asyncio
    async def test_source_id_format(self):
        provider = MockAcademicSearchProvider()
        result = await provider.search("RAG", 3)
        for s in result.data["results"]:
            assert len(s["source_id"]) == 8  # UUID 前 8 位


# ================================================================
# OpenAlexSearchProvider tests with MockTransport
# ================================================================


@pytest.fixture
def openalex_provider(monkeypatch):
    """创建带 mock API key 的 OpenAlex Provider。"""
    monkeypatch.setenv("OPENALEX_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("OPENALEX_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("OPENALEX_MAX_RETRIES", "1")
    monkeypatch.setenv("OPENALEX_FALLBACK_TO_MOCK", "false")
    monkeypatch.setenv("OPENALEX_CONTENT_MODE", "abstract")
    return OpenAlexSearchProvider()


class TestOpenAlexSearchSuccess:
    """OpenAlex 搜索成功场景。"""

    @pytest.mark.asyncio
    async def test_full_work_mapping(self, openalex_provider):
        """200 OK → 正确的 PaperSource。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: _make_search_response([FULL_WORK])
            )
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("RAG evaluation", 5)

        assert result.success
        assert result.tool_name == "academic_search"
        data = result.data
        assert len(data["results"]) == 1
        source = data["results"][0]
        assert source["source_id"] == "W2741809807"
        assert source["title"] == FULL_WORK["display_name"]
        assert source["provider"] == "openalex"

    @pytest.mark.asyncio
    async def test_search_requires_abstract_for_evidence_pipeline(self, openalex_provider):
        captured = {}

        def handler(request):
            captured.update(dict(request.url.params))
            return _make_search_response([FULL_WORK])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("RAG evaluation", 2)

        assert result.success
        assert captured["filter"] == "has_abstract:true"

    @pytest.mark.asyncio
    async def test_empty_results(self, openalex_provider):
        """空结果。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: _make_search_response([], meta_count=0)
            )
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("xyznonexistent", 5)

        assert result.success
        assert result.data["total_found"] == 0
        assert result.data["results"] == []

    @pytest.mark.asyncio
    async def test_minimal_work_safe(self, openalex_provider):
        """缺字段 Work 安全降级。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: _make_search_response([MINIMAL_WORK])
            )
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("test", 1)

        assert result.success
        source = result.data["results"][0]
        assert source["source_id"] == "W12345678"
        assert source["title"] == "Minimal Paper"
        assert source["authors"] == []
        assert source["year"] is None
        # 不应 KeyError

    @pytest.mark.asyncio
    async def test_dedup_by_openalex_id(self, openalex_provider):
        """相同 OpenAlex ID 去重。"""
        dup_work = dict(FULL_WORK)  # same ID
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: _make_search_response([FULL_WORK, dup_work])
            )
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("RAG", 5)

        assert result.success
        assert len(result.data["results"]) == 1  # deduped


class TestOpenAlexSearchErrors:
    """OpenAlex 错误状态码处理。"""

    @pytest.mark.asyncio
    async def test_400_no_retry(self, openalex_provider, monkeypatch):
        """400 不重试，直接返回失败。"""
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "2")
        provider = OpenAlexSearchProvider()

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: _make_error_response(400))
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await provider.search("test", 5)

        assert not result.success
        assert "400" in result.error

    @pytest.mark.asyncio
    async def test_401_no_retry(self, openalex_provider):
        """401 不重试。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: _make_error_response(401))
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("test", 5)

        assert not result.success
        assert "401" in result.error

    @pytest.mark.asyncio
    async def test_403_no_retry(self, openalex_provider):
        """403 不重试。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: _make_error_response(403))
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("test", 5)

        assert not result.success
        assert "403" in result.error

    @pytest.mark.asyncio
    async def test_429_with_retry_after(self, openalex_provider, monkeypatch):
        """429 尊重 Retry-After header，有界重试后失败。"""
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "1")
        provider = OpenAlexSearchProvider()

        call_count = [0]

        def handler(req):
            call_count[0] += 1
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "rate limited"})

        transport = httpx.MockTransport(handler)

        with mock.patch.object(httpx, "AsyncClient", side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw)):
            with mock.patch("asyncio.sleep", return_value=None) as mock_sleep:
                result = await provider.search("test", 5)

        assert not result.success
        assert "429" in result.error
        assert call_count[0] == 2  # 1 initial + 1 retry
        mock_sleep.assert_called_once_with(1)  # Retry-After=1

    @pytest.mark.asyncio
    async def test_5xx_retry_exponential_backoff(self, openalex_provider, monkeypatch):
        """5xx 指数退避重试。"""
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "2")
        provider = OpenAlexSearchProvider()

        call_count = [0]

        def handler(req):
            call_count[0] += 1
            return httpx.Response(503, json={"error": "unavailable"})

        transport = httpx.MockTransport(handler)

        with mock.patch.object(httpx, "AsyncClient", side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw)):
            with mock.patch("asyncio.sleep", return_value=None) as mock_sleep:
                result = await provider.search("test", 5)

        assert not result.success
        assert "503" in result.error
        assert call_count[0] == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_timeout_retry(self, openalex_provider, monkeypatch):
        """Timeout 后重试。"""
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "1")
        provider = OpenAlexSearchProvider()

        call_count = [0]

        def handler(req):
            call_count[0] += 1
            raise httpx.TimeoutException("timed out")

        transport = httpx.MockTransport(handler)

        with mock.patch.object(httpx, "AsyncClient", side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw)):
            with mock.patch("asyncio.sleep", return_value=None):
                result = await provider.search("test", 5)

        assert not result.success
        assert "timeout" in result.error.lower()
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, openalex_provider, monkeypatch):
        """重试上限达到后返回 ToolResult(success=False)，不抛异常。"""
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "2")
        provider = OpenAlexSearchProvider()

        def handler(req):
            raise httpx.TimeoutException("timed out")

        transport = httpx.MockTransport(handler)

        with mock.patch.object(httpx, "AsyncClient", side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw)):
            with mock.patch("asyncio.sleep", return_value=None):
                result = await provider.search("test", 5)

        assert not result.success
        # 不抛异常，安全返回

    @pytest.mark.asyncio
    async def test_fallback_to_mock_enabled(self, monkeypatch):
        """OPENALEX_FALLBACK_TO_MOCK=true → 重试耗尽后回退 Mock。"""
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "1")
        monkeypatch.setenv("OPENALEX_FALLBACK_TO_MOCK", "true")
        provider = OpenAlexSearchProvider()

        def handler(req):
            raise httpx.TimeoutException("timed out")

        transport = httpx.MockTransport(handler)

        with mock.patch.object(httpx, "AsyncClient", side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw)):
            with mock.patch("asyncio.sleep", return_value=None):
                result = await provider.search("RAG evaluation", 3)

        assert result.success
        assert result.data.get("fallback_used") is True
        assert result.data["provider"] == "mock"


class TestAPIKeySafety:
    """API key 不泄漏到错误/Trace/日志中。"""

    @pytest.mark.asyncio
    async def test_api_key_not_in_error(self, openalex_provider):
        """错误信息不含 API key。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: _make_error_response(400, "bad request"))
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("test", 5)

        assert "test-key-123" not in result.error
        assert "test-key-123" not in json.dumps(result.to_dict())

    @pytest.mark.asyncio
    async def test_api_key_not_in_metadata(self, openalex_provider):
        """metadata 不含 API key。"""
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: _make_search_response([FULL_WORK]))
        )

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await openalex_provider.search("RAG", 1)

        assert "test-key-123" not in json.dumps(result.metadata)
        assert "test-key-123" not in json.dumps(result.data or {})


# ================================================================
# AcademicSearchTool integration tests
# ================================================================


class TestAcademicSearchTool:
    """AcademicSearchTool provider 选择。"""

    @pytest.mark.asyncio
    async def test_mock_provider_selected(self, monkeypatch):
        """SEARCH_PROVIDER=mock → Mock Provider。"""
        monkeypatch.setenv("SEARCH_PROVIDER", "mock")
        tool = AcademicSearchTool()
        assert tool.name == "academic_search"
        result = await tool.run(query="RAG evaluation", max_results=3)
        assert result.success
        assert result.tool_name == "academic_search"
        assert result.data["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_academic_provider_selected(self, monkeypatch):
        """SEARCH_PROVIDER=academic → Mock Provider（向后兼容）。"""
        monkeypatch.setenv("SEARCH_PROVIDER", "academic")
        tool = AcademicSearchTool()
        result = await tool.run(query="RAG evaluation", max_results=3)
        assert result.success
        assert result.data["provider"] == "mock"

    @pytest.mark.asyncio
    async def test_max_results_capped(self, monkeypatch):
        """max_results 限制在 1-50。"""
        monkeypatch.setenv("SEARCH_PROVIDER", "mock")
        tool = AcademicSearchTool()
        result = await tool.run(query="RAG", max_results=100)
        assert result.success
        assert len(result.data["results"]) <= 7  # mock 只有 7 篇

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "mock")
        tool = AcademicSearchTool()
        result = await tool.run(query="")
        assert not result.success

    @pytest.mark.asyncio
    async def test_unknown_provider_fails_closed(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "open_alxe")
        result = await AcademicSearchTool().run(query="RAG")
        assert not result.success
        assert "Unsupported SEARCH_PROVIDER" in result.error

    @pytest.mark.asyncio
    async def test_input_schema_has_max_results_bounds(self):
        """input_schema 限制 max_results 1-50。"""
        tool = AcademicSearchTool()
        schema = tool.input_schema
        assert schema["properties"]["max_results"]["minimum"] == 1
        assert schema["properties"]["max_results"]["maximum"] == 50
        assert "query" in schema["required"]


# ================================================================
# TEI Content API tests
# ================================================================


SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Test Paper</title></titleStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <p>This is paragraph one of the body text.</p>
        <p>This is paragraph two with more content.</p>
      </div>
    </body>
  </text>
</TEI>"""


class TestTEIContentFetch:
    """TEI 全文获取与回退。"""

    def test_parse_tei_xml_success(self):
        """成功解析 TEI XML。"""
        text = OpenAlexSearchProvider._parse_tei_xml(SAMPLE_TEI_XML)
        assert text is not None
        assert "paragraph one" in text
        assert "paragraph two" in text

    def test_parse_tei_xml_corrupt(self):
        """损坏 XML 返回 None。"""
        text = OpenAlexSearchProvider._parse_tei_xml("<not><valid>xml")
        assert text is None

    def test_parse_tei_xml_empty(self):
        """空 XML 返回 None。"""
        text = OpenAlexSearchProvider._parse_tei_xml("<TEI xmlns='http://www.tei-c.org/ns/1.0'></TEI>")
        assert text is None

    def test_parse_tei_xml_no_body(self):
        """无 body 返回 None。"""
        xml = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/></TEI>'
        text = OpenAlexSearchProvider._parse_tei_xml(xml)
        assert text is None

    @pytest.mark.asyncio
    async def test_content_mode_abstract_skips_content_api(self, openalex_provider):
        """OPENALEX_CONTENT_MODE=abstract 时不调用 Content API。"""
        sources = [{"openalex_id": "W123", "source_id": "W123", "full_text": "original"}]
        result = await openalex_provider.enrich_full_text(sources)
        assert result[0]["full_text"] == "original"  # unchanged

    @pytest.mark.asyncio
    async def test_content_mode_tei_success(self, monkeypatch):
        """TEI 模式成功获取全文。"""
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_CONTENT_MODE", "tei")
        monkeypatch.setenv("OPENALEX_MAX_CONTENT_FETCHES", "1")
        provider = OpenAlexSearchProvider()

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, content=SAMPLE_TEI_XML.encode())
            )
        )

        sources = [{
            "openalex_id": "W123", "source_id": "W123", "full_text": "",
            "content_url": "https://content.openalex.org/works/W123.grobid-xml",
        }]

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await provider.enrich_full_text(sources)

        assert "paragraph one" in result[0]["full_text"]
        assert result[0]["content_source"] == "openalex_tei"

    @pytest.mark.asyncio
    async def test_search_tei_mode_enriches_results(self, monkeypatch):
        """TEI mode is wired into the public search() path."""
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_CONTENT_MODE", "tei")
        monkeypatch.setenv("OPENALEX_MAX_CONTENT_FETCHES", "1")
        provider = OpenAlexSearchProvider()

        def handler(request):
            if request.url.host == "api.openalex.org":
                return _make_search_response([FULL_WORK])
            return httpx.Response(200, content=SAMPLE_TEI_XML.encode())

        transport = httpx.MockTransport(handler)
        with mock.patch.object(
            httpx, "AsyncClient",
            side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw),
        ):
            result = await provider.search("RAG evaluation", 1)

        assert result.success
        source = result.data["results"][0]
        assert "paragraph one" in source["full_text"]
        assert source["content_source"] == "openalex_tei"

    @pytest.mark.asyncio
    async def test_tei_skips_work_without_content_signal(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_CONTENT_MODE", "tei")
        provider = OpenAlexSearchProvider()
        sources = [{"openalex_id": "W123", "source_id": "W123", "full_text": "abstract"}]

        with mock.patch.object(httpx, "AsyncClient") as client:
            result = await provider.enrich_full_text(sources)

        assert result[0]["full_text"] == "abstract"
        client.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_mode_tei_404_fallback(self, monkeypatch):
        """TEI 404 → 保持 abstract。"""
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_CONTENT_MODE", "tei")
        monkeypatch.setenv("OPENALEX_MAX_CONTENT_FETCHES", "1")
        provider = OpenAlexSearchProvider()

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(404))
        )

        sources = [{"openalex_id": "W123", "source_id": "W123", "full_text": "abstract text"}]

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await provider.enrich_full_text(sources)

        assert result[0]["full_text"] == "abstract text"  # unchanged
        assert result[0].get("content_source") != "openalex_tei"

    @pytest.mark.asyncio
    async def test_content_mode_tei_429_fallback(self, monkeypatch):
        """TEI 429 → 保持 abstract。"""
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_CONTENT_MODE", "tei")
        monkeypatch.setenv("OPENALEX_MAX_CONTENT_FETCHES", "1")
        provider = OpenAlexSearchProvider()

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(429))
        )

        sources = [{"openalex_id": "W123", "source_id": "W123", "full_text": "abstract text"}]

        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await provider.enrich_full_text(sources)

        assert result[0]["full_text"] == "abstract text"


# ================================================================
# Live smoke test (requires OPENALEX_API_KEY to run)
# ================================================================


@pytest.mark.openalex_live
class TestOpenAlexLive:
    """可选的真实 OpenAlex 冒烟测试。"""

    @pytest.mark.asyncio
    async def test_live_search_small_result(self, monkeypatch):
        """真实搜索：请求少量结果，验证格式。"""
        if os.getenv("RUN_OPENALEX_LIVE", "false").lower() != "true":
            pytest.skip("Set RUN_OPENALEX_LIVE=true to enable network tests")
        api_key = os.getenv("OPENALEX_API_KEY", "")
        if not api_key:
            pytest.skip("OPENALEX_API_KEY not set")

        monkeypatch.setenv("SEARCH_PROVIDER", "openalex")
        provider = OpenAlexSearchProvider()
        result = await provider.search("machine learning", max_results=2)

        assert result.success
        assert result.tool_name == "academic_search"
        data = result.data
        assert "results" in data
        assert len(data["results"]) <= 2

        for source in data["results"]:
            assert "source_id" in source
            assert source["source_id"].startswith("W")
            assert "title" in source
            assert "provider" in source
            assert source["provider"] == "openalex"

        # API key 不应出现在结果中
        result_json = json.dumps(result.to_dict())
        assert api_key not in result_json
