"""论文详情检索测试：LocalPaperCatalog + /api/papers/detail 端点 + OpenAlex 标题兜底。

自包含：本地定义确定性 FakeEmbeddingProvider，通过 replace_document 写入
内存 Qdrant（与 test_bm25_hybrid.py 隔离模式一致）；端点测试 monkeypatch
编排模块，保证零网络。OpenAlex search_title 用 httpx.MockTransport。
"""

import hashlib
import math
import re
from pathlib import Path
from unittest import mock

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.local_catalog import LocalPaperCatalog
from app.retrieval.models import (
    LocalPaperChunk,
    LocalPaperDocument,
    PaperPage,
)
from app.retrieval.vector_store import QdrantVectorStore
from app.tools.academic_search_provider import OpenAlexSearchProvider
from app.tools.base import ToolResult
from app.tools.semantic_scholar_provider import SemanticScholarClient

client = TestClient(app)

# 保存原始 httpx.AsyncClient 引用（在 mock 内部使用，避免递归）
_RealAsyncClient = httpx.AsyncClient


def _make_mock_client(transport, *a, **kw):
    return _RealAsyncClient(transport=transport, *a, **kw)


class FakeEmbeddingProvider(EmbeddingProvider):
    """测试专用确定性向量，不下载真实模型。"""

    @property
    def dimension(self) -> int:
        return 64

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)

    def embed_documents(self, texts) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        tokens.extend(re.findall(r"[一-鿿]", text))
        for token in tokens:
            position = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:4],
                "big",
            ) % self.dimension
            vector[position] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def _chunk(
    chunk_id: str,
    text: str,
    *,
    paper_id: str = "local:paper-a",
    year: int = 2025,
    is_reference_section: bool = False,
    title: str = "Test Paper",
) -> LocalPaperChunk:
    return LocalPaperChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=title,
        page=1,
        text=text,
        source_path=f"/tmp/{paper_id}/{chunk_id}.pdf",
        zotero_storage_key="TESTKEY",
        content_hash=f"hash-{chunk_id}",
        chunk_index=0,
        year=year,
        is_reference_section=is_reference_section,
        total_chunks=1,
    )


def _document(
    chunk: LocalPaperChunk, *, doi: str = "10.1234/demo", size_bytes: int = 2048
) -> LocalPaperDocument:
    return LocalPaperDocument(
        paper_id=chunk.paper_id,
        title=chunk.title,
        authors=["Alice", "Bob"],
        year=chunk.year or 2025,
        doi=doi,
        source_path=Path(chunk.source_path),
        zotero_storage_key=chunk.zotero_storage_key,
        content_hash=chunk.content_hash,
        modified_ns=0,
        size_bytes=size_bytes,
        pages=[PaperPage(page=1, text=chunk.text)],
    )


def _build_store(chunks):
    embedder = FakeEmbeddingProvider()
    store = QdrantVectorStore(
        QdrantClient(":memory:"),
        collection_name="catalog_test",
        vector_size=embedder.dimension,
        index_version="v4-section-quality-reference-boundary-late-signal",
    )
    by_source: dict = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_path, []).append(chunk)
    for source_chunks in by_source.values():
        store.replace_document(
            _document(source_chunks[0]),
            source_chunks,
            embedder.embed_documents([c.text for c in source_chunks]),
        )
    return store


# --------------------------------------------------------------------------- #
# LocalPaperCatalog
# --------------------------------------------------------------------------- #


class TestLocalPaperCatalog:
    def test_all_papers_groups_payload_fields(self):
        store = _build_store([
            _chunk("body", "graph based entity alignment"),
            _chunk(
                "ref", "references cited", is_reference_section=True,
            ),
        ])

        papers = store.all_papers()

        assert len(papers) == 1
        paper = papers[0]
        assert paper["paper_id"] == "local:paper-a"
        assert paper["title"] == "Test Paper"
        assert paper["authors"] == ["Alice", "Bob"]
        assert paper["doi"] == "10.1234/demo"
        assert paper["size_bytes"] == 2048
        assert paper["chunk_count"] == 2
        # 摘要片段优先取非参考文献 chunk 的正文。
        assert paper["snippet"].startswith("graph based")

    def test_match_by_title_exact(self):
        store = _build_store([
            _chunk("body", "text", title="Attention Is All You Need"),
        ])
        catalog = LocalPaperCatalog(store)

        match = catalog.match_by_title("Attention Is All You Need")

        assert match is not None
        assert match["paper_id"] == "local:paper-a"
        assert match["match_confidence"] == pytest.approx(1.0)

    def test_match_by_title_partial_token_overlap(self):
        store = _build_store([
            _chunk("body", "text", title="Attention Is All You Need"),
        ])
        catalog = LocalPaperCatalog(store)

        match = catalog.match_by_title("attention is all")

        assert match is not None
        assert match["match_confidence"] > 0.0

    def test_match_by_title_chinese(self):
        store = _build_store([
            _chunk("body", "text", title="多模态实体对齐方法综述"),
        ])
        catalog = LocalPaperCatalog(store)

        match = catalog.match_by_title("多模态实体对齐")

        assert match is not None
        assert match["match_confidence"] > 0.0

    def test_match_by_title_no_match(self):
        store = _build_store([
            _chunk("body", "text", title="Graph Neural Networks"),
        ])
        catalog = LocalPaperCatalog(store)

        assert catalog.match_by_title("Quantum Chemistry Survey") is None

    def test_revision_rebuild_after_replace(self):
        store = _build_store([_chunk("a", "text", title="First Paper")])
        catalog = LocalPaperCatalog(store)
        assert catalog.match_by_title("First Paper") is not None

        new_chunk = _chunk("b", "text", paper_id="local:paper-b", title="Second Paper")
        embedder = FakeEmbeddingProvider()
        store.replace_document(
            _document(new_chunk),
            [new_chunk],
            embedder.embed_documents(["text"]),
        )

        # data_revision 自增后目录自动重建。
        assert catalog.match_by_title("Second Paper") is not None
        assert catalog.match_by_title("First Paper") is not None


# --------------------------------------------------------------------------- #
# /api/papers/detail 端点编排
# --------------------------------------------------------------------------- #


class _FakeCatalog:
    def __init__(self, match):
        self._match = match

    def match_by_title(self, title):
        return self._match


_LOCAL_MATCH = {
    "paper_id": "local:paper-a",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "year": 2017,
    "doi": "10.5555/3295222.3295349",
    "source_path": "/path/to/zotero/ABCDEF/AttentionIsAllYouNeed.pdf",
    "zotero_storage_key": "ABCDEF",
    "size_bytes": 2048,
    "chunk_count": 3,
    "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
    "match_confidence": 1.0,
}

_S2_SOURCE = {
    "source_id": "s2:attention",
    "title": "Attention Is All You Need",
    "url": "https://www.semanticscholar.org/paper/attention",
    "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "year": 2017,
    "venue": "NeurIPS",
    "provider": "semantic_scholar",
    "cited_by_count": 12345,
    "reference_count": 17,
}


class TestPaperDetailEndpoint:
    def test_local_hit_returns_local_detail(self, monkeypatch):
        import app.services.paper_detail as svc

        monkeypatch.setattr(svc, "get_local_rag_enabled", lambda: True)
        monkeypatch.setattr(
            svc, "build_local_paper_catalog", lambda: _FakeCatalog(_LOCAL_MATCH)
        )

        response = client.get(
            "/api/papers/detail", params={"query": "Attention Is All You Need"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["found"] is True
        assert body["matched_local"] is True
        assert body["provider"] == "local_zotero"
        assert body["resolved_via"] == "local"
        assert body["paper"]["title"] == "Attention Is All You Need"
        assert body["paper"]["authors"] == _LOCAL_MATCH["authors"]
        assert body["local"]["source_path"] == _LOCAL_MATCH["source_path"]
        assert body["local"]["chunk_count"] == 3
        assert body["abstract"]

    def test_local_miss_falls_back_to_s2(self, monkeypatch):
        import app.services.paper_detail as svc

        monkeypatch.setattr(svc, "get_local_rag_enabled", lambda: True)
        monkeypatch.setattr(svc, "build_local_paper_catalog", lambda: _FakeCatalog(None))

        async def fake_paper_graph(
            self, paper_query, relation="details", limit=5, offset=0
        ):
            return ToolResult(
                success=True,
                tool_name="semantic_scholar_graph",
                data={"results": [_S2_SOURCE]},
            )

        monkeypatch.setattr(SemanticScholarClient, "paper_graph", fake_paper_graph)

        response = client.get(
            "/api/papers/detail", params={"query": "Attention Is All You Need"}
        )

        body = response.json()
        assert body["found"] is True
        assert body["matched_local"] is False
        assert body["provider"] == "semantic_scholar"
        assert body["resolved_via"] == "online"
        assert body["paper"]["title"] == "Attention Is All You Need"
        assert any("Vaswani" in name for name in body["paper"]["authors"])
        assert body["abstract"]

    def test_local_and_s2_miss_falls_back_to_openalex(self, monkeypatch):
        import app.services.paper_detail as svc

        monkeypatch.setattr(svc, "get_local_rag_enabled", lambda: True)
        monkeypatch.setattr(svc, "build_local_paper_catalog", lambda: _FakeCatalog(None))

        async def fake_paper_graph(
            self, paper_query, relation="details", limit=5, offset=0
        ):
            return ToolResult(
                success=False, tool_name="semantic_scholar_graph", error="boom"
            )

        monkeypatch.setattr(SemanticScholarClient, "paper_graph", fake_paper_graph)

        async def fake_search_title(self, title, limit=3):
            return ToolResult(
                success=True,
                tool_name="academic_search",
                data={
                    "results": [{
                        "source_id": "W2741809807",
                        "openalex_id": "W2741809807",
                        "title": "Attention Is All You Need",
                        "url": "https://openalex.org/works/W2741809807",
                        "snippet": "abstract text",
                        "authors": ["Ashish Vaswani"],
                        "year": 2017,
                        "venue": "NeurIPS",
                        "provider": "openalex",
                        "doi": "10.5555/3295222.3295349",
                        "cited_by_count": 30000,
                    }]
                },
            )

        monkeypatch.setattr(OpenAlexSearchProvider, "search_title", fake_search_title)

        response = client.get(
            "/api/papers/detail", params={"query": "Attention Is All You Need"}
        )

        body = response.json()
        assert body["found"] is True
        assert body["provider"] == "openalex"
        assert body["resolved_via"] == "online"
        assert body["paper"]["openalex_id"] == "W2741809807"
        assert body["paper"]["cited_by_count"] == 30000

    def test_all_sources_miss_returns_not_found(self, monkeypatch):
        import app.services.paper_detail as svc

        monkeypatch.setattr(svc, "get_local_rag_enabled", lambda: True)
        monkeypatch.setattr(svc, "build_local_paper_catalog", lambda: _FakeCatalog(None))

        async def fake_paper_graph(
            self, paper_query, relation="details", limit=5, offset=0
        ):
            return ToolResult(
                success=False, tool_name="semantic_scholar_graph", error="not found"
            )

        monkeypatch.setattr(SemanticScholarClient, "paper_graph", fake_paper_graph)

        async def fake_search_title(self, title, limit=3):
            return ToolResult(
                success=False, tool_name="academic_search", error="no api key"
            )

        monkeypatch.setattr(OpenAlexSearchProvider, "search_title", fake_search_title)

        response = client.get(
            "/api/papers/detail", params={"query": "nonexistent title"}
        )

        body = response.json()
        assert body["found"] is False
        assert body["error"]

    def test_empty_query_rejected(self):
        response = client.get("/api/papers/detail", params={"query": ""})
        assert response.status_code == 422

    def test_local_rag_disabled_skips_local_lookup(self, monkeypatch):
        import app.services.paper_detail as svc

        monkeypatch.setattr(svc, "get_local_rag_enabled", lambda: False)
        called = {}

        async def fake_paper_graph(
            self, paper_query, relation="details", limit=5, offset=0
        ):
            called["query"] = paper_query
            return ToolResult(
                success=True,
                tool_name="semantic_scholar_graph",
                data={"results": [_S2_SOURCE]},
            )

        monkeypatch.setattr(SemanticScholarClient, "paper_graph", fake_paper_graph)

        response = client.get("/api/papers/detail", params={"query": "some title"})

        assert response.json()["provider"] == "semantic_scholar"
        assert called["query"] == "some title"


# --------------------------------------------------------------------------- #
# OpenAlexSearchProvider.search_title 兜底
# --------------------------------------------------------------------------- #


def _minimal_work():
    return {
        "id": "https://openalex.org/works/W2741809807",
        "doi": "https://doi.org/10.48550/arxiv.2309.15217",
        "display_name": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "publication_year": 2023,
        "type": "article",
        "cited_by_count": 245,
        "authorships": [{"author": {"display_name": "Shahul Es"}}],
        "primary_location": {"source": {"display_name": "arXiv preprint (arXiv:2309.15217)"}},
        "abstract_inverted_index": {"We": [0], "introduce": [1], "RAGAS": [2]},
        "open_access": {"is_oa": True, "oa_status": "gold"},
    }


class TestOpenAlexSearchTitle:
    @pytest.mark.asyncio
    async def test_search_title_uses_title_filter_and_maps_work(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key-123")
        provider = OpenAlexSearchProvider()
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={"meta": {"count": 1}, "results": [_minimal_work()]},
            )

        transport = httpx.MockTransport(handler)

        with mock.patch.object(
            httpx,
            "AsyncClient",
            side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw),
        ):
            result = await provider.search_title("RAGAS", limit=3)

        assert result.success
        items = result.data["results"]
        assert items
        assert items[0]["title"].startswith("RAGAS")
        assert items[0]["provider"] == "openalex"
        assert items[0]["openalex_id"] == "W2741809807"
        # 标题聚焦：filter=title.search:... + 相关性 search 参数（: 会被 URL 编码）。
        assert "filter=title.search" in captured["url"]
        assert "search=RAGAS" in captured["url"]
        assert "api_key=test-key-123" in captured["url"]

    @pytest.mark.asyncio
    async def test_search_title_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        provider = OpenAlexSearchProvider()

        result = await provider.search_title("RAGAS", limit=3)

        assert not result.success
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_search_title_empty_query_rejected(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key-123")
        provider = OpenAlexSearchProvider()

        result = await provider.search_title("   ", limit=3)

        assert not result.success
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_title_retries_on_429(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key-123")
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "1")
        provider = OpenAlexSearchProvider()
        call_count = [0]

        def handler(request):
            call_count[0] += 1
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "rate limited"})

        transport = httpx.MockTransport(handler)

        with mock.patch.object(
            httpx,
            "AsyncClient",
            side_effect=lambda *a, **kw: _make_mock_client(transport, *a, **kw),
        ):
            with mock.patch("asyncio.sleep", return_value=None):
                result = await provider.search_title("RAGAS", limit=3)

        assert not result.success
        assert "429" in result.error
        assert call_count[0] == 2  # 1 initial + 1 retry
