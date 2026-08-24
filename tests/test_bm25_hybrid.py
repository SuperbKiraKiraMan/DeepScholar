"""BM25 关键词检索与混合检索（向量 + BM25 + RRF）测试。

自包含：本地定义确定性 FakeEmbeddingProvider，通过 replace_document 写入
内存 Qdrant，与 test_phase_a_retrieval.py 的隔离模式一致。
"""

import hashlib
import math
import re
from pathlib import Path
from typing import List, Tuple

import pytest
from qdrant_client import QdrantClient

from app.retrieval.bm25 import BM25Index, rrf_fuse, tokenize
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.models import (
    LocalPaperChunk,
    LocalPaperDocument,
    PaperPage,
    SearchHit,
)
from app.retrieval.retriever import LocalPaperRetriever
from app.retrieval.vector_store import QdrantVectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    """测试专用确定性向量，不下载真实模型（与 test_phase_a 一致）。"""

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
) -> LocalPaperChunk:
    return LocalPaperChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title="",
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


def _document(chunk: LocalPaperChunk) -> LocalPaperDocument:
    return LocalPaperDocument(
        paper_id=chunk.paper_id,
        title="Test Paper",
        authors=["Alice"],
        year=chunk.year or 2025,
        doi=None,
        source_path=Path(chunk.source_path),
        zotero_storage_key=chunk.zotero_storage_key,
        content_hash=chunk.content_hash,
        modified_ns=0,
        size_bytes=0,
        pages=[PaperPage(page=1, text=chunk.text)],
    )


def _build_stack(
    chunks: List[LocalPaperChunk],
    *,
    min_score: float = 0.25,
    hybrid_enabled: bool = True,
    index_version: str = "v3-section-quality",
) -> Tuple[FakeEmbeddingProvider, QdrantVectorStore, LocalPaperRetriever]:
    embedder = FakeEmbeddingProvider()
    store = QdrantVectorStore(
        QdrantClient(":memory:"),
        collection_name="hybrid_test",
        vector_size=embedder.dimension,
        index_version=index_version,
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
    retriever = LocalPaperRetriever(
        embedder,
        store,
        min_score=min_score,
        hybrid_enabled=hybrid_enabled,
    )
    return embedder, store, retriever


# --------------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------------- #


class TestTokenize:
    def test_english_digits_lowercased(self):
        tokens = tokenize("BERT-base 模型 HNSW v2.0")
        assert "bert" in tokens
        assert "base" in tokens
        assert "hnsw" in tokens
        assert "v2" in tokens
        assert "0" in tokens

    def test_chinese_bigram_segmentation(self):
        # "多模态实体对齐" → 任意相邻双字均可匹配，无空格也能分词。
        tokens = tokenize("多模态实体对齐")
        assert "多模" in tokens
        assert "模态" in tokens
        assert "实体" in tokens
        assert "对齐" in tokens

    def test_minimal_english_stopwords_removed(self):
        tokens = tokenize("the graph and the model")
        assert "graph" in tokens
        assert "model" in tokens
        assert "the" not in tokens
        assert "and" not in tokens


# --------------------------------------------------------------------------- #
# BM25Index
# --------------------------------------------------------------------------- #


class TestBM25Index:
    def test_search_ranks_exact_term_chunk_first(self):
        corpus = [
            _chunk("graph", "graph neural networks learn node representations"),
            _chunk("hnsw", "the hnsw index accelerates nearest neighbor retrieval"),
        ]
        index = BM25Index(corpus)

        hits = index.search("hnsw", top_k=5)

        assert hits
        assert hits[0].chunk.chunk_id == "hnsw"
        assert hits[0].score > 0

    def test_search_returns_empty_for_no_query_term(self):
        index = BM25Index([_chunk("a", "multimodal entity alignment")])

        assert index.search("quantum chemistry", top_k=5) == []

    def test_search_applies_paper_id_filter(self):
        corpus = [
            _chunk("a1", "entity alignment methods", paper_id="local:one"),
            _chunk("b1", "entity alignment survey", paper_id="local:two"),
        ]
        index = BM25Index(corpus)

        hits = index.search("entity alignment", top_k=5, paper_id="local:one")

        assert {hit.chunk.paper_id for hit in hits} == {"local:one"}

    def test_search_applies_year_range_filter(self):
        corpus = [
            _chunk("old", "entity alignment old", year=2020),
            _chunk("new", "entity alignment new", year=2025),
        ]
        index = BM25Index(corpus)

        hits = index.search("entity alignment", top_k=5, year_from=2024)

        assert {hit.chunk.year for hit in hits} == {2025}

    def test_search_excludes_reference_chunks(self):
        corpus = [
            _chunk("body", "graph based entity alignment method"),
            _chunk(
                "ref",
                "wang et al references bibliography cited",
                is_reference_section=True,
            ),
        ]
        index = BM25Index(corpus)

        hits = index.search("references bibliography", top_k=5, exclude_references=True)
        assert hits == []

        hits_including = index.search(
            "references bibliography", top_k=5, exclude_references=False
        )
        assert [hit.chunk.chunk_id for hit in hits_including] == ["ref"]


# --------------------------------------------------------------------------- #
# rrf_fuse
# --------------------------------------------------------------------------- #


class TestRRFFuse:
    def _hit(self, chunk_id: str, score: float) -> SearchHit:
        return SearchHit(chunk=_chunk(chunk_id, f"text {chunk_id}"), score=score)

    def test_keyword_only_hit_enters_fused_result(self):
        # B 只被关键词通道命中，混合后仍应进入结果——这是混合检索的核心价值。
        dense = [self._hit("a", 0.8)]
        keyword = [self._hit("a", 5.0), self._hit("b", 3.0)]

        fused = rrf_fuse(dense, keyword)

        ids = [hit.chunk.chunk_id for hit in fused]
        assert "a" in ids
        assert "b" in ids
        # RRF 排序：a 两通道第 1 → 2/61；b 仅关键词第 2 → 1/62。
        assert ids.index("a") < ids.index("b")

    def test_dense_hit_keeps_cosine_score(self):
        dense = [self._hit("a", 0.82), self._hit("b", 0.31)]
        keyword = [self._hit("b", 7.0), self._hit("a", 4.0)]

        fused = rrf_fuse(dense, keyword)
        by_id = {hit.chunk.chunk_id: hit.score for hit in fused}

        # a、b 两通道均命中 → 沿用向量余弦分。
        assert by_id["a"] == pytest.approx(0.82)
        assert by_id["b"] == pytest.approx(0.31)

    def test_keyword_only_score_saturates_into_unit_interval(self):
        dense = [self._hit("a", 0.9)]
        keyword = [self._hit("a", 4.0), self._hit("b", 3.0)]

        fused = rrf_fuse(dense, keyword)
        by_id = {hit.chunk.chunk_id: hit.score for hit in fused}

        # 仅关键词命中的 b：3/(3+1) = 0.75，保持 [0,1] 相关性量纲。
        assert by_id["b"] == pytest.approx(0.75)
        assert 0.0 <= by_id["b"] <= 1.0


# --------------------------------------------------------------------------- #
# 混合检索集成
# --------------------------------------------------------------------------- #


class TestHybridRetrieval:
    def test_hybrid_recovers_keyword_only_chunk_vector_misses(self):
        # B 含精确术语 hnsw 但 21 个 token 稀释了余弦（约 0.20 < 0.25）；
        # BM25 对稀有词 hnsw 给出高分，饱和映射后约 0.35 ≥ 0.25，混合检索挽回。
        a = _chunk("a", "quantum chemistry exotic matter")
        b = _chunk(
            "b",
            "the hnsw graph index structure organizes high dimensional vectors "
            "supports efficient approximate nearest neighbor search for large "
            "scale retrieval systems and graph based reranking",
        )
        _, store, hybrid = _build_stack([a, b], min_score=0.25)
        _, _, plain = _build_stack([a, b], min_score=0.25, hybrid_enabled=False)

        hybrid_hits = hybrid.retrieve("HNSW", top_k=5)
        plain_hits = plain.retrieve("HNSW", top_k=5)

        # 纯向量：余弦 ~0.20 低于 0.25 → 空；混合：关键词通道挽回 chunk b。
        assert plain_hits == []
        assert [hit.chunk.chunk_id for hit in hybrid_hits] == ["b"]
        assert 0.25 <= hybrid_hits[0].score <= 1.0

    def test_hybrid_disabled_matches_legacy_pure_vector_behavior(self):
        a = _chunk("a", "multimodal entity alignment combines visual textual signals")
        _, store, plain = _build_stack([a], min_score=0.01, hybrid_enabled=False)
        _, _, hybrid = _build_stack([a], min_score=0.01, hybrid_enabled=True)

        plain_hits = plain.retrieve("multimodal entity alignment", top_k=5)
        hybrid_hits = hybrid.retrieve("multimodal entity alignment", top_k=5)

        assert plain_hits
        assert hybrid_hits
        # 语义查询：两通道均命中同一 chunk，score 沿用向量余弦，行为一致。
        assert plain_hits[0].chunk.chunk_id == hybrid_hits[0].chunk.chunk_id
        assert hybrid_hits[0].score == pytest.approx(plain_hits[0].score)

    def test_year_filter_applies_to_both_channels(self):
        new = _chunk(
            "new", "multimodal agent evaluation plans tool use and safety",
            paper_id="local:new", year=2025,
        )
        old = _chunk(
            "old", "multimodal agent evaluation in an older benchmark",
            paper_id="local:old", year=2020,
        )
        _, _, retriever = _build_stack([new, old], min_score=0.01)

        hits = retriever.retrieve("multimodal agent evaluation", top_k=5, year_from=2024)

        assert hits
        assert {hit.chunk.year for hit in hits} == {2025}
        assert {hit.chunk.paper_id for hit in hits} == {"local:new"}

    def test_reference_filter_applies_to_keyword_channel(self):
        body = _chunk("body", "the graph based method achieves strong alignment results")
        ref = _chunk(
            "ref",
            "wang et al references bibliography cited literature",
            is_reference_section=True,
        )
        _, _, retriever = _build_stack([body, ref], min_score=0.01)

        # 查询命中正文 → 默认排除参考文献。
        body_hits = retriever.retrieve("graph alignment", top_k=5)
        assert body_hits
        assert all(not hit.chunk.is_reference_section for hit in body_hits)

        # 查询只命中参考文献 → 默认过滤掉；显式 include_references 时返回。
        default_hits = retriever.retrieve("references bibliography", top_k=5)
        assert default_hits == []

        ref_hits = retriever.retrieve(
            "references bibliography", top_k=5, include_references=True
        )
        assert [hit.chunk.chunk_id for hit in ref_hits] == ["ref"]

    def test_min_score_9999_unrelated_query_returns_empty(self):
        # 回归保护：关键词通道也不能绕过 min_score，避免无关查询返回结果。
        a = _chunk("a", "multimodal entity alignment combines visual textual signals")
        _, _, retriever = _build_stack([a], min_score=0.9999)

        assert retriever.retrieve("unrelated quantum chemistry", top_k=5) == []

    def test_top_k_truncation_respected(self):
        chunks = [
            _chunk(str(index), f"entity alignment discussion number {index}")
            for index in range(5)
        ]
        _, _, retriever = _build_stack(chunks, min_score=0.01)

        hits = retriever.retrieve("entity alignment", top_k=2)

        assert len(hits) <= 2


# --------------------------------------------------------------------------- #
# VectorStore 扩展接口
# --------------------------------------------------------------------------- #


class TestVectorStoreHybridExtensions:
    def test_all_chunks_roundtrips_payload_fields(self):
        body = _chunk("body", "graph based entity alignment", year=2025)
        ref = _chunk(
            "ref",
            "references cited bibliography",
            is_reference_section=True,
        )
        _, store, _ = _build_stack([body, ref])

        chunks = store.all_chunks()
        by_id = {chunk.chunk_id: chunk for chunk in chunks}

        assert set(by_id) == {"body", "ref"}
        assert by_id["body"].text == body.text
        assert by_id["body"].year == 2025
        assert by_id["ref"].is_reference_section is True
        assert by_id["body"].paper_id == "local:paper-a"

    def test_data_revision_increments_on_replace(self):
        _, store, _ = _build_stack([_chunk("a", "entity alignment")])

        revision = store.data_revision
        assert revision >= 1

        store.replace_document(
            _document(_chunk("b", "another chunk")),
            [_chunk("b", "another chunk")],
            FakeEmbeddingProvider().embed_documents(["another chunk"]),
        )
        assert store.data_revision == revision + 1

    def test_supports_section_metadata_depends_on_index_version(self):
        _, v3_store, _ = _build_stack(
            [_chunk("a", "entity alignment")], index_version="v3-section-quality"
        )
        _, legacy_store, _ = _build_stack(
            [_chunk("a", "entity alignment")], index_version="v2-legacy"
        )

        assert v3_store.supports_section_metadata is True
        assert legacy_store.supports_section_metadata is False
