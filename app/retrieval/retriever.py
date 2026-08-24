"""面向 Zotero 本地论文 Chunk 的 Retriever 编排。

默认启用混合检索：向量余弦通道 + BM25 关键词通道，RRF 融合排序。
两条通道应用完全一致的结构化过滤（paper_id / 年份 / 参考文献）。
"""

import threading
from typing import List, Optional

from app.retrieval.bm25 import BM25Index, rrf_fuse
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.models import SearchHit
from app.retrieval.vector_store import VectorStore


class LocalPaperRetriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        min_score: float = 0.05,
        *,
        hybrid_enabled: bool = True,
        rrf_k: int = 60,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        bm25_norm_k: float = 1.0,
        fetch_multiplier: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.min_score = min_score
        self.hybrid_enabled = hybrid_enabled
        self.rrf_k = rrf_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_norm_k = bm25_norm_k
        self.fetch_multiplier = fetch_multiplier
        # BM25 索引懒构建 + 缓存；工具层经 asyncio.to_thread 并发调用，需加锁。
        self._bm25_index: Optional[BM25Index] = None
        self._bm25_revision: Optional[int] = None
        self._bm25_lock = threading.Lock()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_id: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        include_references: bool = False,
    ) -> List[SearchHit]:
        if not query.strip() or self.vector_store.count == 0:
            return []
        if not self.hybrid_enabled:
            # 纯向量路径，保持原行为逐字节不变。
            return self._vector_retrieve(
                query,
                top_k=top_k,
                paper_id=paper_id,
                year_from=year_from,
                year_to=year_to,
                include_references=include_references,
            )

        # BM25 索引构建失败/为空时退化为纯向量检索，不破坏原功能。
        try:
            keyword_index = self._keyword_index()
        except Exception:
            keyword_index = None
        fetch_k = max(top_k * self.fetch_multiplier, top_k + 1)

        dense_hits = self._vector_retrieve(
            query,
            top_k=fetch_k,
            paper_id=paper_id,
            year_from=year_from,
            year_to=year_to,
            include_references=include_references,
        )
        if keyword_index is None or keyword_index.indexed_count == 0:
            return dense_hits[:top_k]

        # 关键词通道与向量通道应用同一参考文献过滤规则。
        exclude_references = (
            not include_references
            and self.vector_store.supports_section_metadata
        )
        keyword_hits = keyword_index.search(
            query,
            top_k=fetch_k,
            paper_id=paper_id,
            year_from=year_from,
            year_to=year_to,
            exclude_references=exclude_references,
        )
        fused = rrf_fuse(
            dense_hits,
            keyword_hits,
            rrf_k=self.rrf_k,
            bm25_norm_k=self.bm25_norm_k,
        )
        # 统一 min_score 过滤（向量阶段已过滤一次，这里兜底仅关键词命中），
        # 保持 score 的 [0,1] 相关性量纲语义不变。
        return [hit for hit in fused if hit.score >= self.min_score][:top_k]

    def _vector_retrieve(
        self,
        query: str,
        *,
        top_k: int,
        paper_id: Optional[str],
        year_from: Optional[int],
        year_to: Optional[int],
        include_references: bool,
    ) -> List[SearchHit]:
        hits = self.vector_store.search(
            self.embedder.embed_query(query),
            top_k=top_k,
            paper_id=paper_id,
            year_from=year_from,
            year_to=year_to,
            include_references=include_references,
        )
        return [hit for hit in hits if hit.score >= self.min_score]

    def _keyword_index(self) -> BM25Index:
        """懒加载 BM25 索引；vector_store 数据修订变化后自动重建。"""
        revision = self.vector_store.data_revision
        if self._bm25_index is None or self._bm25_revision != revision:
            with self._bm25_lock:
                if self._bm25_index is None or self._bm25_revision != revision:
                    chunks = self.vector_store.all_chunks()
                    self._bm25_index = BM25Index(
                        chunks, k1=self.bm25_k1, b=self.bm25_b
                    )
                    self._bm25_revision = revision
        return self._bm25_index
