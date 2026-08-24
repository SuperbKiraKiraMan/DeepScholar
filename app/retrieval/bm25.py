"""BM25 关键词索引与 RRF 融合。

混合检索的第二条通道：用 OKAPI BM25 对 chunk 文本做稀疏关键词检索，
与向量余弦通道互补。BM25 擅长精确术语/编号/限定词，向量擅长同义近义。

与 Qdrant 原生 sparse vector 相比，这里自建 BM25 的好处：
- 不重建 collection、不改 schema，只需从现有 payload 的 ``text`` 字段惰性构建；
- 零新依赖，纯标准库实现；公式与参数显式可讲；
- 构建失败/为空时上层自动退化为纯向量检索。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

from app.retrieval.models import LocalPaperChunk, SearchHit

# 极简英文停用词：对学术检索几乎没有区分度的词。
# 中文不做停用（单字/双字本身即承载语义），靠 bigram 覆盖。
_STOPWORDS = frozenset(
    {"a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with"}
)

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> List[str]:
    """中英混合分词：英文/数字按词提取并小写，中文按连续汉字切 2-gram。

    "BERT-base 对比学习" → ["bert", "base", "对比", "比学", "学习"]
    """
    lowered = text.lower()
    tokens = _ASCII_TOKEN_RE.findall(lowered)
    # 中文没有空格分词：连续汉字串切成 bigram，可匹配任意两个相邻汉字。
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return [token for token in tokens if token not in _STOPWORDS]


class BM25Index:
    """OKAPI BM25 倒排索引。构建后只读，可安全跨线程共享。"""

    def __init__(
        self,
        chunks: List[LocalPaperChunk],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self._chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._build()

    @property
    def indexed_count(self) -> int:
        return len(self._chunks)

    def _build(self) -> None:
        """一次构建：文档长度、avgdl、倒排 postings、doc_freq、IDF。"""
        doc_len: List[int] = []
        # term -> {doc_idx: term_frequency}
        postings: Dict[str, Dict[int, int]] = {}
        for doc_index, chunk in enumerate(self._chunks):
            term_counts: Dict[str, int] = defaultdict(int)
            for token in tokenize(chunk.text):
                term_counts[token] += 1
            doc_len.append(sum(term_counts.values()))
            for token, count in term_counts.items():
                postings.setdefault(token, {})[doc_index] = count

        doc_count = len(self._chunks)
        self._doc_len = doc_len
        self._avgdl = (sum(doc_len) / doc_count) if doc_count else 0.0
        self._postings = postings
        # IDF 加平滑，避免 df==N 时取 ln(0)。
        self._idf = {
            term: math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            for term, doc_freq in {t: len(docs) for t, docs in postings.items()}.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        paper_id: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        exclude_references: bool = False,
    ) -> List[SearchHit]:
        """关键词检索，应用与向量通道一致的结构化过滤。"""
        if not self._chunks or top_k <= 0:
            return []
        terms = [token for token in tokenize(query) if token in self._idf]
        if not terms:
            return []

        # score(d,q) = Σ_t IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))
        raw_scores: Dict[int, float] = defaultdict(float)
        for term in terms:
            idf_t = self._idf[term]
            for doc_index, tf in self._postings[term].items():
                doc_length = self._doc_len[doc_index]
                if doc_length == 0:
                    continue
                norm = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self._avgdl)
                )
                raw_scores[doc_index] += idf_t * tf * (self.k1 + 1) / norm

        scored = [
            (doc_index, raw_scores[doc_index])
            for doc_index in raw_scores
            if raw_scores[doc_index] > 0
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[:top_k]

        hits: List[SearchHit] = []
        for doc_index, score in scored:
            chunk = self._chunks[doc_index]
            if not self._matches(
                chunk,
                paper_id=paper_id,
                year_from=year_from,
                year_to=year_to,
                exclude_references=exclude_references,
            ):
                continue
            hits.append(SearchHit(chunk=chunk, score=score))
        return hits

    @staticmethod
    def _matches(
        chunk: LocalPaperChunk,
        *,
        paper_id: Optional[str],
        year_from: Optional[int],
        year_to: Optional[int],
        exclude_references: bool,
    ) -> bool:
        if paper_id is not None and chunk.paper_id != paper_id:
            return False
        if year_from is not None and (chunk.year is None or chunk.year < year_from):
            return False
        if year_to is not None and (chunk.year is None or chunk.year > year_to):
            return False
        if exclude_references and chunk.is_reference_section:
            return False
        return True


def rrf_fuse(
    dense_hits: List[SearchHit],
    keyword_hits: List[SearchHit],
    *,
    rrf_k: int = 60,
    bm25_norm_k: float = 1.0,
) -> List[SearchHit]:
    """Reciprocal Rank Fusion 合并两个独立排名，返回完整融合排序（未截断）。

    RRF_score(d) = Σ_c 1/(rrf_k + rank_c(d))，rank 从 1 开始。
    融合后按 RRF 分降序，但 ``SearchHit.score`` 保持 [0,1] 相关性量纲：
    - 两通道均命中的 chunk 沿用向量余弦分（与旧行为一致）；
    - 仅关键词命中的 chunk 用饱和映射 bm25/(bm25 + bm25_norm_k) 压到 (0,1)，
      使下游 ``retrieval_score``/``min_score``/graph runtime 的阈值语义不变。
    """
    fused_scores: Dict[str, float] = defaultdict(float)
    # chunk_id -> {chunk, dense_score, keyword_score}
    hit_info: Dict[str, Dict] = {}
    for hits, is_dense in ((dense_hits, True), (keyword_hits, False)):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            fused_scores[chunk_id] += 1.0 / (rrf_k + rank)
            info = hit_info.setdefault(
                chunk_id, {"chunk": hit.chunk, "dense": None, "keyword": None}
            )
            if is_dense:
                info["dense"] = hit.score
            else:
                info["keyword"] = hit.score

    ordered = sorted(fused_scores, key=fused_scores.get, reverse=True)
    fused: List[SearchHit] = []
    for chunk_id in ordered:
        info = hit_info[chunk_id]
        if info["dense"] is not None:
            score = float(info["dense"])
        elif info["keyword"] is not None and info["keyword"] > 0:
            bm25_score = float(info["keyword"])
            score = bm25_score / (bm25_score + bm25_norm_k)
        else:
            score = 0.0
        fused.append(SearchHit(chunk=info["chunk"], score=score))
    return fused
