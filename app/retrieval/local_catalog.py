"""按标题在本地 Zotero 论文库中定位单篇论文的目录视图。

与 BM25 索引共享 data_revision 缓存失效策略：从 VectorStore.all_papers()
构建一次论文级索引，索引变更后自动重建。只读、线程安全。
"""

from __future__ import annotations

import difflib
import re
import threading
from typing import Any, Dict, List, Optional

from app.retrieval.bm25 import tokenize
from app.retrieval.vector_store import VectorStore

# ---- 标题匹配阈值（可调） ----
_TOKEN_OVERLAP_THRESHOLD = 0.6
_TOKEN_OVERLAP_MIN_TOKENS = 3
_SEQUENCE_RATIO_THRESHOLD = 0.8

_SNIPPET_CHARS = 500


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "").strip().casefold())


def _token_overlap(query_tokens: List[str], title_tokens: List[str]) -> float:
    if not query_tokens or not title_tokens:
        return 0.0
    query_set = set(query_tokens)
    overlap = sum(1 for token in title_tokens if token in query_set)
    return overlap / max(len(query_set), len(title_tokens))


def _title_similarity(query: str, title: str) -> Dict[str, float]:
    """返回 (token_overlap, sequence_ratio) 两个互补的标题相似度。"""
    norm_query = _normalize_title(query)
    norm_title = _normalize_title(title)
    query_tokens = tokenize(norm_query)
    title_tokens = tokenize(norm_title)
    overlap = _token_overlap(query_tokens, title_tokens)
    ratio = difflib.SequenceMatcher(None, norm_query, norm_title).ratio()
    return {"overlap": overlap, "ratio": ratio}


class LocalPaperCatalog:
    """按标题定位本地论文；数据来自当前索引版本的论文级视图。"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._papers: Optional[List[Dict[str, Any]]] = None
        self._revision: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        self._refresh()
        return len(self._papers or [])

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        self._refresh()
        for paper in self._papers or []:
            if paper["paper_id"] == paper_id:
                return dict(paper)
        return None

    def match_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """按标题匹配本地论文；返回最佳候选（含 match_confidence）或 None。

        匹配顺序：规范化精确 → token 重叠（英文小写分词 + 中文 bigram，
        复用 bm25.tokenize）→ difflib 序列相似度。多候选取相似度最高者。
        """
        self._refresh()
        if not title or not title.strip():
            return None
        norm_query = _normalize_title(title)
        if not norm_query:
            return None
        query_tokens = tokenize(norm_query)

        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for paper in self._papers or []:
            norm_title = _normalize_title(paper.get("title") or "")
            if not norm_title:
                continue
            sim = _title_similarity(norm_query, norm_title)
            if norm_query == norm_title:
                score = 1.0
            elif (
                len(query_tokens) >= _TOKEN_OVERLAP_MIN_TOKENS
                and sim["overlap"] >= _TOKEN_OVERLAP_THRESHOLD
            ):
                score = sim["overlap"]
            elif sim["ratio"] >= _SEQUENCE_RATIO_THRESHOLD:
                score = sim["ratio"]
            else:
                continue
            if score > best_score:
                best_score = score
                best = dict(paper)
        if best is None:
            return None
        best["match_confidence"] = best_score
        best["snippet"] = str(best.get("snippet") or "")[:_SNIPPET_CHARS]
        return best

    def _refresh(self) -> None:
        revision = self.vector_store.data_revision
        if self._papers is None or self._revision != revision:
            with self._lock:
                if self._papers is None or self._revision != revision:
                    self._papers = self.vector_store.all_papers()
                    self._revision = revision
