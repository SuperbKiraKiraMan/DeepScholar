"""论文详情检索编排：本地 Qdrant 目录优先，Semantic Scholar / OpenAlex 在线兜底。

对外暴露 resolve_paper_detail(query)，供 /api/papers/detail 端点调用。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from app.api.schemas import LocalPaperDetail, PaperDetailResponse, PaperSource
from app.core.config import get_local_rag_enabled
from app.retrieval.factory import build_local_paper_catalog
from app.tools.academic_search_provider import OpenAlexSearchProvider
from app.tools.semantic_scholar_provider import SemanticScholarClient

logger = logging.getLogger(__name__)


async def resolve_paper_detail(query: str) -> PaperDetailResponse:
    """按标题解析论文详情；返回 PaperDetailResponse（永远不抛异常）。"""
    query = (query or "").strip()

    # 1) 本地优先：Qdrant 论文级目录标题匹配（含本地 PDF 来源）。
    if get_local_rag_enabled():
        catalog = build_local_paper_catalog()
        if catalog is not None:
            # 首次构建会整库 scroll，放到线程池避免阻塞事件循环。
            match = await asyncio.to_thread(catalog.match_by_title, query)
            if match:
                return _local_response(query, match)

    # 2) 在线：Semantic Scholar 标题解析 + 论文详情。
    try:
        s2 = SemanticScholarClient()
        result = await s2.paper_graph(
            query, relation="details", limit=1, offset=0
        )
        if result.success:
            items = (result.data or {}).get("results") or []
            if items:
                return _online_response(query, items[0], "semantic_scholar")
    except Exception as exc:
        logger.warning("Semantic Scholar detail lookup failed: %s", exc)

    # 3) 兜底：OpenAlex 标题检索。
    try:
        openalex = OpenAlexSearchProvider()
        result = await openalex.search_title(query, limit=3)
        if result.success:
            items = (result.data or {}).get("results") or []
            if items:
                return _online_response(query, items[0], "openalex")
    except Exception as exc:
        logger.warning("OpenAlex title lookup failed: %s", exc)

    return PaperDetailResponse(
        query=query,
        found=False,
        error="本地库与在线数据源（Semantic Scholar / OpenAlex）均未找到该论文",
    )


def _local_response(query: str, match: Dict[str, Any]) -> PaperDetailResponse:
    """本地命中：返回本地 PDF 来源等论文级详情。"""
    snippet = str(match.get("snippet") or "")
    paper = PaperSource(
        title=str(match.get("title") or "Untitled paper"),
        url=str(match.get("source_path") or ""),
        snippet=snippet,
        authors=list(match.get("authors") or []),
        year=match.get("year"),
        doi=match.get("doi"),
        provider="local_zotero",
        source_type="local",
    )
    local = LocalPaperDetail(
        paper_id=str(match.get("paper_id") or ""),
        source_path=str(match.get("source_path") or ""),
        zotero_storage_key=str(match.get("zotero_storage_key") or ""),
        chunk_count=int(match.get("chunk_count") or 0),
        size_bytes=int(match.get("size_bytes") or 0),
        snippet=snippet,
    )
    return PaperDetailResponse(
        query=query,
        found=True,
        provider="local_zotero",
        resolved_via="local",
        matched_local=True,
        abstract=snippet,
        paper=paper,
        local=local,
    )


def _online_response(
    query: str, source_item: Dict[str, Any], provider: str
) -> PaperDetailResponse:
    """在线命中：S2 / OpenAlex 结果映射为 PaperSource（摘要已在 snippet/full_text）。"""
    item = dict(source_item)
    item.setdefault("url", "")
    paper = PaperSource(**item)
    abstract = (paper.snippet or paper.full_text or "").strip()
    return PaperDetailResponse(
        query=query,
        found=True,
        provider=provider,
        resolved_via="online",
        matched_local=False,
        abstract=abstract,
        paper=paper,
    )
