"""基于项目自有 Zotero 向量索引的 LocalPaperSearchTool。"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import get_local_rag_enabled
from app.retrieval.factory import build_local_retrieval
from app.tools.base import BaseTool, ToolResult


class LocalPaperSearchTool(BaseTool):
    """
    搜索已经建立索引的 Zotero PDF Chunk。

    该工具有意不声明 ``task_types``。它会注册为可用能力，但只有在
    不会把它加入 Planner 或 Worker 的默认工具列表。
    """

    @property
    def name(self) -> str:
        return "local_paper_search"

    @property
    def description(self) -> str:
        return (
            "Search full-text chunks from the user's read-only Zotero PDF library. "
            "Returns traceable local paper passages with paper_id, page, source path, "
            "storage key, and retrieval score. Use only when explicitly selected."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or topic to search in indexed local papers",
                    "minLength": 1,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum chunks to return (1-20, default 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "paper_id": {
                    "type": "string",
                    "description": "Optional stable paper identifier filter",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Optional inclusive minimum publication year",
                    "minimum": 1800,
                    "maximum": 2100,
                },
                "year_to": {
                    "type": "integer",
                    "description": "Optional inclusive maximum publication year",
                    "minimum": 1800,
                    "maximum": 2100,
                },
                "include_references": {
                    "type": "boolean",
                    "description": (
                        "Include bibliography/reference chunks. Defaults to true "
                        "only for an explicit reference-oriented query."
                    ),
                    "default": False,
                },
            },
            "required": ["query"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        if not get_local_rag_enabled():
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Local paper retrieval is disabled",
                metadata={"source_type": "local"},
            )

        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Search query is empty",
            )

        top_k = max(1, min(int(kwargs.get("top_k", 5)), 20))
        # _optional_string: 清洗输入 → 去空格 → 空则返回 None
        paper_id = _optional_string(kwargs.get("paper_id"))
        # _optional_int: None/"" → None，否则 int()
        year_from = _optional_int(kwargs.get("year_from"))
        # _optional_int: None/"" → None，否则 int()
        year_to = _optional_int(kwargs.get("year_to"))
        include_references = _optional_bool(
            kwargs.get("include_references"),
            default=_query_requests_references(query),
        )
        if year_from is not None and year_to is not None and year_from > year_to:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="year_from must be less than or equal to year_to",
            )

        try:
            _, retriever, vector_store = await asyncio.to_thread(
                build_local_retrieval
            )
            # 检索命中结果
            hits = await asyncio.to_thread(
                retriever.retrieve,
                query,
                top_k=top_k,
                paper_id=paper_id,
                year_from=year_from,
                year_to=year_to,
                include_references=include_references,
            )
            indexed_chunk_count = await asyncio.to_thread(
                lambda: vector_store.count
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Local paper search failed: {type(exc).__name__}: {str(exc)[:200]}",
                metadata={"source_type": "local"},
            )

        results = []
        # 处理命中结果，提取必要的字段并格式化
        for hit in hits:
            chunk = hit.chunk
            source_path = Path(chunk.source_path)
            try:
                source_url = source_path.resolve(strict=False).as_uri()
            except ValueError:
                source_url = chunk.source_path
            results.append(
                {
                    "paper_id": chunk.paper_id,
                    # Evidence/Citation 绑定到一个精确检索片段；
                    # paper_id 则用于归并同一 PDF 产生的多个 Chunk。
                    "source_id": chunk.chunk_id,
                    "title": chunk.title,
                    "year": chunk.year,
                    "page": chunk.page,
                    "section": chunk.section,
                    "section_type": chunk.section_type,
                    "is_reference_section": chunk.is_reference_section,
                    "extraction_warning": chunk.extraction_warning,
                    "text": chunk.text,
                    "snippet": chunk.text,
                    "full_text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "retrieval_score": round(hit.score, 6),
                    "source": chunk.source_path,
                    "source_path": chunk.source_path,
                    "url": source_url,
                    "zotero_storage_key": chunk.zotero_storage_key,
                    "source_type": "local",
                    "provider": "local_zotero",
                    "content_source": "zotero_pdf",
                }
            )

        years = [result["year"] for result in results if result["year"] is not None]
        metadata = {
            "result_count": len(results),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "source_type": "local",
            "indexed_chunk_count": indexed_chunk_count,
            "include_references": include_references,
        }
        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "results": results,
                "query": query,
                "total_found": len(results),
                "provider": "local_zotero",
            },
            metadata=metadata,
        )


def _optional_string(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _optional_bool(value: Any, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_requests_references(query: str) -> bool:
    lowered = query.lower()
    markers = (
        "reference",
        "bibliography",
        "citation tracing",
        "citation relationship",
        "参考文献",
        "引用关系",
        "引文关系",
        "论文脉络",
        "文献列表",
    )
    return any(marker in lowered for marker in markers)
