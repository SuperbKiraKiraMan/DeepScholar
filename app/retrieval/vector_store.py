"""VectorStore 抽象及 Qdrant 生产实现。"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.retrieval.models import (
    LocalPaperChunk,
    LocalPaperDocument,
    SearchHit,
)


class VectorStore(ABC):
    """约束索引器和检索器依赖的最小向量存储能力。"""

    @abstractmethod
    def get_document_hash(self, source_path: str) -> Optional[str]:
        """返回当前索引版本下指定 PDF 的内容哈希。"""

    @abstractmethod
    def replace_document(
        self,
        document: LocalPaperDocument,
        chunks: Sequence[LocalPaperChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """原子语义地写入新版本并清理该 PDF 的过期 Chunk。"""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        paper_id: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        include_references: bool = False,
    ) -> List[SearchHit]:
        """执行向量搜索并应用结构化过滤。"""

    @property
    @abstractmethod
    def count(self) -> int:
        """返回当前索引版本的 Chunk 数。"""

    @abstractmethod
    def all_chunks(self) -> List[LocalPaperChunk]:
        """返回当前索引版本的全部 Chunk，作为 BM25 关键词索引的数据源。"""

    @abstractmethod
    def all_papers(self) -> List[Dict[str, Any]]:
        """返回当前索引版本的论文级元数据（按 paper_id 分组）。"""

    @property
    @abstractmethod
    def data_revision(self) -> int:
        """每次写入/替换文档后自增，供上层判定 BM25 缓存是否过期。"""

    @property
    @abstractmethod
    def supports_section_metadata(self) -> bool:
        """索引是否携带 section 元数据，决定参考文献过滤是否生效。"""


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError(
            "缺少 qdrant-client，无法使用本地论文向量库；"
            "请先安装 requirements.txt 中的生产依赖"
        ) from exc
    return models


class QdrantVectorStore(VectorStore):
    """把 Zotero Chunk 与可追踪元数据存储到独立 Qdrant collection。"""

    def __init__(
        self,
        client: Any,
        *,
        collection_name: str,
        vector_size: int,
        index_version: str,
        batch_size: int = 64,
    ):
        if not collection_name.strip():
            raise ValueError("Qdrant collection 名称不能为空")
        if vector_size <= 0:
            raise ValueError("Qdrant vector_size 必须大于 0")
        if batch_size <= 0:
            raise ValueError("Qdrant batch_size 必须大于 0")
        if not index_version.strip():
            raise ValueError("Qdrant index_version 不能为空")

        self.client = client
        self.collection_name = collection_name.strip()
        self.vector_size = vector_size
        self.index_version = index_version.strip()
        self.batch_size = batch_size
        self._supports_section_metadata = self.index_version.startswith(
            ("v3-section-quality", "v4-section-quality")
        )
        # 每次写入自增，供混合检索判定 BM25 索引缓存是否过期。
        self._data_revision = 0
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        models = _qdrant_models()
        if not self.client.collection_exists(self.collection_name):
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            except Exception:
                # 多实例同时首次启动时，另一实例可能已经创建成功。
                if not self.client.collection_exists(self.collection_name):
                    raise

        info = self.client.get_collection(self.collection_name)
        vectors_config = info.config.params.vectors
        configured_size = getattr(vectors_config, "size", None)
        configured_distance = getattr(vectors_config, "distance", None)
        if int(configured_size or 0) != self.vector_size:
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' 的向量维度为 "
                f"{configured_size}，BGE-M3 需要 {self.vector_size}；"
                "请使用新的 collection 名称后重新索引"
            )
        if str(configured_distance).lower().split(".")[-1] != "cosine":
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' 不是 Cosine 距离"
            )
        self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """为高频过滤字段建立索引；本地内存模式不支持时允许跳过。"""
        internal_client = getattr(self.client, "_client", None)
        if (
            internal_client is not None
            and internal_client.__class__.__module__.startswith(
                "qdrant_client.local"
            )
        ):
            return
        models = _qdrant_models()
        fields = {
            "source_path": models.PayloadSchemaType.KEYWORD,
            "paper_id": models.PayloadSchemaType.KEYWORD,
            "index_version": models.PayloadSchemaType.KEYWORD,
            "document_version": models.PayloadSchemaType.KEYWORD,
            "section_type": models.PayloadSchemaType.KEYWORD,
            "is_reference_section": models.PayloadSchemaType.BOOL,
            "year": models.PayloadSchemaType.INTEGER,
        }
        for field_name, field_schema in fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            except (NotImplementedError, ValueError):
                continue

    def get_document_hash(self, source_path: str) -> Optional[str]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._filter(
                source_path=source_path,
                include_current_index=True,
            ),
            limit=1,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        if not points or not points[0].payload:
            return None
        value = points[0].payload.get("content_hash")
        return str(value) if value else None

    def replace_document(
        self,
        document: LocalPaperDocument,
        chunks: Sequence[LocalPaperChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")
        if not chunks:
            raise ValueError("不能写入空 Chunk 集合")
        for vector in vectors:
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"向量维度 {len(vector)} 与 collection 维度 "
                    f"{self.vector_size} 不一致"
                )

        # 数据发生变更，使依赖全量数据的缓存（如 BM25 索引）失效。
        self._data_revision += 1
        models = _qdrant_models()
        source_path = str(document.source_path)
        document_version = f"{document.content_hash}:{self.index_version}"
        points = [
            models.PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=list(vector),
                payload=self._payload(
                    document=document,
                    chunk=chunk,
                    document_version=document_version,
                ),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        for batch in self._batches(points, self.batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        # 先写新版本再删旧版本，索引失败时仍保留上一次可用数据。
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_path",
                            match=models.MatchValue(value=source_path),
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="document_version",
                            match=models.MatchValue(value=document_version),
                        )
                    ],
                )
            ),
            wait=True,
        )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        paper_id: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        include_references: bool = False,
    ) -> List[SearchHit]:
        if not query_vector or top_k <= 0:
            return []
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"查询向量维度 {len(query_vector)} 与 collection 维度 "
                f"{self.vector_size} 不一致"
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=self._filter(
                paper_id=paper_id,
                year_from=year_from,
                year_to=year_to,
                include_current_index=True,
                exclude_references=(
                    not include_references
                    and self._supports_section_metadata
                ),
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            SearchHit(
                chunk=self._chunk_from_payload(point.payload or {}),
                score=float(point.score),
            )
            for point in response.points
        ]

    @property
    def count(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._filter(include_current_index=True),
            exact=True,
        )
        return int(result.count)

    @property
    def data_revision(self) -> int:
        return self._data_revision

    @property
    def supports_section_metadata(self) -> bool:
        return self._supports_section_metadata

    def all_chunks(self) -> List[LocalPaperChunk]:
        """分页 scroll 拉取当前索引版本的全部 Chunk，还原为 LocalPaperChunk。"""
        chunks: List[LocalPaperChunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._filter(include_current_index=True),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload:
                    chunks.append(self._chunk_from_payload(point.payload))
            if offset is None:
                break
        return chunks

    def all_papers(self) -> List[Dict[str, Any]]:
        """论文级视图：按 paper_id 分组原始 payload，补回 all_chunks 丢失的字段。

        镜像 all_chunks 的 scroll 循环（仅当前索引版本），但保留 authors/doi/
        size_bytes 等论文级字段，并统计 chunk 数与首个非参考文献片段摘要。
        """
        papers: Dict[str, Dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._filter(include_current_index=True),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload
                if not payload:
                    continue
                paper_id = str(payload.get("paper_id") or "")
                if not paper_id:
                    continue
                paper = papers.get(paper_id)
                if paper is None:
                    paper = {
                        "paper_id": paper_id,
                        "title": str(payload.get("title") or ""),
                        "authors": list(payload.get("authors") or []),
                        "year": payload.get("year"),
                        "doi": payload.get("doi"),
                        "source_path": str(payload.get("source_path") or ""),
                        "zotero_storage_key": str(
                            payload.get("zotero_storage_key") or ""
                        ),
                        "size_bytes": int(payload.get("size_bytes") or 0),
                        "modified_ns": int(payload.get("modified_ns") or 0),
                        "chunk_count": 0,
                        "snippet": "",
                    }
                    papers[paper_id] = paper
                paper["chunk_count"] += 1
                if not paper["snippet"] and not payload.get(
                    "is_reference_section"
                ):
                    paper["snippet"] = str(payload.get("text") or "")[:500]
            if offset is None:
                break
        return list(papers.values())

    @property
    def document_count(self) -> int:
        source_paths = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._filter(include_current_index=True),
                limit=256,
                offset=offset,
                with_payload=["source_path"],
                with_vectors=False,
            )
            for point in points:
                if point.payload and point.payload.get("source_path"):
                    source_paths.add(str(point.payload["source_path"]))
            if offset is None:
                break
        return len(source_paths)

    def _filter(
        self,
        *,
        source_path: Optional[str] = None,
        paper_id: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        include_current_index: bool = False,
        exclude_references: bool = False,
    ) -> Any:
        models = _qdrant_models()
        conditions = []
        if include_current_index:
            conditions.append(
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchValue(value=self.index_version),
                )
            )
        if source_path:
            conditions.append(
                models.FieldCondition(
                    key="source_path",
                    match=models.MatchValue(value=source_path),
                )
            )
        if paper_id:
            conditions.append(
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchValue(value=paper_id),
                )
            )
        if year_from is not None or year_to is not None:
            conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=year_from, lte=year_to),
                )
            )
        if exclude_references:
            conditions.append(
                models.FieldCondition(
                    key="is_reference_section",
                    match=models.MatchValue(value=False),
                )
            )
        return models.Filter(must=conditions)

    def _payload(
        self,
        *,
        document: LocalPaperDocument,
        chunk: LocalPaperChunk,
        document_version: str,
    ) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "title": chunk.title,
            "authors": list(document.authors),
            "year": chunk.year,
            "doi": document.doi,
            "page": chunk.page,
            "section": chunk.section,
            "section_type": chunk.section_type,
            "is_reference_section": chunk.is_reference_section,
            "extraction_warning": chunk.extraction_warning,
            "total_chunks": chunk.total_chunks,
            "text": chunk.text,
            "source_path": chunk.source_path,
            "zotero_storage_key": chunk.zotero_storage_key,
            "content_hash": chunk.content_hash,
            "chunk_index": chunk.chunk_index,
            "index_version": self.index_version,
            "document_version": document_version,
            "modified_ns": document.modified_ns,
            "size_bytes": document.size_bytes,
            "source_type": "local",
        }

    @staticmethod
    def _chunk_from_payload(payload: Dict[str, Any]) -> LocalPaperChunk:
        return LocalPaperChunk(
            chunk_id=str(payload["chunk_id"]),
            paper_id=str(payload["paper_id"]),
            title=str(payload["title"]),
            page=int(payload["page"]),
            text=str(payload["text"]),
            source_path=str(payload["source_path"]),
            zotero_storage_key=str(payload["zotero_storage_key"]),
            content_hash=str(payload["content_hash"]),
            chunk_index=int(payload["chunk_index"]),
            year=int(payload["year"]) if payload.get("year") is not None else None,
            section=(
                str(payload["section"])
                if payload.get("section") is not None
                else None
            ),
            section_type=str(payload.get("section_type") or "unknown"),
            is_reference_section=bool(
                payload.get("is_reference_section", False)
            ),
            extraction_warning=(
                str(payload["extraction_warning"])
                if payload.get("extraction_warning") is not None
                else None
            ),
            total_chunks=int(payload.get("total_chunks") or 0),
        )

    def _point_id(self, chunk_id: str) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.collection_name}:{chunk_id}",
            )
        )

    @staticmethod
    def _batches(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
        for start in range(0, len(items), size):
            yield list(items[start : start + size])
