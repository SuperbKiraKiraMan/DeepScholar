"""根据配置组装 BGE-M3 与 Qdrant 本地论文检索组件。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import (
    get_bge_m3_batch_size,
    get_bge_m3_cache_dir,
    get_bge_m3_device,
    get_bge_m3_local_files_only,
    get_bge_m3_max_length,
    get_bge_m3_model,
    get_bge_m3_normalize_embeddings,
    get_bge_m3_revision,
    get_local_rag_chunk_overlap,
    get_local_rag_chunk_size,
    get_local_rag_embedding_provider,
    get_local_rag_enabled,
    get_local_rag_hybrid_enabled,
    get_local_rag_index_schema_version,
    get_local_rag_min_score,
    get_qdrant_api_key,
    get_qdrant_batch_size,
    get_qdrant_collection,
    get_qdrant_prefer_grpc,
    get_qdrant_timeout,
    get_qdrant_url,
    get_zotero_storage_path,
)
from app.retrieval.chunker import LocalPaperChunker
from app.retrieval.embedding import BGEM3EmbeddingProvider, EmbeddingProvider
from app.retrieval.indexer import LocalPaperIndexer
from app.retrieval.local_catalog import LocalPaperCatalog
from app.retrieval.pdf_parser import PDFTextExtractor
from app.retrieval.retriever import LocalPaperRetriever
from app.retrieval.vector_store import QdrantVectorStore, VectorStore
from app.retrieval.zotero import ZoteroPDFDiscovery


_QDRANT_CLIENTS: Dict[Tuple[str, str, float, bool], Any] = {}
_QDRANT_CLIENTS_LOCK = threading.Lock()
_DEFAULT_RETRIEVAL_STACK: Optional[
    tuple[LocalPaperIndexer, LocalPaperRetriever, VectorStore]
] = None
_DEFAULT_RETRIEVAL_STACK_LOCK = threading.Lock()


def _get_qdrant_client() -> Any:
    """按连接配置复用 QdrantClient，避免每次 Tool 调用新建连接池。"""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "缺少 qdrant-client，无法连接 Qdrant；"
            "请先安装 requirements.txt 中的生产依赖"
        ) from exc

    url = get_qdrant_url()
    api_key = get_qdrant_api_key()
    timeout = get_qdrant_timeout()
    prefer_grpc = get_qdrant_prefer_grpc()
    cache_key = (url, api_key, timeout, prefer_grpc)
    with _QDRANT_CLIENTS_LOCK:
        client = _QDRANT_CLIENTS.get(cache_key)
        if client is None:
            client = QdrantClient(
                url=url,
                api_key=api_key or None,
                timeout=timeout,
                prefer_grpc=prefer_grpc,
            )
            _QDRANT_CLIENTS[cache_key] = client
        return client


def _build_index_version(
    embedder: EmbeddingProvider,
    *,
    schema_version: Optional[str] = None,
) -> str:
    """索引策略变化时触发安全重建，避免混用不兼容向量。"""
    model_name = getattr(embedder, "model_name", type(embedder).__name__)
    revision = getattr(embedder, "revision", "") or "default"
    selected_schema = (
        schema_version or get_local_rag_index_schema_version()
    ).strip()
    if not selected_schema:
        raise ValueError("LOCAL_RAG_INDEX_SCHEMA_VERSION 不能为空")
    return (
        f"{selected_schema}|{model_name}|{revision}|d{embedder.dimension}|"
        f"c{get_local_rag_chunk_size()}|o{get_local_rag_chunk_overlap()}"
    )


def build_local_retrieval(
    *,
    storage_path: Optional[Path | str] = None,
    embedder: Optional[EmbeddingProvider] = None,
    vector_store: Optional[VectorStore] = None,
    qdrant_client: Optional[Any] = None,
    collection_name: Optional[str] = None,
    index_schema_version: Optional[str] = None,
) -> tuple[LocalPaperIndexer, LocalPaperRetriever, VectorStore]:
    """构造 Retrieval 组件，并允许测试注入假模型与内存 Qdrant。

    两个路径：
    - 全部参数为 None：返回进程级单例，避免反复加载 BGE-M3 模型权重
    - 显式传参：不走单例，重新装配（测试 / 实验用）
    """
    # 全部为 None → 走默认单例路径（生产环境）
    use_default_stack = all(
        value is None
        for value in (
            storage_path,
            embedder,
            vector_store,
            qdrant_client,
            collection_name,
            index_schema_version,
        )
    )
    if use_default_stack:
        global _DEFAULT_RETRIEVAL_STACK
        with _DEFAULT_RETRIEVAL_STACK_LOCK:
            if _DEFAULT_RETRIEVAL_STACK is None:
                _DEFAULT_RETRIEVAL_STACK = _assemble_local_retrieval()
            return _DEFAULT_RETRIEVAL_STACK

    # 显式参数路径：不走单例，直接装配
    return _assemble_local_retrieval(
        storage_path=storage_path,
        embedder=embedder,
        vector_store=vector_store,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        index_schema_version=index_schema_version,
    )


def _assemble_local_retrieval(
    *,
    storage_path: Optional[Path | str] = None,
    embedder: Optional[EmbeddingProvider] = None,
    vector_store: Optional[VectorStore] = None,
    qdrant_client: Optional[Any] = None,
    collection_name: Optional[str] = None,
    index_schema_version: Optional[str] = None,
) -> tuple[LocalPaperIndexer, LocalPaperRetriever, VectorStore]:
    """完成一次组件装配；默认 Tool 路径会在进程内复用装配结果。"""
    storage = Path(storage_path or get_zotero_storage_path()).expanduser()

    if embedder is None:
        embedding_provider = get_local_rag_embedding_provider().strip().lower()
        if embedding_provider not in {"bge-m3", "bge_m3", "bgem3"}:
            raise ValueError(
                "生产环境仅支持 LOCAL_RAG_EMBEDDING_PROVIDER=bge-m3；"
                f"当前配置为 '{embedding_provider}'"
            )
        embedder = BGEM3EmbeddingProvider(
            model_name=get_bge_m3_model(),
            device=get_bge_m3_device(),
            batch_size=get_bge_m3_batch_size(),
            max_length=get_bge_m3_max_length(),
            normalize_embeddings=get_bge_m3_normalize_embeddings(),
            cache_folder=get_bge_m3_cache_dir(),
            revision=get_bge_m3_revision(),
            local_files_only=get_bge_m3_local_files_only(),
        )

    if vector_store is None:
        vector_store = QdrantVectorStore(
            qdrant_client or _get_qdrant_client(),
            collection_name=collection_name or get_qdrant_collection(),
            vector_size=embedder.dimension,
            index_version=_build_index_version(
                embedder,
                schema_version=index_schema_version,
            ),
            batch_size=get_qdrant_batch_size(),
        )

    indexer = LocalPaperIndexer(
        discovery=ZoteroPDFDiscovery(storage),
        parser=PDFTextExtractor(),
        chunker=LocalPaperChunker(
            chunk_size=get_local_rag_chunk_size(),
            chunk_overlap=get_local_rag_chunk_overlap(),
        ),
        embedder=embedder,
        vector_store=vector_store,
    )
    retriever = LocalPaperRetriever(
        embedder,
        vector_store,
        min_score=get_local_rag_min_score(),
        hybrid_enabled=get_local_rag_hybrid_enabled(),
    )
    return indexer, retriever, vector_store


_DEFAULT_LOCAL_CATALOG: Optional[LocalPaperCatalog] = None
_DEFAULT_LOCAL_CATALOG_LOCK = threading.Lock()


def build_local_paper_catalog() -> Optional[LocalPaperCatalog]:
    """构造按标题定位本地论文的目录；失败时返回 None（走在线兜底）。

    复用默认检索栈的 vector_store（与 RAG 检索同源），内部按 data_revision
    懒构建论文级索引。Qdrant 未启用/连接失败时返回 None，且不缓存失败结果，
    下次请求可重试。
    """
    if not get_local_rag_enabled():
        return None
    global _DEFAULT_LOCAL_CATALOG
    if _DEFAULT_LOCAL_CATALOG is None:
        with _DEFAULT_LOCAL_CATALOG_LOCK:
            if _DEFAULT_LOCAL_CATALOG is None:
                try:
                    _DEFAULT_LOCAL_CATALOG = LocalPaperCatalog(
                        build_local_retrieval()[2]
                    )
                except Exception:
                    return None
    return _DEFAULT_LOCAL_CATALOG
