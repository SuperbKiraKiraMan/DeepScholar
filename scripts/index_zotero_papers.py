#!/usr/bin/env python3
"""Zotero 只读发现、Qdrant 索引与检索验收命令。"""

import argparse
import json
import logging
from pathlib import Path

from app.core.config import (
    get_local_rag_index_schema_version,
    get_qdrant_collection,
    get_qdrant_url,
)
from app.core.config import get_zotero_storage_path
from app.retrieval.factory import build_local_retrieval
from app.retrieval.zotero import ZoteroPDFDiscovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 BGE-M3 把 Zotero PDF 索引到 Qdrant"
    )
    parser.add_argument(
        "--storage-path",
        default=get_zotero_storage_path(),
        help="只读的 Zotero storage 根目录",
    )
    parser.add_argument(
        "--collection",
        default=get_qdrant_collection(),
        help="Qdrant collection 名称",
    )
    parser.add_argument(
        "--index-schema-version",
        default=get_local_rag_index_schema_version(),
        help="索引 payload/Chunk schema 版本",
    )
    parser.add_argument(
        "--confirm-in-place-migration",
        action="store_true",
        help="明确允许在活动 collection 中切换索引 schema",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="只扫描并输出统计信息，不加载模型或连接 Qdrant",
    )
    parser.add_argument("--query", default="", help="可选的检索验收查询")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    storage_path = Path(args.storage_path).expanduser()

    if args.discover_only:
        report = ZoteroPDFDiscovery(storage_path).discover()
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    active_collection = get_qdrant_collection()
    active_schema = get_local_rag_index_schema_version()
    if (
        args.collection == active_collection
        and args.index_schema_version != active_schema
        and not args.confirm_in_place_migration
    ):
        print(
            json.dumps(
                {
                    "error": "拒绝在活动 collection 中静默切换索引 schema",
                    "active_collection": active_collection,
                    "active_schema": active_schema,
                    "requested_schema": args.index_schema_version,
                    "suggestion": (
                        "请使用新的 --collection，或显式传入 "
                        "--confirm-in-place-migration"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    # 构建索引器、检索器和向量存储 (组装所有组件)
    logging.info("开始索引 Zotero PDF...")
    indexer, retriever, vector_store = build_local_retrieval(
        storage_path=storage_path,
        collection_name=args.collection,
        index_schema_version=args.index_schema_version,
    )
    # 索引 Zotero PDF
    report = indexer.index()
    payload = report.to_dict()
    payload["qdrant_url"] = get_qdrant_url()
    payload["collection"] = args.collection
    payload["index_schema_version"] = args.index_schema_version
    payload["active_collection"] = active_collection
    payload["active_schema"] = active_schema
    payload["total_indexed_chunks"] = vector_store.count
    payload["activation_ready"] = (
        report.failed_pdf_count == 0 and vector_store.count > 0
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    # 用来测试检索功能
    if args.query:
        # 检索 Zotero PDF
        hits = retriever.retrieve(
            args.query,
            top_k=max(1, min(args.top_k, 20)),
        )
        print("\n本地论文检索结果：")
        print(
            json.dumps(
                [
                    {
                        "paper_id": hit.chunk.paper_id,
                        "title": hit.chunk.title,
                        "page": hit.chunk.page,
                        "score": round(hit.score, 6),
                        "source_path": hit.chunk.source_path,
                        "zotero_storage_key": hit.chunk.zotero_storage_key,
                        "text": hit.chunk.text[:500],
                    }
                    for hit in hits
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    # 损坏或加密的 PDF 会被单独记录并跳过，不影响其他论文。
    return 2 if report.failed_pdf_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
