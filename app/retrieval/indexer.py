"""Zotero PDF 增量索引。"""

from pathlib import Path
from typing import Optional

from app.retrieval.chunker import LocalPaperChunker
from app.retrieval.embedding import EmbeddingProvider
from app.retrieval.models import IndexReport
from app.retrieval.pdf_parser import PDFTextExtractor, sha256_file
from app.retrieval.vector_store import VectorStore
from app.retrieval.zotero import ZoteroPDFDiscovery


class LocalPaperIndexer:
    def __init__(
        self,
        discovery: ZoteroPDFDiscovery,
        parser: PDFTextExtractor,
        chunker: LocalPaperChunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.discovery = discovery
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index(self) -> IndexReport:
        # Step 1: 扫描 Zotero storage 目录，发现所有 .pdf 文件
        discovery_report = self.discovery.discover()
        report = IndexReport(discovery=discovery_report)

        for discovered in discovery_report.discovered_pdfs:
            try:
                # Step 2: 计算文件 SHA256 哈希，对比已索引版本，一致则跳过（增量更新）
                content_hash = sha256_file(discovered.path)
                existing_hash = self.vector_store.get_document_hash(
                    str(discovered.path)
                )
                if existing_hash == content_hash:
                    report.unchanged_pdf_count += 1
                    continue

                # Step 3: 解析 PDF —— 提取文本、元数据和逐页质量检测
                document = self.parser.extract(
                    discovered,
                    content_hash=content_hash,
                )
                for warning in document.extraction_warnings:
                    report.warnings.append(
                        {
                            "source_path": str(discovered.path),
                            **warning,
                        }
                    )
                    report.warning_count += 1

                # Step 4: 智能切块 —— 章节识别 + Unicode 安全分段 + 质量过滤
                chunks = self.chunker.chunk_document(document)
                if not chunks:
                    raise ValueError("PDF 没有通过质量检查的可索引 Chunk")

                # Step 5: BGE-M3 批量向量化
                vectors = self.embedder.embed_documents(
                    [chunk.text for chunk in chunks]
                )

                # Step 6: 写入 Qdrant（先 upsert 新版本，再清理旧版本，保证原子性）
                self.vector_store.replace_document(document, chunks, vectors)
                report.indexed_pdf_count += 1
                report.created_chunk_count += len(chunks)
            except Exception as exc:
                # 单个 PDF 失败不影响其余论文
                report.failed_pdf_count += 1
                report.failures.append(
                    {
                        "source_path": str(discovered.path),
                        "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    }
                )

        return report
