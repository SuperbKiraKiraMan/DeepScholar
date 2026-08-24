"""Phase A：Zotero 本地论文检索与 OpenAlex 年份过滤测试。"""

import hashlib
import json
import math
import re
from pathlib import Path
from unittest import mock

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)
from qdrant_client import QdrantClient

from app.agents.controller import IntentController
from app.agents.planner import PlannerAgent
from app.agents.schemas import ExecutionClass, ExecutionSpec
from app.retrieval import embedding as embedding_module
from app.retrieval.chunker import LocalPaperChunker
from app.retrieval.embedding import BGEM3EmbeddingProvider, EmbeddingProvider
from app.retrieval.factory import build_local_retrieval
from app.retrieval.models import (
    DiscoveredPDF,
    LocalPaperDocument,
    PaperPage,
)
from app.retrieval.pdf_parser import (
    PDFExtractionError,
    PDFTextExtractor,
    _clean_pdf_text,
    analyze_text_quality,
)
from app.retrieval.retriever import LocalPaperRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.zotero import ZoteroPDFDiscovery
from app.tools.academic_search_provider import OpenAlexSearchProvider
from app.tools.academic_search_tool import AcademicSearchTool
from app.tools.citation_check_tool import CitationCheckTool
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.tools import local_paper_search_tool as local_search_module
from app.tools.local_paper_search_tool import LocalPaperSearchTool
from app.tools.registry import ToolRegistry


def _write_text_pdf(
    path: Path,
    text: str,
    *,
    title: str = "Local Retrieval Paper",
    author: str = "Alice; Bob",
    year: int = 2025,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(
        f"BT /F1 11 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata(
        {
            "/Title": title,
            "/Author": author,
            "/CreationDate": f"D:{year}0101",
        }
    )
    with path.open("wb") as output:
        writer.write(output)


def _file_fingerprint(path: Path) -> tuple[str, int, int]:
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_mtime_ns,
        path.stat().st_size,
    )


class FakeEmbeddingProvider(EmbeddingProvider):
    """测试专用确定性向量，不下载真实模型。"""

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
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))
        for token in tokens:
            position = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:4],
                "big",
            ) % self.dimension
            vector[position] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def _build_test_retrieval(storage: Path):
    """通过内存 Qdrant 隔离单元测试与真实服务。"""
    return build_local_retrieval(
        storage_path=storage,
        embedder=FakeEmbeddingProvider(),
        qdrant_client=QdrantClient(":memory:"),
        collection_name="test_local_papers",
    )


def _install_test_retrieval(monkeypatch, storage: Path):
    stack = _build_test_retrieval(storage)
    monkeypatch.setattr(
        local_search_module,
        "build_local_retrieval",
        lambda: stack,
    )
    return stack


@pytest.fixture
def fake_zotero(tmp_path):
    storage = tmp_path / "zotero-storage"
    storage.mkdir()
    pdf = storage / "4F9JKMJE" / "entity-alignment.pdf"
    _write_text_pdf(
        pdf,
        "Multimodal entity alignment combines visual and textual graph signals "
        "for robust cross-modal representation learning and entity matching.",
        title="Multimodal Entity Alignment",
    )
    return storage, pdf


class TestZoteroDiscovery:
    def test_discovers_pdf_and_skips_html_snapshot(self, tmp_path):
        storage = tmp_path / "storage"
        _write_text_pdf(storage / "4F9JKMJE" / "paper.PDF", "A" * 100)
        html = storage / "5CJYBDJW" / "snapshot.html"
        html.parent.mkdir(parents=True)
        html.write_text("<html>snapshot</html>")

        report = ZoteroPDFDiscovery(storage).discover()

        assert report.discovered_pdf_count == 1
        assert report.discovered_pdfs[0].zotero_storage_key == "4F9JKMJE"
        assert report.skipped_html_snapshot_count == 1
        assert report.skipped_no_pdf_directory_count == 1

    def test_multiple_pdfs_and_empty_directory_are_safe(self, tmp_path):
        storage = tmp_path / "storage"
        _write_text_pdf(storage / "MULTIKEY" / "one.pdf", "A" * 100)
        _write_text_pdf(storage / "MULTIKEY" / "two.PdF", "B" * 100)
        (storage / "EMPTYKEY").mkdir(parents=True)

        report = ZoteroPDFDiscovery(storage).discover()

        assert report.discovered_pdf_count == 2
        assert report.skipped_no_pdf_directory_count == 1

    def test_hidden_and_temporary_files_are_ignored(self, tmp_path):
        storage = tmp_path / "storage"
        item = storage / "TEMPKEY"
        item.mkdir(parents=True)
        (item / ".hidden.pdf").write_bytes(b"%PDF")
        (item / "partial.pdf.part").write_bytes(b"%PDF")

        report = ZoteroPDFDiscovery(storage).discover()

        assert report.discovered_pdf_count == 0
        assert report.skipped_hidden_or_temporary_count == 2

    def test_discovery_is_read_only(self, fake_zotero):
        storage, pdf = fake_zotero
        before = _file_fingerprint(pdf)

        ZoteroPDFDiscovery(storage).discover()

        assert _file_fingerprint(pdf) == before


class TestPDFChunkIndex:
    def test_pdf_metadata_and_page_are_preserved(self, fake_zotero):
        storage, pdf = fake_zotero
        discovered = DiscoveredPDF(pdf, "4F9JKMJE")

        document = PDFTextExtractor().extract(discovered)

        assert document.paper_id.startswith("local:")
        assert document.title == "Multimodal Entity Alignment"
        assert document.authors == ["Alice", "Bob"]
        assert document.year == 2025
        assert document.pages[0].page == 1
        assert "entity alignment" in document.pages[0].text.lower()

    def test_corrupt_pdf_isolated_from_valid_pdf(self, fake_zotero, tmp_path, monkeypatch):
        storage, _ = fake_zotero
        corrupt = storage / "BROKEN01" / "broken.pdf"
        corrupt.parent.mkdir()
        corrupt.write_bytes(b"not a pdf")
        monkeypatch.setenv("LOCAL_RAG_MIN_SCORE", "0.01")
        indexer, _, store = _build_test_retrieval(storage)

        report = indexer.index()

        assert report.indexed_pdf_count == 1
        assert report.failed_pdf_count == 1
        assert store.count >= 1
        assert "broken.pdf" in report.failures[0]["source_path"]

    def test_chunk_metadata_and_repeat_index_are_stable(
        self, fake_zotero, tmp_path
    ):
        storage, pdf = fake_zotero
        before = _file_fingerprint(pdf)
        indexer, retriever, store = _build_test_retrieval(storage)

        first = indexer.index()
        first_count = store.count
        second = indexer.index()
        hits = retriever.retrieve("multimodal entity alignment", top_k=3)

        assert first.indexed_pdf_count == 1
        assert first.created_chunk_count == first_count
        assert second.indexed_pdf_count == 0
        assert second.unchanged_pdf_count == 1
        assert store.count == first_count
        assert hits
        chunk = hits[0].chunk
        assert chunk.source_path == str(pdf)
        assert chunk.page == 1
        assert chunk.zotero_storage_key == "4F9JKMJE"
        assert chunk.paper_id.startswith("local:")
        assert _file_fingerprint(pdf) == before

    def test_changed_pdf_replaces_old_chunks(self, fake_zotero, tmp_path):
        storage, pdf = fake_zotero
        indexer, _, store = _build_test_retrieval(storage)
        indexer.index()

        _write_text_pdf(
            pdf,
            "Updated multimodal entity alignment evidence with graph transformers "
            "and cross-modal contrastive learning.",
            title="Updated Entity Alignment",
        )
        report = indexer.index()

        assert report.indexed_pdf_count == 1
        assert store.document_count == 1
        assert store.count >= 1

    def test_unparseable_pdf_raises_controlled_error(self, tmp_path):
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"broken")
        with pytest.raises(PDFExtractionError):
            PDFTextExtractor().extract(DiscoveredPDF(path, "BROKEN"))

    def test_malformed_pdf_unicode_is_sanitized(self):
        assert "\ud835" not in _clean_pdf_text("title \ud835 suffix")
        assert "title" in _clean_pdf_text("title \ud835 suffix")


class TestLocalPaperSearchTool:
    @pytest.mark.asyncio
    async def test_returns_traceable_related_chunk(
        self, fake_zotero, tmp_path, monkeypatch
    ):
        storage, pdf = fake_zotero
        monkeypatch.setenv("LOCAL_RAG_ENABLED", "true")
        monkeypatch.setenv("LOCAL_RAG_MIN_SCORE", "0.01")
        indexer, _, _ = _install_test_retrieval(monkeypatch, storage)
        indexer.index()

        result = await LocalPaperSearchTool().run(
            query="multimodal entity alignment",
            top_k=5,
        )

        assert result.success
        assert result.metadata["source_type"] == "local"
        assert result.metadata["result_count"] >= 1
        source = result.data["results"][0]
        assert source["source_path"] == str(pdf)
        assert source["page"] == 1
        assert source["zotero_storage_key"] == "4F9JKMJE"
        assert source["full_text"] == source["text"]
        assert source["url"].startswith("file:")
        assert source["source_id"] == source["chunk_id"]
        assert source["paper_id"].startswith(("local:", "doi:"))

    @pytest.mark.asyncio
    async def test_no_result_returns_successful_empty_envelope(
        self, fake_zotero, tmp_path, monkeypatch
    ):
        storage, _ = fake_zotero
        monkeypatch.setenv("LOCAL_RAG_MIN_SCORE", "0.9999")
        indexer, _, _ = _install_test_retrieval(monkeypatch, storage)
        indexer.index()

        result = await LocalPaperSearchTool().run(
            query="unrelated quantum chemistry",
            top_k=5,
        )

        assert result.success
        assert result.data["results"] == []
        assert result.metadata["result_count"] == 0

    @pytest.mark.asyncio
    async def test_invalid_year_range_is_rejected(self):
        result = await LocalPaperSearchTool().run(
            query="agents",
            year_from=2026,
            year_to=2025,
        )
        assert not result.success
        assert "year_from" in result.error

    @pytest.mark.asyncio
    async def test_result_reuses_existing_evidence_and_citation_pipeline(
        self, fake_zotero, tmp_path, monkeypatch
    ):
        storage, _ = fake_zotero
        monkeypatch.setenv("LOCAL_RAG_MIN_SCORE", "0.01")
        indexer, _, _ = _install_test_retrieval(monkeypatch, storage)
        indexer.index()
        search = await LocalPaperSearchTool().run(
            query="entity alignment",
            top_k=1,
        )
        source = search.data["results"][0]
        source["quality_score"] = 0.8

        extracted = await EvidenceExtractTool().run(source=source)
        card = extracted.data["evidence_cards"][0]
        checked = await CitationCheckTool().run(
            citations=[
                {
                    "id": 1,
                    "source_id": card["source_id"],
                    "url": card["url"],
                    "quote": card["quote"],
                }
            ],
            sources=[source],
        )

        assert extracted.success
        assert card["source_id"] == source["chunk_id"]
        assert checked.data["all_valid"] is True

    def test_registry_exposes_tool_without_changing_worker_allowlist(self):
        registry = ToolRegistry.get_instance()
        assert registry.get("local_paper_search") is not None
        assert "local_paper_search" not in registry.list_for_task("search")
        assert "academic_search" in registry.list_for_task("search")


class TestProductionRetrievalAdapters:
    def test_bge_m3_batches_and_normalizes_without_real_download(self, monkeypatch):
        calls = {}

        class FakeModel:
            max_seq_length = 8192

            @staticmethod
            def get_sentence_embedding_dimension():
                return 1024

            @staticmethod
            def encode(texts, **kwargs):
                calls["texts"] = texts
                calls["kwargs"] = kwargs
                return [[0.0] * 1024 for _ in texts]

        monkeypatch.setattr(
            embedding_module,
            "_load_sentence_transformer",
            lambda **kwargs: FakeModel(),
        )
        provider = BGEM3EmbeddingProvider(batch_size=4)

        vectors = provider.embed_documents(["agent", "retrieval"])

        assert len(vectors) == 2
        assert len(vectors[0]) == 1024
        assert calls["kwargs"]["batch_size"] == 4
        assert calls["kwargs"]["normalize_embeddings"] is True

    def test_qdrant_year_and_paper_filters(self, tmp_path):
        storage = tmp_path / "storage"
        _write_text_pdf(
            storage / "A" / "new.pdf",
            "Multimodal agent evaluation compares planning, tool use, safety, "
            "and long-horizon task completion across realistic environments.",
            title="New Paper",
            year=2025,
        )
        _write_text_pdf(
            storage / "B" / "old.pdf",
            "Multimodal agent evaluation compares planning, tool use, safety, "
            "and long-horizon task completion across realistic environments.",
            title="Old Paper",
            year=2020,
        )
        indexer, retriever, store = _build_test_retrieval(storage)
        indexer.index()

        hits = retriever.retrieve(
            "multimodal agent evaluation",
            top_k=5,
            year_from=2024,
        )

        assert store.document_count == 2
        assert hits
        assert {hit.chunk.year for hit in hits} == {2025}

    def test_qdrant_rejects_incompatible_collection_dimension(self):
        client = QdrantClient(":memory:")
        QdrantVectorStore(
            client,
            collection_name="dimension_contract",
            vector_size=64,
            index_version="test-v1",
        )

        with pytest.raises(RuntimeError, match="向量维度"):
            QdrantVectorStore(
                client,
                collection_name="dimension_contract",
                vector_size=1024,
                index_version="test-v2",
            )


class TestPhaseA1TextQuality:
    def test_readable_chinese_is_unicode_and_not_mojibake(self):
        text = "多模态实体对齐通过融合视觉、结构和文本信息识别等价实体。"

        quality = analyze_text_quality(text)

        assert isinstance(text, str)
        assert quality.is_reliable
        assert quality.chinese_char_count > 10
        assert quality.replacement_char_count == 0
        assert quality.suspected_mojibake is False

    def test_common_mojibake_is_rejected_without_guessing_conversion(self):
        quality = analyze_text_quality("ä¸­æ–‡çš„å­¦æœ¯æ–‡æœ¬" * 20)

        assert quality.is_reliable is False
        assert quality.suspected_mojibake is True
        assert "mojibake" in quality.extraction_warning

    def test_unrecoverable_pdf_text_layer_has_explicit_error(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "broken-mapping.pdf"
        path.write_bytes(b"%PDF fake")

        class FakePage:
            @staticmethod
            def extract_text():
                return "ä¸­æ–‡çš„å­¦æœ¯æ–‡æœ¬" * 80

        class FakeReader:
            is_encrypted = False
            metadata = {}
            pages = [FakePage()]

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr("pypdf.PdfReader", FakeReader)

        with pytest.raises(PDFExtractionError, match="不可可靠解析"):
            PDFTextExtractor().extract(DiscoveredPDF(path, "BADTEXT1"))

    @pytest.mark.asyncio
    async def test_chinese_text_is_identical_across_full_retrieval_chain(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "CHINESE1" / "paper.pdf"
        path.parent.mkdir()
        path.write_bytes(b"%PDF fake")
        source_text = (
            "摘要\n"
            "多模态实体对齐通过融合视觉、关系结构与文本描述，"
            "在异构知识图谱之间识别语义等价实体。"
        )

        class FakePage:
            @staticmethod
            def extract_text():
                return source_text

        class FakeReader:
            is_encrypted = False
            metadata = {
                "/Title": "多模态实体对齐研究",
                "/Author": "张三;李四",
                "/CreationDate": "D:20250101",
            }
            pages = [FakePage()]

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr("pypdf.PdfReader", FakeReader)
        document = PDFTextExtractor().extract(
            DiscoveredPDF(path, "CHINESE1")
        )
        chunks = LocalPaperChunker().chunk_document(document)
        embedder = FakeEmbeddingProvider()
        store = QdrantVectorStore(
            QdrantClient(":memory:"),
            collection_name="chinese_chain",
            vector_size=embedder.dimension,
            index_version="v3-section-quality",
        )
        store.replace_document(
            document,
            chunks,
            embedder.embed_documents([chunk.text for chunk in chunks]),
        )
        retriever = LocalPaperRetriever(embedder, store, min_score=0.0)
        stack = (mock.Mock(), retriever, store)
        monkeypatch.setattr(
            local_search_module,
            "build_local_retrieval",
            lambda: stack,
        )

        hits = retriever.retrieve("多模态实体对齐", top_k=5)
        result = await LocalPaperSearchTool().run(
            query="多模态实体对齐",
            top_k=5,
        )

        assert document.pages[0].text == source_text
        assert chunks[0].text == source_text
        assert hits[0].chunk.text == source_text
        assert result.data["results"][0]["text"] == source_text
        assert "多模态实体对齐" in json.dumps(
            result.data,
            ensure_ascii=False,
        )
        assert "\ufffd" not in result.data["results"][0]["text"]


class TestPhaseA1ReferenceSections:
    @staticmethod
    def _document(tmp_path) -> LocalPaperDocument:
        return LocalPaperDocument(
            paper_id="local:section-test",
            title="Section-aware Retrieval",
            authors=["Alice"],
            year=2025,
            doi=None,
            source_path=tmp_path / "section.pdf",
            zotero_storage_key="SECTION1",
            content_hash="section-content-v1",
            modified_ns=1,
            size_bytes=100,
            pages=[
                PaperPage(
                    page=1,
                    text=(
                        "Abstract\n"
                        "Multimodal entity alignment combines visual, relational, "
                        "and textual representations for robust entity matching."
                    ),
                ),
                PaperPage(
                    page=2,
                    text=(
                        "Method\n"
                        "Our method uses cross-modal contrastive learning and "
                        "structure-aware neighborhood aggregation."
                    ),
                ),
                PaperPage(
                    page=3,
                    text=(
                        "References\n"
                        "[1] A. Author. Multimodal Entity Alignment. Journal, 2021.\n"
                        "[2] B. Author. Visual Pivoting. Proceedings, 2022.\n"
                        "[3] C. Author. Graph Matching. Conference, 2023.\n"
                        "[4] D. Author. Contrastive Learning. arXiv, 2024."
                    ),
                ),
            ],
        )

    def test_chunker_marks_explicit_reference_boundary(self, tmp_path):
        chunks = LocalPaperChunker(chunk_size=300).chunk_document(
            self._document(tmp_path)
        )

        body = [chunk for chunk in chunks if not chunk.is_reference_section]
        references = [chunk for chunk in chunks if chunk.is_reference_section]

        assert body
        assert references
        assert all(chunk.section_type == "references" for chunk in references)
        assert all(chunk.total_chunks == len(chunks) for chunk in chunks)

    def test_chunker_splits_inline_chinese_reference_boundary(self, tmp_path):
        document = LocalPaperDocument(
            paper_id="local:inline-reference-test",
            title="行内参考文献边界",
            authors=["Alice"],
            year=2025,
            doi=None,
            source_path=tmp_path / "inline-reference.pdf",
            zotero_storage_key="SECTION2",
            content_hash="inline-reference-v1",
            modified_ns=1,
            size_bytes=100,
            pages=[
                PaperPage(
                    page=1,
                    text=(
                        "5 总结\n"
                        "实验结果说明正文方法能够提高实体对齐准确率，"
                        "并在多个公开数据集上保持稳定的召回率和鲁棒性。"
                        " 参考文献: "
                        "[1] A. Author. Entity Alignment. Journal, 2021.\n"
                        "[2] B. Author. Visual Pivoting. Proceedings, 2022.\n"
                        "[3] C. Author. Graph Matching. Conference, 2023."
                    ),
                ),
                PaperPage(
                    page=2,
                    text=(
                        "[4] D. Author. Contrastive Learning. arXiv, 2024.\n"
                        "[5] E. Author. Knowledge Graph. Journal, 2025."
                    ),
                ),
            ],
        )

        chunks = LocalPaperChunker(chunk_size=300).chunk_document(document)

        body = [chunk for chunk in chunks if not chunk.is_reference_section]
        references = [chunk for chunk in chunks if chunk.is_reference_section]
        assert any("提高实体对齐准确率" in chunk.text for chunk in body)
        assert references
        assert all(chunk.section_type == "references" for chunk in references)
        assert any("[4]" in chunk.text for chunk in references)

    def test_chunker_recognizes_fullwidth_reference_entries(self, tmp_path):
        document = LocalPaperDocument(
            paper_id="local:fullwidth-reference-test",
            title="全角参考文献",
            authors=["Alice"],
            year=2025,
            doi=None,
            source_path=tmp_path / "fullwidth-reference.pdf",
            zotero_storage_key="SECTION3",
            content_hash="fullwidth-reference-v1",
            modified_ns=1,
            size_bytes=100,
            pages=[
                PaperPage(
                    page=1,
                    text=(
                        "本文提出多模态实体对齐方法，并通过消融实验验证"
                        "结构信息和视觉信息的互补作用。"
                    ),
                ),
                PaperPage(
                    page=2,
                    text=(
                        "Ｒｅｆｅｒｅｎｃｅｓ：\n"
                        "［２２］ ＺＨＡＮＧ Ｘ，ＬＶ Ｍ，ｅｔ ａｌ． "
                        "Ｍｕｌｔｉ－ｍｏｄａｌ Ｅｎｔｉｔｙ Ａｌｉｇｎｍｅｎｔ［Ｃ］"
                        "／／Ｐｒｏｃｅｅｄｉｎｇｓ，２０２４．\n"
                        "［２３］ ＬＩＵ Ｆ，ＣＨＥＮ Ｍ，ｅｔ ａｌ． "
                        "Ｖｉｓｕａｌ Ｐｉｖｏｔｉｎｇ［Ｊ］．Ｊｏｕｒｎａｌ，２０２３．\n"
                        "［２４］ ＣＨＥＮ Ｌ，ＬＩ Ｚ，ｅｔ ａｌ． "
                        "Ｇｒａｐｈ Ｍａｔｃｈｉｎｇ［Ｃ］／／Ｃｏｎｆｅｒｅｎｃｅ，２０２２．"
                    ),
                ),
            ],
        )

        chunks = LocalPaperChunker(chunk_size=500).chunk_document(document)

        fullwidth_chunk = next(
            chunk for chunk in chunks if "［２２］" in chunk.text
        )
        assert fullwidth_chunk.is_reference_section is True
        assert fullwidth_chunk.section_type == "references"
        assert fullwidth_chunk.section == "Ｒｅｆｅｒｅｎｃｅｓ："
        assert "［２２］" in fullwidth_chunk.text

    def test_chunker_does_not_treat_formula_numbers_as_references(
        self, tmp_path
    ):
        document = LocalPaperDocument(
            paper_id="local:formula-test",
            title="公式编号不是参考文献",
            authors=["Alice"],
            year=2025,
            doi=None,
            source_path=tmp_path / "formula.pdf",
            zotero_storage_key="SECTION4",
            content_hash="formula-v1",
            modified_ns=1,
            size_bytes=100,
            pages=[
                PaperPage(
                    page=1,
                    text="本文介绍模型结构和训练目标，并给出完整推导过程。",
                ),
                PaperPage(
                    page=2,
                    text=(
                        "3. 实验方法\n"
                        "15. 损失函数用于约束实体表示之间的距离。\n"
                        "16. 指示函数用于统计候选集合中的正确实体。\n"
                        "17. 排名倒数用于计算模型的平均检索性能。"
                    ),
                ),
            ],
        )

        chunks = LocalPaperChunker(chunk_size=500).chunk_document(document)

        formula_chunk = next(
            chunk for chunk in chunks if "损失函数" in chunk.text
        )
        assert formula_chunk.is_reference_section is False

    def test_chunker_does_not_mark_first_page_metadata_as_references(
        self, tmp_path
    ):
        document = LocalPaperDocument(
            paper_id="local:first-page-metadata",
            title="首页元数据不是参考文献",
            authors=["Alice"],
            year=2025,
            doi=None,
            source_path=tmp_path / "first-page.pdf",
            zotero_storage_key="SECTION5",
            content_hash="first-page-v1",
            modified_ns=1,
            size_bytes=100,
            pages=[
                PaperPage(
                    page=1,
                    text=(
                        "收稿日期：2024-06-26；基金项目：62067006。\n"
                        "作者简介：高永杰（1998），硕士研究生。\n"
                        "1. 研究方向为知识图谱与实体对齐。\n"
                        "2. 项目编号为2024CXPT-17。\n"
                        "3. 通讯作者从事人工智能研究。"
                    ),
                ),
                PaperPage(page=2, text="方法\n本文提出新的实体对齐模型。"),
                PaperPage(page=3, text="实验\n模型在公开数据集上获得提升。"),
                PaperPage(
                    page=4,
                    text=(
                        "参考文献\n"
                        "[1] A. Author. Entity Alignment. Journal, 2021.\n"
                        "[2] B. Author. Graph Matching. Conference, 2022."
                    ),
                ),
            ],
        )

        chunks = LocalPaperChunker(chunk_size=500).chunk_document(document)

        first_page = [chunk for chunk in chunks if chunk.page == 1]
        assert first_page
        assert all(not chunk.is_reference_section for chunk in first_page)

    @pytest.mark.asyncio
    async def test_default_search_filters_references_but_explicit_query_keeps_them(
        self, tmp_path, monkeypatch
    ):
        document = self._document(tmp_path)
        chunks = LocalPaperChunker(chunk_size=300).chunk_document(document)
        embedder = FakeEmbeddingProvider()
        store = QdrantVectorStore(
            QdrantClient(":memory:"),
            collection_name="reference_filter",
            vector_size=embedder.dimension,
            index_version="v3-section-quality",
        )
        store.replace_document(
            document,
            chunks,
            embedder.embed_documents([chunk.text for chunk in chunks]),
        )
        retriever = LocalPaperRetriever(embedder, store, min_score=0.0)
        monkeypatch.setattr(
            local_search_module,
            "build_local_retrieval",
            lambda: (mock.Mock(), retriever, store),
        )

        default_result = await LocalPaperSearchTool().run(
            query="multimodal entity alignment method",
            top_k=10,
        )
        reference_result = await LocalPaperSearchTool().run(
            query="show the references and bibliography",
            top_k=10,
        )

        assert default_result.success
        assert default_result.data["results"]
        assert all(
            not item["is_reference_section"]
            for item in default_result.data["results"]
        )
        assert reference_result.metadata["include_references"] is True
        assert any(
            item["is_reference_section"]
            for item in reference_result.data["results"]
        )

    def test_schema_reindex_does_not_mix_old_and_new_points(self, tmp_path):
        document = self._document(tmp_path)
        chunks = LocalPaperChunker(chunk_size=300).chunk_document(document)
        embedder = FakeEmbeddingProvider()
        client = QdrantClient(":memory:")
        old_store = QdrantVectorStore(
            client,
            collection_name="schema_migration",
            vector_size=embedder.dimension,
            index_version="v2",
        )
        old_store.replace_document(
            document,
            chunks,
            embedder.embed_documents([chunk.text for chunk in chunks]),
        )
        new_store = QdrantVectorStore(
            client,
            collection_name="schema_migration",
            vector_size=embedder.dimension,
            index_version="v3-section-quality",
        )
        new_store.replace_document(
            document,
            chunks,
            embedder.embed_documents([chunk.text for chunk in chunks]),
        )

        assert new_store.count == len(chunks)
        assert old_store.count == 0
        points, _ = client.scroll(
            collection_name="schema_migration",
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        assert len(points) == len(chunks)
        assert {
            point.payload["index_version"] for point in points
        } == {"v3-section-quality"}


class TestOpenAlexYearFilters:
    @pytest.mark.asyncio
    async def test_provider_sends_date_filters(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_MAX_RETRIES", "0")
        captured = {}

        def handler(request):
            captured.update(dict(request.url.params))
            return httpx.Response(
                200,
                json={
                    "meta": {"count": 1},
                    "results": [
                        {
                            "id": "https://openalex.org/W123",
                            "display_name": "Recent Agent Evaluation",
                            "publication_year": 2025,
                            "publication_date": "2025-06-01",
                            "abstract_inverted_index": {
                                "agent": [0],
                                "evaluation": [1],
                            },
                            "authorships": [],
                            "primary_location": {},
                            "open_access": {},
                            "type": "article",
                        }
                    ],
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with mock.patch.object(httpx, "AsyncClient", return_value=client):
            result = await OpenAlexSearchProvider().search(
                "agent evaluation",
                5,
                year_from=2025,
                year_to=2026,
            )

        assert result.success
        assert captured["filter"] == (
            "has_abstract:true,"
            "from_publication_date:2025-01-01,"
            "to_publication_date:2026-12-31"
        )
        assert result.data["year_from"] == 2025
        assert result.data["year_to"] == 2026
        assert result.data["results"][0]["publication_date"] == "2025-06-01"

    @pytest.mark.asyncio
    async def test_academic_tool_rejects_reversed_years(self):
        result = await AcademicSearchTool().run(
            query="agents",
            year_from=2026,
            year_to=2025,
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_openalex_exception_isolated(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
        monkeypatch.setenv("OPENALEX_FALLBACK_TO_MOCK", "false")
        provider = OpenAlexSearchProvider()
        with mock.patch.object(
            provider,
            "_search_with_retry",
            side_effect=RuntimeError("network failed"),
        ):
            result = await provider.search("agents", 3, year_from=2025)
        assert not result.success
        assert "unexpected exception" in result.error


def _direct_rule_item(decision, user_request: str):
    """把 Controller 决策折算成 Planner 的 ATOMIC WorkItem（规则模式），用于断言工具选型。"""
    spec = ExecutionSpec(
        request_id="test",
        user_request=user_request,
        intent=decision.intent,
        execution_class=ExecutionClass.ATOMIC,
        execution_route=decision.execution_route,
        research_topic=decision.research_topic,
        metadata={
            "agent_mode": "rule",
            "controller_decision": {"requested_count": decision.requested_count},
        },
    )
    return PlannerAgent().plan(spec).items[0]


class TestPhaseACompatibility:
    @pytest.mark.asyncio
    async def test_recommendation_route_is_unchanged(self):
        decision = await IntentController().decide(
            "推荐 5 篇关于 LLM Agent 的论文",
            max_sources=5,
            agent_mode="rule",
        )
        assert decision.execution_route == "direct_tool"
        # 关键步骤：工具选型已从 Controller 下沉到 Planner 规则计划。
        item = _direct_rule_item(decision, user_request="推荐 5 篇关于 LLM Agent 的论文")
        assert item.allowed_tools == ["semantic_scholar_recommendations"]
        assert item.input_data["limit"] == 5

    @pytest.mark.asyncio
    async def test_citation_route_is_unchanged(self):
        decision = await IntentController().decide(
            "Attention Is All You Need 被哪些论文引用",
            max_sources=5,
            agent_mode="rule",
        )
        item = _direct_rule_item(
            decision, user_request="Attention Is All You Need 被哪些论文引用"
        )
        assert item.allowed_tools == ["semantic_scholar_graph"]
        assert item.input_data["relation"] == "citations"
        assert item.input_data["paper_query"] == "Attention Is All You Need"

    @pytest.mark.asyncio
    async def test_reference_route_is_unchanged(self):
        decision = await IntentController().decide(
            "查询 Attention Is All You Need 的参考文献",
            max_sources=5,
            agent_mode="rule",
        )
        item = _direct_rule_item(
            decision, user_request="查询 Attention Is All You Need 的参考文献"
        )
        assert item.allowed_tools == ["semantic_scholar_graph"]
        assert item.input_data["relation"] == "references"
        assert item.input_data["paper_query"] == "Attention Is All You Need"

    def test_semantic_scholar_capabilities_remain_registered(self):
        names = ToolRegistry.get_instance().list_names()
        assert "semantic_scholar_search" in names
        assert "semantic_scholar_recommendations" in names
        assert "semantic_scholar_graph" in names
