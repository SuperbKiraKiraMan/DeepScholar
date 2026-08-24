"""保留页码、章节和文本质量信息的 Unicode 字符切块实现。"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import replace
from typing import List, Optional, Tuple

from app.retrieval.models import LocalPaperChunk, LocalPaperDocument
from app.retrieval.pdf_parser import analyze_text_quality


_HEADING_PREFIX = r"(?:\d+(?:\.\d+)*|[A-Z])?[\s.\-:：]*"
_SECTION_PATTERNS = (
    (
        "references",
        re.compile(
            rf"^{_HEADING_PREFIX}"
            r"(?:references?|bibliography|参考文献|主要参考文献)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "abstract",
        re.compile(
            rf"^{_HEADING_PREFIX}(?:abstract|摘要)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "introduction",
        re.compile(
            rf"^{_HEADING_PREFIX}(?:introduction|引言|绪论)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "method",
        re.compile(
            rf"^{_HEADING_PREFIX}"
            r"(?:methods?|methodology|proposed method|approach|方法|研究方法)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "experiment",
        re.compile(
            rf"^{_HEADING_PREFIX}"
            r"(?:experiments?|experimental setup|evaluation|实验|实验设置)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "result",
        re.compile(
            rf"^{_HEADING_PREFIX}"
            r"(?:results?(?: and discussion)?|discussion|结果|结果与分析)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        re.compile(
            rf"^{_HEADING_PREFIX}"
            r"(?:conclusions?|summary|总结|结论)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "appendix",
        re.compile(
            rf"^{_HEADING_PREFIX}(?:appendix|appendices|附录)(?:\s+[A-Z])?\s*$",
            re.IGNORECASE,
        ),
    ),
)
_NUMBERED_REFERENCE = re.compile(
    r"(?:^|\n)\s*(?:\[\d{1,3}\]|\d{1,3}[.)])\s+"
)
_REFERENCE_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_REFERENCE_MARKER = re.compile(
    r"\b(?:doi|vol\.?|volume|pp\.?|pages?|proceedings|journal|conference|"
    r"arxiv|et\s+al\.?)\b",
    re.IGNORECASE,
)
_AUTHOR_YEAR = re.compile(
    r"\b[A-Z][A-Za-z'’-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'’-]+)?"
    r"\s*[,([]\s*(?:19|20)\d{2}"
)
_INLINE_CHINESE_REFERENCE_HEADING = re.compile(
    r"(?<![\u4e00-\u9fffA-Za-z])"
    r"(?P<heading>(?:主要)?参考文献)\s*[:：]"
)


class LocalPaperChunker:
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        min_chunk_size: int = 40,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须位于 0 和 chunk_size 之间")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: LocalPaperDocument) -> List[LocalPaperChunk]:
        chunks: List[LocalPaperChunk] = []
        chunk_index = 0
        current_section_type = "unknown"
        current_heading: Optional[str] = None
        total_pages = max((page.page for page in document.pages), default=0)

        for page in document.pages:
            if not page.is_reliable:
                continue
            segments, current_section_type, current_heading = self._split_sections(
                page.text,
                current_section_type=current_section_type,
                current_heading=current_heading,
            )
            for section_type, heading, segment_text in segments:
                if (
                    section_type != "references"
                    and self._looks_like_reference_block(
                        segment_text,
                        page_number=page.page,
                        total_pages=total_pages,
                    )
                ):
                    section_type = "references"
                    heading = heading or "References（模式识别）"

                for text in self._split_text(segment_text):
                    clean = text.strip()
                    if len(clean) < self.min_chunk_size:
                        continue
                    chunk_quality = analyze_text_quality(clean)
                    if (
                        not chunk_quality.is_reliable
                        or chunk_quality.replacement_char_count > 0
                    ):
                        continue
                    raw_id = (
                        f"{document.paper_id}|{document.content_hash}|"
                        f"{document.source_path}|{page.page}|{chunk_index}"
                    )
                    chunk_id = hashlib.sha256(
                        raw_id.encode("utf-8")
                    ).hexdigest()[:24]
                    chunks.append(
                        LocalPaperChunk(
                            chunk_id=chunk_id,
                            paper_id=document.paper_id,
                            title=document.title,
                            page=page.page,
                            text=clean,
                            source_path=str(document.source_path),
                            zotero_storage_key=document.zotero_storage_key,
                            content_hash=document.content_hash,
                            chunk_index=chunk_index,
                            year=document.year,
                            section=heading,
                            section_type=section_type,
                            is_reference_section=section_type == "references",
                            extraction_warning=chunk_quality.extraction_warning,
                        )
                    )
                    chunk_index += 1

        total_chunks = len(chunks)
        return [replace(chunk, total_chunks=total_chunks) for chunk in chunks]

    def _split_sections(
        self,
        text: str,
        *,
        current_section_type: str,
        current_heading: Optional[str],
    ) -> Tuple[List[Tuple[str, Optional[str], str]], str, Optional[str]]:
        segments: List[Tuple[str, Optional[str], str]] = []
        buffer: List[str] = []
        section_type = current_section_type
        heading = current_heading
        # 中文 PDF 常把“参考文献:”粘在结论末行，先恢复明确的章节边界。
        text = _INLINE_CHINESE_REFERENCE_HEADING.sub(
            lambda match: (
                f"\n{match.group('heading')}\n"
            ),
            text,
        )

        def flush() -> None:
            segment_text = "\n".join(buffer).strip()
            if segment_text:
                segments.append((section_type, heading, segment_text))
            buffer.clear()

        for line in text.splitlines():
            detected = self._classify_heading(line)
            if detected is not None:
                flush()
                section_type = detected
                heading = line.strip()
            buffer.append(line)
        flush()

        if not segments and text.strip():
            segments.append((section_type, heading, text.strip()))
        return segments, section_type, heading

    @staticmethod
    def _classify_heading(line: str) -> Optional[str]:
        # 仅对分类副本做 NFKC，保留写入 Evidence 的原始 PDF 文本。
        candidate = (
            unicodedata.normalize("NFKC", line)
            .strip()
            .rstrip(":：")
            .strip()
        )
        if not candidate or len(candidate) > 100:
            return None
        for section_type, pattern in _SECTION_PATTERNS:
            if pattern.fullmatch(candidate):
                return section_type
        return None

    @staticmethod
    def _looks_like_reference_block(
        text: str,
        *,
        page_number: int,
        total_pages: int,
    ) -> bool:
        # 全角编号、拉丁字母和数字常见于中文期刊 PDF 的参考文献区。
        normalized = unicodedata.normalize("NFKC", text)
        numbered_entries = len(_NUMBERED_REFERENCE.findall(normalized))
        year_mentions = len(_REFERENCE_YEAR.findall(normalized))
        publication_markers = len(_REFERENCE_MARKER.findall(normalized))
        author_year_entries = len(_AUTHOR_YEAR.findall(normalized))
        late_in_document = (
            total_pages >= 2
            and page_number >= math.ceil(total_pages * 0.6)
        )

        score = 0
        if numbered_entries >= 3:
            score += 3
        elif numbered_entries >= 2:
            score += 2
        if author_year_entries >= 3:
            score += 2
        if year_mentions >= 4:
            score += 1
        if publication_markers >= 3:
            score += 1
        if late_in_document:
            score += 1

        has_bibliographic_signals = (
            author_year_entries >= 2
            or (year_mentions >= 2 and publication_markers >= 1)
            or publication_markers >= 3
        )
        has_reference_structure = (
            numbered_entries >= 2 and has_bibliographic_signals
        ) or author_year_entries >= 3
        return (
            late_in_document
            and has_reference_structure
            and score >= 4
        )

    def _split_text(self, text: str) -> List[str]:
        # Python 字符串切片按 Unicode code point 工作，不按 UTF-8 byte 切分。
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n", text)
            if part.strip()
        ]
        merged: List[str] = []
        current = ""
        for paragraph in paragraphs:
            if not current:
                current = paragraph
            elif len(current) + len(paragraph) + 2 <= self.chunk_size:
                current = f"{current}\n\n{paragraph}"
            else:
                merged.extend(self._window(current))
                current = paragraph
        if current:
            merged.extend(self._window(current))
        return merged

    def _window(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return chunks
