"""本地论文检索流水线共享的数据模型。"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DiscoveredPDF:
    path: Path
    zotero_storage_key: str


@dataclass
class DiscoveryReport:
    storage_path: Path
    scanned_directory_count: int = 0
    discovered_pdfs: List[DiscoveredPDF] = field(default_factory=list)
    skipped_no_pdf_directory_count: int = 0
    skipped_html_snapshot_count: int = 0
    skipped_hidden_or_temporary_count: int = 0

    @property
    def discovered_pdf_count(self) -> int:
        return len(self.discovered_pdfs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_path": str(self.storage_path),
            "scanned_directory_count": self.scanned_directory_count,
            "discovered_pdf_count": self.discovered_pdf_count,
            "skipped_no_pdf_directory_count": self.skipped_no_pdf_directory_count,
            "skipped_html_snapshot_count": self.skipped_html_snapshot_count,
            "skipped_hidden_or_temporary_count": self.skipped_hidden_or_temporary_count,
        }


@dataclass(frozen=True)
class PaperPage:
    page: int
    text: str
    replacement_char_count: int = 0
    control_char_ratio: float = 0.0
    printable_char_ratio: float = 1.0
    chinese_char_count: int = 0
    suspected_mojibake: bool = False
    extraction_warning: Optional[str] = None
    is_reliable: bool = True


@dataclass
class LocalPaperDocument:
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    source_path: Path
    zotero_storage_key: str
    content_hash: str
    modified_ns: int
    size_bytes: int
    pages: List[PaperPage]
    extraction_warnings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LocalPaperChunk:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    source_path: str
    zotero_storage_key: str
    content_hash: str
    chunk_index: int
    year: Optional[int] = None
    section: Optional[str] = None
    section_type: str = "unknown"
    is_reference_section: bool = False
    extraction_warning: Optional[str] = None
    total_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchHit:
    chunk: LocalPaperChunk
    score: float


@dataclass
class IndexReport:
    discovery: DiscoveryReport
    indexed_pdf_count: int = 0
    unchanged_pdf_count: int = 0
    failed_pdf_count: int = 0
    created_chunk_count: int = 0
    failures: List[Dict[str, str]] = field(default_factory=list)
    warning_count: int = 0
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.discovery.to_dict(),
            "indexed_pdf_count": self.indexed_pdf_count,
            "unchanged_pdf_count": self.unchanged_pdf_count,
            "failed_pdf_count": self.failed_pdf_count,
            "created_chunk_count": self.created_chunk_count,
            "failures": list(self.failures),
            "warning_count": self.warning_count,
            "warnings": list(self.warnings),
        }
