"""最小且可靠的 PDF 文本、元数据与抽取质量检测。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from app.retrieval.models import DiscoveredPDF, LocalPaperDocument, PaperPage


_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"(?:D:)?((?:19|20)\d{2})")
_COMMON_MOJIBAKE = (
    "ä¸",
    "äº",
    "æ–",
    "æœ",
    "çš",
    "åœ",
    "å­",
    "ï¼",
    "â€™",
    "â€œ",
    "â€",
)
_EXPECTED_LETTER_SCRIPTS = (
    "LATIN",
    "CJK",
    "IDEOGRAPH",
    "GREEK",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
)


class PDFExtractionError(RuntimeError):
    """表示 PDF 无法形成可信的可检索文本。"""


@dataclass(frozen=True)
class TextQualityMetrics:
    """轻量文本质量结果，不承担文本修复。"""

    replacement_char_count: int
    control_char_ratio: float
    printable_char_ratio: float
    chinese_char_count: int
    unexpected_script_ratio: float
    suspected_mojibake: bool
    extraction_warning: Optional[str]
    is_reliable: bool


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _metadata_value(metadata: Any, key: str) -> str:
    if not metadata:
        return ""
    value = metadata.get(key, "") if hasattr(metadata, "get") else ""
    return _clean_pdf_text(str(value or "")).strip()


def _clean_pdf_text(value: str) -> str:
    """保留 Unicode 字符，仅显式标记代理字符和非法控制字符。"""
    if not isinstance(value, str):
        raise TypeError("PDF Parser 输出必须是 Python Unicode str")

    cleaned = []
    for character in value:
        category = unicodedata.category(character)
        if category == "Cs":
            cleaned.append("\ufffd")
        elif category.startswith("C") and character not in "\n\r\t":
            cleaned.append("\ufffd")
        else:
            cleaned.append(character)
    return "".join(cleaned)


def analyze_text_quality(text: str) -> TextQualityMetrics:
    """检测不可读文本，不进行 latin1/utf8 等猜测性转换。"""
    if not isinstance(text, str):
        raise TypeError("文本质量检查只接受 Python Unicode str")

    total = max(len(text), 1)
    replacement_char_count = text.count("\ufffd")
    control_char_count = sum(
        1
        for character in text
        if unicodedata.category(character).startswith("C")
        and character not in "\n\r\t"
    )
    printable_char_count = sum(
        1 for character in text if character.isprintable() or character in "\n\r\t"
    )
    chinese_char_count = sum(1 for character in text if _is_chinese(character))

    letter_count = 0
    unexpected_script_count = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        letter_count += 1
        unicode_name = unicodedata.name(character, "")
        if unicode_name and not any(
            script in unicode_name for script in _EXPECTED_LETTER_SCRIPTS
        ):
            unexpected_script_count += 1

    control_char_ratio = control_char_count / total
    printable_char_ratio = printable_char_count / total
    unexpected_script_ratio = unexpected_script_count / max(letter_count, 1)
    common_mojibake_found = any(marker in text for marker in _COMMON_MOJIBAKE)
    script_corruption = (
        unexpected_script_count >= 12 and unexpected_script_ratio >= 0.08
    )
    suspected_mojibake = common_mojibake_found or script_corruption

    warnings = []
    if replacement_char_count:
        warnings.append(f"包含 {replacement_char_count} 个 Unicode 替换字符")
    if control_char_ratio > 0.01:
        warnings.append(f"控制字符比例过高：{control_char_ratio:.3f}")
    if printable_char_ratio < 0.92:
        warnings.append(f"可打印字符比例过低：{printable_char_ratio:.3f}")
    if common_mojibake_found:
        warnings.append("检测到常见 UTF-8 mojibake 模式")
    if script_corruption:
        warnings.append(
            f"异常文字脚本比例过高：{unexpected_script_ratio:.3f}"
        )

    replacement_ratio = replacement_char_count / total
    is_reliable = not (
        replacement_char_count >= 10
        or replacement_ratio > 0.01
        or control_char_ratio > 0.01
        or printable_char_ratio < 0.92
        or suspected_mojibake
    )
    return TextQualityMetrics(
        replacement_char_count=replacement_char_count,
        control_char_ratio=control_char_ratio,
        printable_char_ratio=printable_char_ratio,
        chinese_char_count=chinese_char_count,
        unexpected_script_ratio=unexpected_script_ratio,
        suspected_mojibake=suspected_mojibake,
        extraction_warning="；".join(warnings) if warnings else None,
        is_reliable=is_reliable,
    )


def _is_chinese(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _extract_doi(values: Iterable[str]) -> Optional[str]:
    for value in values:
        match = _DOI_PATTERN.search(value or "")
        if match:
            return match.group(0).rstrip(".,;)]}").lower()
    return None


def _extract_year(*values: str) -> Optional[int]:
    for value in values:
        match = _YEAR_PATTERN.search(value or "")
        if match:
            return int(match.group(1))
    return None


def _extract_authors(raw: str) -> list[str]:
    if not raw:
        return []
    if ";" in raw:
        candidates = raw.split(";")
    elif " and " in raw.lower():
        candidates = re.split(r"\s+and\s+", raw, flags=re.IGNORECASE)
    else:
        candidates = [raw]
    return [candidate.strip() for candidate in candidates if candidate.strip()]


class PDFTextExtractor:
    """按页抽取 PDF 文本，并把不可恢复的文本层标记为 warning。"""

    def extract(
        self,
        discovered: DiscoveredPDF,
        *,
        content_hash: Optional[str] = None,
    ) -> LocalPaperDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise PDFExtractionError(
                "本地论文索引需要 pypdf，请先安装项目依赖"
            ) from exc

        path = discovered.path
        digest = content_hash or sha256_file(path)
        extraction_warnings = []
        try:
            stat = path.stat()
            with path.open("rb") as stream:
                reader = PdfReader(stream, strict=False)
                if reader.is_encrypted:
                    try:
                        unlocked = reader.decrypt("")
                    except Exception as exc:
                        raise PDFExtractionError("加密 PDF 无法打开") from exc
                    if unlocked == 0:
                        raise PDFExtractionError("加密 PDF 需要密码")

                metadata = reader.metadata or {}
                title = _metadata_value(metadata, "/Title")
                author = _metadata_value(metadata, "/Author")
                subject = _metadata_value(metadata, "/Subject")
                creation_date = _metadata_value(metadata, "/CreationDate")

                pages = []
                first_reliable_page_text = ""
                for page_number, page in enumerate(reader.pages, start=1):
                    try:
                        raw_text = page.extract_text()
                    except Exception as exc:
                        extraction_warnings.append(
                            {
                                "page": page_number,
                                "warning": (
                                    "页面文本抽取失败："
                                    f"{type(exc).__name__}"
                                ),
                            }
                        )
                        continue

                    if raw_text is not None and not isinstance(raw_text, str):
                        raise PDFExtractionError(
                            f"第 {page_number} 页 Parser 返回值不是 Unicode str"
                        )
                    text = _clean_pdf_text(raw_text or "").strip()
                    if not text:
                        extraction_warnings.append(
                            {
                                "page": page_number,
                                "warning": (
                                    "页面没有可提取文本，可能是扫描页或缺少文本层"
                                ),
                            }
                        )
                        continue

                    quality = analyze_text_quality(text)
                    if quality.extraction_warning:
                        extraction_warnings.append(
                            {
                                "page": page_number,
                                "warning": quality.extraction_warning,
                                "suspected_mojibake": quality.suspected_mojibake,
                            }
                        )
                    if quality.is_reliable and not first_reliable_page_text:
                        first_reliable_page_text = text[:5000]
                    pages.append(
                        PaperPage(
                            page=page_number,
                            text=text,
                            replacement_char_count=quality.replacement_char_count,
                            control_char_ratio=quality.control_char_ratio,
                            printable_char_ratio=quality.printable_char_ratio,
                            chinese_char_count=quality.chinese_char_count,
                            suspected_mojibake=quality.suspected_mojibake,
                            extraction_warning=quality.extraction_warning,
                            is_reliable=quality.is_reliable,
                        )
                    )
        except PDFExtractionError:
            raise
        except Exception as exc:
            raise PDFExtractionError(
                f"PDF 解析失败：{type(exc).__name__}"
            ) from exc

        reliable_pages = [page for page in pages if page.is_reliable]
        if not pages:
            raise PDFExtractionError(
                "PDF 没有可提取文本，可能是扫描型 PDF 或缺少字符映射"
            )
        if not reliable_pages:
            details = "；".join(
                str(item.get("warning", "")) for item in extraction_warnings[:3]
            )
            raise PDFExtractionError(
                "PDF 文本层不可可靠解析，未写入向量库"
                + (f"：{details}" if details else "")
            )

        metadata_title_quality = analyze_text_quality(title) if title else None
        if (
            not title
            or title.lower() in {"untitled", "unknown"}
            or (metadata_title_quality and not metadata_title_quality.is_reliable)
        ):
            title = path.stem
        doi = _extract_doi([subject, first_reliable_page_text])
        year = _extract_year(creation_date)
        paper_id = f"doi:{doi}" if doi else f"local:{digest[:24]}"

        return LocalPaperDocument(
            paper_id=paper_id,
            title=title,
            authors=_extract_authors(author),
            year=year,
            doi=doi,
            source_path=path,
            zotero_storage_key=discovered.zotero_storage_key,
            content_hash=digest,
            modified_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
            pages=pages,
            extraction_warnings=extraction_warnings,
        )
