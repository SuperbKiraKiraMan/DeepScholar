"""Paper-title localization helpers used only for display.

The original title and source_id remain the authoritative traceability fields.
Translated titles are presentation metadata and are never treated as evidence.
"""

import re
from typing import Any, Dict, Mapping


_CJK = re.compile(r"[\u3400-\u9fff]")


def is_chinese_title(title: Any) -> bool:
    """Return whether a title already contains Chinese characters."""
    return bool(_CJK.search(str(title or "")))


def validated_title_translations(
    raw: Any,
    sources: list[Dict[str, Any]],
) -> Dict[str, str]:
    """Keep only translations bound to supplied sources and containing Chinese."""
    if not isinstance(raw, Mapping):
        return {}
    valid_ids = {str(source.get("source_id") or "") for source in sources}
    result: Dict[str, str] = {}
    for source_id, translated in raw.items():
        key = str(source_id or "").strip()
        value = str(translated or "").strip()
        if key in valid_ids and value and is_chinese_title(value):
            result[key] = value
    return result


def display_paper_title(
    source: Mapping[str, Any],
    translations: Mapping[str, str] | None = None,
) -> str:
    """Render Chinese titles unchanged and English titles as 中文名（English）.

    If no trustworthy Chinese translation is available, retain the original
    title instead of inventing one.  LLM chapter generation supplies the normal
    translation mapping; imported/local sources may also carry one already.
    """
    original = str(source.get("title") or "Unknown").strip()
    if not original or is_chinese_title(original):
        return original or "Unknown"

    source_id = str(source.get("source_id") or "")
    candidates = [
        (translations or {}).get(source_id),
        source.get("title_zh"),
        source.get("chinese_title"),
        source.get("translated_title_zh"),
    ]
    translated = next(
        (str(item).strip() for item in candidates if item and is_chinese_title(item)),
        "",
    )
    if not translated:
        return original
    if original.casefold() in translated.casefold():
        return translated
    return f"{translated}（{original}）"


def safe_source_url(source: Mapping[str, Any]) -> str:
    """Return a reproducible public/internal identifier, never a local file path."""
    url = str(source.get("url") or "").strip()
    if url.lower().startswith("file://"):
        url = ""
    if url.startswith(("http://", "https://")):
        return url
    doi = str(source.get("doi") or "").strip()
    if doi:
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        return f"https://doi.org/{doi}"
    openalex_id = str(source.get("openalex_id") or "").strip()
    if openalex_id:
        return openalex_id if openalex_id.startswith("http") else f"https://openalex.org/{openalex_id}"
    semantic_id = str(source.get("semantic_scholar_id") or "").strip()
    if semantic_id:
        return f"https://www.semanticscholar.org/paper/{semantic_id}"
    return ""


def source_evidence_role(source: Mapping[str, Any], evidence_types: set[str] | None = None) -> str:
    """Label original research versus secondary/review evidence for readers."""
    title = str(source.get("title") or "").lower()
    source_type = str(source.get("source_type") or "").lower()
    types = evidence_types or set()
    if "secondary_summary" in types:
        return "二手概述"
    if source_type in {"survey", "review"} or any(term in title for term in ("survey", "review", "taxonomy")):
        return "综述"
    return "原始研究"


def format_reference(
    index: int,
    source: Mapping[str, Any],
    translations: Mapping[str, str] | None = None,
    evidence_types: set[str] | None = None,
) -> str:
    """Build a metadata-complete reference line without exposing local paths."""
    authors = source.get("authors") or []
    if isinstance(authors, list):
        author_text = ", ".join(str(author) for author in authors[:6])
        if len(authors) > 6:
            author_text += ", et al."
    else:
        author_text = str(authors)
    author_text = author_text or "作者信息缺失"
    year = source.get("year") or "年份缺失"
    venue = str(source.get("venue") or "").strip()
    source_type = str(source.get("source_type") or "unknown").strip()
    role = source_evidence_role(source, evidence_types)
    publication = venue or source_type
    doi = str(source.get("doi") or "").strip()
    identifiers = f" DOI: {doi}." if doi else ""
    url = safe_source_url(source)
    link = f" — {url}" if url else ""
    return (
        f"- [{index}] {author_text}. ({year}). "
        f"{display_paper_title(source, translations)}. {publication}. [{role}]."
        f"{identifiers}{link}"
    )
