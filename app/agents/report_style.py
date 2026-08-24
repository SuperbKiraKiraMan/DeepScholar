"""Reader-facing Chinese academic report conventions.

This module deliberately separates the user's command text from the title and
section labels printed in the report.  Prompts may be multilingual, but the
formal report structure must stay stable and idiomatic.
"""

import re
from typing import Iterable


def is_chinese(language: str) -> bool:
    return str(language or "").lower().replace("_", "-").startswith("zh")


def is_mmea_topic(topic: str) -> bool:
    text = str(topic or "").lower()
    return any(
        cue in text
        for cue in ("mmea", "multimodal entity alignment", "multi-modal entity alignment", "多模态实体对齐")
    )


def academic_report_title(topic: str, language: str = "zh") -> str:
    """Return a concise academic title instead of echoing the user command."""
    raw = re.sub(r"\s+", " ", str(topic or "").strip()).strip(" ：:，,。")
    if not is_chinese(language):
        return f"Research Review of {raw}" if raw else "Research Review"
    if is_mmea_topic(raw):
        return "多模态实体对齐研究综述"

    cleaned = re.sub(r"^(?:请|帮我|为我)?(?:生成|撰写|写一份|调研|研究)\s*(?:关于)?", "", raw)
    cleaned = re.sub(r"(?:的)?(?:深度)?(?:调研|研究)?报告$", "", cleaned).strip(" ：:，,。")
    return f"{cleaned or raw or '相关主题'}研究综述"


def report_keywords(topic: str, headings: Iterable[str] = (), language: str = "zh") -> str:
    if not is_chinese(language):
        values = [str(topic or "").strip(), "literature review", "methods", "datasets", "evaluation"]
        return "; ".join(item for item in values if item)
    if is_mmea_topic(topic):
        return "多模态实体对齐；知识图谱；多模态融合；数据集；评价指标"

    cleaned_topic = re.sub(r"^(?:请|帮我|生成|调研|研究)(?:关于)?", "", str(topic or "")).strip()
    values = [cleaned_topic]
    for heading in headings:
        text = str(heading or "").strip()
        if text and text not in {"引言", "结论", "局限"} and text not in values:
            values.append(text)
        if len(values) >= 5:
            break
    return "；".join(item for item in values if item) or "学术研究"
