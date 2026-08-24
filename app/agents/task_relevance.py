"""Deterministic research-task boundary checks for ambiguous academic terms."""

import re
from typing import Any, Dict, List, Tuple


_MMEA_TOPIC_CUES = ("mmea", "multimodal entity alignment", "multi-modal entity alignment", "多模态实体对齐")

_MMEA_EXCLUDED_PRIMARY_TASKS = (
    ("multimodal entity linking", "多模态实体链接"),
    ("knowledge graph completion", "multimodal knowledge graph completion", "知识图谱补全"),
    ("sentiment classification", "sentiment analysis", "情感分类", "情感分析"),
    ("human preference", "preference alignment", "人类偏好对齐"),
    ("retrieval-augmented generation", "retrieval augmented generation", "multimodal rag", "检索增强生成"),
    ("classroom discourse", "课堂话语"),
    ("time series forecasting", "时间序列预测"),
    ("text-attributed graph", "text attributed graph", "文本属性图"),
)


def is_mmea_request(topic: str) -> bool:
    text = str(topic or "").casefold()
    return any(cue in text for cue in _MMEA_TOPIC_CUES)


def classify_source_for_task(topic: str, source: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stable task label; keyword overlap alone never establishes task identity."""
    if not is_mmea_request(topic):
        return {
            "eligible": True,
            "research_task": "topic_default",
            "task_relevance": "core",
            "reason": "No specialized disambiguation profile applies to this topic.",
        }

    title = str(source.get("title") or "")
    abstract = " ".join(str(source.get(key) or "") for key in ("abstract", "snippet", "full_text"))[:8000]
    title_text = re.sub(r"[-–—_/]+", " ", title.casefold())
    body_text = re.sub(r"[-–—_/]+", " ", abstract.casefold())
    combined = f"{title_text} {body_text}"

    for task_cues in _MMEA_EXCLUDED_PRIMARY_TASKS:
        if any(cue in title_text for cue in task_cues):
            return {
                "eligible": False,
                "research_task": task_cues[0],
                "task_relevance": "excluded",
                "reason": f"The paper's primary task is {task_cues[0]}, not multimodal entity alignment.",
            }

    explicit_mmea = (
        "多模态实体对齐" in title
        or bool(re.search(r"\b(?:multi\s*modal|multimodal)\s+(?:knowledge\s+graph\s+)?entity\s+alignment\b", title_text))
        or bool(re.search(r"\bmmea\b", title_text))
    )
    entity_alignment = bool(re.search(r"\bentity\s+alignment\b", title_text)) or "实体对齐" in title
    multimodal_mechanism = bool(re.search(
        r"\b(?:multi\s*modal|multimodal|visual|image|textual|cross\s*modal|modality)\b",
        combined,
    )) or any(cue in combined for cue in ("多模态", "视觉", "图像模态", "文本模态"))

    if explicit_mmea or (entity_alignment and multimodal_mechanism):
        return {
            "eligible": True,
            "research_task": "multimodal_entity_alignment",
            "task_relevance": "core",
            "reason": "The primary paper title defines entity alignment and the source exposes multimodal mechanisms.",
        }

    return {
        "eligible": False,
        "research_task": "adjacent_or_ambiguous",
        "task_relevance": "excluded",
        "reason": (
            "The source does not establish multimodal entity alignment as its primary task; "
            "generic 'multimodal' or 'alignment' overlap is insufficient."
        ),
    }


def filter_sources_for_task(
    topic: str, sources: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter sources before selection and return an audit record for generation details."""
    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for raw in sources:
        source = dict(raw)
        decision = classify_source_for_task(topic, source)
        source.update({
            "research_task": decision["research_task"],
            "task_relevance": decision["task_relevance"],
            "task_relevance_reason": decision["reason"],
        })
        if decision["eligible"]:
            eligible.append(source)
        else:
            rejected.append({
                "source_id": str(source.get("source_id") or ""),
                "title": str(source.get("title") or ""),
                "research_task": decision["research_task"],
                "reason": decision["reason"],
            })
    return eligible, {
        "profile": "multimodal_entity_alignment" if is_mmea_request(topic) else "topic_default",
        "input_count": len(sources),
        "eligible_count": len(eligible),
        "rejected_count": len(rejected),
        "rejected_sources": rejected,
    }
