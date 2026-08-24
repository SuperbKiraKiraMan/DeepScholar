"""Evidence-grounded comparison of two papers already present in a session."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator

from app.llm.client import get_llm_client
from app.llm.prompts import build_output_language_instruction, build_system_prompt_with_memory


class ComparisonDimension(BaseModel):
    dimension: str = ""
    paper_a: str = ""
    paper_b: str = ""
    analysis: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("dimension", data.get("name") or data.get("aspect") or "")
        data.setdefault("paper_a", data.get("paper_a_evidence") or data.get("a") or "")
        data.setdefault("paper_b", data.get("paper_b_evidence") or data.get("b") or "")
        data.setdefault("analysis", data.get("comparison") or data.get("difference") or "")
        return data


class PaperCompareResult(BaseModel):
    paper_a_id: str = ""
    paper_a_title: str = ""
    paper_b_id: str = ""
    paper_b_title: str = ""
    comparison_dimensions: List[ComparisonDimension] = Field(default_factory=list)
    summary: str = ""
    evidence_citations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        nested = data.get("comparison")
        if isinstance(nested, dict):
            nested.update({key: item for key, item in data.items() if key != "comparison"})
            data = nested
        dimensions = data.get("comparison_dimensions") or data.get("dimensions")
        if isinstance(dimensions, dict):
            data["comparison_dimensions"] = [
                {"dimension": key, **(item if isinstance(item, dict) else {"analysis": str(item)})}
                for key, item in dimensions.items()
            ]
        elif dimensions is not None:
            data["comparison_dimensions"] = dimensions
        data.setdefault("summary", data.get("overall_summary") or data.get("conclusion") or "")
        citations = data.get("evidence_citations")
        if isinstance(citations, list):
            data["evidence_citations"] = [
                str(item) if not isinstance(item, dict) else str(
                    item.get("citation") or item.get("paper_id") or item.get("source_id") or ""
                )
                for item in citations
                if not isinstance(item, dict) or any(item.get(key) for key in ("citation", "paper_id", "source_id"))
            ]
        return data


class PaperCompareStrategy:
    async def compare(
        self,
        paper_a: Dict[str, Any],
        paper_b: Dict[str, Any],
        question: str,
        *,
        evidence_cards: List[Dict[str, Any]] | None = None,
        agent_mode: str = "llm",
        language: str = "zh",
        memory_prompt: str = "",
        llm_client=None,
    ) -> tuple[PaperCompareResult, dict]:
        a_id = self._paper_id(paper_a)
        b_id = self._paper_id(paper_b)
        cards = evidence_cards or []
        payload = {
            "question": question,
            "paper_a": self._paper_payload(paper_a),
            "paper_b": self._paper_payload(paper_b),
            "evidence_cards": [
                card for card in cards if card.get("source_id") in {a_id, b_id}
            ],
        }
        llm_result = {"success": False, "error": "LLM mode disabled"}
        if agent_mode == "llm":
            llm_result = await (llm_client or get_llm_client()).generate_structured(
                system_prompt=build_system_prompt_with_memory(
                    build_paper_compare_system_prompt(language), memory_prompt
                ),
                user_prompt=json.dumps(payload, ensure_ascii=False)[:140_000],
                output_schema=PaperCompareResult,
                temperature=0.0,
            )
            if llm_result.get("success"):
                output = llm_result["data"]
                # IDs and titles are runtime-owned, not model-owned.
                output.paper_a_id = a_id
                output.paper_a_title = str(paper_a.get("title") or "")
                output.paper_b_id = b_id
                output.paper_b_title = str(paper_b.get("title") or "")
                output.evidence_citations = [
                    citation for citation in output.evidence_citations
                    if a_id in citation or b_id in citation
                ]
                return output, llm_result

        return self._rule_compare(paper_a, paper_b, language), llm_result

    def _rule_compare(
        self,
        paper_a: Dict[str, Any],
        paper_b: Dict[str, Any],
        language: str = "zh",
    ) -> PaperCompareResult:
        is_zh = str(language or "zh").lower().replace("_", "-").startswith("zh")
        dimension_names = (
            {
                "problem": "研究问题",
                "method": "方法",
                "results": "结果",
                "limitations": "局限性",
                "novelty": "创新点",
            }
            if is_zh else {
                "problem": "problem",
                "method": "method",
                "results": "results",
                "limitations": "limitations",
                "novelty": "novelty",
            }
        )
        both_available = (
            "两边均有当前会话提供的材料。"
            if is_zh else "Both sides contain supplied material for this dimension."
        )
        missing_evidence = (
            "至少一方缺少当前会话提供的证据。"
            if is_zh else "At least one side lacks supplied evidence."
        )
        # 规则回退也遵循同一语言契约，避免 LLM 不可用时混入英文 UI 文本。
        dimensions = []
        for dimension, keys in {
            "problem": ("research_task", "snippet"),
            "method": ("method", "full_text", "snippet"),
            "results": ("result", "snippet"),
            "limitations": ("limitation",),
            "novelty": ("key_contribution", "snippet"),
        }.items():
            a_text = self._first(paper_a, keys)
            b_text = self._first(paper_b, keys)
            dimensions.append(ComparisonDimension(
                dimension=dimension_names[dimension],
                paper_a=a_text[:600] or ("当前证据中未提供" if is_zh else "Not available in supplied evidence"),
                paper_b=b_text[:600] or ("当前证据中未提供" if is_zh else "Not available in supplied evidence"),
                analysis=both_available if a_text and b_text else missing_evidence,
            ))
        return PaperCompareResult(
            paper_a_id=self._paper_id(paper_a),
            paper_a_title=str(paper_a.get("title") or ""),
            paper_b_id=self._paper_id(paper_b),
            paper_b_title=str(paper_b.get("title") or ""),
            comparison_dimensions=dimensions,
            summary=(
                "这是基于当前会话已有文本的保守对比；缺失维度已明确标注。"
                if is_zh else
                "This is a conservative comparison based on text available in the current session; missing dimensions are marked explicitly."
            ),
            evidence_citations=[self._paper_id(paper_a), self._paper_id(paper_b)],
        )

    @staticmethod
    def _paper_payload(paper: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value for key, value in paper.items()
            if key not in {"embedding"}
        }

    @staticmethod
    def _paper_id(paper: Dict[str, Any]) -> str:
        return str(paper.get("source_id") or paper.get("paper_id") or "")

    @staticmethod
    def _first(paper: Dict[str, Any], keys: tuple[str, ...]) -> str:
        return next((str(paper.get(key) or "") for key in keys if paper.get(key)), "")


PAPER_COMPARE_SYSTEM = """Compare exactly two supplied academic papers. Use the
dimensions problem, method, results, limitations, and novelty, plus a user-requested
dimension when needed. Every comparison point must state the evidence for both sides;
if one side is absent, explicitly say it is unavailable. Do not use outside knowledge.
Evidence citations must contain an exact paper ID (and evidence ID when available).
Return only JSON in this shape:
{"paper_a_id":"exact ID","paper_a_title":"title","paper_b_id":"exact ID",
"paper_b_title":"title","comparison_dimensions":[{"dimension":"method",
"paper_a":"evidence for A","paper_b":"evidence for B","analysis":"comparison"}],
"summary":"overall answer","evidence_citations":["paper ID: quoted evidence"],"warnings":[]}.
Include all five required dimensions."""

# 默认系统提示直接包含简体中文约束，便于静态检查和未经过 builder 的调用入口也能安全工作。
PAPER_COMPARE_SYSTEM += "\n\n" + build_output_language_instruction("zh")


def build_paper_compare_system_prompt(language: str = "zh") -> str:
    """将语言契约追加到论文对比系统提示，确保所有调用入口一致。"""
    default_instruction = build_output_language_instruction("zh")
    requested_instruction = build_output_language_instruction(language)
    if requested_instruction == default_instruction:
        return PAPER_COMPARE_SYSTEM
    return PAPER_COMPARE_SYSTEM.replace(default_instruction, requested_instruction, 1)
