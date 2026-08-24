"""Read-only follow-up operations over an existing generated report."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm.client import get_llm_client
from app.llm.prompts import build_output_language_instruction, build_system_prompt_with_memory


class ReportFollowUpRequest(BaseModel):
    query: str
    report_id: str
    language: str = "zh"
    section: Optional[str] = None
    report_text: str = ""
    evidence_cards: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ReportFollowUpResponse(BaseModel):
    answer: str = ""
    operation_type: str = "explain_concept"
    referenced_section: Optional[str] = None
    referenced_sources: List[str] = Field(default_factory=list)
    referenced_evidence: List[str] = Field(default_factory=list)
    cannot_answer: bool = False
    cannot_answer_reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("operation_type", data.get("operation") or data.get("type") or "explain_concept")
        data.setdefault("referenced_section", data.get("section"))
        data.setdefault(
            "answer",
            data.get("expanded_text") or data.get("explanation")
            or data.get("expanded_section") or data.get("detailed_explanation")
            or data.get("analysis") or data.get("response") or data.get("content")
            or data.get("summary") or data.get("result") or "",
        )
        data.setdefault(
            "referenced_sources",
            data.get("source_ids") or data.get("cited_sources") or [],
        )
        data.setdefault(
            "referenced_evidence",
            data.get("evidence_ids") or data.get("cited_evidence") or [],
        )
        for key, id_keys in (
            ("referenced_sources", ("source_id", "paper_id", "id")),
            ("referenced_evidence", ("evidence_id", "id")),
        ):
            if isinstance(data.get(key), str):
                data[key] = [data[key]]
            if isinstance(data.get(key), list):
                data[key] = [
                    str(item) if not isinstance(item, dict) else next(
                        (str(item.get(candidate)) for candidate in id_keys if item.get(candidate)), ""
                    )
                    for item in data[key]
                ]
        return data


class ReportFollowUpStrategy:
    OPERATIONS = {"expand_section", "trace_evidence", "explain_concept", "fill_gap"}

    async def answer(
        self,
        request: ReportFollowUpRequest,
        *,
        agent_mode: str = "llm",
        memory_prompt: str = "",
        llm_client=None,
    ) -> tuple[ReportFollowUpResponse, dict]:
        operation = self.detect_operation(request.query)
        section_text = self._section_text(request.report_text, request.section)
        if not request.report_text:
            response = ReportFollowUpResponse(
                answer="当前会话中没有可用报告。",
                operation_type=operation,
                referenced_section=request.section,
                cannot_answer=True,
                cannot_answer_reason="The active report could not be loaded.",
            )
            return response, {"success": False, "error": response.cannot_answer_reason}

        payload = {
            "query": request.query,
            "report_id": request.report_id,
            "requested_operation": operation,
            "section": request.section,
            "section_text": section_text,
            "report": request.report_text[:80_000],
            "evidence_cards": request.evidence_cards[:80],
            "sources": [
                {key: source.get(key) for key in ("source_id", "title", "url", "snippet")}
                for source in request.sources[:50]
            ],
        }
        llm_result = {"success": False, "error": "LLM mode disabled"}
        if agent_mode == "llm":
            llm_result = await (llm_client or get_llm_client()).generate_structured(
                # 重要：报告追问沿用首轮研究请求的语言，保证多轮 conversation 体验一致。
                system_prompt=build_system_prompt_with_memory(
                    REPORT_FOLLOW_UP_SYSTEM + "\n\n"
                    + build_output_language_instruction(request.language),
                    memory_prompt,
                ),
                user_prompt=json.dumps(payload, ensure_ascii=False)[:150_000],
                output_schema=ReportFollowUpResponse,
                temperature=0.0,
            )
            if llm_result.get("success"):
                output = llm_result["data"]
                output.operation_type = operation
                output.referenced_section = request.section or output.referenced_section
                valid_sources = {str(item.get("source_id") or "") for item in request.sources}
                valid_evidence = {str(item.get("evidence_id") or "") for item in request.evidence_cards}
                output.referenced_sources = [item for item in output.referenced_sources if item in valid_sources]
                output.referenced_evidence = [item for item in output.referenced_evidence if item in valid_evidence]
                if not output.answer.strip():
                    output.answer = section_text or "现有报告没有足够信息回答这个问题。"
                    output.cannot_answer = not bool(section_text)
                    if output.cannot_answer:
                        output.cannot_answer_reason = "The model returned no answer and no matching section text was available."
                return output, llm_result

        answer = section_text or request.report_text[:1600]
        return ReportFollowUpResponse(
            answer=(
                "基于现有报告，可确认的内容如下：\n" + answer
                if answer else "现有报告没有足够信息回答这个问题。"
            ),
            operation_type=operation,
            referenced_section=request.section,
            referenced_sources=self._citations(answer, request.sources),
            cannot_answer=not bool(answer),
            cannot_answer_reason="No matching report section was found." if not answer else "",
        ), llm_result

    @staticmethod
    def detect_operation(query: str) -> str:
        lowered = query.lower()
        if any(marker in lowered for marker in ("溯源", "引用来自", "证据", "trace", "source")):
            return "trace_evidence"
        if any(marker in lowered for marker in ("空白", "缺口", "覆盖", "gap", "missing")):
            return "fill_gap"
        if any(marker in lowered for marker in ("展开", "扩写", "详细", "expand", "elaborate")):
            return "expand_section"
        return "explain_concept"

    @staticmethod
    def _section_text(report: str, section: Optional[str]) -> str:
        if not section:
            return ""
        pattern = re.compile(
            rf"^#+\s*{re.escape(section)}\s*$\n(.*?)(?=^#+\s|\Z)", re.M | re.S | re.I
        )
        match = pattern.search(report)
        if match:
            return match.group(1).strip()
        position = report.lower().find(section.lower())
        return report[position:position + 2000] if position >= 0 else ""

    @staticmethod
    def _citations(text: str, sources: List[Dict[str, Any]]) -> List[str]:
        cited_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", text or "")}
        return [
            str(source.get("source_id") or "")
            for index, source in enumerate(sources, start=1)
            if index in cited_numbers and source.get("source_id")
        ]


REPORT_FOLLOW_UP_SYSTEM = """Answer a follow-up about an existing research report using
only the supplied report, Evidence Cards, and source metadata. Supported operations:
expand_section, trace_evidence, explain_concept, fill_gap. Never modify the original
report and never invent evidence. Keep referenced source/evidence IDs exact. If the
requested explanation is not supported, set cannot_answer=true. Return only JSON
in this exact shape: {"answer":"grounded answer","operation_type":"expand_section",
"referenced_section":"exact section or null","referenced_sources":["exact source ID"],
"referenced_evidence":["exact evidence ID"],"cannot_answer":false,
"cannot_answer_reason":""}."""
