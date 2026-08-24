"""Grounded question answering over one active paper."""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm.client import get_llm_client
from app.llm.prompts import build_output_language_instruction, build_system_prompt_with_memory


class PaperQARequest(BaseModel):
    question: str
    paper_id: str
    language: str = "zh"
    paper_full_text: str = ""
    paper_abstract: str = ""
    paper_title: str = ""


class PaperQAResponse(BaseModel):
    answer: str
    supporting_quotes: List[str] = Field(default_factory=list)
    quote_sections: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cannot_answer: bool = False
    cannot_answer_reason: str = ""
    paper_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_quotes(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        quotes = data.get("supporting_quotes") or []
        normalized = []
        sections = list(data.get("quote_sections") or [])
        for item in quotes:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                quote = item.get("quote") or item.get("text") or item.get("content")
                if quote:
                    normalized.append(str(quote))
                    if item.get("section"):
                        sections.append(str(item["section"]))
        data["supporting_quotes"] = normalized
        data["quote_sections"] = sections
        return data


class PaperQAStrategy:
    async def answer(
        self,
        request: PaperQARequest,
        *,
        agent_mode: str = "llm",
        memory_prompt: str = "",
        llm_client=None,
    ) -> tuple[PaperQAResponse, dict]:
        text = (request.paper_full_text or request.paper_abstract or "").strip()
        if not text:
            response = PaperQAResponse(
                answer="当前会话没有这篇论文的正文或摘要，无法可靠回答。",
                cannot_answer=True,
                cannot_answer_reason="No paper text is available in the session.",
                paper_id=request.paper_id,
            )
            return response, {"success": False, "error": response.cannot_answer_reason}

        llm_result = {"success": False, "error": "LLM mode disabled"}
        if agent_mode == "llm":
            client = llm_client or get_llm_client()
            llm_result = await client.generate_structured(
                # 重要：单篇问答也必须接收同一语言约束，避免 conversation 分支输出语言不一致。
                system_prompt=build_system_prompt_with_memory(
                    PAPER_QA_SYSTEM + "\n\n" + build_output_language_instruction(request.language),
                    memory_prompt,
                ),
                user_prompt=(
                    f"Paper ID: {request.paper_id}\nTitle: {request.paper_title}\n"
                    f"Question: {request.question}\n\nPaper text:\n{text[:80_000]}"
                ),
                output_schema=PaperQAResponse,
                temperature=0.0,
            )
            if llm_result.get("success"):
                output = llm_result["data"]
                output.paper_id = request.paper_id
                if not output.supporting_quotes:
                    output.supporting_quotes = [
                        value.strip()
                        for value in re.findall(r'["“](.*?)["”]', output.answer, re.S)
                        if value.strip()
                    ]
                output.supporting_quotes = [
                    quote for quote in output.supporting_quotes if self._quote_grounded(quote, text)
                ]
                if not output.supporting_quotes:
                    derived_quote = self._derive_grounding_quote(
                        output.answer, request.question, text
                    )
                    if derived_quote:
                        output.supporting_quotes = [derived_quote]
                        output.quote_sections = output.quote_sections or ["supplied_text"]
                        output.confidence = max(output.confidence, 0.65)
                if not output.cannot_answer and not output.supporting_quotes:
                    output.cannot_answer = True
                    output.cannot_answer_reason = "The model answer had no verifiable quote in the supplied text."
                    output.confidence = min(output.confidence, 0.3)
                return output, llm_result

        return self._rule_answer(request, text), llm_result

    @staticmethod
    def _quote_grounded(quote: str, text: str) -> bool:
        normalized_quote = re.sub(r"\s+", " ", quote.strip()).lower()
        normalized_text = re.sub(r"\s+", " ", text).lower()
        return bool(normalized_quote) and normalized_quote in normalized_text

    @staticmethod
    def _derive_grounding_quote(answer: str, question: str, text: str) -> str:
        """Recover an exact source sentence when a provider paraphrases its quote field."""
        sentences = [
            item.strip() for item in re.split(r"(?<=[。！？.!?])\s+|\n+", text)
            if len(item.strip()) >= 20
        ]
        terms = {
            term for term in re.findall(r"[a-z][a-z0-9_-]{3,}", (answer + " " + question).lower())
            if term not in {"this", "that", "with", "from", "paper", "method"}
        }
        markers = []
        lowered = question.lower()
        if any(item in lowered for item in ("方法", "method", "使用", "采用")):
            markers.extend(["methods", "system", "retrieval", "encoder"])
        if any(item in lowered for item in ("局限", "limitation")):
            markers.extend(["limitations", "does not", "evaluation"])
        if any(item in lowered for item in ("结果", "result", "指标")):
            markers.extend(["results", "accuracy", "percent"])
        ranked = sorted(
            sentences,
            key=lambda sentence: (
                sum(term in sentence.lower() for term in terms),
                sum(marker in sentence.lower() for marker in markers),
                len(sentence),
            ),
            reverse=True,
        )
        if not ranked:
            return ""
        best = ranked[0]
        score = sum(term in best.lower() for term in terms) + sum(
            marker in best.lower() for marker in markers
        )
        return best[:800] if score > 0 else ""

    def _rule_answer(self, request: PaperQARequest, text: str) -> PaperQAResponse:
        sentences = [item.strip() for item in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if item.strip()]
        terms = set(re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", request.question.lower()))
        ranked = sorted(
            sentences,
            key=lambda sentence: sum(term in sentence.lower() for term in terms),
            reverse=True,
        )
        quotes = [sentence[:500] for sentence in ranked if sentence][:2]
        if not quotes or not any(term in " ".join(quotes).lower() for term in terms):
            return PaperQAResponse(
                answer="提供的论文文本不足以回答这个问题。",
                cannot_answer=True,
                cannot_answer_reason="No relevant passage was found in the supplied paper text.",
                paper_id=request.paper_id,
            )
        return PaperQAResponse(
            answer="根据论文原文，可定位到以下相关信息：" + " ".join(quotes),
            supporting_quotes=quotes,
            quote_sections=["supplied_text"] * len(quotes),
            confidence=0.65,
            paper_id=request.paper_id,
        )


PAPER_QA_SYSTEM = """You answer questions about exactly one academic paper using only
the supplied paper text. Every factual claim in answer must be supported by a verbatim
supporting quote copied from that text, with its section when visible. If the text does
not contain the answer, set cannot_answer=true and explain why. Do not use outside
knowledge. Return JSON matching PaperQAResponse; keep paper_id exact."""
