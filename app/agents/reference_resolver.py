"""把多轮对话中的论文/报告指代解析为具体的 Session 资源。"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from app.llm.client import get_llm_client
from app.services.session_store import SessionContext


class ReferenceResolution(BaseModel):
    reference_expression: str = ""
    resolved_paper_ids: List[str] = Field(default_factory=list)
    resolved_section: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_used: bool = False
    missing_ordinal: Optional[int] = None
    reasoning: str = ""


class _LLMReferenceOutput(BaseModel):
    reference_expression: str = ""
    resolved_paper_ids: List[str] = Field(default_factory=list)
    resolved_section: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""

# 作用:把模糊的指代转化为具体的 Session 资源 ID 或章节名
class ReferenceResolver:
    """常见指代由规则解析,语义模糊的表达才回退 LLM。"""

    async def resolve(
        self,
        query: str,
        session_context: SessionContext,
        *,
        llm_client=None,
        allow_llm: bool = True,
    ) -> ReferenceResolution:
        # 规则优先:置信度足够高(或禁用 LLM)就直接采用,不付 LLM 成本
        rule = self._rule_resolve(query, session_context)
        # 关键步骤：明确的越界序号不能交给 LLM 猜测，也不能被后续流程降级为深度调研。
        if rule.missing_ordinal is not None:
            return rule
        if rule.confidence >= 0.7 or not allow_llm:
            return rule

        # 规则未命中 → 把 session 候选论文打包成清单,交给 LLM 做语义解析。
        # 候选按“最近批次优先 + 显示编号”构造，使 LLM 看到的编号与屏幕上一致。
        candidates = self._candidates(session_context)
        result = await (llm_client or get_llm_client()).generate_structured(
            system_prompt=_REFERENCE_SYSTEM,
            user_prompt=json.dumps({
                "query": query,
                "papers": candidates,
                "active_paper_id": session_context.active_paper_id,
                "recent_paper_ids": session_context.last_mentioned_paper_ids,
                "active_report_id": session_context.active_report_id,
                "report_sections": session_context.last_report_sections,
            }, ensure_ascii=False),
            output_schema=_LLMReferenceOutput,
            temperature=0.0,
        )
        if not result.get("success"):
            return rule
        output = result["data"]
        # 安全阀:LLM 返回的论文 ID 必须真实存在于 session,防止编造
        valid_ids = {self._paper_id(item) for item in session_context.recommended_papers}
        valid_ids.update(session_context.last_mentioned_paper_ids)
        resolved = [paper_id for paper_id in output.resolved_paper_ids if paper_id in valid_ids]
        # 章节也须精确匹配报告章节,否则回退规则结果
        valid_sections = set(session_context.last_report_sections)
        section = output.resolved_section if output.resolved_section in valid_sections else rule.resolved_section
        return ReferenceResolution(
            reference_expression=output.reference_expression or rule.reference_expression,
            resolved_paper_ids=list(dict.fromkeys(resolved or rule.resolved_paper_ids)),
            resolved_section=section,
            confidence=output.confidence if (resolved or section) else rule.confidence,
            fallback_used=True,
            missing_ordinal=rule.missing_ordinal,
            reasoning=output.reasoning or rule.reasoning,
        )

    def _rule_resolve(
        self, query: str, session: SessionContext
    ) -> ReferenceResolution:
        papers = session.recommended_papers
        ids: List[str] = []
        expressions: List[str] = []

        # 1) 直接命中来源 ID 或精确标题
        for paper in papers:
            paper_id = self._paper_id(paper)
            title = str(paper.get("title") or "").strip()
            if paper_id and re.search(rf"(?<![\w-]){re.escape(paper_id)}(?![\w-])", query, re.I):
                ids.append(paper_id)
                expressions.append(paper_id)
            elif title and len(title) >= 5 and title.lower() in query.lower():
                ids.append(paper_id)
                expressions.append(title)

        # 2) 序数指代:第一篇 / 第 2 个 / paper 3 / first paper
        # 关键步骤：序号按用户可见编号解析——最近批次优先（对应屏幕上刚返回的论文），
        # 批次外回退到全局累积池。跨主题搜索时“第 1 篇”因此指到最新搜索的首篇，而非旧主题。
        ordinal_patterns = [
            r"第\s*([一二两三四五六七八九十百千万\d]+)\s*(?:篇论文|篇|个|项)",
            r"(?:paper|article)\s*#?\s*(\d+)",
        ]
        missing_ordinal: Optional[int] = None
        for pattern in ordinal_patterns:
            for match in re.finditer(pattern, query, re.I):
                position = self._ordinal(match.group(1))
                target = self._ordinal_target(session, position)
                if target is not None and self._paper_id(target):
                    ids.append(self._paper_id(target))
                    expressions.append(match.group(0))
                elif position > 0 and missing_ordinal is None:
                    missing_ordinal = position
                    expressions.append(match.group(0))

        english_ordinals = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        }
        for word, position in english_ordinals.items():
            match = re.search(rf"\b{word}\s+(?:paper|article)\b", query, re.I)
            if match:
                target = self._ordinal_target(session, position)
                if target is not None and self._paper_id(target):
                    ids.append(self._paper_id(target))
                    expressions.append(match.group(0))

        # 3) 配对指代(前者/后者)与最近提及论文指代
        if re.search(r"前者|former", query, re.I) and session.last_mentioned_paper_ids:
            ids.append(session.last_mentioned_paper_ids[-1])
            expressions.append("前者")
        if re.search(r"后者|latter", query, re.I) and session.last_mentioned_paper_ids:
            ids.append(session.last_mentioned_paper_ids[0])
            expressions.append("后者")
        if not ids and re.search(r"这篇|该论文|它(?:的|采用|使用)|这项工作|this paper|it\b", query, re.I):
            if session.active_paper_id:
                ids.append(session.active_paper_id)
                expressions.append("active paper")

        # 4) 报告章节指代
        section = self._resolve_section(query, session.last_report_sections)
        if section:
            expressions.append(section)

        ids = [item for item in dict.fromkeys(ids) if item]
        # 有解析结果 → 高置信直接采用;无结果 → 低置信,触发上游 LLM 兜底
        if ids or section:
            confidence = 0.95 if expressions else 0.75
            reason = "Resolved from session ordinals, IDs, active resources, or section names"
        else:
            confidence = 0.2
            reason = "No deterministic reference matched the active session"
        return ReferenceResolution(
            reference_expression="; ".join(dict.fromkeys(expressions)),
            resolved_paper_ids=ids,
            resolved_section=section,
            confidence=1.0 if missing_ordinal is not None else confidence,
            missing_ordinal=missing_ordinal,
            reasoning=reason,
        )

    @staticmethod
    def _resolve_section(query: str, sections: List[str]) -> Optional[str]:
        # 三种章节指代:直接命中章节名 → 引号包裹 → "展开X章节"动宾结构
        lowered = query.lower()
        for section in sections:
            if section and section.lower() in lowered:
                return section
        quoted = re.search(r"[“\"《](.+?)[”\"》](?:章节|部分|一节)?", query)
        if quoted:
            target = quoted.group(1).strip()
            for section in sections:
                if target in section or section in target:
                    return section
        generic = re.search(r"(?:展开|解释|追溯|补充|检查)\s*(.+?)(?:章节|部分|一节)", query)
        if generic:
            target = generic.group(1).strip(" 的")
            for section in sections:
                if target in section:
                    return section
        return None

    @staticmethod
    def _ordinal_target(session: SessionContext, position: int) -> Optional[dict]:
        """按用户可见编号解析论文：最近批次优先，其次全局累积池。

        跨主题搜索时批次与全局池的编号会错位（新搜索从 1 重新编号，
        但全局池仍从旧主题论文开始）。批次优先保证“第 N 篇”指到屏幕上
        刚返回的那批论文，避免指回上一主题的陈旧论文。
        """
        if position <= 0:
            return None
        batch = list(session.last_recommendation_batch or [])
        start = int(getattr(session, "last_recommendation_batch_start", 1) or 1)
        if batch and start <= position < start + len(batch):
            return batch[position - start]
        papers = session.recommended_papers
        if 1 <= position <= len(papers):
            return papers[position - 1]
        return None

    @staticmethod
    def _candidates(session: SessionContext) -> List[dict]:
        """构造供 LLM 语义解析的候选清单。

        最近批次在前并携带其显示编号，批次外的历史论文按续排编号列出，
        保证编号唯一且与用户可见的编号尽量对齐。
        """
        candidates: List[dict] = []
        seen: set[str] = set()

        def push(paper: dict, position: int) -> None:
            paper_id = ReferenceResolver._paper_id(paper)
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                candidates.append({
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "position": position,
                })

        batch = list(session.last_recommendation_batch or [])
        start = int(getattr(session, "last_recommendation_batch_start", 1) or 1)
        for index, paper in enumerate(batch):
            push(paper, start + index)
        next_position = start + len(batch)
        for index, paper in enumerate(session.recommended_papers):
            push(paper, next_position + index)
        return candidates

    @staticmethod
    def _paper_id(paper: dict) -> str:
        return str(paper.get("source_id") or paper.get("paper_id") or "")

    @staticmethod
    def _ordinal(value: str) -> int:
        # 把中文数字(一/两/十/十一/二十/一百…)解析为阿拉伯数字。
        if value.isdigit():
            return int(value)
        digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}
        units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
        total = 0
        section = 0
        number = 0
        for char in value:
            if char in digits:
                number = digits[char]
            elif char in units:
                unit = units[char]
                if unit == 10000:
                    section = (section + (number or 1)) * unit
                    total += section
                    section = 0
                else:
                    section += (number or 1) * unit
                number = 0
            else:
                return 0
        return total + section + number


_REFERENCE_SYSTEM = """Resolve references in an academic-research follow-up using only
the supplied session resources. Return exact candidate paper IDs and an exact report
section. Never invent an ID. If ambiguous, return low confidence. Output JSON with
reference_expression, resolved_paper_ids, resolved_section, confidence, reasoning."""
