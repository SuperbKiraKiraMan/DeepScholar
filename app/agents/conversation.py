"""基于论文、报告、证据与对话历史的统一有据对话代理。

对话历史仅作为语义上下文，论文/报告/证据才是 grounded_claims 与
supporting_quotes 的可信证据来源。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agents.paper_compare import PaperCompareStrategy, build_paper_compare_system_prompt
from app.agents.paper_qa import PAPER_QA_SYSTEM, PaperQARequest, PaperQAStrategy
from app.agents.report_follow_up import (
    REPORT_FOLLOW_UP_SYSTEM,
    ReportFollowUpRequest,
    ReportFollowUpStrategy,
)
from app.llm.client import get_llm_client
from app.llm.prompts import build_output_language_instruction, build_system_prompt_with_memory
from app.services.run_store import run_store
from app.services.session_store import SessionContext


@dataclass
class FollowUpContext:
    query: str
    papers: List[Dict[str, Any]]
    report: Optional[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    resolved_section: Optional[str]
    report_id: Optional[str]
    missing_paper_ids: List[str] = None


class ConversationRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    context: FollowUpContext
    operation_hint: str = ""
    language: str = "zh"


class ConversationResponse(BaseModel):
    answer: str = ""
    mode: str = "history_only"
    strategy: str = "general"
    cannot_answer: bool = False
    cannot_answer_reason: str = ""
    referenced_papers: List[str] = Field(default_factory=list)
    referenced_section: Optional[str] = None
    referenced_evidence: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)
    quote_sections: List[str] = Field(default_factory=list)
    comparison_dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_claims: List[str] = Field(default_factory=list)
    analysis: str = ""
    warnings: List[str] = Field(default_factory=list)


def _ctx_value(context: Any, key: str, default: Any = None) -> Any:
    """兼容 SessionContext 对象与 dict 两种形态的取值工具。"""
    if isinstance(context, SessionContext):
        return getattr(context, key, default)
    if isinstance(context, dict):
        return context.get(key, default)
    return default


def _paper_id(paper: Dict[str, Any]) -> str:
    """统一获取论文标识，优先 source_id，其次 paper_id。"""
    return str(paper.get("source_id") or paper.get("paper_id") or "")


def _effective_paper_ids(state: Dict[str, Any], resolved_ids: List[str]) -> List[str]:
    """计算本次对话生效的论文 ID 列表。

    优先使用解析阶段给出的 resolved_ids；缺失时回退到会话上下文中的
    active_paper_id 与 last_mentioned_paper_ids，并保持顺序、去重。
    """
    if resolved_ids:
        return list(dict.fromkeys(str(item) for item in resolved_ids if item))
    context = state.get("session_context")
    active = _ctx_value(context, "active_paper_id")
    fallback = [str(active)] if active else []
    for paper_id in _ctx_value(context, "last_mentioned_paper_ids", []) or []:
        if paper_id and str(paper_id) not in fallback:
            fallback.append(str(paper_id))
    return fallback


def load_report(state: Dict[str, Any], report_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """按 report_id 从 run_store 载入报告，并归一化正文/证据/来源字段。"""
    if not report_id:
        return None
    record = run_store.get(report_id)
    if not record:
        return None
    return {
        **record,
        "report_id": report_id,
        "report_text": str(record.get("final_report") or record.get("draft_report") or ""),
        "evidence_cards": list(record.get("evidence_cards") or []),
        "sources": list(record.get("sources") or []),
    }


def load_papers(state: Dict[str, Any], paper_ids: List[str]) -> List[Dict[str, Any]]:
    """按 paper_ids 顺序载入论文，候选池为会话推荐论文与报告来源的并集。"""
    wanted = list(dict.fromkeys(paper_ids))
    session_papers = list(_ctx_value(state.get("session_context"), "recommended_papers", []) or [])
    report_id = _ctx_value(state.get("session_context"), "active_report_id")
    report = run_store.get(report_id) if report_id else None
    candidates = session_papers + list((report or {}).get("sources") or [])
    by_id: Dict[str, Dict[str, Any]] = {}
    for paper in candidates:
        identifier = _paper_id(paper)
        if identifier and identifier not in by_id:
            by_id[identifier] = paper
    # 仅返回实际命中的论文，缺失项交由 missing_paper_ids 上报
    return [by_id[paper_id] for paper_id in wanted if paper_id in by_id]


def load_evidence(state: Dict[str, Any], paper_ids: List[str]) -> List[Dict[str, Any]]:
    """载入报告关联的 Evidence Cards，并按 paper_ids 做白名单过滤。"""
    resolved = state.get("resolved_refs") or {}
    report_id = resolved.get("report_id") if isinstance(resolved, dict) else None
    report_id = report_id or _ctx_value(state.get("session_context"), "active_report_id")
    report = run_store.get(report_id) if report_id else None
    cards = list((report or {}).get("evidence_cards") or [])
    if not paper_ids:
        return cards
    allowed = set(paper_ids)
    return [card for card in cards if str(card.get("source_id") or "") in allowed]


def context_builder(state: Dict[str, Any]) -> FollowUpContext:
    """从 graph state 构造对话所需的 FollowUpContext。"""
    resolved = state.get("resolved_refs") or {}
    if not isinstance(resolved, dict):
        resolved = {
            "paper_ids": getattr(resolved, "paper_ids", []),
            "resolved_section": getattr(resolved, "resolved_section", None),
            "report_id": getattr(resolved, "report_id", None),
        }
    paper_ids = _effective_paper_ids(state, list(resolved.get("paper_ids") or []))
    report_id = resolved.get("report_id") or _ctx_value(
        state.get("session_context"), "active_report_id"
    )
    # 对话历史按轮数截断，仅作为语义上下文使用
    history_turns = max(1, int(os.getenv("CONVERSATION_HISTORY_TURNS", "6")))
    messages = (
        state.get("messages")
        or state.get("conversation_messages")
        or _ctx_value(state.get("session_context"), "conversation_messages", [])
        or []
    )
    papers = load_papers(state, paper_ids)
    loaded_ids = {_paper_id(paper) for paper in papers}
    return FollowUpContext(
        query=str(state.get("topic") or ""),
        papers=papers,
        report=load_report(state, report_id),
        evidence=load_evidence(state, paper_ids),
        history=list(messages)[-(history_turns * 2):],
        resolved_section=resolved.get("resolved_section") or state.get("resolved_section"),
        report_id=report_id,
        missing_paper_ids=[paper_id for paper_id in paper_ids if paper_id not in loaded_ids],
    )


class ConversationAgent:
    """根据上下文形态、operation_hint 与查询语义，选择有据回答的落地策略。"""

    def __init__(self) -> None:
        self.paper_qa = PaperQAStrategy()
        self.paper_compare = PaperCompareStrategy()
        self.report_follow_up = ReportFollowUpStrategy()

    @staticmethod
    def mode_for(context: FollowUpContext) -> str:
        """依据论文数量与是否含报告判定上下文形态（与意图无关）。"""
        count = len(context.papers)
        has_report = context.report is not None
        if count == 1 and not has_report:
            return "single_paper"
        if count >= 2 and not has_report:
            return "multi_paper"
        if has_report and count == 0:
            return "report"
        if has_report and count >= 1:
            return "mixed"
        return "history_only"

    @staticmethod
    def _query_signals(query: str) -> tuple[bool, bool, bool, bool]:
        """从查询文本中识别四类语义信号：论文 / 对比 / 报告 / 分析建议。"""
        lowered = query.lower()
        compare = any(item in lowered for item in (
            "比较", "对比", "区别", "异同", "difference", "compare", "versus", " vs ",
        ))
        report = any(item in lowered for item in (
            "报告", "章节", "部分", "展开", "扩写", "溯源", "证据空白",
            "report", "section", "expand", "trace evidence", "fill gap",
        ))
        paper = any(item in lowered for item in (
            "论文", "这篇", "这些论文", "方法", "创新", "实验", "结论", "局限",
            "数据集", "指标", "结果", "paper", "method", "novelty", "experiment",
            "conclusion", "limitation", "dataset", "result",
        ))
        # 分析/建议类信号：优化、建议、评价、为什么、如何改进等，需综合研判而非单点问答
        analysis = any(item in lowered for item in (
            "优化", "建议", "评价", "评估", "改进", "为什么", "如何改进", "怎么改进",
            "如何优化", "怎么优化", "如何提升", "如何完善",
            "optimize", "suggest", "suggestion", "recommend", "recommendation",
            "evaluate", "evaluation", "assess", "assessment",
            "improve", "improvement", "how to improve", "why",
        ))
        return paper, compare, report, analysis

    def strategy_for(self, request: ConversationRequest, mode: str) -> str:
        # operation_hint 不在已知能力集合内时，直接走 general 综合
        if request.operation_hint and request.operation_hint not in {
            "paper_qa", "paper_compare", "report_follow_up"
        }:
            return "general"
        hint = request.operation_hint if request.operation_hint in {
            "paper_qa", "paper_compare", "report_follow_up"
        } else ""
        paper_query, compare_query, report_query, analysis_query = self._query_signals(request.query)
        # 跨能力请求或分析/建议类问题需要综合所有可用材料，统一路由到 general
        if sum((compare_query, report_query)) > 1 or analysis_query:
            return "general"
        if compare_query and mode in {"multi_paper", "mixed"} and hint in {"", "paper_compare"}:
            return "paper_compare"
        if report_query and mode in {"report", "mixed"} and hint == "report_follow_up":
            return "report_follow_up"
        if (
            paper_query and not compare_query and not report_query
            and mode in {"single_paper", "multi_paper"}
            and hint == "paper_qa"
        ):
            return "paper_qa"
        return "general"

    async def answer(
        self,
        request: ConversationRequest,
        *,
        agent_mode: str = "llm",
        memory_prompt: str = "",
        llm_client=None,
    ) -> tuple[ConversationResponse, dict]:
        # 先由上下文形态与查询语义选定策略，再分发到对应落地方法
        mode = self.mode_for(request.context)
        strategy = self.strategy_for(request, mode)
        if strategy == "paper_qa":
            return await self._paper_qa(request, mode, agent_mode, memory_prompt, llm_client)
        if strategy == "paper_compare":
            return await self._paper_compare(request, mode, agent_mode, memory_prompt, llm_client)
        if strategy == "report_follow_up":
            return await self._report(request, mode, agent_mode, memory_prompt, llm_client)
        return await self._general(request, mode, agent_mode, memory_prompt, llm_client)

    async def _paper_qa(self, request, mode, agent_mode, memory_prompt, llm_client):
        # 单篇论文问答：复用 PaperQAStrategy，引用与结论均来自该论文原文
        paper = request.context.papers[0]
        identifier = _paper_id(paper)
        result, llm_result = await self.paper_qa.answer(
            PaperQARequest(
                question=request.query,
                paper_id=identifier,
                language=request.language,
                paper_full_text=str(paper.get("full_text") or paper.get("text") or ""),
                paper_abstract=str(paper.get("abstract") or paper.get("snippet") or ""),
                paper_title=str(paper.get("title") or ""),
            ),
            agent_mode=agent_mode,
            memory_prompt=memory_prompt,
            llm_client=llm_client,
        )
        return ConversationResponse(
            answer=result.answer, mode=mode, strategy="paper_qa",
            cannot_answer=result.cannot_answer,
            cannot_answer_reason=result.cannot_answer_reason,
            referenced_papers=[identifier] if identifier else [],
            supporting_quotes=result.supporting_quotes,
            quote_sections=result.quote_sections,
            grounded_claims=[result.answer] if not result.cannot_answer else [],
        ), llm_result

    async def _paper_compare(self, request, mode, agent_mode, memory_prompt, llm_client):
        papers = request.context.papers
        # 两篇论文走专用对比策略，ID/标题由运行时注入，避免模型臆造
        if len(papers) == 2:
            result, llm_result = await self.paper_compare.compare(
                papers[0], papers[1], request.query,
                evidence_cards=request.context.evidence,
                agent_mode=agent_mode,
                language=request.language,
                memory_prompt=memory_prompt,
                llm_client=llm_client,
            )
            dimensions = [item.model_dump() for item in result.comparison_dimensions]
            answer = result.summary + "\n\n" + "\n".join(
                f"- {item.dimension}: {item.analysis}\n  - A: {item.paper_a}\n  - B: {item.paper_b}"
                for item in result.comparison_dimensions
            )
            ids = [_paper_id(paper) for paper in papers if _paper_id(paper)]
            return ConversationResponse(
                answer=answer, mode=mode, strategy="paper_compare",
                referenced_papers=ids, comparison_dimensions=dimensions,
                grounded_claims=[result.summary] if result.summary else [],
                warnings=result.warnings,
            ), llm_result

        # 三篇及以上：先构造规则回退，再尝试 LLM 全量对比（不截断）
        response = self._rule_n_paper_compare(request, mode)
        llm_result = {"success": False, "error": "LLM mode disabled"}
        if agent_mode == "llm":
            llm_result = await (llm_client or get_llm_client()).generate_structured(
                system_prompt=build_system_prompt_with_memory(
                    build_paper_compare_system_prompt(request.language)
                    + "\nCompare every supplied paper; do not truncate.",
                    memory_prompt,
                ),
                user_prompt=json.dumps({
                    "query": request.query,
                    "papers": request.context.papers,
                    "evidence": request.context.evidence,
                }, ensure_ascii=False)[:150_000],
                output_schema=ConversationResponse,
                temperature=0.0,
            )
            if llm_result.get("success"):
                candidate = llm_result["data"]
                expected = {_paper_id(item) for item in papers if _paper_id(item)}
                # 对 comparison_dimensions 中的 paper_id 做严格白名单过滤，剔除模型臆造的论文 ID
                dropped = 0
                for dimension in candidate.comparison_dimensions:
                    raw_entries = dimension.get("papers") or []
                    filtered_entries = [
                        entry for entry in raw_entries
                        if isinstance(entry, dict)
                        and str(entry.get("paper_id") or "") in expected
                    ]
                    dropped += len(raw_entries) - len(filtered_entries)
                    dimension["papers"] = filtered_entries
                candidate_ids = set(candidate.referenced_papers)
                matrix_ids = {
                    str(entry.get("paper_id") or "")
                    for dimension in candidate.comparison_dimensions
                    for entry in (dimension.get("papers") or [])
                    if isinstance(entry, dict)
                }
                # 校验：所有期望论文都必须被引用列表或对比矩阵覆盖，否则回退到规则结果
                if expected.issubset(candidate_ids | matrix_ids):
                    response = candidate
                    response.mode = mode
                    response.strategy = "paper_compare"
                    response.referenced_papers = [
                        item for item in response.referenced_papers if item in expected
                    ]
                    if dropped:
                        response.warnings.append(
                            f"已从对比维度中剔除 {dropped} 个不在会话白名单内的论文 ID。"
                        )
        return response, llm_result

    def _rule_n_paper_compare(self, request, mode):
        # 规则版 N 篇对比：逐维度逐篇从会话材料中抽取证据，缺失项明确标注
        is_zh = str(request.language or "zh").lower().replace("_", "-").startswith("zh")
        dimensions = []
        keys = {
            "problem": ("research_task", "snippet"),
            "method": ("method", "full_text", "snippet"),
            "results": ("result", "snippet"),
            "limitations": ("limitation",),
            "novelty": ("key_contribution", "snippet"),
        }
        for dimension, fields in keys.items():
            entries = []
            for paper in request.context.papers:
                evidence = next((str(paper.get(key)) for key in fields if paper.get(key)), "")
                entries.append({
                    "paper_id": _paper_id(paper), "title": str(paper.get("title") or ""),
                    "evidence": evidence[:600] or (
                        "当前证据中未提供" if is_zh else "Not available in supplied evidence"
                    ),
                })
            dimensions.append({
                "dimension": {
                    "problem": "研究问题", "method": "方法", "results": "结果",
                    "limitations": "局限性", "novelty": "创新点",
                }.get(dimension, dimension) if is_zh else dimension,
                "papers": entries,
                "analysis": (
                    "逐篇列出当前会话中可核实的材料；缺失项已明确标注。"
                    if is_zh else
                    "Each paper is listed using verifiable material from the current session; missing items are marked explicitly."
                ),
            })
        ids = [_paper_id(item) for item in request.context.papers if _paper_id(item)]
        answer = (
            f"已基于当前会话中的 {len(ids)} 篇论文，从五个维度进行完整对比。"
            if is_zh else
            f"A complete comparison across five dimensions was made for {len(ids)} papers in the current session."
        )
        return ConversationResponse(
            answer=answer, mode=mode, strategy="paper_compare",
            referenced_papers=ids, comparison_dimensions=dimensions,
            grounded_claims=[answer],
        )

    async def _report(self, request, mode, agent_mode, memory_prompt, llm_client):
        # 报告追问：复用 ReportFollowUpStrategy，引用来源严格限定在报告 sources 白名单内
        report = request.context.report or {}
        result, llm_result = await self.report_follow_up.answer(
            ReportFollowUpRequest(
                query=request.query,
                report_id=request.context.report_id or "",
                language=request.language,
                section=request.context.resolved_section,
                report_text=str(report.get("report_text") or report.get("final_report") or report.get("draft_report") or ""),
                evidence_cards=request.context.evidence,
                sources=list(report.get("sources") or []),
            ),
            agent_mode=agent_mode,
            memory_prompt=memory_prompt,
            llm_client=llm_client,
        )
        valid_papers = {_paper_id(item) for item in list(report.get("sources") or [])}
        return ConversationResponse(
            answer=result.answer, mode=mode, strategy="report_follow_up",
            cannot_answer=result.cannot_answer,
            cannot_answer_reason=result.cannot_answer_reason,
            referenced_papers=[item for item in result.referenced_sources if item in valid_papers],
            referenced_section=result.referenced_section,
            referenced_evidence=result.referenced_evidence,
            grounded_claims=[result.answer] if not result.cannot_answer else [],
        ), llm_result

    async def _general(self, request, mode, agent_mode, memory_prompt, llm_client):
        # general 综合：跨能力 / 分析建议 / 无法归入专用策略的问题都在此处理
        context = request.context
        _paper_query, compare_query, report_query, _analysis_query = self._query_signals(request.query)
        # 所指论文不在当前会话可用材料中，直接拒答
        if context.missing_paper_ids:
            reason = "PAPER_NOT_IN_SESSION：所指论文不在当前会话可用材料中。"
            return ConversationResponse(
                answer="无法在当前会话中找到所指论文，请重新选择或明确论文编号。",
                mode=mode, strategy="general", cannot_answer=True,
                cannot_answer_reason=reason,
            ), {"success": False, "error": reason}
        # 追问报告但当前会话没有可核实报告
        if report_query and not compare_query and context.report is None:
            reason = "REPORT_NOT_FOUND：当前会话中没有可核实的报告材料。"
            return ConversationResponse(
                answer="当前会话没有可追问的研究报告。",
                mode=mode, strategy="general", cannot_answer=True,
                cannot_answer_reason=reason,
            ), {"success": False, "error": reason}
        paper_ids = {_paper_id(item) for item in context.papers if _paper_id(item)}
        evidence_ids = {
            str(item.get("evidence_id") or item.get("card_id") or "")
            for item in context.evidence
            if item.get("evidence_id") or item.get("card_id")
        }
        # 可信材料：仅论文/报告/证据，对话历史不作为可信证据来源
        trusted_material = self._trusted_material(context)
        # 完全没有任何可用上下文（连对话历史都没有）时才拒答；历史单独存在时仍可作为语义上下文
        if not trusted_material and not context.history:
            reason = "当前会话缺少可核实的论文、报告、证据或历史材料。"
            if report_query:
                reason = "REPORT_NOT_FOUND：当前会话中没有可核实的报告材料。"
            return ConversationResponse(
                answer="当前上下文的证据不足，暂时无法可靠回答。",
                mode=mode, strategy="general", cannot_answer=True,
                cannot_answer_reason=reason,
            ), {"success": False, "error": reason}

        llm_result = {"success": False, "error": "LLM mode disabled"}
        if agent_mode == "llm":
            llm_result = await (llm_client or get_llm_client()).generate_structured(
                system_prompt=build_system_prompt_with_memory(
                    GENERAL_CONVERSATION_SYSTEM + "\n\n"
                    + build_output_language_instruction(request.language),
                    memory_prompt,
                ),
                user_prompt=json.dumps({
                    "query": request.query,
                    "papers": context.papers,
                    "report": context.report,
                    "evidence": context.evidence,
                    "history": context.history,
                    "resolved_section": context.resolved_section,
                }, ensure_ascii=False)[:160_000],
                output_schema=ConversationResponse,
                temperature=0.0,
            )
            if llm_result.get("success"):
                output = llm_result["data"]
                output.mode = mode
                output.strategy = "general"
                # 引用论文/证据 ID 必须命中会话白名单，剔除模型臆造的 ID
                output.referenced_papers = [item for item in output.referenced_papers if item in paper_ids]
                output.referenced_evidence = [item for item in output.referenced_evidence if item in evidence_ids]
                # supporting_quotes 必须能在可信材料中逐字命中；对话历史不参与该校验
                output.supporting_quotes = [
                    quote for quote in output.supporting_quotes
                    if any(self._grounded(quote, text) for text in trusted_material)
                ]
                # grounded_claims 必须有可信引用/证据/章节支撑，否则清除并告警
                if output.grounded_claims and not (output.supporting_quotes or output.referenced_evidence or output.referenced_section):
                    output.warnings.append("Grounded claims lacked a verifiable quote/evidence/section reference.")
                    output.grounded_claims = []
                if output.answer.strip():
                    # 仅有对话历史、无可信材料时，回答仅基于语义上下文，需明确告知
                    if not trusted_material:
                        output.warnings.append("回答仅基于对话历史的语义上下文，未包含可核实的论文/报告/证据引用。")
                    if compare_query and len(context.papers) == 1:
                        output.warnings.append("只解析到一篇论文，无法完成多论文对比；已降级为单篇上下文回答。")
                    return output, llm_result

        # 规则回退：仅从可信材料中选取与问题最匹配的片段，绝不引用对话历史
        quote = self._best_passage(request.query, trusted_material)
        if not quote:
            warnings = []
            if compare_query and len(context.papers) == 1:
                warnings.append("只解析到一篇论文，无法完成多论文对比；已降级为单篇上下文回答。")
            if not trusted_material:
                reason = "仅有对话历史作为语义上下文，缺少可核实的论文/报告/证据，无法形成可信回答。"
            else:
                reason = "可核实材料存在，但缺少与问题匹配的引用或证据。"
            return ConversationResponse(
                answer="现有材料与这个问题的关联不足，无法形成可信回答。",
                mode=mode, strategy="general", cannot_answer=True,
                cannot_answer_reason=reason,
                warnings=warnings,
            ), llm_result
        answer = "基于当前会话可核实的材料：" + quote
        warnings = []
        if compare_query and len(context.papers) == 1:
            warnings.append("只解析到一篇论文，无法完成多论文对比；已降级为单篇上下文回答。")
        return ConversationResponse(
            answer=answer, mode=mode, strategy="general",
            referenced_papers=list(paper_ids),
            referenced_section=context.resolved_section,
            referenced_evidence=list(evidence_ids),
            supporting_quotes=[quote], quote_sections=["session_context"],
            grounded_claims=[quote],
            analysis="分析：以上结论仅覆盖当前会话已经保存的材料，未覆盖部分不能据此推断。",
            warnings=warnings,
        ), llm_result

    @staticmethod
    def _trusted_material(context: FollowUpContext) -> List[str]:
        """收集可信证据文本：论文正文/摘要/方法/结果、报告正文、证据卡内容。

        对话历史不在此列——历史仅作为语义上下文供 LLM 理解语境，
        不得作为 supporting_quotes / grounded_claims 的可信来源。
        """
        values: List[str] = []
        for paper in context.papers:
            for key in ("full_text", "text", "abstract", "snippet", "method", "result"):
                if paper.get(key):
                    values.append(str(paper[key]))
        if context.report:
            text = context.report.get("report_text") or context.report.get("final_report") or context.report.get("draft_report")
            if text:
                values.append(str(text))
        for card in context.evidence:
            for key in ("claim", "evidence", "quote", "snippet"):
                if card.get(key):
                    values.append(str(card[key]))
        return values

    @staticmethod
    def _grounded(quote: str, text: str) -> bool:
        """判断 quote 是否能在 text 中逐字命中（忽略空白与大小写差异）。"""
        needle = re.sub(r"\s+", " ", quote.strip()).lower()
        haystack = re.sub(r"\s+", " ", text).lower()
        return bool(needle) and needle in haystack

    @staticmethod
    def _best_passage(query: str, material: List[str]) -> str:
        """从 material 中切分句子，按查询词命中数排序，返回最匹配的片段。

        调用方应只传入可信材料（不含对话历史），避免把历史当作可引用证据。
        """
        terms = set(re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query.lower()))
        passages = [
            item.strip()[:800]
            for text in material
            for item in re.split(r"(?<=[。！？.!?])\s+|\n+", text)
            if item.strip()
        ]
        ranked = sorted(passages, key=lambda item: sum(term in item.lower() for term in terms), reverse=True)
        return ranked[0] if ranked and (not terms or any(term in ranked[0].lower() for term in terms)) else ""


GENERAL_CONVERSATION_SYSTEM = """Answer using only the supplied session papers,
report, evidence, and conversation history. This is a general synthesis task, not a
closed intent classifier. Put directly verifiable facts in grounded_claims and
evidence-based interpretation in analysis. Quotes must be verbatim. Paper and evidence
IDs must exactly match supplied IDs. If only part is supported, answer that part and
put the boundary in warnings. Set cannot_answer only when the material cannot support
a trustworthy answer; describe the missing evidence, never say the intent is unsupported.
Conversation history is semantic context only: never copy a supporting_quote or ground a
grounded_claim from it; quotes and grounded claims must come from the supplied papers,
report, or evidence. Return JSON matching ConversationResponse."""
