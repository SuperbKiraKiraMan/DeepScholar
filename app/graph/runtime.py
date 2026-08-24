"""
app/graph/runtime.py

LangGraph Agent Runtime —— 项目内封装层。

使用 StateGraph、条件边和 LangGraph Send API 实现有界并行任务分发。
支持来源去重、引用重试、失败隔离和可选的 LLM Agent Intelligence。

关键修复：
- Send payload 必须显式包含子 Worker 所需的全部字段
- Planner 生成 2-3 个不同 search query 以证明动态分发
- 并行 worker 写入内部 accumulator，merge 节点显式去重
- trace/warnings 使用 operator.add reducer，节点只返回 delta
- 结构化 trace event
- 单 Worker 失败不中断整个 Run（try/except 保护）
- 去重优先：DOI → arXiv ID → normalized URL → source_id
- citation retry 仅重新处理失败 source，旧无效 EvidenceCard 被替换
- agent_mode=rule 完全离线；agent_mode=llm 可选启用 DeepSeek
"""

import asyncio
import functools
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.agents.planner import Planner, PlannerAgent, Task, TaskDAG
from app.agents.worker import Worker, WorkerAgent, WorkerContext
from app.agents.draft_reviewer import DraftReviewer
from app.agents.evaluator import Evaluator
from app.agents.final_reviewer import FinalReviewer
from app.agents.controller import ControllerAgent, ControllerDecision, IntentController
from app.agents.reviewer import ReviewerAgent
from app.agents.conversation import ConversationAgent, ConversationRequest, context_builder
from app.agents.protocol import agent_protocol
from app.agents.schemas import (
    AgentRole,
    AgentTask,
    AgentTaskStatus,
    WorkerProfile,
    WorkerStrategy,
    WorkItem,
    WorkPlan,
    WorkerResult,
    ExecutionBudget,
    ExecutionSpec,
    ReviewVerdict,
)
from app.agents.source_selector import AdaptiveSourceSelector
from app.agents.task_relevance import filter_sources_for_task
from app.services.run_store import run_store
from app.tools.source_quality_scorer import SourceQualityScorer
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.tools.registry import ToolRegistry
from app.core.config import get_local_rag_min_score
from app.services.session_store import SessionContext

# 可选的按运行作用域的生命周期观察器；未安装时为 no-op。
from app.observability.lifecycle import (
    clear_runtime_budget,
    emit_after_plan,
    emit_before_run,
    emit_error,
    register_runtime_budget,
    reset_execution_context,
    set_execution_context,
)
from app.observability.metrics import aggregate_run_metrics


_chapter_semaphore: Optional[asyncio.Semaphore] = None
_chapter_semaphore_loop = None


def _get_chapter_semaphore() -> asyncio.Semaphore:
    """限制 provider 并发，同时保留 Send API 的并行能力。"""
    global _chapter_semaphore, _chapter_semaphore_loop
    loop = asyncio.get_running_loop()
    limit = max(1, int(os.getenv("LLM_CHAPTER_MAX_CONCURRENCY", "3")))
    if _chapter_semaphore is None or _chapter_semaphore_loop is not loop:
        _chapter_semaphore = asyncio.Semaphore(limit)
        _chapter_semaphore_loop = loop
    return _chapter_semaphore


# ================================================================
# 工具函数
# ================================================================

def _open_protocol_task(
    state: Dict[str, Any],
    *,
    role: AgentRole,
    task_id: str,
    input_data: Dict[str, Any],
    allowed_tools: Optional[List[str]] = None,
    depends_on: Optional[List[str]] = None,
) -> AgentTask:
    """在角色执行前验证输入 Schema 和本任务工具授权。"""
    return agent_protocol.create_task(
        task_id=task_id,
        role=role,
        input_data=input_data,
        allowed_tools=allowed_tools,
        depends_on=depends_on,
        run_id=str(state.get("run_id") or ""),
        session_id=str(state.get("session_id") or ""),
        metadata={"backend": state.get("backend", "")},
    )


def _protocol_tool_calls(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 Worker trace 中提取真实且已注册的工具调用。"""
    calls = []
    registry = ToolRegistry.get_instance()
    for item in trace:
        tool_name = str(item.get("tool_name") or "")
        if not tool_name or registry.get(tool_name) is None:
            continue
        calls.append({
            "tool_name": tool_name,
            "success": bool(item.get("success", False)),
            "latency_ms": max(0, int(item.get("latency_ms", 0) or 0)),
            "error": str(item.get("error") or "")[:200],
        })
    return calls


def _close_protocol_task(
    task: AgentTask,
    *,
    output_data: Dict[str, Any],
    trace: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[str]] = None,
    error: str = "",
) -> Dict[str, Any]:
    """验证角色输出与实际工具调用，并生成可供 Harness 审计的 trace。"""
    status = (
        AgentTaskStatus.FAILED if error
        else AgentTaskStatus.PARTIAL_SUCCESS if warnings
        else AgentTaskStatus.SUCCESS
    )
    result = agent_protocol.create_result(
        task,
        output_data=output_data,
        status=status,
        tool_calls=_protocol_tool_calls(trace or []),
        warnings=warnings or [],
        error=(
            agent_protocol.error(
                "TOOL_UNAVAILABLE",
                message=error,
                recoverable=True,
                suggested_action="检查工具可用性后重试任务",
            )
            if error else None
        ),
    )
    return _tr_event(
        "agent_protocol_validated",
        node=task.task_id,
        graph_node=False,
        protocol_version=result.protocol_version,
        agent_role=task.role.value,
        task_id=task.task_id,
        allowed_tools=task.allowed_tools,
        tools_called=[item.tool_name for item in result.tool_calls],
        status=result.status.value,
    )


@asynccontextmanager
async def _agent_permission_scope(task: AgentTask):
    """把角色与任务级授权注入 contextvar，供 BaseTool 在执行前强制判权。"""
    token = set_execution_context(
        agent_role=task.role.value,
        protocol_task_id=task.task_id,
        allowed_tools=list(task.allowed_tools),
    )
    try:
        yield
    finally:
        reset_execution_context(token)

def _normalize_url(url: str) -> str:
    """规范化 URL 用于去重比较。"""
    if not url:
        return ""
    url = url.strip().rstrip("/").lower()
    # 移除协议前缀
    url = re.sub(r'^https?://', '', url)
    # 移除 www.
    url = re.sub(r'^www\.', '', url)
    return url


def _extract_doi(source: Dict) -> str:
    """从 source 中提取 DOI（如存在）。"""
    doi = source.get("doi", "")
    if doi:
        return doi.strip().lower()
    # 尝试从 full_text 或 url 中提取 DOI
    full_text = source.get("full_text", "")
    url = source.get("url", "")
    for text in [full_text, url]:
        m = re.search(r'(10\.\d{4,}/[^\s\"\']+)', text)
        if m:
            return m.group(1).strip().lower()
    return ""


def _extract_arxiv_id(source: Dict) -> str:
    """从 source 中提取 arXiv ID（如存在）。"""
    arxiv = source.get("arxiv_id", "")
    if arxiv:
        return arxiv.strip().lower()
    url = source.get("url", "")
    full_text = source.get("full_text", "")
    for text in [url, full_text]:
        m = re.search(r'arxiv\.org/(?:abs|pdf)/([\d.]+(?:v\d+)?)', text)
        if m:
            return m.group(1).strip().lower()
    return ""


def _dedup_key(source: Dict) -> str:
    """
    为 source 计算去重键。

    优先级：DOI → arXiv ID → normalized URL → source_id
    同一 URL 即使 source_id 不同也必须合并。
    """
    doi = _extract_doi(source)
    if doi:
        return f"doi:{doi}"

    arxiv = _extract_arxiv_id(source)
    if arxiv:
        return f"arxiv:{arxiv}"

    url = _normalize_url(source.get("url", ""))
    if url:
        return f"url:{url}"

    sid = source.get("source_id", "")
    if sid:
        return f"sid:{sid}"

    # 兜底方案：基于标题的 key
    title = source.get("title", "")
    if title:
        return f"title:{title.strip().lower()[:100]}"

    return f"hash:{hash(str(sorted(source.items())))})"


def _dedup_sources(sources_list: List[Dict]) -> List[Dict]:
    """
    按 去重键 去重 sources。
    防止把同一片论文的预印本和正式本作为两个Work返回，他们的URL,DOI,ID都不同，但标题完全相同。
    优先级：DOI → arXiv ID → normalized URL → source_id。

    合并策略：
    - 同一去重键 → 保留 metadata/full_text 更完整或 quality_score 更高的版本
    - 同一 URL 即使 source_id 不同也必须合并
    """
    merged: Dict[str, Dict] = {}
    title_to_key: Dict[str, str] = {}
    for s in sources_list:
        key = _dedup_key(s)
        normalized_title = re.sub(
            r"[^a-z0-9]+", " ", str(s.get("title", "")).lower()
        ).strip()
        # OpenAlex 可能用不同 ID 和 URL 暴露同一篇论文的预印本/正式版，但标题完全相同，视为同一来源。
        existing_key = key if key in merged else title_to_key.get(normalized_title)

        if existing_key:
            existing = merged[existing_key]
            existing_qs = existing.get("quality_score", 0) or 0
            new_qs = s.get("quality_score", 0) or 0

            existing_fields = sum(1 for v in existing.values() if v)
            new_fields = sum(1 for v in s.values() if v)

            # 保留更好的版本 按照分数来进行比较，保留分数更高的版本，相同则保留字段多/有全文更长的版本
            should_replace = False 
            if new_qs > existing_qs:
                should_replace = True
            elif new_qs == existing_qs and new_fields > existing_fields:  # 字段多的优先
                should_replace = True
            elif "full_text" in s and "full_text" not in existing:  # 有全文的优先
                should_replace = True
            elif len(s.get("full_text", "")) > len(existing.get("full_text", "")):
                should_replace = True

            if should_replace:
                merged[existing_key] = s
        else:
            merged[key] = s
            existing_key = key

        if normalized_title:
            title_to_key[normalized_title] = existing_key

    return list(merged.values())


def _rank_sources_for_topic(sources: List[Dict], topic: str) -> List[Dict]:
    """在应用来源预算之前，先对合并后的搜索结果排序。

    Send worker 完成顺序不确定，因此列表顺序不能代表相关性。
    此处复用确定性的来源打分器，防止不相关的早期结果占用有限的阅读名额。
    """
    if not sources or not topic:
        return sources

    scores = SourceQualityScorer().score_batch(sources, topic)
    ranked = []
    for source in sources:
        item = dict(source)
        score = scores.get(item.get("source_id", ""))
        item["quality_score"] = score.total if score else 0.0
        ranked.append(item)

    return sorted(
        ranked,
        key=lambda source: (
            source.get("quality_score", 0.0),
            bool(source.get("full_text")),
            source.get("cited_by_count") or 0,
        ),
        reverse=True,
    )


def _is_local_full_text_source(source: Dict[str, Any]) -> bool:
    """判断候选项是否为带有证据的本地论文命中。"""
    provider = str(source.get("provider") or "").lower()
    content_source = str(source.get("content_source") or "").lower()
    if provider not in {"local_zotero", "local_rag"} and content_source != "zotero_pdf":
        return False
    if not str(source.get("full_text") or source.get("text") or "").strip():
        return False

    # Zotero 向量检索已应用该阈值，但此处复查可防止格式错误或人为注入的低相关
    # 本地结果仅因 provider 配额而混入。
    if provider == "local_zotero":
        try:
            retrieval_score = float(source.get("retrieval_score") or 0.0)
        except (TypeError, ValueError):
            return False
        if retrieval_score < get_local_rag_min_score():
            return False
    return True


def _cap_sources_with_local_coverage(
    ranked: List[Dict[str, Any]],
    max_sources: int,
) -> List[Dict[str, Any]]:
    """在保留相关本地全文证据的前提下应用来源预算。

    单一的全局排序会系统性偏向新的外部摘要：时效性计入质量分，而许多 Zotero PDF
    没有解析出年份。为符合条件的本地命中预留最多 25% 的预算，并在所选集合内
    保持原有排序。
    """
    cap = max(1, int(max_sources or 1))
    # 绝不因总结果数低于上限，就保留未通过相关性门槛的本地命中。
    ranked = [
        source for source in ranked
        if not (
            str(source.get("provider") or "").lower() in {"local_zotero", "local_rag"}
            or str(source.get("content_source") or "").lower() == "zotero_pdf"
        ) or _is_local_full_text_source(source)
    ]
    if len(ranked) <= cap:
        return ranked

    selected = list(ranked[:cap])
    if cap < 2:
        return selected

    eligible_local = [source for source in ranked if _is_local_full_text_source(source)]
    if not eligible_local:
        return selected

    local_target = min(len(eligible_local), max(1, cap // 4))
    selected_keys = {_dedup_key(source) for source in selected}
    selected_local_count = sum(_is_local_full_text_source(source) for source in selected)
    missing = max(0, local_target - selected_local_count)
    if not missing:
        return selected

    additions = [
        source for source in eligible_local
        if _dedup_key(source) not in selected_keys
    ][:missing]
    if not additions:
        return selected

    removable_indexes = [
        index for index in range(len(selected) - 1, -1, -1)
        if not _is_local_full_text_source(selected[index])
    ]
    for source, index in zip(additions, removable_indexes):
        selected[index] = source

    rank_index = {_dedup_key(source): index for index, source in enumerate(ranked)}
    return sorted(selected, key=lambda source: rank_index[_dedup_key(source)])


def _merge_evidence_cards(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """
    evidence_cards 的 reducer：按 (source_id, claim前80字符) 去重。

    - 保留 confidence 更高的
    - retry 产生的新 card（timestamp 更新）替换同 key 的旧 card
    - 旧无效 EvidenceCard 必须能被替换或删除
    """
    merged: Dict[str, Dict] = {}
    for card in left + right:
        sid = card.get("source_id", "")
        claim_prefix = card.get("claim", "")[:80].strip().lower()
        key = f"{sid}::{claim_prefix}" if sid else f"__noid__::{claim_prefix}"

        if key in merged:
            existing = merged[key]
            # 新 card 的 confidence 更高 → 替换
            new_conf = card.get("confidence", 0)
            old_conf = existing.get("confidence", 0)
            # 如果新 card 有一个更近的 timestamp（retry 产生），优先使用
            new_ts = card.get("_retry_round", 0)
            old_ts = existing.get("_retry_round", 0)
            if new_ts > old_ts or (new_ts == old_ts and new_conf > old_conf):
                merged[key] = card
        else:
            merged[key] = card

    return list(merged.values())


async def _select_sources_with_session_seeds(
    state: Dict[str, Any],
    ranked: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """在自适应来源预算内强制保留 Session Seed，再选择新增候选。"""
    max_sources = max(1, min(int(state.get("max_sources", 5) or 5), 50))
    seed_ids = {str(item) for item in state.get("seed_paper_ids", []) if item}
    if not seed_ids:
        return await AdaptiveSourceSelector().select(
            state.get("topic", ""), ranked, max_sources,
            state.get("agent_mode", "rule"),
            allow_rule_fallback=not _is_llm_only(state),
        )

    def source_id(source: Dict[str, Any]) -> str:
        return str(source.get("source_id") or source.get("paper_id") or "")

    # 关键步骤：Seed 是用户明确指定的研究起点，不参与候选淘汰，仅受 50 篇安全上限约束。
    seeds = [source for source in ranked if source_id(source) in seed_ids][:max_sources]
    seed_keys = {_dedup_key(source) for source in seeds}
    remaining = max_sources - len(seeds)
    extras: List[Dict[str, Any]] = []
    selection: Dict[str, Any] = {"mode": "session_seed", "requested_count": max_sources}
    if remaining > 0:
        candidates = [source for source in ranked if _dedup_key(source) not in seed_keys]
        extras, selection = await AdaptiveSourceSelector().select(
            state.get("topic", ""), candidates, remaining,
            state.get("agent_mode", "rule"),
            allow_rule_fallback=not _is_llm_only(state),
        )
    selection = dict(selection)
    selection.update({
        "mode": "session_seed+" + str(selection.get("mode", "bounded")),
        "seed_paper_ids": [source_id(source) for source in seeds],
        "seed_count": len(seeds),
        "new_candidate_count": len(extras),
        "requested_count": max_sources,
    })
    return seeds + extras, selection


def _restore_session_seeds(
    state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    original_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """将用户明确指定的 Seed 放回过滤后的候选集。"""
    seed_ids = {str(item) for item in state.get("seed_paper_ids", []) if item}
    if not seed_ids:
        return candidates
    seeds = [
        source for source in original_sources
        if str(source.get("source_id") or source.get("paper_id") or "") in seed_ids
    ]
    # Session Seed 是显式用户输入，通用关键词边界和本地来源门不能静默删除它。
    return _dedup_sources(seeds + candidates)


def _tr_event(event: str, **kwargs) -> Dict[str, Any]:
    """创建结构化 trace event（仅返回 delta）。"""
    return {"event": event, "timestamp_ms": int(time.time() * 1000), **kwargs}


def _safe_llm_error_code(purpose: str, status: str) -> str:
    """把内部状态映射为稳定错误码，禁止 provider 原始错误进入 trace/SSE。"""
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(status or "failed").upper()).strip("_")
    return f"{purpose.upper()}_{normalized or 'FAILED'}"


def _is_llm_only(state: Dict[str, Any]) -> bool:
    return bool(state.get("llm_only")) and state.get("agent_mode") == "llm"


# ================================================================
# ResearchAgentState（研究智能体运行状态）
# ================================================================

class ResearchAgentState(TypedDict, total=False):
    topic: str
    research_topic: str
    intent: str
    execution_route: str
    selected_tools: List[str]
    selected_tool_args: Dict[str, Any]
    intent_confidence: float
    requested_count: int
    discovered_source_count: int
    analyzed_source_count: int
    analysis_selection: Dict[str, Any]
    controller_reasoning: str
    max_sources: int
    language: str
    mode: str
    run_eval: bool
    run_id: str
    status: str
    backend: str
    agent_mode: str  # 取值："rule" | "llm"
    llm_only: bool
    task_dag: Dict[str, Any]
    execution_spec: Dict[str, Any]
    work_plan: Dict[str, Any]
    direct_task: Dict[str, Any]
    # ---- 并行 worker accumulator（operator.add append，merge 节点显式去重后追加写入正式字段） ----
    _search_bucket: Annotated[List[Dict[str, Any]], operator.add]
    _reading_bucket: Annotated[List[Dict[str, Any]], operator.add]
    _chapter_bucket: Annotated[List[Dict[str, Any]], operator.add]
    _worker_route_bucket: Annotated[List[str], operator.add]
    _worker_result_bucket: Annotated[List[Dict[str, Any]], operator.add]
    # ---- 正式数据字段（由 merge 节点写入，无 reducer） ----
    sources: List[Dict[str, Any]]
    scored_sources: List[Dict[str, Any]]
    # evidence_cards 仅由当前轮 Merge 写入，可真正覆盖旧轮结果。
    evidence_cards: List[Dict[str, Any]]
    trace: Annotated[List[Dict[str, Any]], operator.add]
    warnings: Annotated[List[str], operator.add]
    # ---- 非并发字段 ----
    citation_check_results: List[Dict[str, Any]]
    citation_summary: Dict[str, Any]
    outline: Dict[str, Any]
    draft_report: str
    final_report: str
    eval_metrics: Dict[str, Any]
    eval_metric_details: Dict[str, Any]
    eval_feedback: List[str]
    review_verdict: Dict[str, Any]
    worker_results: Dict[str, Dict[str, Any]]
    fixes_applied: List[str]
    unresolved_issues: List[str]
    report_completion_ready: bool
    expected_chapter_count: int
    written_chapter_count: int
    report_completion_issues: List[str]
    strict_completion_required: bool
    source_matrix: List[Dict[str, Any]]
    # 循环控制
    replan_count: int
    repair_count: int
    execution_round: int
    retry_count: int
    retry_attempted: bool
    should_retry: bool
    # 延迟
    start_time_ms: int
    total_latency_ms: int
    total_timeout_ms: int
    runtime_budget_id: str
    observability_metrics: Dict[str, Any]
    # ---- Send API 动态分发 ----
    search_tasks: List[Dict[str, Any]]
    reading_tasks: List[Dict[str, Any]]
    current_search_task: Dict[str, Any]
    current_reading_task: Dict[str, Any]
    target_source: Dict[str, Any]  # Send payload 中传递的 source 对象
    current_chapter_task: Dict[str, Any]
    # 统一 Send Worker 只接收自包含 WorkItem；旧字段仅由兼容适配器构造。
    work_item: Dict[str, Any]
    chapter_sources: List[Dict[str, Any]]
    chapter_cards: List[Dict[str, Any]]
    source_number: Dict[str, int]
    # ---- 多轮对话上下文 ----
    session_id: str
    session_context: SessionContext
    conversation_messages: List[Dict[str, Any]]
    memory_prompt: str
    is_follow_up: bool
    reference_expression: str
    resolved_paper_ids: List[str]
    seed_paper_ids: List[str]
    seed_papers: List[Dict[str, Any]]
    resolved_section: Optional[str]
    resolved_refs: Dict[str, Any]
    conversation_operation: str
    clarification_message: str
    fallback_used: bool
    answer: str
    conversation_result: Dict[str, Any]


# ================================================================
# 共享 Agent 实例
# ================================================================

_planner = Planner()
_intent_controller = IntentController()
_controller_agent = ControllerAgent(_intent_controller)
_planner_agent = PlannerAgent()
_draft_reviewer = DraftReviewer()
_evaluator = Evaluator()
_reviewer_agent = ReviewerAgent(_evaluator)
_final_reviewer = FinalReviewer()
_conversation_agent = ConversationAgent()

# LLMWorker——Function Calling 驱动（延迟导入避免循环依赖）
_llm_worker_class = None


def _get_worker_for_task(agent_mode: str):
    """
    根据 agent_mode 返回合适的 Worker 实例。

    agent_mode="llm" → LLMWorker（Function Calling，内部会检查 LLM 可用性）
    agent_mode="rule" / 任何其他 → 规则 Worker
    """
    if agent_mode == "llm":
        global _llm_worker_class
        if _llm_worker_class is None:
            from app.agents.llm_worker import LLMWorker
            _llm_worker_class = LLMWorker
        return _llm_worker_class()
    return Worker()


def _get_fresh_worker(state: ResearchAgentState = None):
    """
    每次调用创建新 Worker 实例（独立 messages/called/budget）。

    有 state 时根据 agent_mode 选择 Worker 类型；
    无 state 时默认 rule Worker（向后兼容）。
    """
    if state is not None:
        return _get_worker_for_task(state.get("agent_mode", "rule"))
    return Worker()


def _get_search_worker(state: ResearchAgentState):
    """在不改变工具超时的前提下，使用更小的自主检索预算。"""
    if state.get("agent_mode") != "llm":
        return Worker()
    from app.agents.llm_worker import LLMWorker, LLMWorkerConfig

    max_calls = max(2, int(os.getenv("SEARCH_MAX_TOOL_CALLS", "4")))
    worker_timeout_ms = max(30_000, int(os.getenv("SEARCH_WORKER_TIMEOUT_MS", "75000")))
    return LLMWorker(LLMWorkerConfig(
        max_tool_calls=max_calls,
        tool_timeout_ms=max(5_000, int(os.getenv("SEARCH_TOOL_TIMEOUT_MS", "30000"))),
        worker_timeout_ms=worker_timeout_ms,
        max_iterations=max(4, max_calls * 2),
    ))


def _build_deps_from_state(state: ResearchAgentState, task_id: str) -> Dict[str, Any]:
    """从 state 构建 Worker 期望的 dependency_results。"""
    deps = {}
    if task_id in ("read", "analyze", "cite"):
        search_ctx = WorkerContext(Task("search", "search", ""))
        search_ctx.add_result("sources", state.get("sources", []))
        search_ctx.add_result("search_results", state.get("sources", []))
        deps["search"] = search_ctx
    if task_id in ("analyze", "cite"):
        read_ctx = WorkerContext(Task("read", "read", ""))
        scored = state.get("scored_sources", [])
        read_ctx.add_result("scored_sources", scored if scored else state.get("sources", []))
        deps["read"] = read_ctx
    if task_id == "cite":
        analyze_ctx = WorkerContext(Task("analyze", "analyze", ""))
        analyze_ctx.add_result("evidence_cards", state.get("evidence_cards", []))
        deps["analyze"] = analyze_ctx
    return deps


def _merge_worker_trace(ctx: WorkerContext) -> List[Dict[str, Any]]:
    """
    将 WorkerContext.trace 转换为结构化 ResearchAgentState trace 事件。

    Worker trace 格式（来自 WorkerContext.add_trace）:
        {"step": N, "task_id": "...", "tool_name": "...", "input_summary": "...",
         "success": bool, "latency_ms": int, "error": "..."}

    Runtime trace 格式（来自 _tr_event）:
        {"event": "...", "timestamp_ms": int, ...}

    本函数将 Worker trace 条目转换为统一的 trace event 格式，
    带 graph_node=False 标记以区分 Worker Tool 事件和 Runtime 节点完成事件。
    """
    events = []
    # 遍历 WorkerContext.trace 中的每个条目
    for entry in ctx.trace:
        tool_name = entry.get("tool_name", entry.get("tool", ""))
        operation_name = entry.get("operation_name") or tool_name
        input_summary = entry.get("input_summary", "")
        success = entry.get("success", True)
        latency = entry.get("latency_ms", 0)
        error = entry.get("error", "")

        # 确定 event type
        if tool_name in ("function_call_started", "tool_selected", "tool_started",
                         "tool_finished", "tool_args_rejected", "tool_loop_finished",
                         "tool_loop_limit_reached", "tool_loop_fallback",
                         "tool_rejected", "llm_finish", "llm_finished", "llm_failed",
                         "provider_fallback", "retrieval_observed",
                         "retrieval_query_rewritten", "retrieval_source_switched",
                         "retrieval_finished"):
            event_type = tool_name
        else:
            event_type = "tool_finished"
        # 构建转换后的事件
        ev = {
            "event": event_type,
            "timestamp_ms": int(time.time() * 1000),
            "graph_node": False,
            "task_id": entry.get("task_id", ""),
            "tool_name": operation_name if event_type == "tool_finished" else tool_name,
            "input_summary": input_summary[:200],
            "success": success,
            "latency_ms": latency,
        }
        if error:
            ev["error"] = str(error)[:200]
        if event_type == "provider_fallback":
            ev["provider"] = entry.get("provider", "mock")
            ev["fallback_reason"] = entry.get("fallback_reason", input_summary)
        if event_type in ("llm_finished", "llm_failed"):
            metadata = entry.get("metadata") or {}
            if isinstance(metadata, dict):
                ev["agent"] = metadata.get("agent", "llm_worker")
                ev["model"] = metadata.get("model", "")
                usage = metadata.get("usage", {})
                if isinstance(usage, dict):
                    ev["usage"] = {
                        key: max(0, int(usage.get(key) or 0))
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
        if event_type.startswith("retrieval_"):
            for key, value in entry.items():
                if key not in {
                    "step", "task_id", "tool_name", "operation_name",
                    "input_summary", "success", "latency_ms", "error", "metadata",
                }:
                    ev[key] = value
            ev["tool_name"] = entry.get("retrieval_tool_name", tool_name)
        events.append(ev)
    return events


def _worker_context_failed(ctx: WorkerContext) -> bool:
    """当本任务中确有工具执行失败时返回 True。"""
    pseudo_tools = {
        "function_call_started", "tool_selected", "tool_started", "tool_finished",
        "tool_args_rejected", "tool_loop_finished", "tool_loop_limit_reached",
        "tool_loop_fallback", "tool_rejected", "llm_finish", "llm_function_call",
        "llm_finished", "llm_failed", "provider_fallback",
        "retrieval_observed", "retrieval_query_rewritten",
        "retrieval_source_switched", "retrieval_finished",
    }
    return any(
        entry.get("tool_name") not in pseudo_tools
        and entry.get("tool_name")
        and not entry.get("success", True)
        for entry in ctx.trace
    )


# ================================================================
# 节点（所有节点只返回 delta）
# ================================================================

# ---- controller 节点 ----

async def node_controller(state: ResearchAgentState) -> Dict[str, Any]:
    # Gate B: 支持已有 run_id（async 模式）
    existing_run_id = state.get("run_id", "")
    if existing_run_id:
        run_id = existing_run_id
        run_store.update(run_id, status="running", backend=state.get("backend", "graph"))
    else:
        run_id = run_store.create(topic=state["topic"])
        run_store.update(run_id, status="running", backend=state.get("backend", "graph"))

    now_ms = int(time.time() * 1000)
    from app.agents.context_loader import context_resource_adapter

    context_resource_adapter.bind_session_snapshot(state.get("session_context"))
    # 关键步骤：生产入口直接调用 ControllerAgent，IntentController 仅作为其内部适配器。
    execution_spec = await _controller_agent.execute(
        user_request=state["topic"],
        request_id=run_id,
        max_sources=state.get("max_sources", 5),
        agent_mode=state.get("agent_mode", "rule"),
        llm_only=bool(state.get("llm_only")),
        session_context=state.get("session_context"),
        memory_prompt=state.get("memory_prompt", ""),
        budget=ExecutionBudget(
            max_workers=max(1, min(int(state.get("max_sources", 5) or 5), 8)),
            # 关键步骤：ReAct 搜索会按来源多次调用工具（多提供商 × 查询改写），
            # 再加上 read/analyze/cite/write/outline/draft/review 各阶段，默认 20
            # 会被 3 个搜索 worker 就耗尽（RUNTIME_BUDGET_EXHAUSTED）。这里放宽为
            # RUNTIME_MAX_TOOL_CALLS（默认 120），墙钟由 total_timeout_ms 兜底。
            max_tool_calls=int(os.getenv("RUNTIME_MAX_TOOL_CALLS", "120")),
            total_timeout_ms=max(
                1,
                int(
                    state.get("total_timeout_ms")
                    or os.getenv("RESEARCH_TOTAL_TIMEOUT_MS", "480000")
                ),
            ),
        ),
    )
    decision = ControllerDecision.model_validate(
        execution_spec.metadata.get("controller_decision") or {}
    )
    register_runtime_budget(
        run_id,
        max_tool_calls=execution_spec.budget.max_tool_calls,
        total_timeout_ms=execution_spec.budget.total_timeout_ms,
    )
    controller_trace = [
        _tr_event("controller_start",
                  node="controller", graph_node=True,
                  run_id=run_id),
        _tr_event(
            "agent_protocol_validated", node="controller", graph_node=False,
            protocol_version="2.0", agent_role="controller",
            architectural_role="controller", task_id="controller",
            allowed_tools=[], tools_called=[], status="success",
        ),
        _tr_event(
            "intent_classified",
            node="intent_controller",
            graph_node=True,
            intent=decision.intent,
            execution_route=decision.execution_route,
            selected_tools=decision.selected_tools,
            confidence=decision.confidence,
            classifier=decision.classifier,
            reasoning=decision.reasoning[:200],
            research_topic=decision.research_topic,
            session_id=decision.session_id or state.get("session_id", ""),
            is_follow_up=decision.is_follow_up,
            reference_expression=decision.reference_expression,
            resolved_resource_ids=decision.resolved_paper_ids,
            resolved_section=decision.resolved_section,
            conversation_operation=decision.conversation_operation,
            clarification_message=decision.clarification_message,
            missing_ordinal=decision.missing_ordinal,
            route_name=decision.execution_route,
            fallback_used=decision.fallback_used,
        ),
    ]
    if decision.llm_result:
        llm_success = bool(decision.llm_result.get("success"))
        controller_trace.append(_tr_event(
            "llm_finished" if llm_success else "llm_failed",
            node="intent_controller",
            agent="intent_controller",
            graph_node=False,
            success=llm_success,
            latency_ms=decision.llm_result.get("latency_ms", 0),
            model=decision.llm_result.get("model", ""),
            usage=decision.llm_result.get("usage", {}),
            error=decision.llm_result.get("error", "")[:200],
        ))
    # 触发 before_run 事件
    emit_before_run({"run_id": run_id, "topic": state["topic"], "node": "controller"})
    session = state.get("session_context")
    seed_ids = set(decision.seed_paper_ids)
    seed_papers = [
        dict(paper) for paper in (session.recommended_papers if isinstance(session, SessionContext) else [])
        if str(paper.get("source_id") or paper.get("paper_id") or "") in seed_ids
    ]
    effective_topic = (
        decision.research_topic
        if decision.intent == "research_from_session"
        else state["topic"]
    )
    return {
        # 关键步骤：续接调研后续所有搜索、筛选和报告节点都使用原推荐主题。
        "topic": effective_topic,
        "run_id": run_id,
        "status": "running",
        "start_time_ms": now_ms,
        "research_topic": decision.research_topic,
        "execution_spec": execution_spec.model_dump(mode="json"),
        "runtime_budget_id": run_id,
        "intent": decision.intent,
        "execution_route": decision.execution_route,
        "selected_tools": decision.selected_tools,
        "selected_tool_args": decision.selected_tool_args,
        "requested_count": decision.requested_count,
        "intent_confidence": decision.confidence,
        "controller_reasoning": decision.reasoning,
        "session_id": decision.session_id or state.get("session_id", ""),
        "is_follow_up": decision.is_follow_up,
        "reference_expression": decision.reference_expression,
        "resolved_paper_ids": decision.resolved_paper_ids,
        "seed_paper_ids": decision.seed_paper_ids,
        "seed_papers": seed_papers,
        "max_sources": max(state.get("max_sources", 5), len(seed_papers)),
        # Send 路径在搜索 worker 返回前先把 Session Seed 放入同一 accumulator。
        "_search_bucket": seed_papers,
        "resolved_section": decision.resolved_section,
        "resolved_refs": {
            "paper_ids": decision.resolved_paper_ids,
            "resolved_section": decision.resolved_section,
            "report_id": _active_report_id(state.get("session_context")),
        },
        "conversation_operation": decision.conversation_operation,
        "clarification_message": decision.clarification_message,
        "missing_ordinal": decision.missing_ordinal,
        "fallback_used": decision.fallback_used,
        "trace": controller_trace,
        "warnings": [],
        "replan_count": 0,
        "repair_count": 0,
        "execution_round": 0,
        "retry_count": 0,
        "retry_attempted": False,
        "should_retry": False,
    }


def route_after_controller(state: ResearchAgentState) -> str:
    """所有请求都必须经过 Planner，Controller 不直接短路到业务节点。"""
    return "planner"


def _active_report_id(context: Any) -> Optional[str]:
    if isinstance(context, SessionContext):
        return context.active_report_id
    if isinstance(context, dict):
        return context.get("active_report_id")
    return None


def _llm_trace(agent: str, result: Dict[str, Any]) -> Dict[str, Any]:
    success = bool(result.get("success"))
    return _tr_event(
        "llm_finished" if success else "llm_failed",
        node=agent,
        agent=agent,
        graph_node=False,
        success=success,
        latency_ms=result.get("latency_ms", 0),
        model=result.get("model", ""),
        usage=result.get("usage", {}),
        error=str(result.get("error") or "")[:200],
    )


async def node_conversation(state: ResearchAgentState) -> Dict[str, Any]:
    if state.get("conversation_operation") == "reference_not_found":
        message = state.get("clarification_message") or "当前会话找不到所指论文。"
        conversation_result = {
            "answer": message,
            "mode": "clarification",
            "strategy": "reference_not_found",
            "cannot_answer": True,
            "cannot_answer_reason": message,
            "conversation_operation": "reference_not_found",
        }
        return {
            "answer": message,
            "final_report": message,
            "conversation_result": conversation_result,
            "resolved_paper_ids": [],
            "route_name": "conversation",
            "trace": [_tr_event(
                "conversation_clarification",
                node="conversation",
                conversation_operation="reference_not_found",
                message=message,
                success=True,
            )],
            "warnings": [message],
        }
    context = context_builder(state)
    response, llm_result = await _conversation_agent.answer(
        ConversationRequest(
            query=state.get("topic", ""),
            context=context,
            operation_hint=state.get("conversation_operation", ""),
            # 重要：把 ResearchRequest.language 沿 Graph state 传入 conversation，
            # 避免多轮追问丢失首轮请求的输出语言约束。
            language=state.get("language", "zh"),
        ),
        agent_mode=state.get("agent_mode", "rule"),
        memory_prompt=state.get("memory_prompt", ""),
    )
    paper_ids = [
        str(paper.get("source_id") or paper.get("paper_id") or "")
        for paper in context.papers
        if paper.get("source_id") or paper.get("paper_id")
    ]
    resources = paper_ids + ([context.report_id] if context.report_id else [])
    conversation_result = response.model_dump()
    # 保留兼容性提示，同时不改变 ResearchResponse 的 schema。
    conversation_result["conversation_operation"] = state.get("conversation_operation", "")
    return {
        "answer": response.answer,
        "final_report": response.answer,
        "conversation_result": conversation_result,
        "resolved_paper_ids": paper_ids,
        "route_name": "conversation",
        "trace": [
            _llm_trace("conversation", llm_result),
            _tr_event(
                "conversation_answered", node="conversation", route_name="conversation",
                mode=response.mode, strategy=response.strategy,
                conversation_operation=state.get("conversation_operation", ""),
                success=not response.cannot_answer,
                session_id=state.get("session_id", ""),
                resolved_resource_ids=resources,
            ),
        ],
        "warnings": response.warnings + (
            [response.cannot_answer_reason] if response.cannot_answer_reason else []
        ),
    }


# ---- planner 节点 ----

async def node_planner(state: ResearchAgentState) -> Dict[str, Any]:
    """生成 Task DAG + Send API 并行子任务（支持 LLM 模式）。"""
    agent_mode = state.get("agent_mode", "rule")
    topic = state["topic"]
    max_sources = state.get("max_sources", 5)

    trace_events = []
    planner_name = "planner"
    use_llm = False

    # ---- Contextual：固定 context_load → answer，两步均无业务工具循环 ----
    if state.get("execution_route") == "conversation":
        resources = list(state.get("seed_papers") or [])
        tasks = [
            Task("context_load", "context_load", "Load explicit session resources"),
            Task("answer", "answer", state.get("topic", ""), depends_on=["context_load"]),
        ]
        task_dag = TaskDAG(topic=topic, tasks=tasks)
        trace_events.append(_tr_event(
            "planner_complete", node="intent_planner", graph_node=True,
            task_count=2, search_count=0, read_count=0, llm_used=False,
            execution_route="conversation",
            work_profiles=["read", "answer"], resource_count=len(resources),
        ))
        return {
            "task_dag": task_dag.to_dict(),
            "search_tasks": [],
            "reading_tasks": [],
            "trace": trace_events,
        }

    # ---- Controller 短路路径：构建满足意图的最小 DAG ----
    if state.get("execution_route") == "direct_tool": # 判断从controller传入的execution_route，true的话创建微型dag
        selected_tools = state.get("selected_tools", [])
        task = Task(
            task_id="direct_1",
            task_type="direct_tool", # 特殊任务类型
            description=state.get("research_topic", topic), # 研究主题
            depends_on=[], # 无依赖
            tool_plan=selected_tools, # 工具规划，例如：["mcp__academic_research_tools__semantic_scholar_recommendations"]
        )
        task_dag = TaskDAG(topic=topic, tasks=[task]) # 创建只有一个节点的微型DAG
        trace_events.append(_tr_event(
            "planner_complete",
            node="intent_planner",
            graph_node=True,
            task_count=1,
            search_count=0,
            read_count=0,
            llm_used=False,
            execution_route="direct_tool",
        ))
        emit_after_plan({
            "task_count": 1,
            "search_count": 0,
            "read_count": 0,
            "task_ids": [task.task_id],
            "dependencies": {task.task_id: []},
            "tool_plan": {task.task_id: selected_tools},
        })
        return {
            "task_dag": task_dag.to_dict(),
            "direct_task": task.to_dict(),
            "search_tasks": [],
            "reading_tasks": [],
            "trace": trace_events,
        }

    # ---- 尝试 LLM Planner ----
    if agent_mode == "llm":
        try:
            from app.agents.llm_planner import LLMPlanner
            llm_planner = LLMPlanner()
            task_dag, llm_result = await llm_planner.plan(
                topic=topic, max_sources=max_sources, mode=state.get("mode", "quick"),
            )
            if llm_result.get("success"):
                use_llm = True
                planner_name = "llm_planner"
                trace_events.append(_tr_event("llm_started",
                                              node="planner", agent="llm_planner",
                                              success=True))
                trace_events.append(_tr_event("llm_finished",
                                              node="planner", agent="llm_planner",
                                              success=True,
                                              latency_ms=llm_result.get("latency_ms", 0),
                                              model=llm_result.get("model", ""),
                                              usage=llm_result.get("usage", {})))
            else:
                # LLM 失败 → 回退规则 Planner
                trace_events.append(_tr_event("llm_failed",
                                              node="planner", agent="llm_planner",
                                              model=llm_result.get("model", ""),
                                              usage=llm_result.get("usage", {}),
                                              latency_ms=llm_result.get("latency_ms", 0),
                                              success=False,
                                              error=llm_result.get("error", "unknown")))
                trace_events.append(_tr_event("llm_fallback",
                                              node="planner",
                                              from_agent="llm_planner",
                                              to_agent="rule_planner"))
                task_dag = _planner.plan_for_send(topic=topic, max_sources=max_sources)
        except Exception as e:
            trace_events.append(_tr_event("llm_failed",
                                          node="planner", agent="llm_planner",
                                          success=False,
                                          error=str(e)[:200]))
            trace_events.append(_tr_event("llm_fallback",
                                          node="planner",
                                          from_agent="llm_planner",
                                          to_agent="rule_planner"))
            task_dag = _planner.plan_for_send(topic=topic, max_sources=max_sources)
    else:
        task_dag = _planner.plan_for_send(topic=topic, max_sources=max_sources)

    search_tasks = []
    reading_tasks = []
    for t in task_dag.tasks:
        if t.task_type == "search":
            search_tasks.append(t.to_dict())
        elif t.task_type == "read":
            reading_tasks.append(t.to_dict())
    # 记录 Planner 完成事件
    trace_events.append(_tr_event("planner_complete",
                                  node=planner_name, graph_node=True,
                                  task_count=len(task_dag.tasks),
                                  search_count=len(search_tasks),
                                  read_count=len(reading_tasks),
                                  llm_used=use_llm))
    # 触发 after_plan 事件
    emit_after_plan({
        "task_count": len(task_dag.tasks),
        "search_count": len(search_tasks),
        "read_count": len(reading_tasks),
        "task_ids": [t.task_id for t in task_dag.tasks],
        "dependencies": {t.task_id: t.depends_on for t in task_dag.tasks},
        "tool_plan": {t.task_id: t.tool_plan for t in task_dag.tasks},
    })

    return {
        "task_dag": task_dag.to_dict(),
        "search_tasks": search_tasks,
        "reading_tasks": reading_tasks,
        "trace": trace_events,
    }


# 长期/短期判断路由
def route_after_planner_send(state: ResearchAgentState):
    """感知 Send 的 planner 路由：直接任务或并行搜索 worker。"""
    if state.get("execution_route") == "conversation":
        return "conversation"
    if state.get("execution_route") == "direct_tool":
        return "capability_worker"
    return send_search_work_items(state)


async def node_four_agent_planner(state: ResearchAgentState) -> Dict[str, Any]:
    """生产 Planner 节点；旧 Planner/LLMPlanner 不再作为主图角色。"""
    spec = ExecutionSpec.model_validate(state.get("execution_spec") or {})
    previous = state.get("work_plan")
    verdict_data = state.get("review_verdict") or {}
    is_replan = bool(previous and verdict_data.get("outcome") == "replan")
    execution_round = int(state.get("execution_round", 0) or 0) + (1 if is_replan else 0)
    if is_replan:
        plan = await _planner_agent.replan_hybrid(
            WorkPlan.model_validate(previous),
            ReviewVerdict.model_validate(verdict_data),
        )
    else:
        plan = await _planner_agent.plan_hybrid(spec)

    for item in plan.items:
        item.metadata["revision"] = plan.revision
        item.metadata["round_id"] = execution_round
        item.metadata["runtime_budget_id"] = str(state.get("runtime_budget_id") or "")
        if item.profile == WorkerProfile.SEARCH:
            item.input_data.update({
                "max_sources": state.get("max_sources", 5),
                "agent_mode": state.get("agent_mode", "rule"),
            })

    task_dag = {
        "topic": spec.research_topic,
        "tasks": [
            {
                "task_id": item.task_id,
                "task_type": item.profile.value,
                "description": item.instruction,
                "depends_on": item.depends_on,
                "tool_plan": item.allowed_tools,
            }
            for item in plan.items
        ],
        "task_count": len(plan.items),
        "revision": plan.revision,
    }
    emit_after_plan({
        "task_count": len(plan.items),
        "task_ids": [item.task_id for item in plan.items],
        "dependencies": {item.task_id: item.depends_on for item in plan.items},
        "tool_plan": {item.task_id: item.allowed_tools for item in plan.items},
        "revision": plan.revision,
    })
    trace_events = [_tr_event(
        "planner_complete", node="planner_agent", graph_node=True,
        task_count=len(plan.items), revision=plan.revision,
        execution_class=spec.execution_class.value,
    ), _tr_event(
        "agent_protocol_validated", node="planner_agent", graph_node=False,
        protocol_version="2.0", agent_role="planner",
        architectural_role="planner", task_id="planner",
        allowed_tools=[], tools_called=[], status="success",
    )]
    hybrid_info = dict(plan.metadata.get("hybrid_planner") or {})
    if hybrid_info.get("attempted"):
        llm_success = bool(hybrid_info.get("success"))
        trace_events.append(_tr_event(
            "llm_finished" if llm_success else "llm_failed",
            node="planner_agent", graph_node=True,
            purpose="hybrid_planning", status=hybrid_info.get("status", ""),
            success=llm_success,
            error_code=("" if llm_success else _safe_llm_error_code(
                "hybrid_planner", hybrid_info.get("status", "failed"),
            )),
            recoverable=not llm_success,
            model=hybrid_info.get("model", ""),
            latency_ms=int(hybrid_info.get("latency_ms") or 0),
            usage=dict(hybrid_info.get("usage") or {}),
        ))
    for item in plan.items:
        if item.metadata.get("fallback_from"):
            trace_events.append(_tr_event(
                "tool_loop_fallback", node="planner_agent", graph_node=True,
                from_tool=item.metadata["fallback_from"],
                to_tool=(item.allowed_tools[0] if item.allowed_tools else ""),
                explicit_work_item=True, revision=plan.revision,
            ))
    return {
        "work_plan": plan.model_dump(mode="json"),
        "task_dag": task_dag,
        "search_tasks": [
            task for task in task_dag["tasks"] if task["task_type"] == "search"
        ],
        "replan_count": plan.replan_count,
        "execution_round": execution_round,
        "trace": trace_events,
    }


def _send_explicit_work_items(
    state: ResearchAgentState,
    profiles: set[WorkerProfile],
    task_ids: Optional[set[str]] = None,
) -> List[Send]:
    # 关键步骤：从 WorkPlan 挑出指定 Profile（可再限定 task_ids）的 WorkItem，
    # 逐个深拷贝成 Send 消息——保证每次执行的 Worker 状态互不共享。
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    round_id = int(state.get("execution_round", 0) or 0)
    selected = [
        item for item in plan.items
        if item.profile in profiles and (task_ids is None or item.task_id in task_ids)
    ]
    sends: List[Send] = []
    for original in selected:
        item = original.model_copy(deep=True)
        item.metadata.update({"round_id": round_id, "revision": plan.revision})
        sends.append(Send("worker_send", {
            "work_item": item.model_dump(mode="json"),
            "explicit_resources": list(item.resources),
        }))
    return sends


def route_after_four_agent_planner(state: ResearchAgentState):
    """Planner 仅派发当前执行类所需的最小首批 Worker。

    关键步骤：整个 DAG 只从这里"扇出"第一批 Worker。
    - 后续阶段不靠这里触发，而是靠各 node_merge_* 在汇总完成后再 Send 下一批。
    - 重规划时只重派受影响子图的就绪根，未受影响节点保留原结果，不整条重跑。
    """
    spec = ExecutionSpec.model_validate(state.get("execution_spec") or {})
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    # ── 重规划路径（revision > 0）：只重派受影响的"就绪根"节点。
    if plan.revision > 0:
        affected = [item for item in plan.items if item.metadata.get("replanned")]
        affected_ids = {item.task_id for item in affected}
        # 就绪根 = 受影响节点中，其依赖也都不在受影响集合内的节点
        # （即该子图最上游，无需等待本次重规划的其它节点）。
        roots = [
            item for item in affected
            if not affected_ids.intersection(item.depends_on)
        ]
        profiles = {item.profile for item in roots}
        root_ids = {item.task_id for item in roots}
        # 关键步骤：按受影响子图的就绪根调度，READ 可直接复用成功 search/sources。
        # 依根节点的 Profile 分发到对应阶段：多合一走通用 Send，READ/ANALYZE/CITE 单阶段各走专用 Send。
        if profiles <= {WorkerProfile.DIRECT, WorkerProfile.CONTEXT_LOAD, WorkerProfile.SEARCH}:
            return _send_explicit_work_items(state, profiles, root_ids)
        if profiles == {WorkerProfile.READ}:
            return send_reading_work_items(state)
        if profiles == {WorkerProfile.ANALYZE}:
            return send_analysis_work_item(state)
        if profiles == {WorkerProfile.CITE}:
            return send_citation_work_item(state)
        if profiles == {WorkerProfile.WRITE}:
            # 降级子图（证据抽取失败）整篇出一份受控报告，否则按章节分派。
            if any(item.metadata.get("degraded_source_only") for item in roots):
                return send_report_work_item(state)
            return send_chapter_work_items(state)
        raise RuntimeError(f"replan 产生无法调度的多阶段就绪根: {sorted(p.value for p in profiles)}")
    # ── 首次执行：按执行类只派 DAG 根节点，后续阶段由 Merge 完成事件逐级触发。
    # atomic -> 只派 DIRECT（单工具直连，不检索不阅读）。
    if spec.execution_class.value == "atomic":
        return _send_explicit_work_items(state, {WorkerProfile.DIRECT})
    # contextual -> 只派 CONTEXT_LOAD（先装载 Controller 解析出的会话资源）。
    if spec.execution_class.value == "contextual":
        return _send_explicit_work_items(state, {WorkerProfile.CONTEXT_LOAD})
    # full_research -> 只派 SEARCH（三路并行检索的根节点）。
    return _send_explicit_work_items(state, {WorkerProfile.SEARCH})

# 执行工具
async def node_capability_worker(state: ResearchAgentState) -> Dict[str, Any]:
    """无报告管道，直接执行选中的能力。"""
    selected = list(state.get("selected_tools", []))
    # 获取工具名
    tool_name = selected[0] if selected else "academic_search"
    # 从ToolRegistry拿到MCPToolAdaptor实例
    registry = ToolRegistry.get_instance() 
    tool = registry.get(tool_name) # → 返回 MCPToolAdapter 实例!
    research_topic = state.get("research_topic") or state.get("topic", "")
    limit = min(int(state.get("requested_count", state.get("max_sources", 5)) or 5), 50)
    session = state.get("session_context")
    prior_papers = (
        session.recommended_papers
        if state.get("intent") == "recommend_more" and isinstance(session, SessionContext)
        else []
    )
    prior_keys = {_dedup_key(item) for item in prior_papers}
    # “再推荐”需要多取候选后排除历史论文，否则工具只返回 10 条时可能一条新论文都没有。
    search_limit = min(50, limit + len(prior_keys)) if prior_keys else limit
    trace_events = [_tr_event(
        "worker_started",
        node="capability_worker",
        worker_type="capability",
        task_id="direct_1",
        tool_name=tool_name,
        success=True,
    )]
    warnings = []

    if tool is None:
        warnings.append(f"Selected capability unavailable: {tool_name}")
        tool_name = "academic_search"
        tool = registry.get(tool_name)

    result = None
    # 构建参数 (research_topic + limit 映射到 MCP 工具的 schema)
    if tool is not None:
        args = _build_direct_tool_args(
            tool.input_schema,
            research_topic,
            search_limit,
            state.get("selected_tool_args", {}),
        )
        # 调用的是BaseTool.run(),它负责计时和异常兜底，然后委托给_arun()
        result = await tool.run(**args)
        trace_events.append(_tr_event(
            "tool_finished",
            node="capability_worker",
            graph_node=False,
            task_id="direct_1",
            tool_name=tool_name,
            success=result.success,
            latency_ms=result.latency_ms,
            error=(result.error or "")[:200],
            metadata=result.metadata,
        ))

    # 远程 MCP 推荐服务能暂时不可用。
    # 回退到内置 provider，不重新进入完整 DAG。
    if (result is None or not result.success) and tool_name != "academic_search":
        fallback = registry.get("academic_search")
        warnings.append(
            f"Capability '{tool_name}' failed; used academic_search fallback"
        )
        trace_events.append(_tr_event(
            "tool_loop_fallback",
            node="capability_worker",
            from_tool=tool_name,
            to_tool="academic_search",
        ))
        tool_name = "academic_search"
        if fallback is not None:
            # 参数按 MCP Schema 格式构建
            args = _build_direct_tool_args(fallback.input_schema, research_topic, search_limit)
            result = await fallback.run(**args)
            trace_events.append(_tr_event(
                "tool_finished",
                node="capability_worker",
                graph_node=False,
                task_id="direct_1",
                tool_name=tool_name,
                success=result.success,
                latency_ms=result.latency_ms,
                error=(result.error or "")[:200],
            ))

    data = result.data if result is not None and result.success else {}
    # 对MCP工具返回的结果进行解析，解析出sources,score_sources并与记录单 traces 和 warnings 一起返回
    sources = []
    if isinstance(data, dict):
        sources = data.get("sources") or data.get("results") or []
    sources = _rank_sources_for_topic(
        _dedup_sources(sources if isinstance(sources, list) else []),
        research_topic,
    )
    if prior_keys:
        # 关键步骤：只向本轮结果暴露从未推荐过的论文，SessionStore 再负责原子追加。
        sources = [source for source in sources if _dedup_key(source) not in prior_keys]
    sources = sources[:limit]
    success = bool(result and result.success and sources)
    if not success:
        warnings.append(
            (result.error if result is not None else "No capability available")
            or "Selected capability returned no sources"
        )
    trace_events.append(_tr_event(
        "worker_finished",
        node="capability_worker",
        worker_type="capability",
        task_id="direct_1",
        tool_name=tool_name,
        success=success,
        source_count=len(sources),
    ))
    return {
        "sources": sources,
        "scored_sources": sources,
        "trace": trace_events,
        "warnings": warnings,
    }

# 回到 node_capability_worker，得到 result: ToolResult 后
# 将 sources 格式化为MD报告
async def node_direct_reviewer(state: ResearchAgentState) -> Dict[str, Any]:
    """基于已有来源生成受限的推荐回答，不做虚假的研究断言。"""
    intent = state.get("intent", "literature_search")
    research_topic = state.get("research_topic") or state.get("topic", "")
    sources = state.get("sources", [])
    session = state.get("session_context")
    prior_papers = (
        list(session.recommended_papers)
        if intent == "recommend_more" and isinstance(session, SessionContext)
        else []
    )
    prior_numbers = {
        _dedup_key(source): index
        for index, source in enumerate(prior_papers, start=1)
    }
    next_number = len(prior_papers) + 1
    display_numbers: List[int] = []
    # 关键步骤：继续推荐沿用会话总序号，新论文从历史数量之后开始编号。
    for source in sources:
        key = _dedup_key(source)
        number = prior_numbers.get(key)
        if number is None:
            number = next_number
            prior_numbers[key] = number
            next_number += 1
        display_numbers.append(number)
    heading = (
        "论文推荐"
        if intent in {"paper_recommendation", "recommend_more"}
        else ("论文图谱结果" if intent == "paper_graph_lookup" else "论文检索结果")
    )
    lines = [f"# {heading}：{research_topic}", ""]
    if not sources:
        lines.append("当前可用工具没有返回匹配的论文来源。")
    else:
        if intent == "recommend_more":
            lines.append(
                f"本次新增 {len(sources)} 个可追溯来源，"
                f"会话累计 {len(prior_numbers)} 篇论文："
            )
        else:
            lines.append(f"共找到 {len(sources)} 个可追溯来源：")
        lines.append("")
        for index, source in zip(display_numbers, sources):
            title = source.get("title") or "Untitled"
            url = source.get("url") or ""
            year = source.get("year")
            venue = source.get("venue") or ""
            authors = source.get("authors") or []
            author_text = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors)
            metadata = " · ".join(
                str(value) for value in (year, venue, author_text) if value
            )
            lines.append(f"## {index}. [{title}]({url})" if url else f"## {index}. {title}")
            if metadata:
                lines.append(metadata)
            graph_metadata = []
            if source.get("cited_by_count") is not None:
                graph_metadata.append(f"被引 {source['cited_by_count']}")
            if source.get("reference_count") is not None:
                graph_metadata.append(f"参考文献 {source['reference_count']}")
            if source.get("provider"):
                graph_metadata.append(f"来源 {source['provider']}")
            if graph_metadata:
                lines.append(" · ".join(graph_metadata))
            snippet = (source.get("snippet") or source.get("full_text") or "").strip()
            if snippet:
                lines.append("")
                lines.append(snippet[:500].rstrip())
            lines.append("")

    report = "\n".join(lines).strip()
    urls_valid = all(bool(source.get("url")) for source in sources)
    metrics = {
        "no_fake_citation": True,
        "min_sources": bool(sources),
        "citation_id_exists": True,
        "source_url_valid": urls_valid,
        "answer_not_empty": bool(report),
        "evidence_available": True,
        "task_success_rate": 1.0 if sources else 0.0,
        "tool_error_rate": 0.0 if sources else 1.0,
        "latency_under_threshold": True,
    }
    now_ms = int(time.time() * 1000)
    return {
        "draft_report": report,
        "final_report": report,
        "eval_metrics": metrics,
        "eval_feedback": [],
        "review_verdict": {
            "outcome": "pass" if sources else "fail",
            "failed_task_ids": [] if sources else ["direct_1"],
            "repair_scope": [],
            "feedback": [] if sources else [{
                "code": "DIRECT_NO_RESULTS",
                "message": "直接能力没有返回可验收来源",
            }],
            "summary": "直接结果通过验收" if sources else "直接结果验收失败",
        },
        "citation_check_results": [],
        "citation_summary": {"total_checked": 0, "valid_count": 0, "invalid_count": 0},
        "total_latency_ms": now_ms - state.get("start_time_ms", now_ms),
        "trace": [_tr_event(
            "direct_reviewer_complete",
            node="direct_reviewer",
            graph_node=True,
            intent=intent,
            source_count=len(sources),
            recommendation_number_start=(display_numbers[0] if display_numbers else None),
            recommendation_number_end=(display_numbers[-1] if display_numbers else None),
            report_len=len(report),
        )],
    }


def _build_direct_tool_args(
    schema: Dict[str, Any], # 工具的 input_schema 格式定义
    research_topic: str,
    limit: int,
    selected_tool_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """将 Controller 的可信请求字段映射到 MCP/本地工具 schema。"""
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    args: Dict[str, Any] = {
        key: value
        for key, value in (selected_tool_args or {}).items()
        if key in properties # 仅保留工具定义的参数
    }
    for key in ("topic", "query"): # 把 research_topic 赋值给 topic 或 query 参数
        if key in properties:
            args[key] = research_topic
            break
    for key in ("limit", "max_results", "top_k"): # 把 limit 赋值给 limit 或 max_results 或 top_k 参数
        if key in properties:
            args[key] = limit
            break
    return args


# ---- analysis_worker（支持 retry 时只处理失败的 source） ----

async def node_analysis_worker(state: ResearchAgentState) -> Dict[str, Any]:
    # 输入准备
    deps = _build_deps_from_state(state, "analyze")
    all_sources = state.get("scored_sources", [])
    if not all_sources:
        all_sources = state.get("sources", [])
    retry_count = state.get("retry_count", 0)

    if retry_count > 0:
        # 重试分支：本节点被 evaluator 重新进入，只修复上次没达标的论文，不全量重来
        prev_citations = state.get("citation_check_results", [])
        # 收集上一次引用校验失败（is_valid=False）的论文 id —— 这就是"需要重抽"的目标集合
        failed_source_ids = {
            r["source_id"] for r in prev_citations
            if not r.get("is_valid", False)
        }
        if failed_source_ids:
            # 定向筛选：只保留失败的那几篇，作为本轮重新抽取证据的来源
            sources_to_reprocess = [
                s for s in all_sources
                if s.get("source_id") in failed_source_ids
            ]
            # 重建 "read" 依赖上下文，使其只含本轮重抽的子集，
            # 与协议 depends_on=["read"] 保持一致（依赖摘要不再携带全量旧论文）
            read_ctx = WorkerContext(Task("read", "read", ""))
            read_ctx.add_result("scored_sources", sources_to_reprocess)
            deps["read"] = read_ctx
        else:
            # 没有失败论文（重试但校验结果为空/全部通过）→ 兜底：重抽全部
            sources_to_reprocess = all_sources
    else:
        # 首次执行：没有重试，直接处理全部来源
        sources_to_reprocess = all_sources
    # 协议包裹开启：限制 LLM 调用范围，只允许 evidence_extract 工具调用
    protocol_task = _open_protocol_task(
        state,
        role=AgentRole.READING,
        task_id="analyze",
        input_data={
            "topic": state.get("topic", ""),
            "sources": sources_to_reprocess,
            "operation": "evidence_extraction",
        },
        allowed_tools=["evidence_extract"],
        depends_on=["read"],
    )

    # 证据提取是确定性的且按来源隔离。并行运行所有选中的论文，
    # 避免自适应语料库超过旧的 20 次调用上限时被静默截断。
    extract_tool = EvidenceExtractTool()

    # 单个实例。每个 source 一个协程；返回 (source, tool_result) 二元组，
    # 便于后面把证据卡回挂到它来自的论文（source 隔离，互不污染）。
    async def extract(source: Dict[str, Any]):
        return source, await extract_tool.run(
            source=source, topic=state.get("topic", ""),
        )

    # 在协议权限作用域内并行抽取：所有工具调用都会在 BaseTool.run
    # 里被 role/任务级授权门控（reading 角色，allowed_tools=["evidence_extract"]）。
    # return_exceptions=True 是关键——单篇失败返回 Exception 而不是整批抛异常。
    async with _agent_permission_scope(protocol_task):
        extracted = await asyncio.gather(
            *(extract(source) for source in sources_to_reprocess),
            return_exceptions=True,
        )
    new_cards = []
    worker_trace = []
    analysis_warnings = []
    for item in extracted:
        # 逐项失败隔离：某篇论文抽取失败只记告警并跳过，不影响其他论文的卡。
        if isinstance(item, Exception):
            analysis_warnings.append(f"[evidence_extract] {type(item).__name__}: {str(item)[:160]}")
            continue
        source, tool_result = item  # 解包二元组，拿回该卡所属的 source
        worker_trace.append({      # 写结构化 trace，供 SSE / Harness 审计该次抽取
            "task_id": "analyze",
            "tool_name": "evidence_extract",
            "operation_name": tool_result.tool_name or "evidence_extract",
            "input_summary": f"source_id={source.get('source_id', '?')}",
            "success": bool(tool_result.success),
            "latency_ms": tool_result.latency_ms,
            "error": tool_result.error if not tool_result.success else None,
        })
        if tool_result.success:
            # 成功：把证据卡追加进本次产出；reducer 会按 _retry_round 合并旧卡
            new_cards.extend((tool_result.data or {}).get("evidence_cards", []))
        else:
            # 失败：只记告警，不让单篇失败污染整体证据集
            analysis_warnings.append(f"[evidence_extract] {tool_result.error}")

    if retry_count > 0:
        prev_cards = state.get("evidence_cards", [])
        failed_ids = {
            r["source_id"] for r in state.get("citation_check_results", [])
            if not r.get("is_valid", False)
        }
        # 保留之前非失败 source 的有效 card
        kept_cards = [c for c in prev_cards if c.get("source_id") not in failed_ids]
        # 标记 retry card 的 _retry_round，确保 reducer 优先选择新 card
        for card in new_cards:
            card["_retry_round"] = retry_count
        evidence_cards = kept_cards + new_cards
    else:
        evidence_cards = new_cards

    protocol_event = _close_protocol_task(
        protocol_task,
        output_data={"sources": sources_to_reprocess, "evidence_cards": evidence_cards},
        trace=worker_trace,
        warnings=analysis_warnings,
    )
    return {
        "evidence_cards": evidence_cards,
        "retry_count": retry_count,
        "trace": [protocol_event, _tr_event("analysis_complete",
                            node="analysis_worker", graph_node=True,
                            card_count=len(evidence_cards),
                            retry_count=retry_count,
                            reprocessed=len(sources_to_reprocess))] + worker_trace,
        "warnings": analysis_warnings,
    }


# ---- citation_worker 节点 ----

async def node_citation_worker(state: ResearchAgentState) -> Dict[str, Any]:
    deps = _build_deps_from_state(state, "cite")
    protocol_task = _open_protocol_task(
        state,
        role=AgentRole.CITATION,
        task_id="cite",
        input_data={
            "sources": state.get("sources", []),
            "evidence_cards": state.get("evidence_cards", []),
        },
        allowed_tools=["citation_check"],
        depends_on=["analyze"],
    )
    task = Task(
        "cite", "cite", "Check citation validity", depends_on=["analyze"],
        tool_plan=protocol_task.allowed_tools,
    )
    async with _agent_permission_scope(protocol_task):
        ctx = await _get_fresh_worker(state).execute_task(task, dependency_results=deps)

    summary = ctx.results.get("citation_summary", {})
    citation_results = ctx.results.get("citation_check_results", [])
    protocol_event = _close_protocol_task(
        protocol_task,
        output_data={
            "citation_check_results": citation_results,
            "citation_summary": summary,
        },
        trace=ctx.trace,
        warnings=ctx.warnings,
    )
    return {
        "citation_check_results": citation_results,
        "citation_summary": summary,
        "trace": [protocol_event, _tr_event("citation_complete",
                            node="citation_worker", graph_node=True,
                            total_checked=summary.get("total_checked", 0),
                            valid_count=summary.get("valid_count", 0))] + _merge_worker_trace(ctx),
        "warnings": ctx.warnings,
    }


# ---- outline（已验证证据 -> 明确的章节计划） ----

async def node_outline(state: ResearchAgentState) -> Dict[str, Any]:
    from app.agents.report_outline import ReportOutlineGenerator

    outline, generation = await ReportOutlineGenerator().generate(
        topic=state.get("topic", ""),
        sources=state.get("sources", []),
        evidence_cards=state.get("evidence_cards", []),
        citation_check_results=state.get("citation_check_results", []),
        language=state.get("language", "zh"),
        agent_mode=state.get("agent_mode", "rule"),
        allow_rule_fallback=not _is_llm_only(state),
    )
    events = [_tr_event(
        "outline_created", node="outline", graph_node=True,
        section_count=len(outline.get("sections", [])),
        evidence_gap_count=len(outline.get("evidence_gaps", [])),
        mode="llm" if generation.get("success") and generation.get("mode") != "rule" else "rule",
    )]
    if generation.get("latency_ms") is not None:
        initial_success = bool(generation.get("success"))
        events.append(_tr_event(
            "llm_finished" if initial_success else "llm_failed",
            node="outline", agent="outline_generator", success=initial_success,
            latency_ms=generation.get("latency_ms", 0), model=generation.get("model", ""),
            purpose="outline_generation",
            error_code=("" if initial_success else "OUTLINE_GENERATION_FAILED"),
            recoverable=not initial_success,
            usage=generation.get("usage", {}),
        ))
    repair = dict(generation.get("repair") or {})
    if repair.get("attempted"):
        repair_success = bool(repair.get("success"))
        repair_status = "accepted" if repair_success else (
            "timeout" if "timeout" in str(repair.get("error") or "").lower() else "rejected"
        )
        # 关键步骤：第二次 Outline LLM repair 使用独立安全事件，不携带正文或原始错误。
        events.append(_tr_event(
            "llm_finished" if repair_success else "llm_failed",
            node="outline", agent="outline_repair", purpose="outline_repair",
            success=repair_success, status=repair_status,
            error_code=("" if repair_success else _safe_llm_error_code(
                "outline_repair", repair_status,
            )),
            recoverable=not repair_success,
            latency_ms=int(repair.get("latency_ms") or 0),
            model=str(repair.get("model") or ""), usage=dict(repair.get("usage") or {}),
        ))
    return {"outline": outline, "trace": events}


# ---- draft_reviewer 节点 ----

async def node_draft_reviewer(state: ResearchAgentState) -> Dict[str, Any]:
    """生成报告草稿（支持 LLM 模式，可降级为模板）。"""
    agent_mode = state.get("agent_mode", "rule")
    trace_events = []
    reviewer_node = "draft_reviewer"
    protocol_task = _open_protocol_task(
        state,
        role=AgentRole.REVIEWER,
        task_id="draft_reviewer",
        input_data={
            "topic": state.get("topic", ""),
            "stage": "draft",
            "sources": state.get("sources", []),
            "evidence_cards": state.get("evidence_cards", []),
            "citation_check_results": state.get("citation_check_results", []),
            "outline": state.get("outline", {}),
            "draft_report": "",
        },
        allowed_tools=[],
        depends_on=["cite"],
    )

    result = None
    if agent_mode == "llm":
        try:
            from app.agents.llm_reviewer import LLMDraftReviewer
            llm_reviewer = LLMDraftReviewer()
            result, llm_result = await llm_reviewer.review(
                topic=state["topic"],
                sources=state.get("sources", []),
                evidence_cards=state.get("evidence_cards", []),
                citation_check_results=state.get("citation_check_results", []),
                citation_summary=state.get("citation_summary", {}),
                language=state.get("language", "zh"),
                outline=state.get("outline", {}),
            )
            for chapter in llm_result.get("chapter_results", []):
                trace_events.append(_tr_event(
                    "chapter_generated", node="draft_reviewer",
                    heading=chapter.get("heading", ""), mode=chapter.get("mode", "llm"),
                    success=chapter.get("success", True), latency_ms=chapter.get("latency_ms", 0),
                    model=chapter.get("model", ""), error=chapter.get("error", ""),
                ))
            if llm_result.get("success"):
                reviewer_node = "llm_draft_reviewer"
                trace_events.append(_tr_event("llm_finished",
                                              node="draft_reviewer",
                                              agent="llm_reviewer", success=True,
                                              latency_ms=llm_result.get("latency_ms", 0),
                                              model=llm_result.get("model", ""),
                                              usage=llm_result.get("usage", {})))
            else:
                trace_events.append(_tr_event("llm_failed",
                                              node="draft_reviewer", agent="llm_reviewer",
                                              model=llm_result.get("model", ""),
                                              usage=llm_result.get("usage", {}),
                                              latency_ms=llm_result.get("latency_ms", 0),
                                              success=False,
                                              error=llm_result.get("error", "")))
                trace_events.append(_tr_event("llm_fallback",
                                              node="draft_reviewer",
                                              from_agent="llm_reviewer",
                                              to_agent="rule_reviewer"))
                # LLMDraftReviewer 已返回确定性的降级结果。
                # 保留该结果，使警告与失败原因得以保留。
                reviewer_node = "rule_draft_reviewer"
        except Exception as e:
            trace_events.append(_tr_event("llm_failed",
                                          node="draft_reviewer", agent="llm_reviewer",
                                          success=False,
                                          error=str(e)[:200]))
            trace_events.append(_tr_event("llm_fallback",
                                          node="draft_reviewer",
                                          from_agent="llm_reviewer",
                                          to_agent="rule_reviewer"))
            result = None

    # 降级为基于规则的 DraftReviewer
    if result is None:
        result = _draft_reviewer.review(
            topic=state["topic"],
            sources=state.get("sources", []),
            evidence_cards=state.get("evidence_cards", []),
            citation_check_results=state.get("citation_check_results", []),
            citation_summary=state.get("citation_summary", {}),
            language=state.get("language", "zh"),
            outline=state.get("outline", {}),
        )

    if not any(event.get("event") == "chapter_generated" for event in trace_events):
        for chapter in result.get("chapter_timings", []):
            trace_events.append(_tr_event(
                "chapter_generated", node="draft_reviewer",
                heading=chapter.get("heading", ""), mode=chapter.get("mode", "rule"),
                success=True, latency_ms=chapter.get("latency_ms", 0),
            ))

    trace_events.append(_tr_event("draft_reviewer_complete",
                                  node=reviewer_node, graph_node=True,
                                  report_len=len(result["draft_report"])))
    trace_events.insert(0, _close_protocol_task(
        protocol_task,
        output_data={
            "report": result["draft_report"],
            "completion_ready": True,
            "issues": result.get("warnings", []),
        },
        warnings=result.get("warnings", []),
    ))

    return {
        "draft_report": result["draft_report"],
        "trace": trace_events,
        "warnings": result.get("warnings", []),
    }


# ---- evaluator 节点 ----

async def node_evaluator(state: ResearchAgentState) -> Dict[str, Any]:
    if not state.get("run_eval", True):
        return {"eval_metrics": {}, "eval_feedback": []}

    now_ms = int(time.time() * 1000)
    total_latency_ms = now_ms - state.get("start_time_ms", now_ms)

    eval_result = _evaluator.evaluate(
        topic=state["topic"],
        draft_report=state.get("draft_report", ""),
        sources=state.get("sources", []),
        evidence_cards=state.get("evidence_cards", []),
        citation_check_results=state.get("citation_check_results", []),
        citation_summary=state.get("citation_summary", {}),
        trace=state.get("trace", []),
        task_dag=state.get("task_dag", {}),
        total_latency_ms=total_latency_ms,
    )

    metrics = eval_result.get("metrics", {})

    research_validation_failed = (
        not metrics.get("no_fake_citation", True)
        or not metrics.get("citation_id_exists", True)
        or not metrics.get("source_url_valid", True)
        or not metrics.get("evidence_available", True)
    )
    retry_count = state.get("retry_count", 0)
    should_retry = research_validation_failed and retry_count < 1
    if should_retry:
        retry_count += 1

    detail = eval_result.get("metrics_detail", {})
    return {
        "eval_metrics": metrics,
        "eval_metric_details": detail,
        "eval_feedback": eval_result.get("feedback", []),
        "total_latency_ms": total_latency_ms,
        "retry_count": retry_count,
        "retry_attempted": state.get("retry_attempted", False) or should_retry,
        "should_retry": should_retry,
        "trace": [_tr_event("evaluator_complete",
                            node="evaluator", graph_node=True,
                            passed=detail.get("passed_count", 0),
                            total=detail.get("total_count", 0),
                            latency_ms=total_latency_ms,
                            retry_count=retry_count)],
    }


async def node_reviewer(state: ResearchAgentState) -> Dict[str, Any]:
    """统一 Reviewer 节点：验收并选择 pass/repair/replan/fail。"""
    delta = await node_evaluator(state)
    merged = {**state, **delta}
    feedback = [
        {"code": "EVALUATION_FAILED", "message": str(message)}
        for message in delta.get("eval_feedback", [])
    ]
    failed_tasks: List[str] = []
    repair_scope: List[str] = []
    replan_count = int(state.get("replan_count", 0) or 0)
    if delta.get("should_retry"):
        outcome = "repair"
        failed_tasks = ["analyze", "cite"]
        repair_scope = ["analyze", "cite"]
    elif not merged.get("sources") or not merged.get("evidence_cards"):
        if replan_count < 1:
            outcome = "replan"
            replan_count += 1
            failed_tasks = ["search", "read", "analyze"]
        else:
            outcome = "fail"
            failed_tasks = ["search", "read", "analyze"]
    else:
        outcome = "pass"
    verdict = {
        "outcome": outcome,
        "failed_task_ids": failed_tasks,
        "repair_scope": repair_scope,
        "feedback": feedback,
        "summary": {
            "pass": "研究产物通过统一验收",
            "repair": "仅重跑失败的证据与引用范围",
            "replan": "证据不足，触发唯一一次有界重规划",
            "fail": "重规划预算已用尽",
        }[outcome],
    }
    delta.update({
        "review_verdict": verdict,
        "replan_count": replan_count,
        "trace": list(delta.get("trace", [])) + [_tr_event(
            "reviewer_verdict", node="reviewer", graph_node=True,
            outcome=outcome, failed_task_ids=failed_tasks,
            repair_scope=repair_scope, replan_count=replan_count,
        )],
    })
    return delta


def route_after_reviewer(state: ResearchAgentState) -> str:
    """Reviewer 可结束、局部修复或最多一次回到 Planner。"""
    outcome = (state.get("review_verdict") or {}).get("outcome", "pass")
    if outcome == "repair":
        return "analysis_worker"
    if outcome == "replan":
        return "planner"
    return "final_reviewer"


async def node_four_agent_reviewer(state: ResearchAgentState) -> Dict[str, Any]:
    """生产 Reviewer：只验收 Worker 结果，不生成或改写正文。"""
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    plan.repair_count = int(state.get("repair_count", plan.repair_count) or 0)
    plan.replan_count = int(state.get("replan_count", plan.replan_count) or 0)
    stored = dict(state.get("worker_results") or {})
    results = [
        WorkerResult.model_validate(stored[item.task_id])
        for item in plan.items if item.task_id in stored
    ]
    final_output = {
        "answer": state.get("answer", ""),
        "final_report": state.get("final_report", ""),
        "expected_chapter_count": state.get("expected_chapter_count"),
        "written_chapter_count": state.get("written_chapter_count"),
    }
    evaluated = bool(state.get("run_eval", True)) and (
        plan.execution_spec.execution_class.value == "research" and bool(results)
    )
    evaluator_input = {
        "topic": state.get("topic", ""),
        "draft_report": state.get("draft_report", ""),
        "sources": state.get("sources", []),
        "evidence_cards": state.get("evidence_cards", []),
        "citation_check_results": state.get("citation_check_results", []),
        "citation_summary": state.get("citation_summary", {}),
        "trace": state.get("trace", []),
        "task_dag": state.get("task_dag", {}),
        "total_latency_ms": max(
            0, int(time.time() * 1000) - int(state.get("start_time_ms") or int(time.time() * 1000)),
        ),
    }
    verdict, semantic_info = await _reviewer_agent.review_hybrid(
        plan, results,
        evaluator_input=evaluator_input,
        final_output=final_output,
        run_eval=bool(state.get("run_eval", True)),
        agent_mode=str(state.get("agent_mode") or "rule"),
    )
    output = dict(verdict.final_output or {})
    metrics = dict(output.get("eval_metrics") or {})
    soft_warnings = []
    if metrics.get("latency_under_threshold") is False:
        soft_warnings.append("运行耗时超过评估阈值，但交付内容已通过验收。")
    trace_events = [
        _tr_event(
            "agent_protocol_validated", node="reviewer", graph_node=False,
            protocol_version="2.0", agent_role="reviewer",
            architectural_role="reviewer", task_id="reviewer",
            allowed_tools=[], tools_called=[], status="success",
        ),
    ]
    evaluator_ran = evaluated and bool(semantic_info.get("evaluator_ran"))
    if evaluator_ran:
        trace_events.append(_tr_event(
            "evaluator_complete", node="reviewer_agent", graph_node=True,
            passed=sum(bool(value) for value in dict(output.get("eval_metrics") or {}).values()),
            total=len(dict(output.get("eval_metrics") or {})),
            retry_count=plan.repair_count,
        ))
    if semantic_info.get("attempted"):
        llm_success = bool(semantic_info.get("success"))
        trace_events.append(_tr_event(
            "llm_finished" if llm_success else "llm_failed",
            node="reviewer_agent", graph_node=True,
            purpose="semantic_review", status=semantic_info.get("status", ""),
            success=llm_success,
            error_code=("" if llm_success else _safe_llm_error_code(
                "hybrid_reviewer", semantic_info.get("status", "failed"),
            )),
            recoverable=not llm_success,
            model=semantic_info.get("model", ""),
            latency_ms=int(semantic_info.get("latency_ms") or 0),
            usage=dict(semantic_info.get("usage") or {}),
        ))
    trace_events.append(_tr_event(
        "reviewer_verdict", node="reviewer_agent", graph_node=True,
        internal_only=True, outcome=verdict.outcome.value,
        failed_task_ids=verdict.failed_task_ids,
        repair_scope=verdict.repair_scope,
        repair_count=plan.repair_count, replan_count=plan.replan_count,
    ))
    return {
        "review_verdict": verdict.model_dump(mode="json"),
        "eval_metrics": metrics,
        "eval_metric_details": dict(output.get("eval_metric_details") or {}),
        "eval_feedback": list(output.get("eval_feedback") or []),
        "warnings": soft_warnings,
        "trace": trace_events,
    }


def route_after_four_agent_reviewer(state: ResearchAgentState) -> str:
    outcome = str((state.get("review_verdict") or {}).get("outcome") or "fail")
    if outcome == "repair":
        return "begin_repair"
    if outcome == "replan":
        return "planner"
    return END


async def node_begin_repair(state: ResearchAgentState) -> Dict[str, Any]:
    """开始唯一一次局部修复，并切换到全新的 reducer round。"""
    repair_count = int(state.get("repair_count", 0) or 0) + 1
    if repair_count > 1:
        raise RuntimeError("局部修复预算已用尽")
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    plan.repair_count = repair_count
    verdict = ReviewVerdict.model_validate(state.get("review_verdict") or {})
    requested_scope = set(verdict.repair_scope)
    supported = {"analyze", "cite", "write"}
    if not requested_scope or not requested_scope <= supported:
        raise RuntimeError("Reviewer repair_scope 超出运行时支持范围")
    downstream = {
        "analyze": {"analyze", "cite", "write"},
        "cite": {"cite", "write"},
        "write": {"write"},
    }
    affected = set().union(*(downstream[item] for item in requested_scope))
    feedback = [dict(item) for item in verdict.feedback]
    messages = [
        str(item.get("message") or item.get("reason") or "").strip()
        for item in feedback
        if str(item.get("message") or item.get("reason") or "").strip()
    ]
    repair_instruction = "；".join(messages)[:500] or "按 Reviewer 结构化反馈修复产物"
    # 关键步骤：把 Reviewer 反馈写入新的执行信封，使实际 Worker 动作发生可审计变化。
    for item in plan.items:
        if item.task_id not in affected:
            continue
        item.instruction = f"{item.instruction}\n修复要求：{repair_instruction}"[:1000]
        item.input_data.update({
            "reviewer_feedback": feedback,
            "failed_task_ids": list(verdict.failed_task_ids),
            "repair_scope": list(verdict.repair_scope),
            "repair_count": repair_count,
        })
        item.metadata.update({
            "repair_count": repair_count,
            "repair_scope": list(verdict.repair_scope),
            "repair_feedback_applied": True,
        })
    plan.reviewer_feedback = messages
    return {
        "repair_count": repair_count,
        "retry_count": repair_count,
        "execution_round": int(state.get("execution_round", 0) or 0) + 1,
        "work_plan": plan.model_dump(mode="json"),
        "trace": [_tr_event(
            "repair_started", node="reviewer_agent", graph_node=True,
            internal_only=True, repair_count=repair_count,
        )],
    }


def route_repair_scope(state: ResearchAgentState):
    """修复必须产生新的 Analyze/Cite/Write WorkItem，不能由 Reviewer 改正文。"""
    scope = set((state.get("review_verdict") or {}).get("repair_scope") or [])
    if scope and scope <= {"write"}:
        # 关键步骤：纯语义写作缺口直接生成新的 Writing WorkItem，不重复消耗证据工具。
        return send_chapter_work_items(state)
    if "analyze" in scope:
        return send_analysis_work_item(state)
    if "cite" in scope:
        return send_citation_work_item(state)
    raise RuntimeError("repair_scope 没有可调度的受支持阶段")

# 只要引用失败，条件边就返回 analysis_worker，仅重试一次
def route_after_evaluator(state: ResearchAgentState) -> str:
    metrics = state.get("eval_metrics", {})
    retry_count = state.get("retry_count", 0)
    research_validation_failed = (
        not metrics.get("no_fake_citation", True)
        or not metrics.get("citation_id_exists", True)
        or not metrics.get("source_url_valid", True)
        or not metrics.get("evidence_available", True)
    )
    if "should_retry" in state:
        return "analysis_worker" if state.get("should_retry", False) else "final_reviewer"
    if research_validation_failed and retry_count < 1:
        return "analysis_worker"
    return "final_reviewer"


# ---- final_reviewer 节点 ----

async def node_final_reviewer(state: ResearchAgentState) -> Dict[str, Any]:
    protocol_task = _open_protocol_task(
        state,
        role=AgentRole.REVIEWER,
        task_id="final_reviewer",
        input_data={
            "topic": state.get("topic", ""),
            "stage": "final",
            "sources": state.get("sources", []),
            "evidence_cards": state.get("evidence_cards", []),
            "citation_check_results": state.get("citation_check_results", []),
            "outline": state.get("outline", {}),
            "draft_report": state.get("draft_report", ""),
        },
        allowed_tools=[],
        depends_on=["draft_reviewer"],
    )
    result = _final_reviewer.review(
        draft_report=state.get("draft_report", ""),
        eval_metrics={
            "metrics": state.get("eval_metrics", {}),
            "metrics_detail": state.get("eval_metric_details", {}),
        },
        eval_feedback=state.get("eval_feedback", []),
        citation_check_results=state.get("citation_check_results", []),
        evidence_cards=state.get("evidence_cards", []),
        sources=state.get("sources", []),
        warnings=state.get("warnings", []),
        language=state.get("language", "zh"),
        topic=state.get("topic", ""),
        outline=state.get("outline", {}),
    )
    protocol_event = _close_protocol_task(
        protocol_task,
        output_data={
            "report": result["final_report"],
            "completion_ready": bool(result.get("completion_ready", False)),
            "issues": result.get("completion_issues", []),
        },
        warnings=result.get("warnings", []),
    )
    return {
        "final_report": result["final_report"],
        "fixes_applied": result.get("fixes_applied", []),
        "unresolved_issues": result.get("unresolved_issues", []),
        "report_completion_ready": bool(result.get("completion_ready", False)),
        "report_completion_issues": result.get("completion_issues", []),
        "strict_completion_required": bool(result.get("strict_completion_required", False)),
        "trace": [protocol_event, _tr_event("final_reviewer_complete",
                            node="final_reviewer", graph_node=True,
                            fixes=len(result.get("fixes_applied", [])),
                            unresolved=len(result.get("unresolved_issues", [])))],
        "warnings": result.get("warnings", []),
    }


# ================================================================
# Send API 动态分发
# ================================================================

def send_to_search_worker(state: ResearchAgentState, *, unified: bool = False):
    """
    Planner 后：为每个 search_task 动态分发 search_worker_send。
    Send payload 显式包含子 Worker 所需的全部字段。
    """
    search_tasks = state.get("search_tasks", [])
    if not search_tasks:
        return []

    sends = []
    for t in search_tasks:
        item = WorkItem(
            task_id=t.get("task_id", "search_?"),
            profile=WorkerProfile.SEARCH,
            instruction=t.get("description") or state.get("topic", ""),
            depends_on=list(t.get("depends_on") or []),
            allowed_tools=list(t.get("tool_plan") or []),
            resources=[],
            input_data={
                "topic": state.get("topic", ""),
                "query": t.get("description") or state.get("topic", ""),
                "max_sources": state.get("max_sources", 5),
                "agent_mode": state.get("agent_mode", "rule"),
                "backend": state.get("backend", "graph_send"),
            },
            strategy=WorkerStrategy.REACT,
        )
        # 关键步骤：Send payload 只有完整 WorkItem，不隐式继承父 State。
        sends.append(Send("worker_send" if unified else "search_worker_send", {
            "work_item": item.model_dump(mode="json"),
            "explicit_resources": [],
            # 迁移兼容：旧测试/调用方仍可读取这些字段，统一 Worker 不消费它们。
            "current_search_task": t,
            "topic": state.get("topic", ""),
            "max_sources": state.get("max_sources", 5),
            "agent_mode": state.get("agent_mode", "rule"),
            "backend": state.get("backend", "graph_send"),
        }))
    return sends


def send_to_reading_worker(state: ResearchAgentState, *, unified: bool = False):
    """
    Search merge 后：为每个 unique source 动态分发 reading_worker_send。
    每个 source 一个 reading task（并行执行 metadata + quality scoring）。

    关键：将 source 对象直接放入 Send payload。
    LangGraph Send 子节点只接收 payload 中显式传递的字段；
    没有 reducer 的 state 字段不会自动继承。
    """
    sources = state.get("sources", [])
    if not sources:
        return []

    sends = []
    for s in sources:
        sid = s.get("source_id", "unknown")
        short_id = sid[:8] if len(sid) > 8 else sid
        item = WorkItem(
            task_id=f"read_{short_id}",
            profile=WorkerProfile.READ,
            instruction=f"Process source: {s.get('title', sid)[:80]}",
            depends_on=["search"],
            allowed_tools=["paper_metadata", "source_quality_scorer"],
            resources=[dict(s)],
            input_data={
                "topic": state.get("topic", ""),
                "max_sources": state.get("max_sources", 5),
                "agent_mode": state.get("agent_mode", "rule"),
                "backend": state.get("backend", "graph_send"),
            },
            strategy=WorkerStrategy.DETERMINISTIC,
        )
        sends.append(Send("worker_send" if unified else "reading_worker_send", {
            "work_item": item.model_dump(mode="json"),
            "explicit_resources": [dict(s)],
            "current_reading_task": {
                "task_id": f"read_{short_id}", "source_id": sid,
                "description": item.instruction,
            },
            "target_source": dict(s),
            "topic": state.get("topic", ""),
            "max_sources": state.get("max_sources", 5),
            "agent_mode": state.get("agent_mode", "rule"),
            "backend": state.get("backend", "graph_send"),
        }))
    return sends


def send_search_work_items(state: ResearchAgentState):
    """新图使用的统一搜索 WorkItem 扇出。"""
    return send_to_search_worker(state, unified=True)


def send_reading_work_items(state: ResearchAgentState):
    """新图使用的统一阅读 WorkItem 扇出。

    关键步骤：从 merge_search_results 汇总出的 sources 里，为每篇论文生成一个
    独立的 READ Send——N 篇论文 = N 个并行阅读 Worker，互不共享状态。
    """
    # 旧图兜底：没有 WorkPlan 时退回旧版扇出（直接按 sources 生成）。
    if not state.get("work_plan"):
        return send_to_reading_worker(state, unified=True)
    plan = WorkPlan.model_validate(state["work_plan"])
    # 以 Planner 产出的 READ WorkItem 为模板，按论文逐份深拷贝。
    template = next(item for item in plan.items if item.profile == WorkerProfile.READ)
    round_id = int(state.get("execution_round", 0) or 0)
    sends = []
    for source in list(state.get("sources") or []):
        sid = str(source.get("source_id") or "unknown")
        item = template.model_copy(deep=True)
        # 每篇论文一个唯一 task_id（read_ + source_id 前 12 位），用于追踪与去重。
        item.task_id = f"read_{sid[:12]}"
        # 关键步骤：resources 绑定单一 source——阅读 Worker 只处理这一篇论文。
        item.resources = [dict(source)]
        item.input_data.update({"topic": state.get("topic", "")})
        item.metadata.update({"round_id": round_id, "revision": plan.revision})
        # Send 到统一 worker_send 节点；explicit_resources 携带该论文全文引用。
        sends.append(Send("worker_send", {
            "work_item": item.model_dump(mode="json"),
            "explicit_resources": [dict(source)],
        }))
    return sends


async def node_worker_send(state: ResearchAgentState) -> Dict[str, Any]:
    """生产统一 Worker Pool；所有 Profile 均调用独立 WorkerAgent 实例。"""
    item = WorkItem.model_validate(state.get("work_item") or {})
    # 关键步骤：每个 Send 都新建 WorkerAgent，messages/预算/去重状态绝不共享。
    result = await WorkerAgent().execute(item)
    round_id = int(item.metadata.get("round_id", 0) or 0)
    revision = int(item.metadata.get("revision", 0) or 0)
    result.metadata.update(item.metadata)
    trace = [
        _tr_event(
            "send_dispatch", node="worker_send", worker_type=item.profile.value,
            task_id=item.task_id, round_id=round_id, revision=revision,
        ),
        _tr_event(
            "worker_started", node="worker_send", worker_type=item.profile.value,
            task_id=item.task_id, round_id=round_id, revision=revision,
        )
    ]
    legacy_role = {
        WorkerProfile.SEARCH: "search",
        WorkerProfile.READ: "reading",
        WorkerProfile.METADATA: "reading",
        WorkerProfile.ANALYZE: "reading",
        WorkerProfile.CITE: "citation",
    }.get(item.profile)
    trace.append(_tr_event(
        "agent_protocol_validated", node=item.task_id, graph_node=False,
        protocol_version="2.0", agent_role="worker",
        architectural_role="worker", worker_profile=item.profile.value,
        task_id=item.task_id, allowed_tools=item.allowed_tools,
        tools_called=[call.tool_name for call in result.tool_calls],
        status=result.status.value,
    ))
    if legacy_role:
        trace.append(_tr_event(
            "agent_protocol_validated", node=item.task_id, graph_node=False,
            protocol_version="2.0", agent_role=legacy_role,
            architectural_role="worker", worker_profile=item.profile.value,
            task_id=item.task_id, allowed_tools=item.allowed_tools,
            tools_called=[call.tool_name for call in result.tool_calls],
            status=result.status.value,
        ))
    trace.extend(_tr_event(
        "tool_finished", node="worker_send", graph_node=False,
        task_id=item.task_id, tool_name=call.tool_name,
        success=call.success, latency_ms=call.latency_ms, error=call.error,
    ) for call in result.tool_calls)
    for entry in list(result.output_data.get("_internal_trace") or []):
        event_name = str(entry.get("tool_name") or "worker_internal")
        trace.append(_tr_event(
            event_name, node="worker_send", graph_node=False,
            task_id=item.task_id,
            reason=entry.get("input_summary", ""),
            success=entry.get("success", True),
        ))
    trace.append(_tr_event(
        "worker_finished", node="worker_send", worker_type=item.profile.value,
        task_id=item.task_id,
        success=result.status == AgentTaskStatus.SUCCESS,
        needs_replan=result.needs_replan, round_id=round_id, revision=revision,
        latency_ms=result.latency_ms,
    ))
    fallback_warnings = []
    internal_names = {
        str(entry.get("tool_name") or "")
        for entry in list(result.output_data.get("_internal_trace") or [])
    }
    if "tool_loop_fallback" in internal_names:
        fallback_warnings.append(
            f"Worker {item.task_id} 的 LLM 决策失败，已按显式工具权限降级执行。"
        )
    return {
        "_worker_result_bucket": [{
            "round_id": round_id,
            "revision": revision,
            "profile": item.profile.value,
            "task_id": item.task_id,
            "result": result.model_dump(mode="json"),
        }],
        "_worker_route_bucket": [f"{round_id}:{item.profile.value}"],
        "trace": trace,
        "warnings": list(result.warnings) + fallback_warnings + (
            [result.error.message] if result.error is not None else []
        ),
    }


def route_after_worker_send(state: ResearchAgentState) -> str:
    """按 WorkItem Profile 把 reducer bucket 交给对应 Merge 基础设施。"""
    routes = list(state.get("_worker_route_bucket") or [])
    current_round = int(state.get("execution_round", 0) or 0)
    current = [route for route in routes if str(route).startswith(f"{current_round}:")]
    profile = WorkerProfile(current[-1].split(":", 1)[1]) if current else None
    if profile == WorkerProfile.DIRECT:
        return "merge_direct_results"
    if profile == WorkerProfile.CONTEXT_LOAD:
        return "merge_context_results"
    if profile == WorkerProfile.ANSWER:
        return "merge_answer_results"
    if profile == WorkerProfile.SEARCH:
        return "merge_search_results"
    if profile in {WorkerProfile.READ, WorkerProfile.METADATA}:
        return "merge_reading_results"
    if profile == WorkerProfile.WRITE:
        return "merge_chapter_results"
    if profile == WorkerProfile.ANALYZE:
        return "merge_analysis_results"
    if profile == WorkerProfile.CITE:
        return "merge_citation_results"
    raise ValueError(f"无法路由 WorkerProfile: {profile}")


# ---- Send API 搜索 worker ----

async def node_search_worker_send(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Send API 版本的 search worker：执行单个子搜索任务。
    写入 _search_bucket（operator.add 追加），不直接写 sources。
    """
    current = state.get("current_search_task", {})
    task_id = current.get("task_id", "search_?")
    query = current.get("description", state.get("topic", "unknown"))

    dispatch_event = _tr_event("send_dispatch",
                               worker_type="search",
                               task_id=task_id,
                               query=query[:80])

    try:
        protocol_task = _open_protocol_task(
            state,
            role=AgentRole.SEARCH,
            task_id=task_id,
            input_data={
                "query": query,
                "max_sources": state.get("max_sources", 5),
                "seed_sources": [],
            },
            allowed_tools=current.get("tool_plan") or None,
            depends_on=current.get("depends_on", []),
        )
        task = Task(
            task_id,
            "search",
            f"Search for academic sources on: {query}",
            tool_plan=protocol_task.allowed_tools,
        )
        async with _agent_permission_scope(protocol_task):
            ctx = await _get_search_worker(state).execute_task(task)
        sources = ctx.results.get("sources", ctx.results.get("search_results", []))
        worker_failed = _worker_context_failed(ctx)
        if worker_failed:
            emit_error({
                "stage": "send_worker",
                "worker_type": "search",
                "task_id": task_id,
                "exception_type": "WorkerToolFailure",
                "error": "; ".join(ctx.warnings)[:200],
            })

        protocol_event = _close_protocol_task(
            protocol_task,
            output_data={
                "sources": sources,
                "discovered_source_count": len(sources),
            },
            trace=ctx.trace,
            warnings=ctx.warnings,
        )
        return {
            "_search_bucket": sources,  # 将搜索结果写入 _search_bucket，实现并行合并
            "trace": [
                dispatch_event,
                protocol_event,
                _tr_event("worker_started",
                          worker_type="search", task_id=task_id,
                          success=not worker_failed),
                _tr_event("worker_finished",
                          worker_type="search", task_id=task_id,
                          success=not worker_failed, source_count=len(sources),
                          latency_ms=sum(t.get("latency_ms", 0) for t in ctx.trace)),
            ] + _merge_worker_trace(ctx),
            "warnings": ctx.warnings,
        }
    except Exception as e:
        emit_error({
            "stage": "send_worker",
            "worker_type": "search",
            "task_id": task_id,
            "exception_type": type(e).__name__,
            "error": str(e)[:200],
        })
        return {
            "_search_bucket": [],
            "trace": [
                dispatch_event,
                _tr_event("worker_started",
                          worker_type="search", task_id=task_id,
                          success=False),
                _tr_event("worker_finished",
                          worker_type="search", task_id=task_id,
                          success=False, error=str(e)[:200]),
            ],
            "warnings": [f"[search_worker_send:{task_id}] Failed: {str(e)[:200]}"],
        }


# ---- Send API 阅读 worker ----

async def node_reading_worker_send(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Send API 版本的 reading worker：为单个 source 执行元数据标准化 + 质量评分。
    写入 _reading_bucket（operator.add 追加），不直接写 scored_sources。

    从 Send payload 的 target_source 获取 source 对象（不依赖 state.sources）。
    """
    current = state.get("current_reading_task", {})
    task_id = current.get("task_id", "read_?")
    target_source_id = current.get("source_id", "")
    topic = state.get("topic", "")

    # 从 payload 直接获取 source 对象（Send 子节点只接收 payload 字段）
    target_source = state.get("target_source", {})

    dispatch_event = _tr_event("send_dispatch",
                               worker_type="reading",
                               task_id=task_id,
                               source_id=target_source_id)

    if not target_source:
        emit_error({
            "stage": "send_worker",
            "worker_type": "reading",
            "task_id": task_id,
            "exception_type": "MissingPayload",
            "error": "target_source not in payload",
        })
        return {
            "_reading_bucket": [],
            "trace": [
                dispatch_event,
                _tr_event("worker_finished",
                          worker_type="reading", task_id=task_id,
                          success=False, error="target_source not in payload"),
            ],
            "warnings": [f"[reading_worker_send:{task_id}] target_source missing from Send payload"],
        }

    try:
        protocol_task = _open_protocol_task(
            state,
            role=AgentRole.READING,
            task_id=task_id,
            input_data={
                "topic": topic,
                "sources": [target_source],
                "operation": "metadata_and_quality",
            },
            allowed_tools=["paper_metadata", "source_quality_scorer"],
            depends_on=["search"],
        )
        # 构建 deps，search task 包含真实 topic（保证 quality scorer 能正确评分）
        search_ctx = WorkerContext(
            Task("search", "search", f"Search for academic sources on: {topic}")
        )
        search_ctx.add_result("sources", [target_source])
        deps = {"search": search_ctx}

        task = Task(task_id, "read",
                    f"Process source: {target_source.get('title', target_source_id)[:80]}",
                    depends_on=["search"], tool_plan=protocol_task.allowed_tools)
        # Read 仅做元数据标准化与确定性质量评分。把每个来源都过一遍 LLM 会让每篇论文
        # 多出两次以上模型往返，却不增加研究判断力；在 max_sources=50 时曾产生
        # 100+ 次可避免调用并导致 provider 压力失败。保留 Send 扇出，
        # 但直接执行这两个有界的工具。
        async with _agent_permission_scope(protocol_task):
            ctx = await Worker().execute_task(task, dependency_results=deps)
        scored_list = ctx.results.get("scored_sources", [target_source])
        worker_failed = _worker_context_failed(ctx)
        if worker_failed:
            emit_error({
                "stage": "send_worker",
                "worker_type": "reading",
                "task_id": task_id,
                "exception_type": "WorkerToolFailure",
                "error": "; ".join(ctx.warnings)[:200],
            })

        protocol_event = _close_protocol_task(
            protocol_task,
            output_data={"sources": scored_list, "evidence_cards": []},
            trace=ctx.trace,
            warnings=ctx.warnings,
        )
        return {
            "_reading_bucket": scored_list,  # 将阅读结果写入 _reading_bucket，实现并行合并
            "trace": [
                dispatch_event,
                protocol_event,
                _tr_event("worker_started",
                          worker_type="reading", task_id=task_id,
                          source_id=target_source_id, success=not worker_failed),
                _tr_event("worker_finished",
                          worker_type="reading", task_id=task_id,
                          source_id=target_source_id, success=not worker_failed,
                          quality_score=scored_list[0].get("quality_score", 0) if scored_list else 0,
                          latency_ms=sum(t.get("latency_ms", 0) for t in ctx.trace)),
            ] + _merge_worker_trace(ctx),
            "warnings": ctx.warnings,
        }
    except Exception as e:
        emit_error({
            "stage": "send_worker",
            "worker_type": "reading",
            "task_id": task_id,
            "exception_type": type(e).__name__,
            "error": str(e)[:200],
        })
        return {
            "_reading_bucket": [target_source],
            "trace": [
                dispatch_event,
                _tr_event("worker_started",
                          worker_type="reading", task_id=task_id,
                          source_id=target_source_id, success=False),
                _tr_event("worker_finished",
                          worker_type="reading", task_id=task_id,
                          source_id=target_source_id, success=False,
                          error=str(e)[:200]),
            ],
            "warnings": [f"[reading_worker_send:{task_id}] Failed: {str(e)[:200]}"],
        }


# ---- Merge 节点 ----

async def node_merge_search_results(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Search workers 完成后的合并节点。
    从 _search_bucket 读取所有并行结果，显式去重 + 截断，写入 sources。
    """
    current_results = _current_worker_results(state, WorkerProfile.SEARCH)
    if current_results:
        revision = int((state.get("work_plan") or {}).get("revision", 0) or 0)
        # 关键步骤：replan 只执行受影响检索根，成功缓存作为显式 Merge 输入复用。
        # 首次执行从 seed_papers 起步，重规划则复用上一轮已成功保留的 sources。
        raw = list(state.get("sources") or []) if revision > 0 else list(state.get("seed_papers") or [])
        for result in current_results:
            # 关键步骤：合并 3 路并行检索的产物到统一列表。
            raw.extend(
                result.output_data.get("sources")
                or result.output_data.get("search_results")
                or []
            )
    else:
        # 旧入口兼容：专项 search_worker_send 仍使用历史 bucket 格式。
        raw = state.get("_search_bucket", [])
    # 关键步骤：显式去重 → 过滤与任务无关来源 → 恢复会话种子 → 按主题相关性排序。
    unique = _dedup_sources(raw)
    eligible, task_boundary = filter_sources_for_task(state.get("topic", ""), unique)
    eligible = _restore_session_seeds(state, eligible, unique)
    ranked = _rank_sources_for_topic(eligible, state.get("topic", ""))
    # 本地 Zotero/RAG 来源若无全文则剔除，避免无证据支撑的引用进入下游。
    ranked = [
        source for source in ranked
        if not (
            str(source.get("provider") or "").lower() in {"local_zotero", "local_rag"}
            or str(source.get("content_source") or "").lower() == "zotero_pdf"
        ) or _is_local_full_text_source(source)
    ]
    ranked = _restore_session_seeds(state, ranked, unique)
    # 关键步骤：截断到 max_sources 上限，selection 记录选源模式供追踪。
    capped, selection = await _select_sources_with_session_seeds(state, ranked)
    selection["task_boundary"] = task_boundary
    provider_counts: Dict[str, int] = {}
    for source in capped:
        provider = str(source.get("provider") or "unknown")
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    worker_results = dict(state.get("worker_results") or {})
    for result in current_results:
        worker_results[result.task_id] = result.model_dump(mode="json")
    partial_warnings = [
        f"检索分支 {result.task_id} 未完成，已使用其他可用来源继续。"
        for result in current_results
        if result.status != AgentTaskStatus.SUCCESS
    ]
    return {
        "sources": capped,
        "discovered_source_count": len(unique),
        "analyzed_source_count": len(capped),
        "analysis_selection": selection,
        "worker_results": worker_results,
        "_search_bucket": [],  # 清空 accumulator
        "trace": [_tr_event("merge_result",
                            node="merge_search_results", graph_node=True,
                            merge_type="search",
                            raw_count=len(raw),
                            unique_count=len(unique),
                            task_eligible_count=len(eligible),
                            task_rejected_count=task_boundary.get("rejected_count", 0),
                            capped_count=len(capped),
                            provider_counts=provider_counts,
                            local_source_count=sum(
                                _is_local_full_text_source(source)
                                for source in capped
                            ),
                            selection_mode=selection.get("mode", "rule"),
                            requested_count=selection.get("requested_count", state.get("max_sources", 5)),
                            analysis_count=len(capped))],
        "warnings": partial_warnings,
    }


async def node_merge_reading_results(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Reading workers 完成后的合并节点。
    从 _reading_bucket 读取所有并行结果，显式去重，写入 scored_sources。
    同时更新 sources（用带 quality_score 的版本覆盖）。
    """
    current_results = _current_worker_results(state, WorkerProfile.READ)
    if current_results:
        raw = []
        for result in current_results:
            # 关键步骤：汇集 N 个阅读 Worker 各自的 scored_sources（已带 quality_score）。
            raw.extend(result.output_data.get("scored_sources") or result.output_data.get("sources") or [])
    else:
        raw = state.get("_reading_bucket", [])
    # 关键步骤：去重后得到最终证据源集合，作为 ANALYZE 阶段的输入。
    unique = _dedup_sources(raw)

    # 把 N 个动态扇出的阅读结果合成 Planner DAG 里唯一的 read 阶段结果。
    stage = _aggregate_stage_result("read", WorkerProfile.READ, current_results, {
        "scored_sources": unique,
    }) if current_results else None
    worker_results = dict(state.get("worker_results") or {})
    if stage is not None:
        worker_results["read"] = stage.model_dump(mode="json")

    return {
        "scored_sources": unique,
        "sources": unique,  # 用带 quality_score 的版本覆盖
        "analyzed_source_count": len(unique),
        "worker_results": worker_results,
        "_reading_bucket": [],  # 清空 accumulator
        "trace": [_tr_event("merge_result",
                            node="merge_reading_results", graph_node=True,
                            merge_type="reading",
                            raw_count=len(raw),
                            scored_count=len(unique))],
    }


def _current_worker_results(
    state: ResearchAgentState,
    profile: WorkerProfile,
) -> List[WorkerResult]:
    """只读取当前 round/revision 的 reducer 结果，旧轮次永不参与 Merge。"""
    round_id = int(state.get("execution_round", 0) or 0)
    revision = int((state.get("work_plan") or {}).get("revision", 0) or 0)
    return [
        WorkerResult.model_validate(entry["result"])
        for entry in state.get("_worker_result_bucket", [])
        if int(entry.get("round_id", -1)) == round_id
        and int(entry.get("revision", -1)) == revision
        and entry.get("profile") == profile.value
    ]


def _aggregate_stage_result(
    task_id: str,
    profile: WorkerProfile,
    results: List[WorkerResult],
    output: Dict[str, Any],
) -> WorkerResult:
    """把同 Profile 的动态扇出结果合成为 Planner DAG 中的单一阶段结果。"""
    failed = [result for result in results if result.status != AgentTaskStatus.SUCCESS]
    needs_replan = any(result.needs_replan for result in results)
    # Fail-closed：没有任何 Worker 结果 → 本阶段判失败并请求重规划。
    if not results:
        return WorkerResult(
            task_id=task_id, profile=profile, status=AgentTaskStatus.FAILED,
            needs_replan=True,
            error=agent_protocol.error("TOOL_UNAVAILABLE", message="当前轮没有 Worker 结果", recoverable=True),
        )
    # 有失败 → 阶段整体按首个失败的状态上报，但输出数据仍保留，供 Reviewer 定位问题。
    if failed:
        first = failed[0]
        return WorkerResult(
            task_id=task_id, profile=profile, status=first.status,
            output_data=output, needs_replan=True,
            warnings=[warning for result in results for warning in result.warnings],
            error=first.error or agent_protocol.error(
                "TOOL_UNAVAILABLE", message="动态 Worker 阶段失败", recoverable=True,
            ),
        )
    # 全部成功 → 合并工具调用与警告，产出唯一阶段结果。
    return WorkerResult(
        task_id=task_id, profile=profile, status=AgentTaskStatus.SUCCESS,
        output_data=output, needs_replan=needs_replan,
        tool_calls=[call for result in results for call in result.tool_calls],
        warnings=[warning for result in results for warning in result.warnings],
    )


async def node_merge_direct_results(state: ResearchAgentState) -> Dict[str, Any]:
    """atomic 直连的收口：工具返回即最终答案，直接落成各阶段字段。"""
    results = _current_worker_results(state, WorkerProfile.DIRECT)
    stage = _aggregate_stage_result("direct_1", WorkerProfile.DIRECT, results, {})
    output = dict(results[-1].output_data) if results else {}
    # 关键步骤：单工具直连无检索/阅读管道，工具输出直接映射为来源与答案。
    source_payload = output.get("sources") if "sources" in output else output.get("results")
    sources = _dedup_sources(list(source_payload or []))
    answer = str(output.get("answer") or "")
    stage.output_data = output
    worker_results = dict(state.get("worker_results") or {})
    worker_results["direct_1"] = stage.model_dump(mode="json")
    # ---- 序号事件：补发 direct_reviewer_complete ----
    # 历史会话恢复（前端 openHistoryItem）依赖 run 记录里的
    # recommendation_number_start/end 还原真实显示序号。旧流程该事件由
    # node_direct_reviewer 发出，但该节点未注册进 graph（direct 走
    # merge_direct_results → reviewer），导致事件从不产生、序号恢复退化。
    # 这里在真实收口节点上补发，编号规则与 node_direct_reviewer 一致：
    # recommend_more 沿用会话累计序号，其它推荐从 1 开始。
    intent = state.get("intent", "literature_search")
    session = state.get("session_context")
    prior_papers = (
        list(session.recommended_papers)
        if intent == "recommend_more" and isinstance(session, SessionContext)
        else []
    )
    prior_numbers = {
        _dedup_key(source): index
        for index, source in enumerate(prior_papers, start=1)
    }
    next_number = len(prior_papers) + 1
    display_numbers: List[int] = []
    for source in sources:
        key = _dedup_key(source)
        number = prior_numbers.get(key)
        if number is None:
            number = next_number
            prior_numbers[key] = number
            next_number += 1
        display_numbers.append(number)
    return {
        "sources": sources,
        "scored_sources": sources,
        "answer": answer,
        "draft_report": answer,
        "final_report": answer,
        "worker_results": worker_results,
        "trace": [
            _tr_event("merge_result", node="merge_direct_results", merge_type="direct"),
            _tr_event(
                "direct_reviewer_complete",
                node="direct_reviewer",
                graph_node=True,
                intent=intent,
                source_count=len(sources),
                recommendation_number_start=(display_numbers[0] if display_numbers else None),
                recommendation_number_end=(display_numbers[-1] if display_numbers else None),
                report_len=len(answer),
            ),
        ],
    }


async def node_merge_answer_results(state: ResearchAgentState) -> Dict[str, Any]:
    """conversation 追问的收口：答案即最终输出。"""
    results = _current_worker_results(state, WorkerProfile.ANSWER)
    stage = _aggregate_stage_result("answer", WorkerProfile.ANSWER, results, {})
    output = dict(results[-1].output_data) if results else {}
    # 关键步骤：单发 ANSWER Send → 单一输出，答案直接同时落为 draft/final 报告。
    answer = str(output.get("answer") or "")
    stage.output_data = output
    worker_results = dict(state.get("worker_results") or {})
    worker_results["answer"] = stage.model_dump(mode="json")
    return {
        "answer": answer,
        "draft_report": answer,
        "final_report": answer,
        "conversation_result": dict(output.get("conversation_result") or {}),
        "worker_results": worker_results,
        "trace": [_tr_event("merge_result", node="merge_answer_results", merge_type="answer")],
    }


async def node_merge_context_results(state: ResearchAgentState) -> Dict[str, Any]:
    """提交 context_load 结果；资源只能来自该 Worker 的显式信封。"""
    results = _current_worker_results(state, WorkerProfile.CONTEXT_LOAD)
    # 关键步骤：把 CONTEXT_LOAD 产物写入 worker_results，供 send_answer_work_item 组装追问上下文。
    stage = _aggregate_stage_result(
        "context_load", WorkerProfile.CONTEXT_LOAD, results,
        dict(results[-1].output_data) if results else {},
    )
    worker_results = dict(state.get("worker_results") or {})
    worker_results["context_load"] = stage.model_dump(mode="json")
    return {
        "worker_results": worker_results,
        "trace": [_tr_event(
            "merge_result", node="merge_context_results", merge_type="context_load",
            resource_count=len(stage.output_data.get("resources") or []),
        )],
    }


def send_answer_work_item(state: ResearchAgentState):
    """把 context_load 的显式产物封装进 ANSWER WorkItem。"""
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    template = next(item for item in plan.items if item.profile == WorkerProfile.ANSWER)
    # 关键步骤：读取 context_load Worker 汇总进桶的产物，据此组装追问的会话上下文。
    context_result = dict((state.get("worker_results") or {}).get("context_load") or {})
    output = dict(context_result.get("output_data") or {})
    resources = list(output.get("resources") or [])
    # 三类资源分别取用：paper 用于引用原文，report 携带证据/章节，history 携带历史对话。
    papers = [resource for resource in resources if resource.get("resource_type") in {None, "paper"}]
    report_resource = next(
        (resource for resource in resources if resource.get("resource_type") == "report"), {},
    )
    history_resource = next(
        (resource for resource in resources if resource.get("resource_type") == "history"), {},
    )
    item = template.model_copy(deep=True)
    item.resources = resources
    item.input_data.update({
        "conversation_context": {
            "papers": papers,
            "report": report_resource or None,
            "evidence": list(report_resource.get("evidence_cards") or []),
            "history": list(history_resource.get("history") or []),
            "resolved_section": report_resource.get("resolved_section"),
            "report_id": report_resource.get("report_id"),
        },
        "operation_hint": state.get("conversation_operation", ""),
        "language": state.get("language", "zh"),
        "memory_prompt": state.get("memory_prompt", ""),
    })
    item.metadata.update({
        "round_id": int(state.get("execution_round", 0) or 0),
        "revision": plan.revision,
        "runtime_budget_id": str(state.get("runtime_budget_id") or ""),
    })
    # 单发一条 ANSWER Send：回答是整体生成，无需扇出。
    return [Send("worker_send", {
        "work_item": item.model_dump(mode="json"),
        "explicit_resources": resources,
    })]


def send_analysis_work_item(state: ResearchAgentState):
    """单发一条 ANALYZE Send：证据抽取跨论文汇总进行，不需要逐篇扇出。"""
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    template = next(item for item in plan.items if item.profile == WorkerProfile.ANALYZE)
    item = template.model_copy(deep=True)
    # 关键步骤：以 merge_reading_results 评分后的全部论文为输入，一次抽完全部证据卡。
    item.resources = list(state.get("scored_sources") or state.get("sources") or [])
    item.input_data.update({"topic": state.get("topic", "")})
    item.metadata.update({
        "round_id": int(state.get("execution_round", 0) or 0),
        "revision": plan.revision,
    })
    return [Send("worker_send", {
        "work_item": item.model_dump(mode="json"),
        "explicit_resources": list(item.resources),
    })]


async def node_merge_analysis_results(state: ResearchAgentState) -> Dict[str, Any]:
    """收口 ANALYZE：跨全部来源汇总证据卡，作为 CITE/WRITE 的依据。"""
    results = _current_worker_results(state, WorkerProfile.ANALYZE)
    # 关键步骤：把证据抽取 Worker 产出的全部 evidence_cards 拍平到统一列表。
    cards = [
        card for result in results
        for card in list(result.output_data.get("evidence_cards") or [])
    ]
    stage = _aggregate_stage_result("analyze", WorkerProfile.ANALYZE, results, {
        "evidence_cards": cards,
    })
    worker_results = dict(state.get("worker_results") or {})
    worker_results["analyze"] = stage.model_dump(mode="json")
    return {
        "evidence_cards": cards,
        "worker_results": worker_results,
        "trace": [_tr_event(
            "merge_result", node="merge_analysis_results",
            merge_type="analysis", evidence_count=len(cards),
        ), _tr_event(
            "analysis_complete", node="merge_analysis_results", graph_node=True,
            evidence_count=len(cards), success=stage.status == AgentTaskStatus.SUCCESS,
        )],
    }


def send_citation_work_item(state: ResearchAgentState):
    """单发一条 CITE Send：确定性校验全部引用绑定。"""
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    template = next(item for item in plan.items if item.profile == WorkerProfile.CITE)
    item = template.model_copy(deep=True)
    # 关键步骤：输入 = 论文来源 + ANALYZE 阶段产出的证据卡，逐条校验引用是否落在来源列表内。
    item.resources = list(state.get("sources") or [])
    item.input_data.update({
        "topic": state.get("topic", ""),
        "evidence_cards": list(state.get("evidence_cards") or []),
    })
    item.metadata.update({
        "round_id": int(state.get("execution_round", 0) or 0),
        "revision": plan.revision,
    })
    return [Send("worker_send", {
        "work_item": item.model_dump(mode="json"),
        "explicit_resources": list(item.resources),
    })]


def send_report_work_item(state: ResearchAgentState):
    """派发 Planner 明确授权的整篇 Writing WorkItem。"""
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    template = next(item for item in plan.items if item.profile == WorkerProfile.WRITE)
    item = template.model_copy(deep=True)
    # 关键步骤：整篇 WRITE——把检索、阅读、证据、引用校验的最终产物全部打包交给单一 Writer。
    item.resources = list(state.get("sources") or [])
    item.input_data.update({
        "topic": state.get("topic", ""),
        "evidence_cards": list(state.get("evidence_cards") or []),
        "citation_check_results": list(state.get("citation_check_results") or []),
        "citation_summary": dict(state.get("citation_summary") or {}),
        "language": state.get("language", "zh"),
    })
    item.metadata.update({
        "round_id": int(state.get("execution_round", 0) or 0),
        "revision": plan.revision,
        "whole_report": True,
    })
    # 单发一条 Send：整篇报告一次生成（与逐章节的 send_chapter_work_items 相对）。
    return [Send("worker_send", {
        "work_item": item.model_dump(mode="json"),
        "explicit_resources": list(item.resources),
    })]


async def node_merge_citation_results(state: ResearchAgentState) -> Dict[str, Any]:
    """收口 CITE：取出引用校验明细与汇总，供 WRITE 阶段保证引用合法性。"""
    results = _current_worker_results(state, WorkerProfile.CITE)
    output = dict(results[-1].output_data) if results else {}
    # 关键步骤：兼容新旧两种输出字段名，取出逐条校验结果与总体摘要。
    checks = list(output.get("citation_check_results") or output.get("check_results") or [])
    summary = dict(output.get("citation_summary") or output.get("summary") or {})
    stage = _aggregate_stage_result("cite", WorkerProfile.CITE, results, {
        "citation_check_results": checks, "citation_summary": summary,
    })
    worker_results = dict(state.get("worker_results") or {})
    worker_results["cite"] = stage.model_dump(mode="json")
    return {
        "citation_check_results": checks,
        "citation_summary": summary,
        "worker_results": worker_results,
        "trace": [
            _tr_event("merge_result", node="merge_citation_results", merge_type="citation"),
            _tr_event(
                "citation_complete", node="merge_citation_results", graph_node=True,
                checked=len(checks),
                total_checked=int(summary.get("total_checked", len(checks)) or 0),
                valid_count=int(summary.get("valid_count", 0) or 0),
                success=(
                    stage.status == AgentTaskStatus.SUCCESS
                    and bool(summary.get("all_valid", not checks))
                ),
            ),
        ],
    }


# ---- Send API chapter workers 将章节任务分发给独立的章节写手。----

def send_to_chapter_writer(state: ResearchAgentState, *, unified: bool = False):
    """为大纲中的每个章节扇出一个相互隔离的写手。

    关键步骤：写阶段按章节扇出，而非整篇一次写。
    - 每个章节只拿到分配给它的 sources 与证据卡（章节隔离）。
    - 依赖 outline 中 OutlineSection 的 assigned_source_ids / assigned_evidence_ids。
    """
    sections = list(state.get("outline", {}).get("sections", []))
    cards_by_id = {
        str(card.get("evidence_id") or ""): card
        for card in state.get("evidence_cards", []) if card.get("evidence_id")
    }
    sources = state.get("sources", [])
    source_number = {
        str(source.get("source_id")): index
        for index, source in enumerate(sources, start=1) if source.get("source_id")
    }
    sends = []
    for index, section in enumerate(sections):
        # 该章节被分配的引用集：只拿自己需要的证据卡与论文，跨章节互不可见。
        evidence_ids = set(section.get("assigned_evidence_ids") or [])
        source_ids = set(section.get("assigned_source_ids") or [])
        chapter_sources = [
            source for source in sources if source.get("source_id") in source_ids
        ]
        chapter_cards = [cards_by_id[item] for item in evidence_ids if item in cards_by_id]
        item = WorkItem(
            task_id=f"chapter_{index + 1}",
            profile=WorkerProfile.WRITE,
            instruction=f"撰写章节：{section.get('heading', index + 1)}",
            depends_on=["cite"],
            resources=chapter_sources,
            input_data={
                "current_chapter_task": {"index": index, "section": section},
                "topic": state.get("topic", ""),
                "language": state.get("language", "zh"),
                "agent_mode": state.get("agent_mode", "rule"),
                "llm_only": state.get("llm_only", False),
                "chapter_cards": chapter_cards,
                "source_number": source_number,
            },
            strategy=WorkerStrategy.SYNTHESIS,
        )
        # 旧图走独立 chapter_writer_send 节点，新图统一走 worker_send。
        sends.append(Send("worker_send" if unified else "chapter_writer_send", {
            "work_item": item.model_dump(mode="json"),
            "explicit_resources": chapter_sources,
            "current_chapter_task": {"index": index, "section": section},
            "topic": state.get("topic", ""),
            "language": state.get("language", "zh"),
            "agent_mode": state.get("agent_mode", "rule"),
            "llm_only": state.get("llm_only", False),
            "chapter_cards": chapter_cards,
            "chapter_sources": chapter_sources,
            "source_number": source_number,
        }))
    return sends


def send_chapter_work_items(state: ResearchAgentState):
    """新图使用的统一 Writing WorkItem 扇出。

    关键步骤：以 Planner 的 WRITE WorkItem 为模板，按 outline 每个章节
    深拷贝一份 Send——N 个章节 = N 个相互隔离的写手并行成稿。
    """
    # 旧图兜底：没有 WorkPlan 时退回旧版章节扇出。
    if not state.get("work_plan"):
        return send_to_chapter_writer(state, unified=True)
    plan = WorkPlan.model_validate(state["work_plan"])
    template = next(item for item in plan.items if item.profile == WorkerProfile.WRITE)
    sections = list(state.get("outline", {}).get("sections", []))
    cards_by_id = {
        str(card.get("evidence_id") or ""): card
        for card in state.get("evidence_cards", []) if card.get("evidence_id")
    }
    sources = list(state.get("sources") or [])
    source_number = {
        str(source.get("source_id")): index
        for index, source in enumerate(sources, start=1) if source.get("source_id")
    }
    sends = []
    for index, section in enumerate(sections):
        evidence_ids = set(section.get("assigned_evidence_ids") or [])
        source_ids = set(section.get("assigned_source_ids") or [])
        item = template.model_copy(deep=True)
        item.task_id = f"write_{index + 1}"
        item.instruction = (
            f"{template.instruction}\n撰写章节：{section.get('heading', index + 1)}"
        )[:1000]
        # 关键步骤：按章节隔离 resources/evidence——每个写手只见自己被分配的论文与证据。
        item.resources = [source for source in sources if source.get("source_id") in source_ids]
        item.input_data.update({
            "section": section,
            "evidence_cards": [cards_by_id[key] for key in evidence_ids if key in cards_by_id],
            "language": state.get("language", "zh"),
            "source_number": source_number,
        })
        item.metadata.update({
            "round_id": int(state.get("execution_round", 0) or 0),
            "revision": plan.revision,
            "chapter_index": index,
        })
        sends.append(Send("worker_send", {
            "work_item": item.model_dump(mode="json"),
            "explicit_resources": list(item.resources),
        }))
    return sends


async def node_chapter_writer_send(state: ResearchAgentState) -> Dict[str, Any]:
    """为单个章节生成正文：严格墙钟预算 + 证据隔离，LLM / 规则双路径。"""
    current = state.get("current_chapter_task", {})
    index = int(current.get("index", 0) or 0)
    section = current.get("section") or {}
    heading = str(section.get("heading") or f"Chapter {index + 1}")
    cards = state.get("chapter_cards", [])
    sources = state.get("chapter_sources", [])
    language = state.get("language", "zh")
    source_number = state.get("source_number", {})
    started = time.perf_counter()  # 计时起点，用于计算本章生成耗时
    warnings = []
    llm_only = _is_llm_only(state)  # LLM-only 模式下失败不允许降级为规则
    # 开启协议追踪：以 REVIEWER 角色记录本章的输入、输出与问题列表
    protocol_task = _open_protocol_task(
        state,
        role=AgentRole.REVIEWER,
        task_id=f"chapter_{index + 1}",
        input_data={
            "topic": state.get("topic", ""),
            "stage": "chapter",
            "sources": sources,
            "evidence_cards": cards,
            "citation_check_results": [],
            "outline": {"sections": [section]},
            "draft_report": "",
        },
        allowed_tools=[],
        depends_on=["cite"],
    )

    try:
        if state.get("agent_mode") == "llm":
            # —— LLM 路径：信号量限流 + 墙钟超时 + 带重试 ——
            from app.agents.llm_reviewer import LLMDraftReviewer

            reviewer = LLMDraftReviewer()
            # 从环境变量读取单次调用的墙钟超时与最大重试次数
            wall_timeout = max(20, int(os.getenv("LLM_CHAPTER_WALL_TIMEOUT_SECONDS", "130")))
            attempts = max(1, int(os.getenv("LLM_CHAPTER_MAX_ATTEMPTS", "2")))
            raw_result = {}
            last_error = "unknown error"
            for attempt in range(attempts):  # 重试循环：超时/异常则带失败原因重写
                try:
                    attempt_section = dict(section)
                    if attempt:  # 重试时注入重写指令，要求实际跨来源综合
                        attempt_section["_retry_instruction"] = (
                            "上一次输出未通过严格校验：" + last_error + "。"
                            "请重新撰写；若本章要求跨来源综合，正文必须实际使用至少两个不同 "
                            "source_id 的证据标记，而不是只在 evidence_ids 数组中列出。"
                        )
                    # 信号量并发限流，wait_for 施加墙钟超时；成功即跳出重试循环
                    async with _get_chapter_semaphore():
                        chapter, raw_result = await asyncio.wait_for(
                            # 调用 LLM 生成章节正文，或根据规则生成（非 LLM-only 模式）
                            reviewer.generate_chapter(
                                attempt_section, cards, sources, language,
                                source_number=source_number,
                                allow_rule_fallback=not llm_only,  # LLM-only 时禁止降级
                            ),
                            timeout=wall_timeout,
                        )
                    break
                except asyncio.TimeoutError:  # 超时也作为一次失败计入重试
                    last_error = (
                        f"Chapter timeout after {wall_timeout}s "
                        f"(attempt {attempt + 1}/{attempts})"
                    )
                except Exception as exc:
                    last_error = str(exc)[:500]
                if attempt + 1 < attempts:  # 非最后一次则短暂停顿后重试
                    await asyncio.sleep(0.25)
            else:  # 所有重试均失败：非 LLM-only 降级为规则生成，否则直接抛错
                if not llm_only:
                    chapter = _draft_reviewer.generate_chapter(
                        section, cards, sources, language, source_number=source_number,
                    )
                    raw_result = {"success": False, "error": last_error, "latency_ms": 0}
                else:
                    raise RuntimeError(
                        f"LLM-only chapter '{heading}' failed after {attempts} attempts: {last_error}"
                    )
            # 只透传非空的元数据字段（成功标志、耗时、模型、错误等）
            result = {
                key: raw_result.get(key) for key in (
                    "success", "skipped", "latency_ms", "model", "usage", "error",
                    "source_title_translations", "missing_title_translation_source_ids",
                ) if raw_result.get(key) is not None
            }
            if not raw_result.get("success") and not llm_only:  # 已发生降级，记录警告
                warnings.append(f"Chapter fallback ({heading}): {raw_result.get('error', 'unknown error')}")
        else:
            # —— 规则路径：无重试，直接规则生成 ——
            chapter = _draft_reviewer.generate_chapter(
                section, cards, sources, language, source_number=source_number,
            )
            result = {"success": True, "latency_ms": 0, "mode": "rule"}
    except Exception as exc:  # 兜底异常：LLM-only 上抛，否则规则降级
        if llm_only:
            raise
        chapter = _draft_reviewer.generate_chapter(
            section, cards, sources, language, source_number=source_number,
        )
        result = {"success": False, "error": str(exc)[:200], "latency_ms": 0}
        warnings.append(f"Chapter fallback ({heading}): {str(exc)[:200]}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)  # 汇总本章总耗时
    mode = (  # 仅成功且未跳过的 LLM 输出记作 llm 模式，否则一律按 rule 计
        "llm" if state.get("agent_mode") == "llm" and result.get("success") and not result.get("skipped")
        else "rule"
    )
    # 关闭协议追踪：写入最终章节文本、完成标记与问题列表
    protocol_event = _close_protocol_task(
        protocol_task,
        output_data={
            "report": chapter,
            "completion_ready": True,
            "issues": warnings,
        },
        warnings=warnings,
    )
    return {  # 返回状态增量：章节写入共享 bucket，供合并节点按大纲顺序组装
        "_chapter_bucket": [{
            "index": index,
            "section": section,
            "chapter": chapter,
            "result": {**result, "mode": mode, "latency_ms": result.get("latency_ms", elapsed_ms)},
        }],
        "trace": [
            _tr_event("send_dispatch", worker_type="chapter", task_id=f"chapter_{index + 1}", heading=heading),
            protocol_event,
            _tr_event("worker_started", worker_type="chapter", task_id=f"chapter_{index + 1}", heading=heading),
            _tr_event(
                "chapter_generated", node="chapter_writer_send", worker_type="chapter",
                task_id=f"chapter_{index + 1}", heading=heading, mode=mode,
                success=bool(result.get("success")), latency_ms=elapsed_ms,
                error=result.get("error", ""), model=result.get("model", ""),
            ),
            _tr_event(
                "worker_finished", worker_type="chapter", task_id=f"chapter_{index + 1}",
                heading=heading, success=True, latency_ms=elapsed_ms,
            ),
        ],
        "warnings": warnings,
    }


async def node_merge_chapter_results(state: ResearchAgentState) -> Dict[str, Any]:
    """将并行章节 worker 按大纲顺序合并为正式草稿。"""
    from app.agents.llm_reviewer import LLMDraftReviewer

    current_results = _current_worker_results(state, WorkerProfile.WRITE)
    whole_report = next((
        result for result in current_results
        if result.metadata.get("whole_report")
    ), None)
    if whole_report is not None:
        # 关键步骤：整篇写作也是 Writing Worker 产物，Merge 只负责提交正式 State。
        report = str(whole_report.output_data.get("report") or "")
        stage = _aggregate_stage_result("write", WorkerProfile.WRITE, current_results, {
            "report": report,
        })
        worker_results = dict(state.get("worker_results") or {})
        worker_results["write"] = stage.model_dump(mode="json")
        degraded = bool(whole_report.metadata.get("degraded_source_only"))
        return {
            "draft_report": report,
            "final_report": report,
            "answer": report,
            "report_completion_ready": bool(report),
            "expected_chapter_count": None,
            "written_chapter_count": None,
            "worker_results": worker_results,
            "trace": [_tr_event(
                "merge_result", node="merge_chapter_results", graph_node=True,
                merge_type="whole_report", chapter_count=1,
                degraded_source_only=degraded,
            ), _tr_event(
                "draft_reviewer_complete", node="merge_chapter_results", graph_node=True,
                report_len=len(report), parallel_chapters=False,
            )],
            "warnings": ([
                "证据抽取工具持续失败；报告仅基于可用来源元数据生成。"
            ] if degraded else []),
        }
    if current_results:
        # 关键步骤：把各章节写手的输出 + 章节序号/成败/耗时整理成统一清单。
        generated = [
            {
                "index": int(result.metadata.get("chapter_index", index) or index),
                "section": dict(result.output_data.get("section") or {}),
                "chapter": str(result.output_data.get("chapter") or result.output_data.get("report") or ""),
                "result": {
                    "success": result.status == AgentTaskStatus.SUCCESS,
                    "mode": str(result.output_data.get("mode") or result.metadata.get("mode") or ""),
                    "latency_ms": result.latency_ms,
                    "error": (result.error.message if result.error else ""),
                },
            }
            for index, result in enumerate(current_results)
        ]
    else:
        generated = list(state.get("_chapter_bucket", []))
    expected = len(state.get("outline", {}).get("sections", []))
    # 关键步骤：统计成功章节集合，只有"全部章节齐全"才算完整草稿。
    successful_indices = {
        int(item.get("index", -1)) for item in generated
        if item.get("result", {}).get("success") and str(item.get("chapter") or "").strip()
    }
    # 失败/缺失章节（含"成功但空产出"）——它们不参与正式组装，以占位符保留结构位置。
    failed_chapters = [
        item for item in generated
        if not item.get("result", {}).get("success")
        or not str(item.get("chapter") or "").strip()
    ]
    # LLM-only 门禁：只禁止"规则降级"内容混入（Worker 层已拦截，这里兜底）。
    # "LLM 真失败"的章节不再拒绝整份组装——降级为部分报告并标记完成度，
    # 交由 Reviewer 的章节数量反馈触发修复回路；修复预算耗尽时部分交付而非整跑失败。
    if _is_llm_only(state):
        rule_smuggled = [
            item for item in generated
            if item.get("result", {}).get("mode") == "rule"
        ]
        if rule_smuggled:
            raise RuntimeError(
                "LLM-only report assembly rejected rule-degraded chapters: "
                f"{[item.get('index') for item in rule_smuggled]}"
            )
    # 为失败章节注入显式"缺口"占位，保证部分报告结构完整、缺口对用户透明可见。
    # 失败 Worker 的 output_data 不含 section，需按 chapter_index 回填大纲章节名。
    outline_sections = list(state.get("outline", {}).get("sections", []))
    for item in failed_chapters:
        index = int(item.get("index", -1))
        outline_section = (
            outline_sections[index] if 0 <= index < len(outline_sections) else {}
        )
        section = dict(item.get("section") or {})
        item["section"] = section or dict(outline_section)
        heading = str(
            section.get("heading")
            or outline_section.get("heading")
            or f"章节 {index + 1}"
        )
        reason = str(item.get("result", {}).get("error") or "章节生成失败")[:300]
        item["chapter"] = (
            f"## {heading}\n\n"
            f"> ⚠️ 本章未能由真实 LLM 生成，报告为部分完成。失败原因：{reason}"
        )
    # 关键步骤：由 DraftReviewer 按大纲顺序把并行章节拼成一份正式草稿。
    result, _ = LLMDraftReviewer().assemble_generated_chapters(
        state.get("topic", ""), state.get("outline", {}), state.get("sources", []),
        state.get("evidence_cards", []), state.get("citation_check_results", []),
        state.get("citation_summary", {}), state.get("language", "zh"), generated,
    )
    stage = None
    if current_results:
        if failed_chapters and len(failed_chapters) < len(generated):
            # 关键步骤：部分章节成功时以 PARTIAL_SUCCESS 上报整个 Write 阶段，
            # Reviewer 因此走"局部修复"分支（REPAIR→begin_repair），
            # 而不是整阶段 FAILED 触发重规划；needs_replan=False 避免二次重规划。
            first_failed = next(
                (result for result in current_results
                 if result.status != AgentTaskStatus.SUCCESS),
                None,
            )
            stage = WorkerResult(
                task_id="write", profile=WorkerProfile.WRITE,
                status=AgentTaskStatus.PARTIAL_SUCCESS,
                output_data={"report": result["draft_report"]},
                needs_replan=False,
                tool_calls=[
                    call for result in current_results
                    if result.status == AgentTaskStatus.SUCCESS
                    for call in result.tool_calls
                ],
                warnings=[warning for result in current_results for warning in result.warnings],
                error=(first_failed.error if first_failed else None),
            )
        else:
            stage = _aggregate_stage_result("write", WorkerProfile.WRITE, current_results, {
                "report": result["draft_report"],
            })
    worker_results = dict(state.get("worker_results") or {})
    if stage is not None:
        worker_results["write"] = stage.model_dump(mode="json")
    return {
        "draft_report": result["draft_report"],
        "final_report": result["draft_report"],
        "answer": result["draft_report"],
        "report_completion_ready": bool(result["draft_report"]) and not failed_chapters,
        "expected_chapter_count": expected,
        "written_chapter_count": len(successful_indices),
        "report_completion_issues": [
            "章节「{}」未生成：{}".format(
                str((item.get("section") or {}).get("heading") or "未知章节"),
                str(item.get("result", {}).get("error") or "章节生成失败"),
            )
            for item in failed_chapters
        ],
        "worker_results": worker_results,
        "_chapter_bucket": [],
        "trace": [
            _tr_event(
                "merge_result", node="merge_chapter_results", graph_node=True,
                merge_type="chapters", chapter_count=len(generated),
            ),
            _tr_event(
                "draft_reviewer_complete", node="merge_chapter_results", graph_node=True,
                report_len=len(result["draft_report"]), parallel_chapters=True,
                critical_chapter_latency_ms=max(
                    (int(item.get("result", {}).get("latency_ms", 0) or 0) for item in generated),
                    default=0,
                ),
            ),
        ],
        "warnings": result.get("warnings", []),
    }


# ================================================================
# StateGraph 构建
# ================================================================

def _observe_node(node_name: str, node_fn):
    """在不改变节点结果契约的前提下，为节点透明地统计耗时。"""
    @functools.wraps(node_fn)
    async def observed(state: ResearchAgentState) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            original_delta = await node_fn(state)
        except Exception as exc:
            # 记录错误摘要，包含节点名称、异常类型、异常信息的前200个字符等
            emit_error({
                "stage": "graph_node",
                "node": node_name,
                "exception_type": type(exc).__name__,
                "error": str(exc)[:200],
            })
            raise

        latency_ms = int((time.perf_counter() - started) * 1000)
        delta = dict(original_delta or {})
        # 将节点观测事件追加到 trace 中，记录节点名称、执行耗时等信息
        delta["trace"] = list(delta.get("trace", [])) + [
            _tr_event(
                "node_observed",
                observed_node=node_name,
                graph_node=True,          # 标记为 graph 节点事件（非工具调用）
                internal_only=True,       # 内部事件，不对外暴露给前端
                success=True,
                latency_ms=latency_ms,
            )
        ]
        return delta

    return observed


def route_after_search_merge(state: ResearchAgentState):
    plan = WorkPlan.model_validate(state.get("work_plan") or {})
    current = _current_worker_results(state, WorkerProfile.SEARCH)
    reusable_read = dict((state.get("worker_results") or {}).get("read") or {})
    if (
        plan.revision > 0 and current
        and not any(result.status == AgentTaskStatus.SUCCESS for result in current)
        and reusable_read.get("status") == AgentTaskStatus.SUCCESS.value
    ):
        # 关键步骤：重规划检索没有产生新来源时复用成功下游缓存，不重复消耗工具预算。
        return "reviewer"
    return send_reading_work_items(state) if state.get("sources") else "reviewer"


def route_after_reading_merge(state: ResearchAgentState):
    result = (state.get("worker_results") or {}).get("read") or {}
    return send_analysis_work_item(state) if result.get("status") == "success" else "reviewer"


def route_after_analysis_merge(state: ResearchAgentState):
    result = (state.get("worker_results") or {}).get("analyze") or {}
    return send_citation_work_item(state) if result.get("status") == "success" else "reviewer"


def route_after_citation_merge(state: ResearchAgentState):
    result = (state.get("worker_results") or {}).get("cite") or {}
    return "outline" if result.get("status") == "success" else "reviewer"


def route_after_outline_for_writing(state: ResearchAgentState):
    sends = send_chapter_work_items(state)
    return sends if sends else "reviewer"


def build_graph_with_send() -> StateGraph:
    """生产四 Agent 主图；旧节点仅保留为可直接调用的兼容函数。"""
    graph = StateGraph(ResearchAgentState)

    graph.add_node("controller", _observe_node("controller", node_controller))
    graph.add_node("planner", _observe_node("planner", node_four_agent_planner))
    graph.add_node("worker_send", _observe_node("worker_send", node_worker_send))
    graph.add_node("merge_direct_results", _observe_node("merge_direct_results", node_merge_direct_results))
    graph.add_node("merge_context_results", _observe_node("merge_context_results", node_merge_context_results))
    graph.add_node("merge_answer_results", _observe_node("merge_answer_results", node_merge_answer_results))
    graph.add_node("merge_search_results", _observe_node("merge_search_results", node_merge_search_results))
    graph.add_node("merge_reading_results", _observe_node("merge_reading_results", node_merge_reading_results))
    graph.add_node("merge_analysis_results", _observe_node("merge_analysis_results", node_merge_analysis_results))
    graph.add_node("merge_citation_results", _observe_node("merge_citation_results", node_merge_citation_results))
    graph.add_node("outline", _observe_node("outline", node_outline))
    graph.add_node("merge_chapter_results", _observe_node("merge_chapter_results", node_merge_chapter_results))
    graph.add_node("reviewer", _observe_node("reviewer", node_four_agent_reviewer))
    graph.add_node("begin_repair", _observe_node("begin_repair", node_begin_repair))

    graph.add_edge(START, "controller")
    graph.add_edge("controller", "planner")
    graph.add_conditional_edges("planner", route_after_four_agent_planner, ["worker_send"])
    graph.add_conditional_edges(
        "worker_send", route_after_worker_send,
        [
            "merge_direct_results", "merge_context_results", "merge_answer_results", "merge_search_results",
            "merge_reading_results", "merge_analysis_results",
            "merge_citation_results", "merge_chapter_results",
        ],
    )
    graph.add_edge("merge_direct_results", "reviewer")
    graph.add_conditional_edges("merge_context_results", send_answer_work_item, ["worker_send"])
    graph.add_edge("merge_answer_results", "reviewer")
    graph.add_conditional_edges("merge_search_results", route_after_search_merge, ["worker_send", "reviewer"])
    graph.add_conditional_edges("merge_reading_results", route_after_reading_merge, ["worker_send", "reviewer"])
    graph.add_conditional_edges("merge_analysis_results", route_after_analysis_merge, ["worker_send", "reviewer"])
    graph.add_conditional_edges("merge_citation_results", route_after_citation_merge, ["outline", "reviewer"])
    graph.add_conditional_edges("outline", route_after_outline_for_writing, ["worker_send", "reviewer"])
    graph.add_edge("merge_chapter_results", "reviewer")
    graph.add_conditional_edges(
        "reviewer", route_after_four_agent_reviewer,
        ["begin_repair", "planner", END],
    )
    graph.add_conditional_edges("begin_repair", route_repair_scope, ["worker_send"])

    return graph.compile()


# 编译好的实例
_graph_send_instance = build_graph_with_send()


# ================================================================
# Trace → SSE 事件适配器（Gate B）
# ================================================================

# V1.0 公开 SSE 契约：1.x 版本内现有事件名保持稳定；
# 可以新增事件类型，但不得破坏旧的消费方。
PUBLIC_SSE_EVENT_TYPES = frozenset({
    "run_started",
    "intent_classified",
    "plan_created",
    "send_dispatch",
    "worker_started",
    "worker_finished",
    "function_call_started",
    "tool_selected",
    "tool_started",
    "tool_finished",
    "tool_args_rejected",
    "tool_rejected",
    "tool_loop_finished",
    "tool_loop_limit_reached",
    "tool_loop_fallback",
    "provider_fallback",
    "citation_checked",
    "eval_finished",
    "draft_reviewer_complete",
    "final_reviewer_complete",
    "direct_reviewer_complete",
    "merge_result",
    "source_found",
    "evidence_created",
    "outline_created",
    "chapter_generated",
    "run_finished",
    "error",
    "heartbeat",
})

_TRACE_TO_SSE_EVENT = {
    "controller_start": "run_started",
    "intent_classified": "intent_classified",
    "planner_complete": "plan_created",
    "send_dispatch": "send_dispatch",
    "worker_started": "worker_started",
    "worker_finished": "worker_finished",
    "function_call_started": "function_call_started",
    "tool_selected": "tool_selected",
    "tool_started": "tool_started",
    "tool_finished": "tool_finished",
    "tool_args_rejected": "tool_args_rejected",
    "tool_loop_finished": "tool_loop_finished",
    "tool_loop_limit_reached": "tool_loop_limit_reached",
    "tool_loop_fallback": "tool_loop_fallback",
    "analysis_complete": "worker_finished",
    "citation_complete": "citation_checked",
    "outline_created": "outline_created",
    "chapter_generated": "chapter_generated",
    "evaluator_complete": "eval_finished",
    "merge_result": "merge_result",
    "llm_started": "worker_started",
    "llm_finished": "worker_finished",
    "llm_failed": "error",
    "llm_fallback": "tool_loop_fallback",
    "direct_reviewer_complete": "direct_reviewer_complete",
}


def _trace_to_sse_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    """将内部 trace 事件转换为公开的 SSE payload。"""
    payload = {k: v for k, v in event.items()
               if k not in ("Authorization", "api_key")
               and not (isinstance(v, str) and "sk-" in v)}
    if "timestamp_ms" not in payload:
        payload["timestamp_ms"] = int(time.time() * 1000)
    return payload


# ================================================================
# 执行入口（同步：ainvoke）
# ================================================================

async def run_graph(
    topic: str,
    max_sources: int = 5,
    language: str = "zh",
    mode: str = "quick",
    run_eval: bool = True,
    agent_mode: str = None,
    run_id: str = None,
    session_context: Optional[SessionContext] = None,
    conversation_messages: Optional[List[Dict[str, Any]]] = None,
    memory_prompt: str = "",
    total_timeout_ms: Optional[int] = None,
    max_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """
    LangGraph Runtime 执行入口（同步 ainvoke，Send API 动态分发）。

    agent_mode="rule" → 完全离线；"llm" → DeepSeek V4 Flash（需 API key）
    run_id → 使用已有 run_id（async 模式），不重复创建
    max_concurrency → Send 扇出并发上限；None 保持 max(1, min(max_sources, 8))，
        传 1 作为"同图串行基线"（LangGraph 一次只调度一个 worker_send）。
    """
    max_sources = max(1, min(int(max_sources), 50))
    if agent_mode is None:
        agent_mode = os.getenv("AGENT_MODE", "llm")

    backend = "graph_send"
    effective_timeout_ms = max(
        1, int(total_timeout_ms or os.getenv("RESEARCH_TOTAL_TIMEOUT_MS", "480000"))
    )

    # 关键步骤：入口预先持有 run_id，确保图中途异常也能释放同一运行级预算。
    effective_run_id = run_id or run_store.create(topic=topic)
    initial_state: ResearchAgentState = {
        "topic": topic, "max_sources": max_sources,
        "language": language, "mode": mode, "run_eval": run_eval,
        "backend": backend, "agent_mode": agent_mode,
        "llm_only": (
            agent_mode == "llm"
            and os.getenv("LLM_ONLY_MODE", "true").lower() == "true"
        ),
        "trace": [], "warnings": [],
        "replan_count": 0, "retry_count": 0, "repair_count": 0,
        "execution_round": 0, "worker_results": {},
        "sources": [], "scored_sources": [], "evidence_cards": [],
        "_search_bucket": [], "_reading_bucket": [], "_chapter_bucket": [],
        "_worker_result_bucket": [], "_worker_route_bucket": [],
        "session_id": session_context.session_id if session_context else "",
        "session_context": session_context,
        "conversation_messages": conversation_messages or [],
        "memory_prompt": memory_prompt,
        "total_timeout_ms": effective_timeout_ms,
    }

    initial_state["run_id"] = effective_run_id

    graph = _graph_send_instance
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(
                initial_state,
                config={"max_concurrency": (
                    max_concurrency
                    if max_concurrency is not None
                    else max(1, min(max_sources, 8))
                )},
            ),
            timeout=effective_timeout_ms / 1000.0,
        )
        if initial_state["llm_only"]:
            llm_only_errors = [
                str(warning) for warning in final_state.get("warnings", [])
                if "LLM-only" in str(warning)
            ]
            # 规则降级已在 Worker/Merge 层拦截；走到这里仍残留的 "LLM-only" 警告
            # 属于"LLM 真失败"。此时只要已组装出可交付的报告——无论是修复后完整
            # （ready=True）还是部分组装（ready=False 但有已写出章节）——都不再整跑
            # raise，交由 _finalize_run 判定为 partial/completed_with_warnings；
            # 只有"没有任何可交付章节"时才保持 fail-closed。
            # 注：warnings 是 operator.add 累积字段，round-0 失败警告在修复成功后仍残留，
            #     所以不能只看"无 LLM-only 警告"，而要判断是否已有可用交付物。
            usable_report = (
                final_state.get("report_completion_ready") is True
                or int(final_state.get("written_chapter_count") or 0) > 0
            )
            if llm_only_errors and not usable_report:
                raise RuntimeError(llm_only_errors[0])
        return _finalize_run(final_state, topic, backend, agent_mode)
    except asyncio.TimeoutError:
        # 关键步骤：整图墙钟到期统一落库为 failed，不能留下永久 running 记录。
        run_store.update(
            effective_run_id,
            status="failed",
            error="Runtime total timeout exceeded",
            warnings=["Runtime total timeout exceeded"],
        )
        run_store.finish(effective_run_id, "failed")
        return run_store.get(effective_run_id) or {
            "run_id": effective_run_id, "status": "failed", "topic": topic,
            "error": "Runtime total timeout exceeded",
        }
    finally:
        clear_runtime_budget(effective_run_id)


async def _stream_with_deadline(stream, timeout_seconds: float):
    """在不破坏逐块 streaming 的前提下对整个异步迭代施加同一截止时间。"""
    iterator = stream.__aiter__()
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        try:
            yield await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return


async def _run_graph_async_impl(
    topic: str,
    max_sources: int = 5,
    language: str = "zh",
    mode: str = "quick",
    run_eval: bool = True,
    agent_mode: str = None,
    run_id: str = None,
    session_context: Optional[SessionContext] = None,
    conversation_messages: Optional[List[Dict[str, Any]]] = None,
    memory_prompt: str = "",
    total_timeout_ms: Optional[int] = None,
    max_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """
    LangGraph Runtime 执行入口（astream 渐进 SSE，Send API 动态分发）。

    使用 astream(stream_mode="updates") 在每个节点完成后立即发布事件。
    实现了真正的渐进式 SSE，不是等整个 run 完成后批量回放。
    max_concurrency → Send 扇出并发上限；None 保持 max(1, min(max_sources, 8))，
        传 1 作为"同图串行基线"。
    """
    max_sources = max(1, min(int(max_sources), 50))
    if agent_mode is None:
        agent_mode = os.getenv("AGENT_MODE", "llm")

    backend = "graph_send"
    effective_timeout_ms = max(
        1, int(total_timeout_ms or os.getenv("RESEARCH_TOTAL_TIMEOUT_MS", "480000"))
    )

    # 关键步骤：SSE 与预算使用同一预分配 run_id，异常路径也不会遗留共享预算。
    effective_run_id = run_id or run_store.create(topic=topic)
    initial_state: ResearchAgentState = {
        "topic": topic, "max_sources": max_sources,
        "language": language, "mode": mode, "run_eval": run_eval,
        "backend": backend, "agent_mode": agent_mode,
        "llm_only": (
            agent_mode == "llm"
            and os.getenv("LLM_ONLY_MODE", "true").lower() == "true"
        ),
        "trace": [], "warnings": [],
        "replan_count": 0, "retry_count": 0, "repair_count": 0,
        "execution_round": 0, "worker_results": {},
        "sources": [], "scored_sources": [], "evidence_cards": [],
        "_search_bucket": [], "_reading_bucket": [], "_chapter_bucket": [],
        "_worker_result_bucket": [], "_worker_route_bucket": [],
        "session_id": session_context.session_id if session_context else "",
        "session_context": session_context,
        "conversation_messages": conversation_messages or [],
        "memory_prompt": memory_prompt,
        "total_timeout_ms": effective_timeout_ms,
    }

    initial_state["run_id"] = effective_run_id

    # 编译 LangGraph 实例（Send API 并行分发）
    graph = _graph_send_instance
    from app.services.event_broker import event_broker

    final_run_id = effective_run_id
    aggregated_state = dict(initial_state)
    seen_source_ids = set()
    seen_card_keys = set()

    try:
        # ---- 使用 astream 实现真正的渐进式 SSE ----
        async for chunk in _stream_with_deadline(
            graph.astream(
                initial_state, stream_mode="updates",
                config={"max_concurrency": (
                    max_concurrency
                    if max_concurrency is not None
                    else max(1, min(max_sources, 8))
                )},
            ),
            effective_timeout_ms / 1000.0,
        ):
            # 示例：chunk = {node_name: state_delta}
            for node_name, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue

                # 发布 trace 事件 -> 映射为 SSE 事件
                trace_events = delta.get("trace", [])
                for ev in trace_events:
                    if ev.get("internal_only"):
                        continue
                    event_type = ev.get("event", "unknown")
                    sse_type = _TRACE_TO_SSE_EVENT.get(event_type, event_type)
                    payload = _trace_to_sse_payload(ev)
                    payload["node"] = node_name
                    # 发布 SSE 事件
                    event_broker.publish(final_run_id, sse_type, payload)

                # 发布 source_found（去重）
                new_sources = delta.get("sources", [])
                for s in new_sources:
                    sid = s.get("source_id", "")
                    if sid and sid not in seen_source_ids:
                        seen_source_ids.add(sid)
                        # 把论文信息发布到 SSE 事件
                        event_broker.publish(final_run_id, "source_found", {
                            "source_id": sid,
                            "title": (s.get("title", "") or "")[:100],
                            "source_type": s.get("source_type", "unknown"),
                            "url": s.get("url", ""),
                            "authors": (s.get("authors", []) or [])[:8],
                            "year": s.get("year"),
                            "venue": s.get("venue", ""),
                            "provider": s.get("provider", ""),
                            "snippet": (s.get("snippet", "") or "")[:320],
                            "cited_by_count": s.get("cited_by_count"),
                        })

                # 发布 evidence_created（去重）
                new_cards = delta.get("evidence_cards", [])
                for c in new_cards:
                    cid = c.get("source_id", "")
                    claim = (c.get("claim", "") or "")[:80]
                    card_key = f"{cid}::{claim}"
                    if card_key not in seen_card_keys:
                        seen_card_keys.add(card_key)
                        # 把证据卡片信息发布到 SSE 事件
                        event_broker.publish(final_run_id, "evidence_created", {
                            "source_id": cid,
                            "claim": claim,
                            "confidence": c.get("confidence", 0),
                        })

                # 聚合状态
                _apply_delta(aggregated_state, delta)

                # 发送 heartbeat
                event_broker.publish(final_run_id, "heartbeat", {"node": node_name})

    except asyncio.CancelledError:
        total_latency_ms = max(
            0,
            int(time.time() * 1000) - int(aggregated_state.get("start_time_ms") or int(time.time() * 1000)),
        )
        warnings = list(aggregated_state.get("warnings", [])) + ["Run cancelled"]
        observability_metrics = aggregate_run_metrics(
            aggregated_state.get("trace", []),
            total_latency_ms=total_latency_ms,
            status="cancelled",
            backend=backend,
            agent_mode=agent_mode,
            retry_count=aggregated_state.get("retry_count", 0),
            replan_count=aggregated_state.get("replan_count", 0),
            warnings=warnings,
            sources=aggregated_state.get("sources", []),
            evidence_cards=aggregated_state.get("evidence_cards", []),
            citation_check_results=aggregated_state.get("citation_check_results", []),
        )
        event_broker.publish(final_run_id, "error", {
            "message": "Run cancelled", "error_type": "CancelledError",
        })
        event_broker.finish_run(final_run_id)
        # 关键步骤:取消路径同样保留已产生的中间产物,不伪造最终报告
        run_store.update(
            final_run_id,
            status="cancelled",
            warnings=warnings,
            trace=aggregated_state.get("trace", []),
            total_latency_ms=total_latency_ms,
            observability_metrics=observability_metrics,
            **_partial_artifact_fields(aggregated_state),
        )
        run_store.finish(final_run_id, "cancelled")
        return run_store.get(final_run_id) or {
            "run_id": final_run_id, "status": "cancelled", "topic": topic,
            "observability_metrics": observability_metrics,
        }
    except Exception as e:
        total_latency_ms = max(
            0,
            int(time.time() * 1000) - int(aggregated_state.get("start_time_ms") or int(time.time() * 1000)),
        )
        warnings = list(aggregated_state.get("warnings", [])) + [f"Runtime error: {str(e)[:200]}"]
        observability_metrics = aggregate_run_metrics(
            aggregated_state.get("trace", []),
            total_latency_ms=total_latency_ms,
            status="failed",
            backend=backend,
            agent_mode=agent_mode,
            retry_count=aggregated_state.get("retry_count", 0),
            replan_count=aggregated_state.get("replan_count", 0),
            warnings=warnings,
            sources=aggregated_state.get("sources", []),
            evidence_cards=aggregated_state.get("evidence_cards", []),
            citation_check_results=aggregated_state.get("citation_check_results", []),
        )
        event_broker.publish(final_run_id, "error", {
            "message": str(e)[:500], "error_type": type(e).__name__,
        })
        event_broker.finish_run(final_run_id, str(e)[:200])
        # 关键步骤:失败时保留异常发生前真实产生的中间产物
        # (检索/证据/引用/大纲/草稿等),status 保持 failed,final_report 不伪造
        run_store.update(
            final_run_id,
            status="failed",
            error=str(e)[:500],
            warnings=warnings,
            trace=aggregated_state.get("trace", []),
            total_latency_ms=total_latency_ms,
            observability_metrics=observability_metrics,
            **_partial_artifact_fields(aggregated_state),
        )
        run_store.finish(final_run_id, "failed")
        return run_store.get(final_run_id) or {
            "run_id": final_run_id, "status": "failed", "topic": topic,
            "error": str(e)[:200], "observability_metrics": observability_metrics,
        }

    result = _finalize_run(aggregated_state, topic, backend, agent_mode)
    return result


async def run_graph_async(
    topic: str,
    max_sources: int = 5,
    language: str = "zh",
    mode: str = "quick",
    run_eval: bool = True,
    agent_mode: str = None,
    run_id: str = None,
    session_context: Optional[SessionContext] = None,
    conversation_messages: Optional[List[Dict[str, Any]]] = None,
    memory_prompt: str = "",
    total_timeout_ms: Optional[int] = None,
    max_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """异步公开入口；所有成功、异常、取消与 finalizer 异常都统一释放预算。"""
    effective_run_id = run_id or run_store.create(topic=topic)
    try:
        return await _run_graph_async_impl(
            topic=topic,
            max_sources=max_sources,
            language=language,
            mode=mode,
            run_eval=run_eval,
            agent_mode=agent_mode,
            run_id=effective_run_id,
            session_context=session_context,
            conversation_messages=conversation_messages,
            memory_prompt=memory_prompt,
            total_timeout_ms=total_timeout_ms,
            max_concurrency=max_concurrency,
        )
    finally:
        # 关键步骤：清理由公开入口 finally 承担，stream/finalizer 任一层异常都不会泄漏预算。
        clear_runtime_budget(effective_run_id)


def _partial_artifact_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """提取失败/取消前已真实产生的中间产物字段,供 RunStore 保存。

    只透传真实状态里的值:没有产生的阶段保持空,final_report 不在此列,
    避免把部分成功包装成端到端成功。
    """
    sources = list(state.get("sources", []) or [])
    evidence_cards = list(state.get("evidence_cards", []) or [])
    return {
        "draft_report": state.get("draft_report", "") or "",
        "sources": sources,
        "discovered_source_count": max(
            int(state.get("discovered_source_count", 0) or 0), len(sources),
        ),
        "analyzed_source_count": max(
            int(state.get("analyzed_source_count", 0) or 0),
            len(state.get("scored_sources", []) or sources),
        ),
        "analysis_selection": state.get("analysis_selection", {}) or {},
        "scored_sources": state.get("scored_sources", []) or [],
        "evidence_cards": evidence_cards,
        "citation_check_results": state.get("citation_check_results", []) or [],
        "citation_summary": state.get("citation_summary", {}) or {},
        "outline": state.get("outline", {}) or {},
        "source_matrix": _build_source_matrix(sources, evidence_cards),
        "task_dag": state.get("task_dag", {}) or {},
        "selected_tools": state.get("selected_tools", []) or [],
        "selected_tool_args": state.get("selected_tool_args", {}) or {},
        "eval_metrics": state.get("eval_metrics", {}) or {},
        "eval_feedback": state.get("eval_feedback", []) or [],
        "intent": state.get("intent", "deep_research"),
        "execution_route": state.get("execution_route", "full_research"),
        "retry_count": state.get("retry_count", 0),
        "replan_count": state.get("replan_count", 0),
    }


def _apply_delta(aggregated: Dict[str, Any], delta: Dict[str, Any]):
    """将节点 delta 合并到聚合状态。"""
    reducer_fields = {
        "trace", "warnings", "_search_bucket", "_reading_bucket",
        "_chapter_bucket", "_worker_result_bucket", "_worker_route_bucket",
    }
    for k, v in delta.items():
        if k in reducer_fields:
            if isinstance(v, list):
                aggregated.setdefault(k, [])
                aggregated[k] = aggregated[k] + v
        else:
            # 关键步骤：正式 State 必须允许空容器、0、False 与空字符串覆盖旧轮数据。
            aggregated[k] = v


def _finalize_run(
    final_state: Dict[str, Any],
    topic: str,
    backend: str,
    agent_mode: str,
) -> Dict[str, Any]:
    """结束运行：构建来源矩阵、确定状态、保存到 run_store、发布 run_finished。"""
    from app.services.event_broker import event_broker

    # 1. 构建来源矩阵：将 sources 和 evidence_cards 合并为可展示的矩阵
    source_matrix = _build_source_matrix(
        final_state.get("sources", []),
        final_state.get("evidence_cards", []),
    )

    run_id = final_state.get("run_id", str(uuid.uuid4())[:8])
    
    # 2. 根据 worker 执行结果判断运行状态
    #    - 所有 worker 都失败 → failed
    #    - 有警告信息 → completed_with_warnings
    #    - 其他情况 → completed
    worker_events = [t for t in final_state.get("trace", [])
                     if t.get("event") == "worker_finished"]
    all_failed = worker_events and all(
        not t.get("success", True) for t in worker_events
    )
    verdict_outcome = str((final_state.get("review_verdict") or {}).get("outcome") or "")
    if verdict_outcome in {"fail", "repair", "replan"}:
        # 部分组装兜底：修复/重规划预算耗尽后，仍有成功章节组成的可交付草稿时，
        # 不把整跑判死为 failed，而是降级为 partial（部分交付），由前端标注未完成章节。
        partial_report_delivered = (
            final_state.get("expected_chapter_count") is not None
            and int(final_state.get("written_chapter_count") or 0) > 0
            and final_state.get("report_completion_ready") is False
        )
        status = "partial" if partial_report_delivered else "failed"
    elif verdict_outcome == "clarify":
        status = "partial"
    elif all_failed:
        status = "failed"
    elif final_state.get("strict_completion_required") and final_state.get("report_completion_ready") is False:
        status = "partial"
    elif final_state.get("warnings") or int(final_state.get("retry_count", 0) or 0) > 0:
        status = "completed_with_warnings"
    else:
        status = "completed"

    # 3. 聚合可观测性指标（工具调用次数、LLM 调用次数、token 消耗等）
    observability_metrics = aggregate_run_metrics(
        final_state.get("trace", []),
        total_latency_ms=final_state.get("total_latency_ms", 0),
        status=status,
        backend=backend,
        agent_mode=agent_mode,
        retry_count=final_state.get("retry_count", 0),
        replan_count=final_state.get("replan_count", 0),
        warnings=final_state.get("warnings", []),
        sources=final_state.get("sources", []),
        evidence_cards=final_state.get("evidence_cards", []),
        citation_check_results=final_state.get("citation_check_results", []),
    )

    # 4. 将所有运行结果持久化到 run_store
    run_store.update(
        run_id, status=status, topic=topic,
        session_id=final_state.get("session_id", ""),
        research_topic=final_state.get("research_topic", topic),
        intent=final_state.get("intent", "deep_research"),
        execution_route=final_state.get("execution_route", "full_research"),
        selected_tools=final_state.get("selected_tools", []),
        selected_tool_args=final_state.get("selected_tool_args", {}),
        requested_count=final_state.get("requested_count", 0),
        intent_confidence=final_state.get("intent_confidence", 0.0),
        controller_reasoning=final_state.get("controller_reasoning", ""),
        is_follow_up=final_state.get("is_follow_up", False),
        reference_expression=final_state.get("reference_expression", ""),
        resolved_paper_ids=final_state.get("resolved_paper_ids", []),
        seed_paper_ids=final_state.get("seed_paper_ids", []),
        resolved_section=final_state.get("resolved_section"),
        clarification_message=final_state.get("clarification_message", ""),
        missing_ordinal=final_state.get("missing_ordinal"),
        conversation_operation=final_state.get("conversation_operation", ""),
        fallback_used=final_state.get("fallback_used", False),
        route_name=final_state.get("execution_route", "full_research"),
        answer=final_state.get("answer", final_state.get("final_report", "")),
        conversation_result=final_state.get("conversation_result", {}),
        final_report=final_state.get("final_report", ""),
        draft_report=final_state.get("draft_report", ""),
        sources=final_state.get("sources", []),
        discovered_source_count=max(
            int(final_state.get("discovered_source_count", 0) or 0),
            len(final_state.get("sources", [])),
        ),
        analyzed_source_count=max(
            int(final_state.get("analyzed_source_count", 0) or 0),
            len(final_state.get("scored_sources", []) or final_state.get("sources", [])),
        ),
        analysis_selection=final_state.get("analysis_selection", {}),
        evidence_cards=final_state.get("evidence_cards", []),
        citation_check_results=final_state.get("citation_check_results", []),
        citation_summary=final_state.get("citation_summary", {}),
        outline=final_state.get("outline", {}),
        source_matrix=source_matrix,
        eval_metrics=final_state.get("eval_metrics", {}),
        eval_metric_details=final_state.get("eval_metric_details", {}),
        eval_feedback=final_state.get("eval_feedback", []),
        review_verdict=final_state.get("review_verdict", {}),
        fixes_applied=final_state.get("fixes_applied", []),
        unresolved_issues=final_state.get("unresolved_issues", []),
        report_completion_ready=final_state.get("report_completion_ready", False),
        report_completion_issues=final_state.get("report_completion_issues", []),
        warnings=final_state.get("warnings", []),
        trace=final_state.get("trace", []),
        task_dag=final_state.get("task_dag", {}),
        retry_attempted=final_state.get("retry_count", 0) > 0,
        total_latency_ms=final_state.get("total_latency_ms", 0),
        replan_count=final_state.get("replan_count", 0),
        retry_count=final_state.get("retry_count", 0),
        backend=backend,
        agent_mode=agent_mode,
        observability_metrics=observability_metrics,
    )
    run_store.finish(run_id, status)

    # 5. Gate B: 向前端推送 run_finished SSE 事件，通知运行结束
    event_broker.publish(run_id, "run_finished", {
        "run_id": run_id,
        "status": status,
        "total_latency_ms": final_state.get("total_latency_ms", 0),
        "observability": {
            "tool_calls": observability_metrics["tools"]["call_count"],
            "tool_errors": observability_metrics["tools"]["error_count"],
            "llm_calls": observability_metrics["llm"]["call_count"],
            "total_tokens": observability_metrics["llm"]["total_tokens"],
        },
    })
    # 关闭 SSE 流
    event_broker.finish_run(run_id)

    # 6. 返回完整的 run 记录
    return run_store.get(run_id) or {"run_id": run_id, "status": status, "topic": topic}


def _build_source_matrix(sources, evidence_cards) -> List[Dict]:
    matrix = []
    for s in (sources or []):
        sid = s.get("source_id", "")
        card_count = sum(1 for c in (evidence_cards or []) if c.get("source_id") == sid)
        authors = s.get("authors", [])
        author_str = ", ".join(authors[:2]) if isinstance(authors, list) else str(authors or "")
        if isinstance(authors, list) and len(authors) > 2:
            author_str += " et al."
        matrix.append({
            "source_id": sid, "title": s.get("title", ""),
            "authors": author_str, "year": s.get("year"),
            "venue": s.get("venue", ""),
            "source_type": s.get("source_type", "unknown"),
            "quality_score": s.get("quality_score", 0.0),
            "key_contribution": f"Contributed {card_count} evidence card(s)",
        })
    return matrix
    register_runtime_budget,
