"""Intent-driven Controller：把用户请求分类为意图，并决定执行路由。

Controller 只回答三个问题：
1. intent —— 用户到底要什么（推荐/搜索/图谱/深度调研/会话追问）
2. execution_route —— 走哪条执行链路（direct_tool / full_research / conversation）
3. research_topic + requested_count —— 供 Planner 与 Worker 使用的输入参数

关键架构决策：Controller 不再替 direct 意图挑选具体工具。
工具选择完全下沉到 Worker 的 ReAct 循环（LLM 从注册表目录自主挑选）。
因此 direct 意图的 selected_tools / selected_tool_args 恒为空，仅作兼容占位。
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.llm.client import get_llm_client
from app.llm.prompts import build_system_prompt_with_memory
from app.llm.schemas import LLMIntentOutput
from app.agents.reference_resolver import ReferenceResolver, ReferenceResolution
from app.services.session_store import SessionContext
from app.tools.registry import ToolRegistry
from app.agents.schemas import (
    ExecutionBudget,
    ExecutionClass,
    ExecutionSpec,
    SafetyPolicy,
)


class ControllerDecision(BaseModel):
    intent: str
    execution_route: str
    research_topic: str
    selected_tools: List[str] = Field(default_factory=list)
    selected_tool_args: Dict[str, Any] = Field(default_factory=dict)
    requested_count: int = Field(default=5, ge=1, le=50)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""
    classifier: str = "rule"
    llm_result: Dict[str, Any] = Field(default_factory=dict)
    is_follow_up: bool = False
    reference_expression: str = ""
    resolved_paper_ids: List[str] = Field(default_factory=list)
    seed_paper_ids: List[str] = Field(default_factory=list)
    resolved_section: Optional[str] = None
    conversation_operation: str = ""
    session_id: Optional[str] = None
    fallback_used: bool = False
    clarification_message: str = ""
    missing_ordinal: Optional[int] = None


class IntentController:
    """Classify natural-language requests before planning their execution path."""

    INTENTS = {
        "paper_recommendation",
        "recommend_more",
        "literature_search",
        "paper_graph_lookup",
        "deep_research",
        "research_from_session",
        "paper_qa",
        "paper_compare",
        "report_follow_up",
        "conversation",
    }
    ROUTES = {
        "direct_tool", "full_research", "conversation"
    }

    def __init__(self):
        self.reference_resolver = ReferenceResolver()

    async def decide(
        self,
        user_request: str,
        max_sources: int = 5,
        agent_mode: str = "llm",
        session_context: Optional[SessionContext] = None,
        memory_prompt: str = "",
    ) -> ControllerDecision:
        max_sources = max(1, min(int(max_sources), 50))
        registry = ToolRegistry.get_instance()
        available_tools = registry.list_for_task("search")
        resolution = ReferenceResolution()
        if session_context is not None:
            resolution = await self.reference_resolver.resolve(
                user_request, session_context, allow_llm=(agent_mode == "llm")
            )
            if resolution.missing_ordinal is not None:
                # 关键步骤：第 N 篇明确不存在时固定走会话澄清，不允许回退到 full_research。
                return self._missing_reference_decide(
                    user_request, max_sources, session_context, resolution
                )
            # 续接动作必须先于普通 follow-up 判断，否则“这些论文/生成报告”会被 conversation 吞掉。
            continuation = self._continuation_decide(
                user_request, max_sources, session_context, resolution
            )
            if continuation is not None:
                return continuation
            if self._is_follow_up(user_request, session_context, resolution):
                follow_up = self._conversation_decide(
                    user_request, max_sources, session_context, resolution
                )
                if follow_up is not None:
                    return follow_up
        rule_decision = self._rule_decide(user_request, max_sources)
        if agent_mode != "llm":
            return rule_decision

        catalog = "\n".join(
            f"- {name}: {registry.get(name).description}"
            for name in available_tools
            if registry.get(name) is not None
        )
        result = await get_llm_client().classify_intent(
            system_prompt=build_system_prompt_with_memory(_CONTROLLER_SYSTEM, memory_prompt),
            user_prompt=_CONTROLLER_USER.format(
                user_request=user_request,
                max_sources=max_sources,
                tool_catalog=catalog,
            ),
            output_schema=LLMIntentOutput,
        )
        if not result.get("success"):
            rule_decision.llm_result = result
            rule_decision.reasoning += "; LLM classifier unavailable, used rule fallback"
            return rule_decision

        output: LLMIntentOutput = result["data"]
        # 关键步骤：只校验意图与路由是否合法；工具名由 Worker 的 ReAct 循环选择，
        # 不再在 Controller 层约束（selected_tool 字段仅作 schema 兼容，值被忽略）。
        if output.intent not in self.INTENTS or output.execution_route not in self.ROUTES:
            rule_decision.llm_result = {
                **result,
                "success": False,
                "error": "Controller output violated route allowlist",
            }
            rule_decision.reasoning += "; invalid LLM decision, used rule fallback"
            return rule_decision

        research_topic = output.research_topic.strip() or user_request.strip()
        return ControllerDecision(
            intent=output.intent,  # 意图：如 paper_recommendation / paper_graph_lookup
            execution_route=output.execution_route,  # 执行路由：direct_tool / full_research / conversation
            research_topic=research_topic,  # 清理后的研究主题，供 Planner 与 Worker 使用
            # 关键步骤：Controller 不选工具——direct 意图的工具与参数
            # 由 Planner 确定性构造（规则模式）或 Worker 的 ReAct 循环自主决定（LLM 模式）。
            selected_tools=[],
            selected_tool_args={},
            requested_count=min(max_sources, output.requested_count),
            confidence=output.confidence,
            reasoning=output.reasoning,
            classifier="llm",
            llm_result=result,
            session_id=session_context.session_id if session_context else None,
        )


    def _continuation_decide(
        self,
        query: str,
        max_sources: int,
        session: SessionContext,
        resolution: ReferenceResolution,
    ) -> Optional[ControllerDecision]:
        """识别需要继续执行工具或 Research DAG 的会话动作。"""
        if not session.recommended_papers:
            return None

        lowered = query.lower().strip()
        # 继续推荐论文（"推荐更多/想推荐更多"是自然口语变体，与"再推荐/继续推荐"同义）
        recommend_more_patterns = (
            r"推荐\s*更多", r"想推荐更多", r"再(?:给我)?推荐", r"继续推荐",
            r"更多(?:的)?(?:论文|文献)", r"再来\s*\d*\s*(?:篇|个)",
            r"recommend\s+(?:me\s+)?more", r"(?:another|more)\s+\d*\s*(?:papers?|articles?)",
        )
        if any(re.search(pattern, lowered, re.I) for pattern in recommend_more_patterns):
            requested_count = _requested_count(query, max_sources)
            topic = session.last_recommendation_topic or _extract_research_topic(query)
            return ControllerDecision(
                intent="recommend_more",
                execution_route="direct_tool",
                research_topic=topic,
                # 关键步骤：会话已推荐论文作为资源引用注入 WorkItem，
                # Worker 的 ReAct 循环会读取并去重，Controller 不再选工具。
                selected_tools=[],
                selected_tool_args={},
                requested_count=requested_count,
                confidence=0.99,
                reasoning="Session continuation requests additional non-duplicate recommendations",
                classifier="session_rule",
                is_follow_up=True,
                reference_expression="已推荐论文",
                session_id=session.session_id,
            )
        # 基于会话累计论文进行研究
        research_from_session_patterns = (
            r"基于(?:这些|上述|上面|刚才)(?:的)?(?:论文|文献)",
            r"用(?:这些|上述|上面|刚才)(?:的)?(?:论文|文献).*(?:报告|调研)",
            r"根据(?:这些|上述|上面|刚才)(?:的)?(?:论文|文献).*(?:报告|调研)",
            r"(?:based on|using) (?:these|the above) (?:papers|articles|studies)",
        )
        requests_research = any(
            marker in lowered
            for marker in ("报告", "调研", "综述", "分析", "report", "research", "review", "analyze")
        )
        if requests_research and any(
            re.search(pattern, lowered, re.I) for pattern in research_from_session_patterns
        ):
            # “这些论文”默认指向会话累计论文；显式解析到具体论文时则缩小 Seed 范围。
            all_ids = self._session_paper_ids(session.recommended_papers)
            seed_ids = list(dict.fromkeys(resolution.resolved_paper_ids or all_ids))[:50]
            topic = session.last_recommendation_topic or query.strip()
            return ControllerDecision(
                intent="research_from_session",
                execution_route="full_research",
                research_topic=topic,
                requested_count=max(max_sources, min(len(seed_ids), 50)),
                confidence=0.99,
                reasoning="Session papers are injected as seeds into the full research DAG",
                classifier="session_rule",
                is_follow_up=True,
                reference_expression=resolution.reference_expression or "这些论文",
                resolved_paper_ids=seed_ids,
                seed_paper_ids=seed_ids,
                session_id=session.session_id,
                fallback_used=resolution.fallback_used,
            )
        return None

    @staticmethod
    def _session_paper_ids(papers: List[Dict[str, Any]]) -> List[str]:
        """按会话顺序提取可传入 DAG 的稳定论文 ID。"""
        return [
            str(item.get("source_id") or item.get("paper_id") or "")
            for item in papers
            if item.get("source_id") or item.get("paper_id")
        ]

    @staticmethod
    def _is_follow_up(
        query: str,
        session: SessionContext,
        resolution: ReferenceResolution,
    ) -> bool:
        if not (session.active_paper_id or session.active_report_id or session.recommended_papers):
            return False
        lowered = query.lower()
        markers = (
            "这篇", "该论文", "它", "第一", "第二", "第三", "第1", "第2", "第3",
            "前者", "后者", "上面", "刚才", "报告", "章节", "部分", "展开", "溯源",
            "对比", "比较", "this paper", "first paper", "second paper", "report",
            "section", "compare", "former", "latter",
            "继续", "接着", "详细说", "进一步", "为什么", "它们", "这些",
            "continue", "go on", "elaborate", "more detail", "why",
        )
        return bool(
            resolution.resolved_paper_ids
            or resolution.resolved_section
            or (len(query.strip()) <= 120 and any(marker in lowered for marker in markers))
        )

    def _conversation_decide(
        self,
        query: str,
        max_sources: int,
        session: SessionContext,
        resolution: ReferenceResolution,
    ) -> Optional[ControllerDecision]:
        lowered = query.lower()
        report_markers = (
            "报告", "章节", "部分", "展开", "扩写", "溯源", "证据空白",
            "report", "section", "expand", "trace evidence", "fill gap",
        )
        compare_markers = ("比较", "对比", "区别", "异同", "compare", "versus", " vs ")
        paper_question_markers = (
            "方法", "创新", "实验", "结论", "局限", "数据集", "指标", "结果",
            "method", "novelty", "experiment", "conclusion", "limitation", "dataset",
        )
        ids = list(dict.fromkeys(resolution.resolved_paper_ids))
        if session.active_report_id and (
            resolution.resolved_section or any(marker in lowered for marker in report_markers)
        ):
            operation = "report_follow_up"
        elif len(ids) >= 2 or (len(ids) >= 1 and any(marker in lowered for marker in compare_markers)):
            operation = "paper_compare"
        elif ids or (
            session.active_paper_id and any(marker in lowered for marker in paper_question_markers)
        ):
            if not ids and session.active_paper_id:
                ids = [session.active_paper_id]
            operation = "paper_qa"
        else:
            operation = ""
        return ControllerDecision(
            intent=operation or "conversation",
            execution_route="conversation",
            conversation_operation=operation,
            research_topic=query.strip(),
            requested_count=max_sources,
            confidence=max(0.75, resolution.confidence),
            reasoning=f"Follow-up routed with session reference resolution: {resolution.reasoning}",
            classifier="session_rule",
            is_follow_up=True,
            reference_expression=resolution.reference_expression,
            resolved_paper_ids=ids,
            resolved_section=resolution.resolved_section,
            session_id=session.session_id,
            fallback_used=resolution.fallback_used,
        )

    @staticmethod
    def _missing_reference_decide(
        query: str,
        max_sources: int,
        session: SessionContext,
        resolution: ReferenceResolution,
    ) -> ControllerDecision:
        ordinal = resolution.missing_ordinal
        message = f"当前会话找不到第 {ordinal} 篇"
        return ControllerDecision(
            intent="conversation",
            execution_route="conversation",
            conversation_operation="reference_not_found",
            research_topic=query.strip(),
            requested_count=max_sources,
            confidence=1.0,
            reasoning=message,
            classifier="session_rule",
            is_follow_up=True,
            reference_expression=resolution.reference_expression,
            session_id=session.session_id,
            clarification_message=message,
            missing_ordinal=ordinal,
        )

    def _rule_decide(
        self,
        user_request: str,
        max_sources: int,
    ) -> ControllerDecision:
        lowered = user_request.lower()
        requested_count = _requested_count(user_request, max_sources)
        research_topic = _extract_research_topic(user_request)

        deep_markers = (
            "调研", "研究报告", "写报告", "生成报告", "总结", "分析",
            "比较", "对比", "综合这些", "归纳", "给出结论",
            "compare", "analyze", "summarize", "synthesize",
            "write a report", "generate a report",
        )
        recommend_markers = ("推荐", "recommend", "suggest")
        search_markers = ("搜索", "查找", "检索", "列出", "find", "search", "list")
        paper_markers = ("论文", "文献", "paper", "article", "study")

        needs_synthesis = any(marker in lowered for marker in deep_markers)
        is_paper_request = any(marker in lowered for marker in paper_markers)
        asks_recommendation = any(marker in lowered for marker in recommend_markers)
        asks_search = any(marker in lowered for marker in search_markers)
        explicit_semantic_scholar = any(
            marker in lowered
            for marker in ("semantic scholar", "semanticscholar", "语义学者")
        )
        graph_markers = (
            "被哪些论文引用", "被引论文", "引用它的论文", "参考文献",
            "引用了哪些", "引用关系", "引用文章", "后续引用", "引用工作",
            "论文详情", "论文元数据", "元数据", "作者是谁", "论文作者",
            "paper details", "authors of", "citations of", "citing papers",
            "references of", "references for", "paper metadata",
        )
        # 处理图谱标记请求（工具与参数的解析下沉给 Planner/Worker，这里只定意图）
        if any(marker in lowered for marker in graph_markers) and not needs_synthesis:
            return ControllerDecision(
                intent="paper_graph_lookup",
                execution_route="direct_tool",
                research_topic=user_request.strip(),
                selected_tools=[],
                selected_tool_args={},
                requested_count=requested_count,
                confidence=0.94,
                reasoning=(
                    "Paper graph lookup is routed to the direct ReAct worker, "
                    "which parses the paper name and relation from the user request"
                ),
            )
        # 处理推荐请求（规则模式由 Planner 按意图确定性选工具，LLM 模式由 ReAct 自主选）
        if (
            (is_paper_request or explicit_semantic_scholar)
            and asks_recommendation
            and not needs_synthesis
        ):
            return ControllerDecision(
                intent="paper_recommendation",
                execution_route="direct_tool",
                research_topic=research_topic,
                selected_tools=[],
                selected_tool_args={},
                requested_count=requested_count,
                confidence=0.95,
                reasoning="Recommendation request is served by the direct ReAct worker",
            )
        # 处理搜索请求
        if (
            (is_paper_request or explicit_semantic_scholar)
            and asks_search
            and not needs_synthesis
        ):
            return ControllerDecision(
                intent="literature_search",
                execution_route="direct_tool",
                research_topic=research_topic,
                selected_tools=[],
                selected_tool_args={},
                requested_count=requested_count,
                confidence=0.9,
                reasoning="Source lookup does not require evidence synthesis",
            )

        return ControllerDecision(
            intent="deep_research",
            execution_route="full_research",
            research_topic=user_request.strip(),
            selected_tools=[],
            requested_count=max_sources,
            confidence=0.85,
            reasoning="Request requires planning, evidence extraction, synthesis, or evaluation",
        )

def _requested_count(request: str, max_sources: int) -> int:
    match = re.search(r"(?<!\d)(\d{1,2})\s*(?:篇|个|papers?|articles?)", request, re.I)
    if not match:
        return max_sources
    # 用户显式给出的数量优先；max_sources 只作为未写数量时的默认值。
    return max(1, min(int(match.group(1)), 50))


def _extract_research_topic(request: str) -> str:
    cleaned = request.strip()
    cleaned = re.sub(r"^(?:请|麻烦)?(?:帮我|给我)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(
        r"^(?:用|通过|use|via)?\s*(?:semantic\s*scholar|semanticscholar|语义学者)\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(?:推荐|查找|搜索|检索|列出|recommend|find|search|list)\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^\d{1,2}\s*(?:篇|个|papers?|articles?)?\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(?:相关的?|关于|on|about)\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(
        r"\s*(?:相关的?)?(?:的)?(?:论文|文献|papers?|articles?)\s*$",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。.!？?")
    return cleaned or request.strip()


_CONTROLLER_SYSTEM = """You are the intent Controller of an academic research agent.
Choose the cheapest route that fully satisfies the request. You ONLY classify
intent and pick a route — you never choose specific tools.

Intents:
- paper_recommendation: recommend a bounded list of papers; no synthesis requested.
- recommend_more: continue a session recommendation and return only unseen papers.
- literature_search: find/list paper metadata; no synthesis requested.
- paper_graph_lookup: inspect one paper's details, citations, or references.
- deep_research: compare, analyze, summarize, review, evaluate, or generate a report.
- research_from_session: run full research with the current session papers as seeds.

Routes:
- direct_tool for recommendation or metadata lookup.
- full_research for any request requiring claims, comparison, evidence, or a report.

The actual tool selection is delegated to the downstream Worker's ReAct loop.
Leave selected_tool as an empty string. Output only valid JSON."""


_CONTROLLER_USER = """User request: {user_request}
Maximum sources: {max_sources}

Available search capabilities (informational only — you do not select tools):
{tool_catalog}

Return intent, execution_route, research_topic, requested_count,
confidence, and concise reasoning."""


class ControllerAgent:
    """四 Agent 架构中的 Controller；不负责任务规划和业务工具执行。"""

    # 遗留协议字段：仅供标识角色身份；工具权限已由 protocol.AgentProtocol/AgentRole 强制，
    # 生产代码并不读取这两个属性（tests/test_four_agent_architecture.py 仍断言其值）。
    role = "controller"
    allowed_tools: tuple[str, ...] = ()

    def __init__(self, intent_controller: Optional[IntentController] = None):
        # 默认自建内部意图控制器；外部可注入复用实例（runtime 中的 _controller_agent 单例）。
        self.intent_controller = intent_controller or IntentController()

    async def execute(
        self,
        user_request: str,
        *,
        request_id: str = "request",
        max_sources: int = 5,
        agent_mode: str = "llm",
        llm_only: bool = False,
        session_context: Optional[SessionContext] = None,
        memory_prompt: str = "",
        budget: Optional[ExecutionBudget] = None,
        safety_policy: Optional[SafetyPolicy | Dict[str, Any]] = None,
    ) -> ExecutionSpec:
        # 关键步骤：唯一入口，先让内部意图控制器产出路由决策（direct_tool/conversation/full_research）。
        decision = await self.intent_controller.decide(
            user_request=user_request,
            max_sources=max_sources,
            agent_mode=agent_mode,
            session_context=session_context,
            memory_prompt=memory_prompt,
        )
        # 关键步骤：把 Controller 路由翻译成 Planner 的执行复杂度（ATOMIC/CONTEXTUAL/RESEARCH）。
        execution_class = {
            "direct_tool": ExecutionClass.ATOMIC,
            "conversation": ExecutionClass.CONTEXTUAL,
            "full_research": ExecutionClass.RESEARCH,
        }.get(decision.execution_route, ExecutionClass.RESEARCH)
        # 关键步骤：只产出稳定资源引用（论文/报告/历史），不把正文塞给下游；
        # 具体内容由 ContextLoad Worker 按引用加载。
        resources: List[Dict[str, Any]] = []
        if session_context is not None:
            # 引用范围以决策解析出的论文/种子集为准；recommend_more 回退到会话累计论文。
            resolved = set(decision.resolved_paper_ids or decision.seed_paper_ids)
            if decision.intent == "recommend_more" and not resolved:
                resolved = {
                    str(item.get("source_id") or item.get("paper_id") or "")
                    for item in session_context.recommended_papers
                }
            resources = [{
                "resource_type": "paper_ref",
                "source_id": str(item.get("source_id") or item.get("paper_id") or ""),
                "paper_id": str(item.get("source_id") or item.get("paper_id") or ""),
                "session_id": session_context.session_id,
            } for item in session_context.recommended_papers
                if str(item.get("source_id") or item.get("paper_id") or "") in resolved]
            # 报告追问：带上已定位的章节作为 report_ref，供下游精准取用。
            if decision.intent == "report_follow_up" and session_context.active_report_id:
                resources.append({
                    "resource_type": "report_ref",
                    "source_id": f"report:{session_context.active_report_id}",
                    "report_id": session_context.active_report_id,
                    "session_id": session_context.session_id,
                    "resolved_section": decision.resolved_section,
                })
            # 有对话历史时追加 history_ref，让会话型回答可引用上下文。
            if session_context.conversation_messages:
                resources.append({
                    "resource_type": "history_ref",
                    "source_id": f"history:{session_context.session_id}",
                    "session_id": session_context.session_id,
                })
        # 关键步骤：汇总为统一的 ExecutionSpec，交由 Planner 规划 Task DAG。
        return ExecutionSpec(
            request_id=request_id,
            user_request=user_request,
            intent=decision.intent,
            execution_class=execution_class,
            execution_route=decision.execution_route,
            research_topic=decision.research_topic,
            resource_ids=list(dict.fromkeys(
                decision.resolved_paper_ids + decision.seed_paper_ids + [
                    str(resource.get("source_id") or "") for resource in resources
                    if resource.get("source_id")
                ]
            )),
            resources=resources,
            selected_tools=decision.selected_tools,
            selected_tool_args=decision.selected_tool_args,
            budget=budget or ExecutionBudget(max_workers=max(1, min(max_sources, 8))),
            safety_policy=SafetyPolicy.model_validate(safety_policy or {}),
            metadata={
                "confidence": decision.confidence,
                "classifier": decision.classifier,
                "is_follow_up": decision.is_follow_up,
                # 兼容适配器需要的字段只作为 Controller 输出元数据传递。
                "controller_decision": decision.model_dump(),
                "agent_mode": agent_mode,
                "llm_only": bool(llm_only),
            },
        )
