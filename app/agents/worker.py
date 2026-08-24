"""
app/agents/worker.py

Worker Tool Loop —— Agent MVP 的工具调用执行器。

这是 ReAct 范式在本项目的工程化落地。

和 ReAct 的关系：
- Thought：Worker 分析当前 task 和已有结果，决定调用什么工具
- Action：调用工具（search / fetch / score / extract / check）
- Observation：验证 ToolResult，记录 trace，决定下一步

和一次性 Tool Calling 的区别：
- 不是 "LLM 决定调用一个工具 → 拿到结果 → 继续"
- 而是 "按 task 计划顺序调用多个工具 → 每个 ToolResult 验证 → 累积结果 → 输出"
- 有 max_tool_calls 限制（防止无限循环）
- 有去重检测（相同 tool + 相同 args 不重复调用）
- 有 timeout（每个 tool 有延迟记录）
- 有 trace（每次调用记录到 trace 列表）
- 失败不崩溃（ToolResult.success=False 时记录 warning，继续执行）

在项目调用链中的位置：
Orchestrator → Planner → Task DAG → Worker Tool Loop → Tools
"""

import asyncio
import json
import time
from typing import Any, Dict, List

from app.tools.base import BaseTool, ToolResult
from app.tools.academic_search_tool import AcademicSearchTool
from app.tools.paper_metadata_tool import PaperMetadataTool
from app.tools.source_quality_scorer import SourceQualityScorer
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.tools.citation_check_tool import CitationCheckTool
from app.agents.planner import Task
from app.observability.lifecycle import (
    reset_execution_context,
    runtime_deadline_remaining,
    set_execution_context,
)
from app.agents.schemas import (
    AgentError,
    AgentTaskStatus,
    AgentToolCall,
    WorkerProfile,
    WorkerResult,
    WorkerStrategy,
    WorkItem,
)
from app.tools.registry import ToolRegistry


class WorkerContext:
    """Worker 执行上下文 —— 收集每次工具调用的结果和 trace。"""

    def __init__(self, task: Task):
        self.task = task
        self.trace: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}       # tool_name → ToolResult.data
        self.warnings: List[str] = []
        self.tool_call_count = 0

    def add_trace(self, tool_name: str, input_summary: str, result: ToolResult):
        self.tool_call_count += 1
        # 记录每次工具调用的 trace
        entry = {
            "step": self.tool_call_count,
            "task_id": self.task.task_id,
            "tool_name": tool_name,
            "operation_name": result.tool_name or tool_name,
            "input_summary": input_summary,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "error": result.error if not result.success else None,
        }
        if result.metadata:
            entry["metadata"] = result.metadata
            if tool_name.startswith("retrieval_"):
                # Retrieval events are bounded structured trace payloads rather
                # than a second state/logging system. Keep their contract fields
                # directly addressable for tests, SSE, and run history.
                for key, value in result.metadata.items():
                    if key == "tool_name":
                        entry["retrieval_tool_name"] = value
                    else:
                        entry[key] = value
        self.trace.append(entry)
        if not result.success:
            self.warnings.append(f"[{tool_name}] {result.error}")

    def add_result(self, key: str, data: Any):
        self.results[key] = data

    def record_provider_fallback(self, data: Any):
        """Surface a real-provider fallback without counting it as a tool call."""
        if not isinstance(data, dict) or not data.get("fallback_used"):
            return
        reason = str(data.get("fallback_reason") or "unknown")
        warning = f"Academic search provider fallback used: mock ({reason})"
        if warning not in self.warnings:
            self.warnings.append(warning)
        self.trace.append({
            "step": self.tool_call_count,
            "task_id": self.task.task_id,
            "tool_name": "provider_fallback",
            "input_summary": reason,
            "success": True,
            "latency_ms": 0,
            "error": None,
            "provider": data.get("provider", "mock"),
            "fallback_reason": reason,
        })


class Worker:
    """
    Worker Tool Loop —— 按 Task DAG 中的任务执行工具调用。

    核心设计：
    1. 每类 task 有预设的工具调用顺序（tool_plan）
    2. 工具调用之间可以传递结果（上一个 tool 的输出作为下一个 tool 的输入）
    3. 每次调用记录 trace
    4. 失败时记录 warning，不中断整个 task

    规则 Worker 按 task_type 选择工具并拼装结果。
    LLM Worker 可在受控预算内动态选择工具。
    """

    MAX_TOOL_CALLS = 20  # 全局最大工具调用次数

    def __init__(self):
        # 初始化所有可用工具
        self._tools: Dict[str, BaseTool] = {
            "academic_search": AcademicSearchTool(),
            "paper_metadata": PaperMetadataTool(),
            "source_quality_scorer": SourceQualityScorer(),
            "evidence_extract": EvidenceExtractTool(),
            "citation_check": CitationCheckTool(),
        }

    async def execute_task(
        self,
        task: Task,
        dependency_results: Dict[str, Any] = None,
    ) -> WorkerContext:
        """
        执行单个 task。

        参数：
        - task: 要执行的 Task
        - dependency_results: 前置任务的执行结果 {task_id: WorkerContext}

        返回：WorkerContext（包含 trace 和累积结果）
        """
        ctx = WorkerContext(task)
        deps = dependency_results or {}

        if ctx.tool_call_count >= self.MAX_TOOL_CALLS:
            ctx.warnings.append(f"Max tool calls ({self.MAX_TOOL_CALLS}) reached before task {task.task_id}")
            return ctx

        token = set_execution_context(task_id=task.task_id, task_type=task.task_type)
        try:
            # 按 task_type 分发, 执行对应的逻辑
            if task.task_type == "search":
                await self._execute_search(task, ctx)
            elif task.task_type == "read":
                await self._execute_read(task, ctx, deps)
            elif task.task_type == "analyze":
                await self._execute_analyze(task, ctx, deps)
            elif task.task_type == "cite":
                await self._execute_cite(task, ctx, deps)
            else:
                ctx.warnings.append(f"Unknown task type: {task.task_type}")
        finally:
            reset_execution_context(token)

        return ctx

    # ---- 各 task 类型的执行逻辑 ----

    async def _execute_search(self, task: Task, ctx: WorkerContext):
        """执行搜索任务。"""
        # 从 task description 中提取 query（简化：直接用 description 中的 topic）
        topic = task.description.replace("Search for academic sources on: ", "").strip()
        planned = list(task.tool_plan or ["academic_search"])
        tool_name = next(
            (name for name in planned if ToolRegistry.get_instance().get(name) is not None),
            "",
        )
        if not tool_name:
            raise ValueError("Search Worker 没有已授权且可用的工具")
        tool = ToolRegistry.get_instance().get(tool_name)
        properties = dict((tool.input_schema or {}).get("properties") or {})
        args: Dict[str, Any] = {}
        if "query" in properties:
            args["query"] = topic
        elif "topic" in properties:
            args["topic"] = topic
        for key in ("max_results", "limit", "top_k"):
            if key in properties:
                args[key] = 7
                break
        result = await tool.run(**args)
        # 记录搜索任务的 trace
        ctx.add_trace(tool.name, f"query={topic[:50]}", result)

        if result.success:
            search_data = result.data or {}
            ctx.record_provider_fallback(search_data)
            ctx.add_result("search_results", search_data.get("results", []))
            ctx.add_result("sources", search_data.get("results", []))
        else:
            ctx.add_result("search_results", [])
            ctx.add_result("sources", [])

    async def _execute_read(self, task: Task, ctx: WorkerContext, deps: Dict[str, Any]):
        """执行阅读任务：元数据标准化 + 质量评分。"""
        # 获取搜索阶段的 sources
        sources = self._get_sources_from_deps(deps)

        if not sources:
            ctx.warnings.append("No sources to process in read task")
            ctx.add_result("scored_sources", [])
            return

        # Step 1: 元数据标准化
        self._require_planned_tool(task, "paper_metadata")
        meta_tool = self._tools["paper_metadata"]
        meta_result = await meta_tool.run(sources=sources)
        ctx.add_trace(meta_tool.name, f"sources_count={len(sources)}", meta_result)

        if meta_result.success:
            normalized = meta_result.data.get("sources", sources)
        else:
            normalized = sources

        # Step 2: 来源质量评分
        topic = task.description.replace("Fetch metadata, score quality, and normalize ", "")\
                                 .replace(" sources", "").strip()
        # topic 可能包含 max_sources 数字，从 task 描述中提取真正的话题：
        # 简化处理：从 deps 中推断 topic
        self._require_planned_tool(task, "source_quality_scorer")
        scorer_tool = self._tools["source_quality_scorer"]

        # 从搜索任务的 description 推断 topic
        search_task_desc = ""
        for tid, dep_ctx in deps.items():
            if hasattr(dep_ctx, 'task') and dep_ctx.task.task_type == "search":
                search_task_desc = dep_ctx.task.description
                break

        topic_for_scoring = self._extract_topic(search_task_desc, task.description)

        score_result = await scorer_tool.run(sources=normalized, topic=topic_for_scoring)
        ctx.add_trace(scorer_tool.name, f"sources_count={len(normalized)}, topic={topic_for_scoring[:50]}", score_result)

        # 将评分合并到 sources
        if score_result.success:
            scores_by_id = score_result.data.get("scores_by_id", {})
            for source in normalized:
                sid = source.get("source_id", "")
                if sid in scores_by_id:
                    source["quality_score"] = scores_by_id[sid]["total"]
                else:
                    source["quality_score"] = 0.0
            ctx.add_result("scored_sources", normalized)
            ctx.add_result("scores_by_id", scores_by_id)
        else:
            for s in normalized:
                s.setdefault("quality_score", 0.0)
            ctx.add_result("scored_sources", normalized)

    async def _execute_analyze(self, task: Task, ctx: WorkerContext, deps: Dict[str, Any]):
        """执行分析任务：为每个 source 抽取证据。"""
        sources = self._get_scored_sources_from_deps(deps)

        if not sources:
            ctx.warnings.append("No scored sources to analyze")
            ctx.add_result("evidence_cards", [])
            return

        self._require_planned_tool(task, "evidence_extract")
        extract_tool = self._tools["evidence_extract"]
        all_cards = []

        for source in sources:
            if ctx.tool_call_count >= self.MAX_TOOL_CALLS:
                break

            # 抽取证据
            topic = task.description.partition(":")[2].strip() if ":" in task.description else ""
            result = await extract_tool.run(source=source, topic=topic)
            ctx.add_trace(extract_tool.name, f"source_id={source.get('source_id', '?')}", result)

            if result.success:
                cards = result.data.get("evidence_cards", [])
                all_cards.extend(cards)

        ctx.add_result("evidence_cards", all_cards)

    async def _execute_cite(self, task: Task, ctx: WorkerContext, deps: Dict[str, Any]):
        """执行引用校验任务。"""
        sources = self._get_scored_sources_from_deps(deps) # 从阅读任务中获取评分后的 sources(即原文章)
        evidence_cards = self._get_evidence_cards_from_deps(deps) # 引用片段，从分析任务中获取，每个 source 对应多个引用片段

        if not sources:
            ctx.warnings.append("No sources available for citation check")
            ctx.add_result("citation_check_results", [])
            return

        # 将 evidence_cards 转为 citation 格式
        citations = []
        for i, card in enumerate(evidence_cards):
            citations.append({
                "id": i + 1,
                "source_id": card.get("source_id", ""),
                "url": card.get("url", ""),
                "quote": card.get("quote", ""),
            })

        self._require_planned_tool(task, "citation_check")
        check_tool = self._tools["citation_check"]
        result = await check_tool.run(citations=citations, sources=sources)
        ctx.add_trace(check_tool.name, f"citation_count={len(citations)}, source_count={len(sources)}", result)

        if result.success:
            ctx.add_result("citation_check_results", result.data.get("results", []))
            ctx.add_result("citation_summary", {
                "total_checked": result.data.get("total_checked", 0),
                "valid_count": result.data.get("valid_count", 0),
                "invalid_count": result.data.get("invalid_count", 0),
                "all_valid": result.data.get("all_valid", False),
            })
        else:
            ctx.add_result("citation_check_results", [])
            ctx.add_result("citation_summary", {})

    # ---- 辅助方法 ----

    @staticmethod
    def _require_planned_tool(task: Task, tool_name: str) -> None:
        """有显式 tool_plan 时禁止旧适配器偷偷补工具。"""
        if task.tool_plan and tool_name not in set(task.tool_plan):
            raise ValueError(f"任务未授权工具 {tool_name}")

    def _get_sources_from_deps(self, deps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从前置任务结果中提取 sources。"""
        for dep_ctx in deps.values():
            if hasattr(dep_ctx, 'results'):
                sources = dep_ctx.results.get("sources", [])
                if sources:
                    return sources
                sources = dep_ctx.results.get("search_results", [])
                if sources:
                    return sources
        return []

    def _get_scored_sources_from_deps(self, deps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从前置任务结果中提取 scored_sources。"""
        for dep_ctx in deps.values():
            if hasattr(dep_ctx, 'results'):
                sources = dep_ctx.results.get("scored_sources", [])
                if sources:
                    return sources
        return self._get_sources_from_deps(deps)

    def _get_evidence_cards_from_deps(self, deps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从前置任务结果中提取 evidence_cards。"""
        for dep_ctx in deps.values():
            if hasattr(dep_ctx, 'results'):
                cards = dep_ctx.results.get("evidence_cards", [])
                if cards:
                    return cards
        return []

    def _extract_topic(self, search_desc: str, read_desc: str) -> str:
        """从 task description 中推断 topic。"""
        for desc in [search_desc, read_desc]:
            if not desc:
                continue
            if "Search for academic sources on: " in desc:
                return desc.replace("Search for academic sources on: ", "").strip()
            if "on: " in desc:
                return desc.split("on: ", 1)[1].strip()
        return "unknown topic"


PROFILE_TOOL_ALLOWLIST: Dict[WorkerProfile, frozenset[str]] = {
    # direct 与 search 共用同一检索目录授权；direct 的策略（REACT / DETERMINISTIC）
    # 决定是 LLM 从目录自主选工具，还是 Planner 指定的单工具。
    WorkerProfile.DIRECT: frozenset({
        "academic_search", "local_paper_search", "semantic_scholar_search",
        "semantic_scholar_graph", "semantic_scholar_recommendations",
    }),
    WorkerProfile.CONTEXT_LOAD: frozenset(),
    WorkerProfile.SEARCH: frozenset({
        "academic_search", "local_paper_search", "semantic_scholar_search",
        "semantic_scholar_graph", "semantic_scholar_recommendations",
    }),
    WorkerProfile.READ: frozenset({"paper_metadata", "source_quality_scorer"}),
    WorkerProfile.METADATA: frozenset({"paper_metadata", "source_quality_scorer"}),
    WorkerProfile.ANALYZE: frozenset({"evidence_extract"}),
    WorkerProfile.CITE: frozenset({"citation_check"}),
    WorkerProfile.ANSWER: frozenset(),
    WorkerProfile.WRITE: frozenset(),
}


class WorkerAgent:
    """统一、单任务、隔离的 Worker 执行器。"""

    role = "worker"

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry.get_instance()
        self.messages: List[Dict[str, str]] = []
        self._called: set[str] = set()

    @staticmethod
    def strategy_for(profile: WorkerProfile) -> WorkerStrategy:
        """按 Profile 决定执行策略：确定性 / 自主检索循环 / 文本综合。"""
        # 确定性链路：固定工具顺序、不进入 LLM 循环（DIRECT/READ/METADATA/CITE）。
        if profile in {
            WorkerProfile.DIRECT, WorkerProfile.READ,
            WorkerProfile.METADATA, WorkerProfile.CITE,
        }:
            return WorkerStrategy.DETERMINISTIC
        # 检索与证据抽取需要观察结果后继续决策，走受控 ReAct 循环。
        if profile in {WorkerProfile.SEARCH, WorkerProfile.ANALYZE}:
            return WorkerStrategy.REACT
        # 其余（ANSWER/WRITE/CONTEXT_LOAD）是生成型任务，直接综合产出文本。
        return WorkerStrategy.SYNTHESIS

    def allowed_tools(self, item: WorkItem) -> List[str]:
        # 关键步骤：安全策略 + Profile 白名单双重校验，fail-closed——越权工具直接抛错。
        requested = list(item.allowed_tools)
        policy = item.safety_policy.model_dump()
        if policy.get("allow_business_tools") is False and requested:
            raise ValueError("安全策略禁止业务工具调用")
        denied_by_policy = set(policy.get("denied_tools") or [])
        if denied_by_policy.intersection(requested):
            raise ValueError(f"安全策略禁止工具: {sorted(denied_by_policy.intersection(requested))}")
        for name in requested:
            capability = self.registry.get_capability(name)
            if capability is None:
                continue
            if capability.network_access and policy.get("allow_network") is False:
                raise ValueError(f"安全策略禁止网络工具: {name}")
            if capability.external_write and policy.get("allow_external_writes") is False:
                raise ValueError(f"安全策略禁止外部写入工具: {name}")
            if capability.destructive and policy.get("allow_destructive_actions") is False:
                raise ValueError(f"安全策略禁止破坏性工具: {name}")
            if (
                capability.resource_scope == "explicit"
                and policy.get("require_explicit_resources", False)
                and not item.resources
            ):
                raise ValueError(f"工具 {name} 要求 WorkItem 显式资源")
        allowed = PROFILE_TOOL_ALLOWLIST[item.profile]
        if item.profile in {WorkerProfile.SEARCH, WorkerProfile.DIRECT}:
            # 关键步骤：MCP 公共名先映射到 canonical 检索能力，再执行
            # Profile 授权；Planner 与 Worker 共用 Registry 的同一判定。
            denied = [
                name for name in requested
                if self.registry.retrieval_capability(name) not in allowed
            ]
        else:
            denied = [name for name in requested if name not in allowed]
        if denied:
            raise ValueError(f"WorkerProfile {item.profile.value} 无权调用: {denied}")
        unknown = [name for name in requested if self.registry.get(name) is None]
        if unknown:
            raise ValueError(f"Worker 工具未注册: {unknown}")
        return requested

    async def execute(self, item: WorkItem) -> WorkerResult:
        """执行一个 WorkItem；不读取任务信封外的父 State。"""
        started = time.perf_counter()
        # 关键步骤：先查运行时全局预算/期限，已过期就就地失败，不进入执行。
        runtime_budget_id = str(item.metadata.get("runtime_budget_id") or "")
        deadline_ok, deadline_reason, runtime_remaining = runtime_deadline_remaining(
            runtime_budget_id
        )
        if not deadline_ok:
            return self._failure(
                item, "RUNTIME_TIMEOUT", deadline_reason, started, timeout=True,
            )
        self.messages = []
        self._called = set()
        strategy = item.strategy or self.strategy_for(item.profile)
        if item.profile == WorkerProfile.CITE and strategy != WorkerStrategy.DETERMINISTIC:
            raise ValueError("Citation Worker 必须使用 deterministic 策略")
        explicit_resource_ids = [
            str(resource.get("source_id") or resource.get("paper_id") or "")
            for resource in item.resources
            if resource.get("source_id") or resource.get("paper_id")
        ]
        local_budget = {"remaining": item.max_tool_calls}
        # 关键步骤：注入执行上下文（角色/白名单/预算/截止时间），结束 finally 时统一回收。
        permission_token = set_execution_context(
            agent_role="worker",
            worker_profile=item.profile.value,
            allowed_tools=list(item.allowed_tools),
            safety_policy=item.safety_policy.model_dump(),
            explicit_resource_ids=explicit_resource_ids,
            local_tool_budget=local_budget,
            runtime_budget_id=runtime_budget_id,
            per_tool_timeout_ms=item.per_tool_timeout_ms,
            item_deadline=time.monotonic() + item.timeout_ms / 1000.0,
        )
        try:
            # 关键步骤：授权 fail-closed——工具型 Worker 未获授权直接拒绝，不进入执行。
            allowed = self.allowed_tools(item)
            if (
                item.profile in {
                    WorkerProfile.DIRECT, WorkerProfile.SEARCH, WorkerProfile.READ,
                    WorkerProfile.METADATA, WorkerProfile.ANALYZE, WorkerProfile.CITE,
                }
                and not allowed
            ):
                raise ValueError("工具型 Worker 的 allowed_tools 为空，按 fail-closed 拒绝执行")
            execution_timeout = min(
                item.timeout_ms / 1000,
                runtime_remaining if runtime_remaining is not None else item.timeout_ms / 1000,
            )
            # 关键步骤：带全局超时的有界执行，产出真实调用轨迹。
            output, calls, rationales, iterations = await asyncio.wait_for(
                self._execute_bounded(item, strategy, allowed),
                timeout=execution_timeout,
            )
            # 关键步骤：防适配器隐藏调用——实际调用的工具必须在授权白名单内，否则整体失败。
            actual = [call.tool_name for call in calls]
            unauthorized = [name for name in actual if name not in set(allowed)]
            if unauthorized:
                raise ValueError(f"Worker 适配器发生未授权隐藏工具调用: {unauthorized}")
            if len(calls) > item.max_tool_calls:
                raise ValueError("Worker 实际工具调用超过 WorkItem 预算")
            policy_error = str(output.pop("_policy_error", "") or "")
            tool_error = str(output.pop("_tool_error", "") or "")
            if policy_error or tool_error:
                # 关键步骤：失败结果保留此前真实执行的完整调用轨迹；被拒绝的调用不伪装成已执行。
                return WorkerResult(
                    task_id=item.task_id,
                    profile=item.profile,
                    status=AgentTaskStatus.FAILED,
                    output_data=output,
                    tool_calls=calls,
                    needs_replan=True,
                    error=AgentError(
                        error_code="TOOL_POLICY_DENIED" if policy_error else "TOOL_CALL_FAILED",
                        message=policy_error or tool_error,
                        recoverable=True,
                    ),
                    action_rationale=rationales,
                    iterations=iterations,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            failed_calls = [call for call in calls if not call.success]
            if strategy == WorkerStrategy.DETERMINISTIC and failed_calls:
                message = failed_calls[0].error or f"{failed_calls[0].tool_name} 执行失败"
                timed_out = "timeout" in message.lower() or "timed out" in message.lower()
                # 关键步骤：确定性链路任一必需调用失败即整体失败，不能用残留输出伪装成功。
                return WorkerResult(
                    task_id=item.task_id,
                    profile=item.profile,
                    status=AgentTaskStatus.TIMEOUT if timed_out else AgentTaskStatus.FAILED,
                    output_data=output,
                    tool_calls=calls,
                    needs_replan=True,
                    error=AgentError(
                        error_code="TOOL_TIMEOUT" if timed_out else "TOOL_CALL_FAILED",
                        message=message,
                        recoverable=True,
                    ),
                    action_rationale=rationales,
                    iterations=iterations,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            mode = str(output.get("mode") or "")
            if (
                item.input_data.get("llm_only")
                and item.profile in {
                    WorkerProfile.SEARCH, WorkerProfile.ANALYZE,
                    WorkerProfile.WRITE, WorkerProfile.DIRECT,
                }
                and mode != "llm"
            ):
                raise RuntimeError(f"LLM-only Worker 禁止 {mode} 模式降级")
            # 关键步骤：按 Profile 判定完成门，不满足则 needs_replan 交由 Reviewer 处理。
            needs_replan = not self._completion_gate(item, output)
            if (
                item.profile == WorkerProfile.DIRECT
                and needs_replan and calls and all(call.success for call in calls)
            ):
                return WorkerResult(
                    task_id=item.task_id,
                    profile=item.profile,
                    status=AgentTaskStatus.FAILED,
                    output_data=output,
                    tool_calls=calls,
                    error=AgentError(
                        error_code="DIRECT_NO_RESULTS",
                        message="直接能力成功执行但没有新的可返回结果",
                        recoverable=False,
                    ),
                    action_rationale=rationales,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            status = AgentTaskStatus.PARTIAL_SUCCESS if needs_replan else AgentTaskStatus.SUCCESS
            return WorkerResult(
                task_id=item.task_id,
                profile=item.profile,
                status=status,
                output_data=output,
                tool_calls=calls,
                needs_replan=needs_replan,
                action_rationale=rationales,
                iterations=iterations,
                latency_ms=int((time.perf_counter() - started) * 1000),
                metadata={"mode": mode} if mode else {},
            )
        except asyncio.TimeoutError:
            return self._failure(item, "TIMEOUT", "Worker 总超时", started, timeout=True)
        except Exception as exc:
            return self._failure(item, "WORKER_FAILED", str(exc), started)
        finally:
            reset_execution_context(permission_token)

    async def _execute_bounded(
        self,
        item: WorkItem,
        strategy: WorkerStrategy,
        allowed: List[str],
    ) -> tuple[Dict[str, Any], List[AgentToolCall], List[str], int]:
        # 关键步骤：只记录简短行动依据，不保存或返回内部思维链。
        rationales = [f"observe: profile={item.profile.value}, resources={len(item.resources)}"]
        # 关键步骤：按 Profile 分派——CONTEXT_LOAD 走资源适配器、SYNTHESIS 走新综合路径、
        # DIRECT 单工具直连；其余检索/证据/引用 Profile 统一回落到 _run_legacy_adapter（旧 Worker）。
        if item.profile == WorkerProfile.CONTEXT_LOAD:
            from app.agents.context_loader import context_resource_adapter

            # 关键步骤：这里只解析信封中的稳定引用，实际正文由只读基础设施适配器裁剪加载。
            resources = context_resource_adapter.load(list(item.resources))
            rationales.append("validate: 仅按 WorkItem 引用加载裁剪资源，不读取父 State")
            return {
                "resources": resources,
                "resource_ids": list(item.input_data.get("resource_ids") or []),
                "mode": "deterministic",
            }, [], rationales, 1
        if strategy == WorkerStrategy.SYNTHESIS:
            rationales.append("decide: 使用隔离资源进行文本综合")
            return await self._synthesize(item), [], rationales, 1
        if item.profile == WorkerProfile.DIRECT:
            rationales.append("validate: 直接任务工具授权已校验")
            # 关键步骤：direct 按策略分流——REACT 走 LLM 自主检索循环（推荐/搜索/图谱），
            # DETERMINISTIC 走规则单发（Planner 已指定唯一工具）。
            if strategy == WorkerStrategy.REACT:
                return await self._run_direct_react(item, allowed, rationales)
            return await self._run_direct(item, allowed, rationales)
        return await self._run_legacy_adapter(item, strategy, rationales)

    async def _run_direct(
        self, item: WorkItem, allowed: List[str], rationales: List[str],
    ) -> tuple[Dict[str, Any], List[AgentToolCall], List[str], int]:
        # 关键步骤：DIRECT 规则模式只调 Planner 指定的一个工具；
        # 同工具+同参数视为重复动作并拒绝，防死循环。
        if not allowed:
            return {"answer": item.instruction, "resources": item.resources,
                    "mode": "rule"}, [], rationales, 1
        tool_name = allowed[0]
        tool = self.registry.get(tool_name)
        properties = dict((tool.input_schema or {}).get("properties") or {})
        args = {key: value for key, value in item.input_data.items() if key in properties}
        # 查询参数兜底：优先用 Planner 清理后的 topic，其次退到完整用户请求。
        fallback_topic = str(item.input_data.get("topic") or item.instruction)
        if "query" in properties and not args.get("query"):
            args["query"] = fallback_topic
        elif "topic" in properties and not args.get("topic"):
            args["topic"] = fallback_topic
        requested_count = int(item.input_data.get("_requested_count", 5) or 5)
        for key in ("limit", "max_results", "top_k"):
            if key in properties and key not in args:
                args[key] = requested_count
                break
        signature = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        if signature in self._called:
            raise ValueError("检测到重复工具动作")
        self._called.add(signature)
        rationales.append(f"act: 调用 {tool_name}")
        # BaseTool 是统一的单工具超时、权限及预算入口，Direct 不再叠加第二层超时。
        result = await tool.run(**args)
        call = AgentToolCall(
            tool_name=tool_name, success=result.success,
            latency_ms=result.latency_ms, error=result.error or "",
        )
        rationales.append(f"observe: {tool_name} {'成功' if result.success else '失败'}")
        if not result.success:
            return {
                "_tool_error": result.error or f"{tool_name} 执行失败",
            }, [call], rationales, 1
        output = dict(result.data or {})
        output["mode"] = "rule"
        output = self._finalize_direct_output(item, output)
        return output, [call], rationales, 1

    async def _run_direct_react(
        self, item: WorkItem, allowed: List[str], rationales: List[str],
    ) -> tuple[Dict[str, Any], List[AgentToolCall], List[str], int]:
        # 关键步骤：direct 的 ReAct 执行——Controller 不再选工具，LLMWorker 的
        # 受控检索循环从目录自主挑选，并把会话资源注入依赖供其感知已推荐论文。
        if not allowed:
            return {"answer": item.instruction, "resources": item.resources,
                    "mode": "llm"}, [], rationales, 1
        # 复用 LLMWorker 的 search 通道：Task 携带完整用户请求与目录授权。
        task = Task(item.task_id, "search", item.instruction, item.depends_on, list(allowed))
        deps = self._direct_dependency_results(item)
        from app.agents.llm_worker import LLMWorker, LLMWorkerConfig

        legacy = LLMWorker(LLMWorkerConfig(
            max_tool_calls=item.max_tool_calls,
            tool_timeout_ms=item.per_tool_timeout_ms,
            worker_timeout_ms=item.timeout_ms,
            max_iterations=item.max_iterations,
            # 关键步骤：把用户请求的篇数透传给 ReAct 循环——提示 LLM 目标篇数、
            # 并把工具 limit clamp 到剩余需求（源头约束，兜底裁剪在 _finalize_direct_output）。
            requested_count=int(item.input_data.get("_requested_count", 0) or 0) or None,
        ))
        ctx = await legacy.execute_task(
            task, deps, topic=str(item.input_data.get("topic") or item.instruction),
        )
        calls, denied, internal = self._trace_to_tool_calls(ctx)
        if denied:
            ctx.results["_policy_error"] = str(
                denied[0].get("error") or "工具调用被权限或预算门拒绝"
            )
        if internal:
            ctx.results["_internal_trace"] = internal
        output = dict(ctx.results)
        fallback_used = any(
            str(entry.get("tool_name") or "") == "tool_loop_fallback"
            for entry in internal
        )
        output["mode"] = "rule" if fallback_used else "llm"
        # 统一收口：与规则单发一样做会话去重 + markdown 答案格式化。
        output = self._finalize_direct_output(item, output)
        rationales.append(
            f"observe: direct ReAct 产出 {len(output.get('sources', []))} 篇新论文"
        )
        return output, calls, rationales, min(item.max_iterations, max(1, len(calls)))

    def _direct_dependency_results(self, item: WorkItem) -> Dict[str, WorkerContext]:
        """把会话资源作为 search 依赖注入，供 LLM 感知已推荐论文（recommend_more 去重）。"""
        deps: Dict[str, WorkerContext] = {}
        if not item.resources:
            return deps
        source_ctx = WorkerContext(
            Task("search", "search", str(item.input_data.get("topic") or ""))
        )
        source_ctx.add_result("sources", list(item.resources))
        source_ctx.add_result("search_results", list(item.resources))
        deps["search"] = source_ctx
        return deps

    def _trace_to_tool_calls(
        self, ctx: WorkerContext,
    ) -> tuple[List[AgentToolCall], List[dict], List[dict]]:
        """从 LLMWorker 的 trace 重建可观测的工具调用列表。

        返回 (calls, denied_attempts, internal_events)：
        - calls           → 真实执行且工具已注册的调用
        - denied_attempts → 被权限/预算门拒绝的调用（不能伪装成已执行）
        - internal_events → 内部流程事件（改写/换源/回退/完成）
        """
        denied_attempts = [
            entry for entry in ctx.trace
            if isinstance(entry.get("metadata"), dict)
            and entry["metadata"].get("executed") is False
        ]
        calls = [
            AgentToolCall(
                tool_name=str(entry.get("operation_name") or entry.get("tool_name")),
                success=bool(entry.get("success", False)),
                latency_ms=max(0, int(entry.get("latency_ms", 0) or 0)),
                error=str(entry.get("error") or ""),
            )
            for entry in ctx.trace
            if self.registry.get(str(entry.get("operation_name") or entry.get("tool_name"))) is not None
            and not (
                isinstance(entry.get("metadata"), dict)
                and entry["metadata"].get("executed") is False
            )
        ]
        internal_events = [
            dict(entry) for entry in ctx.trace
            if str(entry.get("tool_name") or "") in {
                "tool_loop_fallback", "retrieval_query_rewritten",
                "retrieval_source_switched", "retrieval_finished",
                "llm_failed", "llm_finished",
            }
        ]
        return calls, denied_attempts, internal_events

    @staticmethod
    def _finalize_direct_output(item: WorkItem, output: Dict[str, Any]) -> Dict[str, Any]:
        """direct 结果的统一收口：剔除已见论文 + 生成助手口吻的自然语言回复。

        DIRECT 路由无论 deterministic 单发还是 ReAct 循环都必须经过这里，
        保证合并节点（node_merge_direct_results）拿到一致的 sources/answer 结构。

        回答策略：answer 只承载助手的开场与收尾，不重复回显用户问题，也不把
        论文列表写进 markdown——条目交给前端来源卡片按编号渲染。这样对话里既有
        "助手在发言"的部分，又保留可点击、可翻页的结构化论文列表，避免两套样式并存。
        """
        sources = output.get("sources") or output.get("results") or []
        if not isinstance(sources, list):
            return output
        # 去重：剔除 item.resources 中已在会话出现的论文（recommend_more 语义）。
        excluded = {
            identity
            for resource in item.resources
            for identity in WorkerAgent._resource_identities(resource)
        }
        sources = [
            source for source in sources
            if not WorkerAgent._resource_identities(source).intersection(excluded)
        ]
        # 篇数兜底：把交付的来源收敛到用户请求的数量。
        # 源头 clamp（工具 limit）只是提示性约束，工具仍可能返回超量来源，
        # 这里对去重后的最终列表再裁一刀，保证本轮交付数不超 requested_count。
        requested_count = int(item.input_data.get("_requested_count", 0) or 0)
        if requested_count > 0 and len(sources) > requested_count:
            sources = sources[:requested_count]
        instruction = str(item.instruction or "你的请求").strip()
        if sources:
            # 开场只交代整理结果，编号列表由前端卡片呈现，不在这里重复。
            lead = (
                f"针对您的请求《{instruction}》，为您整理出 {len(sources)} 篇相关论文，"
                f"详情见下方列表。"
            )
            closing = "每篇论文都标注了作者、年份与发表渠道，点击标题可查看原文。"
            answer = f"{lead}\n\n{closing}"
        else:
            answer = (
                f"针对您的请求《{instruction}》，暂时没有检索到直接匹配的论文。"
                f"可以补充更具体的关键词、论文 ID 或 DOI 后重新检索。"
            )
        output["sources"] = sources
        output["answer"] = answer
        return output

    async def _run_legacy_adapter(
        self, item: WorkItem, strategy: WorkerStrategy, rationales: List[str],
    ) -> tuple[Dict[str, Any], List[AgentToolCall], List[str], int]:
        # 关键步骤：SEARCH/READ/ANALYZE/CITE 复用旧 Worker/LLMWorker（方法名即 legacy）——
        # 把 WorkItem 折算回 Task + deps，同时施加当前 WorkItem 的工具预算与权限。
        task_type = {
            WorkerProfile.SEARCH: "search",
            WorkerProfile.READ: "read",
            WorkerProfile.METADATA: "read",
            WorkerProfile.ANALYZE: "analyze",
            WorkerProfile.CITE: "cite",
        }[item.profile]
        description = item.instruction
        if task_type == "search":
            description = f"Search for academic sources on: {item.input_data.get('query', item.instruction)}"
        task = Task(item.task_id, task_type, description, item.depends_on, item.allowed_tools)
        deps: Dict[str, WorkerContext] = {}
        if item.resources:
            source_ctx = WorkerContext(Task("search", "search", item.input_data.get("topic", "")))
            source_ctx.add_result("sources", list(item.resources))
            source_ctx.add_result("search_results", list(item.resources))
            deps["search"] = source_ctx
            read_ctx = WorkerContext(Task("read", "read", ""))
            read_ctx.add_result("scored_sources", list(item.resources))
            deps["read"] = read_ctx
        cards = list(item.input_data.get("evidence_cards") or [])
        if cards:
            analyze_ctx = WorkerContext(Task("analyze", "analyze", ""))
            analyze_ctx.add_result("evidence_cards", cards)
            deps["analyze"] = analyze_ctx
        rationales.extend([
            f"decide: 使用 {strategy.value} 有界执行",
            "validate: 完成权限、重复与预算检查",
        ])
        if strategy == WorkerStrategy.REACT and item.input_data.get("agent_mode") == "llm":
            # 关键步骤：ReAct 路径复用现有受控循环，支持查询改写、来源切换和证据补充。
            from app.agents.llm_worker import LLMWorker, LLMWorkerConfig

            legacy = LLMWorker(LLMWorkerConfig(
                max_tool_calls=item.max_tool_calls,
                tool_timeout_ms=item.per_tool_timeout_ms,
                worker_timeout_ms=item.timeout_ms,
                max_iterations=item.max_iterations,
            ))
            ctx = await legacy.execute_task(
                task, deps, topic=str(item.input_data.get("topic") or item.instruction),
            )
        else:
            legacy = Worker()
            # 关键步骤：旧 Worker 适配器也必须服从当前 WorkItem 的工具预算。
            legacy.MAX_TOOL_CALLS = item.max_tool_calls
            ctx = await legacy.execute_task(task, deps)
        calls, denied_attempts, internal_events = self._trace_to_tool_calls(ctx)
        if denied_attempts:
            ctx.results["_policy_error"] = str(
                denied_attempts[0].get("error") or "工具调用被权限或预算门拒绝"
            )
        if internal_events:
            ctx.results["_internal_trace"] = internal_events
        fallback_used = any(
            str(entry.get("tool_name") or "") == "tool_loop_fallback"
            for entry in internal_events
        )
        if item.input_data.get("agent_mode") == "llm":
            ctx.results["mode"] = "rule" if fallback_used else "llm"
        else:
            ctx.results["mode"] = "rule"
        rationales.append(f"observe: 产出 {len(ctx.results)} 类结果")
        return dict(ctx.results), calls, rationales, min(item.max_iterations, max(1, len(calls)))

    @staticmethod
    async def _synthesize(item: WorkItem) -> Dict[str, Any]:
        # 关键步骤：ANSWER/WRITE 走独立综合路径——问答交给 ConversationAgent，
        # 章节/整稿交给 DraftReviewer/FinalReviewer（LLM-only 时用 LLMDraftReviewer）。
        if item.profile == WorkerProfile.ANSWER:
            from app.agents.conversation import (
                ConversationAgent, ConversationRequest, FollowUpContext,
            )

            context_data = dict(item.input_data.get("conversation_context") or {})
            context = FollowUpContext(
                query=item.instruction,
                papers=list(context_data.get("papers") or item.resources),
                report=context_data.get("report"),
                evidence=list(context_data.get("evidence") or []),
                history=list(context_data.get("history") or []),
                resolved_section=context_data.get("resolved_section"),
                report_id=context_data.get("report_id"),
                missing_paper_ids=list(context_data.get("missing_paper_ids") or []),
            )
            response, llm_result = await ConversationAgent().answer(
                ConversationRequest(
                    query=item.instruction,
                    context=context,
                    operation_hint=str(item.input_data.get("operation_hint") or ""),
                    language=str(item.input_data.get("language") or "zh"),
                ),
                agent_mode=str(item.input_data.get("agent_mode") or "rule"),
                memory_prompt=str(item.input_data.get("memory_prompt") or ""),
            )
            return {
                "answer": response.answer,
                "conversation_result": response.model_dump(),
                "llm_result": llm_result,
                "resources_used": response.referenced_papers,
            }

        from app.agents.draft_reviewer import DraftReviewer
        from app.agents.final_reviewer import FinalReviewer

        data = item.input_data
        if data.get("repair_payload"):
            if data.get("llm_only"):
                raise RuntimeError("LLM-only 禁止使用规则 FinalReviewer 修复正文")
            repaired = FinalReviewer().review(**dict(data["repair_payload"]))
            return {"report": repaired.get("final_report", ""), "mode": "rule", **repaired}
        if data.get("section"):
            if data.get("agent_mode") == "llm":
                from app.agents.llm_reviewer import LLMDraftReviewer

                report, generation = await LLMDraftReviewer().generate_chapter(
                    data["section"], list(data.get("evidence_cards") or []),
                    list(item.resources), str(data.get("language") or "zh"),
                    source_number=dict(data.get("source_number") or {}),
                    allow_rule_fallback=not bool(data.get("llm_only")),
                )
                mode = (
                    "llm" if generation.get("success") and not generation.get("skipped")
                    else str(generation.get("mode") or "rule")
                )
                if data.get("llm_only") and mode != "llm":
                    raise RuntimeError("LLM-only 章节未由真实 LLM 生成")
                return {
                    "report": report, "chapter": report, "section": data["section"],
                    "mode": mode, "generation": generation,
                }
            report = DraftReviewer().generate_chapter(
                data["section"], list(data.get("evidence_cards") or []),
                list(item.resources), str(data.get("language") or "zh"),
                source_number=dict(data.get("source_number") or {}),
            )
            return {
                "report": report, "chapter": report, "section": data["section"],
                "mode": "rule",
            }
        if data.get("llm_only"):
            raise RuntimeError("LLM-only 整篇写作必须拆分为显式 LLM 章节 WorkItem")
        drafted = DraftReviewer().review(
            topic=str(data.get("topic") or item.instruction),
            sources=list(item.resources),
            evidence_cards=list(data.get("evidence_cards") or []),
            citation_check_results=list(data.get("citation_check_results") or []),
            citation_summary=dict(data.get("citation_summary") or {}),
            language=str(data.get("language") or "zh"),
        )
        return {"report": drafted.get("draft_report", ""), "mode": "rule", **drafted}

    @staticmethod
    def _completion_gate(item: WorkItem, output: Dict[str, Any]) -> bool:
        if not output:
            return False
        if item.profile == WorkerProfile.SEARCH:
            return bool(output.get("sources") or output.get("search_results"))
        if item.profile == WorkerProfile.DIRECT and "sources" in output:
            return bool(output.get("sources"))
        if item.profile == WorkerProfile.ANALYZE:
            return bool(output.get("evidence_cards"))
        if item.profile == WorkerProfile.CITE:
            summary = output.get("citation_summary") or output.get("summary")
            checks = output.get("citation_check_results")
            return (
                isinstance(summary, dict)
                and {"total_checked", "valid_count", "invalid_count", "all_valid"} <= set(summary)
                and isinstance(checks, list)
            )
        if item.profile == WorkerProfile.ANSWER:
            return bool(str(output.get("answer") or "").strip())
        if item.profile == WorkerProfile.WRITE:
            return bool(str(output.get("report") or "").strip())
        return True

    @staticmethod
    def _resource_key(resource: Dict[str, Any]) -> str:
        doi = str(resource.get("doi") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        url = str(resource.get("url") or "").strip().rstrip("/").lower()
        if url:
            return f"url:{url}"
        source_id = str(resource.get("source_id") or resource.get("paper_id") or "")
        return f"id:{source_id}" if source_id else ""

    @staticmethod
    def _resource_identities(resource: Dict[str, Any]) -> set[str]:
        """同时保留 DOI/URL/稳定 ID，确保轻量引用也能与完整来源去重。"""
        identities: set[str] = set()
        doi = str(resource.get("doi") or "").strip().lower()
        if doi:
            identities.add(f"doi:{doi}")
        url = str(resource.get("url") or "").strip().rstrip("/").lower()
        if url:
            identities.add(f"url:{url}")
        for key in ("source_id", "paper_id", "semantic_scholar_id", "openalex_id"):
            value = str(resource.get(key) or "").strip()
            if value:
                identities.add(f"id:{value}")
        return identities

    @staticmethod
    def _failure(
        item: WorkItem, code: str, message: str, started: float, timeout: bool = False,
    ) -> WorkerResult:
        # 关键步骤：统一失败出口，标记可恢复并建议 Reviewer 做有界 replan。
        return WorkerResult(
            task_id=item.task_id,
            profile=item.profile,
            status=AgentTaskStatus.TIMEOUT if timeout else AgentTaskStatus.FAILED,
            error=AgentError(
                error_code=code, message=message, recoverable=True,
                suggested_action="交由 Reviewer 判断是否需要有界 replan",
            ),
            needs_replan=True,
            action_rationale=["validate: 执行未通过完成门"],
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
