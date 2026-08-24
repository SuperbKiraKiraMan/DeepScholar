"""
app/agents/planner.py

Planner —— 任务规划器。

Agent MVP 的第一步：将研究主题拆解为 Task DAG（有向无环图）。

规则 Planner 设计：
- 规则版 Planner：根据 max_sources 生成固定结构的 Task DAG
- 不需要 LLM，也可由 LLM Planner 提供动态候选

Task DAG 结构：
  search_task ──→ read_task ──→ analyze_task ──→ cite_task

每个 task 包含：
- task_id：唯一标识
- task_type：search / read / analyze / cite
- description：任务描述
- depends_on：依赖的前置 task_id 列表
- tool_plan：计划调用的工具列表

和 Plan-and-Solve 的关系：
- Planner 负责 Plan（生成 Task DAG）
- Worker Tool Loop 负责 Solve（逐步执行每个 task）
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional, Set

from app.tools.registry import ToolRegistry
from app.llm.protocol import StructuredLLMClient
from app.agents.schemas import (
    ExecutionClass,
    ExecutionSpec,
    ReviewOutcome,
    ReviewVerdict,
    WorkerProfile,
    WorkerStrategy,
    WorkItem,
    WorkPlan,
)


def _retrieval_tool_plan() -> List[str]:
    """Resolve the bounded search allowlist from the existing registry."""
    return ToolRegistry.get_instance().list_retrieval_capabilities()


def _direct_rule_plan(spec: ExecutionSpec) -> tuple[str, Dict[str, Any]]:
    """按意图确定性挑选 direct 工具并构造参数（规则模式的单发兜底）。

    返回 (tool_name, tool_args)。工具偏好顺序：带命名空间的 MCP 包装器
    （用户显式点名的 Semantic Scholar 实现）→ canonical 内置工具 → academic_search。
    该函数只在 agent_mode=rule 时被 _atomic_item 调用，不依赖 LLM。
    """
    registry = ToolRegistry.get_instance()
    catalog = registry.list_retrieval_capabilities()
    requested_count = int(
        (spec.metadata.get("controller_decision") or {}).get("requested_count", 5) or 5
    )
    topic = spec.research_topic

    def pick(suffix: str) -> str:
        """在注册目录中优先 MCP 包装器，其次内置 canonical 工具。"""
        for name in catalog:
            if name.startswith("mcp__") and registry.retrieval_capability(name) == suffix:
                return name
        for name in catalog:
            if name == suffix:
                return name
        fallback = "academic_search"
        return fallback if fallback in catalog else (catalog[0] if catalog else "")

    if spec.intent == "paper_graph_lookup":
        tool = pick("semantic_scholar_graph")
        return tool, {
            "paper_query": _extract_graph_paper_query(spec.user_request),
            "relation": _graph_relation(spec.user_request),
            "limit": requested_count,
        }
    if spec.intent in {"paper_recommendation", "recommend_more"}:
        return pick("semantic_scholar_recommendations"), {
            "topic": topic,
            "limit": requested_count,
        }
    if spec.intent == "literature_search":
        lowered = spec.user_request.lower()
        explicit_s2 = any(
            marker in lowered
            for marker in ("semantic scholar", "semanticscholar", "语义学者")
        )
        if explicit_s2:
            return pick("semantic_scholar_search"), {
                "query": topic,
                "max_results": requested_count,
            }
        return "academic_search", {"query": topic, "max_results": requested_count}
    # 未知意图：退化为通用学术搜索，保证至少能产出来源。
    return "academic_search", {"query": topic, "max_results": requested_count}


def _graph_relation(request: str) -> str:
    """从请求原文判定图谱查询的关系类型（references / citations / details）。"""
    lowered = request.lower()
    if any(
        marker in lowered
        for marker in ("参考文献", "引用了哪些", "references of", "references for")
    ):
        return "references"
    if any(
        marker in lowered
        for marker in (
            "被哪些论文引用", "被引论文", "引用它的论文", "引用文章",
            "后续引用", "引用工作", "citations of", "citing papers",
        )
    ):
        return "citations"
    return "details"


def _extract_graph_paper_query(request: str) -> str:
    """从请求原文抽取被查论文名/标识符（规则模式的确定性解析）。"""
    quoted = re.search(r"[“\"《](.+?)[”\"》]", request)
    if quoted:
        return quoted.group(1).strip()
    cleaned = re.sub(
        r"^(?:请|麻烦)?(?:帮我|给我)?(?:查看|查询|查找|找出|show|find)?\s*",
        "",
        request.strip(),
        flags=re.I,
    )
    cleaned = re.sub(
        r"(?:被哪些论文引用|的被引论文|引用它的论文|的参考文献|引用了哪些"
        r"|的引用文章|的后续引用工作有哪些|的后续引用工作|的引用工作"
        r"|的引用关系|的论文详情|的论文元数据|的元数据|的作者是谁|的论文作者"
        r"|paper details|paper metadata|authors of|citations of|citing papers"
        r"|references of|references for)",
        " ",
        cleaned,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ，,。.!？?") or request.strip()


class Task:
    """Task DAG 中的一个任务节点。"""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        description: str,
        depends_on: List[str] = None,
        tool_plan: List[str] = None,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.depends_on = depends_on or []
        self.tool_plan = tool_plan or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "depends_on": self.depends_on,
            "tool_plan": self.tool_plan,
        }

    @staticmethod
    def _safety_args(spec: ExecutionSpec) -> Dict[str, Any]:
        return {"safety_policy": spec.safety_policy.model_copy(deep=True)}


class TaskDAG:
    """Task DAG —— Planner 的输出。"""

    def __init__(self, topic: str, tasks: List[Task]):
        self.topic = topic
        self.tasks = tasks

    def get_task(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task not found: {task_id}")

    def get_ready_tasks(self, completed_ids: set) -> List[Task]:
        """返回所有依赖已满足且尚未完成的任务。"""
        ready = []
        for t in self.tasks:
            if t.task_id in completed_ids:
                continue
            if all(dep in completed_ids for dep in t.depends_on):
                ready.append(t)
        return ready

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "tasks": [t.to_dict() for t in self.tasks],
            "task_count": len(self.tasks),
        }


class Planner:
    """
    任务规划器。

    规则版。根据研究主题生成固定的 Task DAG：
    1. search  → 搜索学术来源
    2. read    → 抓取 + 评分 + 元数据
    3. analyze → 提取证据
    4. cite    → 校验引用

    LLM Planner 可根据主题动态生成 task。
    """

    def plan(self, topic: str, max_sources: int = 5) -> TaskDAG:
        """
        生成 Task DAG。

        参数：
        - topic: 研究主题
        - max_sources: 最大来源数

        返回：TaskDAG
        """
        tasks = [
            Task(
                task_id="search",
                task_type="search",
                description=f"Search for academic sources on: {topic}",
                depends_on=[],
                tool_plan=_retrieval_tool_plan(),
            ),
            Task(
                task_id="read",
                task_type="read",
                description=f"Fetch metadata, score quality, and normalize {max_sources} sources",
                depends_on=["search"],
                tool_plan=["paper_metadata", "source_quality_scorer"],
            ),
            Task(
                task_id="analyze",
                task_type="analyze",
                description="Extract evidence cards from source full_text",
                depends_on=["read"],
                tool_plan=["evidence_extract"],
            ),
            Task(
                task_id="cite",
                task_type="cite",
                description="Check citation validity against source list",
                depends_on=["analyze"],
                tool_plan=["citation_check"],
            ),
        ]
        return TaskDAG(topic=topic, tasks=tasks)

    def plan_for_send(self, topic: str, max_sources: int = 5) -> TaskDAG:
        """兼容旧 Runtime 的并行搜索 DAG。"""
        search_queries = [
            ("search_1", f"{topic} overview survey"),
            ("search_2", f"{topic} methods benchmarks evaluation"),
            ("search_3", f"{topic} limitations challenges recent work"),
        ]
        tasks = [
            Task(
                task_id=task_id,
                task_type="search",
                description=query,
                tool_plan=_retrieval_tool_plan(),
            )
            for task_id, query in search_queries
        ]
        tasks.extend([
            Task(
                task_id="analyze", task_type="analyze",
                description="Extract evidence cards from source full_text",
                depends_on=["read"], tool_plan=["evidence_extract"],
            ),
            Task(
                task_id="cite", task_type="cite",
                description="Check citation validity against source list",
                depends_on=["analyze"], tool_plan=["citation_check"],
            ),
        ])
        return TaskDAG(topic=topic, tasks=tasks)


class PlannerAgent:
    """把所有 ExecutionSpec 规划为统一 WorkPlan。"""

    role = "planner"
    allowed_tools: tuple[str, ...] = ()

    _PROFILE_TOOLS = {
        WorkerProfile.SEARCH: None,
        WorkerProfile.READ: {"paper_metadata", "source_quality_scorer"},
        WorkerProfile.ANALYZE: {"evidence_extract"},
        WorkerProfile.CITE: {"citation_check"},
        WorkerProfile.WRITE: set(),
    }

    def __init__(self, llm_client: StructuredLLMClient | None = None):
        self._llm_client = llm_client

    async def plan_hybrid(self, spec: ExecutionSpec) -> WorkPlan:
        """规则基线之上，仅为 LLM research 接受经过硬校验的候选。"""
        baseline = self.plan(spec)
        if (
            spec.execution_class != ExecutionClass.RESEARCH
            or str(spec.metadata.get("agent_mode") or "rule") != "llm"
        ):
            baseline.metadata["hybrid_planner"] = self._hybrid_info(False, False, "not_applicable")
            return baseline
        return await self._apply_llm_candidate(spec, baseline)

    async def replan_hybrid(self, plan: WorkPlan, verdict: ReviewVerdict) -> WorkPlan:
        """先确定受影响闭包，再允许 LLM 在该闭包内做一次有界修订。"""
        revised = self.replan(plan, verdict)
        spec = plan.execution_spec
        if (
            spec.execution_class != ExecutionClass.RESEARCH
            or str(spec.metadata.get("agent_mode") or "rule") != "llm"
        ):
            revised.metadata["hybrid_planner"] = self._hybrid_info(False, False, "not_applicable")
            return revised
        affected = {
            item.task_id for item in revised.items
            if bool(item.metadata.get("replanned"))
        }
        if not affected:
            revised.metadata["hybrid_planner"] = self._hybrid_info(False, False, "no_affected_tasks")
            return revised
        return await self._apply_llm_candidate(
            spec,
            revised,
            previous_plan=plan,
            affected_task_ids=affected,
            feedback=list(verdict.feedback),
        )

    async def _apply_llm_candidate(
        self,
        spec: ExecutionSpec,
        baseline: WorkPlan,
        *,
        previous_plan: Optional[WorkPlan] = None,
        affected_task_ids: Optional[Set[str]] = None,
        feedback: Optional[List[Dict[str, Any]]] = None,
    ) -> WorkPlan:
        from app.agents.llm_planner import LLMPlanner

        planner = LLMPlanner(self._llm_client)
        timeout_seconds = max(0.001, min(
            float(os.getenv("HYBRID_PLANNER_TIMEOUT_SECONDS", "30")),
            spec.budget.total_timeout_ms / 1000,
        ))
        payload = self._llm_spec_payload(spec)
        previous_payload = previous_plan.model_dump(mode="json") if previous_plan else None
        if previous_payload:
            # 关键步骤：提示中仅保留规划结构，绝不复制成功 Worker 的业务产物。
            previous_payload = {
                "plan_id": previous_payload["plan_id"],
                "revision": previous_payload["revision"],
                "items": [
                    {key: item[key] for key in (
                        "task_id", "profile", "instruction", "depends_on",
                        "allowed_tools", "strategy",
                    )}
                    for item in previous_payload["items"]
                    if not affected_task_ids or item["task_id"] in affected_task_ids
                ],
            }
        try:
            result = await asyncio.wait_for(
                planner.generate_work_plan_candidate(
                    payload,
                    previous_plan=previous_payload,
                    affected_task_ids=sorted(affected_task_ids or []),
                    feedback=feedback,
                    timeout_seconds=max(1, int(timeout_seconds)),
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            baseline.metadata["hybrid_planner"] = self._hybrid_info(True, False, "timeout")
            return baseline
        except Exception:
            baseline.metadata["hybrid_planner"] = self._hybrid_info(True, False, "client_error")
            return baseline
        if not result.get("success"):
            baseline.metadata["hybrid_planner"] = self._hybrid_info(
                True, False, "generation_failed", result,
            )
            return baseline
        try:
            candidate_plan = self._validated_candidate_plan(
                baseline,
                result["data"],
                affected_task_ids=affected_task_ids,
                previous_plan=previous_plan,
            )
        except Exception:
            baseline.metadata["hybrid_planner"] = self._hybrid_info(
                True, False, "candidate_rejected", result,
            )
            return baseline
        candidate_plan.metadata["hybrid_planner"] = self._hybrid_info(
            True, True, "accepted", result,
        )
        return candidate_plan

    def _validated_candidate_plan(
        self, baseline, candidate, *, affected_task_ids=None, previous_plan=None,
    ):
        raw_items = list(candidate.items)
        candidate_ids = {item.task_id for item in raw_items}
        baseline_ids = {item.task_id for item in baseline.items}
        if (
            len(candidate_ids) != len(raw_items)
            or len(raw_items) > 16
            or len(raw_items) > len(baseline.items)
        ):
            raise ValueError("LLM 候选规模或 task_id 非法")
        if affected_task_ids and candidate_ids != set(affected_task_ids):
            raise ValueError("LLM 重规划必须精确覆盖允许修改的任务集合")
        if not affected_task_ids and candidate_ids != baseline_ids:
            # 关键步骤：首次规划保持规则基线的稳定屏障，模型不得增删任务绕过规模预算。
            raise ValueError("LLM 候选必须保留规则基线任务集合")
        original = {item.task_id: item for item in baseline.items}
        templates = {item.profile: item for item in baseline.items}
        replacements = {}
        registry = ToolRegistry.get_instance()
        retrieval = set(_retrieval_tool_plan())
        tool_slots = 0
        for raw in raw_items:
            profile = WorkerProfile(raw.profile)
            base = original.get(raw.task_id) or templates.get(profile)
            if base is None or (raw.task_id in original and profile != base.profile):
                raise ValueError("LLM 不得改变任务 profile")
            if affected_task_ids and raw.task_id not in original:
                raise ValueError("LLM 重规划不得新增任务")
            allowed = retrieval if profile == WorkerProfile.SEARCH else self._PROFILE_TOOLS[profile]
            if set(raw.allowed_tools) - set(allowed or set()):
                raise ValueError("LLM 候选工具越权")
            if list(raw.allowed_tools) != list(base.allowed_tools):
                raise ValueError("LLM 候选不得缩减或扩张规则工具授权")
            tool_slots += len(raw.allowed_tools)
            expected_strategy = {
                WorkerProfile.SEARCH: WorkerStrategy.REACT,
                WorkerProfile.READ: WorkerStrategy.DETERMINISTIC,
                WorkerProfile.ANALYZE: WorkerStrategy.REACT,
                WorkerProfile.CITE: WorkerStrategy.DETERMINISTIC,
                WorkerProfile.WRITE: WorkerStrategy.SYNTHESIS,
            }[profile]
            if WorkerStrategy(raw.strategy) != expected_strategy:
                raise ValueError("LLM 候选执行策略越权")
            for tool_name in raw.allowed_tools:
                capability = registry.get_capability(tool_name)
                if capability is None:
                    raise ValueError("LLM 候选引用未知工具")
                policy = baseline.execution_spec.safety_policy
                if ((not policy.allow_business_tools)
                        or (capability.network_access and not policy.allow_network)
                        or (capability.external_write and not policy.allow_external_writes)
                        or (capability.destructive and not policy.allow_destructive_actions)
                        or tool_name in policy.denied_tools):
                    raise ValueError("LLM 候选违反安全策略")
            replacements[raw.task_id] = base.model_copy(update={
                "task_id": raw.task_id,
                "instruction": raw.instruction,
                "depends_on": list(raw.depends_on),
                "allowed_tools": list(raw.allowed_tools),
                "strategy": WorkerStrategy(raw.strategy),
                "input_data": {
                    **base.input_data,
                    **({"query": raw.instruction} if profile == WorkerProfile.SEARCH else {}),
                },
            }, deep=True)
        if tool_slots > baseline.execution_spec.budget.max_tool_calls:
            raise ValueError("LLM 候选超过总工具调用预算")
        items = (
            [replacements[item.task_id] for item in raw_items]
            if not affected_task_ids
            else [replacements.get(item.task_id, item) for item in baseline.items]
        )
        by_profile = {
            profile: {item.task_id for item in items if item.profile == profile}
            for profile in (
                WorkerProfile.SEARCH, WorkerProfile.READ, WorkerProfile.ANALYZE,
                WorkerProfile.CITE, WorkerProfile.WRITE,
            )
        }
        if any(not by_profile[profile] for profile in by_profile):
            raise ValueError("研究计划缺少必需阶段")
        if any(
            len(by_profile[profile]) != 1
            for profile in (
                WorkerProfile.READ, WorkerProfile.ANALYZE,
                WorkerProfile.CITE, WorkerProfile.WRITE,
            )
        ):
            raise ValueError("研究计划的下游阶段必须各有一个任务")
        fixed_stage_ids = {
            WorkerProfile.READ: "read",
            WorkerProfile.ANALYZE: "analyze",
            WorkerProfile.CITE: "cite",
            WorkerProfile.WRITE: "write",
        }
        if any(by_profile[profile] != {task_id} for profile, task_id in fixed_stage_ids.items()):
            raise ValueError("LLM 候选必须保留运行时稳定阶段 ID")
        search_ids = by_profile[WorkerProfile.SEARCH]
        baseline_search_ids = {
            item.task_id for item in baseline.items if item.profile == WorkerProfile.SEARCH
        }
        if not search_ids or search_ids != baseline_search_ids:
            raise ValueError("LLM 候选必须保留全部稳定 Search ID")
        exact_dependencies = {
            "read": search_ids,
            "analyze": {"read"},
            "cite": {"analyze"},
            "write": {"cite"},
        }
        if any(
            set(next(item for item in items if item.task_id == task_id).depends_on) != dependencies
            for task_id, dependencies in exact_dependencies.items()
        ):
            raise ValueError("LLM 候选破坏稳定阶段屏障")
        if previous_plan is not None:
            previous = {item.task_id: item for item in previous_plan.items}
            for item in items:
                if item.task_id not in set(affected_task_ids or ()):
                    continue
                old = previous.get(item.task_id)
                if old is None:
                    raise ValueError("重规划任务缺少上一版本")
                if item.profile == WorkerProfile.SEARCH:
                    if item.instruction == old.instruction:
                        raise ValueError("受影响 Search 必须改变真实查询")
                    rule_action = original[item.task_id].instruction
                    if rule_action not in item.instruction:
                        raise ValueError("LLM 不得撤销规则查询扩展")
                    if item.metadata.get("strategy_changed") != "query_expanded":
                        raise ValueError("Search strategy_changed 与真实动作不一致")
        for item in items:
            dependencies = set(item.depends_on)
            if item.profile == WorkerProfile.SEARCH and dependencies:
                raise ValueError("search 必须是根任务")
            required_upstream = {
                WorkerProfile.READ: by_profile[WorkerProfile.SEARCH],
                WorkerProfile.ANALYZE: by_profile[WorkerProfile.READ],
                WorkerProfile.CITE: by_profile[WorkerProfile.ANALYZE],
                WorkerProfile.WRITE: by_profile[WorkerProfile.CITE],
            }.get(item.profile, set())
            if required_upstream and not dependencies.intersection(required_upstream):
                raise ValueError("LLM 候选破坏研究阶段依赖")
        # 关键步骤：WorkPlan 的 schema/依赖/无环校验是接受模型候选前的最终硬门。
        return WorkPlan(
            plan_id=baseline.plan_id,
            execution_spec=baseline.execution_spec,
            items=items,
            replan_count=baseline.replan_count,
            repair_count=baseline.repair_count,
            revision=baseline.revision,
            reviewer_feedback=list(baseline.reviewer_feedback),
            metadata=dict(baseline.metadata),
        )

    @staticmethod
    def _llm_spec_payload(spec: ExecutionSpec) -> Dict[str, Any]:
        """Prompt 只携带协议与显式资源引用，不携带父 State 业务内容。"""
        return {
            "request_id": spec.request_id,
            "user_request": spec.user_request,
            "research_topic": spec.research_topic,
            "execution_class": spec.execution_class.value,
            "execution_route": spec.execution_route,
            "resource_ids": list(spec.resource_ids),
            "resources": list(spec.resources),
            "budget": spec.budget.model_dump(mode="json"),
            "safety_policy": spec.safety_policy.model_dump(mode="json"),
            "allowed_profiles": ["search", "read", "analyze", "cite", "write"],
            "allowed_tools": {
                "search": _retrieval_tool_plan(),
                "read": ["paper_metadata", "source_quality_scorer"],
                "analyze": ["evidence_extract"],
                "cite": ["citation_check"],
                "write": [],
            },
            "max_plan_items": 16,
        }

    @staticmethod
    def _hybrid_info(attempted, success, status, result=None):
        result = result or {}
        # 只记录可观测元数据，不记录 prompt、原始响应或内部推理。
        return {
            "attempted": bool(attempted),
            "success": bool(success),
            "status": status,
            "model": str(result.get("model") or ""),
            "latency_ms": int(result.get("latency_ms") or 0),
            "usage": dict(result.get("usage") or {}),
        }

    def plan(self, spec: ExecutionSpec) -> WorkPlan:
        # 关键步骤：唯一入口，按 Controller 定的执行复杂度分派为三类 WorkPlan。
        if spec.execution_class == ExecutionClass.ATOMIC:
            items = [self._atomic_item(spec)]
        elif spec.execution_class == ExecutionClass.CONTEXTUAL:
            items = self._contextual_items(spec)
        else:
            items = self._research_items(spec)
        return WorkPlan(
            plan_id=f"plan:{spec.request_id}",
            execution_spec=spec,
            items=items,
        )

    def replan(self, plan: WorkPlan, verdict: ReviewVerdict) -> WorkPlan:
        """Reviewer 仅能触发一次有界重规划。"""
        # 关键步骤：只有 Reviewer 判 REPLAN 才进入重规划，且整个 DAG 全局最多重规划一次。
        if verdict.outcome != ReviewOutcome.REPLAN:
            return plan
        if plan.replan_count >= 1:
            raise ValueError("有界 replan 次数已用尽")
        candidate = self.plan(plan.execution_spec)
        failed = set(verdict.failed_task_ids)
        # 关键步骤：只修订失败任务及其所有下游，未受影响节点保持原对象和配置。
        affected = set(failed)
        changed = True
        while changed:
            changed = False
            for item in plan.items:
                if item.task_id not in affected and affected.intersection(item.depends_on):
                    affected.add(item.task_id)
                    changed = True
        previous = {item.task_id: item for item in plan.items}
        revised_items = [
            item if item.task_id in affected else previous.get(item.task_id, item)
            for item in candidate.items
        ]
        reusable_analyze_input = any(
            item.get("task_id") == "analyze"
            and "read" in set(item.get("reusable_dependency_ids") or [])
            for item in verdict.feedback
        )
        if (
            plan.execution_spec.execution_class == ExecutionClass.RESEARCH
            and "analyze" in failed
            and reusable_analyze_input
        ):
            # 关键步骤：证据工具持续不可用时，由 Planner 显式收缩受影响子图；
            # Writing Worker 只能基于已完成的来源元数据生成受控降级报告。
            unaffected = [
                item for item in plan.items
                if item.task_id not in affected
            ]
            fallback_write = next(
                item for item in candidate.items if item.profile == WorkerProfile.WRITE
            )
            fallback_write.depends_on = ["read"]
            fallback_write.instruction = (
                f"仅基于可用来源元数据撰写关于 "
                f"{plan.execution_spec.research_topic} 的受控降级报告；"
                "明确说明证据抽取失败，不扩展来源未支持的结论"
            )
            fallback_write.metadata["degraded_source_only"] = True
            revised_items = unaffected + [fallback_write]
        revised = WorkPlan(
            plan_id=plan.plan_id,
            execution_spec=plan.execution_spec,
            items=revised_items,
            revision=1,
        )
        revised.replan_count = 1
        revised.reviewer_feedback = [
            str(item.get("message") or item.get("reason") or item)
            for item in verdict.feedback
        ]
        if plan.execution_spec.execution_class == ExecutionClass.ATOMIC:
            direct = revised.items[0]
            # 规则模式的 direct 首轮失败后，revision=1 退化为单工具 academic_search
            # 确定性重试；LLM 模式保留目录授权，让 ReAct 循环在新的一轮中自行换源。
            if str(plan.execution_spec.metadata.get("agent_mode") or "rule") != "llm":
                previous_tool = direct.allowed_tools[0] if direct.allowed_tools else ""
                if previous_tool != "academic_search" and ToolRegistry.get_instance().get("academic_search"):
                    # 关键步骤：fallback 成为 revision=1 的显式 WorkItem 授权，而非 Worker 偷调第二工具。
                    direct.allowed_tools = ["academic_search"]
                    direct.strategy = WorkerStrategy.DETERMINISTIC
                    decision = dict(plan.execution_spec.metadata.get("controller_decision") or {})
                    direct.input_data = {
                        **direct.input_data,
                        "_requested_count": int(decision.get("requested_count", 5) or 5),
                    }
                    direct.metadata["fallback_from"] = previous_tool
        for item in revised.items:
            item.metadata["revision"] = revised.revision
            if item.task_id in affected or item.metadata.get("degraded_source_only"):
                item.metadata["reviewer_feedback"] = list(revised.reviewer_feedback)
                item.metadata["replanned"] = True
                if item.profile == WorkerProfile.SEARCH:
                    # 关键步骤：重规划必须改变实际检索动作，避免原查询原样重跑。
                    feedback_suffix = " alternative independent sources"
                    item.instruction = f"{item.instruction}{feedback_suffix}"
                    item.input_data["query"] = item.instruction
                    item.metadata["strategy_changed"] = "query_expanded"
                elif item.task_id in failed:
                    # 关键步骤：非检索失败也必须改变实际执行策略，避免原参数原样重跑。
                    strategy_changes = {
                        WorkerProfile.READ: ("metadata_refresh", "force_refresh"),
                        WorkerProfile.ANALYZE: ("evidence_recovery", "extraction_strategy"),
                        WorkerProfile.CITE: ("strict_recheck", "citation_strategy"),
                        WorkerProfile.WRITE: ("bounded_rewrite", "writing_strategy"),
                    }
                    changed_strategy = strategy_changes.get(item.profile)
                    if changed_strategy:
                        value, key = changed_strategy
                        item.input_data[key] = value
                        item.metadata["strategy_changed"] = value
        return revised

    @staticmethod
    def _budget_args(spec: ExecutionSpec) -> Dict[str, int]:
        budget = spec.budget
        return {
            "max_tool_calls": budget.max_tool_calls,
            "max_iterations": budget.max_iterations,
            "per_tool_timeout_ms": budget.per_tool_timeout_ms,
            "timeout_ms": budget.total_timeout_ms,
        }

    @staticmethod
    def _safety_args(spec: ExecutionSpec) -> Dict[str, Any]:
        return {"safety_policy": spec.safety_policy.model_copy(deep=True)}

    def _atomic_item(self, spec: ExecutionSpec) -> WorkItem:
        # 关键步骤：direct_tool 只产生一个 DIRECT 任务，但按模式分两种形态：
        #   - LLM 模式：授权整个检索目录 + REACT 策略，Worker 的 ReAct 循环
        #     用 LLM 从目录自主挑工具（推荐/搜索/图谱都由它解析原文）。
        #   - 规则模式：按意图确定性选一个工具并构造参数，作为无 LLM 兜底。
        decision = dict(spec.metadata.get("controller_decision") or {})
        agent_mode = str(spec.metadata.get("agent_mode") or "rule")
        common_input = {
            "_requested_count": int(decision.get("requested_count", 5) or 5),
            "topic": spec.research_topic,
            "intent": spec.intent,
            "agent_mode": agent_mode,
            "llm_only": bool(spec.metadata.get("llm_only")),
        }
        budget = self._budget_args(spec)
        budget.pop("max_tool_calls")  # 每分支单独收紧调用次数
        common = {
            "task_id": "direct_1",
            "profile": WorkerProfile.DIRECT,
            "instruction": spec.user_request,
            "resources": list(spec.resources),
            **budget,
            **self._safety_args(spec),
        }
        if agent_mode == "llm":
            # LLM 模式：授权整个检索目录，交 ReAct 循环自主决策。
            # max_tool_calls 收紧到 3，控制一轮 direct 意图的调用成本。
            return WorkItem(
                **common,
                allowed_tools=_retrieval_tool_plan(),
                input_data=common_input,
                strategy=WorkerStrategy.REACT,
                max_tool_calls=min(spec.budget.max_tool_calls, 3),
            )
        # 规则模式：确定性单工具，参数由 _direct_rule_plan 构造（不依赖 LLM）。
        tool_name, tool_args = _direct_rule_plan(spec)
        return WorkItem(
            **common,
            allowed_tools=[tool_name] if tool_name else [],
            input_data={**common_input, **tool_args},
            strategy=WorkerStrategy.DETERMINISTIC,
        )

    def _contextual_items(self, spec: ExecutionSpec) -> List[WorkItem]:
        # 关键步骤：conversation 追问——先 context_load 装载会话资源，answer 再据引用回答。
        budget = self._budget_args(spec)
        common = {
            "agent_mode": str(spec.metadata.get("agent_mode") or "rule"),
            "llm_only": bool(spec.metadata.get("llm_only")),
        }
        return [
            WorkItem(
                task_id="context_load",
                profile=WorkerProfile.CONTEXT_LOAD,
                instruction="装载 Controller 显式解析的会话资源",
                resources=list(spec.resources),
                input_data={"resource_ids": spec.resource_ids, **common},
                strategy=WorkerStrategy.DETERMINISTIC,
                **budget,
                **self._safety_args(spec),
            ),
            WorkItem(
                task_id="answer",
                profile=WorkerProfile.ANSWER,
                instruction=spec.user_request,
                depends_on=["context_load"],
                input_data={"resource_ids": spec.resource_ids, **common},
                strategy=WorkerStrategy.SYNTHESIS,
                **budget,
                **self._safety_args(spec),
            ),
        ]

    def _research_items(self, spec: ExecutionSpec) -> List[WorkItem]:
        # 关键步骤：full_research——3 路并行检索 → read → analyze → cite → write 的完整 DAG。
        budget = self._budget_args(spec)
        topic = spec.research_topic
        search_tools = _retrieval_tool_plan()
        searches = [
            ("search_1", f"{topic} overview survey"),
            ("search_2", f"{topic} methods benchmarks evaluation"),
            ("search_3", f"{topic} limitations challenges recent work"),
        ]
        mode_data = {
            "agent_mode": str(spec.metadata.get("agent_mode") or "rule"),
            "llm_only": bool(spec.metadata.get("llm_only")),
        }
        items = [
            WorkItem(
                task_id=task_id,
                profile=WorkerProfile.SEARCH,
                instruction=query,
                allowed_tools=search_tools,
                resources=list(spec.resources),
                input_data={"query": query, "topic": topic, **mode_data},
                strategy=WorkerStrategy.REACT,
                **budget,
                **self._safety_args(spec),
            )
            for task_id, query in searches
        ]
        search_ids = [item.task_id for item in items]
        items.extend([
            WorkItem(
                task_id="read",
                profile=WorkerProfile.READ,
                instruction="标准化来源元数据并确定性评分",
                depends_on=search_ids,
                allowed_tools=["paper_metadata", "source_quality_scorer"],
                input_data=dict(mode_data),
                strategy=WorkerStrategy.DETERMINISTIC,
                **budget,
                **self._safety_args(spec),
            ),
            WorkItem(
                task_id="analyze",
                profile=WorkerProfile.ANALYZE,
                instruction=f"围绕 {topic} 抽取证据",
                depends_on=["read"],
                allowed_tools=["evidence_extract"],
                input_data={"topic": topic, **mode_data},
                strategy=WorkerStrategy.REACT,
                **budget,
                **self._safety_args(spec),
            ),
            WorkItem(
                task_id="cite",
                profile=WorkerProfile.CITE,
                instruction="确定性校验所有引用绑定",
                depends_on=["analyze"],
                allowed_tools=["citation_check"],
                input_data=dict(mode_data),
                strategy=WorkerStrategy.DETERMINISTIC,
                **budget,
                **self._safety_args(spec),
            ),
            WorkItem(
                task_id="write",
                profile=WorkerProfile.WRITE,
                instruction=f"撰写关于 {topic} 的最终报告",
                depends_on=["cite"],
                input_data={"topic": topic, **mode_data},
                strategy=WorkerStrategy.SYNTHESIS,
                **budget,
                **self._safety_args(spec),
            ),
        ])
        return items

    # ── 已废弃：与旧 Planner.plan_for_send 内容重复，生产实际调用 _planner.plan_for_send ──
    # 全局零引用（含测试），注释保留备查，如确定删除可整块移除。
    # def plan_for_send(self, topic: str, max_sources: int = 5) -> TaskDAG:
    #     # 注意：与旧 Planner.plan_for_send 内容重复；生产实际调用 _planner.plan_for_send，
    #     # 本方法已无人引用（含测试），仅作为历史兼容保留。
    #     """
    #     生成 Send API 专用的 Task DAG —— 多个不同搜索查询以证明动态分发。
    #
    #     规则版 Planner 生成 3 个不同角度的 search query：
    #     1. topic overview/survey
    #     2. topic methods/benchmarks
    #     3. topic limitations/recent work
    #
    #     每个 task_id 唯一（search_1, search_2, search_3）。
    #     不需要 LLM。
    #
    #     Reading tasks 在 search merge 后由 send_to_reading_worker 动态生成。
    #     """
    #     search_queries = [
    #         (f"search_1", f"{topic} overview survey"),
    #         (f"search_2", f"{topic} methods benchmarks evaluation"),
    #         (f"search_3", f"{topic} limitations challenges recent work"),
    #     ]
    #
    #     tasks = []
    #     for task_id, query in search_queries:
    #         tasks.append(Task(
    #             task_id=task_id,
    #             task_type="search",
    #             description=query,
    #             depends_on=[],
    #             tool_plan=_retrieval_tool_plan(),
    #         ))
    #
    #     # 后续的 analyze 和 cite task（read 由 Send API 动态生成）
    #     tasks.append(Task(
    #         task_id="analyze",
    #         task_type="analyze",
    #         description="Extract evidence cards from source full_text",
    #         depends_on=["read"],
    #         tool_plan=["evidence_extract"],
    #     ))
    #     tasks.append(Task(
    #         task_id="cite",
    #         task_type="cite",
    #         description="Check citation validity against source list",
    #         depends_on=["analyze"],
    #         tool_plan=["citation_check"],
    #     ))
    #
    #     return TaskDAG(topic=topic, tasks=tasks)
