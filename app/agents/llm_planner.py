"""
app/agents/llm_planner.py

LLMPlanner —— DeepSeek V4 Flash 驱动的研究任务规划器。

可选 LLM 模式，失败时自动回退规则 Planner。

输入：
- user topic, max_sources, mode
- available tool schemas

输出：
- TaskDAG 包含 2-3 个不同搜索角度的 search tasks
"""

import json
from typing import Any, Dict, Optional, Tuple

from app.agents.planner import Planner, Task, TaskDAG
from app.llm.client import get_llm_client
from app.llm.protocol import StructuredLLMClient
from app.llm.prompts import PLANNER_SYSTEM, PLANNER_USER
from app.llm.schemas import LLMPlannerOutput, LLMWorkPlanCandidate
from app.tools.registry import ToolRegistry


class LLMPlanner:
    """
    LLM 驱动的任务规划器。

    - DeepSeek V4 Flash 生成结构化搜索计划
    - Pydantic 校验输出（非法 JSON / 缺少字段 → 回退）
    - allowed_tools 白名单校验
    - API 失败 → 回退 Planner.plan_for_send()
    """

    def __init__(self, llm_client: StructuredLLMClient | None = None):
        self._rule_planner = Planner()
        self._llm = llm_client or get_llm_client()

    async def generate_work_plan_candidate(
        self,
        payload: Dict[str, Any],
        *,
        previous_plan: Optional[Dict[str, Any]] = None,
        affected_task_ids: Optional[list[str]] = None,
        feedback: Optional[list[Dict[str, Any]]] = None,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """生成统一 WorkPlan 候选；调用方继续负责全部确定性安全门。"""
        system_prompt = (
            "你是研究任务规划器。仅输出符合 schema 的任务结构，不输出思维链。"
            "不得新增资源、权限、预算或工具；search/read/analyze/cite/write 必须组成无环 DAG。"
        )
        # 关键步骤：重规划只向模型开放受影响子图，成功缓存不会进入可改写范围。
        user_payload = {
            "execution_spec": payload,
            "previous_plan": previous_plan or {},
            "affected_task_ids": affected_task_ids or [],
            "reviewer_structured_feedback": feedback or [],
        }
        return await self._llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False, default=str),
            output_schema=LLMWorkPlanCandidate,
            temperature=0.0,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )

    async def plan(
        self,
        topic: str,
        max_sources: int = 5,
        mode: str = "quick",
    ) -> Tuple[TaskDAG, Dict[str, Any]]:
        """
        使用 LLM 生成 TaskDAG。

        Returns:
            (TaskDAG, llm_result_dict)
            llm_result_dict contains "success", "latency_ms", "error" etc.
        """
        registry = ToolRegistry.get_instance()
        available_tools = registry.list_retrieval_capabilities()
        tool_catalog = "\n".join(
            f"- {name}: {registry.get(name).description}"
            for name in available_tools
            if registry.get(name) is not None
        )

        user_prompt = PLANNER_USER.format(
            topic=topic,
            max_sources=max_sources,
            mode=mode,
            available_tools=tool_catalog,
        )

        # 调用 LLM 生成结构化输出，符合 LLMPlannerOutput schema 的 JSON 字符串
        # 如果 LLM 输出不符合 schema，会自动触发回退
        result = await self._llm.generate_structured(
            system_prompt=PLANNER_SYSTEM,
            user_prompt=user_prompt,
            output_schema=LLMPlannerOutput,
        )

        if not result.get("success"):
            # 回退规则版
            dag = self._rule_planner.plan_for_send(topic=topic, max_sources=max_sources)
            return dag, result

        # 校验
        llm_output: LLMPlannerOutput = result["data"]  # 校验输出是否符合 LLMPlannerOutput schema
        errors = llm_output.validate_tools(available_tools)

        if errors:
            result["success"] = False
            result["error"] = "; ".join(errors)
            dag = self._rule_planner.plan_for_send(topic=topic, max_sources=max_sources)
            return dag, result

        # Pydantic → Task DAG
        tasks = []
        for st in llm_output.search_tasks:
            tasks.append(Task(
                task_id=st.task_id,
                task_type="search",
                description=st.query, # 对类型进行转换，LLM 只管"搜什么"（query），不需要关心"task_type 该叫什么"
                                    # —— 那是系统内部的约定。所以这一步转换把 LLM 的自由输出纳入系统的类型体系。
                depends_on=st.depends_on,
                # Planner defines the research angle; the existing Worker loop
                # autonomously chooses among the bounded registered retrieval
                # capabilities. This is an allowlist, not an execution order.
                tool_plan=list(available_tools),
            ))

        # 确保至少有 2 个 search task
        if len(tasks) < 2:
            dag = self._rule_planner.plan_for_send(topic=topic, max_sources=max_sources)
            result["success"] = False
            result["error"] = f"LLMPlanner generated only {len(tasks)} search task(s), minimum 2 required"
            return dag, result

        # 追加 analyze + cite tasks
        tasks.append(Task(
            task_id="analyze",
            task_type="analyze",
            description="Extract evidence cards from source full_text",
            depends_on=["read"],
            tool_plan=["evidence_extract"],
        ))
        tasks.append(Task(
            task_id="cite",
            task_type="cite",
            description="Check citation validity against source list",
            depends_on=["analyze"],
            tool_plan=["citation_check"],
        ))

        dag = TaskDAG(topic=topic, tasks=tasks)
        return dag, result
