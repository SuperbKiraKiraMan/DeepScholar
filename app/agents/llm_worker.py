"""
app/agents/llm_worker.py

LLMWorker —— Agentic RAG 的核心执行器：LLM 通过 Function Calling 自主决策检索策略。

与规则 Worker 的关键区别：规则 Worker 按 task_type 硬编码调用哪些工具；
LLMWorker 让 LLM 观察每轮检索结果后自主决定——改写 query、切换检索源、或 finish。

受控循环：
  Task + WorkerContext
  → 构造 allowed tool schemas（OpenAI FC 格式）
  → LLM 返回 finish 或 tool_calls[{name, arguments}]
  → 白名单校验 + JSON Schema 严格校验 + 去重检测
  → 可信上下文注入（source_id → 完整 PaperSource，防止 LLM 伪造）
  → 执行 Tool（asyncio.wait_for per-tool timeout）
  → 构建有界 observation（≤6KB，top_papers ≤5）反馈给 LLM
  → LLM 根据 observation 决定：改写 query 再搜 / 切换检索源 / finish
  → 超限或连续失败时回退规则 Worker

检索任务的闭环决策（Agentic RAG 的关键）：
  - retrieval_query_rewritten：LLM 观察结果后改写查询词
  - retrieval_source_switched：LLM 判断当前源不够，切换到另一个检索能力
  - retrieval_finished：记录终止原因（evidence_sufficient / budget_exhausted / ...）

安全边界：
  - 白名单：LLM 只能调用 task.tool_plan 中列出的工具
  - citation_check 禁止 LLM 调用，必须由 Runtime 确定性执行
  - 参数严格校验：required / type / 数值范围 / 字符串长度 / enum / 数组 items
  - 去重：相同 tool + 相同 args 不重复执行
  - 预算：max_tool_calls(8) + max_iterations + worker_timeout_ms
  - 连续错误计数达到阈值 → 回退规则 Worker
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Set

from app.agents.planner import Task
from app.agents.worker import Worker, WorkerContext
from app.tools.registry import (
    ToolRegistry,
    canonicalize_args,
    retrieval_capability_for_tool,
    validate_tool_args_against_schema,
)
from app.tools.base import ToolResult
from app.tools.evidence_extract_tool import (
    available_source_identity,
    is_evidence_eligible_source,
)
from app.llm.client import get_llm_client
from app.llm.prompts import WORKER_SYSTEM, WORKER_USER
from app.observability.lifecycle import (
    emit_error,
    reset_execution_context,
    set_execution_context,
)


class LLMWorkerConfig:
    """LLM Worker 配置——每个 Send 子 Worker 独立一份。"""

    def __init__(
        self,
        max_tool_calls: int = 8,
        tool_timeout_ms: int = 30_000,
        worker_timeout_ms: int = 120_000,
        max_consecutive_errors: int = 3,
        max_iterations: Optional[int] = None,
        requested_count: Optional[int] = None,
    ):
        self.max_tool_calls = max_tool_calls
        self.tool_timeout_ms = tool_timeout_ms
        self.worker_timeout_ms = worker_timeout_ms
        self.max_consecutive_errors = max_consecutive_errors
        self.max_iterations = max_iterations or max(4, max_tool_calls * 2 + 2)
        # 用户期望的论文篇数（direct 推荐意图）。传入后用于：
        # 1) 提示 LLM 目标篇数；2) 工具调用时把 limit clamp 到剩余需求。
        # None = 未指定，不施加任何篇数约束（搜索/图谱/深度调研不受影响）。
        self.requested_count = requested_count


class LLMWorker:
    """
    Function Calling 驱动的 Worker。

    每个 LLMWorker 实例拥有独立的：
    - messages（对话历史）
    - called_set（去重检测）
    - error_count
    - tool_call_count

    不共享跨 Worker / 跨 Run 的全局可变状态。
    """

    def __init__(self, config: LLMWorkerConfig = None):
        self.config = config or LLMWorkerConfig()
        self._registry = ToolRegistry.get_instance()
        self._llm = get_llm_client()
        self._rule_worker = Worker()

        # 实例级隔离状态
        self._messages: List[Dict[str, Any]] = []
        self._called: Set[str] = set()
        self._tool_call_count = 0
        self._iteration_count = 0
        self._consecutive_errors = 0
        self._start_ms = 0
        self._retrieval_observations: List[Dict[str, Any]] = []
        self._last_retrieval_query = ""
        self._last_retrieval_capability = ""
        self._last_retrieval_provider = ""
        self._retrieval_finished_recorded = False

    # ================================================================
    # 主入口
    # ================================================================

    async def execute_task(
        self,
        task: Task,
        dependency_results: Dict[str, Any] = None,
        topic: str = "",
    ) -> WorkerContext:
        """
        LLM Function Calling 驱动的任务执行。

        Returns WorkerContext（与规则 Worker 相同接口）。
        失败时自动回退规则 Worker。
        """
        ctx = WorkerContext(task)
        deps = dependency_results or {}

        # Citation Worker 禁止 LLM 参与
        if task.task_type == "cite":
            return await self._rule_worker.execute_task(task, deps)

        # 获取 allowed_tools
        allowed_tool_names = task.tool_plan if task.tool_plan else self._default_tools(task.task_type)

        # ---- 注入可信 sources 到 ctx ----
        self._seed_trusted_sources(ctx, task.task_type, deps)

        # 初始化 run-scoped loop state
        self._called = set()
        self._tool_call_count = 0
        self._iteration_count = 0
        self._consecutive_errors = 0
        self._start_ms = int(time.time() * 1000)
        self._retrieval_observations = []
        self._last_retrieval_query = ""
        self._last_retrieval_capability = ""
        self._last_retrieval_provider = ""
        self._retrieval_finished_recorded = False

        # LLM 不可用时继续复用现有确定性 fallback。
        if not self._llm.is_available:
            ctx.warnings.append("LLM not available, using rule Worker")
            return await self._fallback_to_rule(ctx, task, deps, "llm_unavailable")

        # 初始化 messages
        self._messages = self._build_initial_messages(
            task, deps, allowed_tool_names, topic or task.description
        )

        # 获取 OpenAI function schemas
        tool_schemas = self._registry.get_function_schemas(allowed_tool_names)

        if not tool_schemas:
            ctx.warnings.append("No tools available for LLM Worker, using rule Worker")
            return await self._fallback_to_rule(ctx, task, deps, "no_allowed_tools")

        # ---- 记录 function_call_started trace ----
        ctx.add_trace("function_call_started", "llm_worker_loop_start",
                      ToolResult(success=True, data={
                          "task_id": task.task_id, "task_type": task.task_type,
                          "allowed_tools": allowed_tool_names,
                      }))

        # ---- 主循环（"搜索结果不满意就重来"的核心）----
        # 每轮让 LLM 自主决策下一步：看完上一轮检索的有界观察后，若结果不满意
        # 就再发一次工具调用（改写查询 / 换检索能力 / 换提供方）；
        # 直到 LLM 主动 finish 或工具调用 / 迭代预算耗尽。
        try:
            while (
                self._tool_call_count < self.config.max_tool_calls
                and self._iteration_count < self.config.max_iterations
            ):
                # Worker 总 timeout
                elapsed_ms = int(time.time() * 1000) - self._start_ms
                if elapsed_ms > self.config.worker_timeout_ms:
                    ctx.warnings.append("LLM Worker timeout; returning bounded partial results")
                    ctx.add_trace("tool_loop_limit_reached", "worker_timeout",
                                  ToolResult(success=False, error=f"worker_timeout_ms={self.config.worker_timeout_ms}"))
                    if _is_retrieval_task(task):
                        self._record_retrieval_finished(ctx, "worker_timeout", forced=True)
                        self._map_results_to_ctx(task.task_type, ctx)
                        return ctx
                    return await self._fallback_to_rule(ctx, task, deps, "worker_timeout")

                # LLM 决策
                self._iteration_count += 1
                fc_result = await self._llm.function_call(
                    messages=self._messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                )

                llm_success = bool(fc_result.get("success"))
                ctx.add_trace(
                    "llm_finished" if llm_success else "llm_failed",
                    "function_call",
                    ToolResult(
                        success=llm_success,
                        tool_name=str(fc_result.get("model") or "llm_worker"),
                        error="" if llm_success else str(fc_result.get("error") or "LLM call failed"),
                        latency_ms=int(fc_result.get("latency_ms") or 0),
                        metadata={
                            "agent": "llm_worker",
                            "model": fc_result.get("model", ""),
                            "usage": fc_result.get("usage", {}),
                        },
                    ),
                )

                if not fc_result.get("success"):
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self.config.max_consecutive_errors:
                        ctx.warnings.append("LLM Function Calling consecutive errors, falling back to rule")
                        return await self._fallback_to_rule(ctx, task, deps, "consecutive_errors")
                    continue

                self._consecutive_errors = 0  # reset

                # LLM 选择 finish
                if fc_result.get("finish"):
                    # ---- 任务完成验证 ----
                    validation_error = self._validate_task_completion(task.task_type, ctx)
                    if validation_error:
                        # 规则层兜底"重来"：LLM 自称完成但产出不达标
                        # （如搜索没找到任何带稳定 ID 的来源），就把"任务未完成"
                        # 消息塞回对话并 continue，强制 LLM 再调用一次工具。
                        ctx.warnings.append(f"Task completion validation failed: {validation_error}")
                        ctx.add_trace("tool_loop_limit_reached", f"incomplete_task:{validation_error[:100]}",
                                      ToolResult(success=False, error=validation_error))
                        self._messages.append({
                            "role": "user",
                            "content": f"Task not complete: {validation_error}. "
                                       f"Please call a tool to produce the required results.",
                        })
                        continue

                    ctx.add_result("llm_finish_summary", fc_result.get("content", ""))
                    ctx.add_trace("tool_loop_finished", "function_calling_complete",
                                  ToolResult(success=True, data={"summary": fc_result.get("content", "")[:200]}))
                    if _is_retrieval_task(task):
                        finish_reason = self._successful_retrieval_finish_reason(ctx)
                        self._record_retrieval_finished(ctx, finish_reason, forced=False)
                    self._map_results_to_ctx(task.task_type, ctx)
                    return ctx

                # LLM 返回 tool_calls
                tool_calls = fc_result.get("tool_calls", [])
                if not tool_calls:
                    ctx.warnings.append("LLM returned no tool_calls and no finish")
                    return await self._fallback_to_rule(ctx, task, deps, "empty_response")

                # ---- 阶段 2D.1: 多工具调用的预算检查 ----
                remaining = self.config.max_tool_calls - self._tool_call_count  # 计算剩余可用的工具调用次数
                if len(tool_calls) > remaining:
                    ctx.warnings.append(
                        f"LLM requested {len(tool_calls)} tool calls but only {remaining} remaining. "
                        f"Capping to {remaining}."
                    )
                    tool_calls = tool_calls[:remaining]  # 截断超出预算的工具调用，避免超限

                # ---- 阶段 2D.2: 将所有工具调用封装为一条 assistant 消息（arguments 使用 json.dumps）----
                self._messages.append({
                    "role": "assistant",
                    "content": None,  # 工具调用消息不需要 content
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{self._tool_call_count + i}"),  # 生成工具调用的唯一 ID
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),  # 工具名称
                                "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),  # 参数序列化，保留中文
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                })  # 将工具调用记录到对话历史，用于后续 LLM 理解上下文

                # 处理每个 tool_call
                for tc in tool_calls:
                    await self._handle_tool_call(tc, allowed_tool_names, ctx)

                # 检查是否全部失败
                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    ctx.warnings.append("Too many consecutive tool errors")
                    return await self._fallback_to_rule(ctx, task, deps, "tool_errors")

            # 达到 tool-call budget 或 iteration 上限。检索任务不得通过
            # Rule fallback 绕过这个安全边界。
            limit_name = (
                "max_tool_calls"
                if self._tool_call_count >= self.config.max_tool_calls
                else "max_iterations"
            )
            limit_value = (
                self.config.max_tool_calls
                if limit_name == "max_tool_calls"
                else self.config.max_iterations
            )
            ctx.warnings.append(
                f"LLM Worker reached {limit_name} ({limit_value}), "
                f"returning partial results"
            )
            ctx.add_trace("tool_loop_limit_reached", f"{limit_name}={limit_value}",
                          ToolResult(success=True))
            completion_error = self._validate_task_completion(task.task_type, ctx)
            if _is_retrieval_task(task):
                self._record_retrieval_finished(ctx, "budget_exhausted", forced=True)
                self._map_results_to_ctx(task.task_type, ctx)
                return ctx
            if completion_error:
                return await self._fallback_to_rule(
                    ctx, task, deps, f"{limit_name}_incomplete:{completion_error}"
                )
            self._map_results_to_ctx(task.task_type, ctx)
            return ctx

        except Exception as e:
            ctx.warnings.append(f"LLM Worker exception: {str(e)[:200]}")
            return await self._fallback_to_rule(ctx, task, deps, f"exception:{type(e).__name__}")

    # ================================================================
    # Tool Call 处理
    # ================================================================

    async def _handle_tool_call(
        self,
        tc: Dict[str, Any],
        allowed_tool_names: List[str],
        ctx: WorkerContext,
    ):
        """处理单个 tool_call：校验 → 去重 → 执行 → 记录 observation。"""
        tool_name = tc.get("name", "")
        tool_args = tc.get("arguments", {})
        tool_call_id = tc.get("id", f"call_{self._tool_call_count}")

        # ---- 0. Remaining budget check ----
        if self._tool_call_count >= self.config.max_tool_calls:
            ctx.add_trace("tool_loop_limit_reached", "budget_exceeded",
                          ToolResult(success=False, error=f"max_tool_calls={self.config.max_tool_calls}"))
            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json_dumps({"error": "Tool call budget exceeded"}),
            })
            return

        # ---- 1. Registry 校验（工具存在、白名单、citation_check 拦截、args 类型） ----
        error = self._registry.validate_tool_call(tool_name, tool_args, allowed_tool_names)
        if error:
            self._consecutive_errors += 1
            ctx.add_trace("tool_args_rejected", f"{tool_name}: {error[:100]}",
                          ToolResult(success=False, error=error))
            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json_dumps({"error": error}),
            })
            return

                    # ---- Trusted context injection（必须在 schema 校验之前） ----
        # 将 FC-friendly 参数（source_id / source_ids）解析为完整对象后再校验 -> 把 LLM 返回的 FC-friendly 参数转换成工具内部参数
        tool_args = self._inject_trusted_context(tool_name, tool_args, ctx)

        # ---- 验证 tool_args 是否符合 input_schema 校验 ----
        schema_error = validate_tool_args_against_schema(tool_name, tool_args, self._registry)
        if schema_error:
            self._consecutive_errors += 1
            ctx.add_trace("tool_args_rejected", f"{tool_name}: {schema_error[:100]}",
                          ToolResult(success=False, error=schema_error))
            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json_dumps({"error": schema_error}),
            })
            return

        # ---- 2. 去重校验 ----
        canon = canonicalize_args(tool_name, tool_args)
        dedup_key = f"{tool_name}:{canon}"
        if dedup_key in self._called:
            ctx.add_trace("tool_rejected", f"{tool_name}: duplicate call skipped",
                          ToolResult(success=True, data={"dedup": True}))
            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json_dumps({"result": "duplicate_call_skipped", "dedup_key": dedup_key[:80]}),
            })
            return
        self._called.add(dedup_key)

        # ---- 3. 执行 ----
        ctx.add_trace("tool_selected", f"{tool_name}({_summarize_args(tool_args)})",
                      ToolResult(success=True, data={"tool_name": tool_name}))

        tool = self._registry.get(tool_name)
        if tool is None:
            self._consecutive_errors += 1
            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json_dumps({"error": f"Tool '{tool_name}' not found"}),
            })
            return

        # ---- 篇数约束（源头 clamp）：用户目标篇数已知时，把检索类工具的
        # limit 收敛到"剩余需求"，防止 LLM 一轮把工具上限（通常 20）拉满，
        # 叠加多次工具调用后远超请求篇数。兜底裁剪见 _finalize_direct_output。
        if self.config.requested_count:
            collected = len(ctx.results.get("sources") or []) or 0
            remaining = max(1, self.config.requested_count - collected)
            props = (tool.input_schema or {}).get("properties") or {}
            for key in ("limit", "max_results", "top_k"):
                if key in props and isinstance(tool_args.get(key), int):
                    tool_args[key] = min(tool_args[key], remaining)
                    break

        self._tool_call_count += 1

        # ---- asyncio.wait_for per-tool timeout ----
        ctx.add_trace("tool_started", f"{tool_name}",
                      ToolResult(success=True, data={"tool_name": tool_name}))
        try:
            result: ToolResult = await asyncio.wait_for(
                tool.run(**tool_args),
                timeout=self.config.tool_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                success=False, tool_name=tool_name,
                error=f"Tool execution timed out after {self.config.tool_timeout_ms}ms",
            )
        except Exception as e:
            result = ToolResult(success=False, tool_name=tool_name,
                                error=f"{type(e).__name__}: {str(e)[:200]}")

        # 记录 trace
        ctx.add_trace("tool_finished", f"{tool_name}:{_summarize_args(tool_args)[:80]}", result)

        if result.success:
            self._consecutive_errors = 0
            self._accumulate_result(ctx, tool_name, result.data)
        else:
            self._consecutive_errors += 1

        # ---- 4. Observation（带匹配的 tool_call_id） ----
        if retrieval_capability_for_tool(tool_name):
            # 检索反馈环：把本次结果压缩成"有界观察"塞回对话，供 LLM 判断是否满意；
            # 不满意 → 下一轮循环里 LLM 自己再调一次检索（改写查询/换能力/换提供方）。
            # _retrieval_result_to_observation 负责判断论文的证据是否足够。
            observation = _retrieval_result_to_observation(
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
            )
            self._record_retrieval_action_trace(ctx, observation)
            self._retrieval_observations.append(observation)
            ctx.add_trace(
                "retrieval_observed",
                (
                    f"LLM selected {tool_name} after "
                    f"{len(self._retrieval_observations) - 1} prior observation(s)"
                ),
                ToolResult(
                    success=result.success,
                    tool_name="retrieval_observed",
                    error=result.error,
                    metadata={
                        **observation,
                        "selection_basis": (
                            f"Allowed capability selected by the LLM for task "
                            f"{ctx.task.task_id} after reviewing prior bounded observations"
                        ),
                    },
                ),
            )
            obs_content = _bounded_json_observation(observation)
        else:
            obs_content = _tool_result_to_observation(result)
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": obs_content,
        })

    # ================================================================
    # Helpers
    # ================================================================

    def _seed_trusted_sources(self, ctx: WorkerContext, task_type: str, deps: Dict[str, Any]):
        """从依赖结果中提取可信 sources 存入 ctx。"""
        dependency_sources = _sources_from_dependencies(deps)
        if task_type in ("read", "analyze") and dependency_sources:
            ctx.add_result("_trusted_sources", dependency_sources)
        elif task_type == "search" and dependency_sources:
            # Search tasks may legitimately perform No Retrieval when their
            # dependencies already carry sufficient source/evidence content.
            ctx.add_result("_dependency_sources", dependency_sources)
            ctx.add_result("search_results", list(dependency_sources))
            ctx.add_result("sources", list(dependency_sources))

    def _build_initial_messages(
        self, task: Task, deps: Dict[str, Any],
        allowed_tool_names: List[str], topic: str,
    ) -> List[Dict[str, Any]]:
        """构建初始 messages 列表，包含可信上下文。"""
        dep_summary = _summarize_deps(deps)
        remaining = self.config.max_tool_calls

        return [
            {"role": "system", "content": WORKER_SYSTEM},
            {"role": "user", "content": WORKER_USER.format(
                task_description=task.description,
                task_type=task.task_type, # 任务类型
                research_topic=topic,
                allowed_tools=", ".join(allowed_tool_names),
                dependency_summary=dep_summary,
                remaining_calls=remaining,
                requested_count=(
                    str(self.config.requested_count)
                    if self.config.requested_count
                    else "not specified"
                ),
            )},
        ]

    def _default_tools(self, task_type: str) -> List[str]:
        """各 task_type 的默认工具列表。"""
        mapping = {
            "search": self._registry.list_retrieval_capabilities(),
            "read": ["paper_metadata", "source_quality_scorer"],
            "analyze": ["evidence_extract"],
        }
        return mapping.get(task_type, [])

    def _record_retrieval_action_trace(
        self,
        ctx: WorkerContext,
        observation: Dict[str, Any],
    ) -> None:
        """记录"重来"的可观测证据（写入 trace / SSE）：
        - 查询被明显改写     → retrieval_query_rewritten
        - 切换检索能力/提供方 → retrieval_source_switched
        """
        query = str(observation.get("query") or "")
        provider = str(observation.get("provider") or "")
        capability = retrieval_capability_for_tool(
            str(observation.get("tool_name") or "")
        ) or str(observation.get("tool_name") or "")

        if (
            self._last_retrieval_query
            and _normalize_retrieval_query(query)
            != _normalize_retrieval_query(self._last_retrieval_query)
        ):
            ctx.add_trace(
                "retrieval_query_rewritten",
                "LLM issued a materially different query after a retrieval observation",
                ToolResult(
                    success=True,
                    tool_name="retrieval_query_rewritten",
                    metadata={
                        "previous_query": _clip(self._last_retrieval_query, 500),
                        "new_query": _clip(query, 500),
                        "rewrite_reason": (
                            "The LLM refined the query after evaluating prior "
                            "relevance, coverage, recency, or evidence sufficiency"
                        ),
                        "tool_name": _clip(str(observation.get("tool_name") or ""), 160),
                        "provider": _clip(provider, 80),
                    },
                ),
            )

        if self._last_retrieval_capability and capability != self._last_retrieval_capability:
            ctx.add_trace(
                "retrieval_source_switched",
                "LLM selected a different retrieval capability after observation",
                ToolResult(
                    success=True,
                    tool_name="retrieval_source_switched",
                    metadata={
                        "from_capability": self._last_retrieval_capability,
                        "from_provider": _clip(self._last_retrieval_provider, 80),
                        "to_capability": capability,
                        "to_provider": _clip(provider, 80),
                        "switch_reason": (
                            "The LLM switched capability after judging the prior "
                            "observation against the current task"
                        ),
                    },
                ),
            )

        self._last_retrieval_query = query
        self._last_retrieval_capability = capability
        self._last_retrieval_provider = provider

    def _successful_retrieval_finish_reason(self, ctx: WorkerContext) -> str:
        """定义"什么算搜索满意"的收尾原因：依赖已足 / 证据充分 / 仅有可发现来源 / 无可用来源。"""
        if not self._retrieval_observations and ctx.results.get("_dependency_sources"):
            return "dependencies_sufficient"
        if _available_text_count(_context_sources(ctx)):
            return "evidence_sufficient"
        if _available_sources(_context_sources(ctx)):
            return "discovery_sources_sufficient"
        return "no_usable_sources"

    def _record_retrieval_finished(
        self,
        ctx: WorkerContext,
        finish_reason: str,
        *,
        forced: bool,
    ) -> None:
        """Record exactly one bounded terminal event for a retrieval task."""
        if self._retrieval_finished_recorded:
            return
        sources = _available_sources(_context_sources(ctx))
        available_text_count = _available_text_count(sources)
        ctx.add_trace(
            "retrieval_finished",
            finish_reason,
            ToolResult(
                success=bool(sources),
                tool_name="retrieval_finished",
                error="" if sources else finish_reason,
                metadata={
                    "finish_reason": finish_reason,
                    "tool_call_count": self._tool_call_count,
                    "observed_source_count": len(sources),
                    "available_text_count": available_text_count,
                    "evidence_sufficient": available_text_count > 0 and not forced,
                    "forced_stop": forced,
                },
            ),
        )
        self._retrieval_finished_recorded = True

    def _inject_trusted_context(
        self, tool_name: str, tool_args: Dict[str, Any], ctx: WorkerContext,
    ) -> Dict[str, Any]:
        """
        可信上下文注入（严格 source_id 解析）。

        LLM 只选择 source_id / source_ids，程序从 ctx 解析完整 PaperSource 并注入。
        LLM 无法改写 canonical URL、source_id 或 full_text。

        FC-friendly 参数映射：
        - evidence_extract: source_id (str) → source (dict)
        - paper_metadata/source_quality_scorer: source_ids (str[]) → sources (dict[])
        """
        ctx_sources = ctx.results.get("_trusted_sources", [])

        if tool_name in ("paper_metadata", "source_quality_scorer"):
            # Some OpenAI-compatible models occasionally use the internal field
            # name but still send IDs. Treat it as an alias, then resolve only
            # against the run-scoped trusted source set.
            if "sources" in tool_args and isinstance(tool_args["sources"], list):
                if all(isinstance(item, str) for item in tool_args["sources"]):
                    tool_args["source_ids"] = tool_args.pop("sources")

            # FC-friendly: source_ids (str[]) → resolve to full sources
            if "source_ids" in tool_args and isinstance(tool_args["source_ids"], list):
                resolved = []
                source_ids_set = set(tool_args["source_ids"])
                for src in (ctx_sources or []):
                    if src.get("source_id") in source_ids_set:
                        resolved.append(_safe_source_copy(src))
                if resolved:
                    tool_args["sources"] = resolved
                # Remove FC-friendly key after resolution
                tool_args.pop("source_ids", None)

            # Also handle direct sources array (backward compat)
            elif "sources" in tool_args:
                if ctx_sources:
                    tool_args["sources"] = ctx_sources

        elif tool_name == "evidence_extract":
            # Normalize the common FC deviation source="W..." to source_id.
            # The ID still has to resolve to a trusted source below.
            if "source" in tool_args and isinstance(tool_args["source"], str):
                tool_args["source_id"] = tool_args.pop("source")

            # FC-friendly: source_id (str) → resolve to full source object
            if "source_id" in tool_args and isinstance(tool_args["source_id"], str):
                sid = tool_args["source_id"]
                found = None
                for src in (ctx_sources or []):
                    if src.get("source_id") == sid:
                        found = _safe_source_copy(src)
                        break
                if found:
                    tool_args["source"] = found
                # Remove FC-friendly key
                tool_args.pop("source_id", None)

            # Backward compat: LLM passed a source dict → override with trusted
            if "source" in tool_args and isinstance(tool_args["source"], dict):
                llm_sid = tool_args["source"].get("source_id", "")
                for src in (ctx_sources or []):
                    if src.get("source_id") == llm_sid:
                        tool_args["source"] = _safe_source_copy(src)
                        break

        return tool_args

    def _accumulate_result(self, ctx: WorkerContext, tool_name: str, data: Any):
        """将工具执行结果累积到 ctx.results 中。"""
        if data is None:
            return

        if retrieval_capability_for_tool(tool_name) and isinstance(data, dict):
            ctx.record_provider_fallback(data)
            results_list = data.get("sources") or data.get("results") or []
            if not isinstance(results_list, list):
                results_list = []
            existing = ctx.results.get("search_results", [])
            ctx.add_result("search_results", existing + results_list)
            ctx.add_result("sources", ctx.results["search_results"])
            cards = data.get("evidence_cards") or []
            if isinstance(cards, list) and cards:
                existing_cards = ctx.results.get("evidence_cards", [])
                ctx.add_result("evidence_cards", existing_cards + cards)
            if tool_name.startswith("mcp__"):
                ctx.add_result(
                    "mcp_results",
                    {
                        **ctx.results.get("mcp_results", {}),
                        tool_name: data,
                    },
                )

        elif tool_name == "paper_metadata":
            normalized = data.get("sources", [])
            ctx.add_result("_meta_normalized", normalized)
            if "_trusted_sources" not in ctx.results:
                ctx.add_result("_trusted_sources", normalized)

        elif tool_name == "source_quality_scorer":
            scores_by_id = data.get("scores_by_id", {})
            meta_sources = ctx.results.get("_meta_normalized", [])
            for source in (meta_sources or []):
                sid = source.get("source_id", "")
                if sid in scores_by_id:
                    source["quality_score"] = scores_by_id[sid]["total"]
                else:
                    source.setdefault("quality_score", 0.0)
            ctx.add_result("scored_sources", meta_sources)
            ctx.add_result("_trusted_sources", meta_sources)

        elif tool_name == "evidence_extract":
            cards = data.get("evidence_cards", [])
            existing = ctx.results.get("evidence_cards", [])
            ctx.add_result("evidence_cards", existing + cards)

    def _map_results_to_ctx(self, task_type: str, ctx: WorkerContext):
        """LLM 完成后，确保 ctx.results 包含 task_type 所需的标准 key。"""
        if task_type == "search":
            ctx.results.setdefault("sources", [])
            ctx.results.setdefault("search_results", [])
        elif task_type == "read":
            ctx.results.setdefault("scored_sources", [])
        elif task_type == "analyze":
            ctx.results.setdefault("evidence_cards", [])

    def _validate_task_completion(self, task_type: str, ctx: WorkerContext) -> Optional[str]:
        """
        验证任务是否真正产生必要结果，不能空手 finish。

        Returns None if task is complete, otherwise an error message.

        Analyze 必须对有文本的来源形成最小证据覆盖。仅调用过工具
        但返回空证据不再视为任务完成。
        """
        if task_type == "search":
            sources = _available_sources(_context_sources(ctx))
            if not sources:
                if not self._retrieval_observations:
                    return (
                        "retrieval task has no usable dependency sources and has not "
                        "called any retrieval tool"
                    )
                return (
                    "retrieval task must produce at least one titled source with a "
                    "stable paper/source ID, DOI, or URL"
                )
        elif task_type == "read":
            scored = ctx.results.get("scored_sources", [])
            if not scored:
                return "read task must produce scored_sources (metadata + quality scores)"
        elif task_type == "analyze":
            if ctx.tool_call_count == 0:
                return "analyze task must attempt evidence extraction (no tools called)"
            trusted = ctx.results.get("_trusted_sources", [])
            analyzable_ids = {
                source.get("source_id") for source in trusted
                if source.get("source_id") and is_evidence_eligible_source(source)
            }
            cards = ctx.results.get("evidence_cards", [])
            covered_ids = {
                card.get("source_id") for card in cards if card.get("source_id")
            }
            required = min(3, len(analyzable_ids))
            if required and len(covered_ids & analyzable_ids) < required:
                return (
                    "analyze task must produce evidence for at least "
                    f"{required} source(s); covered={len(covered_ids & analyzable_ids)}"
                )
        return None

    async def _fallback_to_rule(
        self, ctx: WorkerContext, task: Task,
        deps: Dict[str, Any], reason: str,
    ) -> WorkerContext:
        """回退到规则 Worker，保留 LLM 阶段的 trace。"""
        if _is_retrieval_task(task):
            finish_reason = {
                "worker_timeout": "worker_timeout",
                "consecutive_errors": "consecutive_errors",
                "tool_errors": "consecutive_errors",
                "empty_response": "no_usable_sources",
                "llm_unavailable": "provider_unavailable",
                "no_allowed_tools": "provider_unavailable",
            }.get(reason.split(":", 1)[0], "provider_unavailable")
            self._record_retrieval_finished(ctx, finish_reason, forced=True)
        emit_error({
            "stage": "llm_worker_fallback",
            "task_id": task.task_id,
            "exception_type": "Fallback",
            "fallback_reason": reason,
            "error": reason,
        })
        ctx.add_trace("tool_loop_fallback", reason,
                      ToolResult(success=True, data={"fallback_reason": reason}))
        rule_ctx = await self._rule_worker.execute_task(task, deps)
        rule_ctx.trace = ctx.trace + rule_ctx.trace
        rule_ctx.warnings = ctx.warnings + rule_ctx.warnings
        rule_ctx.results = {**ctx.results, **rule_ctx.results}
        return rule_ctx


# ================================================================
# 工具函数
# ================================================================

def json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _summarize_args(args: Dict[str, Any]) -> str:
    """安全摘要 args（不记录 API key / full_text）。"""
    if not args:
        return "{}"
    parts = []
    for k, v in args.items():
        if k in ("full_text", "Authorization", "api_key"):
            continue
        if isinstance(v, str):
            parts.append(f"{k}={v[:50]}")
        elif isinstance(v, list):
            parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{...{len(v)} keys}}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)[:200]


def _summarize_deps(deps: Dict[str, Any]) -> str:
    """安全摘要 dependency results（包含 source_id 列表）。"""
    if not deps:
        return "(no dependencies)"
    parts = []
    for tid, dep_ctx in deps.items():
        if hasattr(dep_ctx, 'results'):
            keys = list(dep_ctx.results.keys())
            detail = f"{tid}: {keys}"
            sources = dep_ctx.results.get("sources", []) or dep_ctx.results.get("search_results", [])
            if sources:
                # Stable identifiers are an execution contract, not display text.
                # Never truncate them: the model must return the exact ID so the
                # trusted-context gate can resolve the canonical source object.
                sids = [str(s.get("source_id", "?")) for s in sources[:12]]
                if len(sources) > 12:
                    sids.append(f"...+{len(sources)-12}")
                detail += f" [source_ids: {', '.join(sids)}]"
            parts.append(detail)
        else:
            parts.append(f"{tid}: (no results)")
    return "; ".join(parts)


def _tool_result_to_observation(result: ToolResult) -> str:
    """Keep the legacy bounded observation for non-retrieval tools."""
    try:
        return json.dumps({
            "success": result.success,
            "tool_name": result.tool_name,
            "summary": str(result.data)[:500] if result.data else "",
            "error": result.error[:200] if result.error else "",
        }, default=str, ensure_ascii=False)
    except Exception:
        return f"success={result.success}, error={result.error[:100]}"


def _retrieval_result_to_observation(
    *,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: ToolResult,
    top_papers_limit: int = 5,
) -> Dict[str, Any]:
    """将检索工具结果转换为有证据的有界观察。"""
    data = result.data if isinstance(result.data, dict) else {}
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    raw_results = data.get("sources") or data.get("results") or []
    if not isinstance(raw_results, list):
        raw_results = []
    paper_results = [item for item in raw_results if isinstance(item, dict)]

    years = []
    for item in paper_results:
        year = _parse_year(item.get("year") or item.get("publication_year"))
        if year is not None:
            years.append(year)
    year_min = _parse_year(metadata.get("year_min"))
    year_max = _parse_year(metadata.get("year_max"))
    if year_min is None and years:
        year_min = min(years)
    if year_max is None and years:
        year_max = max(years)

    reported_count = _first_int(
        metadata.get("result_count"),
        data.get("total_found"),
        data.get("result_count"),
        data.get("total"),
    )
    result_count = reported_count if reported_count is not None else len(paper_results)

    provider = _first_text(
        metadata.get("provider"),
        data.get("provider"),
        paper_results[0].get("provider") if paper_results else None,
        _provider_for_capability(tool_name),
    )
    fallback_used = bool(
        metadata.get("fallback_used") or data.get("fallback_used")
    )
    warning = _first_text(
        metadata.get("warning"),
        data.get("warning"),
        "; ".join(str(item) for item in data.get("warnings", [])[:3])
        if isinstance(data.get("warnings"), list)
        else None,
        (
            f"fallback: {metadata.get('fallback_reason') or data.get('fallback_reason')}"
            if fallback_used
            else None
        ),
    )

    top_papers = []
    for item in paper_results[: max(1, min(top_papers_limit, 5))]:
        paper_id = _first_text(
            item.get("paper_id"),
            item.get("semantic_scholar_id"),
            item.get("openalex_id"),
            item.get("source_id"),
            item.get("doi"),
            item.get("url"),
        )
        score = _first_number(
            item.get("retrieval_score"),
            item.get("score"),
            item.get("relevance_score"),
            item.get("quality_score"),
        )
        top_papers.append({
            "paper_id": _clip(paper_id, 180) or None,
            "title": _clip(str(item.get("title") or ""), 240),
            "score": score,
        })

    query = _first_text(
        tool_args.get("query"),
        tool_args.get("topic"),
        tool_args.get("paper_query"),
        data.get("query"),
    )
    return {
        "tool_name": _clip(tool_name, 180),
        "provider": _clip(provider, 80),
        "query": _clip(query, 500),
        "result_count": max(0, result_count),
        "year_min": year_min,
        "year_max": year_max,
        "top_papers": top_papers,
        "fallback_used": fallback_used,
        "error": _clip(result.error, 300) or None,
        "warning": _clip(warning, 300) or None,
        "available_text_count": _available_text_count(paper_results),
    }


def _bounded_json_observation(
    observation: Dict[str, Any],
    max_bytes: int = 6 * 1024,
) -> str:
    """Serialize without ever truncating the JSON byte stream itself."""
    bounded = json.loads(json.dumps(observation, default=str, ensure_ascii=False))

    def encode() -> str:
        return json.dumps(
            bounded,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    encoded = encode()
    while len(encoded.encode("utf-8")) > max_bytes and bounded.get("top_papers"):
        bounded["top_papers"].pop()
        encoded = encode()

    if len(encoded.encode("utf-8")) > max_bytes:
        for key, limit in (
            ("query", 200),
            ("error", 160),
            ("warning", 160),
            ("provider", 48),
            ("tool_name", 96),
        ):
            if isinstance(bounded.get(key), str):
                bounded[key] = _clip(bounded[key], limit)
        encoded = encode()

    # The fixed scalar envelope is far below this cap. This final defensive
    # fallback still returns a complete JSON object if unusual input expands.
    if len(encoded.encode("utf-8")) > max_bytes:
        bounded["top_papers"] = []
        bounded["query"] = _clip(str(bounded.get("query") or ""), 96)
        bounded["error"] = _clip(str(bounded.get("error") or ""), 96) or None
        bounded["warning"] = _clip(str(bounded.get("warning") or ""), 96) or None
        encoded = encode()
    return encoded


def _provider_for_capability(tool_name: str) -> str:
    return {
        "local_paper_search": "local_zotero",
        "academic_search": "openalex",
        "semantic_scholar_search": "semantic_scholar",
        "semantic_scholar_graph": "semantic_scholar",
        "semantic_scholar_recommendations": "semantic_scholar",
    }.get(retrieval_capability_for_tool(tool_name) or "", "unknown")


def _sources_from_dependencies(deps: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for dep_ctx in deps.values():
        if not hasattr(dep_ctx, "results"):
            continue
        candidate = (
            dep_ctx.results.get("sources")
            or dep_ctx.results.get("search_results")
            or dep_ctx.results.get("scored_sources")
            or []
        )
        if isinstance(candidate, list):
            sources.extend(item for item in candidate if isinstance(item, dict))
    return _deduplicate_sources(sources)


def _context_sources(ctx: WorkerContext) -> List[Dict[str, Any]]:
    sources = (
        ctx.results.get("sources")
        or ctx.results.get("search_results")
        or ctx.results.get("_dependency_sources")
        or []
    )
    return _deduplicate_sources(
        [item for item in sources if isinstance(item, dict)]
        if isinstance(sources, list)
        else []
    )


def _available_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [source for source in sources if available_source_identity(source)]


def _available_text_count(sources: List[Dict[str, Any]]) -> int:
    identities = {
        available_source_identity(source)
        for source in sources
        if is_evidence_eligible_source(source)
    }
    return len({identity for identity in identities if identity})


def _deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for source in sources:
        identity = available_source_identity(source)
        key = identity or f"anonymous:{id(source)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _is_retrieval_task(task: Task) -> bool:
    if task.task_type == "search":
        return True
    return any(retrieval_capability_for_tool(name) for name in task.tool_plan)


def _normalize_retrieval_query(query: str) -> str:
    """Ignore only casing and whitespace for meaningful rewrite detection."""
    return re.sub(r"\s+", " ", str(query or "")).strip().casefold()


def _parse_year(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1000 <= value <= 3000:
        return value
    if isinstance(value, str):
        match = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", value)
        if match:
            return int(match.group(1))
    return None


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            if value is not None and str(value).strip():
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return round(float(value), 6)
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _safe_source_copy(source: Dict[str, Any]) -> Dict[str, Any]:
    """创建 source 的安全副本（去除内部标记字段，保留权威字段）。"""
    safe_keys = {
        "source_id", "paper_id", "title", "url", "doi", "snippet",
        "full_text", "abstract", "text", "quote", "authors", "year",
        "venue", "source_type", "quality_score", "page", "section",
        "source_path", "provider", "content_source",
    }
    return {k: v for k, v in source.items() if k in safe_keys}
