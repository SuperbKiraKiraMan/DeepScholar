"""
app/tools/base.py

Tool 抽象基类 + ToolResult 统一返回结构。

类比 Spring Boot：
- BaseTool ≈ interface Tool（定义统一契约）
- ToolResult ≈ 统一的 R<?> 包装类（success + data + error）

设计原则：
1. 每个 Tool 必须定义 name、description、input_schema。
2. run() 永远不抛异常——失败返回 ToolResult(success=False, error="...")。
3. 调用方通过 result.success 判断成功，而不是 try-catch。
4. run() 自动计时并做异常兜底，子类只需实现 _arun()。
"""

import asyncio
import time
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ToolResult:
    """
    工具执行结果，类比 Spring 的统一响应包装类 R<T>。

    无论成功还是失败，都返回这个对象，不抛异常。
    """

    def __init__(
        self,
        success: bool,
        tool_name: str = "",
        data: Any = None,
        error: str = "",
        latency_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.tool_name = tool_name
        self.data = data
        self.error = error
        self.latency_ms = latency_ms
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，方便写入 trace / 日志。"""
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "data": self.data,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL: {self.error}"
        return f"ToolResult({self.tool_name}, {status}, {self.latency_ms}ms)"


class BaseTool(ABC):
    """
    工具抽象基类。

    每个子类必须实现：
    - name: 工具名（如 "mock_academic_search", "citation_check"）
    - description: 工具描述（给 LLM 看的，也给人看的）
    - input_schema: 输入参数的 JSON Schema
    - _arun(input_data): 异步执行逻辑

    类比 Java：
        public interface Tool {
            ToolResult execute(Map<String, Object> input);
            String getName();
            JsonSchema getInputSchema();
        }
    """

    def __init__(self):
        self._call_count = 0

    @staticmethod
    def _hash_input(kwargs: Dict[str, Any]) -> str:
        """计算稳定输入摘要，兼容历史调用方的重复检测入口。"""
        return str(hash(json.dumps(kwargs, sort_keys=True, default=str)))

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，如 'mock_academic_search', 'citation_check'。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述。"""
        ...

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        输入参数的 JSON Schema。子类可以 override 来提供更精确的 schema。
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """输出结果的 JSON Schema。"""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {"type": "object"},
                "error": {"type": "string"},
            },
        }

    @abstractmethod
    async def _arun(self, **kwargs) -> ToolResult:
        """
        子类实现具体逻辑。

        约定：
        - 不要在这里处理 timeout——timeout 由外层 run() 包装。
        - 返回 ToolResult，不抛异常。
        """
        ...

    async def run(self, **kwargs) -> ToolResult:
        """
        公共执行入口。类比 Spring Service 方法外层的事务代理。

        负责：
        1. 计时（latency_ms）
        2. 异常兜底（任何未捕获异常 → ToolResult(success=False)）
        3. 调用计数
        4. Harness observer hooks (before_tool / after_tool / on_error)

        子类不应 override 这个方法——只实现 _arun。
        """
        from app.observability.lifecycle import (
            emit_after_tool,
            emit_before_tool,
            emit_error,
            get_execution_context,
            reserve_runtime_tool_call,
        )

        start = time.time()
        execution_context = get_execution_context()
        role = execution_context.get("agent_role")
        if role:
            from app.agents.protocol import AgentProtocolViolation, agent_protocol

            allowed_tools = set(execution_context.get("allowed_tools") or [])
            try:
                # 关键步骤：权限、安全和预算必须在业务工具实现 _arun 之前逐次校验。
                if role != "worker":
                    agent_protocol.authorize_tool(role, self.name)
                if self.name not in allowed_tools:
                    raise AgentProtocolViolation(
                        "TOOL_NOT_GRANTED_FOR_TASK", f"任务未授予工具 {self.name}",
                    )
                _validate_worker_safety(self.name, kwargs, execution_context)
                local_budget = execution_context.get("local_tool_budget")
                if isinstance(local_budget, dict):
                    if int(local_budget.get("remaining", 0)) <= 0:
                        raise AgentProtocolViolation("TOOL_BUDGET_EXHAUSTED", "Worker 工具预算已耗尽")
                    local_budget["remaining"] = int(local_budget["remaining"]) - 1
                reserved, reason, runtime_remaining = reserve_runtime_tool_call(
                    str(execution_context.get("runtime_budget_id") or "")
                )
                if not reserved:
                    raise AgentProtocolViolation("RUNTIME_BUDGET_EXHAUSTED", reason)
            except AgentProtocolViolation as exc:
                denied = ToolResult(
                    success=False, tool_name=self.name,
                    error=f"{exc.code}: {exc}", metadata={"executed": False},
                )
                emit_error({
                    **execution_context, "tool_name": self.name,
                    "stage": "agent_permission", "exception_type": exc.code,
                    "error": str(exc)[:200],
                })
                return denied
        else:
            runtime_remaining = None

        self._call_count += 1
        event_context = {
            **execution_context,
            "tool_name": self.name,
            "call_count": self._call_count,
            "arg_keys": sorted(kwargs.keys()),
        }
        # 触发 before_tool 事件
        emit_before_tool(event_context)

        try:
            # 调用工具执行逻辑
            per_tool_seconds = max(
                0.001, int(execution_context.get("per_tool_timeout_ms") or 30_000) / 1000.0,
            )
            item_deadline = float(execution_context.get("item_deadline") or 0.0)
            item_remaining = item_deadline - time.monotonic() if item_deadline else None
            candidates = [per_tool_seconds]
            if runtime_remaining is not None:
                candidates.append(runtime_remaining)
            if item_remaining is not None:
                candidates.append(item_remaining)
            timeout_seconds = min(candidates)
            if timeout_seconds <= 0:
                raise asyncio.TimeoutError
            result = await asyncio.wait_for(self._arun(**kwargs), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            elapsed = int((time.time() - start) * 1000)
            result = ToolResult(
                success=False, tool_name=self.name,
                error="Tool execution timed out", latency_ms=elapsed,
                metadata={"executed": True},
            )
        except asyncio.CancelledError:
            elapsed = int((time.time() - start) * 1000)
            cancelled = ToolResult(
                success=False,
                tool_name=self.name,
                error="Tool execution cancelled or timed out",
                latency_ms=elapsed,
            )
            # 触发 after_tool 事件
            emit_after_tool({**event_context, **_result_event_data(cancelled)})
            # 触发 error 事件
            emit_error({
                **event_context,
                "stage": "tool_timeout",
                "exception_type": "CancelledError",
                "error": cancelled.error,
            })
            raise
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            result = ToolResult(
                success=False,
                tool_name=self.name,
                error=f"{type(e).__name__}: {str(e)}",
                latency_ms=elapsed,
            )
        else:
            elapsed = int((time.time() - start) * 1000)
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Invalid tool result type: {type(result).__name__}",
                    latency_ms=elapsed,
                )
            result.tool_name = self.name
            result.latency_ms = elapsed
            result.metadata.setdefault("executed", True)
        # 触发 after_tool 事件
        emit_after_tool({**event_context, **_result_event_data(result)})
        # 触发 error 事件
        if not result.success:
            emit_error({
                **event_context,
                "stage": "tool_execution",
                "exception_type": _error_type(result.error),
                "error": (result.error or "")[:200],
            })

        return result


def _validate_worker_safety(
    tool_name: str, kwargs: Dict[str, Any], execution_context: Dict[str, Any],
) -> None:
    """在统一工具入口校验网络、写入、破坏性动作和显式资源边界。"""
    from app.agents.protocol import AgentProtocolViolation
    from app.tools.registry import ToolRegistry

    policy = dict(execution_context.get("safety_policy") or {})
    capability = ToolRegistry.get_instance().get_capability(tool_name)
    if capability is None:
        raise AgentProtocolViolation("CAPABILITY_METADATA_MISSING", f"工具 {tool_name} 缺少 capability 元数据")
    if tool_name in set(policy.get("denied_tools") or []):
        raise AgentProtocolViolation("SAFETY_POLICY_DENIED", f"安全策略禁止工具 {tool_name}")
    if not policy.get("allow_network", True) and capability.network_access:
        raise AgentProtocolViolation("NETWORK_DENIED", f"安全策略禁止网络工具 {tool_name}")
    if not policy.get("allow_external_writes", False) and capability.external_write:
        raise AgentProtocolViolation("EXTERNAL_WRITE_DENIED", f"安全策略禁止外部写入工具 {tool_name}")
    if not policy.get("allow_destructive_actions", False) and capability.destructive:
        raise AgentProtocolViolation("DESTRUCTIVE_ACTION_DENIED", f"安全策略禁止破坏性工具 {tool_name}")
    if policy.get("require_explicit_resources", False) and capability.resource_scope == "explicit":
        explicit = set(execution_context.get("explicit_resource_ids") or [])
        referenced = _referenced_resource_ids(kwargs)
        if not explicit or not referenced or not referenced.issubset(explicit):
            raise AgentProtocolViolation(
                "RESOURCE_SCOPE_DENIED", f"工具 {tool_name} 引用了 WorkItem 未显式提供的资源",
            )


def _referenced_resource_ids(kwargs: Dict[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ("source", "sources", "citations"):
        value = kwargs.get(key)
        values.extend(value if isinstance(value, list) else [value] if isinstance(value, dict) else [])
    for key in ("source_id", "paper_id"):
        if kwargs.get(key):
            values.append({key: kwargs[key]})
    return {
        str(value.get("source_id") or value.get("paper_id") or "")
        for value in values if isinstance(value, dict)
        and (value.get("source_id") or value.get("paper_id"))
    }

def _result_event_data(result: ToolResult) -> Dict[str, Any]:
    return {
        "success": result.success,
        "latency_ms": result.latency_ms,
        "error_summary": (result.error or "")[:100] if not result.success else "",
        "result_summary": str(result.data)[:200] if result.data and result.success else "",
    }


def _error_type(error: str) -> str:
    if not error:
        return "ToolFailure"
    prefix = error.split(":", 1)[0].strip()
    return prefix if prefix and " " not in prefix else "ToolFailure"
