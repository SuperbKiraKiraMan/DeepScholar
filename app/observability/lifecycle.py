"""
Agent Runtime 的最小生命周期观察者协议。

应用层定义这个协议。可选的观察者（包括 Harness）通过 contextvars 注入，
确保并发的 Send worker 能继承正确的运行上下文，而无需依赖全局可变状态。
"""

import contextvars
import threading
import time
from typing import Any, Dict, Optional, Protocol


class ObserverProtocol(Protocol):
    def on_before_run(self, data: Dict[str, Any]) -> None: ...
    def on_after_plan(self, data: Dict[str, Any]) -> None: ...
    def on_before_tool(self, data: Dict[str, Any]) -> None: ...
    def on_after_tool(self, data: Dict[str, Any]) -> None: ...
    def on_after_run(self, data: Dict[str, Any]) -> None: ...
    def on_error(self, data: Dict[str, Any]) -> None: ...


_current_observer: contextvars.ContextVar[Optional[ObserverProtocol]] = (
    contextvars.ContextVar("agent_lifecycle_observer", default=None)
)
_execution_context: contextvars.ContextVar[Dict[str, Any]] = (
    contextvars.ContextVar("agent_execution_context", default={})
)
_runtime_budget_lock = threading.RLock()
_runtime_budgets: Dict[str, Dict[str, Any]] = {}


def register_runtime_budget(budget_id: str, *, max_tool_calls: int, total_timeout_ms: int) -> None:
    """注册一次运行共享的工具次数与墙钟截止时间。"""
    with _runtime_budget_lock:
        _runtime_budgets[budget_id] = {
            "remaining_tool_calls": max(0, int(max_tool_calls)),
            "deadline": time.monotonic() + max(1, int(total_timeout_ms)) / 1000.0,
        }


def reserve_runtime_tool_call(budget_id: str) -> tuple[bool, str, float | None]:
    """在业务工具执行前原子扣减运行级预算，并返回剩余墙钟秒数。"""
    if not budget_id:
        return True, "", None
    with _runtime_budget_lock:
        budget = _runtime_budgets.get(budget_id)
        if budget is None:
            return False, "运行级预算不存在或已释放", 0.0
        remaining_seconds = float(budget["deadline"]) - time.monotonic()
        if remaining_seconds <= 0:
            return False, "运行总超时预算已耗尽", 0.0
        if int(budget["remaining_tool_calls"]) <= 0:
            return False, "运行工具调用预算已耗尽", remaining_seconds
        budget["remaining_tool_calls"] = int(budget["remaining_tool_calls"]) - 1
        return True, "", remaining_seconds


def runtime_deadline_remaining(budget_id: str) -> tuple[bool, str, float | None]:
    """只检查运行级墙钟截止时间，不消耗工具调用次数。"""
    if not budget_id:
        return True, "", None
    with _runtime_budget_lock:
        budget = _runtime_budgets.get(budget_id)
        if budget is None:
            return False, "运行级预算不存在或已释放", 0.0
        remaining_seconds = float(budget["deadline"]) - time.monotonic()
        if remaining_seconds <= 0:
            return False, "运行总超时预算已耗尽", 0.0
        return True, "", remaining_seconds


def clear_runtime_budget(budget_id: str) -> None:
    with _runtime_budget_lock:
        _runtime_budgets.pop(budget_id, None)


def runtime_budget_snapshot(budget_id: str) -> Dict[str, Any]:
    """测试与可观测性读取入口，不暴露可变预算对象。"""
    with _runtime_budget_lock:
        return dict(_runtime_budgets.get(budget_id) or {})


def set_observer(observer: Optional[ObserverProtocol]) -> contextvars.Token:
    return _current_observer.set(observer)


def reset_observer(token: contextvars.Token) -> None:
    _current_observer.reset(token)


def set_execution_context(**values: Any) -> contextvars.Token:
    merged = dict(_execution_context.get())
    merged.update({k: v for k, v in values.items() if v not in (None, "")})
    return _execution_context.set(merged)


def reset_execution_context(token: contextvars.Token) -> None:
    _execution_context.reset(token)

# 获取当前运行上下文
def get_execution_context() -> Dict[str, Any]:
    return dict(_execution_context.get())

# 通过 contextvars 注入的观察者触发事件
def _emit(method_name: str, data: Dict[str, Any]) -> None:
    observer = _current_observer.get()
    if observer is None:
        return
    try:
        # 调用观察者方法，传递数据
        getattr(observer, method_name)(data)
    except Exception:
        # Observability must never change Agent routing or business results.
        return


def emit_before_run(data: Dict[str, Any]) -> None:
    _emit("on_before_run", data)


def emit_after_plan(data: Dict[str, Any]) -> None:
    _emit("on_after_plan", data)


def emit_before_tool(data: Dict[str, Any]) -> None:
    _emit("on_before_tool", data)


def emit_after_tool(data: Dict[str, Any]) -> None:
    _emit("on_after_tool", data)


def emit_after_run(data: Dict[str, Any]) -> None:
    _emit("on_after_run", data)


def emit_error(data: Dict[str, Any]) -> None:
    _emit("on_error", data)
