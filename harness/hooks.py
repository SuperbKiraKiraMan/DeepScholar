"""
harness/hooks.py

HookBus — 运行作用域的生命周期钩子总线，实现 ObserverProtocol 观察者协议。
通过 contextvar 注入调用 app/ 的生命周期节点。

核心设计:
- 每次 emit 必然创建一条 HookRecord 事件记录（即使没有注册任何 callback）
- callback 异常会被隔离捕获，转为 Harness 警告，不会中断主流程
- 敏感数据（API key、token、长文本）在记录前统一脱敏处理
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional

from harness.models import HookRecord


Callback = Callable[[str, Dict[str, Any]], None]

# 敏感字段名匹配模式 —— 命中则整个字段值替换为 [REDACTED]
_SENSITIVE_KEY_PATTERNS = re.compile(
    r'(api[_-]?key|deepseek_api_key|openalex_api_key|authorization'
    r'|x-api-key|access_token|secret|password|bearer|full[_-]?text'
    r'|chain[_-]?of[_-]?thought|reasoning[_-]?content)',
    re.IGNORECASE,
)
# 敏感值匹配模式 —— 命中则替换对应的子串为 [REDACTED_KEY]
_SENSITIVE_VALUE_PATTERN = re.compile(
    r'(sk-[a-zA-Z0-9_-]{10,}|Bearer\s+[a-zA-Z0-9_\-\.]+)',
    re.IGNORECASE,
)


class HookBus:
    """
    运行作用域生命周期事件总线，实现 ObserverProtocol。

    设计要点:
    - 每次 emit 必然创建一条事件记录，即便没有注册任何 callback
    - callback 抛出的异常会被隔离为 Harness 警告，不影响其他 callback 和主流程
    - 事件数据在记录前统一脱敏
    """

    def __init__(self):
        # 六种生命周期钩子，每种可注册多个 callback
        self._hooks: Dict[str, List[Callback]] = {
            "before_run": [],       # 运行开始前
            "after_plan": [],       # 规划完成后
            "before_tool": [],      # 工具调用前
            "after_tool": [],       # 工具调用后
            "after_run": [],        # 运行结束后
            "on_error": [],         # 发生错误时
        }
        self.records: List[HookRecord] = []   # 累积的事件记录
        self.warnings: List[str] = []          # 累积的警告信息

    # ---- ObserverProtocol 实现 ----
    # 每个 on_xxx 方法对应一个生命周期事件，统一委托给 emit()

    def on_before_run(self, data: Dict[str, Any]) -> None:
        """运行开始前触发"""
        self.emit("before_run", data)

    def on_after_plan(self, data: Dict[str, Any]) -> None:
        """规划完成后触发"""
        self.emit("after_plan", data)

    def on_before_tool(self, data: Dict[str, Any]) -> None:
        """工具调用前触发"""
        self.emit("before_tool", data)

    def on_after_tool(self, data: Dict[str, Any]) -> None:
        """工具调用后触发"""
        self.emit("after_tool", data)

    def on_after_run(self, data: Dict[str, Any]) -> None:
        """运行结束后触发"""
        self.emit("after_run", data)

    def on_error(self, data: Dict[str, Any]) -> None:
        """发生错误时触发"""
        self.emit("on_error", data)

    # ---- 注册 / 注销 ----

    def on(self, hook_name: str, callback: Callback): # 追加 callback
        """向指定钩子注册一个 callback 函数"""
        if hook_name in self._hooks:
            self._hooks[hook_name].append(callback)

    def clear(self): # 清空所有 callback
        """移除所有已注册的 callback"""
        for k in self._hooks:
            self._hooks[k].clear()

    # ---- 事件触发 ----

    def emit(self, hook_name: str, data: Dict[str, Any]) -> List[HookRecord]:
        """
        触发一个钩子事件。

        流程:
        1. 对原始数据做脱敏处理
        2. 创建一条 HookRecord 事件记录（与 callback 数量解耦，保证 hook_summary 统计不受 callback 数量影响）
        3. 按注册顺序依次调用所有 callback，每个 callback 异常独立捕获
        4. 若有 callback 失败，标记 record 为失败并附加错误信息
        """
        if hook_name not in self._hooks:
            return []

        safe_data = _sanitize(data) # 脱敏：去掉 API key、token、长文本截断

        # 每个生命周期事件只产生一条 HookRecord，callback 数量不影响记录数
        record = HookRecord( # 每个事件一条记录(与callback数量解耦)
            hook_name=hook_name,
            stage=hook_name,
            timestamp_ms=int(time.time() * 1000),
            data=safe_data,
            success=True,
        )
        callback_errors = []
        for cb in self._hooks[hook_name]:
            try:
                cb(hook_name, safe_data)
            except Exception as e:
                # 隔离捕获：单个 callback 异常不影响其他 callback 执行
                error = f"{type(e).__name__}: {str(e)[:200]}"
                callback_errors.append(error)
                self.warnings.append(f"[hook:{hook_name}] {error}")

        if callback_errors:
            record.success = False
            record.error = "; ".join(callback_errors)
        self.records.append(record)
        return [record]


def _sanitize(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归脱敏处理，移除敏感字段和敏感值。

    脱敏规则:
    1. 字典: 若 key 命中 _SENSITIVE_KEY_PATTERNS → 值替换为 "[REDACTED]"
    2. 字符串: 长度 > 1000 → 截断为前 200 字符 + "...[truncated]"
               值命中 _SENSITIVE_VALUE_PATTERN → 替换为 "[REDACTED_KEY]"
    3. 列表/元组: 递归处理每个元素
    4. 其他类型: 原样返回
    """
    if isinstance(data, dict):
        safe = {}
        for k, v in data.items():
            if _SENSITIVE_KEY_PATTERNS.search(str(k)):
                safe[k] = "[REDACTED]"
                continue
            safe[k] = _sanitize(v)
        return safe
    elif isinstance(data, (list, tuple)):
        return [_sanitize(v) for v in data]
    elif isinstance(data, str):
        if len(data) > 1000:
            data = data[:200] + "...[truncated]"
        data = _SENSITIVE_VALUE_PATTERN.sub("[REDACTED_KEY]", data)
        return data
    else:
        return data
