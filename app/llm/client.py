"""
app/llm/client.py

LLM 客户端抽象层，集成 DeepSeek V4 Flash。

核心设计：
- 异步 HTTP 调用 DeepSeek API
- 超时 + 单次重试
- 通过 Pydantic 校验结构化 JSON 输出
- 任何失败时自动降级到基于规则的实现
- 永不记录 API key
- FakeLLMClient 用于离线测试
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from app.llm.schemas import LLMConfig
from app.tools.http_client import build_httpx_client


# ================================================================
# 配置加载
# ================================================================

def load_llm_config() -> LLMConfig:
    """从环境变量加载 LLM 配置。"""
    return LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "deepseek"),
        agent_mode=os.getenv("AGENT_MODE", "llm"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "8")),
    )


# ================================================================
# LLM 客户端接口
# ================================================================

class LLMClient:
    """
    DeepSeek V4 Flash 客户端，支持结构化输出和降级方案。

    用法：
        client = LLMClient(config)
        try:
            result = await client.generate_structured(system_prompt, user_prompt, schema)
        except Exception:
            # 降级到基于规则的实现
            result = fallback_implementation()
    """

    def __init__(self, config: LLMConfig = None):
        self.config = config or load_llm_config()

    @property
    def is_available(self) -> bool:
        """检查 LLM 是否已配置且可用。"""
        return (
            self.config.agent_mode == "llm"
            and bool(self.config.api_key)
            and bool(self.config.api_key.strip())
        )

    @property
    def mode(self) -> str:
        """返回当前有效模式（llm 或 rule）。"""
        if self.is_available:
            return "llm"
        return "rule"

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
        temperature: float = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        生成结构化 JSON 输出，并通过 Pydantic schema 校验。

        返回：
            {"success": True, "data": <已校验的 pydantic 对象>, "latency_ms": ..., "model": ...}
            或
            {"success": False, "error": "...", "raw_text": "..."}
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "LLM not available (agent_mode=rule or no API key)",
            }

        temp = temperature if temperature is not None else self.config.temperature
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds
        )
        effective_retries = (
            max_retries if max_retries is not None else self.config.max_retries
        )
        start_ms = int(time.time() * 1000)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(effective_retries + 1):
            try:
                async with build_httpx_client(
                    timeout=effective_timeout,
                ) as client:
                    response = await client.post(
                        f"{self.config.base_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    body = response.json()

                    latency_ms = int(time.time() * 1000) - start_ms
                    usage = body.get("usage", {})
                    raw_text = body["choices"][0]["message"]["content"]

                    # 解析 JSON
                    try:
                        parsed = json.loads(raw_text)
                    except json.JSONDecodeError as e:
                        return {
                            "success": False,
                            "error": f"JSON parse error: {e}",
                            "raw_text": raw_text[:500],
                            "latency_ms": latency_ms,
                        }

                    # 通过 Pydantic schema 校验
                    try:
                        validated = output_schema(**parsed)
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Schema validation error: {e}",
                            "raw_text": raw_text[:500],
                            "latency_ms": latency_ms,
                        }

                    return {
                        "success": True,
                        "data": validated,
                        "latency_ms": latency_ms,
                        "model": self.config.model,
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    }

            except httpx.TimeoutException as e:
                last_error = f"Timeout after {effective_timeout}s"
                if attempt >= effective_retries:
                    break
            except httpx.HTTPStatusError as e:
                response_detail = (e.response.text or "")[:300]
                last_error = f"HTTP {e.response.status_code}: {response_detail}"
                # 4xx 错误（认证错误、请求错误）不重试
                if 400 <= e.response.status_code < 500:
                    break
            except Exception as e:
                last_error = str(e)[:200]
                if attempt >= effective_retries:
                    break

        latency_ms = int(time.time() * 1000) - start_ms
        return {
            "success": False,
            "error": last_error or "Unknown error",
            "latency_ms": latency_ms,
        }

    async def classify_intent(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
    ) -> Dict[str, Any]:
        """Separate semantic entry point so tests can isolate Controller responses."""
        return await self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
            temperature=0.0,
        )

    async def function_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        标准 OpenAI-compatible Function Calling。

        Parameters:
        - messages: 完整对话历史 [{"role": "system"|"user"|"assistant"|"tool", ...}]
        - tools: OpenAI format tool schemas
        - tool_choice: "auto" | "none" | {"type": "function", "function": {"name": "x"}}

        Returns:
            {"success": True, "finish": False, "tool_calls": [...], "latency_ms": ..., "usage": ...}
            {"success": True, "finish": True, "content": "...", "latency_ms": ...}
            {"success": False, "error": "...", "latency_ms": ...}
        """
        # ---- 门禁：LLM 不可用直接返回 ----
        if not self.is_available:
            return {"success": False, "error": "LLM not available", "latency_ms": 0}

        temp = temperature if temperature is not None else self.config.temperature
        start_ms = int(time.time() * 1000)

        # ---- 拼装请求体：messages(对话历史) + tools(可用工具定义) + tool_choice(auto=模型自己决定) ----
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temp,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        # ---- 网络层重试：同一个请求超时/5xx 才重试，4xx 不重试 ----
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with build_httpx_client(
                    timeout=self.config.timeout_seconds,
                ) as client:
                    # ---- 发请求到 DeepSeek API ----
                    response = await client.post(
                        f"{self.config.base_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    body = response.json()

                    latency_ms = int(time.time() * 1000) - start_ms
                    usage = body.get("usage", {})
                    choice = body["choices"][0]
                    message = choice["message"]

                    # ---- 模型返回了 tool_calls → 它想调工具 ----
                    tool_calls = message.get("tool_calls", [])
                    if tool_calls:
                        return {
                            "success": True,
                            "finish": False,       # ← 还没完，调用方要继续循环
                            # 标准化 tool_calls: arguments 可能是 JSON 字符串，统一 parse 成 dict
                            "tool_calls": [
                                {
                                    "id": tc["id"],
                                    "name": tc["function"]["name"],
                                    "arguments": json.loads(tc["function"]["arguments"])
                                    if isinstance(tc["function"]["arguments"], str)
                                    else tc["function"]["arguments"],
                                }
                                for tc in tool_calls
                            ],
                            "latency_ms": latency_ms,
                            "model": self.config.model,
                            "usage": {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            },
                        }

                    # ---- 模型没返回 tool_calls → 任务完成 ----
                    content = message.get("content", "")
                    return {
                        "success": True,
                        "finish": True,           # ← 告诉调用方：结束了
                        "content": content,
                        "latency_ms": latency_ms,
                        "model": self.config.model,
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    }

            # ---- 网络层异常处理：超时可重试，4xx 不重试(认证/参数错误重试没用) ----
            except httpx.TimeoutException:
                last_error = f"Timeout after {self.config.timeout_seconds}s"
                if attempt >= self.config.max_retries:
                    break
            except httpx.HTTPStatusError as e:
                response_detail = (e.response.text or "")[:300]
                last_error = f"HTTP {e.response.status_code}: {response_detail}"
                # 4xx = 客户端错误(api key 无效、参数不对)，重试无意义
                if 400 <= e.response.status_code < 500:
                    break
            except Exception as e:
                last_error = str(e)[:200]
                if attempt >= self.config.max_retries:
                    break

        # ---- 所有重试都失败 → 返回失败结果，调用方会回退规则版 ----
        latency_ms = int(time.time() * 1000) - start_ms
        return {"success": False, "error": last_error or "Unknown error", "latency_ms": latency_ms}

    def build_trace_event(
        self, event: str, success: bool, latency_ms: int = 0, **kwargs
    ) -> Dict[str, Any]:
        """构建 LLM 操作的 trace 事件（排除 API key）。"""
        entry = {
            "event": event,
            "timestamp_ms": int(time.time() * 1000),
            "success": success,
            "model": self.config.model if self.is_available else "rule",
            "mode": self.mode,
            **kwargs,
        }
        if latency_ms:
            entry["latency_ms"] = latency_ms
        return entry


# ================================================================
# 用于测试的 FakeLLMClient
# ================================================================

class FakeLLMClient(LLMClient):
    """
    用于离线测试的 LLM 假客户端。

    返回预定义的结构化响应。
    支持 generate_structured() 和 function_call()。
    永不发起网络调用。

    分离 structured responses 和 FC responses。
    - set_responses() 设置 structured (Planner/Reviewer) 响应
    - set_fc_responses() 设置 Function Calling (Worker) 响应
    - 两个通道互不干扰，避免 Planner 消耗 Worker 的模拟响应
    """

    def __init__(self, responses: List[Dict[str, Any]] = None):
        super().__init__(LLMConfig(agent_mode="llm", api_key="fake-test-key"))
        self._structured_responses: List[Dict[str, Any]] = responses or []
        self._fc_responses: List[Dict[str, Any]] = []
        self._intent_responses: List[Dict[str, Any]] = []
        self._structured_index = 0
        self._fc_index = 0
        self.calls: List[Dict[str, Any]] = []
        self.fc_calls: List[Dict[str, Any]] = []
        self.intent_calls: List[Dict[str, Any]] = []

    @property
    def is_available(self) -> bool:
        return True

    @property
    def mode(self) -> str:
        return "llm"

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
        temperature: float = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """返回下一个假结构化响应，并通过 schema 校验。使用独立的 structured 通道。"""
        self.calls.append({
            "system_prompt": system_prompt[:200],
            "user_prompt": user_prompt[:200],
            "schema": output_schema.__name__,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
        })

        if self._structured_index >= len(self._structured_responses):
            self._structured_index += 1
            return {
                "success": False,
                "error": "FakeLLMClient: no more predefined structured responses",
            }

        raw = self._structured_responses[self._structured_index]
        self._structured_index += 1

        if raw.get("_fail", False):
            return {
                "success": False,
                "error": raw.get("_error", "FakeLLMClient: simulated failure"),
                "raw_text": raw.get("_raw_text", ""),
            }

        try:
            validated = output_schema(**raw)
            return {
                "success": True,
                "data": validated,
                "latency_ms": 5,
                "model": "fake-deepseek-v4-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"FakeLLMClient: schema validation: {e}",
            }

    def set_responses(self, responses: List[Dict[str, Any]]):
        """设置 structured 响应列表（用于 Planner/Reviewer）。"""
        self._structured_responses = responses
        self._structured_index = 0
        self.calls = []

    def set_fc_responses(self, responses: List[Dict[str, Any]]):
        """
        设置 Function Calling 响应列表（用于 Worker）。

        每个元素格式：
        - {"_finish": False, "tool_calls": [{"id": "call_1", "name": "academic_search",
            "arguments": {"query": "test", "max_results": 5}}]}
        - {"_finish": True, "content": "done"}
        - {"_fail": True, "_error": "timeout"}
        """
        self._fc_responses = responses
        self._fc_index = 0
        self.fc_calls = []

    def set_intent_responses(self, responses: List[Dict[str, Any]]):
        """Set Controller-only responses without consuming Planner/Reviewer fixtures."""
        self._intent_responses = list(responses)
        self.intent_calls = []

    async def classify_intent(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
    ) -> Dict[str, Any]:
        self.intent_calls.append({"user_prompt": user_prompt[:200]})
        if not self._intent_responses:
            return {"success": False, "error": "FakeLLMClient: no intent response"}
        raw = self._intent_responses.pop(0)
        if raw.get("_fail"):
            return {"success": False, "error": raw.get("_error", "simulated intent failure")}
        try:
            return {
                "success": True,
                "data": output_schema(**raw),
                "latency_ms": 2,
                "model": "fake-deepseek-v4-flash",
                "usage": {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16},
            }
        except Exception as exc:
            return {"success": False, "error": f"Fake intent schema validation: {exc}"}

    async def function_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = None,
    ) -> Dict[str, Any]:
        """返回下一个假 Function Calling 响应。使用独立的 FC 通道。"""
        self.fc_calls.append({
            "message_count": len(messages),
            "tool_count": len(tools),
        })

        if self._fc_index >= len(self._fc_responses):
            self._fc_index += 1
            return {
                "success": False,
                "error": "FakeLLMClient: no more fc responses",
                "latency_ms": 1,
            }

        raw = self._fc_responses[self._fc_index]
        self._fc_index += 1

        if raw.get("_fail", False):
            return {
                "success": False,
                "error": raw.get("_error", "FakeLLMClient: simulated failure"),
                "latency_ms": 1,
            }

        if raw.get("_finish", False):
            return {
                "success": True,
                "finish": True,
                "content": raw.get("content", ""),
                "latency_ms": 5,
                "model": "fake-deepseek-v4-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        tool_calls = raw.get("tool_calls", [])
        return {
            "success": True,
            "finish": False,
            "tool_calls": tool_calls,
            "latency_ms": 5,
            "model": "fake-deepseek-v4-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }


# ================================================================
# 全局客户端工厂
# ================================================================

_global_client: Optional[LLMClient] = None
_global_config: Optional[LLMConfig] = None


def get_llm_client() -> LLMClient:
    """获取或创建全局 LLM 客户端实例。"""
    global _global_client
    if _global_client is None:
        _global_client = LLMClient()
    return _global_client


def get_llm_config() -> LLMConfig:
    """获取或创建全局 LLM 配置。"""
    global _global_config
    if _global_config is None:
        _global_config = load_llm_config()
    return _global_config


def reset_llm_client():
    """重置全局 LLM 客户端（测试时有用）。"""
    global _global_client, _global_config
    _global_client = None
    _global_config = None
