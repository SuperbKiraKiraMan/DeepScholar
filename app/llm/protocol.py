"""Hybrid Agent 可注入的最小 LLM 协议。"""

from typing import Any, Dict, Protocol


class StructuredLLMClient(Protocol):
    """生产客户端与 Fake 客户端共享的结构化生成接口。"""

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
        temperature: float | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ) -> Dict[str, Any]: ...
