"""
app/tools/academic_search_tool.py

AcademicSearchTool：统一学术搜索工具（provider-neutral）。

Agent 只依赖 "academic_search" 这个名字，不感知底层是 Mock 还是 OpenAlex。
Provider 选择通过 SEARCH_PROVIDER 环境变量控制：

- "mock" | "academic" → MockAcademicSearchProvider（离线，7 篇 mock 论文）
- "openalex" → OpenAlexSearchProvider（真实 OpenAlex API）
- "tavily" → 预留

设计原则：
1. 工具名固定为 "academic_search"，Agent 代码中不出现 provider 名
2. Provider 懒加载（每次 _arun 根据环境变量选择，支持 monkeypatch）
3. max_results 限制 1~50
4. query 由 LLM 传入，API key 只能从环境变量读取

"""

from typing import Any, Dict

from app.core.config import get_search_provider
from app.tools.base import BaseTool, ToolResult
from app.tools.academic_search_provider import (
    MockAcademicSearchProvider,
    OpenAlexSearchProvider,
)


class AcademicSearchTool(BaseTool):
    """
    统一学术搜索工具。

    Provider 选择在 _arun 时根据 SEARCH_PROVIDER 决定，
    支持测试 monkeypatch 切换 Provider。

    """

    @property
    def name(self) -> str:
        return "academic_search"

    @property
    def description(self) -> str:
        return (
            "Provider-neutral external academic discovery, backed by OpenAlex in "
            "production and a mock provider in offline tests. Best for recent work, "
            "broader topic coverage, papers, benchmarks, and datasets. Results always "
            "carry paper metadata; only records with a traceable abstract, snippet, or "
            "full text are eligible for evidence."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research topic or search query, e.g. 'RAG evaluation methods'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-50, default 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                },
                "year_from": {
                    "type": "integer",
                    "description": "Optional inclusive minimum publication year",
                    "minimum": 1800,
                    "maximum": 2100,
                },
                "year_to": {
                    "type": "integer",
                    "description": "Optional inclusive maximum publication year",
                    "minimum": 1800,
                    "maximum": 2100,
                },
            },
            "required": ["query"],
        }

    def _get_provider(self):
        """根据 SEARCH_PROVIDER 懒加载对应 Provider。"""
        provider_name = get_search_provider().strip().lower()

        if provider_name == "openalex":
            return OpenAlexSearchProvider()
        if provider_name in ("mock", "academic"):
            return MockAcademicSearchProvider()
        raise ValueError(
            f"Unsupported SEARCH_PROVIDER '{provider_name}'. "
            "Expected one of: academic, mock, openalex"
        )

    async def _arun(self, **kwargs) -> ToolResult:
        """执行学术搜索。"""
        query = kwargs.get("query", "").strip()
        max_results = kwargs.get("max_results", 5)
        year_from = kwargs.get("year_from")
        year_to = kwargs.get("year_to")

        # 限制 max_results 在 1-50
        max_results = max(1, min(int(max_results), 50))
        year_from = int(year_from) if year_from not in (None, "") else None
        year_to = int(year_to) if year_to not in (None, "") else None
        if year_from is not None and year_to is not None and year_from > year_to:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="year_from must be less than or equal to year_to",
            )

        provider = self._get_provider()
        result = await provider.search(
            query=query,
            max_results=max_results,
            year_from=year_from,
            year_to=year_to,
        )

        # 统一 tool_name
        result.tool_name = self.name
        return result
