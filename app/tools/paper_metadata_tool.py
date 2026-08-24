"""
app/tools/paper_metadata_tool.py

PaperMetadataTool —— 论文元数据标准化工具。

对搜索返回的来源进行元数据标准化和补全：
1. 校验必填字段（title, url）
2. 补全缺失字段的默认值
3. 规范化 source_type 为统一枚举值
4. 标准化作者列表格式

在 Agent 调用链中的位置：
Search Worker -> AcademicSearchTool -> PaperMetadataTool -> SourceQualityScorer

"""

from typing import Any, Dict, List

from app.tools.base import BaseTool, ToolResult

# 合法的 source_type 枚举
VALID_SOURCE_TYPES = {
    "paper", "book", "benchmark", "dataset", "tool", "blog", "report", "other", "unknown",
}


class PaperMetadataTool(BaseTool):
    """
    论文元数据标准化工具。

    职责：
    1. 确保每条 source 有完整的元数据字段
    2. 对缺失字段补默认值
    3. 对不合法字段做规范化
    4. 不改变 source_id（不可变标识）

    """

    @property
    def name(self) -> str:
        return "paper_metadata"

    @property
    def description(self) -> str:
        return (
            "Normalize and validate paper metadata: ensure required fields exist, "
            "fill defaults for missing fields, normalize source_type values, and "
            "standardize author list format."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "description": "List of paper sources to normalize",
                    "items": {"type": "object"},
                },
            },
            "required": ["sources"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        """标准化论文元数据。"""
        sources = kwargs.get("sources", [])

        if not sources:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="No sources provided for metadata normalization",
            )

        normalized = []
        warnings = []

        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                warnings.append(f"Source at index {i} is not a dict, skipping")
                continue

            # 校验必填字段
            title = source.get("title", "").strip()
            url = source.get("url", "").strip()

            if not title:
                warnings.append(f"Source at index {i} has no title, skipping")
                continue
            if not url:
                warnings.append(f"Source '{title[:50]}' has no url, marking as incomplete")

            # 标准化 source_type
            source_type = source.get("source_type", "unknown")
            if source_type not in VALID_SOURCE_TYPES:
                warnings.append(
                    f"Source '{title[:50]}' has unknown source_type '{source_type}', "
                    f"normalized to 'unknown'"
                )
                source_type = "unknown"

            # 标准化作者格式：确保是 list[str]
            authors = source.get("authors", [])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",") if a.strip()]
            if not isinstance(authors, list):
                authors = []

            # 补全默认值
            # Preserve provider-specific provenance (DOI, OpenAlex ID, OA/content
            # metadata) while normalizing the shared PaperSource contract.
            normalized_source = {
                **source,
                "source_id": source.get("source_id", f"unknown_{i}"),
                "title": title,
                "url": url,
                "snippet": source.get("snippet", ""),
                "full_text": source.get("full_text", ""),
                "authors": authors,
                "year": source.get("year"),
                "venue": source.get("venue", ""),
                "source_type": source_type,
            }
            normalized.append(normalized_source)

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "sources": normalized,
                "total": len(normalized),
                "warnings": warnings,
            },
            metadata={"warnings": warnings},
        )
