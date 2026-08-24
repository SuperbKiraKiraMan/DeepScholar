"""
app/tools/citation_check_tool.py

CitationCheckTool —— 引用校验工具。

对报告中的每条引用做规则校验，检测 fake citation。

设计原则（文档明确要求）：
- citation correctness 优先用规则评估，不依赖 LLM
- 每项检查都是确定性的：有就是有，没有就是没有
- 检查项：ID 存在性、URL 匹配、quote 子串匹配

在 Agent 调用链中的位置：
Citation Worker -> CitationCheckTool -> Draft Reviewer -> Evaluator

为什么要有 CitationCheckTool：
- LLM 生成的报告可能包含不存在的引用编号（fake citation）
- LLM 可能把论文 A 的内容标注为论文 B 的引用（张冠李戴）
- LLM 可能编造看似合理的 URL（如虚构 arXiv ID）
- 学术调研系统的可信度取决于引用是否正确
"""

from typing import Any, Dict, List

from app.tools.base import BaseTool, ToolResult
from app.tools.evidence_extract_tool import traceable_source_text


class CitationCheckTool(BaseTool):
    """
    引用校验工具 —— 规则优先，不依赖 LLM。

    对每条 citation 执行三项规则检查：
    1. id_exists：引用编号是否在 source list 中存在
    2. url_matches_source：URL 是否能匹配到某个 source
    3. quote_found_in_source：quote 文本是否能在 source full_text 中找到

    综合判定：三项全部通过 → is_valid = True
    """

    @property
    def name(self) -> str:
        return "citation_check"

    @property
    def description(self) -> str:
        return (
            "Check whether citations in a report are valid: verify that each citation "
            "ID exists in the source list, that URLs match actual sources, and that "
            "quoted text can be found in source full_text. All rules are deterministic."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "description": "List of citations to check. Each must have id, url, quote.",
                    "items": {"type": "object"},
                },
                "sources": {
                    "type": "array",
                    "description": "List of sources to check against. Each must have source_id, url, full_text.",
                    "items": {"type": "object"},
                },
            },
            "required": ["citations", "sources"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        citations = kwargs.get("citations", [])
        sources = kwargs.get("sources", [])

        if not citations:
            return ToolResult(
                success=True,
                tool_name=self.name,
                data={
                    "results": [],
                    "total_checked": 0,
                    "valid_count": 0,
                    "invalid_count": 0,
                    "all_valid": True,
                    "message": "No citations to check",
                },
            )

        if not sources:
            # 没有任何来源时，所有引用都是无效的
            results = []
            for c in citations:
                results.append({
                    "citation_id": c.get("id", 0),
                    "source_id": c.get("source_id", ""),
                    "id_exists": False,
                    "url_matches_source": False,
                    "quote_found_in_source": False,
                    "is_valid": False,
                    "issues": ["No sources available to check against"],
                })

            return ToolResult(
                success=True,
                tool_name=self.name,
                data={
                    "results": results,
                    "total_checked": len(results),
                    "all_valid": False,
                    "valid_count": 0,
                    "invalid_count": len(results),
                },
            )

        # 构建索引：source_id → source（source_id 是主键）
        id_index: Dict[str, dict] = {}
        # 构建 URL 反向索引：url → source_id（辅助排查，不用于 source 定位）
        url_to_source_id: Dict[str, str] = {}
        for s in sources:
            sid = s.get("source_id", "")
            if sid:
                id_index[sid] = s
            url = s.get("url", "").strip().rstrip("/")
            if url and sid:
                url_to_source_id[url] = sid

        results = []
        for c in citations:
            citation_id = c.get("id", 0)
            source_id = c.get("source_id", "")
            url = c.get("url", "").strip().rstrip("/")
            quote = c.get("quote", "")

            issues = []

            # ---- 定位权威来源：source_id 是主键 ----
            # 正确顺序：
            #   1. 用 source_id 定位 matched_source
            #   2. 以 matched_source 的 url / full_text 为准做校验
            # 不能用 URL 反查来定位 source——URL 可能指向另一篇文章
            matched_source = id_index.get(source_id) if source_id else None

            # ---- Check 1: ID 存在性 ----
            id_exists = matched_source is not None
            if not id_exists:
                if source_id:
                    issues.append(
                        f"Citation [{citation_id}]: source_id '{source_id}' not found in source list"
                    )
                else:
                    issues.append(
                        f"Citation [{citation_id}]: missing source_id, cannot verify"
                    )

            # ---- Check 2: URL 匹配（以 matched_source 的 URL 为准） ----
            url_matches_source = False
            if matched_source is not None:
                source_url = matched_source.get("url", "").strip().rstrip("/")
                if url and source_url:
                    if url == source_url:
                        url_matches_source = True
                    else:
                        issues.append(
                            f"Citation [{citation_id}]: URL '{url}' does not match "
                            f"source '{source_id}' URL '{source_url}'"
                        )
                elif not url:
                    # 无 URL → URL 检查通过（不是所有引用都有 URL）
                    url_matches_source = True
                else:
                    # source 无 URL → 无法校验 URL
                    url_matches_source = True
            # 如果 matched_source is None，url_matches_source 保持 False

            # ---- Check 3: Quote 子串匹配（在 matched_source 的 full_text 中查找） ----
            quote_found_in_source = False
            if matched_source is not None:
                full_text = traceable_source_text(matched_source).lower()
                # 严格匹配：检查 quote 是否直接在 full_text 中
                if quote and quote.lower() in full_text:
                    quote_found_in_source = True
                elif quote:
                    # 宽松匹配：检查 quote 中是否有连续的 50 个字符片段在 full_text 中
                    if len(quote) >= 50:
                        for i in range(0, len(quote) - 50, 25):
                            fragment = quote[i:i + 50].lower()
                            if fragment in full_text:
                                quote_found_in_source = True
                                break
                    if not quote_found_in_source:
                        issues.append(
                            f"Citation [{citation_id}]: quote not found in source '{source_id}' full_text"
                        )
                else:
                    # 无 quote → 通过
                    quote_found_in_source = True
            # 如果 matched_source is None，quote_found_in_source 保持 False

            # ---- 综合判定 ----
            is_valid = id_exists and url_matches_source and quote_found_in_source

            results.append({
                "citation_id": citation_id,
                "source_id": source_id,
                "id_exists": id_exists,
                "url_matches_source": url_matches_source,
                "quote_found_in_source": quote_found_in_source,
                "is_valid": is_valid,
                "issues": issues,
            })

        valid_count = sum(1 for r in results if r["is_valid"])

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "results": results,
                "total_checked": len(results),
                "all_valid": valid_count == len(results),
                "valid_count": valid_count,
                "invalid_count": len(results) - valid_count,
            },
        )
