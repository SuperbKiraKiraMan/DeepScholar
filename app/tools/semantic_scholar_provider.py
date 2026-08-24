"""
================================================================================
Semantic Scholar API 客户端 —— HTTP 层 + 数据归一化
================================================================================

【本文件的定位】
  本文件是 Semantic Scholar 三个工具（Search / Recommendations / Graph）的
  "底层发动机"。上层工具类只管定义接口形态（name / description / schema），
  真正的 HTTP 请求、重试策略、数据格式转换全部在本文件完成。

【架构分层 —— 自顶向下】
  semantic_scholar_tools.py            ← 工具定义层（Agent 看到的接口）
      │
  semantic_scholar_provider.py  ← 你在这里（HTTP 客户端 + 数据适配）
      │
  Semantic Scholar Academic Graph API  ← 远程服务 (api.semanticscholar.org)

【核心职责】
  1. HTTP 请求：带超时、API key、重试机制地调用 Semantic Scholar API
  2. 数据归一化：把 API 返回的原始 JSON → 项目统一的 PaperSource 字典
  3. 智能解析：自动识别论文标识符（DOI / arXiv / CorpusId / paperId）
  4. 错误处理：区分可重试错误（429/5xx）和不可重试错误（400/401/403/404）

【设计原则】
  - Agent 层绝不接触原始 API 响应 → 全部通过 semantic_scholar_paper_to_source() 归一化
  - 所有失败路径用 ToolResult(success=False) 表示，绝不抛异常
  - 重试有"预算"机制（max_retries + max_retry_wait），避免无限重试
"""

# ============================================================================
# 阶段一：导入与常量定义
# ============================================================================

import asyncio
from email.utils import parsedate_to_datetime   # 解析 HTTP Retry-After 头中的日期格式
import random
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote                   # URL 编码（处理 paperId 中的特殊字符）

import httpx                                      # 异步 HTTP 客户端（比 aiohttp 更现代）

from app.core.config import (
    get_semantic_scholar_api_key,
    get_semantic_scholar_graph_base_url,
    get_semantic_scholar_max_retries,
    get_semantic_scholar_max_retry_wait,
    get_semantic_scholar_rate_limit_retries,
    get_semantic_scholar_recommendations_base_url,
    get_semantic_scholar_timeout,
)
from app.tools.base import ToolResult
from app.tools.http_client import build_httpx_client


# ---------------------------------------------------------------------------
# PAPER_FIELDS: 请求 Semantic Scholar API 时要求返回的字段列表
# ---------------------------------------------------------------------------
# Semantic Scholar 的 API 支持 fields 参数，只返回你需要的字段，减少响应体大小。
# 这里列出了我们关心的所有字段，每次请求都会带上。
PAPER_FIELDS = ",".join([
    "paperId",           # S2 内部唯一 ID（40 位十六进制）
    "corpusId",          # 数字 ID，和 paperId 一一对应
    "title",             # 论文标题
    "abstract",          # 摘要全文
    "url",               # S2 上的论文页面 URL
    "year",              # 发表年份
    "venue",             # 发表期刊/会议名称
    "authors",           # 作者列表（含 name 和 authorId）
    "externalIds",       # 外部标识符集合（DOI、arXiv、PubMed 等）
    "citationCount",     # 被引次数（反映论文影响力）
    "referenceCount",    # 参考文献数量
    "openAccessPdf",     # 开放获取 PDF 信息（如果有免费版本）
    "publicationTypes",  # 发表类型（如 "Review", "JournalArticle" 等）
    "publicationDate",   # 具体发表日期
    "tldr",              # TL;DR 自动摘要（S2 模型生成的短摘要）
])

# recommendations / citations / references 端点不接受 tldr 字段（会返回 HTTP 400），
# 但 paper search 和 details 端点支持 tldr。因此单独构建一个不含 tldr 的字段集，
# 供 recommendations 和 citations/references 请求使用，避免合法请求被拒绝。
RELATIONSHIP_PAPER_FIELDS = ",".join(
    field for field in PAPER_FIELDS.split(",") if field != "tldr"
)

# ---------------------------------------------------------------------------
# HTTP 状态码分类 —— 决定重试策略的关键
# ---------------------------------------------------------------------------
# 可重试状态码：临时性错误，等一会儿再试可能就成功了
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
#  429: Rate Limit Exceeded（触发限流，需要等 Retry-After 秒）
#  500: Internal Server Error（服务器临时故障）
#  502: Bad Gateway
#  503: Service Unavailable
#  504: Gateway Timeout

# 不可重试状态码：客户端错误，重试多少次都没用
NON_RETRYABLE_STATUS = {400, 401, 403, 404}
#  400: Bad Request（请求参数有问题）
#  401: Unauthorized（API key 无效）
#  403: Forbidden（无权限访问）
#  404: Not Found（论文或端点不存在）


# ============================================================================
# 阶段二：数据归一化 —— 原始 API 响应 → 项目统一格式
# ============================================================================

def semantic_scholar_paper_to_source(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    【核心转换函数】把 Semantic Scholar API 返回的原始论文对象 → 项目统一的
    PaperSource 字典格式。

    【为什么需要这个函数？】
      不同的学术数据源（arXiv、Semantic Scholar、PubMed...）返回的数据结构
      各不相同。通过这个转换函数，让上层 Agent 看到的永远是同一套字段名，
      不关心数据来自哪里。

    【转换示例】
      输入 (S2 原始格式):
        {"paperId": "abc123", "title": "Attention Is All You Need",
         "externalIds": {"DOI": "10.xxx/yyy"}, "citationCount": 95000, ...}

      输出 (项目统一格式):
        {"source_id": "s2:abc123", "title": "Attention Is All You Need",
         "doi": "10.xxx/yyy", "cited_by_count": 95000,
         "provider": "semantic_scholar", ...}

    【关键字段说明】
      - source_id:      格式为 "s2:{paperId}"，确保不同 provider 的 ID 不冲突
      - content_source: 标记摘要来源（"abstract" / "tldr"），方便调试
      - is_oa / oa_status: 是否可免费获取
    """

    # ---- 解析论文 ID 和外部标识符 ----
    paper_id = str(paper.get("paperId") or "").strip()
    external_ids = paper.get("externalIds") or {}
    doi = str(external_ids.get("DOI") or "").strip() or None

    # ---- 构建规范 URL（优先级: S2 原生 URL > DOI 链接 > S2 页面） ----
    open_pdf = paper.get("openAccessPdf") or {}
    canonical_url = str(paper.get("url") or "").strip()
    if not canonical_url and doi:
        canonical_url = f"https://doi.org/{doi}"
    if not canonical_url and paper_id:
        canonical_url = f"https://www.semanticscholar.org/paper/{paper_id}"

    # ---- 解析摘要（优先完整 abstract，其次 TL;DR 短摘要） ----
    abstract = str(paper.get("abstract") or "").strip()
    tldr = paper.get("tldr") or {}
    tldr_text = str(tldr.get("text") or "").strip() if isinstance(tldr, dict) else ""
    snippet = abstract or tldr_text   # 用于列表展示的简短文本

    # ---- 解析发表类型 ----
    publication_types = paper.get("publicationTypes") or []
    source_type = "paper"
    if any(str(item).lower() == "review" for item in publication_types):
        source_type = "paper"

    # ---- 解析作者列表 ----
    authors = []
    for author in paper.get("authors") or []:
        if isinstance(author, dict) and author.get("name"):
            authors.append(str(author["name"]))

    # ---- 解析 PDF 信息 ----
    pdf_url = str(open_pdf.get("url") or "").strip() if isinstance(open_pdf, dict) else ""

    # ---- 构建统一格式的输出字典 ----
    return {
        "source_id": f"s2:{paper_id}" if paper_id else "",
        "title": str(paper.get("title") or "").strip() or "Untitled paper",
        "url": canonical_url,
        "snippet": snippet,
        "full_text": abstract,                          # 完整摘要
        "authors": authors,
        "year": paper.get("year"),
        "venue": str(paper.get("venue") or "").strip(),
        "source_type": source_type,
        "provider": "semantic_scholar",                 # 标记数据来源
        "semantic_scholar_id": paper_id or None,
        "corpus_id": paper.get("corpusId"),
        "doi": doi,
        "cited_by_count": paper.get("citationCount"),   # 被引次数
        "reference_count": paper.get("referenceCount"), # 参考文献数
        "is_oa": bool(pdf_url),                         # 是否开放获取
        "oa_status": "open" if pdf_url else None,
        "content_url": pdf_url or None,
        # 标记摘要来自哪里：完整 abstract 还是 TL;DR
        "content_source": "abstract" if abstract else ("tldr" if tldr_text else None),
        "publication_date": paper.get("publicationDate"),
    }


# ============================================================================
# 阶段三：SemanticScholarClient —— 核心 HTTP 客户端
# ============================================================================

class SemanticScholarClient:
    """
    【核心类】Semantic Scholar API 的异步 HTTP 客户端。

    【职责】
      1. 封装三个业务 API：search（搜索）、recommend（推荐）、paper_graph（图谱）
      2. 统一的 HTTP 请求 + 重试策略（指数退避 + Retry-After 头解析）
      3. 智能解析论文标识符（DOI/arXiv/CorpusId/paperId）

    【为什么用 httpx 而不是 requests？】
      上层 BaseTool.run() → _arun() 是异步的（async/await）。
      如果用同步的 requests 库会阻塞事件循环，所以用 httpx（支持 asyncio）。

    【重试策略概览】
      有两种重试：
        1. 临时故障重试（Timeout/5xx） → 指数退避，最多 max_retries 次
        2. 限流重试（429）            → 优先用 Retry-After 头，最多 rate_limit_retries 次
      两者共享一个"重试等待预算"（max_retry_wait），防止无限重试。
    """

    # ------------------------------------------------------------------
    # 3.1 搜索论文 —— 全文关键词搜索
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int = 5) -> ToolResult:
        """
        【功能】通过关键词搜索学术论文。

        【调用示例】
          await client.search("transformer attention mechanism", max_results=10)

        【API 端点】
          GET {graph_base}/paper/search?query=...&limit=...&fields=...

        【返回】
          ToolResult.data = {
            "results": [...],       # 归一化后的论文列表
            "query": "...",         # 实际使用的搜索词
            "total_found": 1234,    # 搜索命中的总数
            "provider": "semantic_scholar"
          }
        """
        normalized_query = _normalize_search_query(query)
        if not normalized_query:
            return _failure("semantic_scholar_search", "Search query is empty")

        # 限制 result 数在 1-50 范围内
        limit = max(1, min(int(max_results), 50))

        response = await self._request_json(
            "GET",
            f"{get_semantic_scholar_graph_base_url()}/paper/search",
            params={"query": normalized_query, "limit": limit, "fields": PAPER_FIELDS},
        )

        if not response["success"]:
            return _failure("semantic_scholar_search", response["error"], response.get("metadata"))

        papers = response["data"].get("data") or []
        sources = _normalize_sources(papers)[:limit]

        return ToolResult(
            success=True,
            tool_name="semantic_scholar_search",
            data={
                "results": sources,
                "query": normalized_query,
                "total_found": response["data"].get("total", len(sources)),
                "provider": "semantic_scholar",
            },
            metadata=response.get("metadata"),
        )

    # ------------------------------------------------------------------
    # 3.2 推荐论文 —— 基于种子论文的相似推荐
    # ------------------------------------------------------------------

    async def recommend(
        self,
        topic: str,
        limit: int = 5,
        positive_paper_ids: Optional[List[str]] = None,
        negative_paper_ids: Optional[List[str]] = None,
    ) -> ToolResult:
        """
        【功能】基于种子论文，推荐相关论文。

        【调用示例】
          # 已知喜欢的论文 ID，直接推荐
          await client.recommend("nlp", positive_paper_ids=["id1", "id2"], negative_paper_ids=["id3"])

          # 没有种子 ID，自动用 topic 搜索一篇作为种子
          await client.recommend("reinforcement learning", limit=10)

        【推荐流程】
          1. 如果没有 positive_paper_ids → 用 topic 搜一篇作为种子
          2. POST 请求到 /recommendations API，传入正负种子列表
          3. 返回推荐论文列表

        【API 端点】
          POST {recommendations_base}/papers/?limit=...&fields=...
          Body: {"positivePaperIds": [...], "negativePaperIds": [...]}

        【为什么需要自动解析种子？】
          用户可能不知道 paperId，只说"想找类似 xxx 主题的论文"。
          这个自动种子解析让 Agent 不需要先调 search 再调 recommend，
          一次调用就能完成。
        """
        # Agent 工具的 schema 仍把单次精读预算限制在 50；Provider 本身允许
        # 分页接口为“加载更多”请求更大的累计窗口。500 是上游推荐端点的窗口上限，
        # 不是本应用的结果总量配置。
        bounded_limit = max(1, min(int(limit), 500))
        positive_ids = _clean_ids(positive_paper_ids)
        negative_ids = _clean_ids(negative_paper_ids)
        seed_sources: List[Dict[str, Any]] = []

        # ---- 如果没有正面种子，自动从 topic 搜索一篇作为种子 ----
        if not positive_ids:
            seed_result = await self.search(topic, 1)
            if not seed_result.success:
                return _failure(
                    "semantic_scholar_recommendations",
                    f"Could not resolve recommendation seed: {seed_result.error}",
                    seed_result.metadata,
                )
            seed_sources = seed_result.data.get("results") or []
            if not seed_sources:
                return _failure(
                    "semantic_scholar_recommendations",
                    "No seed paper found for the recommendation topic",
                )
            # 取搜索结果的第一篇作为种子论文
            seed_id = seed_sources[0].get("semantic_scholar_id")
            if not seed_id:
                return _failure(
                    "semantic_scholar_recommendations",
                    "Resolved seed paper has no Semantic Scholar paperId",
                )
            positive_ids = [seed_id]

        # ---- 调用推荐 API（POST 方式，传入正负种子） ----
        response = await self._request_json(
            "POST",
            f"{get_semantic_scholar_recommendations_base_url()}/papers/",
            params={"limit": bounded_limit, "fields": RELATIONSHIP_PAPER_FIELDS},
            json_body={
                "positivePaperIds": positive_ids,
                "negativePaperIds": negative_ids,
            },
        )

        if not response["success"]:
            return _failure(
                "semantic_scholar_recommendations",
                response["error"],
                response.get("metadata"),
            )

        papers = response["data"].get("recommendedPapers") or []
        sources = _normalize_sources(papers)[:bounded_limit]

        return ToolResult(
            success=True,
            tool_name="semantic_scholar_recommendations",
            data={
                "results": sources,
                "sources": sources,
                "seed_paper_ids": positive_ids,
                "seed_sources": seed_sources,     # 自动解析得到的种子论文信息
                "provider": "semantic_scholar",
            },
            metadata=response.get("metadata"),
        )

    async def recommend_page(
        self,
        topic: str,
        offset: int = 0,
        limit: int = 20,
        positive_paper_ids: Optional[List[str]] = None,
        negative_paper_ids: Optional[List[str]] = None,
    ) -> ToolResult:
        """按页返回推荐论文；应用不设置总页数，由上游结果耗尽时结束。"""
        page_offset = max(0, int(offset))
        page_limit = max(1, min(int(limit), 50))
        # 推荐端点没有原生 offset。请求累计窗口并只返回当前切片；多取一条用于
        # 判断是否仍有下一页。达到上游窗口后自然停止，而不是暴露应用级硬上限。
        requested_window = min(page_offset + page_limit + 1, 500)
        result = await self.recommend(
            topic=topic,
            limit=requested_window,
            positive_paper_ids=positive_paper_ids,
            negative_paper_ids=negative_paper_ids,
        )
        if not result.success:
            return result

        data = result.data if isinstance(result.data, dict) else {}
        all_sources = data.get("results") or data.get("sources") or []
        items = all_sources[page_offset:page_offset + page_limit]
        has_more = (
            len(all_sources) > page_offset + page_limit
            and page_offset + page_limit < 500
        )
        next_offset = page_offset + page_limit if has_more else None
        return ToolResult(
            success=True,
            tool_name="semantic_scholar_recommendations",
            data={
                **data,
                "results": items,
                "sources": items,
                "offset": page_offset,
                "limit": page_limit,
                "has_more": has_more,
                "next_offset": next_offset,
                "total_found": None,
            },
            metadata=result.metadata,
        )

    # ------------------------------------------------------------------
    # 3.3 论文图谱 —— 详情 / 引用 / 参考文献
    # ------------------------------------------------------------------

    async def paper_graph(
        self,
        paper_query: str,
        relation: str = "details",
        limit: int = 5,
        offset: int = 0,
    ) -> ToolResult:
        """
        【功能】查询一篇论文的详细信息或文献图谱关系。

        【调用示例】
          await client.paper_graph("Attention Is All You Need", relation="details")
          await client.paper_graph("arXiv:1706.03762", relation="citations", limit=10)
          await client.paper_graph("10.48550/arXiv.1706.03762", relation="references")

        【relation 三种模式】
          details     → GET /paper/{id}?fields=...           返回一篇论文的详细信息
          citations   → GET /paper/{id}/citations?limit=...  返回引用了该论文的论文列表
          references  → GET /paper/{id}/references?limit=... 返回该论文引用的文献列表

        【paper_query 的智能解析】
          用户传入的可能是：
          - "Attention Is All You Need"            → 普通标题 → 先调 match API 找 paperId
          - "10.48550/arXiv.1706.03762"            → DOI → 直接提取 "DOI:10.48550/..."
          - "arXiv:1706.03762"                     → arXiv ID → 直接提取
          - "CorpusId:12345678"                    → CorpusId → 直接提取
          - "204e3073870fae3d05bcbc2f6a8e263d9b72e776" → paperId → 直接使用

          解析逻辑在 _resolve_paper_id() 和 _extract_supported_paper_id() 中。
        """
        # 参数校验和默认值处理
        relation = relation if relation in {"details", "citations", "references"} else "details"

        # 第一步：把各种格式的 paper_query 解析成 Semantic Scholar paperId
        paper_id = await self._resolve_paper_id(paper_query)
        if not paper_id:
            return _failure(
                "semantic_scholar_graph",
                f"Could not resolve a Semantic Scholar paper from: {paper_query[:120]}",
            )

        # 对 paperId 做 URL 编码（处理特殊字符，如 DOI 中的 /）
        encoded_id = quote(paper_id, safe=":")
        bounded_limit = max(1, min(int(limit), 50))
        page_offset = max(0, int(offset))

        # ---- 根据 relation 类型，请求不同的 API 端点 ----
        if relation == "details":
            # 论文详情：GET /paper/{paper_id}?fields=...
            url = f"{get_semantic_scholar_graph_base_url()}/paper/{encoded_id}"
            response = await self._request_json("GET", url, params={"fields": PAPER_FIELDS})
            raw_papers = [response["data"]] if response["success"] else []

        else:
            # citations 或 references：GET /paper/{paper_id}/{relation}?limit=...&fields=...
            url = (
                f"{get_semantic_scholar_graph_base_url()}/paper/"
                f"{encoded_id}/{relation}"
            )
            response = await self._request_json(
                "GET",
                url,
                params={
                    "offset": page_offset,
                    "limit": bounded_limit,
                    "fields": RELATIONSHIP_PAPER_FIELDS,
                },
            )
            raw_papers = []
            if response["success"]:
                # citations 和 references 的响应结构中，每篇论文嵌套在不同的 key 下
                # citations  → "citingPaper"   (谁引了它)
                # references → "citedPaper"    (它引了谁)
                nested_key = "citingPaper" if relation == "citations" else "citedPaper"
                for item in response["data"].get("data") or []:
                    if isinstance(item, dict) and isinstance(item.get(nested_key), dict):
                        raw_papers.append(item[nested_key])

        if not response["success"]:
            return _failure("semantic_scholar_graph", response["error"], response.get("metadata"))

        sources = _normalize_sources(raw_papers)[:bounded_limit]
        response_data = response.get("data") if isinstance(response.get("data"), dict) else {}
        provider_next = response_data.get("next") if relation != "details" else None
        if relation == "details":
            has_more = False
            next_offset = None
            total_found = 1 if sources else 0
        else:
            has_more = provider_next is not None
            next_offset = int(provider_next) if has_more else None
            total_found = response_data.get("total")

        # 注意：如果 relation 是 details 且 paper_id 存在但没有任何 sources，
        # success 为 False（表示 API 正常但没有数据），这在 search 场景是合理的
        page_exhausted = page_offset > 0 and relation != "details" and not sources
        return ToolResult(
            success=bool(sources) or page_exhausted,
            tool_name="semantic_scholar_graph",
            data={
                "results": sources,
                "sources": sources,
                "paper_id": paper_id,
                "relation": relation,
                "provider": "semantic_scholar",
                "offset": page_offset,
                "limit": bounded_limit,
                "has_more": has_more,
                "next_offset": next_offset,
                "total_found": total_found,
            },
            error="" if sources or page_exhausted else f"No {relation} records found",
            metadata=response.get("metadata"),
        )

    # ------------------------------------------------------------------
    # 3.4 智能解析论文标识符
    # ------------------------------------------------------------------

    async def _resolve_paper_id(self, paper_query: str) -> str:
        """
        【功能】把用户输入的各种格式（标题/DOI/arXiv/CorpusId/paperId）
        统一解析成 Semantic Scholar paperId。

        【解析策略（两阶段）】
          第一阶段 —— 正则匹配（_extract_supported_paper_id）：
            尝试直接从字符串中提取已知格式的 ID（DOI, arXiv, CorpusId, paperId）
            如果匹配成功 → 直接用这个 ID 去 API 查询，跳过模糊搜索

          第二阶段 —— 模糊匹配（API match 端点）：
            如果正则没匹配上（可能是论文标题），调 Semantic Scholar 的
            /paper/search/match 端点，让 S2 自己去找最匹配的论文

        【为什么要两阶段？】
          1. 正则匹配：快（不涉及网络请求），适合精确 ID
          2. API 匹配：准（S2 的匹配算法比我们的正则强得多），适合模糊标题
        """
        query = str(paper_query or "").strip()
        if not query:
            return ""

        # 第一阶段：尝试精确 ID 提取
        direct = _extract_supported_paper_id(query)
        if direct:
            return direct   # 如 "DOI:10.xxx/yyy" 或 "ARXIV:1706.03762"

        # 第二阶段：调 API 做模糊匹配
        response = await self._request_json(
            "GET",
            f"{get_semantic_scholar_graph_base_url()}/paper/search/match",
            params={"query": _normalize_search_query(query), "fields": "paperId,title"},
        )
        if not response["success"]:
            return ""

        payload = response.get("data") or {}
        if not isinstance(payload, dict):
            return ""

        # Semantic Scholar has returned both shapes for this endpoint:
        # {"paperId": "..."} and {"data": [{"paperId": "..."}]}.
        # Accept both so a successful match is not mistaken for a missing paper.
        paper_id = payload.get("paperId")
        if paper_id:
            return str(paper_id)

        matches = payload.get("data") or []
        if isinstance(matches, list):
            for match in matches:
                if isinstance(match, dict) and match.get("paperId"):
                    return str(match["paperId"])
        return ""

    # ------------------------------------------------------------------
    # 3.5 统一 HTTP 请求 + 智能重试（本文件最复杂的逻辑）
    # ------------------------------------------------------------------

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        【功能】发送 HTTP 请求到 Semantic Scholar API，带完整的重试策略。

        【这是本文件最核心的基础设施方法】
          所有上层方法（search / recommend / paper_graph / _resolve_paper_id）
          最终都通过这个方法发送请求。它封装了：
          1. API key 认证（请求头 x-api-key）
          2. 超时控制（httpx 超时设置）
          3. 指数退避重试（临时性故障: Timeout / 5xx）
          4. Rate Limit 重试（429，优先读取 Retry-After 响应头）
          5. 重试预算机制（防止无限重试，累计等待时间不得超过 max_retry_wait）

        【重试策略详解】

          场景 A：网络超时 / 传输错误（httpx.TimeoutException / TransportError）
            → 指数退避: 1s → 2s → 4s → 8s (上限 8s)
            → 最多重试 max_retries 次
            → 累计等待时间超过 max_retry_wait 则放弃

          场景 B：HTTP 429 (Rate Limit Exceeded)
            → 优先读取响应头 Retry-After（精确到秒或 HTTP 日期）
            → 如果没有 Retry-After 头，退化为指数退避
            → 最多重试 rate_limit_retries 次（通常比 max_retries 多）
            → 同样受 max_retry_wait 预算约束

          场景 C：HTTP 4xx（400/401/403/404）
            → 客户端错误，不重试，直接返回失败

          场景 D：HTTP 5xx（500/502/503/504）
            → 服务器错误，指数退避重试

        【重试等待时间计算】
          延迟 = min(8.0, 2^attempt + random_jitter)
          - 第 1 次重试：~1 秒
          - 第 2 次重试：~2 秒
          - 第 3 次重试：~4 秒
          - 第 4 次及以后：~8 秒
          加上随机抖动可以避免"惊群效应"——多个请求同时重试造成二次雪崩。

        【返回格式】
          成功: {"success": True, "data": <JSON>, "metadata": {...}}
          失败: {"success": False, "error": "...", "metadata": {...}}
          metadata 包含: provider, status_code, attempts, rate_limit_retries,
                         retry_wait_seconds, latency_ms
        """
        max_retries = get_semantic_scholar_max_retries()
        rate_limit_max_retries = get_semantic_scholar_rate_limit_retries()
        max_retry_wait = get_semantic_scholar_max_retry_wait()

        # 设置请求头，如果有 API key 就带上（提升速率限制）
        headers = {"Accept": "application/json"}
        api_key = get_semantic_scholar_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        # ---- 重试循环的初始状态 ----
        started = time.monotonic()    # 整个请求流程的起始时间（单调时钟，不受系统时间调整影响）
        attempts = 0                  # 总尝试次数（含首次请求）
        transient_retries = 0         # 临时故障重试次数（Timeout / 5xx）
        rate_limit_retries = 0        # 限流重试次数（429）
        retry_wait_seconds = 0.0       # 累计等待时间（用于预算检查）

        while True:
            attempts += 1

            # ---- 发送 HTTP 请求 ----
            try:
                async with build_httpx_client(timeout=get_semantic_scholar_timeout()) as client:
                    response = await client.request(
                        method, url, params=params, json=json_body, headers=headers,
                    )

            # ---- 处理网络层异常（超时 / 连接失败） ----
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # 检查重试次数是否已用完
                if transient_retries >= max_retries:
                    return _request_failure(
                        exc.__class__.__name__, started, attempts,
                        rate_limit_retries, retry_wait_seconds,
                    )
                # 检查重试预算是否耗尽
                delay = _backoff_seconds(transient_retries)
                if retry_wait_seconds + delay > max_retry_wait:
                    return _request_failure(
                        "RetryWaitBudgetExceeded", started, attempts,
                        rate_limit_retries, retry_wait_seconds,
                    )
                transient_retries += 1
                retry_wait_seconds += delay
                await asyncio.sleep(delay)
                continue   # ← 回到循环开头，重试

            # ---- 构建元数据（无论成功失败都有） ----
            metadata = {
                "provider": "semantic_scholar",
                "status_code": response.status_code,
                "attempts": attempts,
                "rate_limit_retries": rate_limit_retries,
                "retry_wait_seconds": round(retry_wait_seconds, 3),
                "latency_ms": int((time.monotonic() - started) * 1000),
            }

            # ---- 处理正常响应 ----
            if response.status_code == 200:
                try:
                    return {"success": True, "data": response.json(), "metadata": metadata}
                except ValueError:   # JSON 解析失败
                    return {"success": False, "error": "Semantic Scholar returned invalid JSON", "metadata": metadata}

            # ---- 不可重试的错误 → 立即返回失败 ----
            if response.status_code in NON_RETRYABLE_STATUS:
                return {"success": False, "error": _http_error(response.status_code), "metadata": metadata}

            # ---- Rate Limit (429) 特殊处理 ----
            if response.status_code == 429:
                if rate_limit_retries >= rate_limit_max_retries:
                    return {"success": False, "error": _http_error(response.status_code), "metadata": metadata}
                # 优先用 Retry-After 头，没有的话退化为指数退避
                delay = _retry_delay(response, rate_limit_retries)
                if retry_wait_seconds + delay > max_retry_wait:
                    metadata["retry_wait_budget_exhausted"] = True
                    return {"success": False, "error": f"{_http_error(response.status_code)}; retry wait budget exhausted", "metadata": metadata}
                rate_limit_retries += 1
                retry_wait_seconds += delay
                await asyncio.sleep(delay)
                continue

            # ---- 兜底：其他可重试状态码（5xx 等） ----
            if response.status_code not in RETRYABLE_STATUS or transient_retries >= max_retries:
                return {"success": False, "error": _http_error(response.status_code), "metadata": metadata}
            delay = _retry_delay(response, transient_retries)
            if retry_wait_seconds + delay > max_retry_wait:
                metadata["retry_wait_budget_exhausted"] = True
                return {"success": False, "error": f"{_http_error(response.status_code)}; retry wait budget exhausted", "metadata": metadata}
            transient_retries += 1
            retry_wait_seconds += delay
            await asyncio.sleep(delay)
            # ← 回到循环开头，重试


# ============================================================================
# 阶段四：模块级辅助函数
# ============================================================================

# ---------------------------------------------------------------------------
# _normalize_sources: 论文列表去重 + 归一化
# ---------------------------------------------------------------------------

def _normalize_sources(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    【功能】将原始论文列表转换为归一化格式，并按 semantic_scholar_id / doi / url 去重。

    【为什么需要去重？】
      Semantic Scholar 的某些 API 端点可能返回重复论文（如同一篇论文的
      不同版本），通过唯一标识符去重可以保证列表干净。
    """
    sources = []
    seen = set()   # 已见过论文的唯一标识符集合
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        source = semantic_scholar_paper_to_source(paper)
        # 按优先级选择去重键: S2 ID > DOI > URL
        key = source.get("semantic_scholar_id") or source.get("doi") or source.get("url")
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(source)
    return sources


# ---------------------------------------------------------------------------
# _normalize_search_query: 搜索词预处理
# ---------------------------------------------------------------------------

def _normalize_search_query(query: str) -> str:
    """
    【功能】预处理搜索词，提升 Semantic Scholar 的搜索命中率。

    【处理规则】
      1. 把连字符替换为空格（Semantic Scholar 文档指出连字符可能导致搜索遗漏）
         例如 "state-of-the-art" → "state of the art"
      2. 合并多余空格
    """
    return re.sub(r"\s+", " ", re.sub(r"[-–—]+", " ", str(query or ""))).strip()


# ---------------------------------------------------------------------------
# _clean_ids: 清洗 paper ID 列表
# ---------------------------------------------------------------------------

def _clean_ids(values: Optional[List[str]]) -> List[str]:
    """
    【功能】清洗纸 ID 列表：去掉空白项、空字符串，最多返回 20 个。
    """
    return [str(value).strip() for value in (values or []) if str(value).strip()][:20]


# ---------------------------------------------------------------------------
# _extract_supported_paper_id: 正则提取已知格式的论文标识符
# ---------------------------------------------------------------------------

def _extract_supported_paper_id(text: str) -> str:
    """
    【功能】从字符串中用正则表达式提取已知格式的论文标识符。

    【支持的格式（按匹配顺序）】
      1. DOI:  "10.xxxx/yyyy"              → "DOI:10.xxxx/yyyy"
      2. arXiv: "arXiv:1706.03762"         → "ARXIV:1706.03762"
         or     "arxiv 1706.03762"
      3. CorpusId: "CorpusId:12345678"     → "CorpusId:12345678"
      4. paperId: 40 位十六进制字符串     → 直接返回原值

    【为什么返回带前缀的格式？】
      Semantic Scholar API 接受带前缀的 ID 作为 paper 查询参数，
      例如 GET /paper/DOI:10.xxx 或 GET /paper/ARXIV:1706.03762。
      这比直接传 40 位 paperId 对用户更友好。
    """
    # 匹配 DOI: 以 "10." 开头，后跟 4-9 位数字，斜杠，然后是非空格/逗号/分号的字符串
    doi = re.search(r"\b10\.\d{4,9}/[^\s,;]+", text, re.I)
    if doi:
        return f"DOI:{doi.group(0).rstrip('.?')}"

    # 匹配 arXiv ID: "arxiv" 后跟冒号或空格，然后是数字.数字格式的 ID
    arxiv = re.search(r"\barxiv[:\s]+(\d{4}\.\d{4,5}(?:v\d+)?)", text, re.I)
    if arxiv:
        return f"ARXIV:{arxiv.group(1)}"

    # 匹配 CorpusId: 显式的 corpusid 标记
    corpus = re.search(r"\bcorpusid[:\s]+(\d+)", text, re.I)
    if corpus:
        return f"CorpusId:{corpus.group(1)}"

    # 匹配 40 位十六进制字符串（Semantic Scholar 原生 paperId）
    paper_id = re.search(r"\b[a-f0-9]{40}\b", text, re.I)
    return paper_id.group(0) if paper_id else ""


# ---------------------------------------------------------------------------
# _backoff_seconds: 指数退避延迟计算
# ---------------------------------------------------------------------------

def _backoff_seconds(attempt: int) -> float:
    """
    【功能】计算指数退避延迟，加上随机抖动。

    【公式】
      delay = min(8.0, 2^attempt + random(0, 0.25))

    【为什么加随机抖动（Jitter）？】
      假设 10 个并发请求同时收到 429 → 如果所有客户端都等相同的时间，
      它们会在同一时刻再次同时发送请求 → 再次触发 429 → 无限循环。
      加上随机抖动后，每个客户端等待的时间略有不同，分散了重试时刻。
    """
    return min(8.0, (2 ** attempt) + random.uniform(0.0, 0.25))


# ---------------------------------------------------------------------------
# _retry_delay: 从 HTTP 响应中读取重试等待时间
# ---------------------------------------------------------------------------

def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """
    【功能】确定重试前的等待时间。

    【优先级】
      1. 如果响应头有 Retry-After（秒数）    → 直接使用，上限 30 秒
      2. 如果响应头有 Retry-After（HTTP 日期）→ 计算和目标时间的差值，上限 30 秒
      3. 如果都没有                            → 回退到指数退避

    【Retry-After 头的两种格式】
      - 秒数:  Retry-After: 120           （等 120 秒）
      - 日期:  Retry-After: Wed, 21 Oct 2015 07:28:00 GMT （等到那个时刻）
    """
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        # 尝试解析为秒数
        try:
            return min(30.0, max(0.0, float(retry_after)))
        except ValueError:
            # 尝试解析为 HTTP 日期
            try:
                target = parsedate_to_datetime(retry_after).timestamp()
                return min(30.0, max(0.0, target - time.time()))
            except (TypeError, ValueError, OverflowError):
                pass
    # 没有 Retry-After 头 → 退化为指数退避
    return _backoff_seconds(attempt)


# ---------------------------------------------------------------------------
# _http_error: HTTP 错误信息的人类可读描述
# ---------------------------------------------------------------------------

def _http_error(status_code: int) -> str:
    """把 HTTP 状态码映射为可读的错误描述。"""
    labels = {
        400: "bad request",
        401: "invalid or missing API key",
        403: "request forbidden",
        404: "paper or endpoint not found",
        429: "rate limit exceeded",
    }
    return f"Semantic Scholar HTTP {status_code}: {labels.get(status_code, 'request failed')}"


# ---------------------------------------------------------------------------
# _request_failure: 构建 HTTP 请求失败的统一返回结构
# ---------------------------------------------------------------------------

def _request_failure(
    error_type: str,
    started: float,
    attempts: int,
    rate_limit_retries: int = 0,
    retry_wait_seconds: float = 0.0,
) -> Dict[str, Any]:
    """构建网络层错误的返回字典（不是 ToolResult，由上层组装）。"""
    return {
        "success": False,
        "error": f"Semantic Scholar request failed: {error_type}",
        "metadata": {
            "provider": "semantic_scholar",
            "attempts": attempts,
            "rate_limit_retries": rate_limit_retries,
            "retry_wait_seconds": round(retry_wait_seconds, 3),
            "latency_ms": int((time.monotonic() - started) * 1000),
        },
    }


# ---------------------------------------------------------------------------
# _failure: 构建返回给工具的失败 ToolResult
# ---------------------------------------------------------------------------

def _failure(
    tool_name: str,
    error: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    """
    【便捷函数】快速构建一个失败的 ToolResult。

    用途：业务层错误（搜索词为空、找不到种子论文等），不需要记录重试信息。
    网络层错误用 _request_failure() 直接返回字典。
    """
    return ToolResult(
        success=False,
        tool_name=tool_name,
        error=error,
        metadata=metadata or {"provider": "semantic_scholar"},
    )


# ============================================================================
# 阶段五：架构总结
# ============================================================================
"""
【本文件在整体架构中的位置】

  ┌─────────────────────────────────────────────────┐
  │  semantic_scholar_tools.py (工具定义层)          │
  │  SearchTool / RecommendationsTool / GraphTool   │
  │  只定义 name/description/schema，不处理 HTTP     │
  └──────────────────┬──────────────────────────────┘
                     │ 调用
  ┌──────────────────▼──────────────────────────────┐
  │  semantic_scholar_provider.py (你在这里)         │
  │                                                   │
  │  SemanticScholarClient (HTTP 客户端)              │
  │  ├── search()          调 /paper/search            │
  │  ├── recommend()       调 /recommendations         │
  │  ├── paper_graph()     调 /paper/{id}[/{relation}] │
  │  ├── _resolve_paper_id()  智能 ID 解析               │
  │  └── _request_json()      统一请求 + 重试引擎       │
  │                                                   │
  │  辅助函数（数据归一化）                               │
  │  ├── semantic_scholar_paper_to_source()  格式转换   │
  │  ├── _normalize_sources()               列表去重   │
  │  ├── _normalize_search_query()          搜索预处理  │
  │  ├── _extract_supported_paper_id()      正则 ID 提取│
  │  └── _backoff_seconds / _retry_delay    重试延迟    │
  └──────────────────┬──────────────────────────────┘
                     │ HTTP (httpx)
  ┌──────────────────▼──────────────────────────────┐
  │  Semantic Scholar Academic Graph API             │
  │  api.semanticscholar.org                         │
  └─────────────────────────────────────────────────┘

【本文件的关键设计决策】

  1. "永不抛异常"原则
     所有方法要么返回成功，要么返回 ToolResult(success=False)。
     上层调用方通过 result.success 判断，不需要 try-catch。

  2. 重试预算机制
     不是简单的"最多重试 N 次"，而是有 max_retry_wait 时间上限。
     这样即便并发请求很多，也不会无限堆积等待时间。

  3. 数据归一化在 provider 层完成
     无论 API 返回什么格式，经过 semantic_scholar_paper_to_source()
     后都变成项目统一的 PaperSource 格式。以后如果要接 arXiv 或
     PubMed 数据源，只需实现对应的 provider，上层 Agent 无感知。

  4. 智能 ID 解析
     用户不需要知道 paperId 是什么。传入标题 → 自动 match。
     传入 DOI → 自动识别。 传入 arXiv ID → 自动识别。
     这让 Agent 对用户输入的容错性非常高。
"""
