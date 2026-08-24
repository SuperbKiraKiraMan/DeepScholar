"""
app/tools/academic_search_provider.py

Academic Search Provider：搜索能力抽象层。

定义 AcademicSearchProvider Protocol 以及两个实现：
- MockAcademicSearchProvider: 复用现有 mock 数据（7 篇论文），完全离线
- OpenAlexSearchProvider: 真实 OpenAlex API 搜索 + PaperSource 映射

OpenAlex 到 PaperSource 的映射由 OpenAlexToPaperSource adapter 完成，
保证 OpenAlex 响应结构不泄漏到 Agent Runtime。

设计原则：
1. Provider-neutral: Agent 只知道 "academic_search"，不感知 mock/openalex
2. Lazy config: 所有配置通过 app.core.config 的函数读取，支持 monkeypatch
3. API key 安全: key 只从环境变量读取，不通过 LLM args 传入
4. 安全降级: 缺失字段不会 KeyError；OpenAlex 失败不回退 Mock（除非显式配置）
5. 去重: 优先 OpenAlex Work ID，其次 DOI；结果顺序稳定
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx
import xml.etree.ElementTree as ET

from app.core.config import (
    get_openalex_api_key,
    get_openalex_base_url,
    get_openalex_content_base_url,
    get_openalex_content_mode,
    get_openalex_fallback_to_mock,
    get_openalex_max_content_fetches,
    get_openalex_max_retries,
    get_openalex_max_text_chars,
    get_openalex_timeout,
)
from app.tools.base import ToolResult
from app.tools.http_client import build_httpx_client
from app.tools.mock_academic_search_tool import _MOCK_PAPERS

logger = logging.getLogger(__name__)

# ================================================================
# Provider Protocol
# ================================================================


class AcademicSearchProvider(Protocol):
    """搜索 Provider 的结构化协议（鸭子类型，不需要显式继承）。"""

    async def search(
        self,
        query: str,
        max_results: int,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> ToolResult:
        """
        执行学术搜索。

        Args:
            query: 搜索查询字符串
            max_results: 最大返回结果数

        Returns:
            ToolResult(success=True, data={"results": [...], "query": ..., "total_found": ...})
        """
        ...


# ================================================================
# MockAcademicSearchProvider
# ================================================================


class MockAcademicSearchProvider:
    """
    离线 Mock Provider —— 复用现有 _MOCK_PAPERS 数据。

    完全离线，7 篇带 full_text 的 mock 论文。
    返回格式与 OpenAlexSearchProvider 一致。
    """

    def __init__(self):
        self._papers = list(_MOCK_PAPERS)

    async def search(
        self,
        query: str,
        max_results: int,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                tool_name="academic_search",
                error="Search query is empty",
            )

        max_results = max(1, min(max_results, len(self._papers)))
        query_lower = query.lower()
        keywords = re.findall(r'[a-zA-Z]{2,}', query_lower)

        # 关键词匹配
        matched = []
        for paper in self._papers:
            text = (paper["title"] + " " + paper["snippet"]).lower()
            if any(kw in text for kw in keywords):
                matched.append(paper)

        if not matched:
            matched = list(self._papers)

        if year_from is not None:
            matched = [
                paper for paper in matched
                if paper.get("year") is not None and paper["year"] >= year_from
            ]
        if year_to is not None:
            matched = [
                paper for paper in matched
                if paper.get("year") is not None and paper["year"] <= year_to
            ]

        results = []
        for paper in matched[:max_results]:
            source = {
                "source_id": str(uuid.uuid4())[:8],
                "title": paper["title"],
                "url": paper["url"],
                "snippet": paper["snippet"],
                "full_text": paper["full_text"],
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "source_type": paper.get("source_type", "unknown"),
                # 补充 provider metadata。
                "provider": "mock",
                "openalex_id": None,
                "doi": None,
                "cited_by_count": None,
                "is_oa": None,
                "oa_status": None,
                "content_url": None,
                "content_source": None,
            }
            results.append(source)

        return ToolResult(
            success=True,
            tool_name="academic_search",
            data={
                "results": results,
                "query": query,
                "total_found": len(results),
                "provider": "mock",
                "year_from": year_from,
                "year_to": year_to,
            },
        )


# ================================================================
# OpenAlex Work → PaperSource Adapter
# ================================================================


# OpenAlex type → 项目 source_type 映射
_TYPE_MAPPING: Dict[str, str] = {
    "article": "paper",
    "review": "paper",
    "preprint": "paper",
    "book-chapter": "book",
    "book": "book",
    "dissertation": "paper",
    "proceedings-article": "paper",
    "report": "report",
    "dataset": "other",
    "other": "other",
}


def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    """
    从 OpenAlex abstract_inverted_index 重建自然顺序摘要。

    OpenAlex 的 abstract_inverted_index 格式:
        {"word1": [pos1, pos3], "word2": [pos2], ...}

    按位置排序拼接，恢复自然语序。
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    # 收集所有 (position, word) 对
    positioned: List[Tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, word))

    if not positioned:
        return ""

    # 按位置排序
    positioned.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positioned)


def _select_source_type(openalex_type: Optional[str]) -> str:
    """将 OpenAlex type 映射为项目 source_type。"""
    if not openalex_type:
        return "unknown"
    return _TYPE_MAPPING.get(openalex_type.strip().lower(), "other")


def _select_canonical_url(work: Dict) -> str:
    """
    选择 PaperSource 的 canonical URL。

    优先级: DOI URL → primary_location.landing_page_url → OpenAlex Work ID URL
    """
    doi = work.get("doi", "")
    if doi:
        doi_clean = doi.strip()
        if doi_clean:
            # 如果已经是完整 DOI URL，直接使用；否则补全前缀
            if doi_clean.startswith("https://doi.org/"):
                return doi_clean
            return f"https://doi.org/{doi_clean}"

    primary = work.get("primary_location") or {}
    landing = primary.get("landing_page_url", "")
    if landing:
        return landing

    work_id = work.get("id", "")
    if work_id:
        return work_id  # OpenAlex ID URL: https://openalex.org/works/W...

    return ""


def _extract_work_id(work: Dict) -> str:
    """
    从 OpenAlex Work 对象提取稳定的 Work ID。

    格式: https://openalex.org/works/W2741809807 → W2741809807
    """
    raw = work.get("id", "")
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    return raw


def openalex_work_to_paper_source(work: Dict) -> Dict[str, Any]:
    """
    将 OpenAlex Work 对象映射为项目 PaperSource。

    所有字段使用 .get() 安全访问，不允许 KeyError。
    缺失字段填入合理默认值。

    Mapping:
      id → source_id (extract W...)
      display_name → title
      authorships[].author.display_name → authors
      publication_year → year
      primary_location.source.display_name → venue
      abstract_inverted_index → snippet + full_text
      type → source_type
    """
    work_id = _extract_work_id(work)

    # ---- 标题 ----
    title = (work.get("display_name") or "").strip()

    # ---- 作者 ----
    authors = []
    for authorship in (work.get("authorships") or []):
        if isinstance(authorship, dict):
            author = authorship.get("author") or {}
            if isinstance(author, dict):
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

    # ---- 年份 ----
    year = work.get("publication_year")

    # ---- 期刊/会议名 ----
    primary = work.get("primary_location") or {}
    source_info = primary.get("source") or {}
    venue = (source_info.get("display_name") or "").strip()

    # ---- 摘要 ----
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    snippet = abstract[:300] if abstract else ""

    # 如果没有重建的摘要，尝试从 title 生成简短描述
    if not snippet:
        snippet = title[:300]

    # full_text 默认用摘要（TEI 模式由 _fetch_tei_content 增强）
    full_text = abstract

    # ---- Source type ----
    source_type = _select_source_type(work.get("type"))

    # ---- Canonical URL ----
    url = _select_canonical_url(work)

    # ---- DOI ----
    doi = work.get("doi", "")

    # ---- OpenAlex metadata ----
    openalex_meta = work.get("open_access") or {}

    # OpenAlex Content API availability is exposed on the Work itself.  Publisher
    # landing/PDF URLs are citation/download locations, not Content API signals.
    content_url = work.get("content_url") or ""
    has_content = work.get("has_content") or {}

    return {
        "source_id": work_id,
        "title": title,
        "url": url,
        "snippet": snippet,
        "full_text": full_text,
        "authors": authors,
        "year": year,
        "venue": venue,
        "source_type": source_type,
        # ---- 可信 metadata ----
        "provider": "openalex",
        "openalex_id": work_id,
        "doi": doi,
        "cited_by_count": work.get("cited_by_count"),
        "is_oa": openalex_meta.get("is_oa"),
        "oa_status": openalex_meta.get("oa_status"),
        "content_url": content_url,
        "has_content": has_content,
        "content_source": "openalex",
        "publication_date": work.get("publication_date"),
    }


# ================================================================
# OpenAlexSearchProvider
# ================================================================


class OpenAlexSearchProvider:
    """
    OpenAlex 真实搜索 Provider。

    使用 httpx.AsyncClient 请求 GET /works，映射为 PaperSource。
    包含完整重试、超时、限流和 API key 脱敏逻辑。

    重试策略:
    - 400/401/403: 不重试（客户端错误）
    - 429: 优先尊重 Retry-After header，上限 30s
    - 5xx/Timeout/TransportError: 短指数退避 min(2^attempt, 16)s
    - 达到 OPENALEX_MAX_RETRIES 后返回 ToolResult(success=False)
    - OPENALEX_FALLBACK_TO_MOCK=true 时回退 Mock 并标记 fallback_used=true
    """

    def __init__(self):
        self._base_url = get_openalex_base_url().rstrip("/")
        self._content_base_url = get_openalex_content_base_url().rstrip("/")
        self._timeout = get_openalex_timeout()
        self._max_retries = get_openalex_max_retries()
        self._fallback_to_mock = get_openalex_fallback_to_mock()
        self._content_mode = get_openalex_content_mode()
        self._max_content_fetches = get_openalex_max_content_fetches()
        self._max_text_chars = get_openalex_max_text_chars()
        self._mock = MockAcademicSearchProvider()

    # ---- API key 脱敏 ----

    @staticmethod
    def _safe_error_context(message: str) -> str:
        """移除消息中的 API key 和完整 query URL。"""
        # 移除 sk-... 和类似 key
        message = re.sub(r'(api_key=)[^&\s]+', r'\1[REDACTED]', message)
        message = re.sub(r'(sk-[a-zA-Z0-9_-]{10,})', '[REDACTED_KEY]', message)
        return message

    # ---- 主搜索入口 ----

    async def search(
        self,
        query: str,
        max_results: int,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                tool_name="academic_search",
                error="Search query is empty",
            )

        api_key = get_openalex_api_key()
        if not api_key:
            # 无 key 时按配置决定是否回退
            if self._fallback_to_mock:
                result = await self._mock.search(
                    query,
                    max_results,
                    year_from=year_from,
                    year_to=year_to,
                )
                if result.data:
                    result.data["fallback_used"] = True
                    result.data["fallback_reason"] = "no_api_key"
                result.metadata["fallback_used"] = True
                result.metadata["fallback_reason"] = "no_api_key"
                return result
            return ToolResult(
                success=False,
                tool_name="academic_search",
                error="OPENALEX_API_KEY not configured and fallback to mock is disabled",
            )

        max_results = max(1, min(max_results, 50))

        # 搜索（带完整重试）
        try:
            results = await self._search_with_retry(
                query,
                max_results,
                api_key,
                year_from=year_from,
                year_to=year_to,
            )
        except Exception:
            results = ToolResult(
                success=False,
                tool_name="academic_search",
                error="OpenAlex search raised unexpected exception",
            )

        # 失败时按配置决定是否回退 Mock
        if not results.success:
            if self._fallback_to_mock:
                result = await self._mock.search(
                    query,
                    max_results,
                    year_from=year_from,
                    year_to=year_to,
                )
                if result.data:
                    result.data["fallback_used"] = True
                    result.data["fallback_reason"] = "openalex_failed"
                result.metadata["fallback_used"] = True
                result.metadata["fallback_reason"] = "openalex_failed"
                return result
            return results

        if results.data:
            sources = results.data.get("results", [])
            results.data["results"] = await self.enrich_full_text(sources)

        return results

    # ---- 按标题定位单篇论文（论文详情兜底） ----

    async def search_title(self, title: str, limit: int = 3) -> ToolResult:
        """按标题在 OpenAlex 中精确定位论文，返回前 limit 篇 PaperSource。

        filter=title.search 限定标题命中查询词，search 提供相关性排序。
        无 API key 时与 search() 策略一致：直接失败，不回退 Mock。
        """
        title = (title or "").strip()
        if not title:
            return ToolResult(
                success=False,
                tool_name="academic_search",
                error="Search title is empty",
            )

        api_key = get_openalex_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                tool_name="academic_search",
                error="OPENALEX_API_KEY not configured and fallback to mock is disabled",
            )

        limit = max(1, min(int(limit or 3), 10))
        params = {
            "search": title,
            "filter": f"title.search:{title}",
            "per_page": str(limit),
            "api_key": api_key,
        }
        # 安全日志（不含 API key）
        safe_params = {k: v for k, v in params.items() if k != "api_key"}
        logger.debug("OpenAlex title search: %s", safe_params)

        last_error: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            try:
                async with build_httpx_client(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{self._base_url}/works",
                        params=params,
                    )

                    if response.status_code == 200:
                        return self._handle_success(
                            response, title, limit, None, None
                        )

                    if response.status_code in (400, 401, 403):
                        return ToolResult(
                            success=False,
                            tool_name="academic_search",
                            error=self._safe_error_context(
                                f"OpenAlex HTTP {response.status_code}: "
                                f"{response.text[:500]}"
                            ),
                        )

                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        if attempt < self._max_retries:
                            logger.warning(
                                "OpenAlex 429, retry %s/%s after %ss",
                                attempt + 1, self._max_retries, retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        return ToolResult(
                            success=False,
                            tool_name="academic_search",
                            error="OpenAlex rate limited (429), max retries exhausted",
                            metadata={"http_status": 429},
                        )

                    if response.status_code >= 500:
                        last_error = f"OpenAlex HTTP {response.status_code}"
                        if attempt < self._max_retries:
                            delay = min(2 ** attempt, 16)
                            logger.warning(
                                "OpenAlex 5xx, retry %s/%s after %ss",
                                attempt + 1, self._max_retries, delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        return ToolResult(
                            success=False,
                            tool_name="academic_search",
                            error=f"{last_error}: max retries exhausted",
                            metadata={"http_status": response.status_code},
                        )

                    return ToolResult(
                        success=False,
                        tool_name="academic_search",
                        error=f"OpenAlex unexpected HTTP {response.status_code}",
                        metadata={"http_status": response.status_code},
                    )

            except httpx.TimeoutException:
                last_error = f"OpenAlex timeout after {self._timeout}s"
                if attempt < self._max_retries:
                    delay = min(2 ** attempt, 16)
                    await asyncio.sleep(delay)
                    continue

            except (httpx.TransportError, httpx.ConnectError) as e:
                last_error = f"OpenAlex transport error: {type(e).__name__}"
                if attempt < self._max_retries:
                    delay = min(2 ** attempt, 16)
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_error = (
                    f"OpenAlex unexpected: {type(e).__name__}: {str(e)[:200]}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(min(2 ** attempt, 16))
                    continue

        return ToolResult(
            success=False,
            tool_name="academic_search",
            error=self._safe_error_context(last_error or "Unknown error"),
        )

    # ---- 带重试的 HTTP 搜索 ----

    async def _search_with_retry(
        self,
        query: str,
        max_results: int,
        api_key: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> ToolResult:
        """执行检索请求，带完整重试逻辑。"""
        filters = ["has_abstract:true"]
        if year_from is not None:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to is not None:
            filters.append(f"to_publication_date:{year_to}-12-31")
        params = {
            "search": query,
            # This Agent produces claims from exact source text. Metadata-only
            # works cannot satisfy that contract, so require an abstract.
            "filter": ",".join(filters),
            "per_page": str(max_results),
            "api_key": api_key,
        }

        # 安全日志（不含 API key）
        safe_params = {k: v for k, v in params.items() if k != "api_key"}
        logger.debug("OpenAlex search: %s", safe_params)

        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            try:
                async with build_httpx_client(timeout=self._timeout) as client:
                    response = await client.get(
                        f"{self._base_url}/works",
                        params=params,
                    )

                    # ---- 处理各种 HTTP 状态码 ----
                    if response.status_code == 200:
                        return self._handle_success(
                            response,
                            query,
                            max_results,
                            year_from=year_from,
                            year_to=year_to,
                        )

                    if response.status_code in (400, 401, 403):
                        # 客户端错误，不重试
                        error_body = response.text[:500]
                        return ToolResult(
                            success=False,
                            tool_name="academic_search",
                            error=self._safe_error_context(
                                f"OpenAlex HTTP {response.status_code}: {error_body}"
                            ),
                        )

                    if response.status_code == 429:
                        # 速率限制 → 尊重 Retry-After
                        retry_after = self._parse_retry_after(response)
                        if attempt < self._max_retries:
                            logger.warning(
                                "OpenAlex 429, retry %s/%s after %ss",
                                attempt + 1, self._max_retries, retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        return ToolResult(
                            success=False,
                            tool_name="academic_search",
                            error=f"OpenAlex rate limited (429), max retries exhausted",
                            metadata={"http_status": 429},
                        )

                    if response.status_code >= 500:
                        # 服务端错误 → 指数退避
                        last_error = f"OpenAlex HTTP {response.status_code}"
                        if attempt < self._max_retries:
                            delay = min(2 ** attempt, 16)
                            logger.warning(
                                "OpenAlex 5xx, retry %s/%s after %ss",
                                attempt + 1, self._max_retries, delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        return ToolResult(
                            success=False,
                            tool_name="academic_search",
                            error=f"{last_error}: max retries exhausted",
                            metadata={"http_status": response.status_code},
                        )

                    # 其他状态码
                    return ToolResult(
                        success=False,
                        tool_name="academic_search",
                        error=f"OpenAlex unexpected HTTP {response.status_code}",
                        metadata={"http_status": response.status_code},
                    )

            except httpx.TimeoutException:
                last_error = f"OpenAlex timeout after {self._timeout}s"
                if attempt < self._max_retries:
                    delay = min(2 ** attempt, 16)
                    logger.warning(
                        "OpenAlex timeout, retry %s/%s after %ss",
                        attempt + 1, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

            except (httpx.TransportError, httpx.ConnectError) as e:
                last_error = f"OpenAlex transport error: {type(e).__name__}"
                if attempt < self._max_retries:
                    delay = min(2 ** attempt, 16)
                    logger.warning(
                        "OpenAlex transport error, retry %s/%s after %ss",
                        attempt + 1, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_error = f"OpenAlex unexpected: {type(e).__name__}: {str(e)[:200]}"
                if attempt < self._max_retries:
                    delay = min(2 ** attempt, 16)
                    await asyncio.sleep(delay)
                    continue

        return ToolResult(
            success=False,
            tool_name="academic_search",
            error=self._safe_error_context(last_error or "Unknown error"),
        )

    # ---- 成功响应处理 ----

    def _handle_success(
        self,
        response: httpx.Response,
        query: str,
        max_results: int,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> ToolResult:
        """处理 200 响应：解析、映射、去重。"""
        try:
            body = response.json()
        except Exception:
            return ToolResult(
                success=False,
                tool_name="academic_search",
                error="Failed to parse OpenAlex JSON response",
            )

        works = body.get("results", [])
        if not works:
            return ToolResult(
                success=True,
                tool_name="academic_search",
                data={
                    "results": [],
                    "query": query,
                    "total_found": 0,
                    "provider": "openalex",
                    "year_from": year_from,
                    "year_to": year_to,
                    "meta": body.get("meta", {}),
                },
            )

        # 映射 + 去重
        sources = []
        seen_ids: set = set()
        seen_dois: set = set()

        for work in works:
            source = openalex_work_to_paper_source(work)
            sid = source["source_id"]
            doi = (source.get("doi") or "").strip().lower()

            # 去重：OpenAlex ID 优先，其次 DOI
            if sid and sid in seen_ids:
                continue
            if doi and doi in seen_dois:
                continue

            if sid:
                seen_ids.add(sid)
            if doi:
                seen_dois.add(doi)

            sources.append(source)

            if len(sources) >= max_results:
                break

        return ToolResult(
            success=True,
            tool_name="academic_search",
            data={
                "results": sources,
                "query": query,
                "total_found": body.get("meta", {}).get("count", len(sources)),
                "provider": "openalex",
                "year_from": year_from,
                "year_to": year_to,
                "meta": {
                    "db_response_time_ms": body.get("meta", {}).get("db_response_time_ms"),
                    "page": body.get("meta", {}).get("page"),
                    "per_page": body.get("meta", {}).get("per_page"),
                },
            },
        )

    # ---- Retry-After 解析 ----

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> int:
        """解析 Retry-After header，上限 30s，默认 5s。"""
        raw = response.headers.get("Retry-After", "").strip()
        if not raw:
            return 5

        # 纯数字 = 秒数
        try:
            seconds = int(raw)
            return max(1, min(seconds, 30))
        except ValueError:
            pass

        # HTTP-date 格式（很少见）
        try:
            from email.utils import parsedate_to_datetime
            import datetime
            retry_dt = parsedate_to_datetime(raw)
            now = datetime.datetime.now(datetime.timezone.utc)
            delta = (retry_dt - now).total_seconds()
            return max(1, min(int(delta), 30))
        except Exception:
            pass

        return 5

    # ================================================================
    # TEI 全文获取（Content API）
    # ================================================================

    async def enrich_full_text(self, sources: List[Dict]) -> List[Dict]:
        """
        对前 N 条结果尝试获取 TEI 全文。

        OPENALEX_CONTENT_MODE=tei 时调用 Content API。
        任何失败（404/429/XML损坏/无content_url/配额不足）→ 回退 abstract。
        """
        if self._content_mode != "tei":
            return sources

        fetches = min(self._max_content_fetches, len(sources))
        if fetches == 0:
            return sources

        for i in range(fetches):
            source = sources[i]
            work_id = source.get("openalex_id") or source.get("source_id", "")
            content_url = source.get("content_url", "")
            has_content = source.get("has_content") or {}
            has_tei = bool(has_content.get("grobid_xml"))

            if not work_id or (not content_url and not has_tei):
                continue

            tei_text = await self._fetch_tei_content(work_id)
            if tei_text:
                # 用 TEI 提取的文本增强 full_text
                source["full_text"] = tei_text[: self._max_text_chars]
                source["content_source"] = "openalex_tei"
            # 失败 → 保持 abstract（不做任何事）

        return sources

    async def _fetch_tei_content(self, work_id: str) -> Optional[str]:
        """
        从 Content API 获取 TEI XML 并提取文本。

        Content API endpoint:
          GET https://content.openalex.org/works/{work_id}.grobid-xml?api_key=...

        返回提取的纯文本，失败返回 None。
        """
        api_key = get_openalex_api_key()
        url = f"{self._content_base_url}/works/{work_id}.grobid-xml"

        try:
            async with build_httpx_client(timeout=self._timeout) as client:
                response = await client.get(
                    url,
                    params={"api_key": api_key},
                )

                if response.status_code == 404:
                    logger.debug("Content API 404 for %s", work_id)
                    return None

                if response.status_code == 429:
                    logger.debug("Content API 429 for %s", work_id)
                    return None

                if response.status_code != 200:
                    logger.debug("Content API %s for %s", response.status_code, work_id)
                    return None

                xml_text = response.text
                return self._parse_tei_xml(xml_text)

        except httpx.TimeoutException:
            logger.debug("Content API timeout for %s", work_id)
            return None
        except Exception:
            logger.debug("Content API error for %s", work_id, exc_info=True)
            return None

    @staticmethod
    def _parse_tei_xml(xml_text: str) -> Optional[str]:
        """
        解析 TEI XML，提取 abstract/body/paragraph 的文本。

        使用标准 xml.etree.ElementTree，限制最大字符数。
        XML 损坏或解析失败时返回 None。
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None

        # TEI namespace: http://www.tei-c.org/ns/1.0
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        parts = []

        # 尝试提取 abstract
        for abstract in root.iter():
            tag = abstract.tag.split("}", 1)[-1] if "}" in abstract.tag else abstract.tag
            if tag == "abstract":
                text = " ".join(abstract.itertext()).strip()
                if text:
                    parts.append(text)

        # 提取 body 中的段落
        for body in root.iter():
            tag = body.tag.split("}", 1)[-1] if "}" in body.tag else body.tag
            if tag == "body":
                for p in body.iter():
                    ptag = p.tag.split("}", 1)[-1] if "}" in p.tag else p.tag
                    if ptag == "p":
                        text = " ".join(p.itertext()).strip()
                        if text:
                            parts.append(text)

        if not parts:
            return None

        return " ".join(parts)
