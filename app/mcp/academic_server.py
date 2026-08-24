"""
================================================================================
MCP 学术研究工具服务器 —— 将项目工具暴露为 MCP 协议的标准工具
================================================================================

【本文件的定位】
  这是一个独立的 MCP (Model Context Protocol) Server 进程，把项目中的学术
  搜索工具以 MCP 标准协议暴露出去。任何支持 MCP 的客户端（Claude Desktop、
  Continue、Cline 等）都可以直接调用这些工具，无需依赖项目的 Agent/LangGraph
  运行时。

【MCP 协议简介】
  MCP 是一个开放协议，定义了 AI 应用与外部工具/数据源之间的标准通信方式。
  通信方式有两种：
    - stdio:  标准输入输出流（本地进程通信，低延迟）
    - SSE:    Server-Sent Events（HTTP 长连接，支持远程调用）
  本服务器默认使用 stdio，可通过环境变量 MCP_SERVER_TRANSPORT 切换。

【本文件与项目其他部分的关系】

  semantic_scholar_tools.py  ← 工具定义（name / schema / _arun）
       │
  semantic_scholar_provider.py ← HTTP 客户端 + 数据归一化
       │
  ┌────▼─────────────────────────────────────────┐
  │  academic_server.py  ← 你在这里（MCP 包装层） │
  │                                                   │
  │  把 3 个 Semantic Scholar 工具                   │
  │  + AcademicSearchTool                            │
  │  + Crossref 搜索（MCP 独有）                     │
  │  + 本地 RAG 搜索（MCP 独有）                     │
  │  包装成 MCP 标准工具暴露出去                      │
  └─────────────────────────────────────────────────┘
       │
       ▼  MCP 协议（stdio / SSE）
  Claude Desktop / Continue / Cline / 任何 MCP 客户端

【非 MCP 路径的对比】
  项目的另一条路径是嵌入 Agent 运行时（LangGraph 节点内调用），走 ToolRegistry。
  而本文件让这些工具在 Agent 运行时之外也能独立使用——解耦了工具能力和 Agent 逻辑。

【启动方式】
  python -m app.mcp.academic_server
  或通过环境变量:
  MCP_SERVER_TRANSPORT=sse python -m app.mcp.academic_server
"""

# ============================================================================
# 阶段一：导入依赖
# ============================================================================

import hashlib            # 生成稳定的 source_id（SHA256 哈希）
import html               # 解码 HTML 实体（如 &amp; → &）
import os                 # 读取环境变量配置
import re                 # 正则清理 HTML 标签
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP  # FastMCP: MCP 服务器的快速开发框架

# ---- 导入项目的现有工具 ----
from app.tools.academic_search_tool import AcademicSearchTool
from app.tools.base import ToolResult
from app.tools.http_client import build_httpx_client
from app.tools.semantic_scholar_tools import (
    SemanticScholarGraphTool,
    SemanticScholarRecommendationsTool,
    SemanticScholarSearchTool,
)


# ============================================================================
# 阶段二：创建 MCP 服务器实例
# ============================================================================

mcp = FastMCP(
    "academic-research-tools",   # 服务名称（客户端识别用）
    instructions=(               # 服务说明（客户端展示给用户看）
        "Optional academic research tools. Returned sources use the project's "
        "PaperSource-compatible schema and must still pass local evidence and citation checks."
        # ↑ 强调：MCP 返回的 source 仍是项目标准的 PaperSource 格式，
        #   后续可能还需要经过证据抽取和引用校验
    ),
)


# ============================================================================
# 阶段三：MCP 工具定义 —— @mcp.tool() 装饰器
# ============================================================================
#
# 【@mcp.tool(structured_output=True) 做了什么？】
#   FastMCP 自动从函数签名和 docstring 生成 MCP 标准的 tool schema:
#   - name:       函数名
#   - description: 函数的 docstring
#   - inputSchema:  从参数类型注解自动推导（如 query: str → {"type": "string"}）
#   structured_output=True 表示返回的是结构化数据（dict），不是纯文本
#
# 【和项目内 BaseTool 的对比】
#   项目内: BaseTool.name + description + input_schema → ToolRegistry → Agent 调用
#   MCP:    函数签名 + 类型注解 + docstring   → FastMCP 注册  → MCP 客户端调用
#
#   两种方式殊途同归——都是把工具的能力描述给 LLM，让 LLM 决定何时调用。


# -------------------------------------------------------------------
# 3.1 recommend_papers —— 通用论文推荐（通过 AcademicSearchTool）
# -------------------------------------------------------------------

@mcp.tool(structured_output=True)
async def recommend_papers(topic: str, limit: int = 5) -> Dict[str, Any]:
    """
    【功能】通过配置的学术搜索 provider 推荐与主题相关的论文。

    【与 Semantic Scholar 推荐的区别】
      这个走的是 AcademicSearchTool（可能是 mock provider，也可能是真实 API），
      而 semantic_scholar_recommendations 直接走 Semantic Scholar。
      两者的搜索 provider 和推荐算法不同。

    【参数】
      topic: 研究主题（如 "reinforcement learning"）
      limit: 返回数量，1-50，默认 5
    """
    bounded_limit = max(1, min(int(limit), 50))

    # 在原始 topic 后面追加关键词，让搜索结果更偏向综述和 benchmark
    result = await AcademicSearchTool().run( # 调用 AcademicSearchTool 实例的 run 方法
        query=f"{topic.strip()} related papers surveys benchmarks",
        max_results=bounded_limit,
    )

    # MCP 的错误处理方式：抛异常（而不是返回 ToolResult(success=False)）
    # → FastMCP 会自动捕获并返回标准的 MCP error 响应
    if not result.success:
        raise RuntimeError(result.error or "Academic recommendation search failed")

    return {
        "sources": result.data.get("results", []),
        "provider": result.data.get("provider", "academic"),
        "result_kind": "sources",
    }


# -------------------------------------------------------------------
# 3.2 semantic_scholar_search —— 语义学者论文搜索
# -------------------------------------------------------------------

@mcp.tool(structured_output=True)
async def semantic_scholar_search(
    query: str,
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    【功能】搜索 Semantic Scholar 学术图谱并返回归一化的论文数据。

    【内部流程】
      1. 创建 SemanticScholarSearchTool 实例
      2. 调用 tool.run(query, max_results)
      3. tool.run() → BaseTool.run() 做计时+兜底 → _arun() → SemanticScholarClient.search()
      4. 通过 _source_tool_payload() 将 ToolResult 转为 MCP 格式

    【注意】这里走的是 MCP 路径，不经过 LangGraph / Agent / Planner。
    """
    result = await SemanticScholarSearchTool().run(
        query=query,
        max_results=max_results,
    )
    return _source_tool_payload(result)


# -------------------------------------------------------------------
# 3.3 semantic_scholar_recommendations —— 语义学者论文推荐
# -------------------------------------------------------------------

@mcp.tool(structured_output=True)
async def semantic_scholar_recommendations(
    topic: str,
    limit: int = 5,
    positive_paper_ids: list[str] | None = None,   # Python 3.10+ 联合类型语法
    negative_paper_ids: list[str] | None = None,
) -> Dict[str, Any]:
    """
    【功能】基于主题和可选的种子论文 ID，推荐 Semantic Scholar 相关论文。

    【推荐机制】
      如果有 positive_paper_ids → 以它们为正面种子做推荐
      如果没有                 → 自动用 topic 搜一篇作为种子
      如果有 negative_paper_ids → 排除相似于这些论文的结果
    """
    result = await SemanticScholarRecommendationsTool().run(
        topic=topic,
        limit=limit,
        positive_paper_ids=positive_paper_ids or [],
        negative_paper_ids=negative_paper_ids or [],
    )
    # 把 ToolResult 转换为 MCP 格式
    return _source_tool_payload(result)


# -------------------------------------------------------------------
# 3.4 semantic_scholar_graph —— 语义学者论文图谱
# -------------------------------------------------------------------

@mcp.tool(structured_output=True)
async def semantic_scholar_graph(
    paper_query: str,
    relation: str = "details",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    【功能】返回一篇 Semantic Scholar 论文的详细信息、引用或被引列表。

    【paper_query 支持多种格式】
      - 论文标题: "Attention Is All You Need"
      - DOI:       "10.48550/arXiv.1706.03762"
      - arXiv ID:  "arXiv:1706.03762"
      - CorpusId:  "CorpusId:12345678"
      - paperId:   "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    【relation 参数】
      - "details"    → 论文详情
      - "citations"  → 哪些论文引用了它
      - "references" → 它引用了哪些论文
    """
    result = await SemanticScholarGraphTool().run(
        paper_query=paper_query,
        relation=relation,
        limit=limit,
    )
    return _source_tool_payload(result)


# ============================================================================
# 阶段四：MCP 独有的额外工具（不在项目 ToolRegistry 中）
# ============================================================================
#
# 以下两个工具是 MCP 专属的——它们不在 Agent 运行时中注册，只通过 MCP 暴露。
# 这样做的好处：
#   1. Agent 运行时保持简洁，只包含核心研究管道需要的工具
#   2. MCP 层可以灵活扩展新数据源，不影响现有管道
#   3. 用户可以在 Claude Desktop 中手动调用这些工具

# -------------------------------------------------------------------
# 4.1 crossref_search —— Crossref 元数据搜索
# -------------------------------------------------------------------

@mcp.tool(structured_output=True)
async def crossref_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    【功能】搜索 Crossref——作为以元数据为主的辅助学术数据源。

    【Crossref 是什么？】
      Crossref 是学术出版的 DOI 注册机构，维护了一个覆盖广泛的学术元数据
      数据库。与 Semantic Scholar 的区别：
        - Semantic Scholar: 侧重全文 + 引用关系 + AI 摘要
        - Crossref:        侧重出版元数据（DOI、期刊、发表日期等）

    【适用场景】
      - 通过 DOI 查找论文的出版信息
      - 补充 Semantic Scholar 可能遗漏的元数据
      - 作为辅助数据源进行交叉验证

    【配置方式】
      环境变量：
        CROSSREF_BASE_URL        → API 地址（默认 https://api.crossref.org/works）
        CROSSREF_MAILTO          → 你的邮箱（Crossref 礼貌策略，建议设置）
        CROSSREF_TIMEOUT_SECONDS → 超时时间（默认 15 秒）
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    bounded_limit = max(1, min(int(limit), 50))
    timeout = max(1.0, float(os.getenv("CROSSREF_TIMEOUT_SECONDS", "15")))

    # 构建 Crossref API 请求参数
    params = {"query": query, "rows": bounded_limit}
    mailto = os.getenv("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto   # Crossref 礼貌策略：标识你的身份

    # ---- 发送 HTTP 请求到 Crossref API ----
    async with build_httpx_client(timeout=timeout) as client:
        response = await client.get(
            os.getenv("CROSSREF_BASE_URL", "https://api.crossref.org/works"),
            params=params,
            headers={"User-Agent": "academic-research-copilot/1.2"},  # 标识客户端
        )
        response.raise_for_status()   # 非 2xx 状态码 → 抛异常

    # ---- 解析响应并归一化为项目 PaperSource 格式 ----
    items = response.json().get("message", {}).get("items", [])
    sources = [_crossref_item_to_source(item) for item in items]

    return {
        "sources": [source for source in sources if source],  # 过滤掉无效项
        "provider": "crossref",
        "result_kind": "sources",
    }


# -------------------------------------------------------------------
# 4.2 local_rag_search —— 本地 RAG 知识库搜索
# -------------------------------------------------------------------

@mcp.tool(structured_output=True)
async def local_rag_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    【功能】查询可选的本地 RAG HTTP 端点，将其返回的文本块归一化为 sources。

    【Local RAG 是什么？】
      RAG (Retrieval-Augmented Generation) = 检索增强生成。
      你可以搭建本地向量数据库（如 ChromaDB / Qdrant），把自己收集的论文
      预先索引进去。当搜索时，RAG 系统找到最相关的文本片段返回。

    【与 Semantic Scholar / Crossref 的区别】
      - Semantic Scholar / Crossref: 搜索公开的学术数据库
      - Local RAG:                   搜索你自己的私有论文库

    【配置方式】
      环境变量：
        LOCAL_RAG_ENDPOINT         → RAG HTTP 端点地址（必须设置，否则抛异常）
        LOCAL_RAG_TIMEOUT_SECONDS  → 超时时间（默认 20 秒）

    【RAG 端点协议约定】
      本工具假设 RAG 端点接收:
        POST {"query": "...", "top_k": 5}
      返回:
        {"sources": [...]}  或  {"results": [...]}  或  {"chunks": [...]}
      每个 item 需包含:
        - full_text / text / content / chunk: 文本内容
        - source_id / id: 唯一标识符
        - title / document_name: 文档标题（可选）
    """
    # ---- 检查配置 ----
    endpoint = os.getenv("LOCAL_RAG_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("LOCAL_RAG_ENDPOINT is not configured")
        # ↑ 没有配置端点 → 直接报错，这是预期行为（可选功能）

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    bounded_top_k = max(1, min(int(top_k), 50))
    timeout = max(1.0, float(os.getenv("LOCAL_RAG_TIMEOUT_SECONDS", "20")))

    # ---- 调用 RAG HTTP API ----
    async with build_httpx_client(timeout=timeout) as client:
        response = await client.post(
            endpoint,
            json={"query": query, "top_k": bounded_top_k},
        )
        response.raise_for_status()

    payload = response.json()

    # ---- 兼容多种返回格式 ----
    # 不同 RAG 框架的返回字段名可能不同，这里做兼容处理
    raw_items = payload.get("sources") or payload.get("results") or payload.get("chunks") or []

    # ---- 归一化为 PaperSource 格式 ----
    sources = [
        _local_item_to_source(item, index)
        for index, item in enumerate(raw_items[:bounded_top_k], start=1)
        if isinstance(item, dict)
    ]

    return {
        "sources": sources,
        "provider": "local_rag",
        "result_kind": "sources",
    }


# ============================================================================
# 阶段五：数据归一化辅助函数
# ============================================================================
#
# 以下函数负责将不同数据源的原始响应格式 → 项目统一的 PaperSource 字典。
# 这些逻辑之所以放在 MCP server 里而不是 tool 里，是因为这些数据源
# 只在 MCP 路径中使用，不需要污染核心的 ToolRegistry。

# -------------------------------------------------------------------
# 5.1 _crossref_item_to_source —— Crossref 数据 → PaperSource
# -------------------------------------------------------------------

def _crossref_item_to_source(item: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    【功能】将 Crossref API 返回的单个文献条目 → 项目 PaperSource 格式。

    【Crossref 响应结构示例】
      {
        "title": ["Attention Is All You Need"],
        "DOI": "10.48550/arXiv.1706.03762",
        "abstract": "<p>Transformer model...</p>",
        "issued": {"date-parts": [[2017, 6, 12]]},
        "author": [{"given": "Ashish", "family": "Vaswani"}, ...],
        "container-title": ["NeurIPS 2017"],
        "URL": "https://..."
      }

    【返回 None 的情况】
      如果条目没有 title 也没有 URL，说明数据不完整，返回 None。
      调用方会过滤掉这些 None 值。
    """

    # ---- 提取标题（Crossref 的 title 是列表，取第一个） ----
    title_values = item.get("title") or []
    title = str(title_values[0]).strip() if title_values else ""

    # ---- 提取 DOI 和 URL ----
    doi = str(item.get("DOI") or "").strip()
    url = f"https://doi.org/{doi}" if doi else str(item.get("URL") or "").strip()

    # 数据不完整 → 跳过
    if not title or not url:
        return None

    # ---- 提取摘要（去除 HTML 标签） ----
    abstract = _strip_markup(str(item.get("abstract") or ""))

    # ---- 提取发表年份 ----
    # Crossref 的 date-parts 格式: [[2017, 6, 12]] → 取第一个数组的第一个元素
    issued = (item.get("issued") or {}).get("date-parts") or []
    year = issued[0][0] if issued and issued[0] else None

    # ---- 提取作者列表 ----
    authors = []
    for author in item.get("author") or []:
        name = " ".join(
            part for part in [str(author.get("given") or ""), str(author.get("family") or "")]
            if part
        ).strip()
        if name:
            authors.append(name)

    # ---- 提取期刊/会议名称 ----
    venue_values = item.get("container-title") or []
    venue = str(venue_values[0]).strip() if venue_values else ""

    # ---- 构建统一的 PaperSource 字典 ----
    return {
        "source_id": _stable_id("crossref", doi or url),
        # ↑ 用 SHA256 生成稳定 ID，避免 DOI 中的特殊字符问题
        "title": title,
        "url": url,
        "snippet": abstract[:1000],                     # 截断到 1000 字符
        "full_text": abstract,
        "authors": authors,
        "year": year,
        "venue": venue,
        "source_type": "paper",
        "provider": "crossref",
        "doi": doi or None,
        "content_source": "crossref_abstract" if abstract else "metadata_only",
        # ↑ 标记内容来源：有摘要 → "crossref_abstract"，没有 → "metadata_only"
    }


# -------------------------------------------------------------------
# 5.2 _local_item_to_source —— Local RAG 数据 → PaperSource
# -------------------------------------------------------------------

def _local_item_to_source(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    【功能】将本地 RAG 返回的文本块 → 项目 PaperSource 格式。

    【兼容的字段名】
      full_text / text / content / chunk → 任意一个都可以
      source_id / id                     → 任意一个都可以
      title / document_name              → 任意一个都可以

    这种多字段兼容设计是因为不同的 RAG 框架（ChromaDB、Qdrant、自定义方案）
    返回的字段名各不相同。
    """

    # ---- 提取文本（兼容 4 种常见字段名） ----
    text = str(
        item.get("full_text") or item.get("text") or
        item.get("content") or item.get("chunk") or ""
    ).strip()

    # ---- 提取标识符 ----
    original_id = str(item.get("source_id") or item.get("id") or index)
    url = str(item.get("url") or f"local://rag/{original_id}")
    # ↑ 如果没有 URL，生成一个内部引用链接

    # ---- 提取标题 ----
    title = str(item.get("title") or item.get("document_name") or f"Local document {index}")

    return {
        "source_id": _stable_id("local_rag", original_id),
        "title": title,
        "url": url,
        "snippet": text[:1000],
        "full_text": text,
        "authors": item.get("authors") or [],
        "year": item.get("year"),
        "venue": str(item.get("venue") or "Local Knowledge Base"),
        "source_type": "local_document",         # 标记为本地文档
        "provider": "local_rag",
        "content_source": "local_rag_chunk",
    }


# -------------------------------------------------------------------
# 5.3 _strip_markup —— 清除 HTML 标签和实体
# -------------------------------------------------------------------

def _strip_markup(value: str) -> str:
    """
    【功能】清除字符串中的 HTML 标签和 HTML 实体。

    【处理步骤】
      1. 正则去掉所有 <...> 标签 → "Transformer model..."
      2. html.unescape 解码实体      → "&amp;" → "&", "&lt;" → "<"
      3. 合并多余空白字符            → 多个空格/换行 → 单个空格

    【为什么需要这个？】
      Crossref 的摘要字段经常包含 HTML 标记（如 <jats:p>、<i> 等），
      这些标记对 LLM 没有意义，需要清理。
    """
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


# -------------------------------------------------------------------
# 5.4 _stable_id —— 生成稳定的 source_id
# -------------------------------------------------------------------

def _stable_id(namespace: str, value: str) -> str:
    """
    【功能】生成稳定的、可复现的 source_id。

    【为什么不用 UUID？】
      UUID 是随机的——同一个 DOI 每次生成的 ID 都不同，导致同一篇论文
      被重复入库和去重失败。SHA256 哈希保证同一输入总是产生同一输出。

    【格式】
      {namespace前3位}_{SHA256 前12位十六进制}
      例如: "cro_a1b2c3d4e5f6"

      namespace 前 3 位作为前缀，方便人类快速识别来源。
    """
    digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()[:12]
    return f"{namespace[:3]}_{digest}"


# -------------------------------------------------------------------
# 5.5 _source_tool_payload —— 统一的 MCP 返回格式
# -------------------------------------------------------------------

def _source_tool_payload(result: ToolResult) -> Dict[str, Any]:
    """
    【功能】将项目的 ToolResult → MCP 标准返回格式。

    【这是关键桥接函数】
      项目内部工具返回 ToolResult（项目内部契约），
      MCP 客户端期望的是普通 dict（MCP 协议契约）。
      这个函数完成两者的转换。

    【为什么 Semantic Scholar 工具都通过这个函数返回？】
      1. 统一错误处理：ToolResult 失败 → 抛 MCP RuntimeError
      2. 统一字段映射：保证 sources / results / provider 字段都存在
      3. 透传 metadata：tool_metadata 包含延迟、重试次数等诊断信息
    """
    if not result.success:
        # 把 ToolResult 的 error 转为 MCP 异常
        raise RuntimeError(result.error or f"{result.tool_name} failed")

    data = result.data if isinstance(result.data, dict) else {}

    # 兼容两种字段名：sources 和 results（不同工具可能用不同名字）
    sources = data.get("sources") or data.get("results") or []

    # 展开原始 data 的所有字段，并确保 sources / results 双份存在
    return {
        **data,                                       # 透传所有原始字段
        "sources": sources,
        "results": sources,                          # 两份一样，兼容不同客户端的字段习惯
        "provider": data.get("provider", "semantic_scholar"),
        "result_kind": "sources",                    # 标记返回类型（方便客户端分流处理）
        "tool_metadata": result.metadata,            # 诊断信息（延迟、重试次数等）
    }


# ============================================================================
# 阶段六：启动入口
# ============================================================================

def main() -> None:
    """
    【MCP 服务器启动入口】

    通过环境变量 MCP_SERVER_TRANSPORT 选择通信方式：
      - "stdio" (默认): 标准输入输出，适合 Claude Desktop 等本地客户端
      - "sse":          Server-Sent Events over HTTP，适合远程调用和调试

    启动命令：
      # stdio 模式（默认）
      python -m app.mcp.academic_server

      # SSE 模式
      MCP_SERVER_TRANSPORT=sse python -m app.mcp.academic_server
    """
    transport = os.getenv("MCP_SERVER_TRANSPORT", "stdio")
    mcp.run(transport=transport)
    # ↑ FastMCP.run() 会阻塞当前线程，启动 MCP 协议的事件循环


if __name__ == "__main__":
    main()


# ============================================================================
# 阶段七：架构总结
# ============================================================================
"""
【MCP Server 的工具全景】

  ┌─────────────────────────────────────────────────────────┐
  │              MCP Server: academic-research-tools         │
  │                                                         │
  │  ┌─ 来自项目 Tool 层的工具（共享） ──────────────────┐  │
  │  │ recommend_papers                 AcademicSearchTool│  │
  │  │ semantic_scholar_search          SemanticScholar.. │  │
  │  │ semantic_scholar_recommendations SemanticScholar.. │  │
  │  │ semantic_scholar_graph           SemanticScholar.. │  │
  │  └────────────────────────────────────────────────────┘  │
  │                                                         │
  │  ┌─ MCP 独有的工具 ──────────────────────────────────┐  │
  │  │ crossref_search            直接调 Crossref API      │  │
  │  │ local_rag_search           直接调本地 RAG 端点      │  │
  │  └────────────────────────────────────────────────────┘  │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

【两条调用路径的对比】

  路径 A: Agent 运行时（LangGraph）
    用户问题 → Controller → Planner → Worker → Registry.get(tool) → tool.run()
    → SemanticScholarClient → API
    特点: 有意图分类、任务编排、重试循环、引用校验

  路径 B: MCP 直连（本文件）
    用户问题 → MCP Client → FastMCP → tool.run()
    → SemanticScholarClient → API
    特点: 无 Agent 开销、跨框架调用、手动触发

【什么情况下用 MCP？】
  - 在 Claude Desktop / Continue 中直接搜索论文
  - 把学术搜索能力集成到其他 AI 应用中
  - 开发者调试：手动调用工具查看原始返回结果

【什么情况下用 Agent 运行时？】
  - 需要多阶段研究管道（搜索 → 阅读 → 分析 → 引用）
  - 需要自动引用验证
  - 需要生成结构化研究报告

【MCP 工具配置示例（Claude Desktop 的 claude_desktop_config.json）】
  {
    "mcpServers": {
      "academic-research": {
        "command": "python",
        "args": ["-m", "app.mcp.academic_server"],
        "env": {
          "SEMANTIC_SCHOLAR_API_KEY": "your-key-here",
          "CROSSREF_MAILTO": "your-email@example.com",
          "LOCAL_RAG_ENDPOINT": "http://localhost:8000/query"
        }
      }
    }
  }
"""
