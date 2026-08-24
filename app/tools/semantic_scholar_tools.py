"""
================================================================================
Semantic Scholar 工具集 —— Agent 可调用的学术搜索工具
================================================================================

【总览】
本模块定义了 3 个面向 LLM Agent 的工具（Tool），底层都调用 Semantic Scholar
（语义学者）的 Academic Graph API。每个工具都是一个"能力单元"——Agent 根据
用户意图自动选择合适的工具来执行。

【3 个工具的关系】
  SemanticScholarSearchTool        全文搜索（类似 Google Scholar 搜索框）
  SemanticScholarRecommendationsTool  论文推荐（类似"相关论文"推荐）
  SemanticScholarGraphTool           论文图谱（查看某篇论文的详情/引用/被引）

【架构类比 —— 如果你熟悉 Spring Boot】
  BaseTool      ≈  interface Tool（统一契约）
  ToolResult    ≈  R<T> 统一响应包装（success + data + error）
  本文件的 3 个类 ≈ Controller 层 —— 定义接口形态，真正干活的是 SemanticScholarClient

【为什么这样设计？】
  1. Agent 只需要知道"有什么工具可用"和"怎么调用"，不需要关心 HTTP 细节
  2. 每个工具的 input_schema 就是给 LLM 看的 function-calling 参数定义
  3. 统一返回 ToolResult，上层无需 try-catch
"""

# ---------------------------------------------------------------------------
# 阶段一：导入依赖
# ---------------------------------------------------------------------------
from typing import Any, Dict

# BaseTool: 所有工具的抽象父类，定义了 name / description / input_schema / _arun 契约
from app.tools.base import BaseTool, ToolResult
# SemanticScholarClient: 真正发送 HTTP 请求、处理重试、归一化数据的底层客户端
from app.tools.semantic_scholar_provider import SemanticScholarClient


# ============================================================================
# 工具一：SemanticScholarSearchTool —— 学术论文搜索
# ============================================================================
class SemanticScholarSearchTool(BaseTool):
    """
    【功能】在 Semantic Scholar 的学术图谱中搜索论文。

    【使用场景】
      - "帮我找关于 transformer attention mechanism 的论文"
      - "搜索 2020 年后关于 few-shot learning 的高引论文"
      - "找一下 GPT 相关的论文"

    【与其他工具的区别】
      - 这是"关键词全文搜索"，类似 Google Scholar 的搜索框
      - 适合：宽泛的主题搜索，不知道具体论文标题时使用
      - 不适合：已知某篇论文想查它的引用关系 → 用 SemanticScholarGraphTool
    """

    # task_types: 标记这个工具属于哪种任务类型
    # "search" 表示它用于信息检索类任务
    task_types = ("search",)

    # ------------------------------------------------------------------
    # 阶段二：工具元信息 —— Agent 如何认识这个工具
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        【工具名称】
        这是 Agent 在 function calling 时使用的唯一标识符。
        命名规范：{数据源}_{动作}，如 semantic_scholar_search。
        """
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        """
        【工具描述 —— 给 LLM 看的"使用说明书"】
        LLM 会根据这段描述判断"当前用户的问题是否应该用这个工具"。
        所以描述要说明：能做什么、返回什么信息、适合什么场景。
        """
        return (
            "Search Semantic Scholar's Academic Graph for papers. Best for topic "
            "search with author, citation count, reference count, DOI, venue, "
            "abstract, and open-access metadata. A result without an abstract or "
            "traceable snippet is discovery metadata, not evidence."
        )

    @property
    def input_schema(self) -> Dict[Any, Any]:
        """
        【输入参数定义 —— JSON Schema 格式】
        这是给 LLM 看的"参数说明书"。LLM 会根据这个 schema 自动从用户问题中
        提取参数值。例如用户说"找 3 篇关于 BERT 的论文" →
        LLM 自动填入 {"query": "BERT", "max_results": 3}。

        参数说明：
          - query (必填): 搜索关键词，至少 2 个字符
          - max_results (可选): 返回结果数，1-50，默认 5
        """
        return {
            "type": "object",
            "properties": {
                "query" : {
                  "type" : "string",
                  "minLength" : 2
                },                                                   # 搜索关键词，至少 2 个字符
                "max_results" : {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,                                   # 最多 50 条
                    "default": 5,                                    # 默认返回 5 条
                },
            },
            "required": ["query"],                                   # query 是必填参数
        }

    # ------------------------------------------------------------------
    # 阶段三：核心执行逻辑
    # ------------------------------------------------------------------

    async def _arun(self, **kwargs) -> ToolResult:
        """
        【异步执行入口 —— 子类只需实现这个方法】
        BaseTool.run() 会自动做：计时、异常兜底、事件触发。
        所以子类的 _arun() 只需要关注业务逻辑本身。

        执行流程：
          1. 从 kwargs 中提取 query 参数
          2. 委托给 SemanticScholarClient().search() 发送 HTTP 请求
          3. 返回统一的 ToolResult（由 client 内部构建）

        类比：这就是 Spring Service 的 @Service 方法，
        BaseTool.run() 相当于 Spring AOP 切面（事务/日志/异常处理）。
        """
        return await SemanticScholarClient().search(
            # 从 kwargs 中提取 query，如果为空则传空字符串给 client 处理
            query=str(kwargs.get("query") or ""),
            # max_results 默认 5，和 schema 中的 default 保持一致
            max_results=kwargs.get("max_results", 5),
        )


# ============================================================================
# 工具二：SemanticScholarRecommendationsTool —— 论文推荐
# ============================================================================
class SemanticScholarRecommendationsTool(BaseTool):
    """
    【功能】根据主题或种子论文，推荐相关论文。

    【使用场景】
      - "这篇论文有什么类似的推荐吗？" → 传入这篇的 paper_id
      - "我对 reinforcement learning 感兴趣，推荐几篇必读论文"
      - "我喜欢论文 A 和 B，但不喜欢 C，帮我推荐类似的"

    【推荐机制 —— 正负种子论文】
      - positive_paper_ids: "我喜欢这些论文，推荐类似的"  ← 正面种子
      - negative_paper_ids: "我不喜欢这些，别推荐类似的"  ← 负面种子
      - 如果都不传，工具会自动用 topic 搜索一篇作为种子

    【与 Search 的区别】
      Search 是"匹配关键词"，Recommendations 是"基于论文相似度推荐"。
      前者适合"我不知道具体有什么"，后者适合"我找到一篇不错的，想要更多类似的"。
    """

    task_types = ("search",)

    @property
    def name(self) -> str:
        return "semantic_scholar_recommendations"

    @property
    def description(self) -> str:
        return (
            "Recommend related papers with Semantic Scholar. Accepts a topic and "
            "optionally positive/negative seed paper IDs. If no seed is supplied, "
            "the tool resolves a seed paper from the topic first. Primarily a "
            "discovery expansion capability; metadata-only recommendations are not "
            "evidence."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        参数说明：
          - topic (必填):      研究主题，用于在没有种子论文时自动找种子
          - limit (可选):      返回推荐数量，1-50，默认 5
          - positive_paper_ids: 正面种子论文 ID 列表（我希望推荐的论文像这些）
          - negative_paper_ids: 负面种子论文 ID 列表（我不希望推荐的论文像这些）
        """
        return {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "minLength": 2},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                },
                # 正面种子：字符串数组，如 ["paperId1", "paperId2"]
                "positive_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                # 负面种子：字符串数组
                "negative_paper_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["topic"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        """
        执行流程：
          1. 如果用户提供了 positive_paper_ids → 直接用它们作为种子去推荐
          2. 如果没有提供 → 先用 topic 做一次搜索，拿第一篇作为种子
          3. 用种子 paper IDs 调用推荐 API，返回相似论文列表
        """
        return await SemanticScholarClient().recommend(
            topic=str(kwargs.get("topic") or ""),
            limit=kwargs.get("limit", 5),
            # 正面/负面种子是可选的，可能为 None
            positive_paper_ids=kwargs.get("positive_paper_ids"),
            negative_paper_ids=kwargs.get("negative_paper_ids"),
        )


# ============================================================================
# 工具三：SemanticScholarGraphTool —— 论文图谱查询
# ============================================================================
class SemanticScholarGraphTool(BaseTool):
    """
    【功能】解析并查看一篇论文的详细信息及其文献图谱。

    【使用场景】
      - "这篇论文的详细信息是什么？"          → relation="details"
      - "有哪些论文引用了这篇？"               → relation="citations"
      - "这篇论文引用了哪些参考文献？"          → relation="references"

    【paper_query 支持多种输入格式】
      这是本工具最灵活的地方 —— 你可以传入：
        - 论文标题:  "Attention Is All You Need"
        - DOI:       "10.48550/arXiv.1706.03762"
        - arXiv ID:  "arXiv:1706.03762"
        - CorpusId:  "CorpusId:12345678"
        - paperId:   "204e3073870fae3d05bcbc2f6a8e263d9b72e776" (40 位十六进制)

      工具内部会自动识别格式并解析成 Semantic Scholar paperId。

    【relation 参数 —— 三种查询模式】
      details    → 返回该论文的详细信息（标题、摘要、作者、引用数等）
      citations  → 返回引用了该论文的论文列表（谁引了它？）
      references → 返回该论文引用的参考文献列表（它引了谁？）
    """

    task_types = ("search",)

    @property
    def name(self) -> str:
        return "semantic_scholar_graph"

    @property
    def description(self) -> str:
        return (
            "Inspect one Semantic Scholar paper and its literature graph. Resolve "
            "a title, DOI, arXiv ID, CorpusId, or Semantic Scholar paperId, then "
            "return paper details, papers that cite it, or its references. Citation "
            "and reference edges are discovery metadata unless returned papers also "
            "contain traceable text."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        参数说明：
          - paper_query (必填): 论文标识符，支持标题/DOI/arXiv/CorpusId/paperId
          - relation (可选):    查询关系类型
              * "details"    → 论文详情（默认）
              * "citations"  → 谁引用了这篇论文
              * "references" → 这篇论文引用了谁
          - limit (可选):      返回结果数，1-50，默认 5
        """
        return {
            "type": "object",
            "properties": {
                "paper_query": {
                    "type": "string",
                    "minLength": 2,
                    "description": "Paper title, DOI, arXiv ID, CorpusId, or paperId",
                },
                "relation": {
                    "type": "string",
                    # enum 限制了 relation 只能是这三个值之一
                    "enum": ["details", "citations", "references"],
                    "default": "details",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                },
            },
            "required": ["paper_query"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        """
        执行流程：
          1. 接收 paper_query（可能是标题、DOI、arXiv 等任意格式）
          2. SemanticScholarClient 内部自动解析 paper_query → paperId
          3. 根据 relation 参数，请求对应的 API 端点
          4. 返回结构化的论文数据
        """
        return await SemanticScholarClient().paper_graph(
            paper_query=str(kwargs.get("paper_query") or ""),
            relation=str(kwargs.get("relation") or "details"),
            limit=kwargs.get("limit", 5),
        )


# ============================================================================
# 阶段四：总结 —— 整个工具设计的架构要点
# ============================================================================
"""
【文件结构总结】

  semantic_scholar_tools.py    ← 你在这里（本文件）
  ├── SemanticScholarSearchTool        搜索工具
  ├── SemanticScholarRecommendationsTool  推荐工具
  └── SemanticScholarGraphTool         图谱工具

  semantic_scholar_provider.py  ← 底层实现（HTTP 请求、重试、数据归一化）
  └── SemanticScholarClient
       ├── search()         → 调用 /paper/search API
       ├── recommend()      → 调用 /recommendations API
       ├── paper_graph()    → 调用 /paper/{id} + /paper/{id}/citations + /paper/{id}/references
       ├── _resolve_paper_id() → 智能解析论文标识符（DOI/arXiv/CorpusId/paperId）
       └── _request_json()     → 统一 HTTP 客户端（自动重试、错误处理）

  base.py                      ← 基础抽象
  ├── BaseTool                 → 所有工具的抽象父类
  └── ToolResult               → 统一返回结构 {success, data, error, latency_ms}

【数据流向】
  用户问题 → LLM Agent 选择工具 → BaseTool.run() 计时+兜底 → _arun()
  → SemanticScholarClient HTTP 请求 → Semantic Scholar API
  → 原始 JSON 归一化为 PaperSource 字典 → ToolResult → Agent 解读 → 用户

【扩展新工具的方式】
  1. 继承 BaseTool
  2. 实现 name / description / input_schema 三个属性
  3. 实现 _arun() 方法
  4. 注册到 tool registry（让 Agent 能发现它）
  完事。HTTP、重试、异常处理都不用管。
"""
