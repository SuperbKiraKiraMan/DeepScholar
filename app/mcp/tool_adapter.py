"""
================================================================================
MCP 工具适配器 —— 让外部 MCP 工具"伪装"成项目内部的 BaseTool
================================================================================

【本文件的定位】
  这是一个适配器（Adapter Pattern）实现。它让外部 MCP Server 暴露的工具
  能够以项目内部 BaseTool 的身份被 ToolRegistry 注册、被 Agent 调用。

【核心问题：为什么需要适配器？】

  项目内部工具（如 SemanticScholarSearchTool）:
    - 继承 BaseTool
    - 实现 name / description / input_schema / _arun()
    - 由 ToolRegistry 统一管理

  外部 MCP 工具（如通过 FastMCP 暴露的工具）:
    - 运行在另一个进程
    - 通过 MCP 协议通信（stdio / SSE）
    - 有自己的 name / description / inputSchema

  问题：Agent 和 ToolRegistry 只认识 BaseTool，不认识 MCP 工具。
  方案：MCPToolAdapter 把 MCP 工具包装成 BaseTool 的模样。

【数据流向 —— 一次 MCP 工具调用的完整路径】

  Agent (LLMWorker)
    │
    ▼
  ToolRegistry.get("some_mcp_tool")
    │
    ▼
  MCPToolAdapter._arun(**kwargs)          ← 你在这里（适配器）
    │
    │  self._manager.call_tool(name, kwargs)
    ▼
  MCPToolManager (mcp_manager.py)        ← MCP 工具管理器
    │
    │  stdio / SSE 通信
    ▼
  MCP Server 进程 (如 academic_server.py) ← 实际执行工具
    │
    ▼
  返回结果 → ToolResult → Agent

【适配器 vs 原生工具对比】

  ┌──────────────────────────────────────────────────────────┐
  │                  BaseTool (抽象契约)                       │
  │  name / description / input_schema / _arun()             │
  └──────────────┬────────────────┬──────────────────────────┘
                 │                │
    ┌────────────▼──────┐  ┌─────▼──────────────┐
    │ SemanticScholar.. │  │ MCPToolAdapter      │
    │ (原生工具)         │  │ (适配器 / 代理)     │
    │                    │  │                     │
    │ _arun() 直接调     │  │ _arun() 委托给      │
    │ SemanticScholar-   │  │ MCPToolManager      │
    │ Client             │  │ 跨进程调远程工具     │
    └────────────────────┘  └─────────────────────┘
"""

# ============================================================================
# 阶段一：导入
# ============================================================================

from typing import Any, Dict, Iterable

from app.tools.base import BaseTool, ToolResult


# ============================================================================
# 阶段二：MCPToolAdapter —— 适配器核心类
# ============================================================================

class MCPToolAdapter(BaseTool):
    """
    【适配器模式】将外部的 MCP 工具包装为项目内部 BaseTool 的子类。

    【类比 —— 如果你熟悉设计模式】
      这就是经典的 Adapter Pattern（适配器模式，也叫 Wrapper 模式）。
      - Target (目标接口):  BaseTool
      - Adaptee (被适配者): MCP 工具（由 MCPToolManager 管理）
      - Adapter (适配器):   MCPToolAdapter

      类比电源插头转换器：
        - 墙上的插座 = MCP 工具（外部接口）
        - 你的充电器 = BaseTool（内部接口）
        - 转换插头   = MCPToolAdapter（适配器）

    【关键属性说明】
      - _manager:     MCPToolManager 实例，负责实际的 MCP 协议通信
      - _public_name: 在 ToolRegistry 中的注册名（可以加前缀避免冲突）
      - server_name:  MCP Server 的名称（如 "academic-research-tools"）
      - remote_name:  在 MCP Server 中的原始工具名（如 "semantic_scholar_search"）
      - result_kind:  返回结果的类型标识（如 "sources"），用于上游路由
    """

    def __init__(
        self,
        manager: Any,               # MCPToolManager 实例（负责 MCP 协议通信）
        public_name: str,           # 在 ToolRegistry 中注册的名字（如 "mcp_academic_search"）
        server_name: str,           # MCP Server 的名字（如 "academic-research-tools"）
        remote_name: str,           # MCP 工具在 Server 中的原始名字（如 "semantic_scholar_search"）
        description: str,           # 工具描述（从 MCP Server 的 tool schema 中获取）
        input_schema: Dict[str, Any],  # 输入参数的 JSON Schema（从 MCP Server 获取）
        task_types: Iterable[str],  # 任务类型标签（如 ("search", "read")）
        result_kind: str,           # 结果类型标识（如 "sources"），从 MCP 工具的返回字段获取
        capability_metadata: Dict[str, Any] = None,
    ):
        super().__init__()
        self._manager = manager
        self._public_name = public_name
        self.server_name = server_name
        self.remote_name = remote_name
        self._description = description
        self._input_schema = input_schema
        self.task_types = tuple(task_types)    # 确保不可变
        self.result_kind = result_kind
        # 关键步骤：外部 MCP 默认视为网络能力，写入与破坏性必须由配置显式声明。
        self.capability_metadata = {
            "network_access": True,
            "external_write": False,
            "destructive": False,
            "resource_scope": "none",
            **dict(capability_metadata or {}),
        }

    # ------------------------------------------------------------------
    # 2.1 工具元信息 —— 全部透传自 MCP Server
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        【工具名称】
        返回 ToolRegistry 中的注册名（public_name），而非 MCP Server 中的
        原始名（remote_name）。这样可以：
          1. 加命名空间前缀避免冲突（如 "mcp_academic_search"）
          2. 多个 MCP Server 有同名工具时，通过不同 public_name 区分
        """
        return self._public_name

    @property
    def description(self) -> str:
        """
        【工具描述】
        在原始描述前面加上 [MCP:server_name] 标签，让 LLM 知道这个工具
        来自外部 MCP Server，而非项目内置工具。

        截断到 1024 字符——防止过长的描述影响 LLM 的 context window。
        """
        return f"[MCP:{self.server_name}] {self._description}"[:1024]

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        【输入参数 Schema】
        直接透传 MCP Server 提供的 JSON Schema，不做修改。
        这个 schema 在 _arun() 中被用于参数校验（由 ToolRegistry 的
        validate_tool_args_against_schema() 执行）。
        """
        return self._input_schema

    # ------------------------------------------------------------------
    # 2.2 核心执行 —— 委托给 MCPToolManager 跨进程调用
    # ------------------------------------------------------------------

    async def _arun(self, **kwargs) -> ToolResult:
        """
        【核心执行方法 —— 适配器的关键逻辑】

        这是 BaseTool 模板方法模式中的"子类实现"部分。
        BaseTool.run() 负责计时 + 异常兜底 + 事件触发，
        本方法只负责：把参数转发给 MCPToolManager，然后构建 ToolResult。

        【调用链】
          1. BaseTool.run(**kwargs)
               │  计时 + try/except + 事件
               ▼
          2. MCPToolAdapter._arun(**kwargs)         ← 你在这里
               │  委托给 manager.call_tool()
               ▼
          3. MCPToolManager.call_tool(name, kwargs)
               │  找到对应的 MCP session
               │  通过 stdio/SSE 发送 MCP tools/call 请求
               ▼
          4. MCP Server 进程 执行工具
               │  返回 MCP 标准响应
               ▼
          5. 构建 ToolResult → Agent

        【metadata 的作用】
          返回的 metadata 标记了 protocol="mcp"，方便可观测性系统
          区分"内部工具调用"和"MCP 工具调用"的延迟统计。
        """
        # ---- 委托给 MCPToolManager 执行远程调用 ----
        response = await self._manager.call_tool(
            self.name,    # 用 public_name 去调用（manager 内部做名字映射）
            kwargs        # 透传所有参数
        )

        # ---- 将 MCP 响应包装为 ToolResult ----
        # response 已经是统一格式的字典: {"success": bool, "data": ..., "error": ...}
        return ToolResult(
            success=response["success"],
            tool_name=self.name,
            data=response.get("data"),
            error=response.get("error", ""),
            metadata={
                "protocol": "mcp",                # 标记：来自 MCP 协议
                "server": self.server_name,       # 哪个 MCP Server
                "remote_tool": self.remote_name,  # 远程的原始工具名
                "result_kind": self.result_kind,  # 结果类型（用于上游路由）
            },
        )


# ============================================================================
# 阶段三：架构总结
# ============================================================================
"""
【MCPToolAdapter 在整个系统中的位置】

  ┌─────────────────────────────────────────────────────────┐
  │                    Agent 运行时                          │
  │                                                         │
  │  LLMWorker / Worker                                     │
  │    │                                                    │
  │    ▼                                                    │
  │  ToolRegistry                                           │
  │    │                                                    │
  │    ├── 原生工具 (BaseTool 子类)                          │
  │    │   └── SemanticScholarSearchTool                    │
  │    │       _arun() → SemanticScholarClient → API        │
  │    │                                                    │
  │    ├── MCP 适配工具 (MCPToolAdapter)  ← 本文件          │
  │    │   └── MCPToolAdapter                               │
  │    │       _arun() → MCPToolManager.call_tool()         │
  │    │                    │                               │
  │    │                    ▼ (跨进程: stdio/SSE)           │
  │    │              MCP Server 进程                        │
  │    │              └── semantic_scholar_search(...)      │
  │    │                                                    │
  │    └── 其他适配工具...                                   │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

【为什么这个文件如此简洁（56 行）却很重要？】

  它实现了"开放-封闭原则"（Open-Closed Principle）的关键一环：

  - ToolRegistry 对扩展开放：通过 MCPToolAdapter，任何 MCP 工具都能注册进来
  - ToolRegistry 对修改封闭：不需要改 ToolRegistry 的代码，不需要改 BaseTool 的接口

  新增一个外部工具只需：
    1. 在某个 MCP Server 中实现工具
    2. MCPToolManager 自动发现并创建 MCPToolAdapter 实例
    3. 注册到 ToolRegistry
  全程不需要改 ToolRegistry、BaseTool、Agent 的任何代码。

【_arun() 为什么不做重试？】

  原生工具（SemanticScholarSearchTool）的 _arun() 委托给 SemanticScholarClient，
  Client 内部有完整的重试引擎（指数退避 + Retry-After + 预算控制）。

  MCPToolAdapter 的 _arun() 委托给 MCPToolManager.call_tool()。
  重试逻辑不在这里，因为：
    1. MCP 协议层可能有自己的重试策略（由 MCPToolManager 或 FastMCP 处理）
    2. BaseTool.run() 已经有统一的异常兜底（asyncio.CancelledError / Exception）
    3. 适配器保持"薄"——只做转换，不做策略，符合单一职责原则

【与 academic_server.py 的关系】

  academic_server.py          MCPToolAdapter (本文件)
  ───────────────────────     ──────────────────────
  是 MCP "服务端"             是 MCP "客户端适配器"
  把工具暴露出去               把外部工具接进来
  FastMCP 框架                 BaseTool 接口
  运行在独立进程               运行在 Agent 进程内
"""
