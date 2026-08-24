"""MCP 生命周期、路由、命名空间和工具注册表集成。"""

import asyncio
import hashlib
import re
from typing import Any, Callable, Dict

from app.mcp.client import MCPClientConnection
from app.mcp.config import (
    MCPServerConfig,
    get_allowed_servers,
    get_mcp_config_path,
    get_mcp_connect_timeout,
    get_mcp_tool_timeout,
    load_mcp_server_configs,
    mcp_enabled,
    mcp_fail_open,
)
from app.mcp.tool_adapter import MCPToolAdapter
from app.tools.registry import ToolRegistry


class MCPManager:
    def __init__(
        self,
        connection_factory: Callable[..., MCPClientConnection] = MCPClientConnection,
    ):
        self._connection_factory = connection_factory
        self._connections: Dict[str, MCPClientConnection] = {}
        self._routes: Dict[str, tuple[str, str]] = {}
        self._adapters: Dict[str, MCPToolAdapter] = {}
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._registry: ToolRegistry | None = None
        self._initialized = False

    async def initialize(self, registry: ToolRegistry | None = None) -> None:
        if self._initialized:
            return
        self._registry = registry or ToolRegistry.get_instance()
        self._initialized = True
        if not mcp_enabled():
            return

        path = get_mcp_config_path()
        try:
            configs = load_mcp_server_configs(path)
        except Exception as exc:
            self._servers["_config"] = {"status": "failed", "error": _safe_error(exc)}
            if not mcp_fail_open():
                raise
            return

        allowed_servers = get_allowed_servers()
        for config in configs:
            if not config.enabled or (allowed_servers and config.name not in allowed_servers):
                self._servers[config.name] = {"status": "disabled", "tools": []}
                continue
            await self._initialize_server(config)

    async def _initialize_server(self, config: MCPServerConfig) -> None:
        connection = self._connection_factory(
            config,
            read_timeout_seconds=get_mcp_tool_timeout(),
        )
        try:
            async with asyncio.timeout(get_mcp_connect_timeout()):
                await connection.connect()
                descriptors = await connection.list_tools()
            self._connections[config.name] = connection

            names = []
            for descriptor in descriptors:
                policy = config.tools.get(descriptor.name)
                if config.tools and policy is None:
                    continue
                if policy is not None and not policy.enabled:
                    continue
                public_name = make_public_tool_name(config.name, descriptor.name)
                if self._registry.get(public_name) is not None:
                    raise RuntimeError(f"MCP tool name collision: {public_name}")
                adapter = MCPToolAdapter(
                    manager=self,
                    public_name=public_name,
                    server_name=config.name,
                    remote_name=descriptor.name,
                    description=descriptor.description or descriptor.name,
                    input_schema=descriptor.inputSchema or {
                        "type": "object", "properties": {}, "required": []
                    },
                    task_types=(policy.task_types if policy else ("search",)),
                    result_kind=(policy.result_kind if policy else "generic"),
                    capability_metadata={
                        "network_access": policy.network_access if policy else True,
                        "external_write": policy.external_write if policy else False,
                        "destructive": policy.destructive if policy else False,
                        "resource_scope": policy.resource_scope if policy else "none",
                    },
                )
                self._registry.register(adapter)
                self._routes[public_name] = (config.name, descriptor.name)
                self._adapters[public_name] = adapter
                names.append(public_name)

            self._servers[config.name] = {
                "status": "connected",
                "transport": config.transport,
                "tools": names,
            }
        except Exception as exc:
            await connection.close()
            self._servers[config.name] = {
                "status": "failed",
                "transport": config.transport,
                "tools": [],
                "error": _safe_error(exc),
            }
            if not mcp_fail_open():
                raise
    # 调用MCP工具
    # @param public_name: MCP工具的公共名称
    # @param arguments: MCP工具的参数
    # @return: MCP工具的执行结果
    async def call_tool(self, public_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # 路由查找：public_name -> server_name, remote_name
        route = self._routes.get(public_name)
        # → ("academic_research_tools", "semantic_scholar_recommendations")
        if route is None:
            return {"success": False, "data": None, "error": "Unknown MCP tool"}
        server_name, remote_name = route
        # 获取 MCP 连接
        connection = self._connections.get(server_name)
        if connection is None:
            return {"success": False, "data": None, "error": "MCP server unavailable"}
        try:
            return await asyncio.wait_for(
                connection.call_tool(remote_name, arguments),
                #     ↑
                #     发送 JSON-RPC 消息到独立进程的 MCP Server
                timeout=get_mcp_tool_timeout(),
            )
        except asyncio.TimeoutError:
            return {"success": False, "data": None, "error": "MCP tool execution timed out"}
        except Exception as exc:
            return {"success": False, "data": None, "error": _safe_error(exc)}

    async def shutdown(self) -> None:
        if self._registry is not None:
            for name, adapter in self._adapters.items():
                self._registry.unregister(name, expected_tool=adapter)
        # MCP SDK transports own AnyIO cancel scopes and must be closed by the
        # same task that entered them. Do not wrap these closes in gather().
        for connection in self._connections.values():
            try:
                await connection.close()
            except (Exception, asyncio.CancelledError):
                # Shutdown is best-effort; one broken transport must not prevent
                # the remaining sessions from being released.
                continue
        self._connections.clear()
        self._routes.clear()
        self._adapters.clear()
        self._servers.clear()
        self._initialized = False

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": mcp_enabled(),
            "initialized": self._initialized,
            "config_path": str(get_mcp_config_path()) if mcp_enabled() else "",
            "servers": self._servers,
            "tools": sorted(self._adapters),
        }


def make_public_tool_name(server_name: str, remote_name: str) -> str:
    raw = f"mcp__{_slug(server_name)}__{_slug(remote_name)}"
    if len(raw) <= 64:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{raw[:55]}_{digest}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug or "tool"


def _safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"(?i)(api[_-]?key|token|authorization)=\S+", r"\1=***", message)
    return message[:500]


mcp_manager = MCPManager()
