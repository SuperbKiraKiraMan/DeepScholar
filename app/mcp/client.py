"""Thin async client around the official MCP Python SDK."""

import json
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from app.mcp.config import MCPServerConfig


class MCPClientConnection:
    """Own one MCP transport and initialized ClientSession."""

    def __init__(self, config: MCPServerConfig, read_timeout_seconds: float = 30):
        self.config = config
        self.read_timeout_seconds = read_timeout_seconds
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        await self._stack.__aenter__()
        try:
            if self.config.transport == "stdio":
                params = StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args),
                    env=self.config.env or None,
                    cwd=self.config.cwd,
                )
                read_stream, write_stream = await self._stack.enter_async_context(
                    stdio_client(params)
                )
            else:
                read_stream, write_stream, _ = await self._stack.enter_async_context(
                    streamablehttp_client(
                        self.config.url,
                        headers=self.config.headers or None,
                        timeout=self.read_timeout_seconds,
                        sse_read_timeout=self.read_timeout_seconds,
                    )
                )

            self._session = await self._stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.read_timeout_seconds),
                )
            )
            await self._session.initialize()
        except Exception:
            await self.close()
            raise

    async def list_tools(self) -> List[Any]:
        session = self._require_session()
        tools: List[Any] = []
        cursor = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = getattr(result, "nextCursor", None)
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session()
        result = await session.call_tool(
            name,
            arguments=arguments,
            read_timeout_seconds=timedelta(seconds=self.read_timeout_seconds),
        )
        data = result.structuredContent
        if data is None:
            text_parts = [
                block.text for block in result.content
                if getattr(block, "type", "") == "text" and hasattr(block, "text")
            ]
            joined = "\n".join(text_parts)
            try:
                data = json.loads(joined)
            except (json.JSONDecodeError, TypeError):
                data = {"content": joined}
        # FastMCP wraps untyped mapping returns as {"result": {...}}.
        # Keep this SDK detail outside the Agent-facing ToolResult contract.
        if isinstance(data, dict) and set(data) == {"result"} and isinstance(data["result"], dict):
            data = data["result"]

        return {
            "success": not bool(result.isError),
            "data": data,
            "error": _result_error(result) if result.isError else "",
        }

    async def close(self) -> None:
        self._session = None
        try:
            await self._stack.aclose()
        finally:
            self._stack = AsyncExitStack()

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")
        return self._session


def _result_error(result: Any) -> str:
    parts = [
        block.text for block in result.content
        if getattr(block, "type", "") == "text" and hasattr(block, "text")
    ]
    return ("\n".join(parts) or "MCP tool returned an error")[:500]
