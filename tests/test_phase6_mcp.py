"""V1.1 MCP integration and Docker packaging tests."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mcp import academic_server
from app.mcp.client import MCPClientConnection
from app.mcp.config import MCPConfigError, MCPServerConfig, load_mcp_server_configs
from app.mcp.manager import MCPManager, make_public_tool_name
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
from app.tools.semantic_scholar_tools import SemanticScholarSearchTool


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeConnection:
    fail_names = set()

    def __init__(self, config, read_timeout_seconds=30):
        self.config = config
        self.closed = False

    async def connect(self):
        if self.config.name in self.fail_names:
            raise RuntimeError("secret token=must-not-leak")

    async def list_tools(self):
        return [
            SimpleNamespace(
                name="recommend_papers",
                description="Recommend related papers",
                inputSchema={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            )
        ]

    async def call_tool(self, name, arguments):
        return {
            "success": True,
            "data": {
                "sources": [{
                    "source_id": "mcp-1",
                    "title": arguments["topic"],
                    "url": "https://example.org/paper",
                    "full_text": "Evidence text.",
                }]
            },
            "error": "",
        }

    async def close(self):
        self.closed = True


def _write_config(path: Path, servers: dict) -> Path:
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def test_config_loads_stdio_tool_policy(tmp_path):
    path = _write_config(tmp_path / "mcp.json", {
        "academic": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "app.mcp.academic_server"],
            "tools": {
                "recommend_papers": {
                    "task_types": ["search"],
                    "result_kind": "sources",
                }
            },
        }
    })
    config = load_mcp_server_configs(path)[0]
    assert config.transport == "stdio"
    assert config.tools["recommend_papers"].result_kind == "sources"


def test_config_resolves_current_python_interpreter(tmp_path):
    path = _write_config(tmp_path / "mcp.json", {
        "academic": {
            "transport": "stdio",
            "command": "${PYTHON_EXECUTABLE}",
        }
    })
    assert load_mcp_server_configs(path)[0].command == sys.executable


def test_config_inherits_only_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-stay-private")
    path = _write_config(tmp_path / "mcp.json", {
        "academic": {
            "transport": "stdio",
            "command": "${PYTHON_EXECUTABLE}",
            "inherit_env": ["OPENALEX_API_KEY"],
        }
    })
    env = load_mcp_server_configs(path)[0].env
    assert env["OPENALEX_API_KEY"] == "openalex-test"
    assert "DEEPSEEK_API_KEY" not in env


def test_config_rejects_insecure_remote_http(tmp_path, monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_INSECURE_HTTP", raising=False)
    path = _write_config(tmp_path / "mcp.json", {
        "remote": {"transport": "streamable_http", "url": "http://example.org/mcp"}
    })
    with pytest.raises(MCPConfigError, match="insecure"):
        load_mcp_server_configs(path)


def test_public_names_are_namespaced_and_bounded():
    assert make_public_tool_name("academic", "recommend_papers") == (
        "mcp__academic__recommend_papers"
    )
    assert len(make_public_tool_name("a" * 80, "b" * 80)) <= 64


@pytest.mark.asyncio
async def test_manager_registers_routes_calls_and_unloads(tmp_path, monkeypatch):
    path = _write_config(tmp_path / "mcp.json", {
        "academic": {
            "transport": "stdio",
            "command": "python",
            "tools": {
                "recommend_papers": {
                    "task_types": ["search"],
                    "result_kind": "sources",
                }
            },
        }
    })
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_CONFIG_PATH", str(path))
    registry = ToolRegistry()
    manager = MCPManager(connection_factory=FakeConnection)

    await manager.initialize(registry)
    public_name = "mcp__academic__recommend_papers"
    assert public_name in registry.list_for_task("search")
    assert registry.get(public_name).input_schema["required"] == ["topic"]

    result = await registry.get(public_name).run(topic="Agent evaluation")
    assert result.success is True
    assert result.data["sources"][0]["source_id"] == "mcp-1"
    assert result.metadata["protocol"] == "mcp"

    await manager.shutdown()
    assert registry.get(public_name) is None


@pytest.mark.asyncio
async def test_manager_fail_open_isolates_one_server(tmp_path, monkeypatch):
    path = _write_config(tmp_path / "mcp.json", {
        "broken": {"transport": "stdio", "command": "python"},
        "healthy": {"transport": "stdio", "command": "python"},
    })
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_CONFIG_PATH", str(path))
    monkeypatch.setenv("MCP_FAIL_OPEN", "true")
    FakeConnection.fail_names = {"broken"}
    manager = MCPManager(connection_factory=FakeConnection)
    try:
        await manager.initialize(ToolRegistry())
        status = manager.status()
        assert status["servers"]["broken"]["status"] == "failed"
        assert "must-not-leak" not in status["servers"]["broken"]["error"]
        assert status["servers"]["healthy"]["status"] == "connected"
    finally:
        FakeConnection.fail_names = set()
        await manager.shutdown()


@pytest.mark.asyncio
async def test_real_stdio_protocol_lists_and_calls_recommendation(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    config = MCPServerConfig(
        name="academic-real",
        transport="stdio",
        command=sys.executable,
        args=("-m", "app.mcp.academic_server"),
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "SEARCH_PROVIDER": "mock"},
    )
    connection = MCPClientConnection(config, read_timeout_seconds=15)
    try:
        await connection.connect()
        tools = await connection.list_tools()
        assert {
            "recommend_papers",
            "semantic_scholar_search",
            "semantic_scholar_recommendations",
            "semantic_scholar_graph",
            "crossref_search",
            "local_rag_search",
        } <= {
            tool.name for tool in tools
        }
        result = await connection.call_tool(
            "recommend_papers", {"topic": "RAG evaluation", "limit": 2}
        )
        assert result["success"] is True
        assert len(result["data"]["sources"]) == 2
        assert result["data"]["sources"][0]["provider"] == "mock"
    finally:
        await connection.close()


def test_mcp_status_endpoint_is_additive_and_secret_free(monkeypatch):
    monkeypatch.setenv("MCP_ENABLED", "false")
    response = TestClient(app).get("/api/mcp/tools")
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert "headers" not in response.text


def test_docker_packaging_has_non_root_healthcheck_and_sqlite_volume():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "uvicorn" in dockerfile
    assert "./data:/app/data" in compose
    assert "SQLITE_DB_PATH: /app/data/research_history.db" in compose


def test_example_config_keeps_local_rag_opt_in():
    payload = json.loads(
        (PROJECT_ROOT / "mcp_servers.example.json").read_text(encoding="utf-8")
    )
    tools = payload["mcpServers"]["academic-research"]["tools"]
    assert tools["recommend_papers"]["enabled"] is True
    assert tools["semantic_scholar_search"]["enabled"] is True
    assert tools["semantic_scholar_recommendations"]["enabled"] is True
    assert tools["semantic_scholar_graph"]["enabled"] is True
    assert tools["crossref_search"]["enabled"] is True
    assert tools["local_rag_search"]["enabled"] is False


@pytest.mark.asyncio
async def test_semantic_scholar_mcp_wrapper_reuses_builtin_tool(monkeypatch):
    captured = {}

    async def fake_run(self, **kwargs):
        captured.update(kwargs)
        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "results": [{
                    "source_id": "s2:wrapped",
                    "title": "Wrapped Semantic Scholar Result",
                    "url": "https://www.semanticscholar.org/paper/wrapped",
                    "provider": "semantic_scholar",
                }],
                "provider": "semantic_scholar",
            },
            metadata={"attempts": 1},
        )

    monkeypatch.setattr(SemanticScholarSearchTool, "run", fake_run)

    payload = await academic_server.semantic_scholar_search(
        query="agent evaluation",
        max_results=3,
    )

    assert captured == {"query": "agent evaluation", "max_results": 3}
    assert payload["sources"] == payload["results"]
    assert payload["sources"][0]["source_id"] == "s2:wrapped"
    assert payload["result_kind"] == "sources"
    assert payload["tool_metadata"]["attempts"] == 1


@pytest.mark.asyncio
async def test_semantic_scholar_mcp_wrapper_turns_tool_failure_into_protocol_error(
    monkeypatch,
):
    async def fake_run(self, **kwargs):
        return ToolResult(
            success=False,
            tool_name=self.name,
            error="rate limit budget exhausted",
        )

    monkeypatch.setattr(SemanticScholarSearchTool, "run", fake_run)

    with pytest.raises(RuntimeError, match="rate limit budget exhausted"):
        await academic_server.semantic_scholar_search(
            query="agent evaluation",
            max_results=3,
        )
