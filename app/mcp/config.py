"""MCP server configuration loading and validation."""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


class MCPConfigError(ValueError):
    """Raised when an MCP server configuration is invalid."""


@dataclass(frozen=True)
class MCPToolPolicy:
    enabled: bool = True
    task_types: tuple[str, ...] = ("search",)
    result_kind: str = "generic"
    network_access: bool = True
    external_write: bool = False
    destructive: bool = False
    resource_scope: str = "none"


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    enabled: bool = True
    command: str = ""
    args: tuple[str, ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    tools: Dict[str, MCPToolPolicy] = field(default_factory=dict)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def mcp_enabled() -> bool:
    return os.getenv("MCP_ENABLED", "false").lower() == "true"


def get_mcp_config_path() -> Path:
    raw = os.getenv("MCP_CONFIG_PATH", "mcp_servers.json")
    return Path(raw).expanduser().resolve()


def get_mcp_connect_timeout() -> float:
    return max(1.0, float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "10")))


def get_mcp_tool_timeout() -> float:
    return max(1.0, float(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "30")))


def mcp_fail_open() -> bool:
    return os.getenv("MCP_FAIL_OPEN", "true").lower() == "true"


def get_allowed_servers() -> set[str]:
    value = os.getenv("MCP_ALLOWED_SERVERS", "").strip()
    return {item.strip() for item in value.split(",") if item.strip()}


def load_mcp_server_configs(path: Path) -> List[MCPServerConfig]:
    if not path.is_file():
        raise MCPConfigError(f"MCP config file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"Unable to read MCP config: {exc}") from exc

    raw_servers = payload.get("mcpServers")
    if not isinstance(raw_servers, dict):
        raise MCPConfigError("MCP config must contain an 'mcpServers' object")

    configs = []
    for name, raw in raw_servers.items():
        if not isinstance(raw, dict):
            raise MCPConfigError(f"Server '{name}' config must be an object")
        configs.append(_parse_server(name, raw, path.parent))
    return configs


def _parse_server(name: str, raw: Dict[str, Any], base_dir: Path) -> MCPServerConfig:
    transport = str(raw.get("transport", "stdio")).lower().replace("-", "_")
    if transport in {"http", "streamablehttp"}:
        transport = "streamable_http"
    if transport not in {"stdio", "streamable_http"}:
        raise MCPConfigError(f"Server '{name}' has unsupported transport '{transport}'")

    env = _expand_mapping(raw.get("env", {}), f"{name}.env")
    inherit_env = raw.get("inherit_env", [])
    if not isinstance(inherit_env, list):
        raise MCPConfigError(f"{name}.inherit_env must be an array")
    for key in inherit_env:
        key = str(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise MCPConfigError(f"{name}.inherit_env contains an invalid variable name")
        if key in os.environ:
            env[key] = os.environ[key]
    headers = _expand_mapping(raw.get("headers", {}), f"{name}.headers")
    cwd = raw.get("cwd")
    if cwd:
        cwd_path = Path(_expand_text(str(cwd), f"{name}.cwd"))
        cwd = str(cwd_path if cwd_path.is_absolute() else (base_dir / cwd_path).resolve())

    command = _expand_text(str(raw.get("command", "")), f"{name}.command")
    args = tuple(
        _expand_text(str(item), f"{name}.args") for item in raw.get("args", [])
    )
    url = str(raw.get("url", ""))

    if transport == "stdio" and not command:
        raise MCPConfigError(f"Server '{name}' requires 'command' for stdio transport")
    if transport == "streamable_http":
        if not url:
            raise MCPConfigError(f"Server '{name}' requires 'url' for HTTP transport")
        _validate_remote_url(name, url)

    policies: Dict[str, MCPToolPolicy] = {}
    raw_tools = raw.get("tools", {})
    if raw_tools and not isinstance(raw_tools, dict):
        raise MCPConfigError(f"Server '{name}'.tools must be an object")
    for tool_name, policy in raw_tools.items():
        policy = policy if isinstance(policy, dict) else {}
        task_types = tuple(policy.get("task_types", ["search"]))
        policies[tool_name] = MCPToolPolicy(
            enabled=bool(policy.get("enabled", True)),
            task_types=task_types,
            result_kind=str(policy.get("result_kind", "generic")),
            network_access=bool(policy.get("network_access", True)),
            external_write=bool(policy.get("external_write", False)),
            destructive=bool(policy.get("destructive", False)),
            resource_scope=str(policy.get("resource_scope", "none")),
        )

    return MCPServerConfig(
        name=name,
        transport=transport,
        enabled=bool(raw.get("enabled", True)),
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        url=url,
        headers=headers,
        tools=policies,
    )


def _expand_mapping(raw: Any, location: str) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise MCPConfigError(f"{location} must be an object")
    return {str(key): _expand_text(str(value), location) for key, value in raw.items()}


def _expand_text(value: str, location: str) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key == "PYTHON_EXECUTABLE":
            return sys.executable
        if key not in os.environ:
            raise MCPConfigError(f"Environment variable '{key}' required by {location} is not set")
        return os.environ[key]

    return _ENV_PATTERN.sub(replace, value)


def _validate_remote_url(name: str, url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MCPConfigError(f"Server '{name}' has invalid URL")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        if os.getenv("MCP_ALLOW_INSECURE_HTTP", "false").lower() != "true":
            raise MCPConfigError(
                f"Server '{name}' uses insecure remote HTTP; use HTTPS or explicitly opt in"
            )
