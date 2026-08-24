"""
app/tools/registry.py

Tool Registry：统一的工具注册、查找、Function Calling Schema 生成和参数校验。

不重复实现业务工具——所有工具实现复用现有 tools/ 下的 BaseTool 子类。

所有工具参数通过 validate_tool_args_against_schema() 严格校验。
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.tools.base import BaseTool, ToolResult
from app.tools.academic_search_tool import AcademicSearchTool
from app.tools.paper_metadata_tool import PaperMetadataTool
from app.tools.source_quality_scorer import SourceQualityScorer
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.tools.citation_check_tool import CitationCheckTool
from app.tools.local_paper_search_tool import LocalPaperSearchTool
from app.tools.semantic_scholar_tools import (
    SemanticScholarGraphTool,
    SemanticScholarRecommendationsTool,
    SemanticScholarSearchTool,
)

# 手动维护检索能力列表，与 LLM Function Calling 中的工具名保持一致
RETRIEVAL_CAPABILITIES = (
    "local_paper_search",
    "academic_search",
    "semantic_scholar_search",
    "semantic_scholar_graph",
    "semantic_scholar_recommendations",
)


@dataclass(frozen=True)
class ToolCapabilityMetadata:
    """所有工具共用的副作用与资源边界元数据。"""

    network_access: bool = False
    external_write: bool = False
    destructive: bool = False
    resource_scope: str = "none"


_BUILTIN_CAPABILITIES: Dict[str, ToolCapabilityMetadata] = {
    "local_paper_search": ToolCapabilityMetadata(resource_scope="none"),
    "academic_search": ToolCapabilityMetadata(network_access=True),
    "semantic_scholar_search": ToolCapabilityMetadata(network_access=True),
    "semantic_scholar_graph": ToolCapabilityMetadata(network_access=True),
    "semantic_scholar_recommendations": ToolCapabilityMetadata(network_access=True),
    # 元数据标准化、评分、证据和引用工具都只处理信封内资源，不访问网络。
    "paper_metadata": ToolCapabilityMetadata(resource_scope="explicit"),
    "source_quality_scorer": ToolCapabilityMetadata(resource_scope="explicit"),
    "evidence_extract": ToolCapabilityMetadata(resource_scope="explicit"),
    "citation_check": ToolCapabilityMetadata(resource_scope="explicit"),
}


class ToolRegistry:
    """
    统一工具注册表。

    职责：
    - 注册所有可用工具（单例）
    - 生成 OpenAI-compatible Function Calling JSON Schema
    - 按名称查找工具
    - 校验 tool_name 是否在 allowed_tools 白名单中
    """

    _instance: Optional["ToolRegistry"] = None

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._capabilities: Dict[str, ToolCapabilityMetadata] = {}
        self._register_defaults()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def _register_defaults(self):
        """注册所有 MVP 工具。"""
        self.register(AcademicSearchTool())
        self.register(LocalPaperSearchTool())
        self.register(SemanticScholarSearchTool())
        self.register(SemanticScholarRecommendationsTool())
        self.register(SemanticScholarGraphTool())
        self.register(PaperMetadataTool())
        self.register(SourceQualityScorer())
        self.register(EvidenceExtractTool())
        # CitationCheckTool 注册但不通过 LLM Function Calling 暴露——
        # 它必须由 Runtime 确定性调用。
        self.register(CitationCheckTool())

    def register(
        self,
        tool: BaseTool,
        capability: Optional[ToolCapabilityMetadata] = None,
    ):
        self._tools[tool.name] = tool
        declared = getattr(tool, "capability_metadata", None)
        if isinstance(declared, dict):
            declared = ToolCapabilityMetadata(**declared)
        self._capabilities[tool.name] = (
            capability
            or declared
            or _BUILTIN_CAPABILITIES.get(tool.name)
            or ToolCapabilityMetadata()
        )

    def unregister(self, name: str, expected_tool: Optional[BaseTool] = None) -> bool:
        """Remove a dynamic tool without deleting a replacement registered later."""
        current = self._tools.get(name)
        if current is None or (expected_tool is not None and current is not expected_tool):
            return False
        del self._tools[name]
        self._capabilities.pop(name, None)
        return True

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_capability(self, name: str) -> Optional[ToolCapabilityMetadata]:
        """返回注册时冻结的统一 capability 元数据。"""
        return self._capabilities.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_for_task(self, task_type: str) -> List[str]:
        """Return built-in and dynamically registered tools allowed for a task type."""
        builtins = {
            "search": ["academic_search"],
            "read": ["paper_metadata", "source_quality_scorer"],
            "analyze": ["evidence_extract"],
            "cite": ["citation_check"],
        }
        names = list(builtins.get(task_type, []))
        for name, tool in self._tools.items():
            task_types = getattr(tool, "task_types", ())
            if task_type in task_types and name not in names:
                names.append(name)
        return names

    # 返回的真名就是LLM看到的工具名
    def list_retrieval_capabilities(self) -> List[str]:
        """
        返回 research search 任务可用的、有边界的检索能力列表。

        动态 MCP 工具保留其原始名称。带命名空间的工具只有在最终注册段
        匹配某个 canonical capability 时才会被纳入；此方法不会创建别名
        也不会注册重复的包装器。
        """
        names = []
        for name in self._tools:
            if self.retrieval_capability(name) and name not in names:
                names.append(name)
        return names

    def retrieval_capability(self, tool_name: str) -> Optional[str]:
        """返回已注册检索工具对应的 canonical capability。

        内置工具使用 canonical 名称；带命名空间的动态工具除了名称后缀
        必须匹配，还必须显式声明 ``search`` task type，避免仅靠伪装名称
        绕过 Search Worker 的最小权限边界。
        """
        tool = self._tools.get(tool_name)
        capability = retrieval_capability_for_tool(tool_name)
        if tool is None or capability is None:
            return None
        if tool_name == capability:
            return capability
        task_types = set(getattr(tool, "task_types", ()) or ())
        return capability if "search" in task_types else None

    def get_input_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具的 input_schema（用于参数校验）。"""
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.input_schema

    # ---- OpenAI Function Calling Schema ----

    def get_function_schemas(
        self,
        allowed_tool_names: List[str] = None,
        exclude: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成 OpenAI-compatible tools 数组。

        自动排除 citation_check——它不通过 LLM Function Calling 调用。
        """
        exclude = (exclude or []) + ["citation_check"]
        allowed = allowed_tool_names or self.list_names()

        schemas = []
        for name in allowed:
            if name in exclude:
                continue
            tool = self._tools.get(name)
            if tool is None:
                continue
            schemas.append(self._tool_to_openai_schema(tool))
        return schemas

    def _tool_to_openai_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """
        将 BaseTool 转换为 OpenAI Function Calling 格式。

        对需要完整 PaperSource 的工具，FC schema 暴露 source_id（简单类型），
        不暴露完整 source 对象。程序通过 _inject_trusted_context 注入真实对象。

        Returns:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        params_schema = self._build_fc_friendly_schema(tool)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description[:1024],
                "parameters": params_schema,
            },
        }

    def _build_fc_friendly_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """
        构建 FC-friendly parameters schema。

        规则：
        - evidence_extract: 将 "source" 对象参数替换为 "source_id" 字符串
        - paper_metadata / source_quality_scorer: 将 "sources" 数组替换为 "source_ids" 字符串数组
        - 其他工具保持原样
        """
        raw = tool.input_schema
        properties = dict(raw.get("properties", {}))
        required = list(raw.get("required", []))

        if tool.name == "evidence_extract":
            # FC-facing: LLM 只传 source_id，不传完整 source 对象
            fc_props = {}
            fc_req = []
            for key, prop in properties.items():
                if key == "source":
                    # 替换为: source_id (string) — LLM 选择 source，程序注入完整对象
                    fc_props["source_id"] = {
                        "type": "string",
                        "description": "The source_id of the paper to extract evidence from. "
                                       "Choose from the source_ids listed in dependency_summary.",
                    }
                    if key in required:
                        fc_req.append("source_id")
                else:
                    fc_props[key] = prop
                    if key in required:
                        fc_req.append(key)
            properties = fc_props
            required = fc_req

        elif tool.name in ("paper_metadata", "source_quality_scorer"):
            # FC-facing: LLM 只传 source_ids，程序注入完整 sources 列表
            fc_props = {}
            fc_req = []
            for key, prop in properties.items():
                if key == "sources":
                    fc_props["source_ids"] = {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of source_ids to process. "
                                       "Choose from the source_ids listed in dependency_summary.",
                    }
                    if key in required:
                        fc_req.append("source_ids")
                elif key == "topic":
                    fc_props[key] = prop
                    if key in required:
                        fc_req.append(key)
                else:
                    fc_props[key] = prop
                    if key in required:
                        fc_req.append(key)
            properties = fc_props
            required = fc_req

        return {
            "type": raw.get("type", "object"),
            "properties": properties,
            "required": required,
        }

    def _build_parameters_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """从工具的 input_schema 构建原始 parameters schema（用于严格校验）。"""
        raw = tool.input_schema
        return {
            "type": raw.get("type", "object"),
            "properties": raw.get("properties", {}),
            "required": raw.get("required", []),
        }

    # ---- 白名单校验 ----

    def validate_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        allowed_tool_names: List[str],
    ) -> Optional[str]:
        """
        校验一次工具调用是否合法。

        Returns:
            None 表示合法，否则返回错误信息字符串。
        """
        # 1. 工具是否存在
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Unknown tool: '{tool_name}'. Available: {sorted(allowed_tool_names)}"

        # 2. 是否在白名单中
        if tool_name not in allowed_tool_names:
            return (
                f"Tool '{tool_name}' not in allowed_tools for this task. "
                f"Allowed: {sorted(allowed_tool_names)}"
            )

        # 3. citation_check 禁止 LLM 调用
        if tool_name == "citation_check":
            return "citation_check must be called deterministically by the Runtime, not by LLM"

        # 4. 基本 args 校验（至少 args 是 dict）
        if not isinstance(tool_args, dict):
            return f"tool_args must be a dict, got {type(tool_args).__name__}"

        return None


# ================================================================
# args 规范化（用于重复调用检测）
# ================================================================

def canonicalize_args(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """
    将 tool args 规范化为字符串用于去重检测。

    排序 key，序列化为 JSON。
    """
    try:
        simplified = dict(tool_args)
        if "sources" in simplified and isinstance(simplified["sources"], list):
            simplified["sources"] = [
                {"source_id": s.get("source_id", "")} if isinstance(s, dict) else s
                for s in simplified["sources"][:3]
            ]
        if "source" in simplified and isinstance(simplified["source"], dict):
            simplified["source"] = {"source_id": simplified["source"].get("source_id", "")}
        return json.dumps(simplified, sort_keys=True, default=str)
    except Exception:
        return str(hash(str(sorted(tool_args.items()))))

# 工具能力映射函数
def retrieval_capability_for_tool(tool_name: str) -> Optional[str]:
    """Map a built-in or namespaced registered name to its retrieval capability."""
    for capability in RETRIEVAL_CAPABILITIES:
        if tool_name == capability or tool_name.endswith(f"__{capability}"):
            return capability
    return None


# ================================================================
# 严格的 Tool args JSON Schema 校验
# ================================================================

def validate_tool_args_against_schema(
    tool_name: str,
    tool_args: Dict[str, Any],
    registry: ToolRegistry,
) -> Optional[str]:
    """
    根据工具的 input_schema 严格校验 tool_args。

    调用方必须在此函数之前完成 FC-friendly 参数注入
    (_inject_trusted_context)，确保 tool_args 中已是内部字段名
    (source / sources)，而非 FC-friendly 别名 (source_id / source_ids)。

    校验项：
    - required 字段是否存在且非空
    - type 是否正确
    - 数值范围（minimum, maximum）
    - 字符串最小长度（minLength）
    - enum 允许值
    - 数组 items 类型
    - 嵌套 object 的 required

    Returns:
        None 表示合法，否则返回错误信息字符串。
    """
    schema = registry.get_input_schema(tool_name)
    if schema is None:
        return f"No input_schema found for tool '{tool_name}'"

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # ---- 检查 required 字段 ----
    for req_field in required:
        if req_field not in tool_args or tool_args[req_field] is None:
            return f"Missing required field '{req_field}' for tool '{tool_name}'"
        # 字符串字段不能为空
        if req_field in properties:
            prop_schema = properties[req_field]
            if prop_schema.get("type") == "string":
                val = tool_args[req_field]
                if isinstance(val, str) and len(val.strip()) == 0:
                    return f"Required field '{req_field}' must not be empty for tool '{tool_name}'"
            if prop_schema.get("type") == "array":
                val = tool_args[req_field]
                if isinstance(val, list) and len(val) == 0:
                    return f"Required field '{req_field}' must not be an empty array for tool '{tool_name}'"

    # ---- 检查每个字段的 type ----
    type_mapping = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for field_name, field_value in tool_args.items():
        if field_name not in properties:
            continue  # skip unknown fields (allow additional properties)

        prop_schema = properties[field_name]
        expected_type = prop_schema.get("type")

        if expected_type and expected_type in type_mapping:
            expected_py_type = type_mapping[expected_type]
            if not isinstance(field_value, expected_py_type):
                return (
                    f"Field '{field_name}' for tool '{tool_name}' "
                    f"expected type '{expected_type}', got {type(field_value).__name__}"
                )

        # ---- 数值范围检查 ----
        if expected_type in ("number", "integer") and isinstance(field_value, (int, float)):
            if "minimum" in prop_schema and field_value < prop_schema["minimum"]:
                return (
                    f"Field '{field_name}' for tool '{tool_name}' "
                    f"value {field_value} below minimum {prop_schema['minimum']}"
                )
            if "maximum" in prop_schema and field_value > prop_schema["maximum"]:
                return (
                    f"Field '{field_name}' for tool '{tool_name}' "
                    f"value {field_value} above maximum {prop_schema['maximum']}"
                )

        # ---- 字符串最小长度 ----
        if expected_type == "string" and isinstance(field_value, str):
            if "minLength" in prop_schema and len(field_value) < prop_schema["minLength"]:
                return (
                    f"Field '{field_name}' for tool '{tool_name}' "
                    f"length {len(field_value)} below minLength {prop_schema['minLength']}"
                )
            if "enum" in prop_schema and field_value not in prop_schema["enum"]:
                return (
                    f"Field '{field_name}' for tool '{tool_name}' "
                    f"must be one of {prop_schema['enum']}, got '{field_value}'"
                )

        # ---- 数组 items 类型检查 ----
        if expected_type == "array" and isinstance(field_value, list):
            items_schema = prop_schema.get("items", {})
            items_type = items_schema.get("type")
            if items_type and items_type in type_mapping:
                expected_item_type = type_mapping[items_type]
                for i, item in enumerate(field_value):
                    if not isinstance(item, expected_item_type):
                        return (
                            f"Field '{field_name}[{i}]' for tool '{tool_name}' "
                            f"expected item type '{items_type}', got {type(item).__name__}"
                        )

        # ---- object 嵌套 ----
        if expected_type == "object" and isinstance(field_value, dict):
            obj_properties = prop_schema.get("properties", {})
            obj_required = prop_schema.get("required", [])
            for obj_req in obj_required:
                if obj_req not in field_value or field_value[obj_req] is None:
                    return (
                        f"Nested field '{field_name}.{obj_req}' for tool '{tool_name}' "
                        f"is required but missing"
                    )

    return None
