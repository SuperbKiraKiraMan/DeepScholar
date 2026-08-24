"""Agent Harness 用例的确定性隔离 Fixture。
负责注入测试需要的故障，并在测试完成后恢复原始行为。"""

import importlib
import inspect
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.llm.client import FakeLLMClient, reset_llm_client
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry
import app.llm.client as client_mod


DEFAULT_PROFILE = "default"
FaultFn = Callable[[Any, Dict[str, Any], Callable[..., Awaitable[ToolResult]]], Any]

_TOOL_CLASS_PATHS: Dict[str, Tuple[str, str]] = {
    "academic_search": ("app.tools.academic_search_tool", "AcademicSearchTool"),
    "paper_metadata": ("app.tools.paper_metadata_tool", "PaperMetadataTool"),
    "source_quality_scorer": ("app.tools.source_quality_scorer", "SourceQualityScorer"),
    "evidence_extract": ("app.tools.evidence_extract_tool", "EvidenceExtractTool"),
    "citation_check": ("app.tools.citation_check_tool", "CitationCheckTool"),
    "semantic_scholar_recommendations": (
        "app.tools.semantic_scholar_tools", "SemanticScholarRecommendationsTool",
    ),
}


async def _empty_search(_tool: Any, kwargs: Dict[str, Any], _original: Callable) -> ToolResult:
    return ToolResult(
        success=True,
        tool_name="academic_search",
        data={"results": [], "total_found": 0, "query": kwargs.get("query", "")},
    )


async def _tool_exception(tool: Any, _kwargs: Dict[str, Any], _original: Callable) -> ToolResult:
    raise RuntimeError(f"Injected tool failure for {tool.name}")


async def _partial_send_search(
    tool: Any, kwargs: Dict[str, Any], original: Callable,
) -> ToolResult:
    query = str(kwargs.get("query", ""))
    if "methods benchmarks evaluation" in query:
        raise RuntimeError("Injected failure for search_2")
    return await original(tool, **kwargs)


def _invalid_citation_once() -> FaultFn:
    calls = 0

    async def fault(tool: Any, kwargs: Dict[str, Any], original: Callable) -> ToolResult:
        nonlocal calls
        calls += 1
        if calls > 1:
            return await original(tool, **kwargs)

        citations = [dict(item) for item in kwargs.get("citations", [])]
        if citations:
            citations[0]["source_id"] = "fixture_missing_source"
            citations[0]["url"] = "https://invalid.example/fixture"
        return await original(tool, citations=citations, sources=kwargs.get("sources", []))

    return fault


def _conversation_recommendations(profile: str) -> FaultFn:
    """为多轮协议验收提供可重复、完全离线的推荐批次。"""
    calls = 0

    def paper(index: int, *, provider: str = "semantic_scholar", doi: str = "") -> Dict[str, Any]:
        return {
            "source_id": f"conversation-paper-{index}",
            "paper_id": f"conversation-paper-{index}",
            "semantic_scholar_id": f"s2-{index}" if provider == "semantic_scholar" else None,
            "doi": doi or f"10.1000/conversation.{index}",
            "title": f"Conversation Harness Paper {index}",
            "url": f"https://example.org/papers/{index}",
            "snippet": f"Deterministic conversation fixture paper {index}.",
            "full_text": f"Paper {index} studies RAG evaluation methods and limitations.",
            "authors": ["Harness Author"],
            "year": 2025,
            "venue": "HarnessConf",
            "source_type": "paper",
            "provider": provider,
        }

    async def fault(_tool: Any, _kwargs: Dict[str, Any], _original: Callable) -> ToolResult:
        nonlocal calls
        calls += 1

        # 关键步骤：不同 profile 精确模拟空增量、跨提供商重复和并发追加。
        if profile == "conversation_empty_batch":
            batches = [[paper(1), paper(2)], [paper(1), paper(2)]]
        elif profile == "conversation_cross_provider_duplicate":
            duplicate_doi = "10.1000/shared-provider-paper"
            batches = [
                [paper(1, doi=duplicate_doi), paper(2)],
                [
                    {
                        **paper(101, provider="openalex", doi=duplicate_doi),
                        "source_id": "openalex-duplicate",
                        "paper_id": "openalex-duplicate",
                        "semantic_scholar_id": None,
                        "openalex_id": "W-DUPLICATE",
                        "url": "https://openalex.org/W-DUPLICATE",
                    },
                    paper(3, provider="openalex"),
                ],
            ]
        else:
            batches = [
                [paper(1), paper(2), paper(3)],
                [paper(1), paper(4), paper(5)],
                [paper(2), paper(6), paper(7)],
            ]
        batch = batches[min(calls - 1, len(batches) - 1)]
        return ToolResult(
            success=True,
            tool_name="semantic_scholar_recommendations",
            data={"results": batch, "sources": batch, "total_found": len(batch)},
        )

    return fault


# ---- 故障注入函数映射：每个 profile 对应一套故障（替换工具 _arun 方法） ----
def _faults_for_profile(profile: str) -> Dict[str, FaultFn]:
    if profile == "no_sources":
        return {"academic_search": _empty_search}
    if profile == "tool_exception":
        return {"evidence_extract": _tool_exception}
    if profile == "invalid_citation":
        return {"citation_check": _invalid_citation_once()}
    if profile == "send_worker_partial_failure":
        return {"academic_search": _partial_send_search}
    if profile.startswith("conversation_"):
        return {
            "semantic_scholar_recommendations": _conversation_recommendations(profile),
        }
    return {}


# ---- FakeLLMClient 预设的结构化响应（模拟 LLMPlanner 输出） ----
STRUCTURED_RESPONSES: Dict[str, List[Dict[str, Any]]] = {
    "default": [{
        "research_goal": "Evaluate RAG evaluation methods systematically",
        "search_tasks": [
            {"task_id": "search_1", "query": "RAG evaluation overview survey",
             "purpose": "Broad survey", "depends_on": [],
             "allowed_tools": ["academic_search"]},
            {"task_id": "search_2", "query": "RAG evaluation methods benchmarks evaluation",
             "purpose": "Specific benchmarks", "depends_on": [],
             "allowed_tools": ["academic_search"]},
            {"task_id": "search_3", "query": "RAG evaluation limitations recent advances",
             "purpose": "Limitations", "depends_on": [],
             "allowed_tools": ["academic_search"]},
        ],
    }],
    # LLM fallback 只在 Worker 的 Function Calling 通道连续失败；Controller、
    # Source Selector、Outline 和 Chapter 阶段仍提供 schema 正确的响应。
    "llm_fallback": [
        {
            "research_goal": "Evaluate RAG evaluation methods systematically",
            "search_tasks": [
                {"task_id": "search_1", "query": "RAG evaluation overview survey",
                 "purpose": "Broad survey", "depends_on": [],
                 "allowed_tools": ["academic_search"]},
                {"task_id": "search_2", "query": "RAG evaluation methods benchmarks evaluation",
                 "purpose": "Specific benchmarks", "depends_on": [],
                 "allowed_tools": ["academic_search"]},
                {"task_id": "search_3", "query": "RAG evaluation limitations recent advances",
                 "purpose": "Limitations", "depends_on": [],
                 "allowed_tools": ["academic_search"]},
            ],
        },
        {
            "analysis_count": 3,
            "selected_source_ids": [],
            "selection_reasons": {},
            "coverage_plan": {},
            "rationale": "Use all bounded candidates for coverage.",
        },
        {
            "sections": [
                {"heading": "研究范围", "guiding_question": "研究范围是什么？", "assigned_evidence_ids": []},
                {"heading": "方法与指标", "guiding_question": "方法和指标是什么？", "assigned_evidence_ids": []},
                {"heading": "局限与开放问题", "guiding_question": "局限是什么？", "assigned_evidence_ids": []},
            ],
            "cross_cutting_themes": [],
            "evidence_gaps": [],
        },
        # 当前 Graph 的章节阶段可能按章节重试；保留足够的相同 schema 响应，
        # 让非目标阶段不会因为 fixture 不足而伪装成 Worker fallback 失败。
        *[
            {
                "heading": "研究章节",
                "synthesis": "当前来源支持对研究问题进行有限、审慎且可核验的中文概括，未引入未提供的事实。",
                "evidence_ids": [],
            }
            for _ in range(6)
        ],
    ],
}

# ---- FakeLLMClient 预设的 FC 响应（模拟 LLMWorker Function Calling 输出） ----
FC_RESPONSES: Dict[str, List[Dict[str, Any]]] = {
    "default": [],
    # 3 个 search worker × 当前 Worker 的 3 次连续失败预算；多一次调用会被诊断为 fixture 耗尽。
    "llm_fallback": [
        {"_fail": True, "_error": f"Simulated Worker LLM failure {i}"}
        for i in range(1, 10)
    ],
}

INTENT_RESPONSES: Dict[str, List[Dict[str, Any]]] = {
    "default": [],
    "llm_fallback": [{
        "intent": "deep_research",
        "execution_route": "full_research",
        "research_topic": "RAG evaluation methods",
        "selected_tool": "",
        "requested_count": 3,
        "confidence": 0.95,
        "reasoning": "Full research request requires the research DAG.",
    }],
}

EVAL_THRESHOLD_OVERRIDES: Dict[str, Optional[int]] = {
    "latency_exceeded": 1,
}


class FixtureManager:
    """Install one case profile and restore every mutated global afterwards."""

    def __init__(self):
        self._original_env: Dict[str, Optional[str]] = {}
        self._patched_methods: List[Tuple[type, Callable]] = []
        self._original_max_latency: Optional[int] = None
        self._original_llm_client: Any = None
        self._original_registry: Any = None
        self._original_search_provider: Optional[str] = None
        self._installed = False
        self._fake_client: Optional[FakeLLMClient] = None
        self.last_llm_diagnostics: Dict[str, Any] = {}

    def install(self, profile: str, agent_mode: str) -> None:
        """注入故障：保存原始状态 → 替换为假组件。失败时自动恢复。"""
        self._installed = False
        try:
            # ---- 替换 0: 强制 SEARCH_PROVIDER=mock（防止本地 .env 配置 openalex 导致联网） ----
            self._original_search_provider = os.environ.get("SEARCH_PROVIDER")
            os.environ["SEARCH_PROVIDER"] = "mock"

            # ---- 替换 1: 环境变量（AGENT_MODE） ----
            self._original_env["AGENT_MODE"] = os.environ.get("AGENT_MODE")
            os.environ["AGENT_MODE"] = agent_mode
            self._original_env["MCP_ENABLED"] = os.environ.get("MCP_ENABLED")
            os.environ["MCP_ENABLED"] = "false"

            # ---- 替换 2: 全局 LLM 客户端 → FakeLLMClient（不联网） ----
            self._original_llm_client = client_mod._global_client
            self._original_registry = ToolRegistry._instance
            reset_llm_client()
            ToolRegistry.reset_instance()

            structured = STRUCTURED_RESPONSES.get(profile, STRUCTURED_RESPONSES[DEFAULT_PROFILE])
            fc = FC_RESPONSES.get(profile, FC_RESPONSES[DEFAULT_PROFILE])
            fake = FakeLLMClient()
            fake.set_responses(structured)       # 预设 structured 响应
            fake.set_fc_responses(fc)            # 预设 FC 响应
            fake.set_intent_responses(INTENT_RESPONSES.get(profile, INTENT_RESPONSES[DEFAULT_PROFILE]))
            self._install_llm_diagnostics(fake)
            self._fake_client = fake
            client_mod._global_client = fake     # ← 全局替换

            # ---- 替换 3: 工具 _arun 方法 → 故障版本（monkey-patch） ----
            for tool_name, fault in _faults_for_profile(profile).items():
                self._patch_tool(tool_name, fault)

            # ---- 替换 4: Evaluator 延迟阈值（用于测 latency 超限） ----
            threshold = EVAL_THRESHOLD_OVERRIDES.get(profile)
            if threshold is not None:
                from app.agents.evaluator import Evaluator
                self._original_env["RESEARCH_LATENCY_TTL_SECONDS"] = os.environ.get(
                    "RESEARCH_LATENCY_TTL_SECONDS"
                )
                os.environ.pop("RESEARCH_LATENCY_TTL_SECONDS", None)
                self._original_max_latency = Evaluator.MAX_LATENCY_MS
                Evaluator.MAX_LATENCY_MS = threshold

            self._installed = True
        except Exception:
            self.restore()
            raise

    def _install_llm_diagnostics(self, fake: FakeLLMClient) -> None:
        """包装 FakeLLM 通道，只记录调用元数据和 fixture 耗尽，不记录 prompt 内容。"""
        calls: List[Dict[str, Any]] = []

        def record(kind: str, index_before: int, response: Dict[str, Any]) -> Dict[str, Any]:
            error = str(response.get("error", ""))
            exhausted = "no more predefined" in error or "no intent response" in error
            calls.append({
                "kind": kind,
                "response_index": index_before,
                "success": bool(response.get("success")),
                "exhausted": exhausted,
                "error": error[:160] if error else "",
            })
            return response

        original_structured = fake.generate_structured
        original_function_call = fake.function_call
        original_intent = fake.classify_intent

        async def tracked_structured(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            index = fake._structured_index
            return record("structured", index, await original_structured(*args, **kwargs))

        async def tracked_function_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            index = fake._fc_index
            return record("function_call", index, await original_function_call(*args, **kwargs))

        async def tracked_intent(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            index = len(fake.intent_calls)
            return record("intent", index, await original_intent(*args, **kwargs))

        # 实例级异步函数不再触发 descriptor 绑定，因此签名保留为原方法的 *args/**kwargs。
        fake.generate_structured = tracked_structured  # type: ignore[method-assign]
        fake.function_call = tracked_function_call  # type: ignore[method-assign]
        fake.classify_intent = tracked_intent  # type: ignore[method-assign]
        fake._harness_diagnostics = calls  # type: ignore[attr-defined]

    def _patch_tool(self, tool_name: str, fault: FaultFn) -> None:
        """Monkey-patch：把 Tool._arun 替换为故障注入版本。"""
        module_name, class_name = _TOOL_CLASS_PATHS[tool_name]
        tool_class = getattr(importlib.import_module(module_name), class_name)
        original = tool_class._arun   # ← 保存原始方法

        async def patched(instance: Any, **kwargs: Any) -> ToolResult:
            value = fault(instance, kwargs, original)   # 先走故障函数
            return await value if inspect.isawaitable(value) else value

        self._patched_methods.append((tool_class, original))
        tool_class._arun = patched    # ← 替换类方法（影响所有新实例）

    def restore(self) -> None:
        """恢复所有 monkey-patch：工具 → Evaluator 阈值 → LLM 客户端 → 环境变量。"""
        # ---- 恢复工具 _arun（倒序恢复，防止嵌套 patch 出问题） ----
        for tool_class, original in reversed(self._patched_methods):
            tool_class._arun = original
        self._patched_methods.clear()

        if self._original_max_latency is not None:
            from app.agents.evaluator import Evaluator
            Evaluator.MAX_LATENCY_MS = self._original_max_latency
            self._original_max_latency = None

        # ---- 恢复 SEARCH_PROVIDER ----
        if self._original_search_provider is None:
            os.environ.pop("SEARCH_PROVIDER", None)
        else:
            os.environ["SEARCH_PROVIDER"] = self._original_search_provider
        self._original_search_provider = None

        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._original_env.clear()

        if self._fake_client is not None:
            calls = list(getattr(self._fake_client, "_harness_diagnostics", []))
            exhausted = [item for item in calls if item.get("exhausted")]
            self.last_llm_diagnostics = {
                "calls": calls,
                "call_count": len(calls),
                "exhaustion_count": len(exhausted),
                "exhaustions": exhausted,
            }

        reset_llm_client()
        client_mod._global_client = self._original_llm_client
        ToolRegistry._instance = self._original_registry
        self._original_llm_client = None
        self._original_registry = None
        self._fake_client = None
        self._installed = False
