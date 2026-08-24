"""Deterministic Phase B Agentic Retrieval acceptance tests."""

import asyncio
import json
from typing import Any, Dict, List

import pytest

from app.agents.llm_worker import (
    LLMWorker,
    LLMWorkerConfig,
    _bounded_json_observation,
    _retrieval_result_to_observation,
)
from app.agents.controller import IntentController
from app.agents.planner import Planner, PlannerAgent, Task
from app.agents.schemas import ExecutionClass, ExecutionSpec
from app.agents.worker import WorkerContext
from app.llm.client import FakeLLMClient, reset_llm_client
from app.tools.base import BaseTool, ToolResult
from app.tools.citation_check_tool import CitationCheckTool
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.tools.registry import ToolRegistry
from harness.runner import _extract_real_tools


RETRIEVAL_NAMES = {
    "local_paper_search",
    "academic_search",
    "semantic_scholar_search",
    "semantic_scholar_graph",
    "semantic_scholar_recommendations",
}


def _source(
    source_id: str,
    *,
    provider: str = "mock",
    year: int = 2025,
    text: str = "A traceable abstract reports a concrete evaluation result.",
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "paper_id": source_id,
        "title": f"Paper {source_id}",
        "url": f"https://example.org/{source_id}",
        "year": year,
        "snippet": text,
        "full_text": text,
        "provider": provider,
        "retrieval_score": 0.91,
    }


class ScriptedRetrievalTool(BaseTool):
    task_types = ("search",)

    def __init__(
        self,
        name: str,
        results: List[ToolResult],
        *,
        delay: float = 0.0,
    ):
        super().__init__()
        self._name = name
        self.results = list(results)
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Scripted {self._name} retrieval capability."

    @property
    def input_schema(self) -> Dict[str, Any]:
        if self._name == "semantic_scholar_recommendations":
            properties = {
                "topic": {"type": "string", "minLength": 2},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "positive_paper_ids": {"type": "array", "items": {"type": "string"}},
            }
            required = ["topic"]
        elif self._name == "semantic_scholar_graph":
            properties = {
                "paper_query": {"type": "string", "minLength": 2},
                "relation": {
                    "type": "string",
                    "enum": ["details", "citations", "references"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            }
            required = ["paper_query"]
        else:
            properties = {
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "year_from": {"type": "integer", "minimum": 1800, "maximum": 2100},
                "year_to": {"type": "integer", "minimum": 1800, "maximum": 2100},
            }
            required = ["query"]
        return {"type": "object", "properties": properties, "required": required}

    async def _arun(self, **kwargs) -> ToolResult:
        self.calls.append(dict(kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.results:
            return ToolResult(success=False, error="script exhausted")
        return self.results.pop(0)


@pytest.fixture(autouse=True)
def _isolated_registry_and_llm():
    ToolRegistry.reset_instance()
    reset_llm_client()
    yield
    ToolRegistry.reset_instance()
    reset_llm_client()


def _install_fake(responses: List[Dict[str, Any]]) -> FakeLLMClient:
    import app.llm.client as client_module

    fake = FakeLLMClient()
    fake.set_fc_responses(responses)
    client_module._global_client = fake
    return fake


def _register(tool: BaseTool) -> BaseTool:
    ToolRegistry.get_instance().register(tool)
    return tool


def _result(
    sources: List[Dict[str, Any]],
    *,
    provider: str,
    query: str = "query",
    fallback_used: bool = False,
) -> ToolResult:
    return ToolResult(
        success=True,
        data={
            "results": sources,
            "sources": sources,
            "query": query,
            "total_found": len(sources),
            "provider": provider,
            "fallback_used": fallback_used,
            "fallback_reason": "provider_failed" if fallback_used else None,
        },
        metadata={
            "provider": provider,
            "fallback_used": fallback_used,
            "fallback_reason": "provider_failed" if fallback_used else None,
        },
    )


def _trace(ctx: WorkerContext, event: str) -> List[Dict[str, Any]]:
    return [entry for entry in ctx.trace if entry.get("tool_name") == event]


def test_planner_exposes_bounded_registered_retrieval_allowlist():
    plans = [
        Planner().plan("agent evaluation").get_task("search").tool_plan,
        Planner().plan_for_send("agent evaluation").get_task("search_1").tool_plan,
    ]
    for plan in plans:
        assert set(plan) == RETRIEVAL_NAMES
        assert all("openalex" not in name for name in plan)
    academic_schema = ToolRegistry.get_instance().get_input_schema("academic_search")
    assert "provider" not in academic_schema["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("local_paper_search", {"query": "owned papers", "top_k": 3}),
        ("academic_search", {"query": "recent papers", "max_results": 3}),
        ("semantic_scholar_search", {"query": "graph RAG", "max_results": 3}),
        (
            "semantic_scholar_graph",
            {"paper_query": "s2:seed", "relation": "citations", "limit": 3},
        ),
        (
            "semantic_scholar_recommendations",
            {"topic": "agent evaluation", "limit": 3},
        ),
    ],
)
async def test_worker_can_choose_each_retrieval_capability(tool_name, arguments):
    provider = "local_zotero" if tool_name == "local_paper_search" else (
        "openalex" if tool_name == "academic_search" else "semantic_scholar"
    )
    scripted = _register(
        ScriptedRetrievalTool(
            tool_name,
            [_result([_source(f"{tool_name}-1", provider=provider)], provider=provider)],
        )
    )
    _install_fake([
        {
            "_finish": False,
            "tool_calls": [{"id": "call_1", "name": tool_name, "arguments": arguments}],
        },
        {"_finish": True, "content": "sufficient for this task"},
    ])

    ctx = await LLMWorker().execute_task(
        Task("search_1", "search", "Find sources", tool_plan=list(RETRIEVAL_NAMES))
    )

    assert scripted.calls == [arguments]
    assert ctx.results["sources"]
    observed = _trace(ctx, "retrieval_observed")
    assert observed[0]["provider"] == provider
    assert _trace(ctx, "retrieval_finished")[0]["forced_stop"] is False


@pytest.mark.asyncio
async def test_non_retrieval_task_can_finish_without_retrieval():
    _install_fake([{"_finish": True, "content": "nothing else required"}])
    ctx = await LLMWorker().execute_task(
        Task("custom", "custom", "Format dependency output", tool_plan=["paper_metadata"])
    )
    assert ctx.results["llm_finish_summary"] == "nothing else required"
    assert not _trace(ctx, "retrieval_finished")


@pytest.mark.asyncio
async def test_dependencies_sufficient_allows_no_new_retrieval():
    source = _source("dependency-paper")
    dep = WorkerContext(Task("prior", "search", "Prior discovery"))
    dep.add_result("sources", [source])
    fake = _install_fake([{"_finish": True, "content": "dependencies are sufficient"}])

    ctx = await LLMWorker().execute_task(
        Task("search_2", "search", "Expand only if needed", tool_plan=list(RETRIEVAL_NAMES)),
        dependency_results={"prior": dep},
    )

    assert len(fake.fc_calls) == 1
    assert ctx.results["sources"] == [source]
    finished = _trace(ctx, "retrieval_finished")
    assert finished[0]["finish_reason"] == "dependencies_sufficient"
    assert finished[0]["tool_call_count"] == 0


@pytest.mark.asyncio
async def test_empty_search_cannot_finish_successfully_without_retrieval():
    _install_fake([{"_finish": True, "content": "I think no retrieval is needed"}])
    ctx = await LLMWorker(
        LLMWorkerConfig(max_tool_calls=3, max_iterations=1)
    ).execute_task(
        Task("search", "search", "Find papers", tool_plan=["academic_search"])
    )

    assert ctx.results["sources"] == []
    assert any("has not called any retrieval tool" in warning for warning in ctx.warnings)
    finished = _trace(ctx, "retrieval_finished")[0]
    assert finished["finish_reason"] == "budget_exhausted"
    assert finished["forced_stop"] is True
    assert finished["success"] is False


@pytest.mark.asyncio
async def test_empty_local_result_can_switch_to_external_and_rewrite_query():
    local = _register(
        ScriptedRetrievalTool(
            "local_paper_search",
            [_result([], provider="local_zotero", query="Mamba entity alignment")],
        )
    )
    academic = _register(
        ScriptedRetrievalTool(
            "academic_search",
            [
                _result(
                    [_source("W-new", provider="openalex", year=2026)],
                    provider="openalex",
                    query="Mamba multimodal entity alignment 2025 2026 benchmark",
                )
            ],
        )
    )
    _install_fake([
        {
            "_finish": False,
            "tool_calls": [{
                "id": "local",
                "name": "local_paper_search",
                "arguments": {"query": "Mamba entity alignment", "top_k": 3},
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "external",
                "name": "academic_search",
                "arguments": {
                    "query": "Mamba multimodal entity alignment 2025 2026 benchmark",
                    "max_results": 5,
                    "year_from": 2025,
                },
            }],
        },
        {"_finish": True, "content": "recent external coverage is now sufficient"},
    ])

    ctx = await LLMWorker().execute_task(
        Task(
            "search",
            "search",
            "Use my papers and add the latest two years",
            tool_plan=["local_paper_search", "academic_search"],
        )
    )

    assert len(local.calls) == len(academic.calls) == 1
    rewritten = _trace(ctx, "retrieval_query_rewritten")
    switched = _trace(ctx, "retrieval_source_switched")
    assert rewritten[0]["previous_query"] == "Mamba entity alignment"
    assert rewritten[0]["new_query"].endswith("2025 2026 benchmark")
    assert switched[0]["from_capability"] == "local_paper_search"
    assert switched[0]["to_capability"] == "academic_search"
    assert _trace(ctx, "retrieval_finished")[0]["finish_reason"] == "evidence_sufficient"


@pytest.mark.asyncio
async def test_llm_can_add_recent_external_search_after_old_local_observation():
    _register(
        ScriptedRetrievalTool(
            "local_paper_search",
            [
                _result(
                    [_source("local-old", provider="local_zotero", year=2021)],
                    provider="local_zotero",
                    query="multimodal alignment",
                )
            ],
        )
    )
    external = _register(
        ScriptedRetrievalTool(
            "academic_search",
            [
                _result(
                    [_source("W-recent", provider="openalex", year=2026)],
                    provider="openalex",
                    query="multimodal alignment recent work",
                )
            ],
        )
    )
    _install_fake([
        {
            "_finish": False,
            "tool_calls": [{
                "id": "old-local",
                "name": "local_paper_search",
                "arguments": {"query": "multimodal alignment", "top_k": 3},
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "recent-external",
                "name": "academic_search",
                "arguments": {
                    "query": "multimodal alignment recent work",
                    "max_results": 3,
                    "year_from": 2025,
                    "year_to": 2026,
                },
            }],
        },
        {"_finish": True, "content": "recent coverage added"},
    ])

    ctx = await LLMWorker().execute_task(
        Task(
            "search",
            "search",
            "Use local evidence and cover the latest two years",
            tool_plan=["local_paper_search", "academic_search"],
        )
    )

    assert external.calls[0]["year_from"] == 2025
    observed = _trace(ctx, "retrieval_observed")
    assert observed[0]["year_max"] == 2021
    assert observed[1]["year_max"] == 2026


@pytest.mark.asyncio
async def test_rewritten_args_are_not_duplicate_but_exact_duplicate_is_blocked():
    tool = _register(
        ScriptedRetrievalTool(
            "academic_search",
            [
                _result([_source("W1")], provider="openalex", query="agent evaluation"),
                _result([_source("W2")], provider="openalex", query="agent benchmark ablation"),
            ],
        )
    )
    _install_fake([
        {
            "_finish": False,
            "tool_calls": [{
                "id": "one",
                "name": "academic_search",
                "arguments": {"query": "agent evaluation", "max_results": 3},
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "duplicate",
                "name": "academic_search",
                "arguments": {"max_results": 3, "query": "agent evaluation"},
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "rewrite",
                "name": "academic_search",
                "arguments": {"query": "agent benchmark ablation", "max_results": 3},
            }],
        },
        {"_finish": True, "content": "done"},
    ])

    ctx = await LLMWorker().execute_task(
        Task("search", "search", "Find evidence", tool_plan=["academic_search"])
    )

    assert len(tool.calls) == 2
    assert _trace(ctx, "tool_rejected")
    assert len(_trace(ctx, "retrieval_query_rewritten")) == 1


@pytest.mark.asyncio
async def test_local_graph_and_recommendations_can_be_combined_without_calling_all_tools():
    local = _register(
        ScriptedRetrievalTool(
            "local_paper_search",
            [_result([_source("local:1", provider="local_zotero")], provider="local_zotero")],
        )
    )
    graph = _register(
        ScriptedRetrievalTool(
            "semantic_scholar_graph",
            [_result([_source("s2:cite", provider="semantic_scholar", text="")], provider="semantic_scholar")],
        )
    )
    recs = _register(
        ScriptedRetrievalTool(
            "semantic_scholar_recommendations",
            [_result([_source("s2:rec", provider="semantic_scholar", text="")], provider="semantic_scholar")],
        )
    )
    academic = _register(
        ScriptedRetrievalTool(
            "academic_search",
            [_result([_source("unused")], provider="openalex")],
        )
    )
    s2_search = _register(
        ScriptedRetrievalTool(
            "semantic_scholar_search",
            [_result([_source("unused-s2")], provider="semantic_scholar")],
        )
    )
    _install_fake([
        {
            "_finish": False,
            "tool_calls": [{
                "id": "local",
                "name": "local_paper_search",
                "arguments": {"query": "ACAA mechanism", "top_k": 2},
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "graph",
                "name": "semantic_scholar_graph",
                "arguments": {
                    "paper_query": "s2:seed",
                    "relation": "citations",
                    "limit": 2,
                },
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "recs",
                "name": "semantic_scholar_recommendations",
                "arguments": {
                    "topic": "ACAA mechanism",
                    "positive_paper_ids": ["seed"],
                    "limit": 2,
                },
            }],
        },
        {"_finish": True, "content": "enough evidence and discovery sources"},
    ])

    ctx = await LLMWorker().execute_task(
        Task("search", "search", "Expand a key paper", tool_plan=list(RETRIEVAL_NAMES))
    )

    assert len(local.calls) == len(graph.calls) == len(recs.calls) == 1
    assert academic.calls == []
    assert s2_search.calls == []
    assert len(_trace(ctx, "retrieval_source_switched")) == 2


@pytest.mark.asyncio
async def test_budget_timeout_and_provider_fallback_are_visible():
    fallback_tool = _register(
        ScriptedRetrievalTool(
            "academic_search",
            [_result([_source("mock-1")], provider="mock", fallback_used=True)],
        )
    )
    _install_fake([
        {
            "_finish": False,
            "tool_calls": [{
                "id": "fallback",
                "name": "academic_search",
                "arguments": {"query": "fallback query", "max_results": 2},
            }],
        },
        {
            "_finish": False,
            "tool_calls": [{
                "id": "over-budget",
                "name": "academic_search",
                "arguments": {"query": "must not execute", "max_results": 2},
            }],
        },
    ])
    ctx = await LLMWorker(
        LLMWorkerConfig(max_tool_calls=1)
    ).execute_task(
        Task("search", "search", "Search with fallback", tool_plan=["academic_search"])
    )

    assert len(fallback_tool.calls) == 1
    observed = _trace(ctx, "retrieval_observed")[0]
    assert observed["fallback_used"] is True
    assert "fallback" in observed["warning"]
    finished = _trace(ctx, "retrieval_finished")[0]
    assert finished["finish_reason"] == "budget_exhausted"
    assert finished["forced_stop"] is True

    slow = _register(
        ScriptedRetrievalTool(
            "academic_search",
            [_result([_source("too-late")], provider="openalex")],
            delay=0.05,
        )
    )
    _install_fake([{
        "_finish": False,
        "tool_calls": [{
            "id": "slow",
            "name": "academic_search",
            "arguments": {"query": "slow query", "max_results": 2},
        }],
    }])
    timed = await LLMWorker(
        LLMWorkerConfig(max_tool_calls=1, tool_timeout_ms=1)
    ).execute_task(
        Task("search", "search", "Timeout test", tool_plan=["academic_search"])
    )
    assert len(slow.calls) == 1
    assert "timed out" in _trace(timed, "retrieval_observed")[0]["error"]

    _install_fake([{"_finish": True, "content": "unreachable"}])
    worker_timed = await LLMWorker(
        LLMWorkerConfig(worker_timeout_ms=-1)
    ).execute_task(
        Task("search", "search", "Worker timeout", tool_plan=["academic_search"])
    )
    assert _trace(worker_timed, "retrieval_finished")[0]["finish_reason"] == "worker_timeout"


def test_retrieval_observation_is_complete_bounded_json_without_full_text():
    body = "UNIQUE_FULL_BODY_MARKER " + ("正文" * 10_000)
    sources = [
        {
            **_source(f"paper-{index}", text=body),
            "title": "标题" * 500,
        }
        for index in range(20)
    ]
    observation = _retrieval_result_to_observation(
        tool_name="local_paper_search",
        tool_args={"query": "查询" * 1000, "top_k": 20},
        result=ToolResult(
            success=True,
            data={"results": sources, "total_found": 20, "provider": "local_zotero"},
        ),
    )
    encoded = _bounded_json_observation(observation)
    decoded = json.loads(encoded)

    assert set(decoded) >= {
        "tool_name", "provider", "query", "result_count", "year_min",
        "year_max", "top_papers", "fallback_used", "error", "warning",
        "available_text_count",
    }
    assert len(decoded["top_papers"]) <= 5
    assert len(encoded.encode("utf-8")) <= 6 * 1024
    assert "UNIQUE_FULL_BODY_MARKER" not in encoded


def test_retrieval_trace_events_are_not_reported_as_business_tools():
    tools = _extract_real_tools([
        {"event": "retrieval_finished", "tool_name": "retrieval_finished", "success": False},
        {"event": "tool_finished", "tool_name": "academic_search", "success": True},
    ])
    assert tools == ["academic_search"]


@pytest.mark.asyncio
async def test_metadata_only_results_cannot_create_evidence_but_traceable_snippet_can():
    metadata_only = {
        "source_id": "s2:meta",
        "title": "Metadata Only Paper",
        "url": "https://example.org/meta",
        "year": 2026,
        "citation_count": 42,
        "snippet": "Metadata Only Paper",
        "full_text": "",
    }
    denied = await EvidenceExtractTool().run(source=metadata_only)
    assert denied.success is False
    assert "metadata-only" in denied.error

    traceable = {
        **metadata_only,
        "source_id": "s2:text",
        "title": "Paper With Traceable Snippet",
        "url": "https://example.org/text",
        "snippet": (
            "The reported benchmark compares three retrieval methods and finds "
            "that evidence-aware reranking improves citation accuracy."
        ),
    }
    extracted = await EvidenceExtractTool().run(source=traceable)
    assert extracted.success is True
    card = extracted.data["evidence_cards"][0]
    checked = await CitationCheckTool().run(
        citations=[{
            "id": 1,
            "source_id": card["source_id"],
            "url": card["url"],
            "quote": card["quote"],
        }],
        sources=[traceable],
    )
    assert checked.data["all_valid"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_request", "relation"),
    [
        ("论文 X 的后续引用工作有哪些？", "citations"),
        ("查询论文 X 的参考文献", "references"),
        ("查询论文 X 的论文元数据", "details"),
    ],
)
async def test_graph_lookup_regression_stays_on_direct_tool(user_request, relation):
    decision = await IntentController().decide(
        user_request,
        max_sources=5,
        agent_mode="rule",
    )
    assert decision.intent == "paper_graph_lookup"
    assert decision.execution_route == "direct_tool"
    # 关键步骤：图谱的 paper_query/relation 解析已下沉到 Planner 规则计划。
    spec = ExecutionSpec(
        request_id="test",
        user_request=user_request,
        intent=decision.intent,
        execution_class=ExecutionClass.ATOMIC,
        execution_route=decision.execution_route,
        research_topic=decision.research_topic,
        metadata={"agent_mode": "rule"},
    )
    item = PlannerAgent().plan(spec).items[0]
    assert item.input_data["relation"] == relation
    assert item.input_data["paper_query"] == "论文 X"
    assert item.allowed_tools == ["semantic_scholar_graph"]
    assert "local_paper_search" not in item.allowed_tools
