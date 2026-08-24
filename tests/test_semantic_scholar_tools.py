"""V1.3 Semantic Scholar provider, tools, and intent routing tests."""

import os
import httpx
import pytest

from app.agents.controller import IntentController
from app.agents.planner import PlannerAgent
from app.agents.schemas import ExecutionClass, ExecutionSpec
from app.graph.runtime import run_graph
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry, validate_tool_args_against_schema
from app.tools.semantic_scholar_provider import (
    SemanticScholarClient,
    semantic_scholar_paper_to_source,
)


PAPER = {
    "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
    "corpusId": 215416146,
    "title": "Construction of the Literature Graph in Semantic Scholar",
    "abstract": "We describe the construction of a scientific literature graph.",
    "url": "https://www.semanticscholar.org/paper/649def",
    "year": 2018,
    "venue": "NAACL",
    "authors": [{"authorId": "1", "name": "Waleed Ammar"}],
    "externalIds": {"DOI": "10.18653/v1/N18-3011"},
    "citationCount": 120,
    "referenceCount": 42,
    "openAccessPdf": {"url": "https://example.org/paper.pdf", "status": "GREEN"},
    "publicationTypes": ["JournalArticle"],
    "publicationDate": "2018-06-01",
    "tldr": {"text": "A literature graph construction system."},
}


def test_paper_adapter_preserves_graph_metadata():
    source = semantic_scholar_paper_to_source(PAPER)
    assert source["source_id"].startswith("s2:")
    assert source["semantic_scholar_id"] == PAPER["paperId"]
    assert source["corpus_id"] == 215416146
    assert source["authors"] == ["Waleed Ammar"]
    assert source["cited_by_count"] == 120
    assert source["reference_count"] == 42
    assert source["full_text"] == PAPER["abstract"]
    assert source["provider"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_search_normalizes_hyphen_and_maps_results(monkeypatch):
    captured = {}

    async def fake_request(self, method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return {
            "success": True,
            "data": {"total": 1, "data": [PAPER]},
            "metadata": {"attempts": 1},
        }

    monkeypatch.setattr(SemanticScholarClient, "_request_json", fake_request)
    result = await SemanticScholarClient().search("retrieval-augmented generation", 3)

    assert result.success
    assert captured["params"]["query"] == "retrieval augmented generation"
    assert captured["params"]["limit"] == 3
    assert result.data["results"][0]["semantic_scholar_id"] == PAPER["paperId"]


@pytest.mark.asyncio
async def test_recommendation_resolves_seed_then_calls_recommendations(monkeypatch):
    calls = []
    recommended = {**PAPER, "paperId": "b" * 40, "title": "Recommended Paper"}

    async def fake_request(self, method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/paper/search"):
            return {
                "success": True,
                "data": {"total": 1, "data": [PAPER]},
                "metadata": {},
            }
        return {
            "success": True,
            "data": {"recommendedPapers": [recommended]},
            "metadata": {},
        }

    monkeypatch.setattr(SemanticScholarClient, "_request_json", fake_request)
    result = await SemanticScholarClient().recommend("literature graph", limit=4)

    assert result.success
    assert len(calls) == 2
    assert calls[1][0] == "POST"
    assert calls[1][2]["json_body"]["positivePaperIds"] == [PAPER["paperId"]]
    assert "tldr" not in calls[1][2]["params"]["fields"].split(",")
    assert "paperId" in calls[1][2]["params"]["fields"].split(",")
    assert result.data["results"][0]["title"] == "Recommended Paper"


@pytest.mark.asyncio
async def test_recommendation_page_uses_incremental_window_without_app_total(monkeypatch):
    papers = [
        {**PAPER, "paperId": str(index) * 40, "title": f"Recommended {index}"}
        for index in range(1, 6)
    ]
    captured = {}

    async def fake_recommend(self, topic, limit=5, **kwargs):
        captured["limit"] = limit
        return ToolResult(
            success=True,
            tool_name="semantic_scholar_recommendations",
            data={"results": [semantic_scholar_paper_to_source(paper) for paper in papers[:limit]]},
        )

    monkeypatch.setattr(SemanticScholarClient, "recommend", fake_recommend)
    result = await SemanticScholarClient().recommend_page("literature graph", offset=2, limit=2)

    assert result.success
    assert captured["limit"] == 5
    assert [paper["title"] for paper in result.data["results"]] == [
        "Recommended 3", "Recommended 4",
    ]
    assert result.data["has_more"] is True
    assert result.data["next_offset"] == 4


@pytest.mark.asyncio
async def test_graph_maps_citing_papers(monkeypatch):
    citing = {**PAPER, "paperId": "c" * 40, "title": "A Citing Paper"}
    captured = {}

    async def fake_resolve(self, paper_query):
        return PAPER["paperId"]

    async def fake_request(self, method, url, **kwargs):
        captured.update({"method": method, "url": url, **kwargs})
        return {
            "success": True,
            "data": {"data": [{"citingPaper": citing}]},
            "metadata": {},
        }

    monkeypatch.setattr(SemanticScholarClient, "_resolve_paper_id", fake_resolve)
    monkeypatch.setattr(SemanticScholarClient, "_request_json", fake_request)
    result = await SemanticScholarClient().paper_graph(
        "Construction of the Literature Graph", "citations", 5
    )

    assert result.success
    assert result.data["relation"] == "citations"
    assert result.data["results"][0]["title"] == "A Citing Paper"
    assert "tldr" not in captured["params"]["fields"].split(",")
    assert "paperId" in captured["params"]["fields"].split(",")


@pytest.mark.asyncio
async def test_graph_pagination_forwards_offset_and_provider_next(monkeypatch):
    citing = {**PAPER, "paperId": "d" * 40, "title": "Next Citing Paper"}
    captured = {}

    async def fake_resolve(self, paper_query):
        return PAPER["paperId"]

    async def fake_request(self, method, url, **kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "data": {"next": 40, "data": [{"citingPaper": citing}]},
            "metadata": {},
        }

    monkeypatch.setattr(SemanticScholarClient, "_resolve_paper_id", fake_resolve)
    monkeypatch.setattr(SemanticScholarClient, "_request_json", fake_request)
    result = await SemanticScholarClient().paper_graph(
        PAPER["title"], relation="citations", limit=20, offset=20,
    )

    assert result.success
    assert captured["params"]["offset"] == 20
    assert result.data["has_more"] is True
    assert result.data["next_offset"] == 40


@pytest.mark.asyncio
async def test_resolve_paper_id_accepts_match_list_response(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return {
            "success": True,
            "data": {
                "data": [
                    {
                        "paperId": PAPER["paperId"],
                        "title": PAPER["title"],
                        "matchScore": 168.0,
                    }
                ]
            },
            "metadata": {"status_code": 200},
        }

    monkeypatch.setattr(SemanticScholarClient, "_request_json", fake_request)

    paper_id = await SemanticScholarClient()._resolve_paper_id(PAPER["title"])

    assert paper_id == PAPER["paperId"]


@pytest.mark.asyncio
async def test_resolve_paper_id_keeps_legacy_match_response_compatibility(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return {
            "success": True,
            "data": {"paperId": PAPER["paperId"], "title": PAPER["title"]},
            "metadata": {"status_code": 200},
        }

    monkeypatch.setattr(SemanticScholarClient, "_request_json", fake_request)

    paper_id = await SemanticScholarClient()._resolve_paper_id(PAPER["title"])

    assert paper_id == PAPER["paperId"]


@pytest.mark.asyncio
async def test_api_key_is_header_only(monkeypatch):
    captured = {}
    real_client = httpx.AsyncClient

    async def handler(request):
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"total": 0, "data": []})

    def mock_client(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-s2-key")
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.httpx.AsyncClient",
        mock_client,
    )
    result = await SemanticScholarClient().search("agent evaluation", 2)

    assert result.success
    assert captured["headers"]["x-api-key"] == "secret-s2-key"
    assert "secret-s2-key" not in captured["url"]
    assert "secret-s2-key" not in str(result.to_dict())


@pytest.mark.asyncio
async def test_401_fails_without_retry(monkeypatch):
    calls = 0
    real_client = httpx.AsyncClient

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"message": "bad key"})

    def mock_client(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    monkeypatch.setenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "3")
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.httpx.AsyncClient",
        mock_client,
    )
    result = await SemanticScholarClient().search("agent evaluation", 2)

    assert not result.success
    assert calls == 1
    assert "401" in result.error


@pytest.mark.asyncio
async def test_429_retries_with_retry_after(monkeypatch):
    calls = 0
    sleeps = []
    real_client = httpx.AsyncClient

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"total": 1, "data": [PAPER]})

    def mock_client(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout"))

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "1")
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.httpx.AsyncClient",
        mock_client,
    )
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.asyncio.sleep",
        fake_sleep,
    )
    result = await SemanticScholarClient().search("agent evaluation", 2)

    assert result.success
    assert calls == 2
    assert sleeps == [0.0]


@pytest.mark.asyncio
async def test_429_uses_independent_extended_retry_budget(monkeypatch):
    calls = 0
    sleeps = []
    real_client = httpx.AsyncClient

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"total": 1, "data": [PAPER]})

    def mock_client(*args, **kwargs):
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=kwargs.get("timeout"),
        )

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "0")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_RATE_LIMIT_RETRIES", "3")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_MAX_RETRY_WAIT_SECONDS", "10")
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.httpx.AsyncClient",
        mock_client,
    )
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.asyncio.sleep",
        fake_sleep,
    )

    result = await SemanticScholarClient().search("agent evaluation", 2)

    assert result.success
    assert calls == 4
    assert sleeps == [1.0, 1.0, 1.0]
    assert result.metadata["rate_limit_retries"] == 3
    assert result.metadata["retry_wait_seconds"] == 3.0


@pytest.mark.asyncio
async def test_429_stops_at_bounded_retry_count(monkeypatch):
    calls = 0
    real_client = httpx.AsyncClient

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "0"})

    def mock_client(*args, **kwargs):
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=kwargs.get("timeout"),
        )

    async def fake_sleep(delay):
        return None

    monkeypatch.setenv("SEMANTIC_SCHOLAR_RATE_LIMIT_RETRIES", "2")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_MAX_RETRY_WAIT_SECONDS", "10")
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.httpx.AsyncClient",
        mock_client,
    )
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.asyncio.sleep",
        fake_sleep,
    )

    result = await SemanticScholarClient().search("agent evaluation", 2)

    assert not result.success
    assert calls == 3
    assert "429" in result.error
    assert result.metadata["rate_limit_retries"] == 2


@pytest.mark.asyncio
async def test_429_stops_when_total_wait_budget_would_be_exceeded(monkeypatch):
    calls = 0
    sleeps = []
    real_client = httpx.AsyncClient

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "30"})

    def mock_client(*args, **kwargs):
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=kwargs.get("timeout"),
        )

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setenv("SEMANTIC_SCHOLAR_RATE_LIMIT_RETRIES", "6")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_MAX_RETRY_WAIT_SECONDS", "45")
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.httpx.AsyncClient",
        mock_client,
    )
    monkeypatch.setattr(
        "app.tools.semantic_scholar_provider.asyncio.sleep",
        fake_sleep,
    )

    result = await SemanticScholarClient().search("agent evaluation", 2)

    assert not result.success
    assert calls == 2
    assert sleeps == [30.0]
    assert "retry wait budget exhausted" in result.error
    assert result.metadata["retry_wait_budget_exhausted"] is True


def test_registry_exposes_three_semantic_scholar_capabilities():
    registry = ToolRegistry.get_instance()
    search_tools = registry.list_for_task("search")
    assert "semantic_scholar_search" in search_tools
    assert "semantic_scholar_recommendations" in search_tools
    assert "semantic_scholar_graph" in search_tools
    schemas = registry.get_function_schemas(search_tools)
    names = {item["function"]["name"] for item in schemas}
    assert {
        "semantic_scholar_search",
        "semantic_scholar_recommendations",
        "semantic_scholar_graph",
    }.issubset(names)


def test_graph_relation_enum_is_enforced():
    registry = ToolRegistry.get_instance()
    error = validate_tool_args_against_schema(
        "semantic_scholar_graph",
        {"paper_query": "Attention Is All You Need", "relation": "invented"},
        registry,
    )
    assert error and "must be one of" in error


def _direct_rule_item(decision, user_request: str):
    """把 Controller 决策折算成 Planner 的 ATOMIC WorkItem（规则模式），用于断言工具选型。"""
    spec = ExecutionSpec(
        request_id="test",
        user_request=user_request,
        intent=decision.intent,
        execution_class=ExecutionClass.ATOMIC,
        execution_route=decision.execution_route,
        research_topic=decision.research_topic,
        metadata={
            "agent_mode": "rule",
            "controller_decision": {"requested_count": decision.requested_count},
        },
    )
    return PlannerAgent().plan(spec).items[0]


@pytest.mark.asyncio
async def test_controller_routes_explicit_semantic_scholar_recommendation():
    decision = await IntentController().decide(
        "请用 Semantic Scholar 推荐 3 篇关于 Agent evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )
    assert decision.intent == "paper_recommendation"
    assert decision.selected_tools == []
    assert decision.research_topic == "Agent evaluation"
    # 关键步骤：显式 Semantic Scholar 推荐的工具选型在 Planner 规则计划。
    item = _direct_rule_item(
        decision, user_request="请用 Semantic Scholar 推荐 3 篇关于 Agent evaluation 的论文"
    )
    assert item.allowed_tools == ["semantic_scholar_recommendations"]
    assert item.input_data["limit"] == 3


@pytest.mark.asyncio
async def test_controller_routes_paper_references_to_graph_tool():
    decision = await IntentController().decide(
        "查看《Attention Is All You Need》的参考文献",
        max_sources=5,
        agent_mode="rule",
    )
    assert decision.intent == "paper_graph_lookup"
    assert decision.selected_tools == []
    # 关键步骤：图谱分支保留完整请求在 research_topic，paper_query/relation
    # 的解析已下沉到 Planner 规则计划（Controller 只定意图）。
    assert decision.research_topic == "查看《Attention Is All You Need》的参考文献"
    item = _direct_rule_item(
        decision, user_request="查看《Attention Is All You Need》的参考文献"
    )
    assert item.allowed_tools == ["semantic_scholar_graph"]
    assert item.input_data["relation"] == "references"
    assert item.input_data["paper_query"] == "Attention Is All You Need"


class FakeSemanticRecommendationsTool(BaseTool):
    task_types = ("search",)

    @property
    def name(self):
        return "semantic_scholar_recommendations"

    @property
    def description(self):
        return "Fake Semantic Scholar recommendations"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["topic"],
        }

    async def _arun(self, **kwargs):
        count = kwargs["limit"]
        return ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "source_id": f"s2:fake-{index}",
                        "title": f"S2 Recommendation {index}",
                        "url": f"https://www.semanticscholar.org/paper/fake-{index}",
                        "snippet": "A traceable Semantic Scholar recommendation.",
                        "authors": ["Researcher"],
                        "year": 2025,
                        "venue": "Test",
                        "source_type": "paper",
                        "provider": "semantic_scholar",
                    }
                    for index in range(count)
                ]
            },
        )


@pytest.mark.asyncio
async def test_runtime_executes_semantic_scholar_direct_route_without_full_dag():
    ToolRegistry.get_instance().register(FakeSemanticRecommendationsTool())
    result = await run_graph(
        topic="用 Semantic Scholar 推荐 2 篇关于 RAG evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )

    assert result["status"] == "completed"
    assert result["execution_route"] == "direct_tool"
    # 关键步骤：direct 工具选型已移到 Planner，路由输出仅保留兼容占位。
    assert result["selected_tools"] == []
    assert len(result["sources"]) == 2
    assert not any(item.get("event") == "analysis_complete" for item in result["trace"])


@pytest.mark.semantic_scholar_live
@pytest.mark.asyncio
async def test_semantic_scholar_live_search():
    if os.getenv("RUN_SEMANTIC_SCHOLAR_LIVE", "false").lower() != "true":
        pytest.skip("Set RUN_SEMANTIC_SCHOLAR_LIVE=true for the live smoke test")
    if not os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        pytest.skip("SEMANTIC_SCHOLAR_API_KEY is required to avoid the shared anonymous rate limit")
    result = await SemanticScholarClient().search("LLM agent evaluation", 1)
    assert result.success, result.error
    assert result.data["results"]
    assert result.data["results"][0]["provider"] == "semantic_scholar"
