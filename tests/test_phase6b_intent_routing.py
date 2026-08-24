"""V1.2 intent-driven Controller and dynamic route tests."""

from typing import Any, Dict

import pytest

from app.agents.controller import IntentController
from app.agents.planner import PlannerAgent
from app.agents.schemas import ExecutionClass, ExecutionSpec
from app.graph.runtime import run_graph, send_to_search_worker
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry


class FakeRecommendationMCPTool(BaseTool):
    task_types = ("search",)

    def __init__(self):
        super().__init__()
        self.calls = []

    @property
    def name(self) -> str:
        return "mcp__academic-research__recommend_papers"

    @property
    def description(self) -> str:
        return "Recommend academic papers for a topic."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["topic"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        count = kwargs.get("limit", 5)
        return ToolResult(
            success=True,
            data={
                "sources": [
                    {
                        "source_id": f"rec-{index}",
                        "title": f"Recommended Paper {index}",
                        "url": f"https://example.org/paper/{index}",
                        "snippet": f"Evidence-aware paper about {kwargs['topic']}.",
                        "full_text": f"Evidence-aware paper about {kwargs['topic']}.",
                        "authors": ["Researcher"],
                        "year": 2025,
                        "venue": "TestConf",
                        "source_type": "paper",
                        "provider": "mcp-test",
                    }
                    for index in range(1, count + 1)
                ]
            },
        )


class FakeSemanticScholarMCPTool(FakeRecommendationMCPTool):
    @property
    def name(self) -> str:
        return "mcp__academic-research__semantic_scholar_recommendations"


class FailingSemanticScholarMCPTool(FakeSemanticScholarMCPTool):
    async def _arun(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(
            success=False,
            tool_name=self.name,
            error="MCP Semantic Scholar unavailable",
        )


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
            "controller_decision": {
                "requested_count": decision.requested_count,
            },
        },
    )
    return PlannerAgent().plan(spec).items[0]


@pytest.mark.asyncio
async def test_rule_controller_routes_recommendation_to_direct_tool():
    decision = await IntentController().decide(
        "请推荐 3 篇关于 RAG evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )
    assert decision.intent == "paper_recommendation"
    assert decision.execution_route == "direct_tool"
    assert decision.research_topic == "RAG evaluation"
    assert decision.requested_count == 3


@pytest.mark.asyncio
async def test_controller_prefers_mcp_semantic_scholar_wrapper_when_registered():
    # 关键步骤：Controller 只分类意图，不再替 direct 挑工具（selected_tools 恒为空）；
    # MCP 包装器优先的取舍已下沉到 Planner 的规则计划。
    tool = FakeSemanticScholarMCPTool()
    ToolRegistry.get_instance().register(tool)

    decision = await IntentController().decide(
        "请用 Semantic Scholar 推荐 4 篇关于 Agent evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )

    assert decision.execution_route == "direct_tool"
    assert decision.selected_tools == []
    assert decision.selected_tool_args == {}

    # Planner 规则计划：注册了 Semantic Scholar 的 MCP 包装器时优先选它。
    item = _direct_rule_item(decision, user_request="请用 Semantic Scholar 推荐 4 篇关于 Agent evaluation 的论文")
    assert item.allowed_tools == [tool.name]
    assert item.input_data["topic"] == "Agent evaluation"
    assert item.input_data["limit"] == 4


@pytest.mark.asyncio
async def test_mcp_semantic_scholar_direct_route_preserves_trusted_args():
    tool = FakeSemanticScholarMCPTool()
    ToolRegistry.get_instance().register(tool)

    result = await run_graph(
        topic="请用 Semantic Scholar 推荐 2 篇关于 Agent evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )

    assert result["status"] == "completed"
    assert result["selected_tools"] == []
    assert len(result["sources"]) == 2
    assert tool.calls == [{"topic": "Agent evaluation", "limit": 2}]
    assert any(
        event.get("tool_name") == tool.name
        for event in result["trace"]
        if event.get("event") == "tool_finished"
    )


@pytest.mark.asyncio
async def test_mcp_semantic_scholar_failure_falls_back_to_academic_provider():
    tool = FailingSemanticScholarMCPTool()
    ToolRegistry.get_instance().register(tool)

    result = await run_graph(
        topic="请用 Semantic Scholar 推荐 2 篇关于 Agent evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )

    assert result["selected_tools"] == []
    assert len(result["sources"]) == 2
    assert all(source["provider"] == "mock" for source in result["sources"])
    assert any(
        event.get("event") == "tool_loop_fallback"
        and event.get("from_tool") == tool.name
        and event.get("to_tool") == "academic_search"
        for event in result["trace"]
    )


@pytest.mark.asyncio
async def test_controller_routes_comparison_to_full_research():
    decision = await IntentController().decide(
        "推荐几篇 Agent 论文，并比较它们的评估方法",
        max_sources=5,
        agent_mode="rule",
    )
    assert decision.intent == "deep_research"
    assert decision.execution_route == "full_research"


@pytest.mark.asyncio
async def test_generic_recommendation_prefers_retried_builtin_over_generic_mcp():
    tool = FakeRecommendationMCPTool()
    ToolRegistry.get_instance().register(tool)

    decision = await IntentController().decide(
        "请推荐 3 篇关于 LLM Agent evaluation 的论文",
        max_sources=5,
        agent_mode="rule",
    )

    assert decision.intent == "paper_recommendation"
    assert decision.execution_route == "direct_tool"
    assert decision.selected_tools == []
    assert decision.selected_tool_args == {}

    # 关键步骤：Controller 不再选工具，内置 canonical 优于通用 MCP 包装器的取舍在 Planner。
    item = _direct_rule_item(
        decision, user_request="请推荐 3 篇关于 LLM Agent evaluation 的论文"
    )
    assert item.allowed_tools == ["semantic_scholar_recommendations"]
    assert item.input_data["topic"] == "LLM Agent evaluation"
    assert item.input_data["limit"] == 3
    assert tool.calls == []


def test_send_payload_preserves_agent_mode_and_planned_tool():
    sends = send_to_search_worker({
        "topic": "Agent evaluation",
        "max_sources": 5,
        "agent_mode": "llm",
        "backend": "graph_send",
        "search_tasks": [{
            "task_id": "search_1",
            "task_type": "search",
            "description": "recommend Agent evaluation papers",
            "tool_plan": ["mcp__academic-research__recommend_papers"],
        }],
    })
    assert len(sends) == 1
    assert sends[0].arg["agent_mode"] == "llm"
    assert sends[0].arg["current_search_task"]["tool_plan"] == [
        "mcp__academic-research__recommend_papers"
    ]


@pytest.mark.asyncio
async def test_plain_lookup_uses_short_path_without_mcp():
    result = await run_graph(
        topic="查找 2 篇关于 RAG 的论文",
        max_sources=5,
        agent_mode="rule",
    )
    assert result["intent"] == "literature_search"
    assert result["execution_route"] == "direct_tool"
    assert len(result["sources"]) == 2
    assert result["task_dag"]["task_count"] == 1


@pytest.mark.asyncio
async def test_deep_request_keeps_parallel_evidence_pipeline():
    result = await run_graph(
        topic="调研 LLM Agent evaluation methods，比较主要方法并总结局限",
        max_sources=3,
        agent_mode="rule",
    )
    assert result["intent"] == "deep_research"
    assert result["execution_route"] == "full_research"
    assert result["task_dag"]["task_count"] >= 5
    assert any(event.get("event") == "send_dispatch" for event in result["trace"])
    assert any(event.get("event") == "analysis_complete" for event in result["trace"])


@pytest.mark.asyncio
async def test_llm_controller_channel_does_not_consume_planner_responses(monkeypatch):
    import app.llm.client as client_module
    from app.llm.client import FakeLLMClient

    tool = FakeRecommendationMCPTool()
    ToolRegistry.get_instance().register(tool)
    fake = FakeLLMClient(responses=[{"planner": "reserved"}])
    fake.set_intent_responses([{
        "intent": "paper_recommendation",
        "execution_route": "direct_tool",
        "research_topic": "Agent evaluation",
        "selected_tool": tool.name,
        "requested_count": 4,
        "confidence": 0.98,
        "reasoning": "User asks only for recommendations",
    }])
    client_module._global_client = fake

    decision = await IntentController().decide(
        "帮我推荐 Agent evaluation 论文",
        max_sources=5,
        agent_mode="llm",
    )
    assert decision.classifier == "llm"
    assert decision.selected_tools == []
    assert fake._structured_index == 0
    assert len(fake.intent_calls) == 1
