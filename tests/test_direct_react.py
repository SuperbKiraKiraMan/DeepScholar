"""direct_tool 的 ReAct 执行路径测试。

覆盖新架构下 direct 意图的两种执行形态：
- 规则模式：Planner 确定性选工具，Worker 单发（无 LLM）。
- LLM 模式：Worker 的 ReAct 循环从检索目录自主挑工具并格式化答案。

本文件重点验证 LLM 模式（_run_direct_react + _finalize_direct_output）：
1. ReAct 循环实际调用 Planner 授权的目录工具；
2. LLM 从请求原文解析图谱查询（paper_query + relation）；
3. recommend_more 时把会话已推荐论文去重，只返回未见论文。
"""

import asyncio
from typing import Any, Dict, List

import pytest

from app.agents.llm_worker import LLMWorker
from app.graph.runtime import run_graph
from app.llm.client import FakeLLMClient, reset_llm_client
from app.services.session_store import SessionContext
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import ToolRegistry


# ---------------------------------------------------------------
# 复用的假工具与假客户端（与 test_phase_b_agentic_retrieval 同构）
# ---------------------------------------------------------------

def _source(
    source_id: str,
    *,
    provider: str = "semantic_scholar",
) -> Dict[str, Any]:
    return {
        "source_id": source_id,
        "paper_id": source_id,
        "title": f"Paper {source_id}",
        "url": f"https://example.org/{source_id}",
        "year": 2025,
        "snippet": "A traceable source for direct ReAct routing.",
        "full_text": "A traceable source for direct ReAct routing.",
        "provider": provider,
        "retrieval_score": 0.9,
    }


class ScriptedRetrievalTool(BaseTool):
    """按预置序列出结果的检索工具，记录每次入参。"""

    task_types = ("search",)

    def __init__(self, name: str, results: List[ToolResult]):
        super().__init__()
        self._name = name
        self.results = list(results)
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
            }
            required = ["query"]
        return {"type": "object", "properties": properties, "required": required}

    async def _arun(self, **kwargs) -> ToolResult:
        self.calls.append(dict(kwargs))
        if not self.results:
            return ToolResult(success=False, error="script exhausted")
        return self.results.pop(0)


def _result(sources: List[Dict[str, Any]]) -> ToolResult:
    """检索工具返回：同时暴露 sources/results，兼容两种消费者。"""
    return ToolResult(
        success=True,
        data={"results": sources, "sources": sources, "total_found": len(sources)},
    )


def _install_llm(intent, fc_responses: List[Dict[str, Any]]) -> FakeLLMClient:
    """安装假客户端：先给 Controller 一个意图响应，再给 ReAct 循环一串 FC 响应。"""
    import app.llm.client as client_module

    fake = FakeLLMClient()
    fake.set_intent_responses([intent])
    fake.set_fc_responses(fc_responses)
    client_module._global_client = fake
    return fake


def _direct_intent(intent: str, research_topic: str) -> Dict[str, Any]:
    return {
        "intent": intent,
        "execution_route": "direct_tool",
        "research_topic": research_topic,
        "requested_count": 3,
        "confidence": 0.98,
        "reasoning": "scripted intent for direct ReAct test",
    }


def _finish_fc() -> Dict[str, Any]:
    return {"_finish": True, "content": "sufficient for this request"}


# ---------------------------------------------------------------
# 测试
# ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_direct_llm_mode_react_loop_runs_tool_and_formats_answer():
    """LLM 模式下 direct 推荐：ReAct 循环调用目录工具，并产出 markdown 答案。"""
    tool = ScriptedRetrievalTool(
        "semantic_scholar_recommendations",
        [_result([_source("rec-1"), _source("rec-2"), _source("rec-3")])],
    )
    ToolRegistry.get_instance().register(tool)
    _install_llm(
        _direct_intent("paper_recommendation", "RAG evaluation"),
        [
            {
                "_finish": False,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "semantic_scholar_recommendations",
                    "arguments": {"topic": "RAG evaluation", "limit": 3},
                }],
            },
            _finish_fc(),
        ],
    )

    result = await run_graph(
        topic="推荐 3 篇关于 RAG evaluation 的论文",
        max_sources=3,
        agent_mode="llm",
    )

    assert result["status"] == "completed"
    assert result["intent"] == "paper_recommendation"
    assert result["execution_route"] == "direct_tool"
    # ReAct 循环确实调用了一次推荐工具，且入参由 LLM 决策产生。
    assert tool.calls == [{"topic": "RAG evaluation", "limit": 3}]
    assert len(result["sources"]) == 3
    # 统一收口后的答案：助手口吻的自然语言回复，不回显 # 大标题、不写论文列表。
    answer = result["answer"]
    assert answer.startswith("针对您的请求《推荐 3 篇关于 RAG evaluation 的论文》")
    assert "3 篇" in answer
    assert "点击标题可查看原文" in answer
    # 论文条目不重复写进 answer，去重后的 sources 才是前端卡片的唯一事实源。
    assert "Paper rec-1" not in answer


@pytest.mark.asyncio
async def test_direct_llm_react_parses_graph_query_from_request():
    """LLM 模式下 direct 图谱查询：由 ReAct 循环从请求原文解析 paper_query 与 relation。"""
    tool = ScriptedRetrievalTool(
        "semantic_scholar_graph",
        [_result([_source("s2:cited-1"), _source("s2:cited-2")])],
    )
    ToolRegistry.get_instance().register(tool)
    _install_llm(
        _direct_intent("paper_graph_lookup", "Attention Is All You Need"),
        [
            {
                "_finish": False,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "semantic_scholar_graph",
                    "arguments": {
                        "paper_query": "Attention Is All You Need",
                        "relation": "citations",
                        "limit": 3,
                    },
                }],
            },
            _finish_fc(),
        ],
    )

    result = await run_graph(
        topic="Attention Is All You Need 被哪些论文引用",
        max_sources=3,
        agent_mode="llm",
    )

    assert result["status"] == "completed"
    assert result["intent"] == "paper_graph_lookup"
    assert result["execution_route"] == "direct_tool"
    # 关键步骤：图谱解析完全由 ReAct 的 LLM 决策完成，Controller/Planner 不再猜参数。
    assert tool.calls == [{
        "paper_query": "Attention Is All You Need",
        "relation": "citations",
        "limit": 3,
    }]
    assert len(result["sources"]) == 2


@pytest.mark.asyncio
async def test_recommend_more_dedups_session_papers_in_react_loop():
    """recommend_more：ReAct 输出与会话已推荐论文去重，只返回未见论文。"""
    session = SessionContext(
        session_id="sess-recommend-more",
        last_recommendation_topic="RAG evaluation",
        recommended_papers=[
            _source("s2:seen-1", provider="semantic_scholar"),
            _source("s2:seen-2", provider="semantic_scholar"),
        ],
    )
    # 工具返回 3 篇，其中 s2:seen-1 与会话重复，最终只应保留 2 篇新的。
    tool = ScriptedRetrievalTool(
        "semantic_scholar_recommendations",
        [_result([_source("s2:seen-1"), _source("s2:new-1"), _source("s2:new-2")])],
    )
    ToolRegistry.get_instance().register(tool)
    _install_llm(
        _direct_intent("recommend_more", "RAG evaluation"),
        [
            {
                "_finish": False,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "semantic_scholar_recommendations",
                    "arguments": {"topic": "RAG evaluation", "limit": 3},
                }],
            },
            _finish_fc(),
        ],
    )

    result = await run_graph(
        topic="再推荐几篇",
        max_sources=3,
        agent_mode="llm",
        session_context=session,
    )

    assert result["status"] == "completed"
    assert result["intent"] == "recommend_more"
    # 关键步骤：s2:seen-1 被 _finalize_direct_output 去重，未再出现在结果里。
    returned_ids = {source["source_id"] for source in result["sources"]}
    assert returned_ids == {"s2:new-1", "s2:new-2"}
    # 回复只交代数量，论文列表落在 sources（前端卡片渲染），重复论文不会泄漏进 answer。
    assert "s2:seen-1" not in result["answer"]
    assert "s2:new-2" not in result["answer"]
    assert "2 篇" in result["answer"]


@pytest.mark.asyncio
async def test_direct_llm_mode_trims_sources_to_requested_count():
    """LLM 模式篇数约束：工具返回超量来源时，limit 被 clamp 到剩余需求、
    最终交付裁剪到请求篇数（源头 + 兜底双层保险）。"""
    over_capacity = [_source(f"rec-{i}") for i in range(20)]
    tool = ScriptedRetrievalTool(
        "semantic_scholar_recommendations",
        [_result(over_capacity)],
    )
    ToolRegistry.get_instance().register(tool)
    _install_llm(
        _direct_intent("paper_recommendation", "RAG evaluation"),
        [
            {
                "_finish": False,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "semantic_scholar_recommendations",
                    "arguments": {"topic": "RAG evaluation", "limit": 20},
                }],
            },
            _finish_fc(),
        ],
    )

    result = await run_graph(
        topic="推荐 3 篇关于 RAG evaluation 的论文",
        max_sources=3,
        agent_mode="llm",
    )

    assert result["status"] == "completed"
    # 源头 clamp：LLM 填的 limit=20 被收敛到剩余需求 3。
    assert tool.calls == [{"topic": "RAG evaluation", "limit": 3}]
    # 兜底裁剪：即便工具返回 20 篇，最终只交付 3 篇。
    assert len(result["sources"]) == 3
    assert "3 篇" in result["answer"]


def _recommendation_events(result):
    return [
        event for event in (result.get("trace") or [])
        if isinstance(event, dict) and event.get("event") == "direct_reviewer_complete"
    ]


@pytest.mark.asyncio
async def test_direct_merge_emits_recommendation_number_event():
    """merge 收口补发 direct_reviewer_complete：run 记录携带真实起始/结束序号，
    历史会话恢复据此还原显示编号，不再退化为前端按 count 累计。"""
    tool = ScriptedRetrievalTool(
        "semantic_scholar_recommendations",
        [_result([_source("rec-1"), _source("rec-2")])],
    )
    ToolRegistry.get_instance().register(tool)
    _install_llm(
        _direct_intent("paper_recommendation", "RAG evaluation"),
        [
            {
                "_finish": False,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "semantic_scholar_recommendations",
                    "arguments": {"topic": "RAG evaluation", "limit": 5},
                }],
            },
            _finish_fc(),
        ],
    )

    result = await run_graph(
        topic="推荐 3 篇关于 RAG evaluation 的论文",
        max_sources=3,
        agent_mode="llm",
    )

    events = _recommendation_events(result)
    assert events, "merge 收口应补发 direct_reviewer_complete 序号事件"
    event = events[0]
    # 首次推荐从 1 开始编号，2 篇即 1..2。
    assert event["recommendation_number_start"] == 1
    assert event["recommendation_number_end"] == 2
    assert event["source_count"] == 2


@pytest.mark.asyncio
async def test_direct_recommend_more_numbering_continues_from_session():
    """recommend_more 序号延续：merge 补发的事件从会话已推荐数量之后编号。"""
    session = SessionContext(
        session_id="sess-ordinal",
        last_recommendation_topic="RAG evaluation",
        recommended_papers=[_source("s2:seen-1"), _source("s2:seen-2")],
    )
    tool = ScriptedRetrievalTool(
        "semantic_scholar_recommendations",
        [_result([_source("s2:new-1"), _source("s2:new-2")])],
    )
    ToolRegistry.get_instance().register(tool)
    _install_llm(
        _direct_intent("recommend_more", "RAG evaluation"),
        [
            {
                "_finish": False,
                "tool_calls": [{
                    "id": "call_1",
                    "name": "semantic_scholar_recommendations",
                    "arguments": {"topic": "RAG evaluation", "limit": 2},
                }],
            },
            _finish_fc(),
        ],
    )

    result = await run_graph(
        topic="再推荐 2 篇",
        max_sources=5,
        agent_mode="llm",
        session_context=session,
    )

    events = _recommendation_events(result)
    assert events, "recommend_more 也应在 merge 收口补发序号事件"
    # 会话已推荐 2 篇，本轮 2 篇新论文编号 3..4。
    assert events[0]["recommendation_number_start"] == 3
    assert events[0]["recommendation_number_end"] == 4
    assert events[0]["source_count"] == 2
