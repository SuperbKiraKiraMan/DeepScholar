"""Sessions, compaction, memory, and multi-turn routes."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.agents.controller import IntentController
from app.agents.reference_resolver import ReferenceResolver
from app.api.routes import session_store
from app.main import app
from app.services.context_compressor import CompactionConfig, ContextCompressor
from app.services.run_store import run_store
from app.services.session_store import SessionExpiredError, SessionStore
from app.services.user_memory import MemoryEntry, UserMemoryStore


def _papers():
    return [
        {
            "source_id": "p1",
            "title": "Paper One",
            "snippet": "Paper One uses retrieval augmented generation with dense retrieval.",
            "full_text": "Methods. Paper One uses dense retrieval. Limitations. Evaluation is small.",
            "url": "https://example.org/p1",
        },
        {
            "source_id": "p2",
            "title": "Paper Two",
            "snippet": "Paper Two uses sparse retrieval and evaluates accuracy.",
            "full_text": "Methods. Paper Two uses sparse retrieval. Results. Accuracy improved.",
            "url": "https://example.org/p2",
        },
        {
            "source_id": "p3",
            "title": "Paper Three",
            "snippet": "Paper Three is a hybrid system.",
            "url": "https://example.org/p3",
        },
    ]


def _mock_academic_search(monkeypatch):
    """把真实网络检索（Semantic Scholar/OpenAlex/Crossref）替换为固定论文。

    这两个 API 会话集成测试关心的是意图路由、会话回填与多轮指代解析，
    不是真实检索结果；而沙箱里 Semantic Scholar 20s 超时 + 指数退避重试
    会让单次 run 拖到 30s+，造成偶发超时。这里沿用本文件已有的
    test_recommend_more_excludes_history... 的 mock 先例，把搜索工具换成
    确定性数据，让 run 秒级完成。

    覆盖两条检索路径：
    - Worker._execute_search → AcademicSearchTool（rule 模式 search worker）
    - node_capability_worker → ToolRegistry 里的 SemanticScholar* 工具
      （recommendation/search/graph 走 direct_tool 能力调用）
    """
    from app.tools.academic_search_tool import AcademicSearchTool
    from app.tools.base import ToolResult
    from app.tools.semantic_scholar_tools import (
        SemanticScholarGraphTool,
        SemanticScholarRecommendationsTool,
        SemanticScholarSearchTool,
    )

    papers = _papers()

    def _canned(tool_name, with_query=False):
        async def fake_arun(self, **kwargs):
            data = {"sources": papers, "results": papers}
            if with_query:
                data["query"] = kwargs.get("query", "")
                data["total_found"] = len(papers)
            return ToolResult(success=True, tool_name=tool_name, data=data)

        return fake_arun

    monkeypatch.setattr(
        AcademicSearchTool, "_arun", _canned("academic_search", with_query=True)
    )
    monkeypatch.setattr(
        SemanticScholarSearchTool, "_arun", _canned("semantic_scholar_search")
    )
    monkeypatch.setattr(
        SemanticScholarRecommendationsTool,
        "_arun",
        _canned("semantic_scholar_recommendations"),
    )
    monkeypatch.setattr(
        SemanticScholarGraphTool, "_arun", _canned("semantic_scholar_graph")
    )


async def _start_async_run(topic, session_id, max_sources=None):
    """通过 research_async 创建异步 run，并 await 其后台任务完成后再返回。

    TestClient 会在请求返回时取消后台任务，因此这里直接驱动路由函数本身，
    在测试的事件循环里 await 注册的后台任务（与真实服务器行为一致）。
    任务完成即表示 run_store 已终态、session 已回写（_update_session_after_run
    在后台任务返回前执行）。
    """
    from app.api.routes import research_async, _task_registry
    from app.api.schemas import ResearchRequest

    payload = {"topic": topic, "agent_mode": "rule"}
    if max_sources is not None:
        payload["max_sources"] = max_sources
    resp = await research_async(ResearchRequest(**payload, session_id=session_id), backend="graph_send")
    run_id = resp.run_id

    task = _task_registry.get(run_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=30)

    data = run_store.get(run_id)
    assert data is not None and data.get("status") not in {"queued", "running", "started"}, \
        f"Run {run_id} did not finish; last: {data}"
    return data


def test_session_store_tracks_turns_papers_and_expiry():
    store = SessionStore(default_ttl_minutes=30)
    context = store.create("session-a")
    store.set_recommended_papers(context.session_id, _papers())
    store.set_active_paper(context.session_id, "p2")
    for index in range(12):
        store.record_turn(
            context.session_id,
            user_content=f"question {index}",
            assistant_content=f"answer {index}",
            intent="paper_qa",
        )
    current = store.get(context.session_id)
    assert current.turn_count == 12
    assert len(current.recent_messages) == 20
    assert current.active_paper_id == "p2"
    assert current.last_mentioned_paper_ids[0] == "p2"

    store._sessions[context.session_id].expires_at_ms = int(time.time() * 1000) - 1
    with pytest.raises(SessionExpiredError):
        store.get(context.session_id)


def test_session_store_appends_recommendations_and_tracks_latest_batch():
    store = SessionStore()
    context = store.create("recommend-history")
    store.append_recommended_papers(
        context.session_id, _papers()[:2], recommendation_topic="RAG evaluation"
    )
    store.append_recommended_papers(
        context.session_id, [_papers()[1], _papers()[2]], recommendation_topic="RAG evaluation"
    )

    current = store.get(context.session_id)
    assert [paper["source_id"] for paper in current.recommended_papers] == ["p1", "p2", "p3"]
    assert [paper["source_id"] for paper in current.last_recommendation_batch] == ["p3"]
    assert current.last_recommendation_topic == "RAG evaluation"


def test_session_store_restores_papers_active_paper_and_turns_from_runs():
    store = SessionStore()
    restored = store.restore_from_runs(
        "expired-session",
        [
            {
                "run_id": "run-1",
                "topic": "推荐 RAG 论文",
                "intent": "paper_recommendation",
                "research_topic": "RAG evaluation",
                "sources": _papers()[:2],
                "answer": "推荐两篇论文。",
            },
            {
                "run_id": "run-2",
                "topic": "第二篇的局限是什么？",
                "intent": "paper_qa",
                "resolved_paper_ids": ["p2"],
                "sources": [],
                "answer": "第二篇的局限是……",
            },
        ],
    )

    assert restored.session_id != "expired-session"
    assert restored.restored_from_session_id == "expired-session"
    assert [paper["source_id"] for paper in restored.recommended_papers] == ["p1", "p2"]
    assert restored.active_paper_id == "p2"
    assert restored.turn_count == 2
    assert restored.recent_messages[-1].role == "assistant"


@pytest.mark.asyncio
async def test_reference_resolver_handles_ordinals_active_paper_and_section():
    store = SessionStore()
    context = store.create("resolver")
    store.set_recommended_papers(context.session_id, _papers())
    store.set_active_paper(context.session_id, "p3")
    store.set_report_sections(context.session_id, ["方法比较", "研究局限"], "report-1")
    context = store.get(context.session_id)
    resolver = ReferenceResolver()

    comparison = await resolver.resolve("对比第一篇和第二篇", context)
    assert comparison.resolved_paper_ids == ["p1", "p2"]
    active = await resolver.resolve("这篇论文的局限是什么？", context)
    assert active.resolved_paper_ids == ["p3"]
    section = await resolver.resolve("展开研究局限章节", context)
    assert section.resolved_section == "研究局限"


@pytest.mark.asyncio
async def test_controller_routes_three_conversation_short_paths():
    store = SessionStore()
    context = store.create("routes")
    store.set_recommended_papers(context.session_id, _papers())
    store.set_active_paper(context.session_id, "p1")
    store.set_report_sections(context.session_id, ["研究局限"], "report-1")
    context = store.get(context.session_id)
    controller = IntentController()

    qa = await controller.decide("第一篇用了什么方法？", agent_mode="rule", session_context=context)
    assert (qa.intent, qa.execution_route, qa.conversation_operation) == (
        "paper_qa", "conversation", "paper_qa"
    )
    compare = await controller.decide("对比第一篇和第二篇", agent_mode="rule", session_context=context)
    assert (compare.intent, compare.resolved_paper_ids) == ("paper_compare", ["p1", "p2"])
    assert compare.execution_route == "conversation"
    report = await controller.decide("展开研究局限章节", agent_mode="rule", session_context=context)
    assert report.intent == "report_follow_up"
    assert report.execution_route == "conversation"


@pytest.mark.asyncio
async def test_controller_blocks_unresolved_ordinal_without_deep_research_fallback():
    store = SessionStore()
    context = store.create("missing-ordinal")
    store.set_recommended_papers(context.session_id, _papers())

    decision = await IntentController().decide(
        "请说明第 4 篇论文的核心结论",
        agent_mode="rule",
        session_context=store.get(context.session_id),
    )

    assert decision.execution_route == "conversation"
    assert decision.conversation_operation == "reference_not_found"
    assert decision.clarification_message == "当前会话找不到第 4 篇"
    assert decision.intent != "deep_research"


@pytest.mark.asyncio
async def test_controller_routes_session_continuations_before_conversation():
    store = SessionStore()
    context = store.create("continuations")
    store.append_recommended_papers(
        context.session_id, _papers(), recommendation_topic="RAG evaluation"
    )
    store.set_report_sections(context.session_id, ["研究局限"], "old-report")
    context = store.get(context.session_id)
    controller = IntentController()

    more = await controller.decide(
        "再推荐10篇", max_sources=5, agent_mode="rule", session_context=context
    )
    assert more.intent == "recommend_more"
    assert more.execution_route == "direct_tool"
    assert more.requested_count == 10
    assert more.research_topic == "RAG evaluation"

    research = await controller.decide(
        "基于这些论文生成深度调研报告",
        max_sources=2,
        agent_mode="rule",
        session_context=context,
    )
    assert research.intent == "research_from_session"
    assert research.execution_route == "full_research"
    assert research.seed_paper_ids == ["p1", "p2", "p3"]
    assert research.research_topic == "RAG evaluation"


@pytest.mark.asyncio
async def test_runtime_injects_session_papers_as_research_seeds():
    from app.graph.runtime import node_controller

    store = SessionStore()
    context = store.create("seed-runtime")
    store.append_recommended_papers(
        context.session_id, _papers(), recommendation_topic="RAG evaluation"
    )
    context = store.get(context.session_id)

    delta = await node_controller({
        "topic": "基于这些论文生成深度调研报告",
        "max_sources": 2,
        "agent_mode": "rule",
        "backend": "graph_send",
        "session_context": context,
        "session_id": context.session_id,
    })

    assert delta["intent"] == "research_from_session"
    assert delta["topic"] == "RAG evaluation"
    assert delta["seed_paper_ids"] == ["p1", "p2", "p3"]
    assert [paper["source_id"] for paper in delta["_search_bucket"]] == ["p1", "p2", "p3"]
    assert delta["max_sources"] == 3


@pytest.mark.asyncio
async def test_recommend_more_excludes_history_and_returns_only_new_batch(monkeypatch):
    from app.graph.runtime import node_capability_worker
    from app.tools.registry import ToolRegistry

    store = SessionStore()
    context = store.create("recommend-more-worker")
    store.append_recommended_papers(
        context.session_id, _papers(), recommendation_topic="RAG evaluation"
    )
    context = store.get(context.session_id)

    class FakeTool:
        input_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "limit": {"type": "integer"},
            },
        }

        async def run(self, **kwargs):
            from app.tools.base import ToolResult

            assert kwargs["limit"] == 5
            return ToolResult(
                success=True,
                tool_name="semantic_scholar_recommendations",
                data={"sources": [
                    _papers()[0],
                    {"source_id": "p4", "title": "Paper Four", "url": "https://example.org/p4"},
                    {"source_id": "p5", "title": "Paper Five", "url": "https://example.org/p5"},
                ]},
            )

    class FakeRegistry:
        def get(self, _name):
            return FakeTool()

    monkeypatch.setattr(ToolRegistry, "get_instance", classmethod(lambda cls: FakeRegistry()))
    delta = await node_capability_worker({
        "intent": "recommend_more",
        "research_topic": "RAG evaluation",
        "requested_count": 2,
        "max_sources": 2,
        "selected_tools": ["semantic_scholar_recommendations"],
        "selected_tool_args": {},
        "session_context": context,
        "start_time_ms": 0,
    })

    assert [paper["source_id"] for paper in delta["sources"]] == ["p4", "p5"]


@pytest.mark.asyncio
async def test_recommend_more_report_continues_session_paper_numbers():
    """继续推荐必须使用会话总序号，保证后续“第 N 篇”指代唯一。"""
    from app.graph.runtime import node_direct_reviewer

    store = SessionStore()
    context = store.create("continued-numbering")
    store.append_recommended_papers(context.session_id, _papers())
    context = store.get(context.session_id)
    new_sources = [
        {"source_id": "p4", "title": "Paper Four", "url": "https://example.org/p4"},
        {"source_id": "p5", "title": "Paper Five", "url": "https://example.org/p5"},
    ]

    delta = await node_direct_reviewer({
        "intent": "recommend_more",
        "research_topic": "RAG evaluation",
        "sources": new_sources,
        "session_context": context,
        "start_time_ms": int(time.time() * 1000),
    })

    assert "## 4. [Paper Four]" in delta["final_report"]
    assert "## 5. [Paper Five]" in delta["final_report"]
    assert "本次新增 2 个可追溯来源，会话累计 5 篇论文" in delta["final_report"]
    trace = delta["trace"][0]
    assert trace["recommendation_number_start"] == 4
    assert trace["recommendation_number_end"] == 5

    # 会话累积后，“第四篇”仍按总体列表解析，而不是按当前批次解析。
    store.append_recommended_papers(context.session_id, new_sources)
    resolution = await ReferenceResolver().resolve(
        "第四篇论文的核心结论是什么？", store.get(context.session_id)
    )
    assert resolution.resolved_paper_ids == ["p4"]


@pytest.mark.asyncio
async def test_context_compressor_runs_l3_l1_l2_in_fixed_order(tmp_path):
    config = CompactionConfig(
        tool_result_budget_bytes=200,
        single_tool_result_bytes=100,
        persisted_preview_chars=30,
        max_messages=10,
        keep_head_messages=2,
        keep_tail_messages=6,
        recent_tool_results=2,
        l4_token_threshold=1_000_000,
    )
    compressor = ContextCompressor(config, tmp_path)
    session = SessionStore().create("compact")
    messages = []
    for index in range(7):
        messages.extend([
            {"role": "assistant", "content": f"call {index}", "tool_calls": [{"id": index}]},
            {"role": "tool", "tool_call_id": index, "content": "x" * 150},
        ])
    result = await compressor.compress(messages, session)
    assert result.levels_applied[:3] == [
        "L3_tool_result_budget", "L1_snip_compact", "L2_micro_compact"
    ]
    assert result.persisted_outputs
    assert all((tmp_path / ".task_outputs" / "sessions" / "compact") in __import__("pathlib").Path(path).parents
               for path in result.persisted_outputs)
    # A tool request and its result remain adjacent if retained.
    for index, item in enumerate(result.messages):
        if item.get("tool_calls"):
            assert index + 1 < len(result.messages)
            assert result.messages[index + 1].get("role") == "tool"


@pytest.mark.asyncio
async def test_l4_summary_persists_raw_transcript_and_uses_llm(tmp_path):
    from app.llm.client import FakeLLMClient

    fake = FakeLLMClient(responses=[{
        "topic": "RAG evaluation",
        "key_papers_discussed": ["p1"],
        "key_findings": ["Dense retrieval was discussed"],
        "pending_questions": ["Compare p2"],
        "user_constraints": ["Answer in Chinese"],
        "report_progress": "No report yet",
        "summary": "用户正在研究 RAG evaluation，已经讨论 p1，下一步需要比较 p2。",
    }])
    config = CompactionConfig(
        tool_result_budget_bytes=1_000_000,
        max_messages=100,
        recent_tool_results=10,
        l4_token_threshold=10,
        l4_min_turns=1,
        l4_turn_gap=1,
    )
    compressor = ContextCompressor(config, tmp_path)
    store = SessionStore()
    session = store.create("l4")
    session.turn_count = 2
    result = await compressor.compress(
        [{"role": "user", "content": "RAG evaluation " * 30}],
        session,
        llm_client=fake,
    )
    assert result.levels_applied == ["L4_compact_summary"]
    assert result.summary.topic == "RAG evaluation"
    assert result.transcript_path
    assert __import__("pathlib").Path(result.transcript_path).exists()
    assert len(result.messages) == 1


@pytest.mark.asyncio
async def test_user_memory_store_index_relevant_extract_and_deduplicate(tmp_path):
    from app.llm.client import FakeLLMClient

    memory = UserMemoryStore(tmp_path / ".memory")
    memory.write(MemoryEntry(
        name="citation-style", memory_type="user", title="Citation preference",
        body="The user prefers concise Chinese answers with exact evidence citations.",
        tags=["citations", "Chinese"],
    ))
    assert "citation-style" in memory.load_index()
    relevant = await memory.load_relevant("Chinese citation answer", FakeLLMClient())
    assert [item.name for item in relevant] == ["citation-style"]

    store = SessionStore()
    session = store.create("memory")
    for index in range(3):
        session = store.record_turn(
            session.session_id,
            user_content=f"I prefer concise Chinese answers {index}",
            assistant_content="Acknowledged",
            intent="paper_qa",
        )
    fake = FakeLLMClient(responses=[{
        "entries": [{
            "name": "citation-style-copy", "memory_type": "user",
            "title": "Citation preference",
            "body": "The user prefers concise Chinese answers with exact evidence citations.",
            "tags": ["citations", "Chinese"],
        }]
    }])
    stored = await memory.extract(session, llm_client=fake)
    assert stored == []


@pytest.mark.asyncio
async def test_session_api_runs_recommendation_qa_compare_and_isolates_sessions(monkeypatch):
    from app.api.routes import _public_session

    _mock_academic_search(monkeypatch)
    first = session_store.create().session_id
    second = session_store.create().session_id

    recommendation = await _start_async_run(
        "请推荐3篇关于RAG evaluation的论文", first,
    )
    assert recommendation["intent"] == "paper_recommendation"
    assert recommendation["session_id"] == first

    qa = await _start_async_run(
        "第一篇用了什么方法？", first,
    )
    assert qa["intent"] == "paper_qa"
    assert qa["is_follow_up"] is True
    assert len(qa["resolved_paper_ids"]) == 1

    compare = await _start_async_run(
        "对比第一篇和第二篇", first,
    )
    assert compare["intent"] == "paper_compare"
    assert len(compare["resolved_paper_ids"]) == 2

    assert _public_session(session_store.get(first))["turn_count"] == 3
    isolated = _public_session(session_store.get(second))
    assert isolated["turn_count"] == 0
    assert isolated["recommended_papers"] == []


@pytest.mark.asyncio
async def test_session_api_report_follow_up_reuses_existing_report(monkeypatch):
    from app.api.routes import _public_session

    _mock_academic_search(monkeypatch)
    session_id = session_store.create().session_id
    research = await _start_async_run(
        "调研 RAG evaluation methods，总结方法和局限",
        session_id,
        max_sources=2,
    )
    assert research["intent"] == "deep_research"
    context = _public_session(session_store.get(session_id))
    assert context["active_report_id"] == research["run_id"]
    assert context["last_report_sections"]

    section = context["last_report_sections"][0]
    follow_up = await _start_async_run(
        f"展开{section}章节", session_id,
    )
    assert follow_up["intent"] == "report_follow_up"
    assert follow_up["resolved_section"] == section
    assert follow_up["answer"]


def test_deleted_and_expired_session_errors_are_distinct():
    client = TestClient(app)
    deleted = client.post("/api/sessions", json={}).json()["session_id"]
    assert client.delete(f"/api/sessions/{deleted}").status_code == 200
    response = client.get(f"/api/sessions/{deleted}")
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "SESSION_NOT_FOUND"

    expired = session_store.create().session_id
    session_store._sessions[expired].expires_at_ms = int(time.time() * 1000) - 1
    response = client.get(f"/api/sessions/{expired}")
    assert response.status_code == 410
    assert response.json()["detail"]["error_code"] == "SESSION_EXPIRED"


def test_history_runs_create_restored_session_with_paper_context():
    old_session_id = "expired-history-session"
    run_id = "expired-history-run"
    session_store.delete(old_session_id)
    run_store._runs.pop(run_id, None)
    run_store.create("推荐论文", run_id=run_id)
    run_store.update(
        run_id,
        session_id=old_session_id,
        intent="paper_recommendation",
        research_topic="RAG evaluation",
        sources=_papers()[:2],
        answer="推荐两篇论文。",
        resolved_paper_ids=[],
    )
    expired = session_store.create(old_session_id)
    session_store._sessions[expired.session_id].expires_at_ms = int(time.time() * 1000) - 1

    try:
        response = TestClient(app).get(f"/api/conversations/{old_session_id}/runs")
        body = response.json()
        assert response.status_code == 200
        assert body["restored"] is True
        assert body["session_id"] != old_session_id
        assert [item["source_id"] for item in body["session"]["recommended_papers"]] == ["p1", "p2"]
        assert body["session"]["restored_from_session_id"] == old_session_id
    finally:
        run_store._runs.pop(run_id, None)
        session_store.delete(old_session_id)
        if "body" in locals():
            session_store.delete(body.get("session_id", ""))
