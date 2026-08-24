"""Contract tests for the unified conversation path."""

import pytest

from app.agents.conversation import (
    ConversationAgent,
    ConversationResponse,
    ConversationRequest,
    FollowUpContext,
    context_builder,
)
from app.agents.paper_compare import PaperCompareStrategy
from app.agents.controller import IntentController
from app.llm.client import FakeLLMClient
from app.services.run_store import run_store
from app.services.session_store import SessionContext, session_store


def _paper(identifier: str, method: str = "dense retrieval"):
    return {
        "source_id": identifier,
        "title": f"Paper {identifier}",
        "full_text": f"Methods. Paper {identifier} uses {method}. Results are reported.",
        "snippet": f"Paper {identifier} uses {method}.",
    }


def _context(*, papers=None, report=None, evidence=None, history=None, section=None):
    return FollowUpContext(
        query="", papers=papers or [], report=report, evidence=evidence or [],
        history=history or [], resolved_section=section,
        report_id=(report or {}).get("report_id"),
    )


class _PromptCaptureLLM:
    """只捕获 system prompt 的离线 LLM 替身。"""

    def __init__(self):
        self.system_prompt = ""

    async def generate_structured(self, *, system_prompt, **kwargs):
        self.system_prompt = system_prompt
        return {"success": False, "error": "test fallback"}


@pytest.mark.asyncio
async def test_conversation_language_reaches_compare_prompt_and_rule_fallback():
    client = _PromptCaptureLLM()
    papers = [_paper("p1"), _paper("p2", "sparse retrieval")]
    response, _ = await ConversationAgent().answer(
        ConversationRequest(
            query="比较这两篇论文的方法",
            context=_context(papers=papers),
            operation_hint="paper_compare",
            language="zh",
        ),
        agent_mode="llm",
        llm_client=client,
    )

    assert "Required output language: Simplified Chinese." in client.system_prompt
    assert "All summaries, dimension names, comparisons, analyses, warnings," in client.system_prompt
    assert "Paper titles, model names, dataset names, acronyms, and verbatim" in client.system_prompt
    assert "研究问题" in {item["dimension"] for item in response.comparison_dimensions}
    assert "Not available in supplied evidence" not in response.answer


@pytest.mark.asyncio
async def test_paper_compare_language_is_explicit_and_non_chinese_can_be_requested():
    client = _PromptCaptureLLM()
    result, _ = await PaperCompareStrategy().compare(
        _paper("p1"), _paper("p2"), "compare methods",
        language="en", agent_mode="llm", llm_client=client,
    )

    assert "Required output language: English." in client.system_prompt
    assert result.comparison_dimensions[0].dimension == "problem"


@pytest.mark.asyncio
async def test_graph_language_is_forwarded_to_conversation_request(monkeypatch):
    from app.graph import runtime

    class _CaptureConversationAgent:
        def __init__(self):
            self.request = None

        async def answer(self, request, **kwargs):
            self.request = request
            return ConversationResponse(answer="中文回答"), {"success": False}

    capture = _CaptureConversationAgent()
    monkeypatch.setattr(runtime, "_conversation_agent", capture)
    result = await runtime.node_conversation({
        "topic": "这篇论文的方法是什么？",
        "language": "zh",
        "agent_mode": "rule",
        "conversation_operation": "paper_qa",
        "resolved_refs": {"paper_ids": ["p1"]},
        "session_context": SessionContext(
            session_id="conversation-language",
            recommended_papers=[_paper("p1")],
            active_paper_id="p1",
        ),
        "messages": [],
    })

    assert capture.request.language == "zh"
    assert result["answer"] == "中文回答"


def test_mode_is_context_shape_not_operation_hint():
    agent = ConversationAgent()
    assert agent.mode_for(_context(papers=[_paper("p1")])) == "single_paper"
    assert agent.mode_for(_context(papers=[_paper("p1"), _paper("p2")])) == "multi_paper"
    assert agent.mode_for(_context(report={"report_id": "r1"})) == "report"
    assert agent.mode_for(_context(papers=[_paper("p1")], report={"report_id": "r1"})) == "mixed"
    assert agent.mode_for(_context(history=[{"role": "user", "content": "hello"}])) == "history_only"


@pytest.mark.asyncio
async def test_strategy_uses_hint_shape_and_query_compatibility():
    agent = ConversationAgent()
    context = _context(papers=[_paper("p1"), _paper("p2", "sparse retrieval")])
    strict, _ = await agent.answer(ConversationRequest(
        query="比较这两篇论文的方法", context=context, operation_hint="paper_compare"
    ), agent_mode="rule")
    assert (strict.mode, strict.strategy) == ("multi_paper", "paper_compare")

    incompatible, _ = await agent.answer(ConversationRequest(
        query="介绍这几篇论文的方法", context=context, operation_hint="paper_compare"
    ), agent_mode="rule")
    assert (incompatible.mode, incompatible.strategy) == ("multi_paper", "general")


@pytest.mark.asyncio
async def test_llm_mode_preserves_quote_and_id_grounding_guards():
    agent = ConversationAgent()
    paper = _paper("p1")
    qa_client = FakeLLMClient(responses=[{
        "answer": "该论文使用 dense retrieval。",
        "supporting_quotes": ["Paper p1 uses dense retrieval."],
        "quote_sections": ["Methods"],
        "confidence": 0.9,
        "paper_id": "invented",
    }])
    qa, _ = await agent.answer(ConversationRequest(
        query="这篇论文使用什么方法？", context=_context(papers=[paper]),
        operation_hint="paper_qa",
    ), agent_mode="llm", llm_client=qa_client)
    assert qa.referenced_papers == ["p1"]
    assert qa.supporting_quotes == ["Paper p1 uses dense retrieval."]

    general_client = FakeLLMClient(responses=[{
        "answer": "基于已有材料给出保守说明。",
        "referenced_papers": ["p1", "invented-paper"],
        "referenced_evidence": ["invented-evidence"],
        "supporting_quotes": ["invented quote"],
        "grounded_claims": ["invented claim"],
        "analysis": "分析内容",
    }])
    general, _ = await agent.answer(ConversationRequest(
        query="概括当前材料", context=_context(papers=[paper]), operation_hint="unknown",
    ), agent_mode="llm", llm_client=general_client)
    assert general.referenced_papers == ["p1"]
    assert general.referenced_evidence == [] and general.supporting_quotes == []
    assert general.grounded_claims == [] and general.warnings


@pytest.mark.asyncio
async def test_empty_unknown_and_cross_capability_use_general_not_unsupported():
    agent = ConversationAgent()
    report = {"report_id": "r1", "report_text": "第三章说明 dense retrieval 的实验结果。"}
    context = _context(papers=[_paper("p1"), _paper("p2")], report=report)
    for hint, query in (
        ("", "介绍已有材料中的方法"),
        ("unknown", "介绍已有材料中的方法"),
        ("paper_compare", "对比两篇论文并结合报告章节给出分析"),
    ):
        response, _ = await agent.answer(
            ConversationRequest(query=query, context=context, operation_hint=hint),
            agent_mode="rule",
        )
        assert response.strategy == "general"
        assert "unsupported" not in (response.answer + response.cannot_answer_reason).lower()


@pytest.mark.asyncio
async def test_n_paper_comparison_covers_every_paper_without_truncation():
    papers = [_paper("p1"), _paper("p2", "sparse retrieval"), _paper("p3", "hybrid retrieval")]
    response, _ = await ConversationAgent().answer(ConversationRequest(
        query="比较这三篇论文的方法与结果",
        context=_context(papers=papers),
        operation_hint="paper_compare",
    ), agent_mode="rule")
    assert response.referenced_papers == ["p1", "p2", "p3"]
    assert all(
        {entry["paper_id"] for entry in dimension["papers"]} == {"p1", "p2", "p3"}
        for dimension in response.comparison_dimensions
    )


@pytest.mark.asyncio
async def test_analysis_suggestion_queries_route_to_general_synthesis():
    agent = ConversationAgent()
    paper = _paper("p1")
    # 分析/建议类问题即使带 paper_qa hint，也统一进入 general 综合
    for query in (
        "如何改进这篇论文的方法？",
        "给出优化建议",
        "为什么这篇论文效果好？",
        "评价这两篇论文的思路",
    ):
        context = _context(papers=[paper, _paper("p2", "sparse")]) if "两篇" in query else _context(papers=[paper])
        response, _ = await agent.answer(ConversationRequest(
            query=query, context=context, operation_hint="paper_qa",
        ), agent_mode="rule")
        assert response.strategy == "general", query

    # 对照：不含分析/建议信号的单篇方法问答仍走 paper_qa
    qa, _ = await agent.answer(ConversationRequest(
        query="这篇论文的方法是什么？", context=_context(papers=[paper]),
        operation_hint="paper_qa",
    ), agent_mode="rule")
    assert qa.strategy == "paper_qa"


@pytest.mark.asyncio
async def test_history_is_not_trusted_evidence_for_quotes_and_claims():
    agent = ConversationAgent()
    paper = _paper("p1")
    history = [{"role": "assistant", "content": "历史中提到的虚构引用：hallucinated quote from history。"}]
    client = FakeLLMClient(responses=[{
        "answer": "基于历史给出说明。",
        "supporting_quotes": ["hallucinated quote from history"],
        "grounded_claims": ["基于历史得出的结论"],
        "analysis": "仅作语义上下文参考。",
    }])
    response, _ = await agent.answer(ConversationRequest(
        query="概括当前材料", context=_context(papers=[paper], history=history),
        operation_hint="unknown",
    ), agent_mode="llm", llm_client=client)
    # 引用仅在对话历史中出现、不在可信材料中，必须被剔除
    assert response.supporting_quotes == []
    # 缺少可信引用支撑的 grounded_claims 必须被清除并告警
    assert response.grounded_claims == []
    assert response.warnings


@pytest.mark.asyncio
async def test_n_paper_comparison_filters_hallucinated_paper_ids():
    papers = [_paper("p1"), _paper("p2", "sparse retrieval"), _paper("p3", "hybrid retrieval")]
    client = FakeLLMClient(responses=[{
        "answer": "三篇论文的方法对比。",
        "referenced_papers": ["p1", "p2", "p3", "fake-paper"],
        "comparison_dimensions": [
            {
                "dimension": "method",
                "papers": [
                    {"paper_id": "p1", "title": "Paper p1", "evidence": "dense retrieval"},
                    {"paper_id": "p2", "title": "Paper p2", "evidence": "sparse retrieval"},
                    {"paper_id": "p3", "title": "Paper p3", "evidence": "hybrid retrieval"},
                    {"paper_id": "fake-paper", "title": "Hallucinated", "evidence": "invented"},
                ],
                "analysis": "逐篇对比方法。",
            }
        ],
        "grounded_claims": ["三篇论文的方法对比。"],
    }])
    response, _ = await ConversationAgent().answer(ConversationRequest(
        query="比较这三篇论文的方法",
        context=_context(papers=papers),
        operation_hint="paper_compare",
    ), agent_mode="llm", llm_client=client)
    # referenced_papers 仅保留会话白名单内的 ID
    assert response.referenced_papers == ["p1", "p2", "p3"]
    # 每个对比维度的 paper_id 都必须命中白名单，臆造 ID 已剔除
    for dimension in response.comparison_dimensions:
        ids = {entry["paper_id"] for entry in dimension["papers"]}
        assert ids <= {"p1", "p2", "p3"}
        assert "fake-paper" not in ids
    # 剔除臆造 ID 后给出告警
    assert any("剔除" in warning for warning in response.warnings)


def test_context_builder_falls_back_and_report_only_keeps_all_evidence():
    report_id = run_store.create("conversation test", run_id="conversation-context-test")
    run_store.update(
        report_id,
        final_report="# 方法\n报告正文",
        sources=[_paper("p1"), _paper("p2")],
        evidence_cards=[
            {"evidence_id": "e1", "source_id": "p1", "claim": "claim one"},
            {"evidence_id": "e2", "source_id": "p2", "claim": "claim two"},
        ],
    )
    session = SessionContext(
        session_id="s1", recommended_papers=[_paper("p1"), _paper("p2")],
        active_paper_id="p1", last_mentioned_paper_ids=["p2"],
        active_report_id=report_id,
    )
    fallback = context_builder({
        "topic": "继续", "session_context": session,
        "resolved_refs": {"paper_ids": [], "report_id": report_id},
        "messages": [{"role": "user", "content": str(index)} for index in range(20)],
    })
    assert [_paper_item["source_id"] for _paper_item in fallback.papers] == ["p1", "p2"]
    assert [item["evidence_id"] for item in fallback.evidence] == ["e1", "e2"]
    assert len(fallback.history) == 12

    report_only = context_builder({
        "topic": "展开报告", "session_context": SessionContext(
            session_id="s2", active_report_id=report_id
        ),
        "resolved_refs": {"paper_ids": [], "report_id": report_id},
    })
    assert report_only.papers == []
    assert [item["evidence_id"] for item in report_only.evidence] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_missing_resources_and_single_paper_compare_degrade_safely():
    missing = FollowUpContext(
        query="这篇论文的方法", papers=[], report=None, evidence=[], history=[],
        resolved_section=None, report_id=None, missing_paper_ids=["missing"],
    )
    response, _ = await ConversationAgent().answer(ConversationRequest(
        query="这篇论文的方法", context=missing, operation_hint="paper_qa"
    ), agent_mode="rule")
    assert response.cannot_answer and "PAPER_NOT_IN_SESSION" in response.cannot_answer_reason

    degraded, _ = await ConversationAgent().answer(ConversationRequest(
        query="比较这篇论文与另一篇", context=_context(papers=[_paper("p1")]),
        operation_hint="paper_compare",
    ), agent_mode="rule")
    assert degraded.mode == "single_paper" and degraded.strategy == "general"
    assert degraded.warnings


@pytest.mark.asyncio
async def test_controller_new_topic_gate_and_reference_signal():
    session = SessionContext(
        session_id="gate", recommended_papers=[_paper("p1")], active_paper_id="p1",
        conversation_messages=[{"role": "user", "content": "old topic"}],
    )
    controller = IntentController()
    new_topic = await controller.decide(
        "分析量子纠错码的最新研究", agent_mode="rule", session_context=session
    )
    assert new_topic.execution_route == "full_research"
    follow_up = await controller.decide(
        "这篇论文的方法是什么？", agent_mode="rule", session_context=session
    )
    assert follow_up.execution_route == "conversation"
    assert follow_up.conversation_operation == "paper_qa"


def _run_like_api(context: SessionContext, query: str, sources, intent: str) -> SessionContext:
    """按真实 API 归档逻辑模拟一轮搜索/推荐的 Session 更新。"""
    from app.api.routes import _update_session_after_run
    result = {
        "intent": intent,
        "sources": sources,
        "research_topic": query,
        "resolved_paper_ids": [],
        "final_report": f"# {query}",
        "run_id": "run",
    }
    return _update_session_after_run(context, query, result)


@pytest.mark.asyncio
async def test_cross_topic_search_then_followup_points_to_latest_batch():
    """跨主题搜索后，追问必须指到最近批次论文，而不是全局池里的旧主题论文。

    场景：先搜索主题 A（papers a1/a2），再搜索不同主题 B（papers b1/b2），
    然后追问“这篇/第 1 篇/比较”。修复前序数会指回 a1/a2、active paper 为陈旧
    或为空导致拒答；修复后应解析到 b1/b2。
    """
    from app.agents.reference_resolver import ReferenceResolver

    context = session_store.create(ttl_minutes=30)
    context = _run_like_api(context, "实体对齐", [_paper("a1"), _paper("a2")], "literature_search")
    context = _run_like_api(context, "RAG", [_paper("b1"), _paper("b2")], "literature_search")

    # 会话状态：批次与 active paper 都应指向最新搜索的论文。
    assert [item["source_id"] for item in context.recommended_papers] == ["a1", "a2", "b1", "b2"]
    assert [item["source_id"] for item in context.last_recommendation_batch] == ["b1", "b2"]
    assert context.last_recommendation_batch_start == 1
    assert context.active_paper_id == "b1"

    resolver = ReferenceResolver()
    controller = IntentController()

    active = await resolver.resolve("这篇论文的方法是什么？", context)
    assert active.resolved_paper_ids == ["b1"]

    first = await resolver.resolve("第1篇的方法是什么？", context)
    assert first.resolved_paper_ids == ["b1"]

    pair = await resolver.resolve("比较第1篇和第2篇", context)
    assert pair.resolved_paper_ids == ["b1", "b2"]

    qa = await controller.decide("第1篇的方法是什么？", agent_mode="rule", session_context=context)
    assert (qa.intent, qa.conversation_operation, qa.resolved_paper_ids) == (
        "paper_qa", "paper_qa", ["b1"]
    )
    compare = await controller.decide(
        "比较第1篇和第2篇", agent_mode="rule", session_context=context
    )
    assert (compare.intent, compare.resolved_paper_ids) == ("paper_compare", ["b1", "b2"])


@pytest.mark.asyncio
async def test_recommend_more_keeps_global_ordinals_across_batches():
    """续接推荐沿用会话累计编号时，“第 N 篇”仍按全局编号解析。

    会话先有 a1/a2，再“再推荐”新增 b1/b2（显示编号 3、4）。
    “第3篇”应指 b1（累计序号），而不是批次内第 3 个位置（不存在）。
    """
    context = session_store.create(ttl_minutes=30)
    context = _run_like_api(context, "RAG", [_paper("a1"), _paper("a2")], "paper_recommendation")
    context = _run_like_api(context, "再推荐", [_paper("b1"), _paper("b2")], "recommend_more")

    assert context.last_recommendation_batch_start == 3
    assert [item["source_id"] for item in context.last_recommendation_batch] == ["b1", "b2"]

    from app.agents.reference_resolver import ReferenceResolver
    resolver = ReferenceResolver()
    third = await resolver.resolve("第3篇的方法是什么？", context)
    assert third.resolved_paper_ids == ["b1"]
    fourth = await resolver.resolve("第4篇的局限是什么？", context)
    assert fourth.resolved_paper_ids == ["b2"]
    # 会话累计不足 5 篇时，越界序号仍触发澄清而不是落到旧批次。
    fifth = await resolver.resolve("第5篇的局限是什么？", context)
    assert fifth.missing_ordinal == 5
