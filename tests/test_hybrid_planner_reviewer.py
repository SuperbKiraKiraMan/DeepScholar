import asyncio
import json

import pytest

from app.agents.planner import PlannerAgent
from app.agents.reviewer import ReviewerAgent
from app.agents.schemas import (
    AgentError,
    AgentToolCall,
    AgentTaskStatus,
    ExecutionBudget,
    ExecutionClass,
    ExecutionSpec,
    ReviewOutcome,
    ReviewVerdict,
    SafetyPolicy,
    WorkerResult,
    WorkItem,
    WorkPlan,
)


class QueueLLM:
    """按 schema 返回可注入结果，并保留不含密钥的调用记录。"""

    def __init__(self, responses=None, delay=0):
        self.responses = list(responses or [])
        self.delay = delay
        self.calls = []

    async def generate_structured(self, system_prompt, user_prompt, output_schema, **kwargs):
        self.calls.append({
            "schema": output_schema.__name__,
            "system_prompt": system_prompt,
            "payload": json.loads(user_prompt),
        })
        if self.delay:
            await asyncio.sleep(self.delay)
        raw = self.responses.pop(0)
        if raw.get("_fail"):
            return {"success": False, "error": "fake failure", "latency_ms": 1}
        return {
            "success": True,
            "data": output_schema.model_validate(raw),
            "latency_ms": 1,
            "model": "fake-hybrid",
            "usage": {"total_tokens": 9},
        }


class PassingEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, **kwargs):
        self.calls += 1
        return {
            "metrics": {
                "no_fake_citation": True,
                "citation_id_exists": True,
                "source_url_valid": True,
                "evidence_available": True,
            },
            "metrics_detail": {},
            "feedback": [],
        }


def make_spec(execution_class=ExecutionClass.RESEARCH, mode="llm"):
    route = {
        ExecutionClass.ATOMIC: "direct_tool",
        ExecutionClass.CONTEXTUAL: "conversation",
        ExecutionClass.RESEARCH: "full_research",
    }[execution_class]
    return ExecutionSpec(
        request_id="hybrid-test",
        user_request="研究 RAG 的方法、证据与局限",
        intent="deep_research" if execution_class == ExecutionClass.RESEARCH else "lookup",
        execution_class=execution_class,
        execution_route=route,
        research_topic="RAG evaluation",
        # 遗留协议字段：direct 的工具选型已下沉到 Planner，这里恒为空（保持 API/SSE 兼容）。
        selected_tools=[],
        selected_tool_args={},
        budget=ExecutionBudget(total_timeout_ms=5_000, max_tool_calls=20),
        metadata={"agent_mode": mode},
    )


def candidate_for(plan, *, changes=None):
    changes = changes or {}
    return {
        "items": [
            {
                "task_id": item.task_id,
                "profile": item.profile.value,
                "instruction": changes.get(item.task_id, item.instruction),
                "depends_on": list(item.depends_on),
                "allowed_tools": list(item.allowed_tools),
                "strategy": item.strategy.value,
            }
            for item in plan.items
        ],
    }


def successful_results(plan):
    results = []
    for item in plan.items:
        kwargs = {}
        if item.profile.value == "cite":
            kwargs = {
                "tool_calls": [AgentToolCall(tool_name="citation_check", success=True)],
                "output_data": {
                    "citation_check_results": [],
                    "citation_summary": {
                        "total_checked": 0, "valid_count": 0,
                        "invalid_count": 0, "all_valid": True,
                    },
                },
            }
        results.append(WorkerResult(
            task_id=item.task_id,
            profile=item.profile,
            status=AgentTaskStatus.SUCCESS,
            output_data=kwargs.pop("output_data", {"ok": True}),
            **kwargs,
        ))
    return results


@pytest.mark.asyncio
async def test_rule_and_non_research_modes_never_call_planner_llm():
    fake = QueueLLM([])
    planner = PlannerAgent(fake)
    rule_plan = await planner.plan_hybrid(make_spec(mode="rule"))
    atomic = await planner.plan_hybrid(make_spec(ExecutionClass.ATOMIC, "llm"))
    contextual = await planner.plan_hybrid(make_spec(ExecutionClass.CONTEXTUAL, "llm"))
    assert not fake.calls
    assert len(rule_plan.items) == 7
    assert len(atomic.items) == 1
    assert [item.task_id for item in contextual.items] == ["context_load", "answer"]


@pytest.mark.asyncio
async def test_research_llm_accepts_valid_unified_work_plan():
    baseline = PlannerAgent().plan(make_spec())
    raw = candidate_for(baseline, changes={"search_1": "RAG evaluation systematic survey"})
    fake = QueueLLM([raw])
    plan = await PlannerAgent(fake).plan_hybrid(make_spec())
    assert plan.metadata["hybrid_planner"]["success"] is True
    assert plan.items[0].instruction == "RAG evaluation systematic survey"
    payload = fake.calls[0]["payload"]["execution_spec"]
    assert payload["budget"]["max_tool_calls"] == 20
    assert "safety_policy" in payload and "allowed_tools" in payload


@pytest.mark.asyncio
async def test_llm_plan_rejects_single_search_that_weakens_runtime_barrier():
    spec = make_spec()
    baseline = PlannerAgent().plan(spec)
    raw = candidate_for(baseline)
    raw["items"] = [item for item in raw["items"] if item["task_id"] != "search_3"]
    raw["items"][-4]["depends_on"] = ["search_1", "search_2"]
    plan = await PlannerAgent(QueueLLM([raw])).plan_hybrid(spec)
    assert plan.metadata["hybrid_planner"]["status"] == "candidate_rejected"
    assert len([item for item in plan.items if item.profile.value == "search"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["cycle", "unauthorized", "missing"])
async def test_invalid_llm_plan_falls_back_to_rule(mutation):
    spec = make_spec()
    baseline = PlannerAgent().plan(spec)
    raw = candidate_for(baseline)
    if mutation == "cycle":
        raw["items"][0]["depends_on"] = ["write"]
    elif mutation == "unauthorized":
        raw["items"][0]["allowed_tools"] = ["evidence_extract"]
    else:
        raw["items"].pop()
    plan = await PlannerAgent(QueueLLM([raw])).plan_hybrid(spec)
    assert plan.metadata["hybrid_planner"]["status"] == "candidate_rejected"
    assert [item.task_id for item in plan.items] == [item.task_id for item in baseline.items]


@pytest.mark.asyncio
async def test_planner_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("HYBRID_PLANNER_TIMEOUT_SECONDS", "0.001")
    plan = await PlannerAgent(QueueLLM([{}], delay=0.05)).plan_hybrid(make_spec())
    assert plan.metadata["hybrid_planner"]["status"] == "timeout"


@pytest.mark.asyncio
async def test_llm_candidate_cannot_escalate_budget_or_safety():
    budget_spec = make_spec()
    budget_spec.budget.max_tool_calls = 1
    budget_baseline = PlannerAgent().plan(budget_spec)
    budget_plan = await PlannerAgent(QueueLLM([candidate_for(budget_baseline)])).plan_hybrid(
        budget_spec,
    )
    assert budget_plan.metadata["hybrid_planner"]["status"] == "candidate_rejected"

    safety_spec = make_spec()
    safety_spec.safety_policy = SafetyPolicy(allow_network=False)
    safety_baseline = PlannerAgent().plan(safety_spec)
    safety_plan = await PlannerAgent(QueueLLM([candidate_for(safety_baseline)])).plan_hybrid(
        safety_spec,
    )
    assert safety_plan.metadata["hybrid_planner"]["status"] == "candidate_rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["new_search", "partial_barrier", "shrink_tools"])
async def test_llm_candidate_cannot_bypass_stable_plan_barriers(mutation):
    spec = make_spec()
    baseline = PlannerAgent().plan(spec)
    raw = candidate_for(baseline)
    if mutation == "new_search":
        raw["items"][0]["task_id"] = "search_4"
        raw["items"][3]["depends_on"][0] = "search_4"
    elif mutation == "partial_barrier":
        raw["items"][3]["depends_on"] = ["search_1"]
    else:
        raw["items"][0]["allowed_tools"] = [raw["items"][0]["allowed_tools"][0]]
    plan = await PlannerAgent(QueueLLM([raw])).plan_hybrid(spec)
    assert plan.metadata["hybrid_planner"]["status"] == "candidate_rejected"


@pytest.mark.asyncio
async def test_replan_only_exposes_affected_subgraph_and_is_bounded():
    spec = make_spec()
    original = PlannerAgent().plan(spec)
    verdict = ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["analyze"],
        feedback=[{"task_id": "analyze", "message": "change extraction strategy"}],
    )
    deterministic = PlannerAgent().replan(original, verdict)
    affected = [item for item in deterministic.items if item.metadata.get("replanned")]
    raw = candidate_for(
        deterministic.model_copy(update={"items": affected}),
        changes={"analyze": "使用替代抽取策略恢复证据"},
    )
    fake = QueueLLM([raw])
    revised = await PlannerAgent(fake).replan_hybrid(original, verdict)
    assert fake.calls[0]["payload"]["affected_task_ids"] == ["analyze", "cite", "write"]
    assert fake.calls[0]["payload"]["reviewer_structured_feedback"][0]["task_id"] == "analyze"
    assert revised.items[0].instruction == original.items[0].instruction
    assert revised.replan_count == 1
    with pytest.raises(ValueError, match="次数已用尽"):
        await PlannerAgent(fake).replan_hybrid(revised, verdict)


@pytest.mark.asyncio
async def test_replan_rejects_llm_that_restores_failed_search_query():
    spec = make_spec()
    original = PlannerAgent().plan(spec)
    verdict = ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["search_1"],
        feedback=[{"task_id": "search_1", "message": "query failed"}],
    )
    deterministic = PlannerAgent().replan(original, verdict)
    affected = [item for item in deterministic.items if item.metadata.get("replanned")]
    raw = candidate_for(deterministic.model_copy(update={"items": affected}))
    next(item for item in raw["items"] if item["task_id"] == "search_1")["instruction"] = (
        original.items[0].instruction
    )
    revised = await PlannerAgent(QueueLLM([raw])).replan_hybrid(original, verdict)
    search = next(item for item in revised.items if item.task_id == "search_1")
    assert revised.metadata["hybrid_planner"]["status"] == "candidate_rejected"
    assert "alternative independent sources" in search.instruction
    assert search.metadata["strategy_changed"] == "query_expanded"


@pytest.mark.asyncio
async def test_reviewer_hard_failure_cannot_be_relaxed_by_llm():
    plan = PlannerAgent().plan(make_spec())
    results = successful_results(plan)
    results[0] = WorkerResult(
        task_id=results[0].task_id,
        profile=results[0].profile,
        status=AgentTaskStatus.FAILED,
        needs_replan=True,
        error=AgentError(error_code="TOOL_FAILED", message="failed", recoverable=True),
    )
    fake = QueueLLM([{"outcome": "pass"}])
    verdict, info = await ReviewerAgent(PassingEvaluator(), fake).review_hybrid(
        plan, results, evaluator_input={}, final_output={"final_report": "draft"},
        run_eval=True, agent_mode="llm",
    )
    assert verdict.outcome == ReviewOutcome.REPLAN
    assert info["status"] == "hard_gate_blocked"
    assert not fake.calls


@pytest.mark.asyncio
async def test_semantic_reviewer_can_tighten_rule_pass_without_writing_body():
    plan = PlannerAgent().plan(make_spec())
    fake = QueueLLM([{
        "outcome": "repair",
        "failed_task_ids": ["write"],
        "repair_scope": ["write"],
        "feedback": [{"task_id": "write", "code": "TOPIC_GAP", "message": "缺少局限讨论"}],
        "summary": "需要补充主题覆盖",
    }])
    verdict, info = await ReviewerAgent(PassingEvaluator(), fake).review_hybrid(
        plan, successful_results(plan), evaluator_input={},
        final_output={"final_report": "已有正文"}, run_eval=True, agent_mode="llm",
    )
    assert verdict.outcome == ReviewOutcome.REPAIR
    assert verdict.final_output["final_report"] == "已有正文"
    assert info["success"] is True


@pytest.mark.asyncio
async def test_semantic_search_repair_is_converted_to_replan_not_analysis_repair():
    from app.graph import runtime

    plan = PlannerAgent().plan(make_spec())
    fake = QueueLLM([{
        "outcome": "repair",
        "failed_task_ids": ["search_1"],
        "repair_scope": ["search_1"],
        "feedback": [{
            "task_id": "search_1", "code": "COVERAGE_GAP",
            "message": "需要新的检索角度",
        }],
    }])
    verdict, _ = await ReviewerAgent(PassingEvaluator(), fake).review_hybrid(
        plan, successful_results(plan), evaluator_input={},
        final_output={"final_report": "已有正文"}, run_eval=True, agent_mode="llm",
    )
    assert verdict.outcome == ReviewOutcome.REPLAN
    assert verdict.repair_scope == []
    assert runtime.route_after_four_agent_reviewer({
        "review_verdict": verdict.model_dump(mode="json"),
    }) == "planner"


@pytest.mark.asyncio
async def test_begin_repair_injects_write_feedback_into_real_worker_payload():
    from app.graph import runtime

    plan = PlannerAgent().plan(make_spec(mode="rule"))
    verdict = ReviewVerdict(
        outcome=ReviewOutcome.REPAIR,
        failed_task_ids=["write"],
        repair_scope=["write"],
        feedback=[{
            "task_id": "write", "code": "TOPIC_GAP",
            "message": "补充研究局限并删除重复表述",
        }],
    )
    delta = await runtime.node_begin_repair({
        "work_plan": plan.model_dump(mode="json"),
        "review_verdict": verdict.model_dump(mode="json"),
        "repair_count": 0,
        "execution_round": 0,
    })
    revised = WorkPlan.model_validate(delta["work_plan"])
    write = next(item for item in revised.items if item.task_id == "write")
    assert write.input_data["reviewer_feedback"][0]["code"] == "TOPIC_GAP"
    assert "补充研究局限" in write.instruction
    sends = runtime.send_chapter_work_items({
        "work_plan": delta["work_plan"],
        "outline": {"sections": [{
            "heading": "局限", "assigned_evidence_ids": [], "assigned_source_ids": [],
        }]},
        "sources": [], "evidence_cards": [], "execution_round": 1,
        "language": "zh",
    })
    payload = WorkItem.model_validate(sends[0].arg["work_item"])
    assert payload.input_data["reviewer_feedback"][0]["message"] == "补充研究局限并删除重复表述"
    assert "补充研究局限" in payload.instruction


@pytest.mark.asyncio
async def test_invalid_or_timed_out_semantic_review_falls_back_to_pass(monkeypatch):
    plan = PlannerAgent().plan(make_spec())
    results = successful_results(plan)
    invalid = QueueLLM([{"outcome": "repair", "failed_task_ids": ["unknown"]}])
    verdict, info = await ReviewerAgent(PassingEvaluator(), invalid).review_hybrid(
        plan, results, evaluator_input={}, final_output={"final_report": "正文"},
        run_eval=True, agent_mode="llm",
    )
    assert verdict.outcome == ReviewOutcome.PASS
    assert info["status"] == "candidate_rejected"
    monkeypatch.setenv("HYBRID_REVIEWER_TIMEOUT_SECONDS", "0.001")
    slow = QueueLLM([{"outcome": "fail"}], delay=0.05)
    verdict, info = await ReviewerAgent(PassingEvaluator(), slow).review_hybrid(
        plan, results, evaluator_input={}, final_output={"final_report": "正文"},
        run_eval=True, agent_mode="llm",
    )
    assert verdict.outcome == ReviewOutcome.PASS
    assert info["status"] == "timeout"


@pytest.mark.asyncio
async def test_run_eval_false_disables_evaluator_and_semantic_llm():
    plan = PlannerAgent().plan(make_spec())
    evaluator = PassingEvaluator()
    fake = QueueLLM([])
    verdict, info = await ReviewerAgent(evaluator, fake).review_hybrid(
        plan, successful_results(plan), evaluator_input={},
        final_output={"final_report": "正文"}, run_eval=False, agent_mode="llm",
    )
    assert verdict.outcome == ReviewOutcome.PASS
    assert evaluator.calls == 0 and not fake.calls
    assert info["status"] == "not_applicable"


@pytest.mark.asyncio
async def test_production_nodes_use_hybrid_agents_and_trace_never_contains_cot(monkeypatch):
    from app.graph import runtime

    spec = make_spec()
    baseline = PlannerAgent().plan(spec)
    planner_fake = QueueLLM([candidate_for(baseline)])
    monkeypatch.setattr(runtime, "_planner_agent", PlannerAgent(planner_fake))
    planned = await runtime.node_four_agent_planner({
        "execution_spec": spec.model_dump(mode="json"),
        "execution_round": 0,
        "runtime_budget_id": "hybrid-runtime",
        "max_sources": 3,
        "agent_mode": "llm",
    })
    assert planner_fake.calls[0]["schema"] == "LLMWorkPlanCandidate"
    assert any(event.get("purpose") == "hybrid_planning" for event in planned["trace"])

    plan = PlannerAgent().plan(spec)
    reviewer_fake = QueueLLM([{"outcome": "pass", "summary": "semantic pass"}])
    monkeypatch.setattr(
        runtime, "_reviewer_agent", ReviewerAgent(PassingEvaluator(), reviewer_fake),
    )
    reviewed = await runtime.node_four_agent_reviewer({
        "work_plan": plan.model_dump(mode="json"),
        "worker_results": {
            result.task_id: result.model_dump(mode="json")
            for result in successful_results(plan)
        },
        "run_eval": True,
        "agent_mode": "llm",
        "topic": spec.research_topic,
        "final_report": "已有正文",
        "sources": [],
        "evidence_cards": [],
        "citation_check_results": [],
        "citation_summary": {},
        "trace": [],
    })
    assert reviewer_fake.calls[0]["schema"] == "LLMSemanticReviewOutput"
    serialized_trace = json.dumps(reviewed["trace"], ensure_ascii=False).lower()
    assert "chain_of_thought" not in serialized_trace
    assert "reasoning" not in serialized_trace


def test_semantic_write_repair_is_delegated_to_writing_work_items(monkeypatch):
    from app.graph import runtime

    monkeypatch.setattr(runtime, "send_chapter_work_items", lambda state: ["writing-send"])
    monkeypatch.setattr(runtime, "send_analysis_work_item", lambda state: ["analysis-send"])
    state = {"review_verdict": {"outcome": "repair", "repair_scope": ["write"]}}
    assert runtime.route_repair_scope(state) == ["writing-send"]


def test_outline_repair_cannot_rewrite_or_remove_validated_sections():
    from app.agents.report_outline import ReportOutlineGenerator

    original = {"sections": [{
        "heading": "方法", "guiding_question": "有哪些方法？",
        "assigned_evidence_ids": ["S1:e1"], "assigned_source_ids": ["S1"],
    }]}
    additive = {"sections": [
        dict(original["sections"][0]),
        {
            "heading": "局限", "guiding_question": "有哪些局限？",
            "assigned_evidence_ids": ["S2:e1"], "assigned_source_ids": ["S2"],
        },
    ]}
    rewritten = {"sections": [
        {**original["sections"][0], "assigned_evidence_ids": ["S2:e1"]},
        additive["sections"][1],
    ]}
    assert ReportOutlineGenerator._repair_preserves_existing(original, additive) == ""
    assert "rewrote or removed" in ReportOutlineGenerator._repair_preserves_existing(
        original, rewritten,
    )


@pytest.mark.asyncio
async def test_outline_second_llm_call_cannot_rewrite_existing_section():
    from app.agents.report_outline import ReportOutlineGenerator
    from app.llm.client import FakeLLMClient, reset_llm_client
    import app.llm.client as client_mod

    client_mod._global_client = FakeLLMClient(responses=[
        {
            "sections": [
                {
                    "heading": "方法", "guiding_question": "有哪些主要方法？",
                    "assigned_evidence_ids": ["E1"],
                },
                {
                    "heading": "局限", "guiding_question": "有哪些研究局限？",
                    "assigned_evidence_ids": ["E2"],
                },
            ],
        },
        {
            "sections": [
                {
                    "heading": "方法", "guiding_question": "改写后的问题是什么？",
                    "assigned_evidence_ids": ["E2"],
                },
                {
                    "heading": "局限", "guiding_question": "有哪些研究局限？",
                    "assigned_evidence_ids": ["E2"],
                },
                {
                    "heading": "数据集", "guiding_question": "使用哪些数据集？",
                    "assigned_evidence_ids": ["E1"],
                },
            ],
        },
    ])
    cards = [
        {"evidence_id": "S1:e1", "source_id": "S1", "claim": "method", "method": "A"},
        {"evidence_id": "S2:e1", "source_id": "S2", "claim": "limit", "limitation": "B"},
    ]
    try:
        outline, info = await ReportOutlineGenerator().generate(
            "调研 RAG 的主要方法、数据集与研究局限",
            [{"source_id": "S1"}, {"source_id": "S2"}], cards,
            [{"citation_id": 1, "is_valid": True}, {"citation_id": 2, "is_valid": True}],
            agent_mode="llm", allow_rule_fallback=True,
        )
        assert len(client_mod._global_client.calls) == 2
        assert info["success"] is False
        assert "rewrote or removed" in info["repair"]["error"]
        assert outline["sections"]  # 返回规则 fallback，而不是接受恶意 repair。
    finally:
        reset_llm_client()


@pytest.mark.asyncio
async def test_outline_repair_and_hybrid_fallback_events_are_safe(monkeypatch):
    from app.agents.report_outline import ReportOutlineGenerator
    from app.graph import runtime

    async def fake_generate(self, **kwargs):
        return {"sections": []}, {
            "success": False,
            "latency_ms": 3,
            "error": "provider secret body",
            "repair": {
                "attempted": True, "success": False, "latency_ms": 2,
                "error": "provider secret repair body", "model": "fake",
            },
        }

    monkeypatch.setattr(ReportOutlineGenerator, "generate", fake_generate)
    delta = await runtime.node_outline({"agent_mode": "llm"})
    llm_events = [item for item in delta["trace"] if item["event"] == "llm_failed"]
    assert {item["purpose"] for item in llm_events} == {
        "outline_generation", "outline_repair",
    }
    for event in llm_events:
        assert event["success"] is False
        assert event["recoverable"] is True
        assert event["error_code"]
        assert "error" not in event
        sse = runtime._trace_to_sse_payload(event)
        assert "secret" not in json.dumps(sse).lower()


@pytest.mark.asyncio
async def test_planner_and_reviewer_fallback_events_use_sanitized_error_codes(monkeypatch):
    from app.graph import runtime

    spec = make_spec()
    planner_fake = QueueLLM([{"_fail": True}])
    monkeypatch.setattr(runtime, "_planner_agent", PlannerAgent(planner_fake))
    planned = await runtime.node_four_agent_planner({
        "execution_spec": spec.model_dump(mode="json"),
        "execution_round": 0, "runtime_budget_id": "safe-events",
        "max_sources": 3, "agent_mode": "llm",
    })
    planner_event = next(item for item in planned["trace"] if item["event"] == "llm_failed")
    assert planner_event["success"] is False
    assert planner_event["recoverable"] is True
    assert planner_event["error_code"] == "HYBRID_PLANNER_GENERATION_FAILED"
    assert "error" not in planner_event
    assert "fake failure" not in json.dumps(runtime._trace_to_sse_payload(planner_event)).lower()

    plan = PlannerAgent().plan(spec)
    reviewer_fake = QueueLLM([{"_fail": True}])
    monkeypatch.setattr(
        runtime, "_reviewer_agent", ReviewerAgent(PassingEvaluator(), reviewer_fake),
    )
    reviewed = await runtime.node_four_agent_reviewer({
        "work_plan": plan.model_dump(mode="json"),
        "worker_results": {
            result.task_id: result.model_dump(mode="json")
            for result in successful_results(plan)
        },
        "run_eval": True, "agent_mode": "llm", "topic": spec.research_topic,
        "final_report": "正文", "sources": [], "evidence_cards": [],
        "citation_check_results": [], "citation_summary": {}, "trace": [],
    })
    reviewer_event = next(item for item in reviewed["trace"] if item["event"] == "llm_failed")
    assert reviewer_event["success"] is False
    assert reviewer_event["recoverable"] is True
    assert reviewer_event["error_code"] == "HYBRID_REVIEWER_GENERATION_FAILED"
    assert "error" not in reviewer_event
    assert "fake failure" not in json.dumps(runtime._trace_to_sse_payload(reviewer_event)).lower()
