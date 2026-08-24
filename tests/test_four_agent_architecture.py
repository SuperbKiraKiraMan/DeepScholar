"""四 Agent 协议、隔离与有界反馈测试。"""

import asyncio
import re
import uuid

import pytest

from app.agents import ControllerAgent, PlannerAgent, ReviewerAgent, WorkerAgent
from app.llm.client import FakeLLMClient
from app.agents.schemas import (
    AgentError,
    AgentRole,
    AgentTaskStatus,
    AgentToolCall,
    ExecutionBudget,
    ExecutionClass,
    ExecutionSpec,
    ReviewOutcome,
    ReviewVerdict,
    SafetyPolicy,
    WorkerProfile,
    WorkerResult,
    WorkerStrategy,
    WorkItem,
    WorkPlan,
)
from app.graph import runtime
from app.observability.lifecycle import clear_runtime_budget, register_runtime_budget
from app.services.session_store import SessionContext
from app.tools.base import ToolResult


def _spec(execution_class: ExecutionClass, route: str) -> ExecutionSpec:
    return ExecutionSpec(
        request_id=execution_class.value,
        user_request="测试请求",
        intent="test",
        execution_class=execution_class,
        execution_route=route,
        research_topic="RAG evaluation",
    )


def test_four_agents_are_independently_instantiable():
    agents = [ControllerAgent(), PlannerAgent(), WorkerAgent(), ReviewerAgent()]
    assert [agent.role for agent in agents] == [
        AgentRole.CONTROLLER.value,
        AgentRole.PLANNER.value,
        AgentRole.WORKER.value,
        AgentRole.REVIEWER.value,
    ]
    assert ReviewerAgent.allowed_tools == ()


def test_all_execution_classes_are_planned_through_protocol():
    planner = PlannerAgent()
    atomic = planner.plan(_spec(ExecutionClass.ATOMIC, "direct_tool"))
    contextual = planner.plan(_spec(ExecutionClass.CONTEXTUAL, "conversation"))
    research = planner.plan(_spec(ExecutionClass.RESEARCH, "full_research"))

    assert [item.profile for item in atomic.items] == [WorkerProfile.DIRECT]
    assert [item.task_id for item in contextual.items] == ["context_load", "answer"]
    assert contextual.items[1].depends_on == ["context_load"]
    assert len([item for item in research.items if item.profile == WorkerProfile.SEARCH]) == 3
    assert research.items[-1].profile == WorkerProfile.WRITE


def test_profile_permissions_and_citation_strategy_are_bounded():
    worker = WorkerAgent()
    forbidden = WorkItem(
        task_id="read", profile=WorkerProfile.READ, instruction="read",
        allowed_tools=["citation_check"],
    )
    with pytest.raises(ValueError):
        worker.allowed_tools(forbidden)
    with pytest.raises(ValueError):
        WorkItem(
            task_id="cite", profile=WorkerProfile.CITE, instruction="cite",
            strategy=WorkerStrategy.REACT,
        )


def test_search_profile_maps_registered_namespaced_mcp_tools():
    """动态 MCP 公共名应按 canonical capability 授权，而不是整单拒绝。"""
    from app.tools.base import BaseTool
    from app.tools.registry import ToolCapabilityMetadata, ToolRegistry

    class NamespacedSearchTool(BaseTool):
        task_types = ("search",)

        @property
        def name(self):
            return "mcp__academic-research__semantic_scholar_search"

        @property
        def description(self):
            return "namespaced MCP search"

        async def _arun(self, **_kwargs):
            return ToolResult(True, data={"sources": []})

    class SpoofedNonSearchTool(NamespacedSearchTool):
        task_types = ("write",)

        @property
        def name(self):
            return "mcp__untrusted__semantic_scholar_graph"

    registry = ToolRegistry()
    mcp_tool = NamespacedSearchTool()
    spoofed = SpoofedNonSearchTool()
    registry.register(mcp_tool, ToolCapabilityMetadata(network_access=True))
    registry.register(spoofed, ToolCapabilityMetadata(network_access=True))

    requested = ["academic_search", mcp_tool.name]
    item = WorkItem(
        task_id="search_1", profile=WorkerProfile.SEARCH,
        instruction="search", allowed_tools=requested,
    )
    assert registry.list_retrieval_capabilities() == [
        "academic_search", "local_paper_search", "semantic_scholar_search",
        "semantic_scholar_recommendations", "semantic_scholar_graph",
        mcp_tool.name,
    ]
    assert WorkerAgent(registry).allowed_tools(item) == requested

    with pytest.raises(ValueError, match="无权调用"):
        WorkerAgent(registry).allowed_tools(WorkItem(
            task_id="search_2", profile=WorkerProfile.SEARCH,
            instruction="search", allowed_tools=[spoofed.name],
        ))


@pytest.mark.asyncio
async def test_worker_instances_isolate_messages_budget_and_dedup_state():
    first = WorkerAgent()
    second = WorkerAgent()
    item = WorkItem(
        task_id="answer", profile=WorkerProfile.ANSWER,
        instruction="简短回答", strategy=WorkerStrategy.SYNTHESIS,
    )
    result = await first.execute(item)
    first.messages.append({"role": "user", "content": "private"})
    first._called.add("private")
    assert result.status == AgentTaskStatus.SUCCESS
    assert second.messages == []
    assert second._called == set()


def test_repair_and_replan_are_each_limited_to_one():
    planner = PlannerAgent()
    plan = planner.plan(_spec(ExecutionClass.ATOMIC, "direct_tool"))
    failed = WorkerResult(
        task_id="direct_1", profile=WorkerProfile.DIRECT,
        status=AgentTaskStatus.FAILED, needs_replan=True,
        error=AgentError(error_code="X", message="失败", recoverable=True),
    )
    verdict = ReviewerAgent().review(plan, [failed])
    assert verdict.outcome == ReviewOutcome.REPLAN
    replanned = planner.replan(plan, verdict)
    assert replanned.replan_count == 1
    with pytest.raises(ValueError):
        planner.replan(replanned, verdict)

    successful = WorkerResult(
        task_id="direct_1", profile=WorkerProfile.DIRECT,
        status=AgentTaskStatus.SUCCESS,
    )
    hard_failure = {"eval_metrics": {"evidence_available": False}}
    repair = ReviewerAgent().review(plan, [successful], final_output=hard_failure)
    assert repair.outcome == ReviewOutcome.REPAIR
    plan.repair_count = 1
    assert ReviewerAgent().review(
        plan, [successful], final_output=hard_failure,
    ).outcome == ReviewOutcome.FAIL


def test_legacy_execution_route_is_preserved():
    spec = _spec(ExecutionClass.RESEARCH, "full_research")
    assert spec.execution_route == "full_research"


def test_work_plan_rejects_cycles_and_planner_propagates_limits():
    spec = _spec(ExecutionClass.RESEARCH, "full_research")
    spec.budget = ExecutionBudget(
        max_workers=1, max_tool_calls=2, max_iterations=3,
        per_tool_timeout_ms=123, total_timeout_ms=456,
    )
    spec.safety_policy = {"allow_network": False}
    plan = PlannerAgent().plan(spec)
    # max_workers 是调度并发上限，不能改变 Planner 的任务集合。
    assert len([item for item in plan.items if item.profile == WorkerProfile.SEARCH]) == 3
    assert all(item.max_tool_calls == 2 for item in plan.items)
    assert all(item.safety_policy.allow_network is False for item in plan.items)

    with pytest.raises(ValueError, match="环"):
        WorkPlan(
            execution_spec=spec,
            items=[
                WorkItem(
                    task_id="a", profile=WorkerProfile.ANSWER,
                    instruction="a", depends_on=["b"],
                ),
                WorkItem(
                    task_id="b", profile=WorkerProfile.ANSWER,
                    instruction="b", depends_on=["a"],
                ),
            ],
        )


def test_reviewer_rejects_nonterminal_and_success_needs_replan():
    plan = PlannerAgent().plan(_spec(ExecutionClass.ATOMIC, "direct_tool"))
    pending = WorkerResult(
        task_id="direct_1", profile=WorkerProfile.DIRECT,
        status=AgentTaskStatus.PENDING,
    )
    assert ReviewerAgent().review(plan, [pending]).outcome == ReviewOutcome.FAIL

    invalid_success = WorkerResult(
        task_id="direct_1", profile=WorkerProfile.DIRECT,
        status=AgentTaskStatus.SUCCESS, needs_replan=True,
    )
    assert ReviewerAgent().review(plan, [invalid_success]).outcome == ReviewOutcome.REPLAN
    plan.replan_count = 1
    assert ReviewerAgent().review(plan, [invalid_success]).outcome == ReviewOutcome.FAIL


def test_replan_consumes_feedback_and_only_revises_affected_subgraph():
    planner = PlannerAgent()
    plan = planner.plan(_spec(ExecutionClass.RESEARCH, "full_research"))
    verdict = ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["analyze"],
        feedback=[{
                "task_id": "analyze", "code": "TOOL_UNAVAILABLE",
                "message": "证据工具持续失败",
                "reusable_dependency_ids": ["read"],
        }],
    )
    revised = planner.replan(plan, verdict)
    assert revised.revision == revised.replan_count == 1
    assert revised.reviewer_feedback == ["证据工具持续失败"]
    assert {item.task_id for item in revised.items} >= {"read", "write"}
    assert not any(item.profile == WorkerProfile.ANALYZE for item in revised.items)
    writing = next(item for item in revised.items if item.profile == WorkerProfile.WRITE)
    assert writing.depends_on == ["read"]
    assert writing.metadata["degraded_source_only"] is True


def test_merge_reads_only_current_round_and_revision():
    def entry(round_id: int, revision: int, status: AgentTaskStatus):
        result = WorkerResult(
            task_id="analyze", profile=WorkerProfile.ANALYZE,
            status=status,
            output_data={"evidence_cards": [{"round": round_id}]},
            error=(AgentError(error_code="X", message="旧轮失败")
                   if status == AgentTaskStatus.FAILED else None),
        )
        return {
            "round_id": round_id, "revision": revision,
            "profile": "analyze", "task_id": "analyze",
            "result": result.model_dump(mode="json"),
        }

    state = {
        "execution_round": 1,
        "work_plan": {"revision": 1},
        "_worker_result_bucket": [
            entry(0, 0, AgentTaskStatus.FAILED),
            entry(1, 1, AgentTaskStatus.SUCCESS),
            entry(2, 1, AgentTaskStatus.FAILED),
        ],
    }
    current = runtime._current_worker_results(state, WorkerProfile.ANALYZE)
    assert len(current) == 1
    assert current[0].status == AgentTaskStatus.SUCCESS
    assert current[0].output_data["evidence_cards"] == [{"round": 1}]


@pytest.mark.asyncio
async def test_worker_enforces_safety_and_detects_hidden_tool_calls(monkeypatch):
    worker = WorkerAgent()
    blocked = WorkItem(
        task_id="search", profile=WorkerProfile.SEARCH, instruction="search",
        allowed_tools=["academic_search"],
        safety_policy={"allow_business_tools": False},
    )
    blocked_result = await worker.execute(blocked)
    assert blocked_result.status == AgentTaskStatus.FAILED
    assert "安全策略" in blocked_result.error.message

    async def hidden_call(_self, _item, _strategy, _allowed):
        return {}, [AgentToolCall(tool_name="citation_check", success=True)], [], 1

    monkeypatch.setattr(WorkerAgent, "_execute_bounded", hidden_call)
    hidden = await WorkerAgent().execute(WorkItem(
        task_id="answer", profile=WorkerProfile.ANSWER, instruction="answer",
    ))
    assert hidden.status == AgentTaskStatus.FAILED
    assert "未授权隐藏工具" in hidden.error.message


@pytest.mark.asyncio
async def test_answer_and_write_workers_produce_real_text():
    answer = await WorkerAgent().execute(WorkItem(
        task_id="answer", profile=WorkerProfile.ANSWER,
        instruction="请简要说明 RAG。", strategy=WorkerStrategy.SYNTHESIS,
        input_data={"language": "zh", "agent_mode": "rule"},
    ))
    source = {
        "source_id": "s1", "title": "RAG Evaluation", "year": 2025,
        "authors": ["A"], "url": "https://example.org/rag",
    }
    writing = await WorkerAgent().execute(WorkItem(
        task_id="write", profile=WorkerProfile.WRITE,
        instruction="撰写 RAG 报告", strategy=WorkerStrategy.SYNTHESIS,
        resources=[source], input_data={"topic": "RAG", "language": "zh"},
    ))
    assert answer.status == AgentTaskStatus.SUCCESS
    assert answer.output_data["answer"].strip()
    assert writing.status == AgentTaskStatus.SUCCESS
    assert writing.output_data["report"].strip()


def test_fail_and_clarify_verdicts_control_final_status():
    base = {
        "topic": "status", "trace": [], "warnings": [], "sources": [],
        "evidence_cards": [], "citation_check_results": [],
        "worker_results": {}, "final_report": "", "answer": "",
        "execution_route": "conversation", "intent": "conversation",
    }
    failed = runtime._finalize_run(
        {**base, "run_id": f"fail-{uuid.uuid4().hex}",
         "review_verdict": {"outcome": "fail"}},
        "status", "graph_send", "rule",
    )
    clarify = runtime._finalize_run(
        {**base, "run_id": f"clarify-{uuid.uuid4().hex}",
         "review_verdict": {"outcome": "clarify"}},
        "status", "graph_send", "rule",
    )
    assert failed["status"] == "failed"
    assert clarify["status"] == "partial"


def test_reviewer_verdict_is_internal_sse_only():
    assert "reviewer_verdict" not in runtime.PUBLIC_SSE_EVENT_TYPES
    assert "reviewer_verdict" not in runtime._TRACE_TO_SSE_EVENT


@pytest.mark.asyncio
async def test_production_runtime_calls_all_four_agents(monkeypatch):
    calls = {"controller": 0, "planner": 0, "worker": 0, "reviewer": 0}
    original_controller = ControllerAgent.execute
    original_plan = PlannerAgent.plan
    original_worker = WorkerAgent.execute
    original_review = ReviewerAgent.review

    async def controller_spy(self, *args, **kwargs):
        calls["controller"] += 1
        return await original_controller(self, *args, **kwargs)

    def planner_spy(self, *args, **kwargs):
        calls["planner"] += 1
        return original_plan(self, *args, **kwargs)

    async def worker_spy(self, *args, **kwargs):
        calls["worker"] += 1
        return await original_worker(self, *args, **kwargs)

    def reviewer_spy(self, *args, **kwargs):
        calls["reviewer"] += 1
        return original_review(self, *args, **kwargs)

    monkeypatch.setattr(ControllerAgent, "execute", controller_spy)
    monkeypatch.setattr(PlannerAgent, "plan", planner_spy)
    monkeypatch.setattr(WorkerAgent, "execute", worker_spy)
    monkeypatch.setattr(ReviewerAgent, "review", reviewer_spy)

    result = await runtime.run_graph(
        "调研 RAG evaluation，比较主要方法并总结局限",
        max_sources=2, agent_mode="rule",
    )
    assert result["status"] in {"completed", "completed_with_warnings"}
    assert all(count > 0 for count in calls.values())


@pytest.mark.asyncio
async def test_llm_only_write_worker_uses_real_llm_chapter_path():
    """LLM-only 章节必须经过真实结构化 LLM 调用并记录真实模式。"""
    from app.llm.client import FakeLLMClient
    import app.llm.client as client_mod

    fake = FakeLLMClient([{
        "heading": "研究概述",
        "synthesis": "检索增强生成把外部知识检索与生成模型结合，从而增强事实依据并改善回答质量。[[e:e1]]",
        "evidence_ids": ["e1"],
        "findings": [],
    }])
    client_mod._global_client = fake
    source = {
        "source_id": "s1", "title": "Retrieval-Augmented Generation",
        "authors": ["A"], "year": 2024, "url": "https://example.org/s1",
    }
    result = await WorkerAgent().execute(WorkItem(
        task_id="write_1", profile=WorkerProfile.WRITE,
        instruction="撰写研究概述", strategy=WorkerStrategy.SYNTHESIS,
        resources=[source],
        input_data={
            "section": {
                "heading": "研究概述", "guiding_question": "RAG 的主要作用是什么？",
                "assigned_evidence_ids": ["e1"], "assigned_source_ids": ["s1"],
            },
            "evidence_cards": [{
                "evidence_id": "e1", "source_id": "s1",
                "claim": "RAG 可增强生成内容的事实依据。",
            }],
            "source_number": {"s1": 1}, "language": "zh",
            "agent_mode": "llm", "llm_only": True,
        },
    ))

    assert result.status == AgentTaskStatus.SUCCESS
    assert result.metadata["mode"] == "llm"
    assert result.output_data["mode"] == "llm"
    assert "[1]" in result.output_data["chapter"]
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_production_llm_only_research_uses_llm_write_workers(monkeypatch):
    """默认 LLM-only 生产图从检索到章节合并均不能落入规则写作。"""
    import re

    from app.llm.client import FakeLLMClient
    import app.llm.client as client_mod

    class ScenarioLLM(FakeLLMClient):
        """按当前隔离任务信封返回离线的确定性模型决策。"""

        async def function_call(self, messages, tools, tool_choice="auto", temperature=None):
            self.fc_calls.append({"message_count": len(messages), "tool_count": len(tools)})
            initial = str(messages[1].get("content") or "")
            if any(message.get("role") == "tool" for message in messages):
                return {
                    "success": True, "finish": True, "content": "done",
                    "latency_ms": 1, "model": "scenario-llm", "usage": {},
                }
            if "Task type: search" in initial:
                call = {
                    "id": f"search-{len(self.fc_calls)}", "name": "academic_search",
                    "arguments": {"query": "RAG evaluation", "max_results": 1},
                }
            else:
                match = re.search(r"source_ids:\s*([^\],]+)", initial)
                assert match, initial
                call = {
                    "id": "analyze-1", "name": "evidence_extract",
                    "arguments": {"source_id": match.group(1).strip()},
                }
            return {
                "success": True, "finish": False, "tool_calls": [call],
                "latency_ms": 1, "model": "scenario-llm", "usage": {},
            }

        async def generate_structured(
            self, system_prompt, user_prompt, output_schema,
            temperature=None, timeout_seconds=None, max_retries=None,
        ):
            self.calls.append({"schema": output_schema.__name__})
            if output_schema.__name__ == "LLMSourceSelectionOutput":
                source_id = re.search(r"source_id=([^;]+)", user_prompt).group(1).strip()
                raw = {
                    "analysis_count": 1, "selected_source_ids": [source_id],
                    "selection_reasons": {source_id: "evidence-bearing source"},
                    "coverage_plan": {"scope": [source_id]}, "rationale": "bounded",
                }
            elif output_schema.__name__ == "LLMOutlinePlan":
                raw = {
                    "sections": [
                        {
                            "heading": "研究概述", "guiding_question": "RAG 的基本作用是什么？",
                            "assigned_evidence_ids": ["E1"],
                        },
                        {
                            "heading": "主要结论", "guiding_question": "现有证据支持什么结论？",
                            "assigned_evidence_ids": ["E1"],
                        },
                    ],
                    "cross_cutting_themes": [], "evidence_gaps": [],
                }
            elif output_schema.__name__ == "LLMChapterOutput":
                evidence_id = re.search(r"\[([^\]]+:e\d+)\]\s+source_id=", user_prompt).group(1)
                heading = re.search(r"(?:章节标题|Chapter heading)[：:]\s*(.+)", user_prompt)
                raw = {
                    "heading": heading.group(1).strip() if heading else "研究结论",
                    "synthesis": (
                        "现有论文表明，检索增强生成把外部知识与生成过程结合，"
                        f"从而为回答提供可追溯的事实依据与研究边界。[[e:{evidence_id}]]"
                    ),
                    "evidence_ids": [evidence_id], "findings": [],
                    "source_title_translations": {},
                }
            else:
                return {"success": False, "error": f"unexpected schema {output_schema.__name__}"}
            return {
                "success": True, "data": output_schema(**raw), "latency_ms": 1,
                "model": "scenario-llm", "usage": {},
            }

    fake = ScenarioLLM()
    fake.set_intent_responses([{
        "intent": "deep_research", "execution_route": "full_research",
        "research_topic": "RAG 基础概览", "selected_tool": "",
        "requested_count": 1, "confidence": 0.99, "reasoning": "research request",
    }])
    client_mod._global_client = fake
    monkeypatch.setenv("LLM_ONLY_MODE", "true")
    write_results = []
    original_execute = WorkerAgent.execute

    async def capture_write(self, item):
        result = await original_execute(self, item)
        if item.profile == WorkerProfile.WRITE:
            write_results.append(result)
        return result

    monkeypatch.setattr(WorkerAgent, "execute", capture_write)
    result = await runtime.run_graph(
        "请调研 RAG 基础概览", max_sources=1,
        agent_mode="llm", run_eval=False,
    )

    assert result["status"] in {"completed", "completed_with_warnings"}
    assert len(write_results) == 2
    assert all(item.status == AgentTaskStatus.SUCCESS for item in write_results)
    assert all(item.metadata.get("mode") == "llm" for item in write_results)
    assert all(item.output_data.get("mode") == "llm" for item in write_results)


class _ChapterFailScenarioLLM(FakeLLMClient):
    """llm_only 深调研脚本化客户端：可对指定章节注入 LLM 生成失败。

    fail_mode="first"   —— 指定章节仅在首轮失败，修复轮成功（验证修复回路）；
    fail_mode="always"  —— 指定章节每次尝试都失败（验证修复耗尽后的部分交付）。
    """

    def __init__(self, fail_headings=(), fail_mode="first", sections=None):
        super().__init__()
        self.fail_headings = set(fail_headings)
        self.fail_mode = fail_mode
        self.sections = sections or [
            {
                "heading": "研究概述", "guiding_question": "RAG 的基本作用是什么？",
                "assigned_evidence_ids": ["E1"],
            },
            {
                "heading": "主要结论", "guiding_question": "现有证据支持什么结论？",
                "assigned_evidence_ids": ["E1"],
            },
        ]
        self.chapter_attempts = {}

    async def function_call(self, messages, tools, tool_choice="auto", temperature=None):
        self.fc_calls.append({"message_count": len(messages), "tool_count": len(tools)})
        initial = str(messages[1].get("content") or "")
        if any(message.get("role") == "tool" for message in messages):
            return {
                "success": True, "finish": True, "content": "done",
                "latency_ms": 1, "model": "scenario-llm", "usage": {},
            }
        if "Task type: search" in initial:
            call = {
                "id": f"search-{len(self.fc_calls)}", "name": "academic_search",
                "arguments": {"query": "RAG evaluation", "max_results": 1},
            }
        else:
            match = re.search(r"source_ids:\s*([^\],]+)", initial)
            assert match, initial
            call = {
                "id": "analyze-1", "name": "evidence_extract",
                "arguments": {"source_id": match.group(1).strip()},
            }
        return {
            "success": True, "finish": False, "tool_calls": [call],
            "latency_ms": 1, "model": "scenario-llm", "usage": {},
        }

    async def generate_structured(
        self, system_prompt, user_prompt, output_schema,
        temperature=None, timeout_seconds=None, max_retries=None,
    ):
        self.calls.append({"schema": output_schema.__name__})
        if output_schema.__name__ == "LLMSourceSelectionOutput":
            source_id = re.search(r"source_id=([^;]+)", user_prompt).group(1).strip()
            raw = {
                "analysis_count": 1, "selected_source_ids": [source_id],
                "selection_reasons": {source_id: "evidence-bearing source"},
                "coverage_plan": {"scope": [source_id]}, "rationale": "bounded",
            }
        elif output_schema.__name__ == "LLMOutlinePlan":
            raw = {
                "sections": self.sections,
                "cross_cutting_themes": [], "evidence_gaps": [],
            }
        elif output_schema.__name__ == "LLMChapterOutput":
            heading_match = re.search(
                r"(?:章节标题|Chapter heading|规范章节名)[：:]\s*(.+)", user_prompt,
            )
            heading = heading_match.group(1).strip() if heading_match else ""
            attempt = self.chapter_attempts.get(heading, 0) + 1
            self.chapter_attempts[heading] = attempt
            if heading in self.fail_headings and (
                self.fail_mode == "always" or attempt == 1
            ):
                return {
                    "success": False, "error": f"scripted generation failure: {heading}",
                    "latency_ms": 1, "model": "scenario-llm", "usage": {},
                }
            evidence_id = re.search(r"\[([^\]]+:e\d+)\]\s+source_id=", user_prompt).group(1)
            raw = {
                "heading": heading or "研究结论",
                "synthesis": (
                    "现有论文表明，检索增强生成把外部知识与生成过程结合，"
                    f"从而为回答提供可追溯的事实依据与研究边界。[[e:{evidence_id}]]"
                ),
                "evidence_ids": [evidence_id], "findings": [],
                "source_title_translations": {},
            }
        else:
            return {"success": False, "error": f"unexpected schema {output_schema.__name__}"}
        return {
            "success": True, "data": output_schema(**raw), "latency_ms": 1,
            "model": "scenario-llm", "usage": {},
        }


def _install_scenario_llm(fail_headings=(), fail_mode="first", sections=None):
    import app.llm.client as client_mod

    fake = _ChapterFailScenarioLLM(
        fail_headings=fail_headings, fail_mode=fail_mode, sections=sections,
    )
    fake.set_intent_responses([{
        "intent": "deep_research", "execution_route": "full_research",
        "research_topic": "RAG 基础概览", "selected_tool": "",
        "requested_count": 1, "confidence": 0.99, "reasoning": "research request",
    }])
    client_mod._global_client = fake
    return fake


@pytest.mark.asyncio
async def test_llm_only_partial_merge_assembles_partial_report_instead_of_failing():
    """llm_only 下单章失败不再整跑 raise：组装部分报告并标记完成度，供修复回路接手。"""
    sections = [
        {"heading": "研究概述", "guiding_question": "问题 1"},
        {"heading": "研究局限", "guiding_question": "问题 2"},
        {"heading": "主要结论", "guiding_question": "问题 3"},
    ]
    results = []
    for index, section in enumerate(sections):
        success = index != 1
        results.append(WorkerResult(
            task_id=f"write_{index + 1}", profile=WorkerProfile.WRITE,
            status=AgentTaskStatus.SUCCESS if success else AgentTaskStatus.FAILED,
            output_data={
                "section": section,
                "chapter": (
                    f"## {section['heading']}\n\n第 {index + 1} 章正文。[[e:e{index + 1}]]"
                    if success else ""
                ),
                "mode": "llm" if success else "",
            },
            metadata={"chapter_index": index, "mode": "llm" if success else ""},
            error=None if success else AgentError(
                error_code="LLM_CHAPTER_FAILED", message="没有足够的跨来源证据", recoverable=True,
            ),
        ))
    state = {
        "topic": "RAG", "language": "zh", "agent_mode": "llm", "llm_only": True,
        "execution_round": 0, "work_plan": None,
        "outline": {"topic": "RAG", "sections": sections},
        "sources": [], "evidence_cards": [], "citation_check_results": [],
        "citation_summary": {}, "worker_results": {}, "_worker_result_bucket": [
            {
                "round_id": 0, "revision": 0, "profile": "write",
                "task_id": result.task_id, "result": result.model_dump(mode="json"),
            }
            for result in results
        ],
    }

    merged = await runtime.node_merge_chapter_results(state)

    # 不再 raise：部分报告照常组装，成功章节保留，失败章节以占位符标记缺口。
    assert merged["report_completion_ready"] is False
    assert merged["expected_chapter_count"] == 3
    assert merged["written_chapter_count"] == 2
    assert len(merged["report_completion_issues"]) == 1
    assert "研究局限" in merged["report_completion_issues"][0]
    assert "研究概述" in merged["draft_report"]
    assert "主要结论" in merged["draft_report"]
    assert "本章未能由真实 LLM 生成" in merged["draft_report"]
    # 部分章节成功 → Write 阶段以 PARTIAL_SUCCESS 上报，Reviewer 才能走修复回路。
    write_stage = merged["worker_results"]["write"]
    assert write_stage["status"] == AgentTaskStatus.PARTIAL_SUCCESS.value
    assert write_stage["needs_replan"] is False


@pytest.mark.asyncio
async def test_llm_only_merge_still_rejects_rule_smuggled_chapter():
    """llm_only 门禁仍拦截规则降级章节：规则内容不允许混入 LLM-only 报告。"""
    sections = [
        {"heading": "研究概述", "guiding_question": "问题 1"},
        {"heading": "主要结论", "guiding_question": "问题 2"},
    ]
    results = []
    for index, section in enumerate(sections):
        mode = "llm" if index == 0 else "rule"
        results.append(WorkerResult(
            task_id=f"write_{index + 1}", profile=WorkerProfile.WRITE,
            status=AgentTaskStatus.SUCCESS,
            output_data={
                "section": section,
                "chapter": f"## {section['heading']}\n\n{('LLM' if index == 0 else '规则')}版正文。",
                "mode": mode,
            },
            metadata={"chapter_index": index, "mode": mode},
        ))
    state = {
        "topic": "RAG", "language": "zh", "agent_mode": "llm", "llm_only": True,
        "execution_round": 0, "work_plan": None,
        "outline": {"topic": "RAG", "sections": sections},
        "sources": [], "evidence_cards": [], "citation_check_results": [],
        "citation_summary": {}, "worker_results": {}, "_worker_result_bucket": [
            {
                "round_id": 0, "revision": 0, "profile": "write",
                "task_id": result.task_id, "result": result.model_dump(mode="json"),
            }
            for result in results
        ],
    }

    with pytest.raises(RuntimeError, match="rule-degraded"):
        await runtime.node_merge_chapter_results(state)


@pytest.mark.asyncio
async def test_llm_only_single_chapter_failure_recovers_via_repair_loop(monkeypatch):
    """llm_only 单章失败：先触发修复回路重写，重写成功后按 warnings 交付而非 failed。"""
    _install_scenario_llm(fail_headings={"研究概述"}, fail_mode="first")
    monkeypatch.setenv("LLM_ONLY_MODE", "true")

    result = await runtime.run_graph(
        "请调研 RAG 基础概览", max_sources=1, agent_mode="llm", run_eval=False,
    )

    assert result["status"] == "completed_with_warnings"
    assert result["report_completion_ready"] is True
    assert result["report_completion_issues"] == []
    # 修复轮把失败章节重写成功，最终报告两个章节齐全。
    assert "研究概述" in result["final_report"]
    assert "主要结论" in result["final_report"]


@pytest.mark.asyncio
async def test_llm_only_persistent_chapter_failure_delivers_partial(monkeypatch):
    """llm_only 单章持续失败且修复预算耗尽：部分交付（partial），不整跑 failed。"""
    _install_scenario_llm(fail_headings={"研究概述"}, fail_mode="always")
    monkeypatch.setenv("LLM_ONLY_MODE", "true")

    result = await runtime.run_graph(
        "请调研 RAG 基础概览", max_sources=1, agent_mode="llm", run_eval=False,
    )

    assert result["status"] == "partial"
    assert result["report_completion_ready"] is False
    assert len(result["report_completion_issues"]) == 1
    assert "研究概述" in result["report_completion_issues"][0]
    # 成功章节保留，失败章节以显式缺口占位呈现，不再是整跑失败。
    assert "主要结论" in result["final_report"]
    assert "本章未能由真实 LLM 生成" in result["final_report"]


@pytest.mark.asyncio
async def test_llm_only_limitation_chapter_insufficient_sources_delivers_partial(monkeypatch):
    """复现原始 d26540b5 场景：局限章在 llm_only 下仅 1 个来源，两源门禁无法通过，
    修复预算耗尽后交付部分报告（partial），不再整跑 failed。"""
    sections = [
        {
            "heading": "研究概述", "guiding_question": "RAG 的基本作用是什么？",
            "assigned_evidence_ids": ["E1"],
        },
        {
            "heading": "研究局限", "guiding_question": "现有证据支持什么结论？",
            "assigned_evidence_ids": ["E1"],
        },
    ]
    _install_scenario_llm(sections=sections)
    monkeypatch.setenv("LLM_ONLY_MODE", "true")

    result = await runtime.run_graph(
        "请调研 RAG 基础概览", max_sources=1, agent_mode="llm", run_eval=False,
    )

    # 局限章因"两源综合门禁"在 llm_only 下必然失败且无法修复 → 部分交付。
    assert result["status"] == "partial"
    assert result["report_completion_ready"] is False
    assert len(result["report_completion_issues"]) == 1
    assert "研究局限" in result["report_completion_issues"][0]
    # 成功的研究概述保留，失败的局限章以显式缺口占位呈现。
    assert "研究概述" in result["final_report"]
    assert "本章未能由真实 LLM 生成" in result["final_report"]


@pytest.mark.asyncio
async def test_max_workers_one_executes_every_planned_chapter():
    """并发上限为一时仍须顺序执行大纲中的全部章节。"""
    spec = _spec(ExecutionClass.RESEARCH, "full_research")
    spec.budget.max_workers = 1
    spec.metadata.update({"agent_mode": "rule", "llm_only": False})
    plan = PlannerAgent().plan(spec)
    sections = [
        {
            "heading": f"章节 {index}", "guiding_question": f"问题 {index}",
            "assigned_evidence_ids": [], "assigned_source_ids": [],
        }
        for index in range(1, 4)
    ]
    state = {
        "topic": "RAG", "language": "zh", "agent_mode": "rule", "llm_only": False,
        "execution_round": 0, "work_plan": plan.model_dump(mode="json"),
        "outline": {"topic": "RAG", "sections": sections},
        "sources": [], "evidence_cards": [], "citation_check_results": [],
        "citation_summary": {}, "worker_results": {}, "_worker_result_bucket": [],
    }

    sends = runtime.send_chapter_work_items(state)
    assert len(sends) == 3
    # 关键步骤：模拟 max_concurrency=1 的串行调度，确认不是只执行首章。
    for send in sends:
        delta = await runtime.node_worker_send(send.arg)
        runtime._apply_delta(state, delta)
    merged = await runtime.node_merge_chapter_results(state)

    assert merged["expected_chapter_count"] == 3
    assert merged["written_chapter_count"] == 3
    assert all(f"章节 {index}" in merged["draft_report"] for index in range(1, 4))


@pytest.mark.asyncio
async def test_empty_allowed_tools_fails_closed_before_tool_call(monkeypatch):
    from app.tools.academic_search_tool import AcademicSearchTool

    called = 0

    async def counted(_self, **_kwargs):
        nonlocal called
        called += 1
        return ToolResult(True, data={"results": []})

    monkeypatch.setattr(AcademicSearchTool, "_arun", counted)
    result = await WorkerAgent().execute(WorkItem(
        task_id="search", profile=WorkerProfile.SEARCH,
        instruction="检索", allowed_tools=[], strategy=WorkerStrategy.REACT,
    ))

    assert result.status == AgentTaskStatus.FAILED
    assert result.tool_calls == []
    assert called == 0


@pytest.mark.asyncio
async def test_single_tool_budget_blocks_second_real_call_without_trace_truncation(monkeypatch):
    from app.tools.paper_metadata_tool import PaperMetadataTool
    from app.tools.source_quality_scorer import SourceQualityScorer

    calls = {"paper_metadata": 0, "source_quality_scorer": 0}

    async def metadata(_self, **kwargs):
        calls["paper_metadata"] += 1
        return ToolResult(True, data={"sources": list(kwargs["sources"])})

    async def scorer(_self, **_kwargs):
        calls["source_quality_scorer"] += 1
        return ToolResult(True, data={"scores_by_id": {}})

    monkeypatch.setattr(PaperMetadataTool, "_arun", metadata)
    monkeypatch.setattr(SourceQualityScorer, "_arun", scorer)
    source = {"source_id": "s1", "title": "Source"}
    result = await WorkerAgent().execute(WorkItem(
        task_id="read", profile=WorkerProfile.READ, instruction="读取",
        resources=[source], allowed_tools=["paper_metadata", "source_quality_scorer"],
        strategy=WorkerStrategy.DETERMINISTIC, max_tool_calls=1,
    ))

    assert result.status == AgentTaskStatus.FAILED
    assert calls == {"paper_metadata": 1, "source_quality_scorer": 0}
    assert [call.tool_name for call in result.tool_calls] == ["paper_metadata"]
    assert "预算" in result.error.message


@pytest.mark.asyncio
async def test_per_tool_timeout_is_enforced_at_business_call(monkeypatch):
    from app.tools.academic_search_tool import AcademicSearchTool

    called = 0

    async def slow(_self, **_kwargs):
        nonlocal called
        called += 1
        await asyncio.sleep(0.05)
        return ToolResult(True, data={"results": [{"source_id": "late"}]})

    monkeypatch.setattr(AcademicSearchTool, "_arun", slow)
    result = await WorkerAgent().execute(WorkItem(
        task_id="direct", profile=WorkerProfile.DIRECT, instruction="检索",
        allowed_tools=["academic_search"], per_tool_timeout_ms=1,
        timeout_ms=1_000,
    ))

    assert called == 1
    assert result.status == AgentTaskStatus.FAILED
    assert "timed out" in result.error.message
    assert [call.tool_name for call in result.tool_calls] == ["academic_search"]


def test_source_only_hard_metric_is_clarify_never_pass():
    planner = PlannerAgent()
    original = planner.plan(_spec(ExecutionClass.RESEARCH, "full_research"))
    revised = planner.replan(original, ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["analyze"],
        feedback=[{
            "task_id": "analyze", "message": "extractor unavailable",
            "reusable_dependency_ids": ["read"],
        }],
    ))
    results = [
        WorkerResult(
            task_id=item.task_id, profile=item.profile,
            status=AgentTaskStatus.SUCCESS,
            output_data={"report": "来源级草稿"} if item.profile == WorkerProfile.WRITE else {},
        )
        for item in revised.items
    ]
    verdict = ReviewerAgent().review(revised, results, final_output={
        "final_report": "来源级草稿",
        "eval_metrics": {"evidence_available": False},
    })

    assert verdict.outcome == ReviewOutcome.CLARIFY
    assert verdict.outcome != ReviewOutcome.PASS


def test_upstream_failure_replan_keeps_valid_topology_and_changes_action():
    planner = PlannerAgent()
    plan = planner.plan(_spec(ExecutionClass.RESEARCH, "full_research"))
    revised = planner.replan(plan, ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["search_2"],
        feedback=[{
            "task_id": "search_2", "code": "PROVIDER_DOWN",
            "message": "OpenAlex unavailable", "reusable_dependency_ids": [],
        }],
    ))

    search = next(item for item in revised.items if item.task_id == "search_2")
    assert search.metadata["strategy_changed"] == "query_expanded"
    assert "alternative independent sources" in search.input_data["query"]
    assert all(set(item.depends_on) <= {candidate.task_id for candidate in revised.items}
               for item in revised.items)
    state = {
        "execution_spec": revised.execution_spec.model_dump(mode="json"),
        "work_plan": revised.model_dump(mode="json"), "execution_round": 1,
    }
    sends = runtime.route_after_four_agent_planner(state)
    assert len(sends) == 1
    assert sends[0].arg["work_item"]["task_id"] == "search_2"


@pytest.mark.asyncio
async def test_run_eval_false_skips_evaluator_events_and_metrics():
    result = await runtime.run_graph(
        "调研 RAG evaluation，比较主要方法并总结局限",
        max_sources=2, agent_mode="rule", run_eval=False,
    )

    assert result["eval_metrics"] == {}
    assert not any(event.get("event") == "evaluator_complete" for event in result["trace"])


@pytest.mark.asyncio
async def test_context_load_answer_uses_only_explicit_worker_resources():
    spec = _spec(ExecutionClass.CONTEXTUAL, "conversation")
    spec.resources = [{
        "resource_type": "paper", "source_id": "visible", "title": "Visible",
    }]
    spec.resource_ids = ["visible"]
    planner_delta = await runtime.node_four_agent_planner({
        "topic": "总结它", "execution_spec": spec.model_dump(mode="json"),
        "execution_round": 0, "runtime_budget_id": "", "run_id": "context-test",
        # 这个父 State 字段不在 Controller 的显式资源信封中，Planner 不得读取。
        "seed_papers": [{"source_id": "hidden", "title": "Hidden"}],
    })
    state = {
        "work_plan": planner_delta["work_plan"], "execution_round": 0,
        "worker_results": {}, "_worker_result_bucket": [],
        "conversation_operation": "summarize", "language": "zh",
    }
    sends = runtime.route_after_four_agent_planner({
        **state, "execution_spec": spec.model_dump(mode="json"),
    })
    delta = await runtime.node_worker_send(sends[0].arg)
    runtime._apply_delta(state, delta)
    runtime._apply_delta(state, await runtime.node_merge_context_results(state))
    answer_sends = runtime.send_answer_work_item(state)
    answer_item = WorkItem.model_validate(answer_sends[0].arg["work_item"])

    assert [resource["source_id"] for resource in answer_item.resources] == ["visible"]
    assert "hidden" not in str(answer_item.input_data)


def test_apply_delta_allows_explicit_empty_and_false_overwrite():
    state = {
        "sources": [{"source_id": "old"}], "eval_metrics": {"old": True},
        "retry_count": 1, "run_eval": True, "answer": "old", "trace": [{"event": "old"}],
    }
    runtime._apply_delta(state, {
        "sources": [], "eval_metrics": {}, "retry_count": 0,
        "run_eval": False, "answer": "", "trace": [],
    })

    assert state["sources"] == []
    assert state["eval_metrics"] == {}
    assert state["retry_count"] == 0
    assert state["run_eval"] is False
    assert state["answer"] == ""
    assert state["trace"] == [{"event": "old"}]


@pytest.mark.asyncio
async def test_shared_runtime_budget_and_safety_fail_before_business_call(monkeypatch):
    from app.tools.academic_search_tool import AcademicSearchTool

    called = 0

    async def counted(_self, **_kwargs):
        nonlocal called
        called += 1
        return ToolResult(True, data={
            "results": [{"source_id": f"s{called}", "title": "Source"}],
        })

    monkeypatch.setattr(AcademicSearchTool, "_arun", counted)
    register_runtime_budget("shared-test", max_tool_calls=1, total_timeout_ms=10_000)
    try:
        common = {
            "profile": WorkerProfile.DIRECT, "instruction": "检索",
            "allowed_tools": ["academic_search"],
            "metadata": {"runtime_budget_id": "shared-test"},
        }
        first = await WorkerAgent().execute(WorkItem(task_id="one", **common))
        second = await WorkerAgent().execute(WorkItem(task_id="two", **common))
        network_denied = await WorkerAgent().execute(WorkItem(
            task_id="denied", profile=WorkerProfile.DIRECT, instruction="检索",
            allowed_tools=["academic_search"],
            safety_policy=SafetyPolicy(allow_network=False),
        ))
    finally:
        clear_runtime_budget("shared-test")

    assert first.status == AgentTaskStatus.SUCCESS
    assert second.status == AgentTaskStatus.FAILED
    assert network_denied.status == AgentTaskStatus.FAILED
    assert called == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["atomic", "contextual", "research"])
async def test_three_routes_emit_complete_four_agent_protocol(kind):
    session = None
    if kind == "atomic":
        topic = "查找 2 篇关于 RAG 的论文"
    elif kind == "contextual":
        topic = "这篇论文的方法是什么？"
        session = SessionContext(
            session_id="four-agent-context",
            active_paper_id="p1",
            recommended_papers=[{
                "source_id": "p1", "paper_id": "p1", "title": "RAG Paper",
                "url": "https://example.org/p1",
                "full_text": "本文提出一种可验证的 RAG 评估方法。",
            }],
        )
    else:
        topic = "调研 RAG evaluation，比较主要方法并总结局限"

    result = await runtime.run_graph(
        topic, max_sources=2, agent_mode="rule", session_context=session,
    )
    roles = {
        event.get("architectural_role")
        for event in result["trace"]
        if event.get("event") == "agent_protocol_validated"
    }
    assert roles >= {"controller", "planner", "worker", "reviewer"}
    worker_events = [
        event for event in result["trace"]
        if event.get("event") == "agent_protocol_validated"
        and event.get("architectural_role") == "worker"
    ]
    if kind == "atomic":
        assert len({event["task_id"] for event in worker_events}) == 1
    elif kind == "contextual":
        assert {event["task_id"] for event in worker_events} == {"context_load", "answer"}
        assert all(not event.get("tools_called") for event in worker_events)
    else:
        assert len({event["task_id"] for event in worker_events}) > 1


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [WorkerProfile.READ, WorkerProfile.CITE])
async def test_deterministic_read_and_cite_timeout_never_succeed(monkeypatch, profile):
    from app.tools.citation_check_tool import CitationCheckTool
    from app.tools.paper_metadata_tool import PaperMetadataTool

    async def slow(_self, **_kwargs):
        await asyncio.sleep(0.03)
        return ToolResult(True, data={})

    source = {
        "source_id": "s-timeout", "title": "Timeout Source",
        "url": "https://example.org/timeout", "full_text": "Abstract: verified claim.",
    }
    if profile == WorkerProfile.READ:
        monkeypatch.setattr(PaperMetadataTool, "_arun", slow)
        item = WorkItem(
            task_id="read", profile=profile, instruction="read",
            resources=[source], allowed_tools=["paper_metadata", "source_quality_scorer"],
            strategy=WorkerStrategy.DETERMINISTIC, per_tool_timeout_ms=1,
        )
    else:
        monkeypatch.setattr(CitationCheckTool, "_arun", slow)
        item = WorkItem(
            task_id="cite", profile=profile, instruction="cite",
            resources=[source], allowed_tools=["citation_check"],
            input_data={"evidence_cards": [{
                "source_id": "s-timeout", "url": source["url"], "quote": "verified claim",
            }]},
            strategy=WorkerStrategy.DETERMINISTIC, per_tool_timeout_ms=1,
        )

    result = await WorkerAgent().execute(item)
    assert result.status == AgentTaskStatus.TIMEOUT
    assert result.needs_replan is True
    assert any(not call.success for call in result.tool_calls)


def test_reviewer_rejects_success_that_contains_failed_tool_call():
    plan = PlannerAgent().plan(_spec(ExecutionClass.ATOMIC, "direct_tool"))
    invalid = WorkerResult(
        task_id="direct_1", profile=WorkerProfile.DIRECT,
        status=AgentTaskStatus.SUCCESS,
        output_data={"answer": "invalid"},
        tool_calls=[AgentToolCall(
            tool_name="academic_search", success=False, error="timeout",
        )],
    )
    verdict = ReviewerAgent().review(plan, [invalid])
    assert verdict.outcome != ReviewOutcome.PASS
    assert "direct_1" in verdict.failed_task_ids


@pytest.mark.asyncio
async def test_run_eval_false_source_only_is_clarify_or_fail_without_metrics():
    planner = PlannerAgent()
    original = planner.plan(_spec(ExecutionClass.RESEARCH, "full_research"))
    revised = planner.replan(original, ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["analyze"],
        feedback=[{
            "task_id": "analyze", "message": "extractor unavailable",
            "reusable_dependency_ids": ["read"],
        }],
    ))
    successful = {
        item.task_id: WorkerResult(
            task_id=item.task_id, profile=item.profile,
            status=AgentTaskStatus.SUCCESS,
            output_data={"report": "来源级草稿"} if item.profile == WorkerProfile.WRITE else {},
        ).model_dump(mode="json")
        for item in revised.items
    }
    base = {
        "work_plan": revised.model_dump(mode="json"), "worker_results": successful,
        "run_eval": False, "answer": "", "expected_chapter_count": None,
        "written_chapter_count": None, "topic": "RAG", "trace": [],
    }
    with_report = await runtime.node_four_agent_reviewer({
        **base, "final_report": "来源级草稿",
    })
    without_report = await runtime.node_four_agent_reviewer({
        **base, "final_report": "",
    })

    assert with_report["review_verdict"]["outcome"] == "clarify"
    assert without_report["review_verdict"]["outcome"] == "fail"
    assert with_report["eval_metrics"] == without_report["eval_metrics"] == {}
    assert not any(event["event"] == "evaluator_complete" for event in with_report["trace"])


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [WorkerProfile.ANSWER, WorkerProfile.WRITE])
async def test_expired_runtime_deadline_blocks_toolless_synthesis(profile):
    budget_id = f"expired-{profile.value}"
    register_runtime_budget(budget_id, max_tool_calls=10, total_timeout_ms=1)
    await asyncio.sleep(0.005)
    try:
        item = WorkItem(
            task_id=profile.value, profile=profile, instruction="生成文本",
            strategy=WorkerStrategy.SYNTHESIS,
            metadata={"runtime_budget_id": budget_id},
            input_data={"agent_mode": "rule"},
        )
        result = await WorkerAgent().execute(item)
    finally:
        clear_runtime_budget(budget_id)

    assert result.status == AgentTaskStatus.TIMEOUT
    assert result.needs_replan is True


@pytest.mark.asyncio
async def test_sync_and_async_graph_deadlines_fail_and_clear_budget(monkeypatch):
    from app.observability.lifecycle import runtime_budget_snapshot

    class SlowGraph:
        async def ainvoke(self, *_args, **_kwargs):
            await asyncio.sleep(0.05)
            return {}

        async def astream(self, *_args, **_kwargs):
            await asyncio.sleep(0.05)
            if False:
                yield {}

    monkeypatch.setattr(runtime, "_graph_send_instance", SlowGraph())
    sync_id = f"deadline-sync-{uuid.uuid4().hex}"
    async_id = f"deadline-async-{uuid.uuid4().hex}"
    sync_result = await runtime.run_graph(
        "deadline", agent_mode="rule", run_id=sync_id, total_timeout_ms=1,
    )
    async_result = await runtime.run_graph_async(
        "deadline", agent_mode="rule", run_id=async_id, total_timeout_ms=1,
    )

    assert sync_result["status"] == async_result["status"] == "failed"
    assert runtime_budget_snapshot(sync_id) == {}
    assert runtime_budget_snapshot(async_id) == {}


@pytest.mark.asyncio
async def test_external_write_and_destructive_capabilities_are_preflight_denied(monkeypatch):
    from app.tools.base import BaseTool
    from app.tools.registry import ToolCapabilityMetadata, ToolRegistry

    class SideEffectProbe(BaseTool):
        @property
        def name(self):
            return "side_effect_probe"

        @property
        def description(self):
            return "probe"

        async def _arun(self, **_kwargs):
            side_effects.append("executed")
            return ToolResult(True, data={"ok": True})

    side_effects = []
    registry = ToolRegistry.get_instance()
    probe = SideEffectProbe()
    registry.register(probe, ToolCapabilityMetadata(
        external_write=True, destructive=True,
    ))
    try:
        result = await WorkerAgent(registry).execute(WorkItem(
            task_id="probe", profile=WorkerProfile.DIRECT,
            instruction="probe", allowed_tools=[probe.name],
        ))
        local_allowed = WorkerAgent(registry).allowed_tools(WorkItem(
            task_id="local", profile=WorkerProfile.DIRECT,
            instruction="local", allowed_tools=["local_paper_search"],
            safety_policy=SafetyPolicy(allow_network=False),
        ))
    finally:
        registry.unregister(probe.name, probe)

    assert result.status == AgentTaskStatus.FAILED
    assert side_effects == []
    assert local_allowed == ["local_paper_search"]


def test_read_root_replan_reuses_sources_and_only_dispatches_read():
    planner = PlannerAgent()
    original = planner.plan(_spec(ExecutionClass.RESEARCH, "full_research"))
    revised = planner.replan(original, ReviewVerdict(
        outcome=ReviewOutcome.REPLAN,
        failed_task_ids=["read"],
        feedback=[{"task_id": "read", "message": "metadata timeout"}],
    ))
    sources = [{"source_id": "s1", "title": "One"}, {"source_id": "s2", "title": "Two"}]
    sends = runtime.route_after_four_agent_planner({
        "execution_spec": revised.execution_spec.model_dump(mode="json"),
        "work_plan": revised.model_dump(mode="json"),
        "execution_round": 1, "sources": sources,
    })

    assert len(sends) == 2
    assert {WorkItem.model_validate(send.arg["work_item"]).profile for send in sends} == {WorkerProfile.READ}
    assert all("force_refresh" in send.arg["work_item"]["input_data"] for send in sends)


@pytest.mark.asyncio
async def test_controller_emits_only_refs_and_context_worker_loads_real_resources():
    from app.services.run_store import run_store
    from app.services.session_store import session_store

    session_id = f"context-ref-{uuid.uuid4().hex}"
    session_store.create(session_id=session_id)
    paper = {"source_id": "p1", "title": "Visible Paper", "full_text": "paper body"}
    session_store.set_recommended_papers(session_id, [paper])
    report_id = run_store.create("report", run_id=f"report-{uuid.uuid4().hex}")
    run_store.update(
        report_id, final_report="secret report body",
        evidence_cards=[{"evidence_id": "e1", "source_id": "p1"}], sources=[paper],
    )
    session_store.update(
        session_id, active_paper_id="p1", active_report_id=report_id,
        conversation_messages=[{"role": "user", "content": "secret history"}],
    )
    session = session_store.get(session_id)
    try:
        spec = await ControllerAgent().execute(
            "展开这份报告", agent_mode="rule", session_context=session,
        )
        serialized = str(spec.resources)
        assert "secret report body" not in serialized
        assert "secret history" not in serialized
        assert all(resource["resource_type"].endswith("_ref") for resource in spec.resources)
        context_item = PlannerAgent().plan(spec).items[0]
        loaded = await WorkerAgent().execute(context_item)
    finally:
        session_store.delete(session_id)

    assert loaded.status == AgentTaskStatus.SUCCESS
    resources = loaded.output_data["resources"]
    assert any(resource.get("report_text") == "secret report body" for resource in resources)
    assert any(resource.get("history", [{}])[-1].get("content") == "secret history" for resource in resources)


@pytest.mark.asyncio
async def test_async_finalizer_exception_still_clears_runtime_budget(monkeypatch):
    from app.observability.lifecycle import runtime_budget_snapshot

    class EmptyGraph:
        async def astream(self, *_args, **_kwargs):
            if False:
                yield {}

    run_id = f"finalizer-error-{uuid.uuid4().hex}"
    register_runtime_budget(run_id, max_tool_calls=1, total_timeout_ms=10_000)
    monkeypatch.setattr(runtime, "_graph_send_instance", EmptyGraph())
    monkeypatch.setattr(runtime, "_finalize_run", lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await runtime.run_graph_async("finalize", run_id=run_id, agent_mode="rule")
    assert runtime_budget_snapshot(run_id) == {}


def test_base_tool_hash_input_compatibility():
    from app.tools.academic_search_tool import AcademicSearchTool
    from app.tools.base import BaseTool

    payload = {"query": "RAG", "filters": {"year": 2025}}
    assert BaseTool._hash_input(payload) == AcademicSearchTool()._hash_input(payload)
