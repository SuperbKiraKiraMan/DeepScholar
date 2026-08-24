"""四 Agent 架构的统一只读 Reviewer。"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Iterable, List

from app.llm.protocol import StructuredLLMClient

from app.agents.schemas import (
    AgentTaskStatus,
    ReviewOutcome,
    ReviewVerdict,
    WorkerResult,
    WorkerProfile,
    WorkPlan,
)


class ReviewerAgent:
    """统一 DirectReviewer、Evaluator 与 FinalReviewer 的验收语义。

    Reviewer 不调用业务工具，也不负责撰写报告；它只检查 Worker 产物并给出
    pass/repair/replan/clarify/fail 之一。
    """

    role = "reviewer"
    allowed_tools: tuple[str, ...] = ()

    HARD_METRICS = (
        "no_fake_citation", "citation_id_exists", "source_url_valid",
        "evidence_available",
    )

    def __init__(self, evaluator=None, llm_client: StructuredLLMClient | None = None):
        # Evaluator 是 Reviewer 内部的只读兼容适配器，不成为主图角色。
        if evaluator is None:
            from app.agents.evaluator import Evaluator
            evaluator = Evaluator()
        self.evaluator = evaluator
        self._llm_client = llm_client

    async def review_hybrid(
        self,
        plan: WorkPlan,
        results: Iterable[WorkerResult],
        *,
        evaluator_input: Dict[str, Any],
        final_output: Dict[str, Any] | None = None,
        run_eval: bool = True,
        agent_mode: str = "rule",
    ) -> tuple[ReviewVerdict, Dict[str, Any]]:
        """先执行不可放宽的硬门，再按需执行 Evaluator 与语义 LLM。"""
        collected = list(results)
        hard_verdict = self.review(plan, collected, final_output=final_output)
        verdict = hard_verdict
        evaluator_ran = False
        if run_eval and collected and plan.execution_spec.execution_class.value == "research":
            # 关键步骤：硬门先定结论；Evaluator 可补充指标，但绝不能放宽已有硬失败。
            evaluation = self.evaluator.evaluate(**evaluator_input)
            evaluator_ran = True
            payload = dict(final_output or {})
            payload["eval_metrics"] = dict(evaluation.get("metrics") or {})
            payload["eval_metric_details"] = dict(evaluation.get("metrics_detail") or {})
            payload["eval_feedback"] = list(evaluation.get("feedback") or [])
            evaluated_verdict = self.review(plan, collected, final_output=payload)
            if hard_verdict.outcome == ReviewOutcome.PASS:
                verdict = evaluated_verdict
            else:
                verdict = hard_verdict.model_copy(update={"final_output": payload}, deep=True)
        if hard_verdict.outcome != ReviewOutcome.PASS:
            info = self._semantic_info(False, False, "hard_gate_blocked")
            info["evaluator_ran"] = evaluator_ran
            return verdict, info
        if verdict.outcome != ReviewOutcome.PASS:
            info = self._semantic_info(False, False, "evaluator_gate_blocked")
            info["evaluator_ran"] = evaluator_ran
            return verdict, info
        if (
            not run_eval
            or agent_mode != "llm"
            or plan.execution_spec.execution_class.value != "research"
        ):
            info = self._semantic_info(False, False, "not_applicable")
            info["evaluator_ran"] = evaluator_ran
            return verdict, info

        from app.llm.client import get_llm_client
        from app.llm.schemas import LLMSemanticReviewOutput

        client = self._llm_client or get_llm_client()
        timeout_seconds = max(0.001, min(
            float(os.getenv("HYBRID_REVIEWER_TIMEOUT_SECONDS", "30")),
            plan.execution_spec.budget.total_timeout_ms / 1000,
        ))
        prompt = self._semantic_payload(plan, collected, evaluator_input, verdict.final_output)
        try:
            result = await asyncio.wait_for(
                client.generate_structured(
                    system_prompt=(
                        "你是只读研究质量审查器。仅输出严格 schema，不输出思维链或正文。"
                        "检查主题覆盖、论证一致性、来源支持、重复空泛与相关性。"
                        "你只能维持 pass 或收紧为 repair/clarify/fail。"
                    ),
                    user_prompt=json.dumps(prompt, ensure_ascii=False, default=str),
                    output_schema=LLMSemanticReviewOutput,
                    temperature=0.0,
                    timeout_seconds=max(1, int(timeout_seconds)),
                    max_retries=0,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            info = self._semantic_info(True, False, "timeout")
            info["evaluator_ran"] = evaluator_ran
            return verdict, info
        except Exception:
            info = self._semantic_info(True, False, "client_error")
            info["evaluator_ran"] = evaluator_ran
            return verdict, info
        if not result.get("success"):
            info = self._semantic_info(True, False, "generation_failed", result)
            info["evaluator_ran"] = evaluator_ran
            return verdict, info
        try:
            tightened = self._tighten_verdict(plan, verdict, result["data"])
        except Exception:
            info = self._semantic_info(True, False, "candidate_rejected", result)
            info["evaluator_ran"] = evaluator_ran
            return verdict, info
        info = self._semantic_info(True, True, "accepted", result)
        info["evaluator_ran"] = evaluator_ran
        return tightened, info

    @staticmethod
    def _semantic_payload(plan, results, evaluator_input, final_output):
        """仅提供语义审查所需材料，并对长文本做确定性裁剪。"""
        sources = list(evaluator_input.get("sources") or [])
        cards = list(evaluator_input.get("evidence_cards") or [])
        return {
            "topic": plan.execution_spec.research_topic,
            "user_request": plan.execution_spec.user_request,
            "report": str((final_output or {}).get("final_report") or "")[:30000],
            "sources": [
                {"source_id": item.get("source_id"), "title": item.get("title")}
                for item in sources[:100]
            ],
            "evidence": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "source_id": item.get("source_id"),
                    "claim": str(item.get("claim") or "")[:500],
                }
                for item in cards[:200]
            ],
            "worker_status": [
                {"task_id": item.task_id, "profile": item.profile.value, "status": item.status.value}
                for item in results
            ],
            "eval_metrics": dict((final_output or {}).get("eval_metrics") or {}),
        }

    @staticmethod
    def _tighten_verdict(plan, deterministic, semantic):
        outcome = ReviewOutcome(semantic.outcome)
        if outcome == ReviewOutcome.PASS:
            return deterministic
        known = {item.task_id for item in plan.items}
        failed_ids = list(dict.fromkeys(semantic.failed_task_ids))
        repair_scope = list(dict.fromkeys(semantic.repair_scope))
        if (set(failed_ids) | set(repair_scope)) - known:
            raise ValueError("语义 Reviewer 引用了未知任务")
        profiles = {item.task_id: item.profile for item in plan.items}
        requested_ids = set(failed_ids) | set(repair_scope)
        unsupported_repair = {
            task_id for task_id in requested_ids
            if profiles.get(task_id) not in {
                WorkerProfile.ANALYZE, WorkerProfile.CITE, WorkerProfile.WRITE,
            }
        }
        if outcome == ReviewOutcome.REPAIR and unsupported_repair:
            # 关键步骤：Search/Read 不是局部写作修复，必须升级为受 Planner 管理的 replan。
            if plan.replan_count < 1:
                outcome = ReviewOutcome.REPLAN
                repair_scope = []
            else:
                usable = bool(str(deterministic.final_output.get("final_report") or "").strip())
                outcome = ReviewOutcome.CLARIFY if usable else ReviewOutcome.FAIL
                repair_scope = []
        if outcome == ReviewOutcome.REPAIR:
            if not repair_scope:
                repair_scope = [
                    task_id for task_id in failed_ids
                    if profiles.get(task_id) in {
                        WorkerProfile.ANALYZE, WorkerProfile.CITE, WorkerProfile.WRITE,
                    }
                ] or ["write"]
            if any(
                profiles.get(task_id) not in {
                    WorkerProfile.ANALYZE, WorkerProfile.CITE, WorkerProfile.WRITE,
                }
                for task_id in repair_scope
            ):
                raise ValueError("语义 repair_scope 超出运行时支持范围")
        if outcome == ReviewOutcome.REPAIR and plan.repair_count >= 1:
            outcome = ReviewOutcome.FAIL
        feedback = [
            {
                "task_id": item.task_id,
                "code": item.code,
                "message": item.message,
                "recoverable": outcome in {ReviewOutcome.REPAIR, ReviewOutcome.REPLAN},
                "source": "semantic_reviewer",
            }
            for item in semantic.feedback
        ]
        return ReviewVerdict(
            outcome=outcome,
            failed_task_ids=failed_ids,
            repair_scope=repair_scope,
            feedback=feedback,
            final_output=dict(deterministic.final_output),
            clarification=(
                semantic.clarification
                or ("检索或来源读取需要重新规划，请确认是否继续。"
                    if outcome == ReviewOutcome.CLARIFY else "")
            ),
            summary=semantic.summary or "语义审查收紧了确定性通过结果",
        )

    @staticmethod
    def _semantic_info(attempted, success, status, result=None):
        result = result or {}
        # 只暴露调用状态与计量信息，禁止记录原始输出和内部推理。
        return {
            "attempted": bool(attempted),
            "success": bool(success),
            "status": status,
            "model": str(result.get("model") or ""),
            "latency_ms": int(result.get("latency_ms") or 0),
            "usage": dict(result.get("usage") or {}),
        }

    def review(
        self,
        plan: WorkPlan,
        results: Iterable[WorkerResult],
        *,
        final_output: Dict[str, Any] | None = None,
    ) -> ReviewVerdict:
        collected = list(results)
        by_id = {result.task_id: result for result in collected}
        missing = [item.task_id for item in plan.items if item.task_id not in by_id]
        non_terminal = [
            result.task_id for result in collected
            if result.status in {
                AgentTaskStatus.PENDING, AgentTaskStatus.RUNNING,
                AgentTaskStatus.CANCELLED,
            }
        ]
        failed = [
            result.task_id for result in collected
            if result.status in {AgentTaskStatus.FAILED, AgentTaskStatus.TIMEOUT}
        ]
        inconsistent_success = [
            result.task_id for result in collected
            if result.status == AgentTaskStatus.SUCCESS
            and any(not call.success for call in result.tool_calls)
        ]
        malformed_citations = [
            result.task_id for result in collected
            if result.status == AgentTaskStatus.SUCCESS
            and result.profile == WorkerProfile.CITE
            and (
                not result.tool_calls
                or any(not call.success for call in result.tool_calls)
                or not isinstance(result.output_data.get("citation_check_results"), list)
                or not {
                    "total_checked", "valid_count", "invalid_count", "all_valid",
                } <= set(result.output_data.get("citation_summary") or {})
            )
        ]
        failed = list(dict.fromkeys(failed + inconsistent_success + malformed_citations))
        partial = [
            result.task_id for result in collected
            if result.status == AgentTaskStatus.PARTIAL_SUCCESS
        ]
        feedback: List[Dict[str, Any]] = []
        for task_id in list(dict.fromkeys(inconsistent_success + malformed_citations)):
            feedback.append({
                "task_id": task_id,
                "code": "INVALID_SUCCESS_RESULT",
                "message": "SUCCESS 结果包含失败工具调用或不完整的确定性引用产物",
                "recoverable": True,
            })
        for task_id in non_terminal:
            feedback.append({
                "task_id": task_id,
                "code": "NON_TERMINAL_RESULT",
                "message": "Reviewer 拒绝非终态 WorkerResult",
                "recoverable": False,
            })
        for task_id in missing:
            feedback.append({
                "task_id": task_id,
                "code": "MISSING_RESULT",
                "message": "计划任务没有返回结果",
            })
        for result in collected:
            if result.error is not None:
                planned = next(
                    (item for item in plan.items if item.task_id == result.task_id), None,
                )
                reusable_dependencies = [
                    dependency for dependency in (planned.depends_on if planned else [])
                    if dependency in by_id
                    and by_id[dependency].status == AgentTaskStatus.SUCCESS
                    and not by_id[dependency].needs_replan
                ]
                feedback.append({
                    "task_id": result.task_id,
                    "code": result.error.error_code,
                    "message": result.error.message,
                    "recoverable": result.error.recoverable,
                    "reusable_dependency_ids": reusable_dependencies,
                })
            elif result.needs_replan:
                feedback.append({
                    "task_id": result.task_id,
                    "code": "COMPLETION_GATE_FAILED",
                    "message": "Worker 未通过完成门",
                    "recoverable": True,
                })

        requested_replan = [
            result.task_id for result in collected
            if result.status == AgentTaskStatus.SUCCESS and result.needs_replan
        ]
        failed_ids = list(dict.fromkeys(missing + non_terminal + failed + partial + requested_replan))
        if not collected:
            return ReviewVerdict(
                outcome=ReviewOutcome.CLARIFY,
                clarification="没有可验收的 Worker 结果",
                feedback=feedback,
                summary="缺少执行结果",
            )
        clarification_results = [
            result for result in collected
            if bool((result.output_data.get("conversation_result") or {}).get("cannot_answer"))
        ]
        if clarification_results:
            conversation = clarification_results[-1].output_data.get("conversation_result") or {}
            return ReviewVerdict(
                outcome=ReviewOutcome.CLARIFY,
                failed_task_ids=[result.task_id for result in clarification_results],
                clarification=str(
                    conversation.get("cannot_answer_reason")
                    or conversation.get("answer")
                    or "需要用户补充上下文"
                ),
                final_output=final_output or {},
                summary="回答需要用户澄清",
            )
        if non_terminal:
            return ReviewVerdict(
                outcome=ReviewOutcome.FAIL,
                failed_task_ids=failed_ids,
                feedback=feedback,
                final_output=final_output or {},
                summary="存在非终态或已取消任务",
            )
        if missing or failed:
            recoverable = any(bool(item.get("recoverable")) for item in feedback)
            if recoverable and plan.replan_count < 1:
                outcome = ReviewOutcome.REPLAN
            else:
                outcome = ReviewOutcome.FAIL
            return ReviewVerdict(
                outcome=outcome,
                failed_task_ids=failed_ids,
                repair_scope=failed_ids,
                feedback=feedback,
                final_output=final_output or {},
                summary="存在缺失或失败任务",
            )
        if requested_replan:
            outcome = ReviewOutcome.REPLAN if plan.replan_count < 1 else ReviewOutcome.FAIL
            return ReviewVerdict(
                outcome=outcome,
                failed_task_ids=failed_ids,
                feedback=feedback,
                final_output=final_output or {},
                summary="SUCCESS 结果声明 needs_replan，不能通过验收",
            )
        if partial:
            profiles = {result.profile for result in collected if result.task_id in partial}
            metrics = dict((final_output or {}).get("eval_metrics") or {})
            hard_failed = [name for name in self.HARD_METRICS if metrics.get(name) is False]
            usable_report = bool(str((final_output or {}).get("final_report") or "").strip())
            if (
                profiles == {WorkerProfile.SEARCH}
                and plan.replan_count >= 1
                and usable_report
                and not hard_failed
            ):
                # 关键步骤：单个检索分支持续失败但其余来源足以形成完整报告时，
                # Reviewer 明确给出受控降级通过，而不是由 Finalizer 忽略 fail。
                feedback.append({
                    "task_id": ",".join(partial),
                    "code": "PARTIAL_SEARCH_DEGRADED",
                    "message": "检索分支已耗尽重规划，但现有来源与硬指标足以交付",
                    "recoverable": False,
                })
                output = dict(final_output or {})
                output["degraded_partial_search"] = True
                return ReviewVerdict(
                    outcome=ReviewOutcome.PASS,
                    failed_task_ids=failed_ids,
                    feedback=feedback,
                    final_output=output,
                    summary="部分检索分支持续失败，使用已验收来源受控交付",
                )
            if profiles.intersection({WorkerProfile.DIRECT, WorkerProfile.SEARCH}):
                outcome = ReviewOutcome.REPLAN if plan.replan_count < 1 else ReviewOutcome.FAIL
            else:
                outcome = ReviewOutcome.REPAIR if plan.repair_count < 1 else ReviewOutcome.FAIL
            return ReviewVerdict(
                outcome=outcome,
                failed_task_ids=failed_ids,
                repair_scope=failed_ids,
                feedback=feedback,
                final_output=final_output or {},
                summary="部分任务需要一次局部修复",
            )
        expected_chapters = (final_output or {}).get("expected_chapter_count")
        written_chapters = (final_output or {}).get("written_chapter_count")
        if expected_chapters is not None and written_chapters != expected_chapters:
            feedback.append({
                "task_id": "write", "code": "CHAPTER_SET_INCOMPLETE",
                "message": f"章节数量不一致: expected={expected_chapters}, actual={written_chapters}",
                "recoverable": plan.repair_count < 1,
            })
            outcome = ReviewOutcome.REPAIR if plan.repair_count < 1 else ReviewOutcome.FAIL
            return ReviewVerdict(
                outcome=outcome, failed_task_ids=["write"], repair_scope=["write"],
                feedback=feedback, final_output=final_output or {},
                summary="Writing Worker 未完整覆盖 outline",
            )
        degraded_source_only = any(
            item.profile == WorkerProfile.WRITE
            and bool(item.metadata.get("degraded_source_only"))
            for item in plan.items
        )
        if degraded_source_only:
            report = str((final_output or {}).get("final_report") or "").strip()
            output = dict(final_output or {})
            output["degraded_source_only"] = True
            if report:
                return ReviewVerdict(
                    outcome=ReviewOutcome.CLARIFY,
                    failed_task_ids=["analyze", "cite"],
                    feedback=feedback + [{
                        "task_id": "review", "code": "DEGRADED_SOURCE_ONLY",
                        "message": "来源级降级计划不具备完整证据链，不能通过验收",
                        "recoverable": False,
                    }],
                    final_output=output,
                    clarification="证据抽取持续失败；已生成来源级草稿，请确认是否接受部分结果。",
                    summary="来源可用但缺少完整证据链，仅能部分交付",
                )
            return ReviewVerdict(
                outcome=ReviewOutcome.FAIL,
                failed_task_ids=["write"],
                feedback=feedback,
                final_output=output,
                summary="来源级降级计划没有产生可交付正文",
            )
        metrics = dict((final_output or {}).get("eval_metrics") or {})
        hard_failed = [name for name in self.HARD_METRICS if metrics.get(name) is False]
        if hard_failed:
            feedback.extend({
                "task_id": "review",
                "code": "HARD_METRIC_FAILED",
                "metric": name,
                "message": f"硬指标未通过: {name}",
                "recoverable": plan.repair_count < 1,
            } for name in hard_failed)
            outcome = ReviewOutcome.REPAIR if plan.repair_count < 1 else ReviewOutcome.FAIL
            return ReviewVerdict(
                outcome=outcome,
                failed_task_ids=["analyze", "cite", "write"],
                repair_scope=["analyze", "cite", "write"],
                feedback=feedback,
                final_output=final_output or {},
                summary="硬指标未通过",
            )
        return ReviewVerdict(
            outcome=ReviewOutcome.PASS,
            final_output=final_output or self._last_output(collected),
            summary="全部计划任务通过验收",
        )

    @staticmethod
    def _last_output(results: List[WorkerResult]) -> Dict[str, Any]:
        return dict(results[-1].output_data) if results else {}

    # 兼容旧调用方的命名入口，内部始终落到统一 review。
    review_direct = review
    evaluate = review
    review_final = review

    def evaluate_and_review(
        self,
        plan: WorkPlan,
        results: Iterable[WorkerResult],
        *,
        evaluator_input: Dict[str, Any],
        final_output: Dict[str, Any] | None = None,
    ) -> ReviewVerdict:
        """在统一 Reviewer 内运行现有 Evaluator，再生成最终 verdict。"""
        evaluation = self.evaluator.evaluate(**evaluator_input)
        payload = dict(final_output or {})
        payload["eval_metrics"] = dict(evaluation.get("metrics") or {})
        payload["eval_metric_details"] = dict(evaluation.get("metrics_detail") or {})
        payload["eval_feedback"] = list(evaluation.get("feedback") or [])
        return self.review(plan, results, final_output=payload)
