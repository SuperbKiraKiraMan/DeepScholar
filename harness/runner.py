"""
Harness 运行器 —— 对现有 Agent Runtime 执行单个用例或完整套件。
通过 contextvar 注入观察者 + Runtime 权威 eval_metrics。
"""

import copy
import time
from typing import Any, Dict, List, Optional, Set

from app.graph.runtime import run_graph
from app.services.orchestrator import Orchestrator

from harness.models import (
    CaseResult, ExpectationResult, HarnessCase, MetricAssertion, SuiteResult,
)
from harness.hooks import HookBus
from harness.fixtures import FixtureManager
from harness.metrics import METRIC_NAMES, assert_metric
from app.observability.lifecycle import emit_after_run, reset_observer, set_observer

# ---- 非真实工具调用的事件名（trace 里混杂了 graph 节点事件和工具事件，需要过滤） ----
_NON_TOOL_NAMES: Set[str] = {
    "orchestrator", "",
    "function_call_started", "tool_selected", "tool_started", "tool_finished",
    "tool_args_rejected", "tool_loop_finished", "tool_loop_limit_reached",
    "tool_loop_fallback", "tool_rejected", "llm_finish", "llm_function_call",
    "controller_start", "planner_complete", "send_dispatch", "merge_result",
    "search_complete", "reading_complete", "analysis_complete", "citation_complete",
    "draft_reviewer_complete", "evaluator_complete", "final_reviewer_complete",
    "worker_started", "worker_finished",
    "retrieval_observed", "retrieval_query_rewritten",
    "retrieval_source_switched", "retrieval_finished",
}


def _extract_real_tools(trace: List[Dict]) -> List[str]:
    """Extract real tool names from trace, excluding event-type names."""
    seen: Set[str] = set()
    for t in trace:
        name = t.get("tool_name", "").strip()
        if not name or name in _NON_TOOL_NAMES:
            continue
        if name.startswith("tool_") or name.startswith("llm_"):
            continue
        seen.add(name)
    return sorted(seen)


def _is_real_tool_trace(event: Dict[str, Any]) -> bool:
    name = str(event.get("tool_name", ""))
    if not name or name in _NON_TOOL_NAMES or name.startswith(("tool_", "llm_")):
        return False
    event_type = event.get("event", "")
    return event_type == "tool_finished" or (not event_type and "success" in event)


class CaseRunner:
    """Execute one HarnessCase against the real Runtime. 执行单个测试用例"""

    def __init__(self, hook_bus: HookBus = None):
        self.hooks = hook_bus or HookBus()
        self.fixtures = FixtureManager()

    async def run(self, case: HarnessCase) -> CaseResult:
        errors: List[str] = []
        llm_diagnostics: Dict[str, Any] = {}

        # ---- 1. 注入故障：替换 LLM 客户端、Hook 工具方法、设置环境变量 ----
        self.fixtures.install(case.fixture_profile, case.request.agent_mode)

        # ---- 2. 注入生命周期观察者（记录 before_run / after_tool / on_error 等事件） ----
        token = set_observer(self.hooks)

        result: Dict[str, Any] = {}
        start_ms = int(time.time() * 1000)
        try:
            # ---- 3. 真正跑 Agent pipeline（和线上完全一样） ----
            orchestrator = Orchestrator()
            if case.backend == "loop":
                result = await orchestrator.run(
                    topic=case.request.topic,
                    max_sources=case.request.max_sources,
                    language=case.request.language,
                    mode=case.request.mode,
                    run_eval=case.request.run_eval,
                )
            else:
                #  目前是串行运行，可以改成并行运行,谁先完成谁就返回
                result = await run_graph(
                    topic=case.request.topic,
                    max_sources=case.request.max_sources,
                    language=case.request.language,
                    mode=case.request.mode,
                    run_eval=case.request.run_eval,
                    agent_mode=case.request.agent_mode,
                )
        except Exception as e:
            errors.append(f"Run exception: {type(e).__name__}: {str(e)[:200]}")
            self.hooks.on_error({
                "stage": "run", "exception_type": type(e).__name__,
                "error": str(e)[:200],
            })
        finally:
            # ---- 无论如何都要恢复：去掉故障注入、还原 LLM 客户端、清理观察者 ----
            total_latency_ms = int(time.time() * 1000) - start_ms
            # 触发 after_run 事件
            emit_after_run({
                "case_id": case.id,
                "run_id": result.get("run_id", ""),
                "status": result.get("status", "error"),
                "total_latency_ms": total_latency_ms,
                "retry_count": result.get("retry_count", 0),
                "replan_count": result.get("replan_count", 0),
            })
            # 清理观察者
            reset_observer(token)
            self.fixtures.restore()     # ← 还原所有 monkey-patch
            llm_diagnostics = copy.deepcopy(self.fixtures.last_llm_diagnostics)

        # ---- 4. 从 Agent 返回结果中提取所有数据 ----
        # 3. Extract from agent output
        run_id = result.get("run_id", "")
        status = result.get("status", "error")
        trace = result.get("trace", [])
        sources = result.get("sources", [])
        citation_check_results = result.get("citation_check_results", [])
        final_report = result.get("final_report", "")
        draft_report = result.get("draft_report", "")
        task_dag = result.get("task_dag", {})
        warnings = result.get("warnings", [])
        unresolved = result.get("unresolved_issues", [])
        fixes_applied = result.get("fixes_applied", [])
        retry_count = result.get("retry_count", 0)
        replan_count = result.get("replan_count", 0)

        # ---- 5. 用 Runtime 的权威 eval_metrics（不复算，直接复用线上逻辑） ----
        eval_metrics = result.get("eval_metrics", {})
        eval_metric_details = result.get("eval_metric_details", {})

        # ---- 6. 执行所有断言：指标 / trace / 工具 / retry / hook ----
        expectation_results = self._assert_expectations(
            case, trace, eval_metrics, status, retry_count, replan_count,
            sources, fixes_applied, unresolved, self.hooks.records,
            llm_diagnostics,
        )

        # 6. Build CaseResult
        tool_names = _extract_real_tools(trace)
        trace_event_names = sorted(set(
            t.get("event", "") for t in trace if t.get("event")
        ))

        # ---- 全部通过的条件：每条断言通过 + 最终状态匹配 + 无异常 ----
        all_passed = (all(r.passed for r in expectation_results)
                      and status == case.expected_status
                      and not errors)

        # Merge hook warnings into case warnings
        all_warnings = list(warnings) + list(self.hooks.warnings)

        return CaseResult(
            case_id=case.id, description=case.description,
            passed=all_passed, status=status,
            expected_status=case.expected_status,
            run_id=run_id, backend=case.backend,
            agent_mode=case.request.agent_mode,
            total_latency_ms=total_latency_ms,
            retry_count=retry_count, replan_count=replan_count,
            eval_metrics=eval_metrics,
            eval_metric_details=eval_metric_details,
            expectation_results=expectation_results,
            tools_called=tool_names, trace_events=trace_event_names,
            warnings=all_warnings, unresolved_issues=unresolved,
            fixes_applied=fixes_applied,
            hooks=list(self.hooks.records),
            error="; ".join(errors) if errors else "",
            llm_diagnostics=llm_diagnostics,
        )

    # 执行所有断言：指标 / trace / 工具 / retry / hook
    def _assert_expectations(
        self, case: HarnessCase, trace: List[Dict], metrics: Dict[str, Any],
        status: str, retry_count: int, replan_count: int,
        sources: List[Dict], fixes_applied: List[str], unresolved: List[str],
        hooks: List[Any], llm_diagnostics: Dict[str, Any],
    ) -> List[ExpectationResult]:
        results: List[ExpectationResult] = []

        # ---- 6a. Rule Metrics 指标断言（eq/gte/lte/gt/lt/ne） ----
        for ma in case.expected_metrics:
            actual = metrics.get(ma.name)
            r = assert_metric(ma.name, ma.expected, actual, ma.op)
            results.append(ExpectationResult(
                name=ma.name, expected=ma.expected, actual=actual,
                passed=r["passed"], reason=r["reason"],
            ))

        # ---- 6b. 必需的 trace 事件（如 controller_start、citation_complete） ----
        trace_event_names = {t.get("event", "") for t in trace if t.get("event")}
        for req_ev in case.required_trace_events:
            passed = req_ev in trace_event_names
            results.append(ExpectationResult(
                name=f"required_trace:{req_ev}",
                expected="present", actual="present" if passed else "missing",
                passed=passed,
                reason="" if passed else f"Trace event '{req_ev}' not found",
            ))

        # ---- 6c. 禁止的 trace 事件（如不应出现 tool_loop_fallback） ----
        for forb_ev in case.forbidden_trace_events:
            present = forb_ev in trace_event_names
            results.append(ExpectationResult(
                name=f"forbidden_trace:{forb_ev}",
                expected="absent", actual="present" if present else "absent",
                passed=not present,
                reason=f"Forbidden event '{forb_ev}' found" if present else "",
            ))

        # ---- 6d. 协议角色必须通过真实 Runtime 的输入、权限与输出校验 ----
        protocol_roles = {
            str(event.get("agent_role") or "")
            for event in trace
            if event.get("event") == "agent_protocol_validated"
        }
        for role in case.required_protocol_roles:
            passed = role in protocol_roles
            results.append(ExpectationResult(
                name=f"required_protocol_role:{role}",
                expected="validated",
                actual="validated" if passed else "missing",
                passed=passed,
                reason="" if passed else f"Agent role '{role}' was not protocol-validated",
            ))

        # ---- 6e. 必须调用的工具（如 mock_academic_search 一定要被执行） ----
        tool_names = set(_extract_real_tools(trace))
        for exp_tool in case.expected_tools:
            passed = exp_tool in tool_names
            results.append(ExpectationResult(
                name=f"expected_tool:{exp_tool}",
                expected="called", actual="called" if passed else "not called",
                passed=passed,
                reason="" if passed else f"Tool '{exp_tool}' not called",
            ))

        # ---- 6f. retry / replan 次数必须在 [min, max] 范围内 ----
        results.append(ExpectationResult(
            name="max_retry_count",
            expected=f"≤{case.max_retry_count}", actual=retry_count,
            passed=retry_count <= case.max_retry_count,
            reason="" if retry_count <= case.max_retry_count
            else f"retry={retry_count} > max={case.max_retry_count}",
        ))
        results.append(ExpectationResult(
            name="min_retry_count",
            expected=f"≥{case.min_retry_count}", actual=retry_count,
            passed=retry_count >= case.min_retry_count,
            reason="" if retry_count >= case.min_retry_count
            else f"retry={retry_count} < min={case.min_retry_count}",
        ))

        # Replan
        results.append(ExpectationResult(
            name="max_replan_count",
            expected=f"≤{case.max_replan_count}", actual=replan_count,
            passed=replan_count <= case.max_replan_count,
            reason="" if replan_count <= case.max_replan_count
            else f"replan={replan_count} > max={case.max_replan_count}",
        ))
        results.append(ExpectationResult(
            name="min_replan_count",
            expected=f"≥{case.min_replan_count}", actual=replan_count,
            passed=replan_count >= case.min_replan_count,
            reason="" if replan_count >= case.min_replan_count
            else f"replan={replan_count} < min={case.min_replan_count}",
        ))

        worker_finishes = [event for event in trace if event.get("event") == "worker_finished"]
        failed_workers = sum(1 for event in worker_finishes if not event.get("success", False))
        successful_workers = sum(1 for event in worker_finishes if event.get("success", False))
        results.extend([
            ExpectationResult(
                name="min_failed_workers", expected=f"≥{case.min_failed_workers}",
                actual=failed_workers, passed=failed_workers >= case.min_failed_workers,
                reason="" if failed_workers >= case.min_failed_workers
                else f"failed_workers={failed_workers} < min={case.min_failed_workers}",
            ),
            ExpectationResult(
                name="min_successful_workers", expected=f"≥{case.min_successful_workers}",
                actual=successful_workers, passed=successful_workers >= case.min_successful_workers,
                reason="" if successful_workers >= case.min_successful_workers
                else f"successful_workers={successful_workers} < min={case.min_successful_workers}",
            ),
        ])

        real_tool_events = [event for event in trace if _is_real_tool_trace(event)]
        failed_tools = sum(1 for event in real_tool_events if not event.get("success", False))
        results.append(ExpectationResult(
            name="min_failed_tools", expected=f"≥{case.min_failed_tools}",
            actual=failed_tools, passed=failed_tools >= case.min_failed_tools,
            reason="" if failed_tools >= case.min_failed_tools
            else f"failed_tools={failed_tools} < min={case.min_failed_tools}",
        ))

        # 检查所有必须的 Hook 都被调用
        hook_names = [hook.stage for hook in hooks]
        for hook_name in case.required_hooks:
            present = hook_name in hook_names
            results.append(ExpectationResult(
                name=f"required_hook:{hook_name}", expected="present",
                actual="present" if present else "missing", passed=present,
                reason="" if present else f"Hook '{hook_name}' not emitted",
            ))
        on_error_count = hook_names.count("on_error")
        results.append(ExpectationResult(
            name="min_on_error_hooks", expected=f"≥{case.min_on_error_hooks}",
            actual=on_error_count, passed=on_error_count >= case.min_on_error_hooks,
            reason="" if on_error_count >= case.min_on_error_hooks
            else f"on_error_hooks={on_error_count} < min={case.min_on_error_hooks}",
        ))

        if case.request.agent_mode == "llm":
            # 关键步骤：LLM fallback 只能在目标 Worker 通道失败；其他阶段的 fixture 耗尽必须显式失败。
            exhaustion_count = int(llm_diagnostics.get("exhaustion_count", 0))
            results.append(ExpectationResult(
                name="fixture_no_unexpected_llm_response_exhaustion",
                expected=0,
                actual=exhaustion_count,
                passed=exhaustion_count == 0,
                reason="" if exhaustion_count == 0
                else f"FakeLLM fixture exhausted {exhaustion_count} time(s): "
                f"{llm_diagnostics.get('exhaustions', [])[:2]}",
            ))

        if case.expect_sources_empty is not None:
            is_empty = not sources
            results.append(ExpectationResult(
                name="sources_empty", expected=case.expect_sources_empty,
                actual=is_empty, passed=is_empty == case.expect_sources_empty,
                reason="" if is_empty == case.expect_sources_empty
                else f"Expected sources_empty={case.expect_sources_empty}, got {is_empty}",
            ))

        if case.require_citation_recovery:
            checks = [event for event in trace if event.get("event") == "citation_complete"]
            saw_failure = any(
                event.get("total_checked", 0) > event.get("valid_count", 0)
                for event in checks
            )
            saw_recovery = saw_failure and any(
                event.get("total_checked", 0) > 0
                and event.get("total_checked") == event.get("valid_count")
                for event in checks[1:]
            )
            results.append(ExpectationResult(
                name="citation_failure_then_recovery", expected=True,
                actual=saw_recovery, passed=saw_recovery,
                reason="" if saw_recovery else "No failed citation check followed by recovery",
            ))

        # Expected status
        results.append(ExpectationResult(
            name="expected_status",
            expected=case.expected_status, actual=status,
            passed=status == case.expected_status,
            reason="" if status == case.expected_status
            else f"Expected '{case.expected_status}', got '{status}'",
        ))

        return results


# ================================================================
# Suite Runner 遍历每个 Case 并执行测试
# ================================================================

class SuiteRunner:
    """Execute a list of HarnessCases and produce a SuiteResult."""

    def __init__(self, cases: List[HarnessCase], suite_name: str = "Agent Harness Suite"):
        # Deep copy cases to avoid mutation of globals
        self.cases = [copy.deepcopy(c) for c in cases]
        self.suite_name = suite_name

    def validate(self) -> List[str]:
        errors = []
        seen_ids: Set[str] = set()
        # 遍历每个测试用例，检查 ID 是否唯一
        for c in self.cases:
            if not c.id or not c.id.strip():
                errors.append("Empty case id")
            elif c.id in seen_ids:
                errors.append(f"Duplicate case id: '{c.id}'")
            seen_ids.add(c.id)
            if c.max_retry_count < 0:
                errors.append(f"Case '{c.id}': max_retry_count cannot be negative")
            if c.max_replan_count < 0:
                errors.append(f"Case '{c.id}': max_replan_count cannot be negative")
            for ma in c.expected_metrics:
                if ma.name not in METRIC_NAMES:
                    errors.append(f"Case '{c.id}': unknown metric '{ma.name}'")
            try:
                HarnessCase.model_validate(c.model_dump())
            except Exception as e:
                errors.append(f"Case '{c.id}' validation: {e}")
        return errors

    # suite runner 运行所有测试用例
    async def run(self) -> SuiteResult:
        # Fixture 使用进程级环境变量和类级 monkey-patch 注入故障。
        # Case 必须顺序运行，否则一个 Case 的故障会污染另一个 Case。
        async def run_one(case: HarnessCase) -> CaseResult:
            hook_bus = HookBus()
            try:
                return await CaseRunner(hook_bus=hook_bus).run(case)
            except Exception as e:
                return CaseResult(
                    case_id=case.id, description=case.description,
                    passed=False, status="error",
                    expected_status=case.expected_status,
                    backend=case.backend, agent_mode=case.request.agent_mode,
                    error=f"Runner exception: {type(e).__name__}: {str(e)[:200]}",
                )

        results: List[CaseResult] = []
        for case in self.cases:
            results.append(await run_one(case))

        passed = sum(1 for r in results if r.passed) # 统计通过的测试用例数量
        failed = len(results) - passed # 统计失败的测试用例数量

        # Metric pass rates
        metric_counts: Dict[str, int] = {}
        metric_passes: Dict[str, int] = {}
        for r in results:
            for er in r.expectation_results:
                metric_counts[er.name] = metric_counts.get(er.name, 0) + 1
                if er.passed:
                    metric_passes[er.name] = metric_passes.get(er.name, 0) + 1
        metric_pass_rates = {
            n: round(metric_passes.get(n, 0) / max(metric_counts[n], 1), 4)
            for n in metric_counts
        }

        # Hook 总计
        hook_summary: Dict[str, int] = {}
        for r in results:
            for h in r.hooks:
                key = f"{h.stage}"
                hook_summary[key] = hook_summary.get(key, 0) + 1

        # Backend/agent_mode info
        backends = sorted(set(r.backend for r in results if r.backend))
        agent_modes = sorted(set(r.agent_mode for r in results if r.agent_mode))
        backend_str = ", ".join(backends) if backends else "mixed"
        agent_str = ", ".join(agent_modes) if agent_modes else "mixed"

        return SuiteResult(
            suite_name=self.suite_name,
            backend=backend_str,
            agent_mode=agent_str,
            total_cases=len(self.cases),
            passed_cases=passed,
            failed_cases=failed,
            results=results,
            metric_pass_rates=metric_pass_rates,
            hook_summary=hook_summary,
            known_limitations=[
                "Mock tool data only; OpenAlex provider available via SEARCH_PROVIDER=openalex",
                "FakeLLMClient for LLM mode tests (no real DeepSeek calls)",
                "Loop backend does not support Send API parallelism",
                "LLM planner fallback can expose task-id mismatches in the fixed graph worker path",
                "Latency thresholds may vary in CI environments",
            ],
        )
