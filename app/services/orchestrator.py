"""
app/services/orchestrator.py

Orchestrator —— Agent MVP 编排器。

串联 Planner、Worker、Draft Reviewer、Evaluator、Final Reviewer。

支持评估后的有界重试回路。
- 如果 Evaluator 发现 citation 相关指标失败，
  Orchestrator 会重新执行 analyze + cite 阶段（最多 1 次重试），
  尝试通过不同的证据抽取消除假引用。
- 重试后仍失败的指标交由 FinalReviewer 处理：
  能删除的删除，删不掉的列入 UNRESOLVED ISSUES。

设计原则：
- 最多重试 1 次（不是无限循环）
- 只为可修复的指标重试（citation 相关），
  不可修复的（min_sources / latency）不浪费重试次数
- 重试不是假装修好——修不好就诚实说修不好
"""

import time
from typing import Any, Dict, List

from app.agents.planner import Planner
from app.agents.worker import Worker, WorkerContext
from app.agents.draft_reviewer import DraftReviewer
from app.agents.evaluator import Evaluator
from app.agents.final_reviewer import FinalReviewer
from app.services.run_store import run_store
from app.observability.lifecycle import emit_after_plan, emit_before_run, emit_error
from app.observability.metrics import aggregate_run_metrics


# 触发重试的指标：这些指标失败意味着 analyze/cite 阶段可能可以改进
_RETRY_TRIGGER_METRICS = {"no_fake_citation", "citation_id_exists", "source_url_valid"}

MAX_RETRIES = 1


class Orchestrator:
    """
    Agent MVP 编排器，含评估驱动的重试回路。
    """

    def __init__(self):
        self.planner = Planner()
        self.worker = Worker()
        self.draft_reviewer = DraftReviewer()
        self.evaluator = Evaluator()
        self.final_reviewer = FinalReviewer()

    async def run(
        self,
        topic: str,
        max_sources: int = 5,
        language: str = "zh",
        mode: str = "quick",
        run_eval: bool = True,
        run_id: str = None,
    ) -> Dict[str, Any]:
        """
        执行完整的 Agent 调研流程（含重试回路）。

        返回：完整的 run 结果字典
        """
        max_sources = max(1, min(int(max_sources), 50))
        start_time = time.time()
        if run_id is None:
            run_id = run_store.create(topic=topic)
        elif run_store.get(run_id) is None:
            run_store.create(topic=topic, run_id=run_id)
        else:
            run_store.update(run_id, topic=topic, status="running")
        all_trace: List[Dict[str, Any]] = []
        retry_attempted = False
        all_sources: List[Dict[str, Any]] = []
        all_evidence_cards: List[Dict[str, Any]] = []
        citation_check_results: List[Dict[str, Any]] = []
        all_warnings: List[str] = []
        emit_before_run({"run_id": run_id, "topic": topic, "node": "orchestrator"})

        try:
            # ---- 1. Planner ----
            task_dag = self.planner.plan(topic=topic, max_sources=max_sources)
            emit_after_plan({
                "task_count": len(task_dag.tasks),
                "task_ids": [task.task_id for task in task_dag.tasks],
                "dependencies": {task.task_id: task.depends_on for task in task_dag.tasks},
                "tool_plan": {task.task_id: task.tool_plan for task in task_dag.tasks},
            })
            run_store.update(run_id, task_dag=task_dag.to_dict())

            # ---- 2. 首次 Worker 执行（search → read → analyze → cite） ----
            dep_results, all_sources, all_evidence_cards = await self._run_all_tasks(
                task_dag, {}
            )
            for ctx in dep_results.values():
                all_trace.extend(ctx.trace)

            # ---- 3. 首次评估 ----
            cite_ctx = dep_results.get("cite")
            citation_check_results, citation_summary = self._extract_citation_results(cite_ctx)
            draft_report, draft_warnings = self._run_draft_review(
                topic, all_sources, all_evidence_cards,
                citation_check_results, citation_summary, language,
            )

            eval_result = {}
            if run_eval:
                eval_result = self.evaluator.evaluate(
                    topic=topic,
                    draft_report=draft_report,
                    sources=all_sources,
                    evidence_cards=all_evidence_cards,
                    citation_check_results=citation_check_results,
                    citation_summary=citation_summary,
                    trace=all_trace,
                    task_dag=task_dag.to_dict(),
                    total_latency_ms=int((time.time() - start_time) * 1000),
                )

            # ---- 4. 重试回路：如果 citation 相关指标失败，重新 analyze + cite ----
            if run_eval and self._should_retry(eval_result):
                retry_attempted = True
                all_trace.append({
                    "step": "retry_triggered",
                    "task_id": "orchestrator",
                    "tool_name": "orchestrator",
                    "input_summary": f"Retrying analyze+cite due to failed metrics: "
                                     f"{self._failed_retry_triggers(eval_result)}",
                    "success": True,
                    "latency_ms": 0,
                    "error": None,
                })

                # 只重新执行 analyze + cite（search 和 read 结果复用）
                retry_dep_results = dict(dep_results)
                # 清除旧的 analyze 和 cite 结果，让它们重新执行
                retry_dep_results.pop("analyze", None)
                retry_dep_results.pop("cite", None)

                retry_results, _, retry_cards = await self._run_analyze_and_cite(
                    task_dag, retry_dep_results
                )
                for ctx in retry_results.values():
                    all_trace.extend(ctx.trace)

                # 更新结果
                all_evidence_cards = retry_cards or all_evidence_cards
                retry_cite_ctx = retry_results.get("cite")
                if retry_cite_ctx:
                    citation_check_results, citation_summary = self._extract_citation_results(
                        retry_cite_ctx
                    )

                # 用新结果重新生成 draft_report
                draft_report, draft_warnings = self._run_draft_review(
                    topic, all_sources, all_evidence_cards,
                    citation_check_results, citation_summary, language,
                )

                # 重新评估
                eval_result = self.evaluator.evaluate(
                    topic=topic,
                    draft_report=draft_report,
                    sources=all_sources,
                    evidence_cards=all_evidence_cards,
                    citation_check_results=citation_check_results,
                    citation_summary=citation_summary,
                    trace=all_trace,
                    task_dag=task_dag.to_dict(),
                    total_latency_ms=int((time.time() - start_time) * 1000),
                )

            # ---- 5. 汇总 warnings ----
            all_warnings = list(draft_warnings)
            for ctx in dep_results.values():
                all_warnings.extend(ctx.warnings)

            # ---- 6. Final Reviewer: 实际修复 ----
            eval_feedback = eval_result.get("feedback", [])
            eval_metrics = eval_result.get("metrics", {})

            final_result = self.final_reviewer.review(
                draft_report=draft_report,
                eval_metrics=eval_result,
                eval_feedback=eval_feedback,
                citation_check_results=citation_check_results,
                evidence_cards=all_evidence_cards,
                sources=all_sources,
                warnings=all_warnings,
                language=language,
            )
            final_report = final_result["final_report"]
            all_warnings = final_result.get("warnings", all_warnings)
            fixes_applied = final_result.get("fixes_applied", [])
            unresolved_issues = final_result.get("unresolved_issues", [])

            # ---- 7. 构建来源矩阵 + 更新 RunStore ----
            source_matrix = self._build_source_matrix(all_sources, all_evidence_cards)
            total_latency_ms = int((time.time() - start_time) * 1000)
            observability_metrics = aggregate_run_metrics(
                all_trace,
                total_latency_ms=total_latency_ms,
                status="completed",
                backend="loop",
                agent_mode="rule",
                retry_count=int(retry_attempted),
                warnings=all_warnings,
                sources=all_sources,
                evidence_cards=all_evidence_cards,
                citation_check_results=citation_check_results,
            )

            run_store.update(
                run_id,
                status="completed",
                final_report=final_report,
                draft_report=draft_report,
                sources=all_sources,
                evidence_cards=all_evidence_cards,
                citation_check_results=citation_check_results,
                citation_summary=citation_summary,
                source_matrix=source_matrix,
                eval_metrics=eval_metrics,
                eval_metric_details=eval_result.get("metrics_detail", {}),
                eval_feedback=eval_feedback,
                fixes_applied=fixes_applied,
                unresolved_issues=unresolved_issues,
                warnings=all_warnings,
                trace=all_trace,
                retry_attempted=retry_attempted,
                latency_ms=total_latency_ms,
                total_latency_ms=total_latency_ms,
                observability_metrics=observability_metrics,
            )
            run_store.finish(run_id, "completed")

        except Exception as e:
            emit_error({
                "stage": "orchestrator",
                "run_id": run_id,
                "exception_type": type(e).__name__,
                "error": str(e)[:200],
            })
            total_latency_ms = int((time.time() - start_time) * 1000)
            observability_metrics = aggregate_run_metrics(
                all_trace,
                total_latency_ms=total_latency_ms,
                status="failed",
                backend="loop",
                agent_mode="rule",
                retry_count=int(retry_attempted),
                warnings=all_warnings or [f"Orchestrator error: {str(e)}"],
                sources=all_sources,
                evidence_cards=all_evidence_cards,
                citation_check_results=citation_check_results,
            )
            run_store.update(
                run_id,
                status="failed",
                warnings=[f"Orchestrator error: {str(e)}"],
                latency_ms=total_latency_ms,
                total_latency_ms=total_latency_ms,
                trace=all_trace,
                observability_metrics=observability_metrics,
            )
            run_store.finish(run_id, "failed")

        return run_store.get(run_id) or {"run_id": run_id, "status": "error", "topic": topic}

    # ================================================================
    # Worker 执行
    # ================================================================

    async def _run_all_tasks(
        self,
        task_dag,
        initial_deps: Dict[str, WorkerContext],
    ) -> tuple:
        """执行 Task DAG 中所有任务（search → read → analyze → cite）。"""
        completed: set = set()
        dep_results: Dict[str, WorkerContext] = dict(initial_deps)
        all_sources: List[Dict[str, Any]] = []
        all_cards: List[Dict[str, Any]] = []

        for task_id in initial_deps:
            completed.add(task_id)

        while True:
            ready = task_dag.get_ready_tasks(completed)
            if not ready:
                break

            for task in ready:
                ctx = await self.worker.execute_task(task, dep_results)
                dep_results[task.task_id] = ctx
                completed.add(task.task_id)

                if task.task_id == "read":
                    all_sources = ctx.results.get("scored_sources", [])
                elif task.task_id == "analyze":
                    all_cards = ctx.results.get("evidence_cards", [])

        return dep_results, all_sources, all_cards

    async def _run_analyze_and_cite(
        self,
        task_dag,
        dep_results: Dict[str, WorkerContext],
    ) -> tuple:
        """只重新执行 analyze 和 cite 任务。"""
        completed = set(dep_results.keys())
        new_results: Dict[str, WorkerContext] = {}
        all_cards: List[Dict[str, Any]] = []

        for task in task_dag.tasks:
            if task.task_id in ("analyze", "cite"):
                if task.task_id in completed:
                    completed.discard(task.task_id)

        while True:
            ready = task_dag.get_ready_tasks(completed)
            if not ready:
                break

            for task in ready:
                ctx = await self.worker.execute_task(task, dep_results)
                new_results[task.task_id] = ctx
                dep_results[task.task_id] = ctx
                completed.add(task.task_id)

                if task.task_id == "analyze":
                    all_cards = ctx.results.get("evidence_cards", [])

        return new_results, [], all_cards

    # ================================================================
    # 重试决策
    # ================================================================

    def _should_retry(self, eval_result: Dict[str, Any]) -> bool:
        """判断是否需要重试：是否有 citation 相关指标失败。"""
        metrics = eval_result.get("metrics", {})
        for key in _RETRY_TRIGGER_METRICS:
            if not metrics.get(key, True):
                return True
        return False

    def _failed_retry_triggers(self, eval_result: Dict[str, Any]) -> List[str]:
        """列出触发重试的失败指标名。"""
        metrics = eval_result.get("metrics", {})
        return [k for k in _RETRY_TRIGGER_METRICS if not metrics.get(k, True)]

    # ================================================================
    # 辅助方法
    # ================================================================

    def _extract_citation_results(self, cite_ctx) -> tuple:
        """从 cite worker context 提取引用校验结果。"""
        if cite_ctx is None:
            return [], {}
        return (
            cite_ctx.results.get("citation_check_results", []),
            cite_ctx.results.get("citation_summary", {}),
        )

    def _run_draft_review(
        self,
        topic: str,
        sources: List[Dict],
        evidence_cards: List[Dict],
        citation_check_results: List[Dict],
        citation_summary: Dict,
        language: str = "zh",
    ) -> tuple:
        """运行 Draft Reviewer。"""
        result = self.draft_reviewer.review(
            topic=topic,
            sources=sources,
            evidence_cards=evidence_cards,
            citation_check_results=citation_check_results,
            citation_summary=citation_summary,
            language=language,
        )
        return result["draft_report"], result.get("warnings", [])

    def _build_source_matrix(
        self,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """构建来源矩阵。"""
        matrix = []
        for s in sources:
            sid = s.get("source_id", "")
            card_count = sum(1 for c in evidence_cards if c.get("source_id") == sid)

            authors = s.get("authors", [])
            if isinstance(authors, list):
                author_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    author_str += " et al."
            else:
                author_str = str(authors) if authors else ""

            matrix.append({
                "source_id": sid,
                "title": s.get("title", ""),
                "authors": author_str,
                "year": s.get("year"),
                "venue": s.get("venue", ""),
                "source_type": s.get("source_type", "unknown"),
                "quality_score": s.get("quality_score", 0.0),
                "key_contribution": f"Contributed {card_count} evidence card(s)",
            })
        return matrix
