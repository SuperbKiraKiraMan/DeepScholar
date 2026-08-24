"""
app/services/run_store.py

RunStore —— 运行状态存储。

运行态由内存字典管理，重启后由 SQLite Repository 提供完成态历史。

类比 Spring Boot：
- RunStore ≈ 一个简单的 Repository（不依赖 JPA，直接内存存储）
- 负责 CRUD：创建 run、更新状态、查询 run

在项目调用链中的位置：
routes.py → Orchestrator → RunStore（读写 run 状态）
routes.py → RunStore（GET /api/runs/{run_id} 直接查询）
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class RunStore:
    """
    内存版运行状态存储。

    设计为简单的 key-value 存储，不引入数据库依赖。
    """

    def __init__(self, repository=None):
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._repository = repository

    def _get_repository(self):
        if self._repository is not None:
            return self._repository
        from app.storage import get_history_repository
        return get_history_repository()

    def create(self, topic: str, run_id: str = None, **kwargs) -> str:
        """创建新的 run 记录，返回 run_id。"""
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        self._runs[run_id] = {
            "run_id": run_id,
            "topic": topic,
            "research_topic": topic,
            "intent": "deep_research",
            "execution_route": "full_research",
            "selected_tools": [],
            "selected_tool_args": {},
            "intent_confidence": 0.0,
            "controller_reasoning": "",
            "session_id": "",
            "is_follow_up": False,
            "reference_expression": "",
            "resolved_paper_ids": [],
            "seed_paper_ids": [],
            "resolved_section": None,
            "route_name": "full_research",
            "fallback_used": False,
            "answer": "",
            "conversation_result": {},
            "status": "running",
            "created_at": now,
            "finished_at": None,
            "final_report": "",
            "draft_report": "",
            "sources": [],
            "discovered_source_count": 0,
            "analyzed_source_count": 0,
            "analysis_selection": {},
            "evidence_cards": [],
            "outline": {},
            "report_completion_ready": False,
            "report_completion_issues": [],
            "citation_check_results": [],
            "citation_summary": {},
            "source_matrix": [],
            "eval_metrics": {},
            "eval_feedback": [],
            "warnings": [],
            "trace": [],
            "task_dag": {},
            "latency_ms": 0,
            "total_latency_ms": 0,
            "observability_metrics": {},
            "prompt_version": os.getenv("PROMPT_VERSION", "v1"),
            **kwargs,
        }
        return run_id

    def update(self, run_id: str, **kwargs):
        """更新 run 的字段。"""
        if run_id in self._runs:
            self._runs[run_id].update(kwargs)

    def finish(self, run_id: str, status: str = "completed"):
        """标记 run 结束，并尽力保存不可变的 SQLite 快照。"""
        if run_id in self._runs:
            self._runs[run_id]["status"] = status
            self._runs[run_id]["finished_at"] = datetime.now().isoformat()
            try:
                repository = self._get_repository()
                if repository is not None:
                    repository.save(dict(self._runs[run_id]))
            except Exception as exc:
                warning = f"Run history persistence failed: {type(exc).__name__}: {str(exc)[:160]}"
                warnings = self._runs[run_id].setdefault("warnings", [])
                if warning not in warnings:
                    warnings.append(warning)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        """优先查询活动内存，找不到时回退 SQLite 历史。"""
        active = self._runs.get(run_id)
        if active is not None:
            return active
        try:
            repository = self._get_repository()
            return repository.get(run_id) if repository is not None else None
        except Exception:
            return None

    def list_session_runs(self, session_id: str) -> List[Dict[str, Any]]:
        """返回一个 Session 的完整轮次，并合并尚在内存中的最新结果。"""
        if not session_id:
            return []
        persisted: List[Dict[str, Any]] = []
        try:
            repository = self._get_repository()
            if repository is not None:
                persisted = repository.list_by_session(session_id)
        except Exception:
            persisted = []

        # 关键步骤：以 run_id 去重，内存数据优先于不可变的持久化快照。
        by_id = {
            str(run.get("run_id") or ""): dict(run)
            for run in persisted if run.get("run_id")
        }
        for run in self._runs.values():
            if str(run.get("session_id") or "") == session_id and run.get("run_id"):
                by_id[str(run["run_id"])] = dict(run)
        runs = list(by_id.values())
        runs.sort(key=lambda run: (str(run.get("created_at") or ""), str(run.get("run_id") or "")))
        return runs

    def list_runs(self) -> List[Dict[str, Any]]:
        """向后兼容：列出当前可查询的前 100 条摘要。"""
        runs, _ = self.list_runs_page(limit=100)
        return runs

    def list_runs_page(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        group_by_session: bool = False,
    ) -> tuple[List[Dict[str, Any]], int]:
        """分页列出完成态历史；禁用 SQLite 时回退内存。"""
        try:
            repository = self._get_repository()
            if repository is not None:
                return repository.list(
                    limit=limit, offset=offset, status=status,
                    group_by_session=group_by_session,
                )
        except Exception:
            pass

        runs = list(self._runs.values())
        if status:
            runs = [run for run in runs if run.get("status") == status]
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        if group_by_session:
            # 关键步骤：内存回退路径与 SQLite 保持相同的“一个 Session 一个窗口”语义。
            grouped: Dict[str, Dict[str, Any]] = {}
            first_topics: Dict[str, str] = {}
            turn_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}
            for run in reversed(runs):
                key = str(run.get("session_id") or run.get("run_id") or "")
                first_topics.setdefault(key, str(run.get("topic") or ""))
                turn_counts[key] = turn_counts.get(key, 0) + 1
                source_counts[key] = max(
                    source_counts.get(key, 0), len(run.get("sources") or [])
                )
            for run in runs:
                key = str(run.get("session_id") or run.get("run_id") or "")
                if key not in grouped:
                    summary = _run_summary(run)
                    summary["conversation_title"] = first_topics.get(key, summary["topic"])
                    summary["turn_count"] = turn_counts.get(key, 1)
                    summary["conversation_source_count"] = source_counts.get(key, 0)
                    grouped[key] = summary
            summaries = list(grouped.values())
            total = len(summaries)
            return summaries[offset:offset + limit], total
        total = len(runs)
        summaries = [_run_summary(run) for run in runs[offset:offset + limit]]
        return summaries, total


def _run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    observability = run.get("observability_metrics") or {}
    models = (observability.get("llm") or {}).get("models") or []
    return {
        "run_id": run.get("run_id", ""),
        "session_id": run.get("session_id", ""),
        "topic": run.get("topic", ""),
        "status": run.get("status", "unknown"),
        "backend": run.get("backend", ""),
        "agent_mode": run.get("agent_mode", ""),
        "model": run.get("model") or (", ".join(models) if models else ""),
        "prompt_version": run.get("prompt_version", "v1"),
        "source_count": len(run.get("sources") or []),
        "evidence_count": len(run.get("evidence_cards") or []),
        "total_latency_ms": run.get("total_latency_ms", run.get("latency_ms", 0)),
        "created_at": run.get("created_at", ""),
        "finished_at": run.get("finished_at"),
    }


# 全局单例：活动 Run 保存在进程内。
run_store = RunStore()
