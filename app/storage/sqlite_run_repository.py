"""SQLite repository for terminal research-run snapshots."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_JSON_COLUMNS = {
    "sources": "sources_json",
    "evidence_cards": "evidence_cards_json",
    "citation_check_results": "citation_check_results_json",
    "citation_summary": "citation_summary_json",
    "source_matrix": "source_matrix_json",
    "eval_metrics": "eval_metrics_json",
    "eval_metric_details": "eval_metric_details_json",
    "observability_metrics": "observability_metrics_json",
    "warnings": "warnings_json",
    "trace": "trace_json",
    "task_dag": "task_dag_json",
    "eval_feedback": "eval_feedback_json",
    "fixes_applied": "fixes_applied_json",
    "unresolved_issues": "unresolved_issues_json",
    "analysis_selection": "analysis_selection_json",
    "report_completion_issues": "report_completion_issues_json",
    "seed_paper_ids": "seed_paper_ids_json",
}


class SQLiteRunRepository:
    """Store one immutable snapshot per run_id using short SQLite transactions."""

    def __init__(self, db_path: str, busy_timeout_ms: int = 5000):
        self.db_path = str(db_path)
        self.busy_timeout_ms = max(0, int(busy_timeout_ms))
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def save(self, snapshot: Dict[str, Any]) -> None:
        self._ensure_schema()
        values = self._snapshot_values(snapshot)
        columns = list(values)
        placeholders = ", ".join("?" for _ in columns)
        sql = (
            f"INSERT INTO research_runs ({', '.join(columns)}) "
            f"VALUES ({placeholders}) ON CONFLICT(run_id) DO NOTHING"
        )
        with self._connect() as conn:
            conn.execute(sql, [values[column] for column in columns])

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """按时间顺序返回一个会话的全部运行，用于恢复多轮对话窗口。"""
        self._ensure_schema()
        if not session_id:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_runs WHERE session_id = ? "
                "ORDER BY created_at ASC, run_id ASC",
                (session_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def list(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        group_by_session: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]:
        self._ensure_schema()
        limit = min(100, max(1, int(limit)))
        offset = max(0, int(offset))
        where = " WHERE status = ?" if status else ""
        params = (status,) if status else ()
        summary_columns = (
            "run_id, session_id, topic, status, backend, agent_mode, model, prompt_version, "
            "source_count, evidence_count, total_latency_ms, created_at, finished_at"
        )
        with self._connect() as conn:
            if group_by_session:
                # 关键步骤：底层仍保留每轮 run，但最近会话按 session_id 聚合成一个窗口。
                filtered = f"SELECT *, CASE WHEN session_id <> '' THEN session_id ELSE run_id END AS conversation_key FROM research_runs{where}"
                total = conn.execute(
                    f"SELECT COUNT(DISTINCT conversation_key) FROM ({filtered})", params,
                ).fetchone()[0]
                rows = conn.execute(
                    f"""
                    WITH filtered AS ({filtered}), ranked AS (
                        SELECT *,
                            ROW_NUMBER() OVER (
                                PARTITION BY conversation_key ORDER BY created_at DESC, run_id DESC
                            ) AS latest_rank,
                            FIRST_VALUE(topic) OVER (
                                PARTITION BY conversation_key ORDER BY created_at ASC, run_id ASC
                            ) AS conversation_title,
                            COUNT(*) OVER (PARTITION BY conversation_key) AS turn_count,
                            MAX(source_count) OVER (
                                PARTITION BY conversation_key
                            ) AS conversation_source_count
                        FROM filtered
                    )
                    SELECT {summary_columns}, conversation_title, turn_count,
                           conversation_source_count
                    FROM ranked WHERE latest_rank = 1
                    ORDER BY created_at DESC LIMIT ? OFFSET ?
                    """,
                    (*params, limit, offset),
                ).fetchall()
                return [dict(row) for row in rows], int(total)
            total = conn.execute(
                f"SELECT COUNT(*) FROM research_runs{where}", params,
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT {summary_columns} FROM research_runs{where} "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            path = Path(self.db_path)
            if self.db_path != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                if self.db_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS research_runs (
                        run_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL DEFAULT '',
                        topic TEXT NOT NULL,
                        status TEXT NOT NULL,
                        backend TEXT NOT NULL DEFAULT '',
                        agent_mode TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        prompt_version TEXT NOT NULL DEFAULT '',
                        draft_report TEXT NOT NULL DEFAULT '',
                        final_report TEXT NOT NULL DEFAULT '',
                        sources_json TEXT NOT NULL DEFAULT '[]',
                        evidence_cards_json TEXT NOT NULL DEFAULT '[]',
                        citation_check_results_json TEXT NOT NULL DEFAULT '[]',
                        citation_summary_json TEXT NOT NULL DEFAULT '{}',
                        source_matrix_json TEXT NOT NULL DEFAULT '[]',
                        eval_metrics_json TEXT NOT NULL DEFAULT '{}',
                        eval_metric_details_json TEXT NOT NULL DEFAULT '{}',
                        observability_metrics_json TEXT NOT NULL DEFAULT '{}',
                        warnings_json TEXT NOT NULL DEFAULT '[]',
                        trace_json TEXT NOT NULL DEFAULT '[]',
                        task_dag_json TEXT NOT NULL DEFAULT '{}',
                        eval_feedback_json TEXT NOT NULL DEFAULT '[]',
                        fixes_applied_json TEXT NOT NULL DEFAULT '[]',
                        unresolved_issues_json TEXT NOT NULL DEFAULT '[]',
                        analysis_selection_json TEXT NOT NULL DEFAULT '{}',
                        report_completion_issues_json TEXT NOT NULL DEFAULT '[]',
                        seed_paper_ids_json TEXT NOT NULL DEFAULT '[]',
                        report_completion_ready INTEGER NOT NULL DEFAULT 0,
                        source_count INTEGER NOT NULL DEFAULT 0,
                        evidence_count INTEGER NOT NULL DEFAULT 0,
                        total_latency_ms INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        error TEXT NOT NULL DEFAULT ''
                    );
                    CREATE INDEX IF NOT EXISTS idx_research_runs_created_at
                        ON research_runs(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_research_runs_status
                        ON research_runs(status);
                    """
                )
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(research_runs)").fetchall()
                }
                migrations = {
                    "session_id": "TEXT NOT NULL DEFAULT ''",
                    "analysis_selection_json": "TEXT NOT NULL DEFAULT '{}'",
                    "report_completion_issues_json": "TEXT NOT NULL DEFAULT '[]'",
                    "report_completion_ready": "INTEGER NOT NULL DEFAULT 0",
                    "seed_paper_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                }
                for column, definition in migrations.items():
                    if column not in columns:
                        conn.execute(f"ALTER TABLE research_runs ADD COLUMN {column} {definition}")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_research_runs_session_id "
                    "ON research_runs(session_id)"
                )

                # 兼容旧数据库：历史快照曾把 session_id 放在 Trace 中，但没有独立列。
                legacy_rows = conn.execute(
                    "SELECT run_id, trace_json FROM research_runs WHERE session_id = ''"
                ).fetchall()
                for legacy in legacy_rows:
                    try:
                        trace = json.loads(legacy["trace_json"] or "[]")
                    except (TypeError, json.JSONDecodeError):
                        continue
                    session_id = next(
                        (
                            str(event.get("session_id") or "")
                            for event in trace
                            if isinstance(event, dict) and event.get("session_id")
                        ),
                        "",
                    )
                    if session_id:
                        conn.execute(
                            "UPDATE research_runs SET session_id = ? WHERE run_id = ?",
                            (session_id, legacy["run_id"]),
                        )
            self._schema_ready = True

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=max(0.001, self.busy_timeout_ms / 1000),
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn

    def _snapshot_values(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        observability = snapshot.get("observability_metrics") or {}
        models = (observability.get("llm") or {}).get("models") or []
        model = snapshot.get("model") or (", ".join(models) if models else "")
        values = {
            "run_id": str(snapshot.get("run_id") or ""),
            "session_id": str(snapshot.get("session_id") or ""),
            "topic": str(snapshot.get("topic") or ""),
            "status": str(snapshot.get("status") or "completed"),
            "backend": str(snapshot.get("backend") or ""),
            "agent_mode": str(snapshot.get("agent_mode") or ""),
            "model": str(model),
            "prompt_version": str(snapshot.get("prompt_version") or os.getenv("PROMPT_VERSION", "v1")),
            "draft_report": str(snapshot.get("draft_report") or ""),
            "final_report": str(snapshot.get("final_report") or ""),
            "source_count": len(snapshot.get("sources") or []),
            "evidence_count": len(snapshot.get("evidence_cards") or []),
            "total_latency_ms": max(0, int(snapshot.get("total_latency_ms") or snapshot.get("latency_ms") or 0)),
            "created_at": str(snapshot.get("created_at") or ""),
            "finished_at": snapshot.get("finished_at"),
            "error": str(snapshot.get("error") or "")[:1000],
            "report_completion_ready": 1 if snapshot.get("report_completion_ready") else 0,
        }
        for field, column in _JSON_COLUMNS.items():
            default = {} if field in {
                "citation_summary", "eval_metrics", "eval_metric_details",
                "observability_metrics", "task_dag", "analysis_selection",
            } else []
            values[column] = json.dumps(
                snapshot.get(field, default),
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        return values

    def _row_to_run(self, row: sqlite3.Row) -> Dict[str, Any]:
        run = {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "topic": row["topic"],
            "status": row["status"],
            "backend": row["backend"],
            "agent_mode": row["agent_mode"],
            "model": row["model"],
            "prompt_version": row["prompt_version"],
            "draft_report": row["draft_report"],
            "final_report": row["final_report"],
            "total_latency_ms": row["total_latency_ms"],
            "latency_ms": row["total_latency_ms"],
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
            "report_completion_ready": bool(row["report_completion_ready"]),
        }
        for field, column in _JSON_COLUMNS.items():
            try:
                run[field] = json.loads(row[column])
            except (TypeError, json.JSONDecodeError):
                run[field] = {} if field in {
                    "citation_summary", "eval_metrics", "eval_metric_details",
                    "observability_metrics", "task_dag", "analysis_selection",
                } else []
        # 旧快照未单独保存路由字段；从 Trace 恢复渲染多轮历史所需的最小元数据。
        intent_event = next(
            (
                event for event in run.get("trace", [])
                if isinstance(event, dict) and event.get("event") == "intent_classified"
            ),
            {},
        )
        run["intent"] = str(intent_event.get("intent") or "deep_research")
        run["research_topic"] = str(
            intent_event.get("research_topic") or run.get("topic") or ""
        )
        run["execution_route"] = str(
            intent_event.get("execution_route") or "full_research"
        )
        run["route_name"] = str(
            intent_event.get("route_name") or run["execution_route"]
        )
        run["selected_tools"] = list(intent_event.get("selected_tools") or [])
        run["resolved_paper_ids"] = list(
            intent_event.get("resolved_resource_ids")
            or intent_event.get("resolved_paper_ids")
            or []
        )
        run["conversation_operation"] = str(
            intent_event.get("conversation_operation") or ""
        )
        run["clarification_message"] = str(
            intent_event.get("clarification_message") or ""
        )
        run["missing_ordinal"] = intent_event.get("missing_ordinal")
        recommendation_event = next(
            (
                event for event in run.get("trace", [])
                if isinstance(event, dict)
                and event.get("event") == "direct_reviewer_complete"
            ),
            {},
        )
        run["recommendation_number_start"] = recommendation_event.get(
            "recommendation_number_start"
        )
        run["recommendation_number_end"] = recommendation_event.get(
            "recommendation_number_end"
        )
        run["answer"] = run.get("final_report") or run.get("draft_report") or ""
        return run


_repository_lock = threading.Lock()
_repository_key: Optional[Tuple[str, int]] = None
_repository: Optional[SQLiteRunRepository] = None


def get_history_repository() -> Optional[SQLiteRunRepository]:
    """Return a lazy repository so tests can disable history before each run."""
    if os.getenv("RUN_HISTORY_ENABLED", "true").lower() != "true":
        return None
    path = os.getenv("SQLITE_DB_PATH", "data/research_history.db")
    busy_timeout = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "5000"))
    key = (path, busy_timeout)
    global _repository, _repository_key
    with _repository_lock:
        if _repository is None or _repository_key != key:
            _repository = SQLiteRunRepository(path, busy_timeout)
            _repository_key = key
    return _repository


def reset_history_repository() -> None:
    global _repository, _repository_key
    with _repository_lock:
        _repository = None
        _repository_key = None
