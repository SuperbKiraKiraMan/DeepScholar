"""Phase 5B SQLite completed-run history tests. Zero network calls."""

import sqlite3

from fastapi.testclient import TestClient

from app.main import app
from app.services.run_store import RunStore, run_store
from app.storage.sqlite_run_repository import SQLiteRunRepository


client = TestClient(app)


def _snapshot(run_id="run-001", topic="Agent evaluation", status="completed", created_at="2026-07-22T10:00:00", session_id=""):
    return {
        "run_id": run_id,
        "session_id": session_id,
        "topic": topic,
        "status": status,
        "backend": "graph_send",
        "agent_mode": "llm",
        "model": "deepseek-v4-flash",
        "prompt_version": "report-v2",
        "draft_report": "draft",
        "final_report": "final report",
        "sources": [{"source_id": "s1", "title": "Paper", "url": "https://example.org/paper"}],
        "evidence_cards": [{"source_id": "s1", "claim": "claim", "quote": "quote", "url": "https://example.org/paper"}],
        "citation_check_results": [{"citation_id": 1, "source_id": "s1", "is_valid": True}],
        "citation_summary": {"valid": 1},
        "source_matrix": [{"source_id": "s1", "title": "Paper"}],
        "eval_metrics": {"answer_not_empty": True},
        "eval_metric_details": {"answer_not_empty": {"passed": True}},
        "observability_metrics": {"run": {"latency_ms": 123}, "llm": {"models": ["deepseek-v4-flash"]}},
        "warnings": [],
        "trace": [{"event": "run_started"}],
        "task_dag": {"tasks": []},
        "eval_feedback": [],
        "fixes_applied": ["removed invalid citation"],
        "unresolved_issues": [],
        "total_latency_ms": 123,
        "created_at": created_at,
        "finished_at": "2026-07-22T10:00:01",
    }


class TestSQLiteRunRepository:
    def test_round_trip_preserves_complete_snapshot(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot())

        restored = repo.get("run-001")
        assert restored["final_report"] == "final report"
        assert restored["sources"][0]["source_id"] == "s1"
        assert restored["evidence_cards"][0]["claim"] == "claim"
        assert restored["observability_metrics"]["run"]["latency_ms"] == 123
        assert restored["prompt_version"] == "report-v2"

    def test_snapshot_is_immutable_on_duplicate_run_id(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot(topic="Original"))
        repo.save(_snapshot(topic="Overwritten"))
        assert repo.get("run-001")["topic"] == "Original"

    def test_list_returns_summaries_with_pagination(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("r1", "First", created_at="2026-07-22T10:00:00"))
        repo.save(_snapshot("r2", "Second", created_at="2026-07-22T11:00:00"))
        repo.save(_snapshot("r3", "Third", created_at="2026-07-22T12:00:00"))

        page, total = repo.list(limit=2, offset=0)
        assert total == 3
        assert [item["run_id"] for item in page] == ["r3", "r2"]
        assert "final_report" not in page[0]
        assert page[0]["source_count"] == 1

        second_page, _ = repo.list(limit=2, offset=2)
        assert [item["run_id"] for item in second_page] == ["r1"]

    def test_list_filters_status(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("ok", status="completed"))
        repo.save(_snapshot("bad", status="failed", created_at="2026-07-22T11:00:00"))
        rows, total = repo.list(status="failed")
        assert total == 1
        assert rows[0]["run_id"] == "bad"

    def test_list_groups_multi_turn_runs_by_session(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("turn-1", "初始研究问题", created_at="2026-07-22T10:00:00", session_id="session-a"))
        follow_up = _snapshot("turn-2", "第一轮追问", created_at="2026-07-22T11:00:00", session_id="session-a")
        follow_up["sources"] = []
        repo.save(follow_up)
        repo.save(_snapshot("standalone", "独立报告", created_at="2026-07-22T12:00:00"))

        rows, total = repo.list(group_by_session=True)

        assert total == 2
        session_row = next(item for item in rows if item["session_id"] == "session-a")
        assert session_row["run_id"] == "turn-2"
        assert session_row["conversation_title"] == "初始研究问题"
        assert session_row["turn_count"] == 2
        assert session_row["conversation_source_count"] == 1

    def test_list_by_session_restores_turns_in_chronological_order(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot(
            "turn-2", "第二轮", created_at="2026-07-22T11:00:00", session_id="session-a"
        ))
        repo.save(_snapshot(
            "turn-1", "第一轮", created_at="2026-07-22T10:00:00", session_id="session-a"
        ))
        repo.save(_snapshot("other", "其他会话", session_id="session-b"))

        rows = repo.list_by_session("session-a")

        assert [item["run_id"] for item in rows] == ["turn-1", "turn-2"]
        assert all(item["session_id"] == "session-a" for item in rows)

    def test_restored_run_includes_persisted_recommendation_range(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        snapshot = _snapshot("recommend-more", session_id="session-a")
        snapshot["trace"] = [
            {
                "event": "intent_classified",
                "intent": "recommend_more",
                "execution_route": "direct_tool",
            },
            {
                "event": "direct_reviewer_complete",
                "recommendation_number_start": 6,
                "recommendation_number_end": 10,
            },
        ]
        repo.save(snapshot)

        restored = repo.get("recommend-more")

        assert restored["recommendation_number_start"] == 6
        assert restored["recommendation_number_end"] == 10

    def test_wal_and_indexes_are_enabled(self, tmp_path):
        db_path = tmp_path / "history.db"
        repo = SQLiteRunRepository(str(db_path))
        repo.save(_snapshot())
        with sqlite3.connect(db_path) as conn:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            indexes = {row[1] for row in conn.execute("PRAGMA index_list('research_runs')")}
        assert journal_mode.lower() == "wal"
        assert "idx_research_runs_created_at" in indexes
        assert "idx_research_runs_status" in indexes


class TestRunStoreHistoryBoundary:
    def test_finish_persists_and_new_store_recovers_after_restart(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        first_process = RunStore(repository=repo)
        run_id = first_process.create("Persistent report", run_id="restart1")
        first_process.update(run_id, final_report="saved", sources=[{"source_id": "s1"}])
        first_process.finish(run_id, "completed")

        second_process = RunStore(repository=repo)
        recovered = second_process.get(run_id)
        assert recovered["final_report"] == "saved"
        assert recovered["status"] == "completed"
        assert second_process._runs == {}

    def test_memory_has_priority_over_persisted_history(self, tmp_path):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("same", topic="Historical"))
        store = RunStore(repository=repo)
        store.create("Active", run_id="same")
        assert store.get("same")["topic"] == "Active"

    def test_persistence_failure_degrades_to_warning(self):
        class BrokenRepository:
            def save(self, snapshot):
                raise sqlite3.OperationalError("disk unavailable")

        store = RunStore(repository=BrokenRepository())
        run_id = store.create("Still return the report")
        store.update(run_id, final_report="important result")
        store.finish(run_id, "completed")
        run = store.get(run_id)
        assert run["final_report"] == "important result"
        assert run["status"] == "completed"
        assert any("history persistence failed" in warning.lower() for warning in run["warnings"])


class TestHistoryAPI:
    def test_api_lists_summary_and_reads_detail_from_sqlite(self, tmp_path, monkeypatch):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("api-hist", topic="Recovered API report"))
        monkeypatch.setattr(run_store, "_repository", repo)
        run_store._runs.pop("api-hist", None)

        listing = client.get("/api/runs?limit=5&offset=0")
        assert listing.status_code == 200
        body = listing.json()
        assert body["total"] == 1
        assert body["runs"][0]["run_id"] == "api-hist"
        assert "final_report" not in body["runs"][0]

        detail = client.get("/api/runs/api-hist")
        assert detail.status_code == 200
        assert detail.json()["final_report"] == "final report"

    def test_api_supports_status_filter(self, tmp_path, monkeypatch):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("completed-one", status="completed"))
        repo.save(_snapshot("failed-one", status="failed", created_at="2026-07-22T11:00:00"))
        monkeypatch.setattr(run_store, "_repository", repo)
        response = client.get("/api/runs?status=failed")
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["runs"][0]["run_id"] == "failed-one"

    def test_api_groups_recent_items_as_conversations_by_default(self, tmp_path, monkeypatch):
        repo = SQLiteRunRepository(str(tmp_path / "history.db"))
        repo.save(_snapshot("api-turn-1", topic="首轮问题", session_id="api-session"))
        repo.save(_snapshot(
            "api-turn-2", topic="后续追问", session_id="api-session",
            created_at="2026-07-22T11:00:00",
        ))
        monkeypatch.setattr(run_store, "_repository", repo)

        body = client.get("/api/runs?limit=5&offset=0").json()

        assert body["total"] == 1
        assert body["runs"][0]["run_id"] == "api-turn-2"
        assert body["runs"][0]["conversation_title"] == "首轮问题"
        assert body["runs"][0]["turn_count"] == 2

        conversation = client.get("/api/conversations/api-session/runs")
        assert conversation.status_code == 200
        assert [item["run_id"] for item in conversation.json()["runs"]] == [
            "api-turn-1", "api-turn-2"
        ]


class TestHistoryFrontend:
    def test_dashboard_contains_recent_reports(self):
        html = client.get("/").text
        assert 'id="history-list"' in html
        assert 'id="btn-history-refresh"' in html
        assert "Recent reports" in html

    def test_frontend_loads_paginated_history_safely(self):
        js = client.get("/static/app.js").text
        assert "function loadRunHistory" in js
        assert 'fetch("/api/runs?limit="' in js
        assert "historyList.appendChild" in js
        assert "innerHTML" not in js

    def test_frontend_restores_expired_sessions_from_history(self):
        js = client.get("/static/app.js").text
        assert '"/api/sessions/" + encodeURIComponent(sessionId) + "/restore"' in js
        assert 'response.status === 404 || response.status === 410) return restoreSession(activeSessionId)' in js
        assert "activeSessionId = data.session_id || sessionId" in js
        assert "if (data.session) updateSessionDisplay(data.session)" in js
