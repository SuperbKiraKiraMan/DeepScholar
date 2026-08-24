"""Workspace library, evidence, and report aggregation tests."""

import json
import sqlite3

from fastapi.testclient import TestClient

from app.main import app
from app.services.workspace_catalog import WorkspaceCatalog


class StubRunStore:
    def __init__(self, runs):
        self.runs = runs

    def list_runs_page(self, *, limit=100, offset=0, status=None):
        items = list(self.runs.values())
        return [
            {"run_id": item["run_id"], "topic": item["topic"]}
            for item in items[offset:offset + limit]
        ], len(items)

    def get(self, run_id):
        return self.runs.get(run_id)


def _local_index(path):
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        CREATE TABLE local_paper_documents (
            source_path TEXT PRIMARY KEY, paper_id TEXT, title TEXT,
            authors_json TEXT, year INTEGER, doi TEXT, zotero_storage_key TEXT,
            content_hash TEXT, modified_ns INTEGER, size_bytes INTEGER, indexed_at TEXT
        );
        CREATE TABLE local_paper_chunks (chunk_id TEXT, source_path TEXT);
        """)
        connection.execute(
            "INSERT INTO local_paper_documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("/papers/a.pdf", "local:a", "Local RAG", json.dumps(["Ada"]), 2025,
             "10.1/a", "ABC", "hash", 1, 2048, "2026-01-01"),
        )
        connection.execute("INSERT INTO local_paper_chunks VALUES (?, ?)", ("c1", "/papers/a.pdf"))


def test_workspace_catalog_aggregates_real_artifacts(tmp_path):
    index = tmp_path / "papers.sqlite3"
    _local_index(index)
    run = {
        "run_id": "run-1", "topic": "RAG evaluation", "status": "completed",
        "created_at": "2026-01-02", "finished_at": "2026-01-02",
        "final_report": "# RAG evaluation\nGrounded report", "total_latency_ms": 1200,
        "sources": [{
            "source_id": "s1", "title": "Searched RAG", "url": "https://example.org/paper",
            "authors": ["Bob"], "year": 2024, "provider": "semantic_scholar",
        }],
        "evidence_cards": [{
            "evidence_id": "e1", "claim": "RAG needs joint evaluation", "quote": "quoted text",
            "source_id": "s1", "url": "https://example.org/paper", "confidence": 0.9,
        }],
    }
    catalog = WorkspaceCatalog(StubRunStore({"run-1": run}), str(index))

    papers = catalog.papers()
    assert papers["local_count"] == 1
    assert papers["searched_count"] == 1
    assert {item["title"] for item in papers["items"]} == {"Local RAG", "Searched RAG"}
    assert papers["items"][0]["chunk_count"] == 1

    evidence = catalog.evidence()["items"][0]
    assert evidence["source_title"] == "Searched RAG"
    assert evidence["run_topic"] == "RAG evaluation"
    assert catalog.reports()["items"][0]["report"].startswith("# RAG")


def test_workspace_collection_routes_are_available():
    client = TestClient(app)
    for path in ("/api/library", "/api/evidence-library", "/api/reports"):
        response = client.get(path)
        assert response.status_code == 200
        assert "items" in response.json()


def test_frontend_navigation_and_routing_state_are_interactive():
    client = TestClient(app)
    html = client.get("/").text
    script = client.get("/static/app.js").text
    css = client.get("/static/styles.css").text
    assert 'data-workspace="library"' in html
    assert 'data-workspace="evidence"' not in html
    assert 'data-workspace="reports"' not in html
    assert '>新对话</button>' in html
    for element_id in ("library-view", "library-list", "library-detail"):
        assert f'id="{element_id}"' in html
    for removed_id in ("evidence-library-view", "reports-view"):
        assert f'id="{removed_id}"' not in html
    assert "function setWorkspace(name)" in script
    assert 'fetch("/api/library?' not in script  # endpoint comes from the route map
    assert 'library: "/api/library?' in script
    assert ".assistant-message.is-routing .deep-response" in css
    assert "if (!force && !followLatest) return;" in script
    assert 'elements.chatScroll.addEventListener("wheel"' in script
