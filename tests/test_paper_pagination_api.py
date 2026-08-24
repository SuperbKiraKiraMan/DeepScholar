"""Pagination API for direct paper recommendation and graph results."""

from fastapi.testclient import TestClient

from app.main import app
from app.services.run_store import run_store
from app.tools.base import ToolResult
from app.tools.semantic_scholar_provider import SemanticScholarClient


client = TestClient(app)


def _source(index: int):
    return {
        "source_id": f"s2:{index}",
        "title": f"Paper {index}",
        "url": f"https://example.org/{index}",
        "provider": "semantic_scholar",
    }


def test_recommendation_run_can_fetch_next_page(monkeypatch):
    run_id = run_store.create("graph RAG", run_id="page-rec")
    run_store.update(
        run_id,
        status="completed",
        intent="paper_recommendation",
        execution_route="direct_tool",
        research_topic="graph RAG",
        selected_tool_args={"topic": "graph RAG"},
    )
    captured = {}

    async def fake_page(self, topic, offset=0, limit=20, **kwargs):
        captured.update({"topic": topic, "offset": offset, "limit": limit})
        return ToolResult(
            success=True,
            tool_name="semantic_scholar_recommendations",
            data={
                "results": [_source(6), _source(7)],
                "has_more": True,
                "next_offset": 7,
                "total_found": None,
            },
        )

    monkeypatch.setattr(SemanticScholarClient, "recommend_page", fake_page)
    response = client.get("/api/research/runs/page-rec/papers?offset=5&limit=2")

    assert response.status_code == 200
    assert captured == {"topic": "graph RAG", "offset": 5, "limit": 2}
    body = response.json()
    assert [item["title"] for item in body["items"]] == ["Paper 6", "Paper 7"]
    assert body["has_more"] is True
    assert body["next_offset"] == 7


def test_deep_research_run_rejects_direct_result_pagination():
    run_id = run_store.create("deep topic", run_id="page-deep")
    run_store.update(
        run_id,
        status="completed",
        intent="deep_research",
        execution_route="full_research",
    )

    response = client.get("/api/research/runs/page-deep/papers?offset=0&limit=20")

    assert response.status_code == 409
