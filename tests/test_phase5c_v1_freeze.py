"""V1.0 public contract and release-document regression tests."""

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.api.schemas import EvidenceCard, PaperSource, ResearchRequest, ResearchResponse
from app.core.config import config
from app.graph.runtime import PUBLIC_SSE_EVENT_TYPES
from app.main import app


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)

CORE_ROUTES = {
    "/health",
    "/api/research/runs",
    "/api/research/runs/{run_id}/cancel",
    "/api/research/stream/{run_id}",
    "/api/runs",
    "/api/runs/{run_id}",
}

CORE_SSE_EVENTS = {
    "run_started",
    "plan_created",
    "send_dispatch",
    "worker_started",
    "worker_finished",
    "tool_started",
    "tool_finished",
    "citation_checked",
    "eval_finished",
    "draft_reviewer_complete",
    "final_reviewer_complete",
    "source_found",
    "evidence_created",
    "run_finished",
    "error",
    "heartbeat",
}


def test_v1_version_is_consistent_across_api_and_frontend():
    assert config.APP_VERSION == "1.3.0"
    assert client.get("/health").json()["version"] == "1.3.0"
    assert "v1.3" in client.get("/").text


def test_v1_core_api_routes_remain_available():
    openapi_paths = set(client.get("/openapi.json").json()["paths"])
    assert CORE_ROUTES <= openapi_paths


def test_request_source_budget_accepts_one_through_fifty():
    assert ResearchRequest(topic="test", max_sources=1).max_sources == 1
    assert ResearchRequest(topic="test", max_sources=50).max_sources == 50
    with pytest.raises(ValidationError):
        ResearchRequest(topic="test", max_sources=0)
    with pytest.raises(ValidationError):
        ResearchRequest(topic="test", max_sources=51)


def test_v1_core_schema_fields_are_stable():
    assert {
        "source_id", "title", "url", "snippet", "full_text", "authors",
        "year", "venue", "source_type", "quality_score", "provider",
    } <= set(PaperSource.model_fields)
    assert {
        "evidence_id", "claim", "quote", "source_id", "url",
        "confidence", "method", "limitation", "key_results",
        "original_quote", "quote_location", "relevance_to_topic",
    } <= set(EvidenceCard.model_fields)
    assert {
        "run_id", "topic", "status", "final_report", "draft_report",
        "sources", "evidence_cards", "outline", "citation_check_results",
        "source_matrix", "eval_metrics", "observability_metrics",
        "warnings", "trace", "created_at",
    } <= set(ResearchResponse.model_fields)


def test_v1_core_sse_event_names_are_frozen():
    assert CORE_SSE_EVENTS <= PUBLIC_SSE_EVENT_TYPES
    frontend = (ROOT / "app/web/app.js").read_text(encoding="utf-8")
    for event_type in CORE_SSE_EVENTS - {"heartbeat"}:
        assert f'"{event_type}"' in frontend


def test_v1_runtime_dependencies_are_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "langgraph>=" in requirements
    assert "sse-starlette>=" in requirements


def test_public_release_documents_exist_and_are_linked():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current_documents = (
        "docs/README.md",
        "docs/ARCHITECTURE.md",
        "docs/本地论文数据.md",
        "docs/工具接入与部署.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CHANGELOG.md",
    )
    for current_document in current_documents:
        # 关键步骤：发布契约只指向当前文档和治理文件，不引入本机运行记录。
        assert (ROOT / current_document).is_file()
        assert current_document in readme or current_document.startswith("docs/")

    docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    for index_entry in (
        "ARCHITECTURE.md",
        "工具接入与部署.md",
        "本地论文数据.md",
        "benchmarks/README.md",
    ):
        # 当前索引本身是发布文档契约。
        assert index_entry in docs_index
