"""Phase 5A run observability tests. All tests are deterministic and offline."""

import asyncio

from fastapi.testclient import TestClient

from app.main import app
from app.observability.metrics import aggregate_run_metrics
from app.services.run_store import RunStore


client = TestClient(app)


def _sample_trace():
    return [
        {"event": "node_observed", "observed_node": "planner", "success": True, "latency_ms": 12},
        {"event": "node_observed", "observed_node": "search_worker", "success": True, "latency_ms": 20},
        {"event": "worker_finished", "worker_type": "search", "success": True, "latency_ms": 18},
        {"event": "send_dispatch", "worker_type": "search", "success": True},
        {"event": "tool_finished", "tool_name": "academic_search", "success": True, "latency_ms": 17},
        {"event": "tool_finished", "tool_name": "evidence_extract", "success": False,
         "latency_ms": 30, "error": "tool timeout"},
        {"event": "function_call_started", "tool_name": "function_call_started", "success": True},
        {"event": "llm_finished", "agent": "llm_worker", "model": "deepseek-test",
         "success": True, "latency_ms": 40,
         "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}},
        {"event": "tool_loop_fallback", "success": True},
    ]


class TestMetricAggregation:
    def test_empty_trace_has_stable_schema(self):
        metrics = aggregate_run_metrics([])
        assert metrics["schema_version"] == "1.0"
        assert metrics["tools"]["call_count"] == 0
        assert metrics["llm"]["estimated_cost_usd"] is None
        assert metrics["llm"]["cost_status"] == "unavailable"

    def test_aggregates_run_node_worker_tool_and_llm_layers(self):
        metrics = aggregate_run_metrics(
            _sample_trace(), total_latency_ms=100, backend="graph_send", agent_mode="llm",
            retry_count=1, replan_count=2, warnings=["warning"],
        )
        assert metrics["run"]["backend"] == "graph_send"
        assert metrics["run"]["latency_ms"] == 100
        assert metrics["run"]["retry_count"] == 1
        assert metrics["run"]["replan_count"] == 2
        assert metrics["run"]["fallback_count"] == 1
        assert metrics["nodes"]["execution_count"] == 2
        assert metrics["workers"]["execution_count"] == 1
        assert metrics["workers"]["dispatch_count"] == 1
        assert metrics["tools"]["call_count"] == 2
        assert metrics["tools"]["error_count"] == 1
        assert metrics["tools"]["timeout_count"] == 1
        assert metrics["llm"]["call_count"] == 1
        assert metrics["llm"]["total_tokens"] == 125
        assert metrics["llm"]["models"] == ["deepseek-test"]

    def test_pseudo_tool_events_are_not_counted_as_tools(self):
        metrics = aggregate_run_metrics(_sample_trace())
        assert set(metrics["tools"]["by_name"]) == {"academic_search", "evidence_extract"}

    def test_legacy_loop_trace_is_supported(self):
        metrics = aggregate_run_metrics([
            {"tool_name": "academic_search", "success": True, "latency_ms": 8},
            {"tool_name": "tool_loop_finished", "success": True, "latency_ms": 0},
        ])
        assert metrics["tools"]["call_count"] == 1
        assert metrics["tools"]["by_name"]["academic_search"]["latency_ms"] == 8

    def test_worker_names_are_normalized(self):
        metrics = aggregate_run_metrics([
            {"event": "analysis_complete", "success": True, "latency_ms": 3},
            {"event": "citation_complete", "success": False, "latency_ms": 4},
        ])
        assert set(metrics["workers"]["by_name"]) == {"analysis", "citation"}
        assert metrics["workers"]["partial_failure_count"] == 1

    def test_quality_counts_and_citation_pass_rate(self):
        metrics = aggregate_run_metrics(
            [], sources=[{"source_id": "s1"}],
            evidence_cards=[{"source_id": "s1"}, {"source_id": "s1"}],
            citation_check_results=[{"is_valid": True}, {"is_valid": False}],
        )
        assert metrics["quality"] == {
            "source_count": 1,
            "evidence_count": 2,
            "citation_count": 2,
            "valid_citation_count": 1,
            "citation_pass_rate": 0.5,
        }

    def test_cost_is_estimated_only_when_rates_are_configured(self, monkeypatch):
        monkeypatch.setenv("LLM_INPUT_COST_PER_1M_TOKENS", "2")
        monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "4")
        metrics = aggregate_run_metrics(_sample_trace())
        assert metrics["llm"]["estimated_cost_usd"] == 0.0003
        assert metrics["llm"]["cost_status"] == "estimated"

    def test_invalid_cost_rates_do_not_break_a_run(self, monkeypatch):
        monkeypatch.setenv("LLM_INPUT_COST_PER_1M_TOKENS", "not-a-number")
        monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "4")
        metrics = aggregate_run_metrics(_sample_trace())
        assert metrics["llm"]["estimated_cost_usd"] is None
        assert metrics["llm"]["cost_status"] == "unavailable"


class TestRuntimeIntegration:
    def test_graph_send_result_contains_run_scoped_metrics(self):
        from app.graph.runtime import run_graph

        result = asyncio.run(run_graph(
            "LLM agent evaluation", max_sources=2, agent_mode="rule",
        ))
        metrics = result["observability_metrics"]
        assert metrics["run"]["backend"] == "graph_send"
        assert metrics["nodes"]["execution_count"] >= 9
        assert metrics["workers"]["dispatch_count"] >= 2
        assert metrics["tools"]["call_count"] >= 1
        assert metrics["quality"]["source_count"] == len(result["sources"])

    def test_loop_result_uses_same_metrics_contract(self):
        from app.services.orchestrator import Orchestrator

        result = asyncio.run(Orchestrator().run("RAG evaluation", max_sources=2))
        metrics = result["observability_metrics"]
        assert metrics["run"]["backend"] == "loop"
        assert metrics["tools"]["call_count"] >= 1
        assert metrics["quality"]["evidence_count"] == len(result["evidence_cards"])

    def test_run_store_default_is_backward_compatible(self):
        store = RunStore()
        run_id = store.create("test")
        run = store.get(run_id)
        assert run["observability_metrics"] == {}
        assert run["total_latency_ms"] == 0


class TestFrontendObservability:
    def test_dashboard_has_observability_panel(self):
        html = client.get("/").text
        assert 'id="observability-section"' in html
        assert 'id="observability-display"' in html
        assert "Run telemetry" in html

    def test_frontend_uses_safe_dom_rendering_for_metrics(self):
        js = client.get("/static/app.js").text
        assert "function renderObservability" in js
        assert "observabilityDisplay.replaceChildren" in js
        assert "innerHTML" not in js

    def test_mobile_observability_layout_exists(self):
        css = client.get("/static/styles.css").text
        assert ".observability-grid" in css
        assert "grid-template-columns: 1fr" in css
