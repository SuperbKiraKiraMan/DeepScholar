"""Run-scoped observability metrics derived from structured Agent trace events."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_VERSION = "1.0"

_PSEUDO_TOOL_NAMES = {
    "", "function_call_started", "llm_finished", "llm_failed",
    "llm_function_call", "llm_finish", "provider_fallback",
    "tool_args_rejected", "tool_loop_fallback", "tool_loop_finished",
    "tool_loop_limit_reached", "tool_rejected", "tool_selected",
    "tool_started", "tool_finished",
    "retrieval_observed", "retrieval_query_rewritten",
    "retrieval_source_switched", "retrieval_finished",
}

_WORKER_COMPLETION_EVENTS = {
    "worker_finished", "search_complete", "reading_complete",
    "analysis_complete", "citation_complete",
}

# 统计运行指标
def aggregate_run_metrics(
    trace: Optional[Iterable[Dict[str, Any]]],
    *,
    total_latency_ms: int = 0,
    status: str = "",
    backend: str = "",
    agent_mode: str = "",
    retry_count: int = 0,
    replan_count: int = 0,
    warnings: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    evidence_cards: Optional[List[Dict[str, Any]]] = None,
    citation_check_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate one run without global counters or business-side effects."""
    events = [event for event in (trace or []) if isinstance(event, dict)]
    node_events = [event for event in events if event.get("event") == "node_observed"]
    tool_events = [event for event in events if _is_tool_completion(event)]
    llm_events = [
        event for event in events
        if event.get("event") in {"llm_finished", "llm_failed"}
    ]
    worker_events = [event for event in events if event.get("event") in _WORKER_COMPLETION_EVENTS]

    node_metrics = _aggregate_operations(node_events, "node")
    worker_metrics = _aggregate_workers(worker_events, events)
    tool_metrics = _aggregate_tools(tool_events)
    llm_metrics = _aggregate_llm(llm_events)

    citations = citation_check_results or []
    valid_citations = sum(1 for item in citations if item.get("is_valid", False))
    citation_pass_rate = round(valid_citations / len(citations), 4) if citations else None

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "status": status,
            "backend": backend,
            "agent_mode": agent_mode,
            "latency_ms": max(0, _as_int(total_latency_ms)),
            "trace_event_count": len(events),
            "warning_count": len(warnings or []),
            "error_event_count": sum(
                1 for event in events
                if event.get("event") in {"error", "llm_failed"}
                or event.get("success") is False
            ),
            "retry_count": max(0, _as_int(retry_count)),
            "replan_count": max(0, _as_int(replan_count)),
            "fallback_count": sum(
                1 for event in events
                if event.get("event") in {"llm_fallback", "tool_loop_fallback", "provider_fallback"}
            ),
            "timeout_count": sum(1 for event in events if _is_timeout(event)),
        },
        "nodes": node_metrics,
        "workers": worker_metrics,
        "tools": tool_metrics,
        "llm": llm_metrics,
        "quality": {
            "source_count": len(sources or []),
            "evidence_count": len(evidence_cards or []),
            "citation_count": len(citations),
            "valid_citation_count": valid_citations,
            "citation_pass_rate": citation_pass_rate,
        },
    }


def _aggregate_operations(events: List[Dict[str, Any]], name_field: str) -> Dict[str, Any]:
    by_name: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"execution_count": 0, "success_count": 0, "error_count": 0, "latency_ms": 0}
    )
    slowest = None

    for event in events:
        name = str(
            event.get(name_field)
            or event.get(f"observed_{name_field}")
            or event.get("event")
            or "unknown"
        )
        latency = max(0, _as_int(event.get("latency_ms")))
        bucket = by_name[name]
        bucket["execution_count"] += 1
        bucket["latency_ms"] += latency
        if event.get("success", True):
            bucket["success_count"] += 1
        else:
            bucket["error_count"] += 1
        if slowest is None or latency > slowest["latency_ms"]:
            slowest = {"name": name, "latency_ms": latency}

    for bucket in by_name.values():
        count = bucket["execution_count"]
        bucket["avg_latency_ms"] = round(bucket["latency_ms"] / count, 2) if count else 0

    return {
        "execution_count": len(events),
        "success_count": sum(1 for event in events if event.get("success", True)),
        "error_count": sum(1 for event in events if not event.get("success", True)),
        "total_latency_ms": sum(max(0, _as_int(event.get("latency_ms"))) for event in events),
        "slowest": slowest,
        "by_name": dict(sorted(by_name.items())),
    }


def _aggregate_workers(
    worker_events: List[Dict[str, Any]],
    all_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = []
    for event in worker_events:
        item = dict(event)
        if not item.get("worker_type"):
            event_name = str(item.get("event") or "worker")
            item["worker_type"] = event_name.removesuffix("_complete").removesuffix("_finished")
        normalized.append(item)
    metrics = _aggregate_operations(normalized, "worker_type")
    metrics["dispatch_count"] = sum(1 for event in all_events if event.get("event") == "send_dispatch")
    metrics["partial_failure_count"] = sum(
        1 for event in normalized if not event.get("success", True)
    )
    return metrics


def _aggregate_tools(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = _aggregate_operations(events, "tool_name")
    metrics["call_count"] = metrics.pop("execution_count")
    metrics["timeout_count"] = sum(1 for event in events if _is_timeout(event))
    return metrics


def _aggregate_llm(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = _aggregate_operations(events, "agent")
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    models = set()

    for event in events:
        usage = event.get("usage") or {}
        if isinstance(usage, dict):
            prompt_tokens += max(0, _as_int(usage.get("prompt_tokens")))
            completion_tokens += max(0, _as_int(usage.get("completion_tokens")))
            total_tokens += max(0, _as_int(usage.get("total_tokens")))
        model = str(event.get("model") or "").strip()
        if model:
            models.add(model)

    input_rate = _optional_float("LLM_INPUT_COST_PER_1M_TOKENS")
    output_rate = _optional_float("LLM_OUTPUT_COST_PER_1M_TOKENS")
    estimated_cost = None
    if input_rate is not None and output_rate is not None:
        estimated_cost = round(
            prompt_tokens / 1_000_000 * input_rate
            + completion_tokens / 1_000_000 * output_rate,
            8,
        )

    metrics.update({
        "call_count": metrics.pop("execution_count"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "models": sorted(models),
        "estimated_cost_usd": estimated_cost,
        "cost_status": "estimated" if estimated_cost is not None else "unavailable",
    })
    return metrics


def _is_tool_completion(event: Dict[str, Any]) -> bool:
    event_name = event.get("event")
    tool_name = str(event.get("tool_name") or "")
    if event_name == "tool_finished" and not event.get("graph_node", False):
        return tool_name not in _PSEUDO_TOOL_NAMES
    return not event_name and tool_name not in _PSEUDO_TOOL_NAMES and "success" in event


def _is_timeout(event: Dict[str, Any]) -> bool:
    text = " ".join(
        str(event.get(key) or "") for key in ("error", "error_summary", "input_summary")
    ).lower()
    return "timeout" in text or "timed out" in text or "cancelled" in text


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _optional_float(name: str) -> Optional[float]:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
