"""
共享规则评估模块 —— 封装现有 Evaluator，确保 Runtime Evaluator 和 Harness
使用同一套指标实现。没有第二套指标算法，没有分叉的评估器。
"""

from typing import Any, Dict, List

from app.agents.evaluator import Evaluator

# Single shared instance
_evaluator = Evaluator()

# Metric definitions for Harness assertions
METRIC_NAMES = [
    "no_fake_citation",
    "min_sources",
    "citation_id_exists",
    "source_url_valid",
    "evidence_available",
    "answer_not_empty",
    "task_success_rate",
    "tool_error_rate",
    "latency_under_threshold",
]

METRIC_DESCRIPTIONS = {
    "no_fake_citation": "No fake citations in report",
    "min_sources": "At least one source retrieved",
    "citation_id_exists": "Every citation ID exists in source list",
    "source_url_valid": "Citation URLs match source URLs",
    "evidence_available": "At least one verified EvidenceCard was extracted",
    "answer_not_empty": "Final report is non-empty (>50 chars)",
    "task_success_rate": "All tasks completed successfully",
    "tool_error_rate": "Tool error rate below threshold (≤50%)",
    "latency_under_threshold": "Run latency below threshold (≤60s)",
}


def evaluate(
    topic: str = "",
    draft_report: str = "",
    sources: List[Dict[str, Any]] = None,
    evidence_cards: List[Dict[str, Any]] = None,
    citation_check_results: List[Dict[str, Any]] = None,
    citation_summary: Dict[str, Any] = None,
    trace: List[Dict[str, Any]] = None,
    task_dag: Dict[str, Any] = None,
    total_latency_ms: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate using the single shared Evaluator instance.

    Returns same dict as Evaluator.evaluate():
        {"metrics": {...}, "metrics_detail": {...}, "feedback": [...], "all_passed": bool}
    """
    return _evaluator.evaluate(
        topic=topic,
        draft_report=draft_report,
        sources=sources,
        evidence_cards=evidence_cards,
        citation_check_results=citation_check_results,
        citation_summary=citation_summary,
        trace=trace,
        task_dag=task_dag,
        total_latency_ms=total_latency_ms,
    )


def get_metric(name: str, eval_result: Dict[str, Any]) -> Any:
    """Get a single metric value from eval_result."""
    return eval_result.get("metrics", {}).get(name)


def assert_metric(name: str, expected: Any, actual: Any, op: str = "eq") -> Dict[str, Any]:
    """
    Assert that an actual metric value satisfies the expected value using the given op.

    Returns {"passed": bool, "reason": str}
    """
    ops = {
        "eq": lambda a, e: a == e,
        "gte": lambda a, e: float(a) >= float(e),
        "lte": lambda a, e: float(a) <= float(e),
        "gt": lambda a, e: float(a) > float(e),
        "lt": lambda a, e: float(a) < float(e),
        "ne": lambda a, e: a != e,
    }

    if op not in ops:
        return {"passed": False, "reason": f"Unknown op: {op}"}

    try:
        passed = ops[op](actual, expected)
        reason = "" if passed else f"Expected {name} {op} {expected}, got {actual}"
        return {"passed": passed, "reason": reason}
    except (TypeError, ValueError) as e:
        return {"passed": False, "reason": f"Cannot compare: {e}"}
