"""
Harness 报告生成 —— JSON 产物 + 人类可读的 Markdown 套件报告。
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from harness.models import CaseResult, SuiteResult
from harness.metrics import METRIC_DESCRIPTIONS

_SENSITIVE_RE = re.compile(
    r'(sk-[a-zA-Z0-9_-]{10,}|Bearer\s+[a-zA-Z0-9_\-\.]+)',
    re.IGNORECASE,
)


def _redact_sensitive(obj: Any) -> Any:
    """Recursively redact sensitive values from any JSON-serializable object."""
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if _is_sensitive_key(k) else _redact_sensitive(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_redact_sensitive(v) for v in obj]
    elif isinstance(obj, str):
        if len(obj) > 1000:
            obj = obj[:200] + "...[truncated]"
        return _SENSITIVE_RE.sub("[REDACTED_KEY]", obj)
    return obj


def _is_sensitive_key(key: str) -> bool:
    k = key.lower().replace("_", "").replace("-", "")
    sensitive = {
        "apikey", "deepseekapikey", "openalexapikey", "authorization",
        "xapikey", "accesstoken", "secret", "password", "bearer",
        "fulltext", "chainofthought", "reasoningcontent",
    }
    return k in sensitive


def write_json_report(suite: SuiteResult, output_dir: str) -> str:
    """Write machine-readable JSON artifact with sensitive data redacted."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"suite_report_{ts}.json")
    raw = suite.model_dump()
    redacted = _redact_sensitive(raw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=2, default=str, ensure_ascii=False)
    return path


def write_markdown_report(suite: SuiteResult, output_dir: str) -> str:
    """Write a redacted human-readable Markdown suite report."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"suite_report_{ts}.md")
    redacted_suite = SuiteResult.model_validate(_redact_sensitive(suite.model_dump()))
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(redacted_suite))
    return path


def _render_markdown(suite: SuiteResult, include_generated_at: bool = True) -> str:
    pct = round(suite.passed_cases / max(suite.total_cases, 1) * 100, 1)
    lines = [
        f"# {suite.suite_name}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Cases | {suite.total_cases} |",
        f"| Passed | {suite.passed_cases} |",
        f"| Failed | {suite.failed_cases} |",
        f"| Pass Rate | {pct}% |",
        f"| Backend | {suite.backend} |",
        f"| Agent Mode | {suite.agent_mode} |",
        "",
        "## Results by Case",
        "",
    ]
    if include_generated_at:
        lines[2:2] = [f"**Generated:** {datetime.now(timezone.utc).isoformat()}", ""]

    for i, cr in enumerate(suite.results):
        status_icon = "✅" if cr.passed else "❌"
        lines.extend([
            f"### {i + 1}. {status_icon} {cr.case_id}",
            "",
            f"- **Description:** {cr.description}",
            f"- **Status:** {cr.status} (expected: {cr.expected_status})",
            f"- **Backend:** {cr.backend} | **Agent Mode:** {cr.agent_mode}",
            f"- **Run ID:** {cr.run_id}",
            f"- **Latency:** {cr.total_latency_ms}ms",
            f"- **Retry Count:** {cr.retry_count} | **Replan Count:** {cr.replan_count}",
            "",
        ])

        if cr.llm_diagnostics:
            # 关键步骤：把 fixture 是否意外耗尽直接写入报告，避免只看 Case 状态而漏掉测试污染。
            lines.extend([
                f"- **FakeLLM Calls:** {cr.llm_diagnostics.get('call_count', 0)}",
                f"- **FakeLLM Fixture Exhaustions:** {cr.llm_diagnostics.get('exhaustion_count', 0)}",
                "",
            ])

        # Expectations
        if cr.expectation_results:
            lines.append("| Expectation | Expected | Actual | Passed | Reason |")
            lines.append("|-------------|----------|--------|--------|--------|")
            for er in cr.expectation_results:
                icon = "✅" if er.passed else "❌"
                reason = er.reason[:80] if er.reason else "-"
                lines.append(
                    f"| {er.name} | {er.expected} | {er.actual} | {icon} | {reason} |"
                )
            lines.append("")

        # Warnings
        if cr.warnings:
            lines.append("**Warnings:**")
            for w in cr.warnings[:5]:
                lines.append(f"- ⚠ {w[:120]}")
            if len(cr.warnings) > 5:
                lines.append(f"- ... and {len(cr.warnings) - 5} more")
            lines.append("")

        # Tools called
        if cr.tools_called:
            lines.append(f"**Tools called:** {', '.join(cr.tools_called)}")
            lines.append("")

        # Unresolved
        if cr.unresolved_issues:
            lines.append("**Unresolved Issues:**")
            for u in cr.unresolved_issues:
                lines.append(f"- {u[:120]}")
            lines.append("")

        # Error
        if cr.error:
            lines.append(f"**Error:** `{cr.error[:200]}`")
            lines.append("")

    # Metric pass rates
    if suite.metric_pass_rates:
        lines.extend([
            "## Per-Metric Pass Rates",
            "",
            "| Metric | Description | Pass Rate |",
            "|--------|-------------|-----------|",
        ])
        for name, rate in suite.metric_pass_rates.items():
            desc = METRIC_DESCRIPTIONS.get(name, name)
            lines.append(f"| {name} | {desc} | {rate * 100:.0f}% |")
        lines.append("")

    # Hook summary
    if suite.hook_summary:
        lines.extend([
            "## Hook Execution Summary",
            "",
            "| Hook | Invocations |",
            "|------|-------------|",
        ])
        for key, count in suite.hook_summary.items():
            lines.append(f"| {key} | {count} |")
        lines.append("")

    # Known limitations
    if suite.known_limitations:
        lines.extend([
            "## Known Limitations",
            "",
        ])
        for lim in suite.known_limitations:
            lines.append(f"- {lim}")
        lines.append("")

    return "\n".join(lines)


def write_stable_example_report(suite: SuiteResult, output_dir: str) -> str:
    """Write a stable example report (no random run_id, no absolute timestamps)."""
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize: remove run_ids and timestamps
    sanitized = suite.model_copy(deep=True)
    for r in sanitized.results:
        r.run_id = "<run_id>"
        r.total_latency_ms = 0
        for er in r.expectation_results:
            pass  # keep expectations
        for h in r.hooks:
            h.timestamp_ms = 0

    lines = [
        "# Agent Harness Suite — Example Report",
        "",
        "> This is a stable example report. Run IDs and timestamps are replaced with placeholders.",
        "",
    ]
    redacted = SuiteResult.model_validate(_redact_sensitive(sanitized.model_dump()))
    lines.append(_render_markdown(redacted, include_generated_at=False))

    path = os.path.join(output_dir, "evaluation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
