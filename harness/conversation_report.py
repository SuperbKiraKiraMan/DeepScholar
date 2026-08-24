"""多轮会话 Harness 的 JSON 与 Markdown 报告。"""

import json
import os
from datetime import datetime, timezone

from harness.models import ConversationSuiteResult
from harness.report import _redact_sensitive


def write_conversation_reports(suite: ConversationSuiteResult, output_dir: str) -> tuple[str, str]:
    """一次写出机器可读 JSON 和面向人的 Markdown 报告。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"conversation_suite_report_{timestamp}"
    json_path = os.path.join(output_dir, f"{stem}.json")
    markdown_path = os.path.join(output_dir, f"{stem}.md")
    safe_payload = _redact_sensitive(suite.model_dump())
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(safe_payload, file, ensure_ascii=False, indent=2, default=str)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(_render_markdown(ConversationSuiteResult.model_validate(safe_payload)))
    return json_path, markdown_path


def _render_markdown(suite: ConversationSuiteResult) -> str:
    rate = round(suite.passed_scenarios / max(suite.total_scenarios, 1) * 100, 1)
    lines = [
        "# Conversation Harness Suite",
        "",
        f"- 场景：{suite.total_scenarios}",
        f"- 通过：{suite.passed_scenarios}",
        f"- 失败：{suite.failed_scenarios}",
        f"- 通过率：{rate}%",
        "",
    ]
    for scenario in suite.results:
        lines.extend([
            f"## {'✅' if scenario.passed else '❌'} {scenario.scenario_id}",
            "",
            scenario.description,
            "",
            f"Session：`{scenario.session_id}`；耗时：{scenario.total_latency_ms}ms；"
            f"最终轮数：{scenario.final_turn_count}；累计论文：{scenario.final_paper_count}",
            "",
            "| 轮次 | HTTP | Intent | Route | 新增论文 | Session 轮数 | 结果 |",
            "|---|---|---|---|---:|---:|---|",
        ])
        for turn in scenario.turns:
            lines.append(
                f"| {turn.turn_id} | {turn.http_statuses} | {turn.intents} | "
                f"{turn.execution_routes} | {turn.new_paper_count} | "
                f"{turn.session_turn_count} | {'✅' if turn.passed else '❌'} |"
            )
        lines.append("")
        failures = [
            item for turn in scenario.turns for item in turn.expectations if not item.passed
        ] + [item for item in scenario.expectations if not item.passed]
        if failures:
            lines.append("失败断言：")
            lines.append("")
            for item in failures:
                lines.append(f"- `{item.name}`：{item.reason}（实际：{item.actual}）")
            lines.append("")
        if scenario.error:
            lines.extend([f"错误：`{scenario.error}`", ""])
    return "\n".join(lines)
