#!/usr/bin/env python3
"""
CLI 入口：运行单个 HarnessCase。

用法：
    PYTHONPATH=. python -m harness.run_case --case happy_path
    PYTHONPATH=. python -m harness.run_case --case tool_exception --output-dir /tmp/reports
"""

import argparse
import asyncio
import copy
import os
import sys

from harness.cases.definitions import ALL_CASES
from harness.runner import CaseRunner, SuiteRunner
from harness.hooks import HookBus
from harness.models import SuiteResult
from harness.report import write_json_report, write_markdown_report


def main():
    parser = argparse.ArgumentParser(description="Run a single HarnessCase")
    parser.add_argument("--case", required=True, help="Case ID")
    parser.add_argument("--backend", default=None, help="Override backend")
    parser.add_argument("--agent-mode", default=None, dest="agent_mode", help="Override agent_mode")
    parser.add_argument("--output-dir", default="harness/reports", dest="output_dir")
    args = parser.parse_args()

    # Find case — deep copy to avoid mutating global ALL_CASES
    case = None
    for c in ALL_CASES:
        if c.id == args.case:
            case = copy.deepcopy(c)
            break

    if case is None:
        print(f"Error: Unknown case '{args.case}'. Available: {[c.id for c in ALL_CASES]}")
        sys.exit(1)

    if args.backend:
        case.backend = args.backend
    if args.agent_mode:
        case.request.agent_mode = args.agent_mode

    try:
        hook_bus = HookBus()
        result = asyncio.run(CaseRunner(hook_bus=hook_bus).run(case))
    except Exception as e:
        print(f"Error running case: {e}")
        sys.exit(1)

    # Produce single-case report
    os.makedirs(args.output_dir, exist_ok=True)
    suite = SuiteResult(
        suite_name=f"Single Case: {case.id}",
        backend=case.backend,
        agent_mode=case.request.agent_mode,
        total_cases=1,
        passed_cases=1 if result.passed else 0,
        failed_cases=0 if result.passed else 1,
        results=[result],
    )
    json_path = write_json_report(suite, args.output_dir)
    md_path = write_markdown_report(suite, args.output_dir)

    # Print summary
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n[{status}] {result.case_id} — {result.description}")
    print(f"  Status: {result.status} | Latency: {result.total_latency_ms}ms")
    print(f"  Retry: {result.retry_count} | Replan: {result.replan_count}")
    print(f"  Tools: {result.tools_called}")
    print(f"  Reports: {json_path}")
    print(f"           {md_path}")

    if result.expectation_results:
        print("  Expectations:")
        for er in result.expectation_results:
            icon = "✅" if er.passed else "❌"
            print(f"    {icon} {er.name}: expected={er.expected} actual={er.actual} {er.reason}")

    if result.error:
        print(f"  Error: {result.error}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
