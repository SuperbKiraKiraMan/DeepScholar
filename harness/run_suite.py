#!/usr/bin/env python3
"""
CLI 入口：运行完整的 Harness 测试套件。

用法：
    PYTHONPATH=. python -m harness.run_suite
    PYTHONPATH=. python -m harness.run_suite --backend graph_send --agent-mode rule
"""

import argparse
import asyncio
import copy
import sys

from harness.cases.definitions import ALL_CASES
from harness.runner import SuiteRunner
from harness.report import write_json_report, write_markdown_report, write_stable_example_report


def main():
    """主函数，解析命令行参数并运行测试套件。"""
    parser = argparse.ArgumentParser(description="Run the Agent Harness Suite")
    parser.add_argument("--backend", default=None, help="Override backend for all cases")
    parser.add_argument("--agent-mode", default=None, dest="agent_mode", help="Override agent_mode")
    parser.add_argument("--output-dir", default="harness/reports", dest="output_dir")
    parser.add_argument("--case", default=None, help="Run only a specific case ID")
    args = parser.parse_args()

    # ---- 选取用例：指定单个 or 全量，深拷贝防止污染全局 ALL_CASES ----
    if args.case:
        cases = [copy.deepcopy(c) for c in ALL_CASES if c.id == args.case]
        if not cases:
            print(f"Error: Unknown case '{args.case}'. Available: {[c.id for c in ALL_CASES]}")
            sys.exit(1)
    else:
        cases = [copy.deepcopy(c) for c in ALL_CASES]

    for c in cases:
        if args.backend:
            c.backend = args.backend
        if args.agent_mode:
            c.request.agent_mode = args.agent_mode

    runner = SuiteRunner(cases)
    errors = runner.validate()
    if errors:
        for e in errors:
            print(f"Validation error: {e}")
        sys.exit(1)

    try:
        # asyncio.run() 异步的驱动方式，确保所有用例完成后再返回
        suite = asyncio.run(runner.run())
    except Exception as e:
        print(f"Suite error: {e}")
        sys.exit(1)

    json_path = write_json_report(suite, args.output_dir)
    md_path = write_markdown_report(suite, args.output_dir)
    write_stable_example_report(suite, "docs")

    pct = round(suite.passed_cases / max(suite.total_cases, 1) * 100, 1)
    print(f"\n{'='*60}")
    print(f"Suite: {suite.suite_name}")
    print(f"Passed: {suite.passed_cases}/{suite.total_cases} ({pct}%)")
    print(f"Backend: {suite.backend} | Agent Mode: {suite.agent_mode}")
    print(f"Reports: {json_path}")
    print(f"         {md_path}")
    print(f"{'='*60}")

    for cr in suite.results:
        icon = "✅" if cr.passed else "❌"
        print(f"  {icon} {cr.case_id}: {cr.status} ({cr.total_latency_ms}ms)")

    sys.exit(0 if suite.failed_cases == 0 else 1)


if __name__ == "__main__":
    main()
