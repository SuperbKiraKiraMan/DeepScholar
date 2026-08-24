#!/usr/bin/env python3
"""运行多轮会话验收套件。"""

import argparse
import asyncio
import sys

from harness.cases.conversation_definitions import ALL_CONVERSATION_SCENARIOS
from harness.conversation_report import write_conversation_reports
from harness.conversation_runner import ConversationSuiteRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Conversation Harness Suite")
    parser.add_argument("--scenario", default="", help="只运行指定场景 ID")
    parser.add_argument("--output-dir", default="harness/reports", dest="output_dir")
    args = parser.parse_args()

    scenarios = [
        scenario.model_copy(deep=True)
        for scenario in ALL_CONVERSATION_SCENARIOS
        if not args.scenario or scenario.id == args.scenario
    ]
    if not scenarios:
        print(f"Unknown scenario: {args.scenario}")
        sys.exit(1)

    runner = ConversationSuiteRunner(scenarios)
    validation_errors = runner.validate()
    if validation_errors:
        print("\n".join(validation_errors))
        sys.exit(1)

    # 关键步骤：场景必须顺序运行，避免进程级 Fixture 互相污染。
    suite = asyncio.run(runner.run())
    json_path, markdown_path = write_conversation_reports(suite, args.output_dir)
    print(
        f"Conversation Harness: {suite.passed_scenarios}/{suite.total_scenarios} passed\n"
        f"JSON: {json_path}\nMarkdown: {markdown_path}"
    )
    for result in suite.results:
        print(f"  {'✅' if result.passed else '❌'} {result.scenario_id}")
    sys.exit(0 if suite.failed_scenarios == 0 else 1)


if __name__ == "__main__":
    main()
