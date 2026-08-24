"""Harness 多轮会话协议验收测试。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.cases.conversation_definitions import (
    ALL_CONVERSATION_SCENARIOS,
    CONCURRENT_SESSION_UPDATE,
    CROSS_PROVIDER_DUPLICATE,
    EMPTY_RECOMMENDATION_BATCH,
    RECOMMEND_MORE_RESEARCH,
    SESSION_EXPIRED,
)
from harness.conversation_report import write_conversation_reports
from harness.conversation_runner import ConversationScenarioRunner, ConversationSuiteRunner
from harness.models import ConversationScenario, ConversationTurn, HarnessRequest


def test_conversation_turn_rejects_invalid_parallel_error_contract():
    """异常协议轮次不能并发，否则错误响应无法与请求稳定配对。"""
    with pytest.raises(ValidationError):
        ConversationTurn(
            id="invalid",
            request=HarnessRequest(topic="test"),
            parallel_requests=2,
            expected_http_status=410,
        )


def test_conversation_suite_rejects_duplicate_scenario_ids():
    scenario = ConversationScenario(
        id="duplicate",
        turns=[ConversationTurn(id="turn", request=HarnessRequest(topic="test"))],
    )
    errors = ConversationSuiteRunner([scenario, scenario]).validate()
    assert errors == ["Duplicate scenario id: 'duplicate'"]


@pytest.mark.asyncio
async def test_recommend_more_then_research_preserves_session_and_seeds():
    result = await ConversationScenarioRunner().run(RECOMMEND_MORE_RESEARCH)
    assert result.passed
    assert result.final_turn_count == 3
    assert result.final_paper_count >= 5
    assert len({turn.session_ids[0] for turn in result.turns}) == 1
    seed_assertion = next(
        item for item in result.turns[-1].expectations
        if item.name == "session_papers_seeded"
    )
    assert seed_assertion.passed


@pytest.mark.asyncio
async def test_empty_and_cross_provider_batches_obey_dedup_protocol():
    empty = await ConversationScenarioRunner().run(EMPTY_RECOMMENDATION_BATCH)
    duplicate = await ConversationScenarioRunner().run(CROSS_PROVIDER_DUPLICATE)
    assert empty.passed and empty.turns[-1].new_paper_count == 0
    assert empty.turns[-1].source_counts == [0]
    assert duplicate.passed and duplicate.final_paper_count == 3
    assert len(duplicate.final_paper_keys) == len(set(duplicate.final_paper_keys))


@pytest.mark.asyncio
async def test_expiry_and_concurrent_updates_are_protocol_checked():
    expired = await ConversationScenarioRunner().run(SESSION_EXPIRED)
    concurrent = await ConversationScenarioRunner().run(CONCURRENT_SESSION_UPDATE)
    assert expired.passed
    assert expired.turns[0].http_statuses == [410]
    assert expired.final_turn_count == 0
    assert concurrent.passed
    assert concurrent.final_turn_count == 3
    assert concurrent.turns[-1].new_paper_count == 4


@pytest.mark.asyncio
async def test_full_conversation_suite_and_reports(tmp_path):
    # 关键步骤：完整套件必须覆盖文档列出的六类多轮与边界协议。
    suite = await ConversationSuiteRunner(ALL_CONVERSATION_SCENARIOS).run()
    assert suite.total_scenarios == 6
    assert suite.failed_scenarios == 0

    json_path, markdown_path = write_conversation_reports(suite, str(tmp_path))
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    markdown = Path(markdown_path).read_text(encoding="utf-8")
    assert payload["passed_scenarios"] == 6
    assert "recommend_more_then_research" in markdown
    assert "concurrent_session_update" in markdown
