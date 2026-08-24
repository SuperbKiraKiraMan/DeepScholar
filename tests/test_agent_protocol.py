"""显式 Agent 协议、角色权限与 Runtime 审计测试。"""

import pytest

from app.agents.protocol import AgentProtocol, AgentProtocolViolation
from app.agents.schemas import AgentResult, AgentRole, AgentTaskStatus
from app.observability.lifecycle import reset_execution_context, set_execution_context
from app.tools.base import BaseTool, ToolResult
from harness.cases.definitions import HAPPY_PATH
from harness.runner import CaseRunner


def _search_input():
    return {"query": "RAG evaluation", "max_sources": 3, "seed_sources": []}


def _reading_input(operation: str = "metadata_and_quality"):
    return {"topic": "RAG evaluation", "sources": [], "operation": operation}


def test_role_permissions_are_explicit_and_least_privilege():
    protocol = AgentProtocol()
    assert "academic_search" in protocol.allowed_tools_for_role(AgentRole.SEARCH)
    assert set(protocol.allowed_tools_for_role(AgentRole.READING)) == {
        "paper_metadata", "source_quality_scorer", "evidence_extract",
    }
    assert protocol.allowed_tools_for_role(AgentRole.CITATION) == ["citation_check"]
    assert protocol.allowed_tools_for_role(AgentRole.REVIEWER) == []


@pytest.mark.parametrize(
    ("role", "tool_name"),
    [
        (AgentRole.SEARCH, "citation_check"),
        (AgentRole.READING, "semantic_scholar_search"),
        (AgentRole.CITATION, "evidence_extract"),
        (AgentRole.REVIEWER, "academic_search"),
    ],
)
def test_cross_role_tool_calls_are_denied(role, tool_name):
    with pytest.raises(AgentProtocolViolation) as error:
        AgentProtocol().authorize_tool(role, tool_name)
    assert error.value.code == "TOOL_PERMISSION_DENIED"


def test_invalid_input_and_output_are_rejected_by_role_schema():
    protocol = AgentProtocol()
    with pytest.raises(AgentProtocolViolation) as input_error:
        protocol.create_task(
            task_id="search", role=AgentRole.SEARCH,
            input_data={"query": "", "unexpected": True},
        )
    assert input_error.value.code == "INVALID_AGENT_INPUT"

    task = protocol.create_task(
        task_id="search", role=AgentRole.SEARCH,
        input_data=_search_input(), allowed_tools=["academic_search"],
    )
    with pytest.raises(AgentProtocolViolation) as output_error:
        protocol.create_result(task, output_data={"unknown": []})
    assert output_error.value.code == "INVALID_AGENT_OUTPUT"


def test_task_grant_is_narrower_than_role_permission():
    protocol = AgentProtocol()
    task = protocol.create_task(
        task_id="read", role=AgentRole.READING,
        input_data=_reading_input(), allowed_tools=["paper_metadata"],
    )
    with pytest.raises(AgentProtocolViolation) as error:
        protocol.create_result(
            task,
            output_data={"sources": [], "evidence_cards": []},
            tool_calls=[{"tool_name": "evidence_extract", "success": True}],
        )
    assert error.value.code == "TOOL_NOT_GRANTED_FOR_TASK"


def test_result_must_match_task_identity_and_role():
    protocol = AgentProtocol()
    task = protocol.create_task(
        task_id="cite", role=AgentRole.CITATION,
        input_data={"sources": [], "evidence_cards": []},
        allowed_tools=["citation_check"],
    )
    mismatched = AgentResult(
        task_id="other",
        role=AgentRole.CITATION,
        status=AgentTaskStatus.SUCCESS,
        output_data={"citation_check_results": [], "citation_summary": {}},
    )
    with pytest.raises(AgentProtocolViolation) as error:
        protocol.validate_result(task, mismatched)
    assert error.value.code == "TASK_RESULT_MISMATCH"


@pytest.mark.asyncio
async def test_base_tool_denies_before_business_execution():
    executed = False

    class GuardedSearchTool(BaseTool):
        @property
        def name(self):
            return "academic_search"

        @property
        def description(self):
            return "用于验证执行前权限门控"

        async def _arun(self, **kwargs):
            nonlocal executed
            executed = True
            return ToolResult(success=True, data={"results": []})

    token = set_execution_context(
        agent_role="reviewer",
        protocol_task_id="reviewer-test",
        allowed_tools=[],
    )
    try:
        result = await GuardedSearchTool().run(query="RAG")
    finally:
        reset_execution_context(token)

    assert not result.success
    assert "TOOL_PERMISSION_DENIED" in result.error
    assert executed is False


@pytest.mark.asyncio
async def test_runtime_and_harness_validate_all_formal_roles():
    result = await CaseRunner().run(HAPPY_PATH)
    assert result.passed
    protocol_events = [
        hook for hook in result.expectation_results
        if hook.name.startswith("required_protocol_role:")
    ]
    assert {item.name.rsplit(":", 1)[-1] for item in protocol_events} == {
        "search", "reading", "citation", "reviewer",
    }
    assert all(item.passed for item in protocol_events)


def test_context_builders_enforce_role_visibility():
    protocol = AgentProtocol()
    source = {"source_id": "s1", "title": "论文", "full_text": "敏感全文",
              "snippet": "摘要", "quality_score": 0.9}
    search = protocol.build_search_input(
        query="agent", topic="agent", max_results=5,
        providers=["academic_search"], seed_sources=[source],
    )
    assert "full_text" not in str(search)
    assert search["search_constraints"]["exclude_ids"] == ["s1"]

    reading = protocol.build_reading_input(source, reading_questions=["核心结论？"])
    assert "sources" not in reading
    assert reading["source"]["source_id"] == "s1"

    reviewer = protocol.build_reviewer_input(
        topic="agent", stage="draft", sources=[source], evidence_cards=[],
        citation_summary={}, outline={},
    )
    assert "full_text" not in reviewer["source_metadata"][0]
    assert "snippet" not in reviewer["source_metadata"][0]
    assert reviewer["source_metadata"][0]["content_available"] is True


def test_standard_failure_result_propagates_error_code():
    protocol = AgentProtocol()
    task = protocol.create_task(
        task_id="read:s1", role=AgentRole.READING,
        input_data=protocol.build_reading_input({"source_id": "s1"}),
        allowed_tools=["evidence_extract"], input_resource_ids=["s1"],
    )
    result = protocol.create_result(
        task, output_data={}, status=AgentTaskStatus.FAILED,
        error=protocol.error("READ_NO_FULL_TEXT", recoverable=True),
    )
    assert result.task_id == "read:s1"
    assert result.error.error_code == "READ_NO_FULL_TEXT"
    assert result.error.recoverable is True
