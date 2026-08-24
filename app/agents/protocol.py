"""显式 Agent 协议、角色权限白名单与运行时校验器。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Type

from pydantic import BaseModel, ValidationError

from app.agents.schemas import (
    AgentError,
    AgentEnvelopePayload,
    AgentResult,
    AgentRole,
    AgentTask,
    AgentTaskStatus,
    AgentToolCall,
    CitationAgentInput,
    CitationAgentOutput,
    ReadingAgentInput,
    ReadingAgentOutput,
    ReviewerAgentInput,
    ReviewerAgentOutput,
    SearchAgentInput,
    SearchAgentOutput,
    SearchConstraints,
)
from app.tools.registry import ToolRegistry, retrieval_capability_for_tool


PROTOCOL_VERSION = "1.0"

AGENT_ERROR_CODES = {
    "SEARCH_NO_RESULTS": "搜索无结果",
    "READ_NO_FULL_TEXT": "论文无可用全文",
    "REVIEW_INSUFFICIENT_EVIDENCE": "证据不足以生成报告",
    "CITATION_INVALID": "引用校验失败",
    "TIMEOUT": "执行超时",
    "TOOL_UNAVAILABLE": "所需工具不可用",
    "PERMISSION_DENIED": "Agent 无权调用该工具",
    "INVALID_AGENT_INPUT": "Agent 输入不符合 Schema",
    "INVALID_AGENT_OUTPUT": "Agent 输出不符合 Schema",
}

_REVIEWER_SOURCE_FIELDS = {
    "source_id", "title", "url", "authors", "year", "venue", "source_type",
    "quality_score", "provider", "openalex_id", "semantic_scholar_id", "doi",
    "publication_date", "research_task", "task_relevance", "content_available",
}

# Reviewer 只消费可信状态并生成文本，不允许直接调用检索或引用工具。
ROLE_TOOL_ALLOWLIST: Dict[AgentRole, frozenset[str]] = {
    AgentRole.CONTROLLER: frozenset(),
    AgentRole.PLANNER: frozenset(),
    AgentRole.WORKER: frozenset(),
    AgentRole.SEARCH: frozenset({
        "academic_search",
        "local_paper_search",
        "semantic_scholar_search",
        "semantic_scholar_graph",
        "semantic_scholar_recommendations",
    }),
    AgentRole.READING: frozenset({
        "paper_metadata", "source_quality_scorer", "evidence_extract",
    }),
    AgentRole.CITATION: frozenset({"citation_check"}),
    AgentRole.REVIEWER: frozenset(),
}

ROLE_INPUT_SCHEMAS: Dict[AgentRole, Type[BaseModel]] = {
    AgentRole.CONTROLLER: AgentEnvelopePayload,
    AgentRole.PLANNER: AgentEnvelopePayload,
    AgentRole.WORKER: AgentEnvelopePayload,
    AgentRole.SEARCH: SearchAgentInput,
    AgentRole.READING: ReadingAgentInput,
    AgentRole.CITATION: CitationAgentInput,
    AgentRole.REVIEWER: ReviewerAgentInput,
}

ROLE_OUTPUT_SCHEMAS: Dict[AgentRole, Type[BaseModel]] = {
    AgentRole.CONTROLLER: AgentEnvelopePayload,
    AgentRole.PLANNER: AgentEnvelopePayload,
    AgentRole.WORKER: AgentEnvelopePayload,
    AgentRole.SEARCH: SearchAgentOutput,
    AgentRole.READING: ReadingAgentOutput,
    AgentRole.CITATION: CitationAgentOutput,
    AgentRole.REVIEWER: ReviewerAgentOutput,
}


class AgentProtocolViolation(ValueError):
    """任务、结果或工具权限违反显式协议。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class AgentProtocol:
    """创建并校验 AgentTask / AgentResult 的唯一入口。"""

    def __init__(self, registry: ToolRegistry | None = None):
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """未显式注入时始终读取当前 Registry，兼容测试与 MCP 动态重载。"""
        return self._registry or ToolRegistry.get_instance()

    def allowed_tools_for_role(self, role: AgentRole | str) -> List[str]:
        normalized = AgentRole(role)
        if normalized == AgentRole.SEARCH:
            return sorted(
                name for name in self.registry.list_names()
                if self._role_allows_tool(normalized, name)
            )
        return sorted(
            name for name in ROLE_TOOL_ALLOWLIST[normalized]
            if self.registry.get(name) is not None
        )

    def create_task(
        self,
        *,
        task_id: str,
        role: AgentRole | str,
        input_data: Dict[str, Any],
        instruction: str = "",
        input_resource_ids: Iterable[str] | None = None,
        constraints: Dict[str, Any] | None = None,
        allowed_tools: Iterable[str] | None = None,
        depends_on: Iterable[str] | None = None,
        run_id: str = "",
        session_id: str = "",
        parent_task_id: str | None = None,
        timeout_ms: int = 60_000,
        metadata: Dict[str, Any] | None = None,
    ) -> AgentTask:
        normalized_role = AgentRole(role)
        input_data = self._normalize_legacy_input(normalized_role, input_data)
        normalized_input = self._validate_payload(
            ROLE_INPUT_SCHEMAS[normalized_role], input_data, "INVALID_AGENT_INPUT",
        )
        requested_tools = (
            list(allowed_tools) if allowed_tools is not None
            else self.allowed_tools_for_role(normalized_role)
        )
        task = AgentTask(
            protocol_version=PROTOCOL_VERSION,
            task_id=task_id,
            role=normalized_role,
            instruction=instruction,
            input_resource_ids=list(input_resource_ids or []),
            constraints=dict(constraints or {}),
            input_data=normalized_input,
            allowed_tools=requested_tools,
            depends_on=list(depends_on or []),
            run_id=run_id,
            session_id=session_id,
            parent_task_id=parent_task_id,
            timeout_ms=timeout_ms,
            metadata=dict(metadata or {}),
        )
        self.validate_task(task)
        return task

    def validate_task(self, task: AgentTask) -> None:
        if task.protocol_version != PROTOCOL_VERSION:
            raise AgentProtocolViolation(
                "UNSUPPORTED_PROTOCOL_VERSION",
                f"不支持的协议版本: {task.protocol_version}",
            )
        self._validate_payload(
            ROLE_INPUT_SCHEMAS[task.role], task.input_data, "INVALID_AGENT_INPUT",
        )
        for tool_name in task.allowed_tools:
            self.authorize_tool(task.role, tool_name)

    def authorize_tool(self, role: AgentRole | str, tool_name: str) -> None:
        normalized_role = AgentRole(role)
        if self.registry.get(tool_name) is None:
            raise AgentProtocolViolation(
                "UNKNOWN_TOOL", f"工具未注册: {tool_name}",
            )
        if not self._role_allows_tool(normalized_role, tool_name):
            raise AgentProtocolViolation(
                "TOOL_PERMISSION_DENIED",
                f"角色 {normalized_role.value} 无权调用工具 {tool_name}",
            )

    def create_result(
        self,
        task: AgentTask,
        *,
        output_data: Dict[str, Any],
        status: AgentTaskStatus | str = AgentTaskStatus.SUCCESS,
        tool_calls: Iterable[AgentToolCall | Dict[str, Any]] | None = None,
        warnings: Iterable[str] | None = None,
        error: AgentError | Dict[str, Any] | None = None,
        latency_ms: int = 0,
        resource_ids_produced: Iterable[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AgentResult:
        normalized_status = AgentTaskStatus(status)
        normalized_output = self._normalize_legacy_output(task.role, output_data)
        if normalized_status not in {AgentTaskStatus.FAILED, AgentTaskStatus.TIMEOUT, AgentTaskStatus.CANCELLED}:
            normalized_output = self._validate_payload(
                ROLE_OUTPUT_SCHEMAS[task.role], normalized_output, "INVALID_AGENT_OUTPUT",
            )
        calls = [
            item if isinstance(item, AgentToolCall) else AgentToolCall(**item)
            for item in (tool_calls or [])
        ]
        normalized_error = (
            error if isinstance(error, AgentError)
            else AgentError(**error) if isinstance(error, dict)
            else None
        )
        result = AgentResult(
            protocol_version=PROTOCOL_VERSION,
            task_id=task.task_id,
            role=task.role,
            status=normalized_status,
            output_data=normalized_output,
            tool_calls=calls,
            warnings=list(warnings or []),
            error=normalized_error,
            latency_ms=latency_ms,
            resource_ids_produced=list(resource_ids_produced or []),
            metadata=dict(metadata or {}),
        )
        self.validate_result(task, result)
        return result

    @staticmethod
    def _normalize_legacy_input(role: AgentRole, data: Dict[str, Any]) -> Dict[str, Any]:
        """迁移旧 Runtime 载荷，同时在协议边界裁剪越权上下文。"""
        if role == AgentRole.SEARCH and "search_constraints" not in data:
            seeds = list(data.get("seed_sources") or [])
            return {
                "query": str(data.get("query") or data.get("topic") or ""),
                "topic": str(data.get("topic") or data.get("query") or ""),
                "search_constraints": {
                    "max_results": int(data.get("max_sources", 10) or 10),
                    "providers": ["academic_search"],
                    "exclude_ids": [str(item.get("source_id") or item.get("paper_id"))
                                    for item in seeds if item.get("source_id") or item.get("paper_id")],
                },
            }
        if role == AgentRole.READING and "source" not in data:
            sources = list(data.get("sources") or [])
            return AgentProtocol.build_reading_input(
                sources[0] if sources else {"source_id": "unknown"},
                reading_questions=[str(data.get("operation") or "阅读并提取证据")],
                extract_full_text=data.get("operation") == "evidence_extraction",
            )
        if role == AgentRole.REVIEWER and "source_metadata" not in data:
            return AgentProtocol.build_reviewer_input(
                topic=str(data.get("topic") or "未命名研究"),
                stage=str(data.get("stage") or "draft"),
                sources=data.get("sources", []),
                evidence_cards=data.get("evidence_cards", []),
                citation_summary=data.get("citation_summary", {}),
                outline=data.get("outline", {}),
                draft_report=str(data.get("draft_report") or data.get("report") or ""),
                language=str(data.get("language") or "zh"),
            )
        return dict(data)

    @staticmethod
    def _normalize_legacy_output(role: AgentRole, data: Dict[str, Any]) -> Dict[str, Any]:
        """将旧 Worker 结果转换为稳定的角色输出 Schema。"""
        if role == AgentRole.SEARCH and "discovered_source_count" in data:
            sources = list(data.get("sources") or [])
            return {"sources": sources, "search_queries_used": [],
                    "total_found": int(data.get("discovered_source_count", len(sources))),
                    "provider_used": "mixed"}
        if role == AgentRole.READING and "sources" in data:
            sources = list(data.get("sources") or [])
            source = sources[0] if sources else {}
            coverage = "full_text" if source.get("full_text") else (
                "abstract_only" if source.get("snippet") else "metadata_only"
            )
            return {"evidence_cards": list(data.get("evidence_cards") or []),
                    "source_quality_score": float(source.get("quality_score", 0) or 0),
                    "reading_coverage": coverage}
        if role == AgentRole.CITATION and "citation_check_results" in data:
            checks = list(data.get("citation_check_results") or [])
            return {"check_results": checks, "summary": dict(data.get("citation_summary") or {}),
                    "invalid_card_ids": [str(x.get("source_id")) for x in checks if not x.get("is_valid", False)]}
        if role == AgentRole.REVIEWER and "coverage" in data:
            report = str(data.get("report") or data.get("draft_report") or "")
            return {"report": report or "未生成正文", "sections": [], "conclusion": "",
                    "evidence_coverage_report": dict(data.get("coverage") or {}),
                    "completion_ready": bool(data.get("completion_ready", bool(report))),
                    "issues": list(data.get("issues") or [])}
        return dict(data)

    def validate_result(self, task: AgentTask, result: AgentResult) -> None:
        if result.task_id != task.task_id or result.role != task.role:
            raise AgentProtocolViolation(
                "TASK_RESULT_MISMATCH",
                "AgentResult 的 task_id 或 role 与 AgentTask 不一致",
            )
        if result.status not in {AgentTaskStatus.FAILED, AgentTaskStatus.TIMEOUT, AgentTaskStatus.CANCELLED}:
            self._validate_payload(
                ROLE_OUTPUT_SCHEMAS[result.role], result.output_data, "INVALID_AGENT_OUTPUT",
            )
        allowed = set(task.allowed_tools)
        for call in result.tool_calls:
            self.authorize_tool(result.role, call.tool_name)
            if call.tool_name not in allowed:
                raise AgentProtocolViolation(
                    "TOOL_NOT_GRANTED_FOR_TASK",
                    f"任务 {task.task_id} 未授予工具 {call.tool_name}",
                )

    def build_search_input(
        self,
        *,
        query: str,
        topic: str,
        max_results: int,
        providers: Iterable[str],
        seed_sources: Iterable[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        """只向 Search 暴露搜索约束和排除 ID，不传证据、报告或全文。"""
        exclude_ids = [
            str(item.get("source_id") or item.get("paper_id") or "")
            for item in seed_sources
            if item.get("source_id") or item.get("paper_id")
        ]
        return SearchAgentInput(
            query=query,
            topic=topic,
            search_constraints=SearchConstraints(
                max_results=max_results,
                providers=list(providers),
                exclude_ids=exclude_ids,
            ),
        ).model_dump()

    @staticmethod
    def build_reading_input(
        source: Dict[str, Any],
        *,
        reading_questions: Iterable[str] = (),
        extract_full_text: bool = True,
        citation_feedback: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """Reading 每个任务只接收一篇目标论文和本轮问题。"""
        return ReadingAgentInput(
            source=dict(source),
            reading_questions=list(reading_questions),
            extract_full_text=extract_full_text,
            citation_feedback=list(citation_feedback),
        ).model_dump()

    @staticmethod
    def build_reviewer_input(
        *,
        topic: str,
        stage: str,
        sources: Iterable[Dict[str, Any]],
        evidence_cards: Iterable[Dict[str, Any]],
        citation_summary: Dict[str, Any],
        outline: Dict[str, Any],
        draft_report: str = "",
        language: str = "zh",
    ) -> Dict[str, Any]:
        """Reviewer 只看到来源元数据和证据卡，绝不接收论文全文或工具 Trace。"""
        metadata = []
        for source in sources:
            item = {
                key: value for key, value in source.items()
                if key in _REVIEWER_SOURCE_FIELDS
            }
            item["content_available"] = bool(
                source.get("full_text") or source.get("snippet") or source.get("content_available")
            )
            metadata.append(item)
        return ReviewerAgentInput(
            topic=topic,
            stage=stage,
            evidence_cards=[dict(item) for item in evidence_cards],
            source_metadata=metadata,
            citation_summary=dict(citation_summary or {}),
            outline=dict(outline or {}),
            draft_report=draft_report,
            language=language,
        ).model_dump()

    @staticmethod
    def build_citation_input(
        sources: Iterable[Dict[str, Any]],
        evidence_cards: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Citation 只接收校验所需来源与证据，不接收查询、报告或 Trace。"""
        return CitationAgentInput(
            sources=[dict(item) for item in sources],
            evidence_cards=[dict(item) for item in evidence_cards],
        ).model_dump()

    def error(
        self,
        code: str,
        *,
        message: str = "",
        recoverable: bool = False,
        suggested_action: str = "",
    ) -> AgentError:
        return AgentError(
            error_code=code,
            message=message or AGENT_ERROR_CODES.get(code, code),
            recoverable=recoverable,
            suggested_action=suggested_action,
        )

    def task_from_planner(
        self,
        planner_task: Any,
        *,
        role: AgentRole | str,
        input_data: Dict[str, Any],
        run_id: str = "",
        session_id: str = "",
    ) -> AgentTask:
        """将旧 Task DAG 节点包装成显式 AgentTask，保持 Planner 兼容。"""
        return self.create_task(
            task_id=planner_task.task_id,
            role=role,
            instruction=planner_task.description,
            input_data=input_data,
            allowed_tools=planner_task.tool_plan or None,
            depends_on=planner_task.depends_on,
            run_id=run_id,
            session_id=session_id,
        )

    @staticmethod
    def _validate_payload(
        schema: Type[BaseModel], payload: Dict[str, Any], code: str,
    ) -> Dict[str, Any]:
        try:
            return schema.model_validate(payload).model_dump()
        except ValidationError as exc:
            raise AgentProtocolViolation(code, str(exc)) from exc

    @staticmethod
    def _role_allows_tool(role: AgentRole, tool_name: str) -> bool:
        if role == AgentRole.SEARCH:
            # 关键步骤：动态 MCP 检索工具按 canonical capability 判权，不能仅靠名称前缀放行。
            return bool(retrieval_capability_for_tool(tool_name))
        return tool_name in ROLE_TOOL_ALLOWLIST[role]


agent_protocol = AgentProtocol()
