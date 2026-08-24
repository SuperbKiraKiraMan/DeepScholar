"""显式 Agent 协议的数据模型与角色输入输出 Schema。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentRole(str, Enum):
    """Agent 身份与旧版专项角色。

    新运行时只使用前四个角色；SEARCH/READING/CITATION 是协议 1.0
    的兼容值，避免历史运行记录和外部调用立即失效。
    """

    CONTROLLER = "controller"
    PLANNER = "planner"
    WORKER = "worker"
    REVIEWER = "reviewer"

    SEARCH = "search"
    READING = "reading"
    CITATION = "citation"


class WorkerProfile(str, Enum):
    """Worker 的最小能力配置；它不是独立 Agent 身份。"""

    DIRECT = "direct"
    CONTEXT_LOAD = "context_load"
    SEARCH = "search"
    READ = "read"
    METADATA = "metadata"
    ANALYZE = "analyze"
    CITE = "cite"
    CITATION = "cite"
    ANSWER = "answer"
    WRITE = "write"


class ExecutionClass(str, Enum):
    """Controller 允许选择的三种执行复杂度。"""

    ATOMIC = "atomic"
    CONTEXTUAL = "contextual"
    RESEARCH = "research"


class WorkerStrategy(str, Enum):
    """Worker 内部执行策略；不会暴露模型思维链。"""

    DETERMINISTIC = "deterministic"
    REACT = "react"
    SYNTHESIS = "synthesis"


class ReviewOutcome(str, Enum):
    """Reviewer 对一次计划执行的唯一结构化结论。"""

    PASS = "pass"
    REPAIR = "repair"
    REPLAN = "replan"
    CLARIFY = "clarify"
    FAIL = "fail"


class ExecutionBudget(BaseModel):
    """一次执行的总预算和单 Worker 上限。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_workers: int = Field(default=8, ge=1, le=64)
    max_tool_calls: int = Field(default=120, ge=0, le=200)
    max_iterations: int = Field(default=8, ge=1, le=50)
    per_tool_timeout_ms: int = Field(default=30_000, ge=1)
    total_timeout_ms: int = Field(default=120_000, ge=1)


class SafetyPolicy(BaseModel):
    """Controller、Planner 与 Worker 共用的强类型安全策略。"""

    model_config = ConfigDict(extra="forbid")

    allow_business_tools: bool = True
    allow_network: bool = True
    allow_external_writes: bool = False
    allow_destructive_actions: bool = False
    require_explicit_resources: bool = True
    denied_tools: List[str] = Field(default_factory=list)

    @field_validator("denied_tools")
    @classmethod
    def unique_denied_tools(cls, values: List[str]) -> List[str]:
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


class ExecutionSpec(BaseModel):
    """Controller 的完整输出；不包含任何任务规划。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: str = Field(default="request", min_length=1)
    user_request: str = Field(..., min_length=1)
    intent: str = Field(..., min_length=1)
    execution_class: ExecutionClass
    # 兼容字段：API/SSE 继续使用 direct_tool/conversation/full_research。
    execution_route: str = Field(..., min_length=1)
    research_topic: str = Field(..., min_length=1)
    resource_ids: List[str] = Field(default_factory=list)
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    selected_tools: List[str] = Field(default_factory=list)
    selected_tool_args: Dict[str, Any] = Field(default_factory=dict)
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    safety_policy: SafetyPolicy = Field(default_factory=SafetyPolicy)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkItem(BaseModel):
    """Planner 分发给 Worker 的自包含任务信封。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(..., min_length=1)
    profile: WorkerProfile
    instruction: str = Field(..., min_length=1)
    depends_on: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=list)
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    strategy: Optional[WorkerStrategy] = None
    max_tool_calls: int = Field(default=120, ge=0, le=200)
    max_iterations: int = Field(default=8, ge=1, le=50)
    per_tool_timeout_ms: int = Field(default=30_000, ge=1)
    timeout_ms: int = Field(default=120_000, ge=1)
    safety_policy: SafetyPolicy = Field(default_factory=SafetyPolicy)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_tools", "depends_on")
    @classmethod
    def unique_work_item_values(cls, values: List[str]) -> List[str]:
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    @model_validator(mode="after")
    def validate_work_item(self):
        if self.task_id in self.depends_on:
            raise ValueError("WorkItem 不能依赖自身")
        if self.profile in {WorkerProfile.ANSWER, WorkerProfile.WRITE} and self.allowed_tools:
            raise ValueError("answer/write Worker 不允许业务工具")
        if self.profile == WorkerProfile.CITE and self.strategy not in {None, WorkerStrategy.DETERMINISTIC}:
            raise ValueError("Citation Worker 必须使用 deterministic 策略")
        return self


class WorkPlan(BaseModel):
    """Planner 的 DAG 输出。"""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(default="plan", min_length=1)
    execution_spec: ExecutionSpec
    items: List[WorkItem] = Field(default_factory=list)
    replan_count: int = Field(default=0, ge=0, le=1)
    repair_count: int = Field(default=0, ge=0, le=1)
    revision: int = Field(default=0, ge=0, le=1)
    reviewer_feedback: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dag(self):
        ids = [item.task_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("WorkPlan task_id 必须唯一")
        known = set(ids)
        for item in self.items:
            missing = set(item.depends_on) - known
            if missing:
                raise ValueError(f"WorkItem 依赖不存在: {sorted(missing)}")
        # 关键步骤：使用三色 DFS 拒绝自洽但成环的依赖图。
        dependencies = {item.task_id: list(item.depends_on) for item in self.items}
        colors: Dict[str, int] = {task_id: 0 for task_id in ids}

        def visit(task_id: str) -> None:
            if colors[task_id] == 1:
                raise ValueError("WorkPlan DAG 不能包含环")
            if colors[task_id] == 2:
                return
            colors[task_id] = 1
            for dependency in dependencies[task_id]:
                visit(dependency)
            colors[task_id] = 2

        for task_id in ids:
            visit(task_id)
        return self


class WorkerResult(BaseModel):
    """Worker 的隔离结果；needs_replan 只能由 Worker 提议。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1)
    profile: WorkerProfile
    status: AgentTaskStatus
    output_data: Dict[str, Any] = Field(default_factory=dict)
    resource_ids_produced: List[str] = Field(default_factory=list)
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    needs_replan: bool = False
    action_rationale: List[str] = Field(default_factory=list)
    error: Optional[AgentError] = None
    iterations: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_worker_failure(self):
        if self.status in {AgentTaskStatus.FAILED, AgentTaskStatus.TIMEOUT} and self.error is None:
            raise ValueError("失败的 WorkerResult 必须提供 error")
        return self


class ReviewVerdict(BaseModel):
    """Reviewer 的统一验收输出。"""

    model_config = ConfigDict(extra="forbid")

    outcome: ReviewOutcome
    failed_task_ids: List[str] = Field(default_factory=list)
    repair_scope: List[str] = Field(default_factory=list)
    feedback: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)
    clarification: str = ""
    summary: str = ""


class AgentTaskStatus(str, Enum):
    """AgentResult 可使用的稳定状态集合。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class SearchConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_results: int = Field(default=10, ge=1, le=50)
    year_from: Optional[int] = Field(default=None, ge=1800, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1800, le=2100)
    providers: List[str] = Field(default_factory=lambda: ["academic_search"])
    exclude_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_range(self):
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from 不能大于 year_to")
        return self


class AgentEnvelopePayload(BaseModel):
    """四角色控制面信封；具体载荷由 ExecutionSpec/WorkPlan 等模型校验。"""

    model_config = ConfigDict(extra="allow")


class SearchAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    search_constraints: SearchConstraints = Field(default_factory=SearchConstraints)


class SearchAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: List[Dict[str, Any]] = Field(default_factory=list)
    search_queries_used: List[str] = Field(default_factory=list)
    total_found: int = Field(default=0, ge=0)
    provider_used: str = ""


class ReadingAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Dict[str, Any]
    reading_questions: List[str] = Field(default_factory=list)
    extract_full_text: bool = True
    citation_feedback: List[str] = Field(default_factory=list)


class ReadingAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_cards: List[Dict[str, Any]] = Field(default_factory=list)
    source_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reading_coverage: Literal["full_text", "abstract_only", "metadata_only"]


class GeneratedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    content: str
    evidence_ids_used: List[str] = Field(default_factory=list)
    unanswered_questions: List[str] = Field(default_factory=list)


class CitationAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_cards: List[Dict[str, Any]] = Field(default_factory=list)


class CitationAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_results: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    invalid_card_ids: List[str] = Field(default_factory=list)


class ReviewerAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=1)
    stage: Literal["chapter", "draft", "final"]
    evidence_cards: List[Dict[str, Any]] = Field(default_factory=list)
    source_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    citation_summary: Dict[str, Any] = Field(default_factory=dict)
    outline: Dict[str, Any] = Field(default_factory=dict)
    draft_report: str = ""
    language: str = "zh"


class ReviewerAgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: str = Field(..., min_length=1)
    sections: List[GeneratedSection] = Field(default_factory=list)
    conclusion: str = ""
    evidence_coverage_report: Dict[str, Any] = Field(default_factory=dict)
    completion_ready: bool = False
    issues: List[str] = Field(default_factory=list)


class AgentError(BaseModel):
    """跨角色传播的标准错误结构。"""

    error_code: str
    message: str
    recoverable: bool = False
    suggested_action: str = ""


class AgentToolCall(BaseModel):
    """AgentResult 中可审计的实际工具调用摘要。"""

    tool_name: str = Field(..., min_length=1)
    success: bool = True
    latency_ms: int = Field(default=0, ge=0)
    error: str = ""


class AgentTask(BaseModel):
    """Runtime 分发给角色的统一任务信封。"""

    protocol_version: str = "1.0"
    task_id: str = Field(..., min_length=1)
    role: AgentRole
    instruction: str = ""
    input_resource_ids: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    input_data: Dict[str, Any]
    allowed_tools: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    run_id: str = ""
    session_id: str = ""
    parent_task_id: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    timeout_ms: int = Field(default=60_000, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_tools", "depends_on")
    @classmethod
    def unique_ordered_values(cls, values: List[str]) -> List[str]:
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    @model_validator(mode="after")
    def reject_self_dependency(self):
        if self.task_id in self.depends_on:
            raise ValueError("AgentTask 不能依赖自身")
        return self


class AgentResult(BaseModel):
    """角色执行完毕后返回给 Runtime 的统一结果信封。"""

    protocol_version: str = "1.0"
    task_id: str = Field(..., min_length=1)
    role: AgentRole
    status: AgentTaskStatus
    output_data: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[AgentError] = None
    latency_ms: int = Field(default=0, ge=0)
    resource_ids_produced: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_failure_error(self):
        if self.status in {AgentTaskStatus.FAILED, AgentTaskStatus.TIMEOUT} and self.error is None:
            raise ValueError("失败的 AgentResult 必须提供 error")
        return self

    @property
    def tool_calls_made(self) -> List[str]:
        return [item.tool_name for item in self.tool_calls]
