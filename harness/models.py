"""
harness/models.py

Harness 数据模型 — Pydantic v2，包含单轮运行和多轮会话验收契约。
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_BACKENDS = {"loop", "graph_send"}
VALID_AGENT_MODES = {"rule", "llm"}


# ================================================================
# Harness Request — mirrors ResearchRequest
# ================================================================

class HarnessRequest(BaseModel):
    topic: str = Field(default="RAG evaluation methods", min_length=1)
    language: str = Field(default="zh")
    max_sources: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="quick")
    run_eval: bool = Field(default=True)
    agent_mode: str = Field(default="rule")

    @field_validator("agent_mode")
    @classmethod
    def valid_agent_mode(cls, v):
        if v not in VALID_AGENT_MODES:
            raise ValueError(f"Invalid agent_mode: '{v}'. Valid: {VALID_AGENT_MODES}")
        return v


# ================================================================
# HarnessCase
# ================================================================

class MetricAssertion(BaseModel):
    name: str = Field(..., min_length=1)
    expected: Any
    op: str = Field(default="eq", description="eq | gte | lte | gt | lt | ne")

    @field_validator("op")
    @classmethod
    def valid_op(cls, v):
        if v not in {"eq", "gte", "lte", "gt", "lt", "ne"}:
            raise ValueError(f"Invalid op: '{v}'")
        return v


class HarnessCase(BaseModel):
    id: str = Field(..., min_length=1, description="Unique case identifier")
    description: str = Field(default="")
    request: HarnessRequest = Field(default_factory=HarnessRequest)
    backend: str = Field(default="graph_send")
    fixture_profile: str = Field(default="default", description="Fixture profile to load")
    expected_metrics: List[MetricAssertion] = Field(default_factory=list)
    expected_tools: List[str] = Field(default_factory=list, description="Tools that must be called")
    required_trace_events: List[str] = Field(default_factory=list)
    forbidden_trace_events: List[str] = Field(default_factory=list)
    max_retry_count: int = Field(default=1)
    max_replan_count: int = Field(default=1)
    min_retry_count: int = Field(default=0, ge=0)
    min_replan_count: int = Field(default=0, ge=0)
    min_failed_workers: int = Field(default=0, ge=0)
    min_successful_workers: int = Field(default=0, ge=0)
    min_failed_tools: int = Field(default=0, ge=0)
    min_on_error_hooks: int = Field(default=0, ge=0)
    expect_sources_empty: Optional[bool] = None
    require_citation_recovery: bool = False
    required_hooks: List[str] = Field(default_factory=list)
    required_protocol_roles: List[str] = Field(default_factory=list)
    expected_status: str = Field(default="completed")

    @field_validator("backend")
    @classmethod
    def valid_backend(cls, v):
        if v not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend: '{v}'. Valid: {VALID_BACKENDS}")
        return v

    @field_validator("expected_status")
    @classmethod
    def valid_status(cls, v):
        valid = {"completed", "completed_with_warnings", "partial", "failed", "cancelled"}
        if v not in valid:
            raise ValueError(f"Invalid expected_status: '{v}'. Valid: {valid}")
        return v


# ================================================================
# HookRecord
# ================================================================

class HookRecord(BaseModel):
    hook_name: str
    stage: str  # before_run, after_plan, before_tool, after_tool, after_run, on_error
    timestamp_ms: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str = ""


# ================================================================
# CaseResult
# ================================================================

class ExpectationResult(BaseModel):
    name: str
    expected: Any
    actual: Any
    passed: bool
    reason: str = ""


class CaseResult(BaseModel):
    case_id: str
    description: str = ""
    passed: bool = False
    status: str = ""
    expected_status: str = ""
    run_id: str = ""
    backend: str = ""
    agent_mode: str = ""
    total_latency_ms: int = 0
    retry_count: int = 0
    replan_count: int = 0
    # default_factory=dict 确保在每个实例例中创建新的字典
    eval_metrics: Dict[str, Any] = Field(default_factory=dict)
    eval_metric_details: Dict[str, Any] = Field(default_factory=dict)
    expectation_results: List[ExpectationResult] = Field(default_factory=list)
    tools_called: List[str] = Field(default_factory=list)
    trace_events: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    unresolved_issues: List[str] = Field(default_factory=list)
    hooks: List[HookRecord] = Field(default_factory=list)
    fixes_applied: List[str] = Field(default_factory=list)
    # 记录 FakeLLM fixture 的调用通道和耗尽情况，防止异常响应被静默吞掉。
    llm_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""


# ================================================================
# SuiteResult
# ================================================================

class SuiteResult(BaseModel):
    suite_name: str = "Agent Harness Suite"
    backend: str = "graph_send"
    agent_mode: str = "rule"
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    results: List[CaseResult] = Field(default_factory=list)
    metric_pass_rates: Dict[str, float] = Field(default_factory=dict)
    hook_summary: Dict[str, int] = Field(default_factory=dict)
    known_limitations: List[str] = Field(default_factory=list)


# ================================================================
# 多轮会话 Harness
# ================================================================

class ConversationTurn(BaseModel):
    """会话场景中的一轮协议请求及其验收条件。"""

    id: str = Field(..., min_length=1)
    request: HarnessRequest = Field(default_factory=HarnessRequest)
    parallel_requests: int = Field(default=1, ge=1, le=10)
    expire_session_before: bool = False
    expected_http_status: int = Field(default=200, ge=100, le=599)
    expected_error_code: str = ""
    expected_intent: str = ""
    expected_execution_route: str = ""
    expected_statuses: List[str] = Field(default_factory=lambda: ["completed", "completed_with_warnings"])
    expected_follow_up: Optional[bool] = None
    expect_answer: Optional[bool] = None
    expect_empty_batch: bool = False
    expect_seed_from_session: bool = False
    min_new_papers: int = Field(default=0, ge=0)
    max_new_papers: Optional[int] = Field(default=None, ge=0)
    required_hooks: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_protocol_expectations(self):
        if self.max_new_papers is not None and self.max_new_papers < self.min_new_papers:
            raise ValueError("max_new_papers 不能小于 min_new_papers")
        if self.expected_http_status != 200 and self.parallel_requests != 1:
            raise ValueError("异常协议轮次不支持并发请求")
        return self


class ConversationScenario(BaseModel):
    """共享一个 Session 的完整多轮验收场景。"""

    id: str = Field(..., min_length=1)
    description: str = ""
    turns: List[ConversationTurn] = Field(..., min_length=1)
    backend: str = "graph_send"
    fixture_profile: str = "conversation_default"
    ttl_minutes: int = Field(default=30, ge=1, le=1440)
    expect_unique_papers: bool = True
    expect_isolated_session: bool = True
    expected_final_turn_count: Optional[int] = Field(default=None, ge=0)
    min_final_papers: int = Field(default=0, ge=0)

    @field_validator("backend")
    @classmethod
    def valid_conversation_backend(cls, value):
        if value not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend: '{value}'. Valid: {VALID_BACKENDS}")
        return value


class ConversationTurnResult(BaseModel):
    """单轮协议执行结果；并发轮次会同时保留每个响应摘要。"""

    turn_id: str
    passed: bool = False
    http_statuses: List[int] = Field(default_factory=list)
    intents: List[str] = Field(default_factory=list)
    execution_routes: List[str] = Field(default_factory=list)
    run_ids: List[str] = Field(default_factory=list)
    session_ids: List[str] = Field(default_factory=list)
    seed_paper_ids: List[List[str]] = Field(default_factory=list)
    source_counts: List[int] = Field(default_factory=list)
    new_paper_count: int = 0
    session_turn_count: int = 0
    session_paper_count: int = 0
    expectations: List[ExpectationResult] = Field(default_factory=list)
    hooks: List[HookRecord] = Field(default_factory=list)
    error: str = ""


class ConversationScenarioResult(BaseModel):
    """多轮场景结果和最终 Session 快照。"""

    scenario_id: str
    description: str = ""
    passed: bool = False
    session_id: str = ""
    total_latency_ms: int = 0
    turns: List[ConversationTurnResult] = Field(default_factory=list)
    final_turn_count: int = 0
    final_paper_count: int = 0
    final_paper_keys: List[str] = Field(default_factory=list)
    expectations: List[ExpectationResult] = Field(default_factory=list)
    hook_summary: Dict[str, int] = Field(default_factory=dict)
    error: str = ""


class ConversationSuiteResult(BaseModel):
    """多轮场景套件汇总。"""

    suite_name: str = "Conversation Harness Suite"
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0
    results: List[ConversationScenarioResult] = Field(default_factory=list)
