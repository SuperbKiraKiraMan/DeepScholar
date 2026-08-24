"""
app/llm/schemas.py

用于 LLM 结构化输出校验的 Pydantic schema。

所有 LLM 输出都会通过这些 schema 校验。
校验失败的输出会触发降级到基于规则的实现。
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LLMIntentOutput(BaseModel):
    """Controller's structured natural-language intent decision."""
    intent: str
    execution_route: str
    research_topic: str = Field(..., min_length=1)
    selected_tool: str = ""
    requested_count: int = Field(default=5, ge=1, le=50)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: str = ""


# ================================================================
# LLM Planner 输出
# ================================================================

class LLMSearchTask(BaseModel):
    """LLM Planner 生成的单个搜索任务。"""
    task_id: str = Field(..., description="Unique task ID, e.g. search_1")
    query: str = Field(..., min_length=3, description="Search query string")
    purpose: str = Field(default="", description="Why this query is needed")
    depends_on: List[str] = Field(
        default_factory=list, description="IDs of tasks this task depends on"
    )
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["academic_search"],
        description="Tool names from the tool registry",
    )


class LLMPlannerOutput(BaseModel):
    """LLM Planner 的结构化输出。"""
    research_goal: str = Field(..., min_length=10)
    search_tasks: List[LLMSearchTask] = Field(..., min_length=1) # 最少一个search task

    def validate_tools(self, available_tool_names: List[str]) -> List[str]:
        # 二次校验：LLM不能编造不存在的工具名
        """校验所有 allowed_tools 是否存在于工具注册表中。"""
        errors = []
        for task in self.search_tasks:
            for tool in task.allowed_tools:
                if tool not in available_tool_names:
                    errors.append(
                        f"Task '{task.task_id}' references unknown tool '{tool}'. "
                        f"Available: {available_tool_names}"
                    )
        return errors


class LLMWorkItemCandidate(BaseModel):
    """LLM 只能建议任务结构，预算、资源与安全策略仍由运行时注入。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., min_length=1, max_length=64)
    profile: Literal["search", "read", "analyze", "cite", "write"]
    instruction: str = Field(..., min_length=3, max_length=1000)
    depends_on: List[str] = Field(default_factory=list, max_length=16)
    allowed_tools: List[str] = Field(default_factory=list, max_length=8)
    strategy: Literal["deterministic", "react", "synthesis"]


class LLMWorkPlanCandidate(BaseModel):
    """Hybrid Planner 的严格候选计划。"""

    model_config = ConfigDict(extra="forbid")
    items: List[LLMWorkItemCandidate] = Field(..., min_length=1, max_length=16)


class LLMSemanticFeedback(BaseModel):
    """不包含思维链的可执行语义审查建议。"""

    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(default="review", min_length=1, max_length=64)
    code: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=1000)


class LLMSemanticReviewOutput(BaseModel):
    """Hybrid Reviewer 只接受收紧式 verdict，不接收正文。"""

    model_config = ConfigDict(extra="forbid")
    outcome: Literal["pass", "repair", "clarify", "fail"]
    failed_task_ids: List[str] = Field(default_factory=list, max_length=16)
    repair_scope: List[str] = Field(default_factory=list, max_length=16)
    feedback: List[LLMSemanticFeedback] = Field(default_factory=list, max_length=16)
    clarification: str = Field(default="", max_length=1000)
    summary: str = Field(default="", max_length=1000)


class LLMSourceSelectionOutput(BaseModel):
    """Model decision for how many discovered papers deserve full analysis."""
    # The candidate pool (and an optional deployment guard) bounds this value at
    # runtime.  Keeping a fixed Pydantic maximum here recreated the exact
    # "retrieved many, always analysed N" failure this decision stage replaces.
    analysis_count: int = Field(..., ge=1)
    selected_source_ids: List[str] = Field(default_factory=list)
    selection_reasons: Dict[str, str] = Field(default_factory=dict)
    coverage_plan: Dict[str, List[str]] = Field(default_factory=dict)
    rationale: str = ""


# ================================================================
# LLM 工具调用输出
# ================================================================

class LLMToolSelection(BaseModel):
    """LLM 决定下一步调用哪个工具。"""
    reasoning: str = Field(default="", description="Why this tool was selected")
    tool_name: str = Field(..., description="Name of the tool to call")
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )
    finish: bool = Field(
        default=False,
        description="True if the LLM wants to finish the task",
    )
    finish_summary: str = Field(
        default="",
        description="Summary of task completion (only when finish=True)",
    )


# ================================================================
# LLM Reviewer Output
# ================================================================

class LLMFinding(BaseModel):
    """LLM Draft Reviewer 生成的单个基于证据的结论。"""
    claim: str = Field(..., min_length=10) # 结论最少10个字符
    source_id: str = Field(..., description="Must be a verified PaperSource.source_id")
    evidence_id: str = Field(
        default="",
        description="Must reference a provided EvidenceCard.evidence_id",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="All EvidenceCard IDs supporting this finding; evidence_id is retained for compatibility",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    analysis: str = Field(
        default="",
        description="Why the finding matters or how it compares with other evidence",
    )

    def validate_source_id(self, valid_source_ids: List[str]) -> Optional[str]:
        # 核心放幻觉：Source_id 必须是已验证的 PaperSource.source_id
        """Return error message if source_id is not in the valid list."""
        if self.source_id not in valid_source_ids:
            return (
                f"Finding references non-existent source_id '{self.source_id}'. "
                f"Valid source_ids: {valid_source_ids[:5]}..."
            )
        return None

    def bound_evidence_ids(self) -> List[str]:
        """Return the de-duplicated old+new evidence bindings."""
        return list(dict.fromkeys([item for item in [self.evidence_id, *self.evidence_ids] if item]))


class LLMSynthesisSection(BaseModel):
    """LLM Draft Reviewer 生成的主题综合段落。"""

    heading: str = Field(..., min_length=3)
    synthesis: str = Field(..., min_length=30)
    evidence_ids: List[str] = Field(..., min_length=1)


class LLMReportOutput(BaseModel):
    """LLM Draft Reviewer 生成的结构化报告。"""
    title: str = Field(..., min_length=5)
    executive_summary: str = Field(default="")
    introduction: str = Field(..., min_length=50)
    synthesis_sections: List[LLMSynthesisSection] = Field(default_factory=list)
    findings: List[LLMFinding] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    limitations: str = Field(default="")
    conclusion: str = Field(default="")


class ReportOutlineSection(BaseModel):
    """One independently grounded report chapter."""
    heading: str = Field(..., min_length=2)
    guiding_question: str = Field(..., min_length=3)
    assigned_evidence_ids: List[str] = Field(default_factory=list)
    assigned_source_ids: List[str] = Field(default_factory=list)
    estimated_length: str = Field(default="medium", pattern="^(short|medium|long)$")


class ReportOutline(BaseModel):
    """Explicit bridge between verified evidence and chapter generation."""
    # The runtime owns the topic and overwrites this field after validation.
    # Keeping it optional prevents an otherwise usable model-created outline
    # from failing only because the model omitted duplicated request metadata.
    topic: str = ""
    evidence_card_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    sections: List[ReportOutlineSection] = Field(default_factory=list)
    cross_cutting_themes: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)


class LLMOutlineAssignment(BaseModel):
    """Compact LLM decision; code derives source bindings and chapter metadata."""
    assignments: Dict[str, List[str]] = Field(default_factory=dict)
    evidence_gaps: List[str] = Field(default_factory=list)


class LLMOutlinePlanSection(BaseModel):
    """Compact chapter decision; runtime derives sources and length metadata."""
    heading: str = Field(..., min_length=2)
    guiding_question: str = Field(..., min_length=3)
    assigned_evidence_ids: List[str] = Field(default_factory=list)


class LLMOutlinePlan(BaseModel):
    """Model-owned headings/questions with runtime-owned source bindings."""
    sections: List[LLMOutlinePlanSection] = Field(default_factory=list)
    cross_cutting_themes: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)

    @field_validator("cross_cutting_themes", mode="before")
    @classmethod
    def normalize_cross_cutting_themes(cls, value: Any) -> List[str]:
        return cls._normalize_text_list(value, ("theme", "name", "title", "text"))

    @field_validator("evidence_gaps", mode="before")
    @classmethod
    def normalize_evidence_gaps(cls, value: Any) -> List[str]:
        return cls._normalize_text_list(
            value, ("gap", "question", "issue", "description", "text"),
        )

    @staticmethod
    def _normalize_text_list(value: Any, preferred_keys: tuple[str, ...]) -> List[str]:
        """Accept common provider object drift while retaining reader-safe text."""
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized: List[str] = []
        for item in items:
            text = ""
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = next(
                    (
                        str(item.get(key) or "").strip()
                        for key in preferred_keys if str(item.get(key) or "").strip()
                    ),
                    "",
                )
            if text.strip() and text.strip() not in normalized:
                normalized.append(text.strip())
        return normalized


class LLMChapterOutput(BaseModel):
    """Structured output for one isolated chapter call."""
    heading: str = Field(..., min_length=2)
    synthesis: str = Field(..., min_length=30)
    evidence_ids: List[str] = Field(default_factory=list)
    findings: List[LLMFinding] = Field(default_factory=list)
    source_title_translations: Dict[str, str] = Field(
        default_factory=dict,
        description="source_id -> Chinese paper title; identifiers and original titles stay unchanged",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_common_model_shapes(cls, value: Any) -> Any:
        """Accept semantically equivalent keys before deciding to use rule fallback.

        Providers commonly return ``chapter_title``/``content`` even when the
        requested schema says ``heading``/``synthesis``.  Those are shape
        differences, not missing research content, so normalize them here while
        retaining the normal field validation afterwards.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        nested = data.get("chapter")
        if isinstance(nested, dict):
            merged = dict(nested)
            merged.update({key: item for key, item in data.items() if key != "chapter"})
            data = merged

        if not data.get("heading"):
            for key in ("chapter_title", "section_title", "section_heading", "title"):
                if data.get(key):
                    data["heading"] = data[key]
                    break
            if not data.get("heading") and isinstance(data.get("chapter"), str):
                data["heading"] = data["chapter"]

        if not data.get("synthesis"):
            for key in ("content", "chapter_content", "body", "analysis", "text", "summary"):
                if data.get(key):
                    data["synthesis"] = data[key]
                    break

        if not data.get("evidence_ids"):
            for key in ("used_evidence_ids", "supporting_evidence_ids", "citations"):
                if data.get(key):
                    data["evidence_ids"] = data[key]
                    break
        if not data.get("evidence_ids") and isinstance(data.get("findings"), list):
            bound_ids = []
            for finding in data["findings"]:
                if not isinstance(finding, dict):
                    continue
                bound_ids.extend(finding.get("evidence_ids") or [])
                if finding.get("evidence_id"):
                    bound_ids.append(finding["evidence_id"])
            if bound_ids:
                data["evidence_ids"] = list(dict.fromkeys(bound_ids))

        # ``findings`` is supplementary metadata; the formal chapter is bound by
        # synthesis-level evidence IDs and paragraph markers.  Some providers
        # return useful dataset/aspect dictionaries or plain strings here instead
        # of LLMFinding objects.  Discard only those malformed optional entries so
        # a fully grounded chapter is not rejected for an unused auxiliary field.
        if isinstance(data.get("findings"), list):
            data["findings"] = [
                finding for finding in data["findings"]
                if isinstance(finding, dict)
                and str(finding.get("claim") or "").strip()
                and str(finding.get("source_id") or "").strip()
            ]
        elif data.get("findings") is not None:
            data["findings"] = []

        if not data.get("source_title_translations"):
            for key in ("title_translations", "translated_titles", "chinese_titles"):
                if isinstance(data.get(key), dict):
                    data["source_title_translations"] = data[key]
                    break
        return data


# ================================================================
# Config Schema
# ================================================================

class LLMConfig(BaseModel):
    """LLM 配置。"""
    provider: str = "deepseek"
    agent_mode: str = "llm"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_seconds: int = 30
    max_retries: int = 1
    temperature: float = 0.1
    max_tool_calls: int = 8
