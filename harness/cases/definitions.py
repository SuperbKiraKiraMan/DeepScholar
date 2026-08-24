"""
25 个覆盖正常链路、边界条件与故障恢复的离线 HarnessCase 用例定义。
每个用例通过 fixture_profile 注入特定故障。
"""

from harness.models import HarnessCase, HarnessRequest, MetricAssertion

_REQ_RULE = HarnessRequest(
    topic="RAG evaluation methods", language="zh",
    max_sources=3, run_eval=True, agent_mode="rule",
)
_REQ_LLM = HarnessRequest(
    topic="RAG evaluation methods", language="zh",
    max_sources=3, run_eval=True, agent_mode="llm",
)

# ================================================================
# Case 1: happy_path — normal execution, all pass
# ================================================================
HAPPY_PATH = HarnessCase(
    id="happy_path",
    description="Normal execution: rule mode, graph_send, all metrics pass",
    request=_REQ_RULE, backend="graph_send", fixture_profile="default",
    expected_metrics=[ # 预期指标断言   
        MetricAssertion(name="no_fake_citation", expected=True),
        MetricAssertion(name="min_sources", expected=True),
        MetricAssertion(name="answer_not_empty", expected=True),
        MetricAssertion(name="latency_under_threshold", expected=True),
    ],
    expected_tools=[ # 必须调用过这些工具
        "academic_search", "paper_metadata", "source_quality_scorer",
        "evidence_extract", "citation_check",
    ],
    required_trace_events=[ # 必须调用过这些事件
        "controller_start", "planner_complete", "send_dispatch",
        "worker_started", "worker_finished", "citation_complete",
        "draft_reviewer_complete", "evaluator_complete",
    ],
    max_retry_count=1, max_replan_count=1, # 最大重试次数和重计划次数不超过 1 次
    required_hooks=["before_run", "after_plan", "before_tool", "after_tool", "after_run"], # 必须调用过这些 Hook
    required_protocol_roles=["search", "reading", "citation", "reviewer"],
    expected_status="completed", # 最终状态为 "completed"
)

# ================================================================
# Case 2: invalid_or_fake_citation — citation error detected & handled
# ================================================================
INVALID_CITATION = HarnessCase(
    id="invalid_or_fake_citation",
    description="Citations checked by CitationCheckTool, retry bounded, issues reflected",
    request=_REQ_RULE, backend="graph_send", fixture_profile="invalid_citation",
    expected_metrics=[
        MetricAssertion(name="no_fake_citation", expected=True),
        MetricAssertion(name="citation_id_exists", expected=True),
    ],
    expected_tools=["citation_check"],
    required_trace_events=["citation_complete", "evaluator_complete"],
    max_retry_count=1, min_retry_count=1, max_replan_count=1,
    require_citation_recovery=True,
    expected_status="completed_with_warnings",
)

# ================================================================
# Case 4: tool_exception — evidence_extract throws
# ================================================================
TOOL_EXCEPTION = HarnessCase(
    id="tool_exception",
    description="evidence_extract 持续异常 → 返回可观察的 source-only 部分结果",
    request=_REQ_RULE, backend="graph_send", fixture_profile="tool_exception",
    expected_metrics=[
        MetricAssertion(name="answer_not_empty", expected=True),
    ],
    required_trace_events=["controller_start", "evaluator_complete"],
    forbidden_trace_events=[],
    max_retry_count=1, max_replan_count=1,
    min_failed_tools=1, min_on_error_hooks=1,
    required_hooks=["before_tool", "after_tool", "on_error"],
    expected_status="partial",
)

# ================================================================
# Case 5: llm_fallback — LLM fails → rule Worker takes over
# ================================================================
LLM_FALLBACK = HarnessCase(
    id="llm_fallback",
    description="LLM Worker fails → falls back to rule Worker, run completes with warnings",
    request=_REQ_LLM, backend="graph_send", fixture_profile="llm_fallback",
    expected_metrics=[
        MetricAssertion(name="answer_not_empty", expected=True),
        MetricAssertion(name="min_sources", expected=True),
    ],
    required_trace_events=[
        "controller_start", "llm_failed", "tool_loop_fallback", "evaluator_complete",
    ],
    expected_tools=[
        "academic_search", "paper_metadata", "source_quality_scorer",
        "evidence_extract", "citation_check",
    ],
    max_retry_count=1, max_replan_count=1, min_on_error_hooks=1,
    required_hooks=["on_error"],
    expected_status="completed_with_warnings",
)

# ================================================================
# Case 6: send_worker_partial_failure — one worker fails, others succeed
# ================================================================
SEND_PARTIAL_FAILURE = HarnessCase(
    id="send_worker_partial_failure",
    description="graph_send: one worker fails, others succeed, merge retains successes",
    request=_REQ_RULE, backend="graph_send", fixture_profile="send_worker_partial_failure",
    expected_metrics=[
        MetricAssertion(name="answer_not_empty", expected=True),
        MetricAssertion(name="min_sources", expected=True),
        MetricAssertion(name="task_success_rate", expected=1.0, op="lt"),
    ],
    required_trace_events=[
        "controller_start", "send_dispatch", "worker_started",
        "worker_finished", "merge_result",
    ],
    max_retry_count=1, max_replan_count=1,
    min_failed_workers=1, min_successful_workers=1, min_failed_tools=1,
    min_on_error_hooks=1, required_hooks=["on_error"],
    expected_status="completed_with_warnings",
)

# ================================================================
# Case 7: latency_threshold_exceeded — forced latency failure
# ================================================================
LATENCY_EXCEEDED = HarnessCase(
    id="latency_threshold_exceeded",
    description="Evaluator threshold set to 1ms → latency_under_threshold=False",
    request=_REQ_RULE, backend="graph_send", fixture_profile="latency_exceeded",
    expected_metrics=[
        MetricAssertion(name="latency_under_threshold", expected=False),
        MetricAssertion(name="answer_not_empty", expected=True),
    ],
    required_trace_events=["controller_start", "evaluator_complete"],
    max_retry_count=1, max_replan_count=1,
    expected_status="completed_with_warnings",
)


# ================================================================
# 扩展 Case：正常链路参数矩阵
# ================================================================

def _clone_case(
    base: HarnessCase,
    *,
    case_id: str,
    description: str,
    backend: str = None,
    fixture_profile: str = None,
    request_updates: dict = None,
    **overrides,
) -> HarnessCase:
    """复制基准 Case 并只修改场景变量，避免不同 Case 共享可变请求对象。"""
    # 关键步骤：每个扩展 Case 都做深复制，防止请求参数或断言列表相互污染。
    case = base.model_copy(deep=True)
    case.id = case_id
    case.description = description
    if backend is not None:
        case.backend = backend
    if fixture_profile is not None:
        case.fixture_profile = fixture_profile
    if request_updates:
        case.request = case.request.model_copy(update=request_updates)
    for key, value in overrides.items():
        setattr(case, key, value)
    return case


def _normal_variant(
    case_id: str,
    description: str,
    *,
    backend: str = "graph_send",
    request_updates: dict = None,
) -> HarnessCase:
    """构造正常完成场景，统一验证端到端结果和五个核心工具。"""
    return _clone_case(
        HAPPY_PATH,
        case_id=case_id,
        description=description,
        backend=backend,
        request_updates=request_updates,
        # 参数矩阵 Case 关注最终质量与工具覆盖，Trace 细节由 happy_path 专门验收。
        expected_metrics=[
            MetricAssertion(name="no_fake_citation", expected=True),
            MetricAssertion(name="min_sources", expected=True),
            MetricAssertion(name="answer_not_empty", expected=True),
            MetricAssertion(name="latency_under_threshold", expected=True),
        ],
        expected_tools=[
            "academic_search", "paper_metadata", "source_quality_scorer",
            "evidence_extract", "citation_check",
        ],
        required_trace_events=[],
        required_protocol_roles=[],
        required_hooks=[],
    )


HAPPY_PATH_LOOP = _normal_variant(
    "happy_path_loop",
    "Loop 后端正常执行并完成完整研究链路",
    backend="loop",
)
HAPPY_PATH_ENGLISH = _normal_variant(
    "happy_path_english",
    "英文输出请求的正常执行链路",
    request_updates={"language": "en"},
)
HAPPY_PATH_MIN_SOURCES = _normal_variant(
    "happy_path_min_sources",
    "最小来源预算（1 篇）下的正常执行",
    request_updates={"max_sources": 1},
)
HAPPY_PATH_EXTENDED_SOURCES = _normal_variant(
    "happy_path_extended_sources",
    "扩展来源预算（5 篇）下的正常执行",
    request_updates={"max_sources": 5},
)
DEEP_RESEARCH_MODE = _normal_variant(
    "deep_research_mode",
    "deep 模式下的正常执行链路",
    request_updates={"mode": "deep"},
)
STANDARD_MODE = _normal_variant(
    "standard_mode",
    "standard 模式下的正常执行链路",
    request_updates={"mode": "standard"},
)
CHINESE_TOPIC = _normal_variant(
    "chinese_topic",
    "中文学术论文引用校验主题的正常执行",
    request_updates={"topic": "中文学术论文引用校验"},
)
MAX_SOURCES_TWO = _normal_variant(
    "max_sources_two",
    "来源预算为 2 篇时的正常执行",
    request_updates={"max_sources": 2},
)
MAX_SOURCES_FOUR = _normal_variant(
    "max_sources_four",
    "来源预算为 4 篇时的正常执行",
    request_updates={"max_sources": 4},
)


# ================================================================
# 扩展 Case：检索空结果、引用恢复与工具异常
# ================================================================

NO_SOURCE_METRICS = [
    MetricAssertion(name="no_fake_citation", expected=True),
    MetricAssertion(name="min_sources", expected=False),
    MetricAssertion(name="answer_not_empty", expected=False),
    MetricAssertion(name="task_success_rate", expected=0.0),
]

NO_SOURCES_GRAPH = _clone_case(
    HAPPY_PATH,
    case_id="no_sources_graph",
    description="Graph 后端检索为空时返回可观察的失败状态",
    fixture_profile="no_sources",
    expected_metrics=NO_SOURCE_METRICS,
    expected_tools=["academic_search"],
    required_trace_events=["controller_start", "evaluator_complete"],
    required_protocol_roles=[],
    required_hooks=[],
    expect_sources_empty=True,
    expected_status="failed",
)
NO_SOURCES_LOOP = _clone_case(
    HAPPY_PATH,
    case_id="no_sources_loop",
    description="Loop 后端检索为空时保留可用的部分结果状态",
    backend="loop",
    fixture_profile="no_sources",
    expected_metrics=[
        MetricAssertion(name="min_sources", expected=False),
        MetricAssertion(name="answer_not_empty", expected=True),
    ],
    expected_tools=["academic_search"],
    required_trace_events=[],
    required_protocol_roles=[],
    required_hooks=[],
    expect_sources_empty=True,
    expected_status="completed",
)
INVALID_CITATION_LOOP = _clone_case(
    INVALID_CITATION,
    case_id="invalid_citation_loop",
    description="Loop 后端执行引用检查并保持无伪引用结果",
    backend="loop",
    expected_metrics=[
        MetricAssertion(name="no_fake_citation", expected=True),
        MetricAssertion(name="citation_id_exists", expected=True),
    ],
    expected_tools=["citation_check"],
    required_trace_events=[],
    required_protocol_roles=[],
    required_hooks=[],
    min_retry_count=0,
    require_citation_recovery=False,
    expected_status="completed",
)
INVALID_CITATION_MIN_SOURCES = _clone_case(
    INVALID_CITATION,
    case_id="invalid_citation_min_sources",
    description="最小来源预算下的引用失败重试与恢复",
    request_updates={"max_sources": 1},
)
TOOL_EXCEPTION_LOOP = _clone_case(
    TOOL_EXCEPTION,
    case_id="tool_exception_loop",
    description="Loop 后端证据抽取异常后的部分成功处理",
    backend="loop",
    expected_metrics=[
        MetricAssertion(name="answer_not_empty", expected=True),
        MetricAssertion(name="task_success_rate", expected=1.0, op="lt"),
    ],
    expected_tools=["evidence_extract"],
    required_trace_events=[],
    required_protocol_roles=[],
    required_hooks=[],
    expected_status="completed",
)
TOOL_EXCEPTION_MIN_SOURCES = _clone_case(
    TOOL_EXCEPTION,
    case_id="tool_exception_min_sources",
    description="最小来源预算下的证据抽取异常降级",
    request_updates={"max_sources": 1},
)


# ================================================================
# 扩展 Case：Worker 部分失败与延迟边界
# ================================================================

PARTIAL_FAILURE_MIN_SOURCES = _clone_case(
    SEND_PARTIAL_FAILURE,
    case_id="partial_failure_min_sources",
    description="最小来源预算下保留成功 Worker 结果",
    request_updates={"max_sources": 1},
)
PARTIAL_FAILURE_EXTENDED_SOURCES = _clone_case(
    SEND_PARTIAL_FAILURE,
    case_id="partial_failure_extended_sources",
    description="扩展来源预算下合并成功 Worker 结果",
    request_updates={"max_sources": 5},
)
PARTIAL_FAILURE_STANDARD_MODE = _clone_case(
    SEND_PARTIAL_FAILURE,
    case_id="partial_failure_standard_mode",
    description="standard 模式下 Worker 部分失败仍可完成合并",
    request_updates={"mode": "standard"},
)
LATENCY_THRESHOLD_MIN_SOURCES = _clone_case(
    LATENCY_EXCEEDED,
    case_id="latency_threshold_min_sources",
    description="最小来源预算下验证延迟阈值告警",
    request_updates={"max_sources": 1},
)

ALL_CASES = [
    HAPPY_PATH,
    INVALID_CITATION,
    TOOL_EXCEPTION,
    LLM_FALLBACK,
    SEND_PARTIAL_FAILURE,
    LATENCY_EXCEEDED,
    HAPPY_PATH_LOOP,
    HAPPY_PATH_ENGLISH,
    HAPPY_PATH_MIN_SOURCES,
    HAPPY_PATH_EXTENDED_SOURCES,
    DEEP_RESEARCH_MODE,
    STANDARD_MODE,
    CHINESE_TOPIC,
    MAX_SOURCES_TWO,
    MAX_SOURCES_FOUR,
    NO_SOURCES_GRAPH,
    NO_SOURCES_LOOP,
    INVALID_CITATION_LOOP,
    INVALID_CITATION_MIN_SOURCES,
    TOOL_EXCEPTION_LOOP,
    TOOL_EXCEPTION_MIN_SOURCES,
    PARTIAL_FAILURE_MIN_SOURCES,
    PARTIAL_FAILURE_EXTENDED_SOURCES,
    PARTIAL_FAILURE_STANDARD_MODE,
    LATENCY_THRESHOLD_MIN_SOURCES,
]
