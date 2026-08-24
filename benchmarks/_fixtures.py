"""
benchmarks/_fixtures.py

Send API 串行 vs 并行基准测试的离线假延迟夹具。

三件套：
1. LatencyLLMClient —— 在 FakeLLMClient 三个通道（structured / FC / intent）
   前注入可控 asyncio.sleep 延迟，保留 schema 校验与响应耗尽语义。
2. patch_tool_latency —— 类级替换业务工具 _arun，给每次工具调用加固定延迟
   （延迟自动计入 ToolResult.latency_ms，可从 observability 校验）。
3. _DeterministicMockProvider —— 固定 source_id 的 mock provider，绕开 uuid
   导致的跨 worker 去重失败，从而不触发 AdaptiveSourceSelector 的 LLM 分支。

外加 FakeLLM 全链路响应脚本（build_fake_responses）与一键安装/恢复
（install_benchmark_env），让基准脚本与冒烟测试共用同一套夹具。

设计约束（见 plans/replicated-tumbling-church.md §3）：
- agent_mode="llm" 但 LLM_ONLY_MODE=false，允许确定性 rule 降级；
- run_eval=False 时跳过 Evaluator 与语义 Reviewer，省去 reviewer 的 LLM 响应；
- FC 通道是「消息历史驱动」而非共享索引：每个 worker 按自己的对话历史
  拿到确定性响应（search worker: [search, finish]；analyze worker: 耗尽→rule），
  因此并发扇出下无竞态、每 run 消耗确定：structured N+2、fc 6(search)+3(analyze 耗尽)、intent 1。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

from app.llm import client as llm_client_mod
from app.llm.client import FakeLLMClient
from app.tools.academic_search_provider import MockAcademicSearchProvider
from app.tools.academic_search_tool import AcademicSearchTool
from app.tools.citation_check_tool import CitationCheckTool
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.tools.paper_metadata_tool import PaperMetadataTool
from app.tools.source_quality_scorer import SourceQualityScorer

# Search / Read / Analyze / Cite 各阶段至少命中一个的业务工具。
TOOL_CLASSES = [
    AcademicSearchTool,
    PaperMetadataTool,
    SourceQualityScorer,
    EvidenceExtractTool,
    CitationCheckTool,
]

# 基准离线环境变量（零网络、确定性延迟、允许 rule 降级）。
BENCHMARK_ENV = {
    "AGENT_MODE": "llm",          # 走真实 LLMWorker ReAct 路径
    "LLM_ONLY_MODE": "false",     # 允许 rule 降级（LLM-only 时降级会抛异常）
    "SEARCH_PROVIDER": "mock",    # 离线 mock provider
    "MCP_ENABLED": "false",       # 不加载 MCP 工具
    "RUN_HISTORY_ENABLED": "false",  # 不写 SQLite 运行历史
}


# ================================================================
# 1. LatencyLLMClient —— 三通道可控延迟的 FakeLLMClient
# ================================================================

class LatencyLLMClient(FakeLLMClient):
    """在 FakeLLMClient 三个通道前注入可控延迟，保留 schema 校验与耗尽语义。"""

    def __init__(
        self,
        responses: Optional[List[Dict[str, Any]]] = None,
        *,
        structured_latency_ms: int = 0,
        fc_latency_ms: int = 0,
        intent_latency_ms: int = 0,
    ):
        super().__init__(responses)
        self._structured_latency_ms = structured_latency_ms
        self._fc_latency_ms = fc_latency_ms
        self._intent_latency_ms = intent_latency_ms

    async def generate_structured(self, system_prompt, user_prompt, output_schema, **kwargs):
        if self._structured_latency_ms:
            await asyncio.sleep(self._structured_latency_ms / 1000.0)
        return await super().generate_structured(
            system_prompt, user_prompt, output_schema, **kwargs
        )

    async def function_call(self, messages, tools, **kwargs):
        """FC 通道：消息历史驱动，天然并发安全。

        FakeLLMClient 的共享 _fc_index 在 Send 并行扇出下会被多个 worker
        竞态消耗（一个 search worker 可能连续拿到 2 个 search），导致并行阶段
        多做无谓工具调用、测量失真。这里改为按「本 worker 自己的对话历史」决定
        响应，不依赖任何共享状态：

        - search worker（tools 含 academic_search）：对话里还没有 tool 结果 →
          返回 search；已有 → 返回 finish。确定性 [search, finish]。
        - 其余 worker（analyze 等）：一律 success=False，连续 3 次耗尽后
          LLMWorker 按 max_consecutive_errors=3 走 rule 降级（确定性）。

        这模拟了生产语义：每个 worker 是独立 LLM 会话，按自己的历史决策。
        """
        if self._fc_latency_ms:
            await asyncio.sleep(self._fc_latency_ms / 1000.0)

        tool_names = {t.get("function", {}).get("name", "") for t in tools}
        self.fc_calls.append({
            "message_count": len(messages),
            "tool_count": len(tools),
            "tool_names": sorted(tool_names),
        })

        if "academic_search" not in tool_names:
            # analyze 等非检索 worker：FC 耗尽 → rule 降级（确定性）
            return {
                "success": False,
                "error": "FakeLLMClient: no more fc responses",
                "latency_ms": 1,
            }

        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if has_tool_result:
            return {
                "success": True,
                "finish": True,
                "content": "Search complete",
                "latency_ms": 5,
                "model": "fake-deepseek-v4-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        return {
            "success": True,
            "finish": False,
            "tool_calls": [{
                "id": f"call_bench_search_{len(self.fc_calls)}",
                "name": "academic_search",
                "arguments": {"query": "research overview", "max_results": 5},
            }],
            "latency_ms": 5,
            "model": "fake-deepseek-v4-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    async def classify_intent(self, system_prompt, user_prompt, output_schema):
        if self._intent_latency_ms:
            await asyncio.sleep(self._intent_latency_ms / 1000.0)
        return await super().classify_intent(system_prompt, user_prompt, output_schema)


# ================================================================
# 2. 工具 _arun 延迟注入（类级替换，延迟计入 ToolResult.latency_ms）
# ================================================================

def patch_tool_latency(
    tool_classes: List[type],
    latency_ms: int,
) -> Callable[[], None]:
    """给 tool_classes 每个类的 _arun 加固定延迟：先 sleep 再调原实现。

    返回恢复函数，把每个类的 _arun 还原为原始实现。
    注意用默认参数绑定 original/_secs，避免循环闭包共享最后一个变量。
    """
    latency_secs = latency_ms / 1000.0
    saved: List[tuple] = []

    for cls in tool_classes:
        original = cls._arun

        async def patched(self, *args, _orig=original, _secs=latency_secs, **kwargs):
            if _secs:
                await asyncio.sleep(_secs)
            return await _orig(self, *args, **kwargs)

        cls._arun = patched
        saved.append((cls, original))

    def restore() -> None:
        for cls, original in saved:
            cls._arun = original

    return restore


# ================================================================
# 3. 确定性 mock provider（固定 source_id，绕开 source-selector LLM）
# ================================================================

class _DeterministicMockProvider(MockAcademicSearchProvider):
    """固定 source_id（mock_paper_01..07）的 mock provider。

    MockAcademicSearchProvider.search 默认生成 uuid source_id，3 个 search
    worker 各自返回 7 篇论文时跨 worker 去重不掉，会触发 AdaptiveSourceSelector
    的 LLM 分支（脆弱、非确定）。这里按标题固定 source_id，去重收敛到 ≤7，
    max_sources=8 时走 all_candidates，彻底绕开 source-selector LLM。
    """

    def __init__(self):
        super().__init__()
        self._id_by_title = {
            str(paper.get("title") or ""): f"mock_paper_{index:02d}"
            for index, paper in enumerate(self._papers, start=1)
        }

    async def search(self, query, max_results, year_from=None, year_to=None):
        result = await super().search(
            query, max_results, year_from=year_from, year_to=year_to,
        )
        if result.data:
            for item in result.data.get("results", []):
                title = str(item.get("title") or "")
                fixed_id = self._id_by_title.get(title)
                if fixed_id:
                    item["source_id"] = fixed_id
        return result


def patch_deterministic_provider() -> Callable[[], None]:
    """让 AcademicSearchTool._get_provider 返回固定 source_id 的 mock provider。"""
    original = AcademicSearchTool._get_provider

    def patched(self):
        return _DeterministicMockProvider()

    AcademicSearchTool._get_provider = patched
    return lambda: setattr(AcademicSearchTool, "_get_provider", original)


# ================================================================
# 4. FakeLLM 全链路响应脚本
# ================================================================

def build_intent_response(topic: str, requested_count: int = 8) -> Dict[str, Any]:
    """Controller 意图：锁定 deep_research / full_research，不依赖 rule 关键词。"""
    return {
        "intent": "deep_research",
        "execution_route": "full_research",
        "research_topic": topic,
        "selected_tool": "",
        "requested_count": requested_count,
        "confidence": 0.98,
        "reasoning": "benchmark: force full_research pipeline",
    }


def build_outline_response() -> Dict[str, Any]:
    """LLMOutlinePlan：3 节，assigned_evidence_ids 用 E 别名。

    E1/E2/E3 由 report_outline._normalize_model_outline 按当次 prompt 的证据卡
    解析回真实 evidence_id；来源绑定由代码从证据卡推导，不信任模型自填。
    heading 避开「局限/limitation」等维度关键词，避免触发必需维度校验。
    """
    return {
        "sections": [
            {
                "heading": "研究背景与核心问题",
                "guiding_question": "该主题的研究背景如何界定，需要回答哪些核心问题？",
                "assigned_evidence_ids": ["E1"],
            },
            {
                "heading": "主流方法与技术路线",
                "guiding_question": "该主题的主流方法家族与技术路线分别是什么，各自的适用条件如何？",
                "assigned_evidence_ids": ["E2"],
            },
            {
                "heading": "证据质量与综合发现",
                "guiding_question": "已核验证据能支撑哪些综合发现，证据之间的冲突与空白在哪里？",
                "assigned_evidence_ids": ["E3"],
            },
        ],
        "cross_cutting_themes": ["证据综合", "方法对比"],
        "evidence_gaps": [],
    }


def build_chapter_responses(n_sections: int = 3) -> List[Dict[str, Any]]:
    """LLMChapterOutput ×N：heading≥2、synthesis≥30、findings 可空。"""
    chapters = [
        {
            "heading": "研究背景与核心问题",
            "synthesis": "综合已核验文献证据，该主题的研究背景清晰，核心问题聚焦于概念界定、研究边界划分以及与相邻工作的差异，为后续各章提供问题框架。",
            "evidence_ids": [],
            "findings": [],
            "source_title_translations": {},
        },
        {
            "heading": "主流方法与技术路线",
            "synthesis": "该主题的主流方法可划分为若干方法家族，各家族在适用条件、计算成本与可解释性上各有取舍；本报告基于证据质量对方法路线做了综合对比。",
            "evidence_ids": [],
            "findings": [],
            "source_title_translations": {},
        },
        {
            "heading": "证据质量与综合发现",
            "synthesis": "已核验证据整体质量较高，能够支撑若干综合发现；部分长尾问题仍存在证据空白，需要在后续研究中进一步补充。",
            "evidence_ids": [],
            "findings": [],
            "source_title_translations": {},
        },
    ]
    return chapters[:n_sections]


def build_fake_responses(
    topic: str,
    *,
    requested_count: int = 8,
    n_chapters: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """组装 FakeLLM 响应脚本。

    FC 通道不在此预设——由 LatencyLLMClient.function_call 按消息历史
    确定性生成（见上方）。每次 run 的预期消耗：
    - structured: planner(1，_fail → rule baseline) + outline(1) + chapter(N) = N+2
    - fc: 3 个 search worker × [search, finish] = 6 + analyze 3 次耗尽 = 9
    - intent: 1
    """
    structured = [
        {"_fail": True},                       # LLMPlanner → rule baseline
        build_outline_response(),              # LLMOutlinePlan
        *build_chapter_responses(n_chapters),  # LLMChapterOutput ×N
    ]
    return {
        "structured": structured,
        "intent": [build_intent_response(topic, requested_count)],
    }


# ================================================================
# 5. 一键安装 / 恢复
# ================================================================

class BenchmarkFixture:
    """一次基准 run 的夹具：注入的 fake 客户端与恢复函数。"""

    def __init__(self, fake: LatencyLLMClient, restore: Callable[[], None]):
        self.fake = fake
        self.restore = restore


def install_benchmark_env(
    *,
    topic: str,
    tool_latency_ms: int = 150,
    structured_latency_ms: int = 200,
    fc_latency_ms: int = 50,
    intent_latency_ms: int = 10,
    requested_count: int = 8,
) -> BenchmarkFixture:
    """安装基准环境：env + 注入 LatencyLLMClient + 确定性 provider + 工具延迟。

    返回 BenchmarkFixture（含 fake 实例，可读 calls/fc_calls/intent_calls 校验
    LLM 覆盖）与 restore()。restore 会恢复 env、还原工具 _arun 与 provider、
    清空全局 LLM 客户端。
    """
    saved_env = {key: os.environ.get(key) for key in BENCHMARK_ENV}
    os.environ.update(BENCHMARK_ENV)

    responses = build_fake_responses(topic, requested_count=requested_count)
    fake = LatencyLLMClient(
        responses["structured"],
        structured_latency_ms=structured_latency_ms,
        fc_latency_ms=fc_latency_ms,
        intent_latency_ms=intent_latency_ms,
    )
    fake.set_intent_responses(responses["intent"])
    llm_client_mod._global_client = fake

    restores = [
        patch_tool_latency(TOOL_CLASSES, tool_latency_ms),
        patch_deterministic_provider(),
    ]

    def restore() -> None:
        for fn in restores:
            fn()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        llm_client_mod._global_client = None

    return BenchmarkFixture(fake=fake, restore=restore)


def llm_coverage(fake: FakeLLMClient) -> Dict[str, int]:
    """返回 fake 各通道实际调用次数，用于校验 LLM 覆盖是否达标。"""
    return {
        "structured": len(fake.calls),
        "fc": len(fake.fc_calls),
        "intent": len(fake.intent_calls),
    }
