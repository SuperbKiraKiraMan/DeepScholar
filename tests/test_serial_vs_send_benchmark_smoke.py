"""
tests/test_serial_vs_send_benchmark_smoke.py

冒烟验证 benchmarks 夹具 + run_condition + 阶段切分（小延迟，跑 1 个固定任务 × 2 条件）：

- serial（max_concurrency=1）与 parallel（默认 min(max_sources, 8)）各完成一次；
- 每条 run status ∈ {completed, completed_with_warnings}；
- LLM 覆盖达标：structured ≥ 5、fc ≥ 6、intent ≥ 1（防静默 rule 降级）；
- 六阶段边界齐全（None 即边界缺失）；
- 串行墙钟 ≈ 阶段和（stages 是墙钟子集，差额 ≤ 30% 覆盖 planner/intent/assembly）；
- 并行墙钟 ≤ 串行墙钟（×1.2 松弛）；
- search / read / chapter 阶段 speedup > 1（Send 扇出的并行收益）。

并发语义用阶段墙钟（merge_result 边界的执行期 timestamp_ms）验证——扁平化的
tool 事件时间戳不可靠，故不依赖 tool_finished 重叠检查。
"""

import pytest

from benchmarks.serial_vs_send_benchmark import (
    BenchConfig,
    STAGE_ORDER,
    run_condition,
)

TOPIC = "请调研并撰写一份关于大语言模型推理能力涌现现象的综述报告"


def _small_config() -> BenchConfig:
    """小延迟配置：跑得快但仍能体现串/并行差异。"""
    return BenchConfig(
        max_sources=8,
        tool_latency_ms=30,
        structured_latency_ms=50,
        fc_latency_ms=10,
        intent_latency_ms=5,
        total_timeout_ms=120_000,
    )


@pytest.mark.asyncio
async def test_serial_and_parallel_complete_with_llm_coverage():
    config = _small_config()
    serial = await run_condition(TOPIC, 1, config)
    parallel = await run_condition(TOPIC, None, config)

    for rec, cond in ((serial, "serial"), (parallel, "parallel")):
        assert rec["status"] in {"completed", "completed_with_warnings"}, (
            f"{cond} status={rec['status']} warnings={rec['warning_count']}"
        )
        # LLM 覆盖：低于预期说明发生静默 rule 降级
        assert rec["llm"]["structured"] >= 5, f"{cond} structured={rec['llm']}"
        assert rec["llm"]["fc"] >= 6, f"{cond} fc={rec['llm']}"
        assert rec["llm"]["intent"] >= 1, f"{cond} intent={rec['llm']}"
        # 六阶段边界齐全
        missing = [s for s in STAGE_ORDER if rec["stages"].get(s) is None]
        assert not missing, f"{cond} 缺失阶段边界：{missing}"


@pytest.mark.asyncio
async def test_serial_stage_sum_close_to_wall():
    config = _small_config()
    serial = await run_condition(TOPIC, 1, config)
    stage_sum = sum(serial["stages"][s] for s in STAGE_ORDER)
    # 阶段是墙钟子集：wall ≥ stage_sum，且差额 ≤ 30%
    assert serial["wall_e2e_ms"] >= stage_sum * 0.9, (
        f"wall={serial['wall_e2e_ms']} < stage_sum={stage_sum}"
    )
    assert (serial["wall_e2e_ms"] - stage_sum) <= 0.3 * serial["wall_e2e_ms"], (
        f"stage_sum={stage_sum} 与 wall={serial['wall_e2e_ms']} 偏差过大"
    )


@pytest.mark.asyncio
async def test_parallel_faster_and_stage_speedups():
    config = _small_config()
    serial = await run_condition(TOPIC, 1, config)
    parallel = await run_condition(TOPIC, None, config)

    # 并行整体墙钟 ≤ 串行（×1.2 松弛覆盖调度抖动）
    assert parallel["wall_e2e_ms"] <= serial["wall_e2e_ms"] * 1.2, (
        f"parallel={parallel['wall_e2e_ms']} > serial={serial['wall_e2e_ms']}"
    )

    # Send 扇出阶段（search/read/chapter）必须有并行收益
    for stage in ("search", "read", "chapter"):
        ratio = serial["stages"][stage] / parallel["stages"][stage]
        assert ratio > 1.0, (
            f"{stage} 无并行收益：serial={serial['stages'][stage]}ms "
            f"parallel={parallel['stages'][stage]}ms"
        )
