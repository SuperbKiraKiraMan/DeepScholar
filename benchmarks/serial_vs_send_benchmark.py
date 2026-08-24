#!/usr/bin/env python3
"""
benchmarks/serial_vs_send_benchmark.py

Send API 串行（max_concurrency=1） vs 并行（默认 min(max_sources, 8)）基准。

测的是「同一编译图 _graph_send_instance、同 DAG、只改并发上限」下的调度墙钟收益：
- 每个任务分别以 max_concurrency=1 与 None 各跑一遍（同一图实例）；
- 记录端到端墙钟 + Search/Read/Analysis/Cite/Outline/Chapter 各阶段耗时；
- 输出 raw.jsonl / summary.json / report.md，统计 mean/median/P95 与 paired speedup。

诚实性边界见 README.md：测的是同机 asyncio 事件循环内的调度效率，零网络、
零真实限流、均匀假延迟，不能外推真实网络/限流场景。

用法：
    python benchmarks/serial_vs_send_benchmark.py                       # 全量 18 任务
    python benchmarks/serial_vs_send_benchmark.py --limit 3            # 冒烟子集
    python benchmarks/serial_vs_send_benchmark.py --tool-latency-ms 30 \\
        --structured-latency-ms 50 --fc-latency-ms 10 --intent-latency-ms 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.graph.runtime import run_graph  # noqa: E402
from benchmarks._fixtures import install_benchmark_env, llm_coverage  # noqa: E402

# 阶段顺序与 merge_type 边界（见 plans/replicated-tumbling-church.md §4）。
STAGE_ORDER = ["search", "read", "analysis", "cite", "outline", "chapter"]
CONDITIONS = [("serial", 1), ("parallel", None)]

DEFAULT_TOOL_LATENCY_MS = 150
DEFAULT_STRUCTURED_LATENCY_MS = 200
DEFAULT_FC_LATENCY_MS = 50
DEFAULT_INTENT_LATENCY_MS = 10
DEFAULT_MAX_SOURCES = 8


# ================================================================
# 配置
# ================================================================

@dataclass
class BenchConfig:
    """基准运行的延迟参数；冒烟测试直接构造小延迟配置复用。"""

    max_sources: int = DEFAULT_MAX_SOURCES
    tool_latency_ms: int = DEFAULT_TOOL_LATENCY_MS
    structured_latency_ms: int = DEFAULT_STRUCTURED_LATENCY_MS
    fc_latency_ms: int = DEFAULT_FC_LATENCY_MS
    intent_latency_ms: int = DEFAULT_INTENT_LATENCY_MS
    total_timeout_ms: int = 180_000


# ================================================================
# 阶段耗时解析（从 trace 事件边界，事件带执行期 timestamp_ms）
# ================================================================

def _last_ts(trace: List[Dict[str, Any]], event: str, **match: Any) -> Optional[int]:
    ts = [
        e.get("timestamp_ms") for e in trace
        if isinstance(e, dict) and e.get("event") == event
        and all(e.get(k) == v for k, v in match.items())
    ]
    return max(ts) if ts else None


def parse_stage_durations(
    result: Dict[str, Any], trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[float]]:
    """从 trace 解析各阶段墙钟耗时（ms）。

    边界事件：planner_complete(node=planner_agent) → 各 merge_result /
    outline_created。每个阶段 = 边界差，链式累加；某边界缺失则该阶段及其后为 None。
    """
    trace = trace if trace is not None else result.get("trace", [])
    boundaries = {
        "search": _last_ts(trace, "merge_result", merge_type="search"),
        "read": _last_ts(trace, "merge_result", merge_type="reading"),
        "analysis": _last_ts(trace, "merge_result", merge_type="analysis"),
        "cite": _last_ts(trace, "merge_result", merge_type="citation"),
        "outline": _last_ts(trace, "outline_created"),
        "chapter": (
            _last_ts(trace, "merge_result", merge_type="chapters")
            or _last_ts(trace, "merge_result", merge_type="whole_report")
        ),
    }
    start = _last_ts(trace, "planner_complete", node="planner_agent")
    stages: Dict[str, Optional[float]] = {}
    prev = start
    for stage in STAGE_ORDER:
        cur = boundaries.get(stage)
        if prev is None or cur is None:
            stages[stage] = None
            prev = None  # 边界缺失，后续无法继续链式累加
        else:
            stages[stage] = round(max(0.0, float(cur - prev)), 2)
            prev = cur
    return stages


# ================================================================
# 单次条件运行
# ================================================================

async def run_condition(
    topic: str,
    max_concurrency: Optional[int],
    config: BenchConfig,
) -> Dict[str, Any]:
    """安装夹具 → 执行一次 run_graph → 恢复，返回解析后的 run 记录。

    max_concurrency=1 为同图串行基线，None 为默认并发 min(max_sources, 8)。
    """
    fixture = install_benchmark_env(
        topic=topic,
        tool_latency_ms=config.tool_latency_ms,
        structured_latency_ms=config.structured_latency_ms,
        fc_latency_ms=config.fc_latency_ms,
        intent_latency_ms=config.intent_latency_ms,
        requested_count=config.max_sources,
    )
    try:
        wall_start = time.perf_counter()
        result = await run_graph(
            topic=topic,
            max_sources=config.max_sources,
            agent_mode="llm",
            run_eval=False,
            total_timeout_ms=config.total_timeout_ms,
            max_concurrency=max_concurrency,
        )
        wall_e2e_ms = round((time.perf_counter() - wall_start) * 1000, 2)
    finally:
        fixture.restore()

    stages = parse_stage_durations(result, result.get("trace", []))
    # 图自身口径的 e2e：首个 trace 事件到最后一个 trace 事件的执行期时间戳差。
    # （run_graph 返回 dict 的 total_latency_ms 恒为 0——final_state 未携带该字段，
    # 故用 trace 时间戳作为交叉校验，而不是那个恒 0 的字段。）
    trace_ts = [
        e.get("timestamp_ms") for e in result.get("trace", [])
        if isinstance(e, dict) and e.get("timestamp_ms") is not None
    ]
    reported_e2e_ms = (max(trace_ts) - min(trace_ts)) if trace_ts else 0
    return {
        "max_concurrency": max_concurrency,
        "status": result.get("status", "unknown"),
        "wall_e2e_ms": wall_e2e_ms,
        "reported_e2e_ms": reported_e2e_ms,
        "stages": stages,
        "stage_sum_ms": round(
            sum(v for v in stages.values() if v is not None), 2,
        ),
        "llm": llm_coverage(fixture.fake),
        "node_metrics": result.get("observability_metrics", {}).get("nodes", {}),
        "warning_count": result.get("observability_metrics", {})
            .get("run", {}).get("warning_count", 0),
    }


def _check_coverage(rec: Dict[str, Any], n_chapters: int = 3) -> List[str]:
    """校验 LLM 覆盖：structured ≥ n_chapters+2、fc ≥ 6、intent ≥ 1。

    低于预期说明发生静默 rule 降级，打印 WARNING 并返回问题列表。
    """
    llm = rec.get("llm", {})
    expected = {
        "structured": n_chapters + 2,  # planner(_fail) + outline + chapter×N
        "fc": 6,                       # 3 search × [search, finish]
        "intent": 1,
    }
    issues = []
    for channel, minimum in expected.items():
        actual = llm.get(channel, 0)
        if actual < minimum:
            issues.append(f"{channel}={actual}（预期 ≥{minimum}）")
    if issues:
        print(f"  ⚠ LLM 覆盖偏低：{'；'.join(issues)}（可能静默 rule 降级）")
    return issues


# ================================================================
# 统计
# ================================================================

def percentile(values: List[float], p: float) -> float:
    """线性插值百分位；空列表返回 0.0。"""
    s = sorted(float(v) for v in values)
    if not s:
        return 0.0
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return round(s[lo] * (1.0 - frac) + s[hi] * frac, 2)


def _describe(values: List[Optional[float]]) -> Dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean_ms": round(statistics.mean(clean), 2),
        "median_ms": round(statistics.median(clean), 2),
        "p95_ms": percentile(clean, 95),
        "min_ms": round(min(clean), 2),
        "max_ms": round(max(clean), 2),
    }


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """per-condition 聚合 + 每任务 paired speedup。"""
    by_cond: Dict[str, List[Dict[str, Any]]] = {"serial": [], "parallel": []}
    for rec in records:
        by_cond[rec["condition"]].append(rec)

    by_task: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for rec in records:
        by_task.setdefault(rec["task_index"], {})[rec["condition"]] = rec

    summary: Dict[str, Any] = {}
    # 1) e2e + 各阶段 per-condition 聚合
    metrics = {"e2e": [r["wall_e2e_ms"] for r in records if r["condition"] == "serial"]}
    # e2e 聚合按 condition 分别存
    summary["conditions"] = {}
    for condition in ("serial", "parallel"):
        rows = by_cond[condition]
        summary["conditions"][condition] = {
            "e2e": _describe([r["wall_e2e_ms"] for r in rows]),
            "stages": {
                stage: _describe([r["stages"].get(stage) for r in rows])
                for stage in STAGE_ORDER
            },
        }

    # 2) paired speedup（每任务 serial/parallel 比值）
    summary["paired_speedup"] = {"e2e": {}, "stages": {}}
    speedup_metrics = {"e2e": []}
    for stage in STAGE_ORDER:
        speedup_metrics[stage] = []
    for task_index, pair in sorted(by_task.items()):
        if "serial" not in pair or "parallel" not in pair:
            continue
        serial, parallel = pair["serial"], pair["parallel"]
        entry: Dict[str, Any] = {"task_index": task_index}
        entry["e2e_speedup"] = _safe_ratio(serial["wall_e2e_ms"], parallel["wall_e2e_ms"])
        speedup_metrics["e2e"].append(entry["e2e_speedup"])
        entry["stages"] = {}
        for stage in STAGE_ORDER:
            s = serial["stages"].get(stage)
            p = parallel["stages"].get(stage)
            ratio = _safe_ratio(s, p)
            entry["stages"][stage] = ratio
            if ratio is not None:
                speedup_metrics[stage].append(ratio)
        summary["paired_speedup"]["e2e"][task_index] = entry["e2e_speedup"]
        summary["paired_speedup"]["stages"][task_index] = entry["stages"]

    summary["speedup_stats"] = {
        "e2e": _describe_speedups(speedup_metrics["e2e"]),
        "stages": {
            stage: _describe_speedups(speedup_metrics[stage])
            for stage in STAGE_ORDER
        },
    }
    return summary


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def _describe_speedups(values: List[Optional[float]]) -> Dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean_speedup": round(statistics.mean(clean), 2),
        "median_speedup": round(statistics.median(clean), 2),
        "min_speedup": round(min(clean), 2),
        "max_speedup": round(max(clean), 2),
        "mean_pct_reduction": round(
            (1 - 1 / statistics.mean(clean)) * 100, 2,
        ),
    }


# ================================================================
# 输出
# ================================================================

def render_report(
    summary: Dict[str, Any],
    config: BenchConfig,
    run_meta: Dict[str, Any],
) -> str:
    """生成人类可读的 markdown 报告。"""
    L = []
    L.append("# Send API 串行 vs 并行 基准报告")
    L.append("")
    L.append(f"- 生成时间：{run_meta['generated_at']}")
    L.append(f"- 任务数：{run_meta['task_count']} × 2 条件")
    L.append(
        f"- 延迟参数：tool={config.tool_latency_ms}ms / "
        f"structured={config.structured_latency_ms}ms / "
        f"fc={config.fc_latency_ms}ms / intent={config.intent_latency_ms}ms"
    )
    L.append(f"- 并发上限：parallel=min({config.max_sources},8)，serial=1")
    L.append("")

    L.append("## 端到端耗时")
    L.append("")
    L.append(_render_metric_table(summary["conditions"]["serial"]["e2e"],
                                  summary["conditions"]["parallel"]["e2e"],
                                  summary["speedup_stats"]["e2e"]))
    L.append("")

    L.append("## 阶段耗时（Search/Read/… 墙钟）")
    L.append("")
    header = ("| 阶段 | 串行 mean | 串行 P95 | 并行 mean | 并行 P95 | "
              "平均 speedup | 耗时下降 |")
    L.append(header)
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for stage in STAGE_ORDER:
        ser = summary["conditions"]["serial"]["stages"][stage]
        par = summary["conditions"]["parallel"]["stages"][stage]
        sp = summary["speedup_stats"]["stages"][stage]
        L.append(
            f"| {stage} | {ser['mean_ms']} | {ser['p95_ms']} | "
            f"{par['mean_ms']} | {par['p95_ms']} | "
            f"{sp.get('mean_speedup', 0) if sp.get('n') else '-'} | "
            f"{sp.get('mean_pct_reduction', 0) if sp.get('n') else '-'}% |"
        )
    L.append("")

    L.append("## 每任务 paired speedup")
    L.append("")
    L.append("| 任务 | 串行 e2e(ms) | 并行 e2e(ms) | e2e speedup | Search speedup | Read speedup |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for task_index in sorted(summary["paired_speedup"]["e2e"]):
        ratio = summary["paired_speedup"]["e2e"][task_index]
        stage_ratios = summary["paired_speedup"]["stages"].get(task_index, {})
        task_label = summary.get("_per_task", {}).get(task_index, {}).get("task_id", task_index)
        serial_e2e = summary.get("_per_task", {}).get(task_index, {}).get("serial", {}).get("wall_e2e_ms", "-")
        parallel_e2e = summary.get("_per_task", {}).get(task_index, {}).get("parallel", {}).get("wall_e2e_ms", "-")
        L.append(
            f"| {task_label} | {serial_e2e} | {parallel_e2e} | "
            f"{ratio if ratio else '-'} | "
            f"{stage_ratios.get('search') if stage_ratios.get('search') else '-'} | "
            f"{stage_ratios.get('read') if stage_ratios.get('read') else '-'} |"
        )
    L.append("")

    L.append("## 诚实性边界")
    L.append("")
    L.append(_CAVEATS)
    return "\n".join(L)


def _render_metric_table(serial: Dict[str, Any], parallel: Dict[str, Any],
                         speedup: Dict[str, Any]) -> str:
    if speedup.get("n"):
        sp = f"{speedup['mean_speedup']}×（减少 {speedup['mean_pct_reduction']}%）"
    else:
        sp = "-"
    return (
        "| 指标 | 串行 | 并行 |\n"
        "|---|--:|--:|\n"
        f"| mean (ms) | {serial['mean_ms']} | {parallel['mean_ms']} |\n"
        f"| median (ms) | {serial['median_ms']} | {parallel['median_ms']} |\n"
        f"| P95 (ms) | {serial['p95_ms']} | {parallel['p95_ms']} |\n"
        f"| min / max (ms) | {serial['min_ms']} / {serial['max_ms']} | "
        f"{parallel['min_ms']} / {parallel['max_ms']} |\n"
        f"| 平均 speedup | {sp} | - |"
    )


_CAVEATS = (
    "1. 测的是调度效率：同机 asyncio 事件循环内 Send 并发调度的墙钟收益，"
    "零网络、零真实限流、零 provider 抖动，不能外推真实网络/限流场景。\n"
    "2. 串行基线仍走 Send 机制（同一编译图，max_concurrency=1），"
    "对比的是「并发 1 vs 并发 N 的墙钟」，不含 Send 机制固定开销差。\n"
    "3. 并行上限受扇出规模限制：SEARCH=3 / READ≤7 / CHAPTER=3，"
    "speedup 上限≈扇出数而非并发上限 8。\n"
    "4. LLM 路径覆盖 = search / chapter / outline；read/cite 天然确定性，"
    "analyze 本设计 FC 耗尽后 rule 降级。\n"
    "5. asyncio.sleep 是均匀假延迟，无长尾，P95 只描述均匀负载下调度器表现。\n"
    "6. 章节证据绑定依赖 outline 的 E 别名解析；某主题证据卡不足导致章节降级为 "
    "rule gap 时，对应行应告警并应从章节 speedup 结论剔除。"
)


# ================================================================
# 任务加载与主流程
# ================================================================

def load_tasks(path: Optional[str], limit: Optional[int], seed: Optional[int]) -> List[Dict[str, Any]]:
    task_path = Path(path or BENCH_DIR / "benchmark_tasks.json")
    if not task_path.exists():
        raise FileNotFoundError(f"任务文件不存在：{task_path}")
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    if seed is not None:
        import random
        random.Random(seed).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("任务列表为空")
    return tasks


async def main(args: argparse.Namespace) -> None:
    config = BenchConfig(
        max_sources=args.max_sources,
        tool_latency_ms=args.tool_latency_ms,
        structured_latency_ms=args.structured_latency_ms,
        fc_latency_ms=args.fc_latency_ms,
        intent_latency_ms=args.intent_latency_ms,
    )
    tasks = load_tasks(args.tasks, args.limit, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"任务数：{len(tasks)}；每个任务跑 serial(max_concurrency=1) + "
          f"parallel(max_concurrency=None)")
    print(f"延迟参数：tool={config.tool_latency_ms}ms / "
          f"structured={config.structured_latency_ms}ms / "
          f"fc={config.fc_latency_ms}ms / intent={config.intent_latency_ms}ms")
    print("-" * 72)

    records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        tid, topic = task["id"], task["topic"]
        for condition, max_concurrency in CONDITIONS:
            rec = await run_condition(topic, max_concurrency, config)
            rec.update({"task_id": tid, "task_index": index - 1, "condition": condition})
            records.append(rec)
            coverage = " ".join(f"{k}={v}" for k, v in rec["llm"].items())
            issues = _check_coverage(rec)
            flag = "⚠" if (rec["status"] not in {"completed", "completed_with_warnings"} or issues) else "✓"
            print(
                f"[{index:02d}/{len(tasks)}] {condition:8s} {tid} "
                f"status={rec['status']:22s} e2e={rec['wall_e2e_ms']:8.1f}ms "
                f"stages={rec['stage_sum_ms']:8.1f}ms llm[{coverage}] {flag}"
            )
        print("-" * 72)

    wall_total = round(time.perf_counter() - t_start, 2)

    # 写 raw.jsonl
    raw_path = out_dir / "raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 汇总
    summary = compute_summary(records)
    summary["_per_task"] = {}
    for rec in records:
        entry = summary["_per_task"].setdefault(rec["task_index"], {})
        entry["task_id"] = rec.get("task_id", str(rec["task_index"]))
        entry[rec["condition"]] = {
            "wall_e2e_ms": rec["wall_e2e_ms"],
            "stages": rec["stages"],
            "status": rec["status"],
        }
    summary["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "task_count": len(tasks),
        "wall_total_s": wall_total,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 报告
    report_path = out_dir / "report.md"
    report_path.write_text(
        render_report(summary, config, summary["meta"]), encoding="utf-8",
    )

    print(f"总耗时：{wall_total}s")
    print(f"输出：")
    print(f"  raw.jsonl      {raw_path}")
    print(f"  summary.json   {summary_path}")
    print(f"  report.md      {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default=str(BENCH_DIR / "benchmark_tasks.json"),
                        help="任务 JSON 路径")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个任务（冒烟用）")
    parser.add_argument("--repeats", type=int, default=1,
                        help="每个任务×条件的重复次数（保留参数，当前固定 1）")
    parser.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES)
    parser.add_argument("--tool-latency-ms", type=int, default=DEFAULT_TOOL_LATENCY_MS)
    parser.add_argument("--structured-latency-ms", type=int, default=DEFAULT_STRUCTURED_LATENCY_MS)
    parser.add_argument("--fc-latency-ms", type=int, default=DEFAULT_FC_LATENCY_MS)
    parser.add_argument("--intent-latency-ms", type=int, default=DEFAULT_INTENT_LATENCY_MS)
    parser.add_argument("--seed", type=int, default=None, help="任务洗牌种子")
    parser.add_argument("--out", default=str(BENCH_DIR / "results"),
                        help="输出目录（默认 benchmarks/results）")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
