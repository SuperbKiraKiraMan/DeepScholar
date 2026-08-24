# Send API 串行 vs 并行 基准

量化 LangGraph **Send API 动态并行调度**与**串行执行**的调度差异，帮助维护者检查并发边界。

- **串行基线**：同一编译图 `_graph_send_instance`，仅把 `max_concurrency` 压到 `1`（同 DAG 的真串行，变量只有并行调度）。
- **并行**：同一编译图，`max_concurrency=None` → 默认 `min(max_sources, 8)`。
- **数据源**：`SEARCH_PROVIDER=mock`（7 篇离线论文）+ `FakeLLMClient` + 工具 `_arun` 注入固定 `asyncio.sleep`，零网络、确定性延迟。
- **样本**：`benchmark_tasks.json` 中 15~20 个固定中文调研任务 × 2 条件各 1 次。

## 运行

```bash
cd Stage3/academic_research_copilot

# 全量（18 任务，默认延迟 tool=150 / structured=200 / fc=50 / intent=10ms）
python benchmarks/serial_vs_send_benchmark.py

# 冒烟子集 + 小延迟
python benchmarks/serial_vs_send_benchmark.py --limit 3 \
    --tool-latency-ms 30 --structured-latency-ms 50 --fc-latency-ms 10 --intent-latency-ms 5

# 自定义输出目录
python benchmarks/serial_vs_send_benchmark.py --out /tmp/bench_results
```

参数：`--tasks --limit --max-sources --tool-latency-ms --structured-latency-ms --fc-latency-ms --intent-latency-ms --seed --out`（`--repeats` 保留参数）。

## 输出

被 Git 忽略的输出目录（默认 `benchmarks/results/`，也可通过 `--out` 指定）：

| 文件 | 内容 |
|---|---|
| `raw.jsonl` | 每行一次 run：任务/条件/status/墙钟 e2e/reported e2e/各阶段耗时/LLM 覆盖 |
| `summary.json` | per-condition 聚合（mean/median/P95）+ 每任务 paired speedup + speedup 统计 |
| `report.md` | 人类可读报告：端到端表、阶段表、paired speedup 表、诚实性边界 |

**阶段切分**（trace 事件边界，事件带执行期 `timestamp_ms`）：

| 阶段 | 起点 | 终点 |
|---|---|---|
| Search | `planner_complete` | `merge_result(merge_type=search)` |
| Read | 上者 | `merge_result(merge_type=reading)` |
| Analysis | 上者 | `merge_result(merge_type=analysis)` |
| Cite | 上者 | `merge_result(merge_type=citation)` |
| Outline | 上者 | `outline_created` |
| Chapter | 上者 | `merge_result(merge_type∈{chapters, whole_report})` |

端到端采用 `time.perf_counter()` 墙钟；交叉校验用 trace 事件执行期时间戳
（首个事件 → 末个事件的 `timestamp_ms` 差）——`run_graph` 返回 dict 的
`total_latency_ms` 恒为 0（`final_state` 未携带该字段），不作为交叉校验。

## 测试

```bash
python -m pytest tests/test_run_graph_max_concurrency.py tests/test_serial_vs_send_benchmark_smoke.py -v
```

## 诚实性边界（写进报告）

1. 测的是**调度效率**：同机 asyncio 事件循环内 Send 并发调度的墙钟差异，零网络、零真实限流、零 provider 抖动，不能外推真实网络/限流场景。
2. 串行基线**仍走 Send 机制**（同一编译图，并发压到 1），对比的是「并发 1 vs 并发 N 的墙钟」，不含 Send 机制固定开销差。
3. 并行上限受扇出规模限制：SEARCH=3 / READ≤7 / CHAPTER=3，speedup 上限≈扇出数而非并发上限 8。
4. LLM 路径覆盖 = search / chapter / outline；read/cite 天然确定性，analyze 本设计 FC 耗尽后 rule 降级。
5. `asyncio.sleep` 是均匀假延迟，无长尾，P95 只描述均匀负载下调度器表现。
6. 章节证据绑定依赖 outline 的 E 别名解析；某主题证据卡不足导致章节降级为 rule gap 时，对应行应告警并从章节 speedup 结论剔除。
