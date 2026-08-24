"""
tests/test_run_graph_max_concurrency.py

验证 run_graph 系列把 max_concurrency 正确透传到 graph.ainvoke 的 config：
- 显式 max_concurrency=1 → 1（同图串行基线）
- max_concurrency=None → max(1, min(max_sources, 8))（默认并发上限）
- 并发上限不超过 8

通过 monkeypatch _graph_send_instance.ainvoke 捕获 config、monkeypatch
_finalize_run 返回最小 dict，纯单元级验证，不执行真实图。
"""

import pytest

import app.graph.runtime as runtime_mod


@pytest.fixture
def captured_config(monkeypatch):
    """捕获传给 graph.ainvoke 的 config，并短路图执行。"""
    captured = {}

    async def fake_ainvoke(state, config=None):
        # 实例字典属性是普通函数（非绑定方法），state 直接收到 initial_state。
        captured["state"] = state
        captured["config"] = config
        return {"warnings": [], "report_completion_ready": True}

    def fake_finalize(final_state, topic, backend, agent_mode):
        return {
            "status": "completed",
            "topic": topic,
            "backend": backend,
            "agent_mode": agent_mode,
            "total_latency_ms": 1,
            "trace": [],
            "observability_metrics": {},
        }

    monkeypatch.setattr(runtime_mod._graph_send_instance, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(runtime_mod, "_finalize_run", fake_finalize)
    return captured


@pytest.mark.asyncio
async def test_explicit_max_concurrency_1_is_passed_through(captured_config):
    """传 max_concurrency=1 → config={"max_concurrency": 1}（同图串行基线）。"""
    await runtime_mod.run_graph(
        topic="测试主题", max_sources=8, agent_mode="rule", max_concurrency=1,
    )
    assert captured_config["config"] == {"max_concurrency": 1}


@pytest.mark.asyncio
async def test_default_uses_min_max_sources_8(captured_config):
    """max_concurrency=None → min(max_sources, 8)。"""
    await runtime_mod.run_graph(
        topic="测试主题", max_sources=8, agent_mode="rule",
    )
    assert captured_config["config"] == {"max_concurrency": 8}


@pytest.mark.asyncio
async def test_default_caps_at_8(captured_config):
    """max_sources=20 时并发上限仍被压到 8。"""
    await runtime_mod.run_graph(
        topic="测试主题", max_sources=20, agent_mode="rule",
    )
    assert captured_config["config"] == {"max_concurrency": 8}


@pytest.mark.asyncio
async def test_default_scales_down_to_small_max_sources(captured_config):
    """max_sources=3 → 并发上限 3。"""
    await runtime_mod.run_graph(
        topic="测试主题", max_sources=3, agent_mode="rule",
    )
    assert captured_config["config"] == {"max_concurrency": 3}


@pytest.mark.asyncio
async def test_run_graph_async_passes_max_concurrency(monkeypatch):
    """run_graph_async 同样透传 max_concurrency 到 impl。"""
    seen = {}

    async def fake_impl(**kwargs):
        seen.update(kwargs)
        return {"status": "completed", "trace": [], "total_latency_ms": 1,
                "observability_metrics": {}}

    monkeypatch.setattr(runtime_mod, "_run_graph_async_impl", fake_impl)
    await runtime_mod.run_graph_async(
        topic="测试主题", max_sources=8, agent_mode="rule", max_concurrency=1,
    )
    assert seen.get("max_concurrency") == 1
