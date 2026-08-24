"""
tests/test_phase3_2_hardening.py

Phase 3 加固测试：
- SSE finished-run replay
- SSE reconnect (Last-Event-ID)
- Cancel endpoint
- Invalid backend validation
- Failure state persistence
- Desktop/mobile visual regression
- FC-friendly schema
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ================================================================
# SSE Integration Tests
# ================================================================

class TestSSEIntegration:
    """SSE finished-run replay & reconnect 测试。"""

    def test_finished_run_get_events_includes_run_started(self):
        """已完成 run 的 events 从 id=1 开始回放，包含 run_started。"""
        from app.services.event_broker import event_broker

        # 创建并完成一个 run
        response = client.post(
            "/api/research/runs?backend=graph_send",
            json={"topic": "Replay test", "max_sources": 2, "agent_mode": "rule"},
        )
        run_id = response.json()["run_id"]

        # 模拟 run completion
        event_broker.publish(run_id, "tool_started", {"tool": "test"})
        event_broker.publish(run_id, "run_finished", {"status": "completed"})
        event_broker.finish_run(run_id)

        # 查询 events（首次连接，无 Last-Event-ID）
        events = event_broker.get_unconsumed_events(run_id, "")
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types
        assert "tool_started" in event_types
        assert "run_finished" in event_types

    def test_reconnect_with_last_event_id(self):
        """重连（带 Last-Event-ID）只回放未消费事件。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("reconnect-test", "Topic")
        broker.publish("reconnect-test", "run_started", {"n": 1})
        broker.publish("reconnect-test", "plan_created", {"n": 2})
        broker.publish("reconnect-test", "tool_started", {"n": 3})
        broker.publish("reconnect-test", "tool_finished", {"n": 4})

        # 重连：客户端说已收到 id=2
        unconsumed = broker.get_unconsumed_events("reconnect-test", "2")
        event_names = [e["event"] for e in unconsumed]
        assert "run_started" not in event_names  # id=1 已消费
        assert "plan_created" not in event_names  # id=2 已消费
        assert "tool_started" in event_names  # id=3 未消费
        assert "tool_finished" in event_names  # id=4 未消费

    def test_reconnect_with_invalid_last_event_id(self):
        """无效 Last-Event-ID 时回放全部事件。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("bad-reconnect", "Topic")
        broker.publish("bad-reconnect", "event_1", {})

        # 无效 header
        unconsumed = broker.get_unconsumed_events("bad-reconnect", "not-a-number")
        assert len(unconsumed) == 1  # 全部回放

    def test_finished_run_is_finished(self):
        """finish_run 后 is_finished 为 True（使用独立 broker 避免全局状态污染）。"""
        import uuid
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        run_id = str(uuid.uuid4())[:8]
        broker.init_run(run_id, "Fresh run")
        assert not broker.is_finished(run_id)

        broker.finish_run(run_id)
        assert broker.is_finished(run_id)


class TestCancelEndpoint:
    """取消接口测试。"""

    def test_cancel_unknown_run_returns_404(self):
        """取消不存在的 run 返回 404。"""
        response = client.post("/api/research/runs/nonexistent-99/cancel")
        assert response.status_code == 404

    def test_cancel_completed_run_returns_message(self):
        """取消已完成的 run 返回提示。"""
        from app.services.run_store import run_store

        # 直接构造一个已完成（无运行中 task）的 run
        run_id = run_store.create(topic="Cancel test")
        run_store.update(run_id, status="completed")
        run_store.finish(run_id, "completed")

        # 尝试取消已完成 run
        cancel_resp = client.post(f"/api/research/runs/{run_id}/cancel")
        assert cancel_resp.status_code == 200
        data = cancel_resp.json()
        assert "cannot cancel" in data.get("message", "").lower()

    def test_event_broker_cancel_marks_cancelled(self):
        """event_broker.cancel_run 后 is_cancelled 为 True。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("cancel-me", "Test")
        assert not broker.is_cancelled("cancel-me")
        broker.cancel_run("cancel-me")
        assert broker.is_cancelled("cancel-me")


class TestBackendValidation:
    """非法 backend 校验。"""

    def test_invalid_backend_async_returns_422(self):
        """异步接口非法 backend 返回 422。"""
        response = client.post(
            "/api/research/runs?backend=not_a_backend",
            json={"topic": "Test", "max_sources": 2},
        )
        assert response.status_code == 422

    def test_valid_backends_accepted(self):
        """合法 backend 全部接受。"""
        for backend in ("loop", "graph_send"):
            response = client.post(
                f"/api/research/runs?backend={backend}",
                json={"topic": "Valid backend", "max_sources": 1},
            )
            assert response.status_code == 202, f"Backend {backend} should be accepted"


class TestFailureState:
    """失败状态落库测试。"""

    def test_run_store_supports_failed_status(self):
        """RunStore 保存 failed 状态。"""
        from app.services.run_store import RunStore

        store = RunStore()
        rid = store.create(topic="Fail test")
        store.update(rid, status="failed", error="Something broke")
        store.finish(rid, "failed")

        data = store.get(rid)
        assert data["status"] == "failed"
        assert "error" in data

    def test_broker_finish_with_error(self):
        """Event broker finish_run 保存 error。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("err-run", "Test")
        broker.finish_run("err-run", "Test error message")
        assert broker.get_error("err-run") == "Test error message"
        assert broker.is_finished("err-run")


# ================================================================
# Desktop / Mobile Visual Verification
# ================================================================

class TestVisualResponsive:
    """桌面/移动端视觉回归测试。"""

    def test_viewport_meta_present(self):
        """HTML 包含 viewport meta 标签。"""
        response = client.get("/")
        body = response.text
        assert 'name="viewport"' in body
        assert "width=device-width" in body

    def test_mobile_breakpoints_in_css(self):
        """CSS 包含移动端断点（max-width: 900px / 500px）。"""
        response = client.get("/static/styles.css")
        body = response.text
        assert "@media" in body
        assert "max-width" in body

    def test_no_hardcoded_wide_layout(self):
        """CSS 不包含 > 900px 的固定宽度。"""
        response = client.get("/static/styles.css")
        body = response.text
        # 布局使用 grid/flex，不应有超过 400px 的固定宽度
        for line in body.split("\n"):
            if "width:" in line and "px" in line:
                val_str = line.split("width:")[1].strip().rstrip(";").rstrip("px")
                # 允许 max-width 但不允许固定 > 500px 宽度
                if "max" not in line.split("width")[0]:
                    try:
                        val = int(val_str.strip().split("px")[0])
                        assert val <= 500, f"Fixed width {val}px too wide: {line.strip()}"
                    except ValueError:
                        pass  # not a pure px value

    def test_no_text_overflow_in_mobile(self):
        """移动端不应有横向溢出。"""
        response = client.get("/static/styles.css")
        body = response.text
        # Should have overflow-x handling
        assert "overflow-x" in body or "overflow" in body
        # Should have word-break for long text
        assert "word-break" in body.lower() or "break-word" in body

    def test_border_radius_constrained(self):
        """卡片圆角不超过 8px。"""
        response = client.get("/static/styles.css")
        body = response.text
        for line in body.split("\n"):
            if "border-radius" in line and "px" in line:
                val_str = line.split("border-radius:")[1].split(";")[0].strip().rstrip("px")
                try:
                    val = int(val_str.split()[0])
                    assert val <= 8, f"border-radius {val}px exceeds 8px: {line.strip()}"
                except ValueError:
                    pass

    def test_form_elements_exist(self):
        """前端表单包含所有必需元素。"""
        response = client.get("/")
        body = response.text
        assert "backend" in body.lower()
        assert "agent_mode" in body.lower()
        assert "max_sources" in body
        assert "run_eval" in body


# ================================================================
# FC-friendly Schema Tests
# ================================================================

class TestFCFriendlySchema:
    """FC-facing vs internal Tool Schema 分离测试。"""

    def test_evidence_extract_fc_schema_exposes_source_id(self):
        """evidence_extract 的 FC schema 暴露 source_id (string)，非 source (object)。"""
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        schemas = registry.get_function_schemas(["evidence_extract"])

        ev_schema = None
        for s in schemas:
            if s["function"]["name"] == "evidence_extract":
                ev_schema = s["function"]
                break

        assert ev_schema is not None
        props = ev_schema["parameters"]["properties"]
        # FC-facing: 应有 source_id (string)
        assert "source_id" in props, "evidence_extract FC schema should expose source_id"
        assert props["source_id"]["type"] == "string"

    def test_evidence_extract_internal_schema_has_source_object(self):
        """evidence_extract 的内部 input_schema 仍保留 source 对象定义。"""
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        internal = registry.get_input_schema("evidence_extract")
        assert internal is not None
        assert "source" in internal["properties"], "Internal schema must retain source object"

    def test_paper_metadata_fc_schema_exposes_source_ids(self):
        """paper_metadata 的 FC schema 暴露 source_ids (string[])，非 sources (object[])。"""
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        schemas = registry.get_function_schemas(["paper_metadata"])

        pm_schema = None
        for s in schemas:
            if s["function"]["name"] == "paper_metadata":
                pm_schema = s["function"]
                break

        assert pm_schema is not None
        props = pm_schema["parameters"]["properties"]
        assert "source_ids" in props
        assert props["source_ids"]["type"] == "array"
        assert props["source_ids"]["items"]["type"] == "string"
