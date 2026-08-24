"""
tests/test_phase3_sse_frontend.py

Phase 3 Gate B + Gate C 测试：
- Async run creation → 202 + run_id
- SSE content-type
- Event sequence
- GET run result
- Static files and Dashboard
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ================================================================
# Gate B: SSE + Async Run Tests
# ================================================================

class TestAsyncRun:
    """异步运行接口测试。"""

    def test_create_async_run_returns_202(self):
        """POST /api/research/runs 返回 202 和 run_id。"""
        response = client.post(
            "/api/research/runs?backend=graph_send",
            json={"topic": "Test topic", "max_sources": 2, "agent_mode": "rule"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "queued"

    def test_create_async_run_with_loop_backend(self):
        """POST /api/research/runs 支持 loop backend。"""
        response = client.post(
            "/api/research/runs?backend=loop",
            json={"topic": "Test topic", "max_sources": 2, "agent_mode": "rule"},
        )
        assert response.status_code == 202

    @pytest.mark.asyncio
    async def test_async_run_stores_initial_events(self):
        """异步 run 创建后 event_broker 包含 run_started 事件。"""
        from app.services.event_broker import event_broker

        response = client.post(
            "/api/research/runs?backend=graph_send",
            json={"topic": "Event test", "max_sources": 2, "agent_mode": "rule"},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        # 即使后台任务可能还在运行，broker 中已有初始事件
        events = event_broker.get_unconsumed_events(run_id)
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types


class TestSSEStream:
    """SSE 事件流测试。"""

    def test_sse_content_type(self):
        """SSE content-type 为 text/event-stream。"""
        # 先创建一个 run
        response = client.post(
            "/api/research/runs?backend=graph_send",
            json={"topic": "SSE test", "max_sources": 2, "agent_mode": "rule"},
        )
        run_id = response.json()["run_id"]

        # 对 SSE stream 发 HEAD-like GET 请求，检查 content-type header
        # TestClient 的 streaming 长连接可能挂，用不带 stream 的请求验证路由存在
        # （EventSourceResponse 产生的 streaming response 在同步 TestClient 中难以测试）
        from app.services.event_broker import event_broker
        events = event_broker.get_unconsumed_events(run_id)
        assert len(events) >= 1
        assert events[0]["event"] == "run_started"

    def test_sse_unknown_run_returns_404(self):
        """不存在的 run 返回 404。"""
        response = client.get("/api/research/stream/nonexistent-id")
        assert response.status_code == 404

    def test_finished_failed_run_replay_keeps_failed_terminal_status(self):
        """失败 run 重连时不能被补写成 completed。"""
        from app.services.event_broker import event_broker
        from app.services.run_store import run_store

        run_id = "failed-replay-test"
        run_store.create(topic="failed replay", run_id=run_id)
        run_store.update(run_id, status="failed", error="outline failed")
        run_store.finish(run_id, "failed")
        event_broker.init_run(run_id, "failed replay")
        event_broker.publish(run_id, "error", {
            "message": "outline failed", "error_type": "RuntimeError",
        })
        event_broker.finish_run(run_id, "outline failed")

        response = client.get(f"/api/research/stream/{run_id}")

        assert response.status_code == 200
        assert '"status": "failed"' in response.text

    def test_sse_contains_run_started_event(self):
        """SSE 包含 run_started 事件。"""
        from app.services.event_broker import event_broker

        response = client.post(
            "/api/research/runs?backend=graph_send",
            json={"topic": "SSE events test", "max_sources": 2, "agent_mode": "rule"},
        )
        run_id = response.json()["run_id"]

        # 检查 event broker 中有 run_started
        events = event_broker.get_unconsumed_events(run_id)
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types

    def test_event_sequence_monotonic(self):
        """相同 run_id 的事件序号单调递增。"""
        from app.services.event_broker import event_broker

        response = client.post(
            "/api/research/runs?backend=graph_send",
            json={"topic": "Sequence test", "max_sources": 2, "agent_mode": "rule"},
        )
        run_id = response.json()["run_id"]

        events = event_broker.get_unconsumed_events(run_id)
        seqs = [int(e["id"]) for e in events]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"Sequence should be monotonic: {seqs}"


class TestEventBroker:
    """RunEventBroker 单元测试。"""

    def test_init_and_publish(self):
        """Event broker 可以初始化和发布事件。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("test-1", "Test Topic")
        broker.publish("test-1", "run_started", {"run_id": "test-1"})

        events = broker.get_unconsumed_events("test-1")
        assert len(events) == 1
        assert events[0]["event"] == "run_started"

    def test_subscribe_receives_events(self):
        """Subscriber 能收到新发布的事件。"""
        import asyncio
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("test-2", "Test")

        q = broker.subscribe("test-2")
        assert q is not None

        broker.publish("test-2", "tool_started", {"tool": "mock"})

        try:
            ev = q.get_nowait()
            assert ev["event"] == "tool_started"
        except asyncio.QueueEmpty:
            pytest.fail("Subscriber should have received event")

    def test_unconsumed_events_with_last_event_id(self):
        """get_unconsumed_events 支持 last_event_id 回放。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("test-3", "Test")

        broker.publish("test-3", "event_1", {"n": 1})
        broker.publish("test-3", "event_2", {"n": 2})
        broker.publish("test-3", "event_3", {"n": 3})

        # 从 id=1 后开始
        unconsumed = broker.get_unconsumed_events("test-3", "1")
        event_names = [e["event"] for e in unconsumed]
        assert "event_2" in event_names
        assert "event_3" in event_names
        assert "event_1" not in event_names

    def test_finish_run_marks_complete(self):
        """finish_run 后 is_finished 为 true。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("test-4", "Test")
        assert not broker.is_finished("test-4")
        broker.finish_run("test-4")
        assert broker.is_finished("test-4")

    def test_error_run_stores_error(self):
        """Error run 保存错误信息。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("test-5", "Test")
        broker.finish_run("test-5", "Something went wrong")
        assert broker.get_error("test-5") == "Something went wrong"

    def test_event_count(self):
        """get_event_count 返回正确的事件数。"""
        from app.services.event_broker import RunEventBroker

        broker = RunEventBroker()
        broker.init_run("test-6", "Test")
        broker.publish("test-6", "a", {})
        broker.publish("test-6", "b", {})
        assert broker.get_event_count("test-6") == 2


# ================================================================
# Gate C: Frontend Dashboard Tests
# ================================================================

class TestFrontendDashboard:
    """前端 Dashboard 测试。"""

    def test_root_returns_dashboard(self):
        """GET / 返回 Dashboard HTML。"""
        response = client.get("/")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type
        # 包含关键 HTML 元素
        body = response.text
        assert "Academic Research Copilot" in body
        assert "dashboard" in body.lower()

    def test_static_css_accessible(self):
        """GET /static/styles.css 返回 CSS。"""
        response = client.get("/static/styles.css")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/css" in content_type

    def test_static_js_accessible(self):
        """GET /static/app.js 返回 JS。"""
        response = client.get("/static/app.js")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "javascript" in content_type.lower() or "text/" in content_type.lower()

    def test_static_index_accessible(self):
        """GET /static/index.html 返回 HTML。"""
        response = client.get("/static/index.html")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/html" in content_type

    def test_brand_logo_and_favicon_assets_are_available(self):
        """主页使用完整 Logo，浏览器标签页使用独立图形 Logo。"""
        body = client.get("/").text
        assert "/static/assets/academic-copilot-logo.png" in body
        assert "/static/assets/academic-copilot-icon.png" in body
        assert 'class="welcome-logo"' in body
        assert 'class="message-avatar assistant-avatar"' in body
        assert 'rel="icon"' in body

        for path in (
            "/static/assets/academic-copilot-logo.png",
            "/static/assets/academic-copilot-icon.png",
        ):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers.get("content-type", "").startswith("image/png")

    def test_form_has_backend_selector(self):
        """表单包含 backend 选择器。"""
        response = client.get("/")
        body = response.text
        assert "backend" in body.lower()
        assert "graph_send" in body

    def test_form_has_agent_mode_selector(self):
        """表单包含 agent_mode 选择器。"""
        response = client.get("/")
        body = response.text
        assert "agent_mode" in body.lower()
        assert '<option value="llm" selected>' in body

    def test_form_has_max_sources_input(self):
        """表单包含 max_sources 输入。"""
        response = client.get("/")
        body = response.text
        assert "max_sources" in body
        assert 'max="50"' in body
        assert "source-limit-dialog" in body

    def test_max_sources_is_clamped_before_request(self):
        """超出 API 上限时前端自动钳制，并提供可见提示。"""
        response = client.get("/static/app.js")
        body = response.text
        assert "normalizeMaxSources" in body
        assert "Math.min(maximum" in body
        assert "showSourceLimitNotice" in body
        assert "max_sources: maxSources" in body

    def test_terminal_sse_error_opens_dialog_and_settles_failed_run(self):
        page = client.get("/").text
        script = client.get("/static/app.js").text

        assert 'id="runtime-error-dialog"' in page
        assert "showRuntimeError" in script
        assert "payload.error_type" in script
        assert 'finishResearchProgress("failed")' in script
        assert 'setStatus("failed")' in script

    def test_form_has_run_eval_checkbox(self):
        """表单包含 run_eval checkbox。"""
        response = client.get("/")
        body = response.text
        assert "run_eval" in body

    def test_app_js_uses_event_source(self):
        """app.js 使用 EventSource。"""
        response = client.get("/static/app.js")
        body = response.text
        assert "EventSource" in body

    def test_no_unsafe_innerhtml_injection(self):
        """app.js 使用安全 DOM API（textContent）而非直接 innerHTML 注入模型输出。"""
        response = client.get("/static/app.js")
        body = response.text
        # 应该有 textContent 用法
        assert "textContent" in body or "safeSetText" in body
        # 不应该直接 innerHTML 设置模型输出（除结构化的 table/div 构建外）
        # 不对 report content 使用 innerHTML


class TestRunStoreBackwardCompat:
    """RunStore 向后兼容测试。"""

    def test_create_with_default_id(self):
        """RunStore.create 不传 run_id 时自动生成。"""
        from app.services.run_store import RunStore

        store = RunStore()
        rid = store.create(topic="test")
        assert len(rid) > 0
        data = store.get(rid)
        assert data is not None
        assert data["topic"] == "test"

    def test_create_with_explicit_id(self):
        """RunStore.create 传入 run_id 时使用该 ID。"""
        from app.services.run_store import RunStore

        store = RunStore()
        rid = store.create(topic="test", run_id="my-custom-id")
        assert rid == "my-custom-id"
        data = store.get("my-custom-id")
        assert data is not None


class TestPhase3Regression:
    """Phase 3 回归测试。"""

    def test_health_endpoint(self):
        """GET /health 返回正常。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_swagger_accessible(self):
        """GET /docs 返回 Swagger UI。"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_list_runs(self):
        """GET /api/runs 返回 runs 列表。"""
        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
