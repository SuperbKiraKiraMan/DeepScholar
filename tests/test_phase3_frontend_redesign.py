"""Phase 3 frontend redesign contract tests."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_dashboard_html_is_not_cached_during_frontend_iteration():
    response = client.get("/")
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"


def test_deep_research_uses_compact_progress_then_report_artifacts():
    body = client.get("/").text
    for element_id in (
        "deep-response",
        "overview-running",
        "activity-feed",
        "deep-report",
        "report-content",
        "research-artifacts",
        "btn-skip-typing",
    ):
        assert f'id="{element_id}"' in body
    assert 'class="phase-rail"' not in body
    for tab in ("trace", "sources", "evidence", "quality"):
        assert f'data-tab="{tab}"' in body
    assert 'data-tab="overview"' not in body
    assert 'data-tab="report"' not in body


def test_report_uses_progressive_gpt_style_typing():
    script = client.get("/static/app.js").text
    css = client.get("/static/styles.css").text
    assert "function renderReport(text, animate, onComplete)" in script
    assert "function typeNext()" in script
    assert "Array.from(content)" in script
    assert "stopReportTyping(true)" in script
    assert ".report-content.is-typing::after" in css
    assert ".research-progress-card" in css


def test_report_typing_hides_future_markdown_structure_until_text_arrives():
    script = client.get("/static/app.js").text
    assert 'const blockSelector = "h1, h2, h3, p, li, blockquote, tr"' in script
    assert 'const containerSelector = "ul, ol, .report-table-wrap"' in script
    assert "element.hidden = true" in script
    assert "element.hidden = false" in script
    assert "segment.structural" in script
    assert "appendText(inlineNode, token)" in script


def test_frontend_calls_existing_cancel_endpoint():
    script = client.get("/static/app.js").text
    assert '"/api/research/runs/"' in script
    assert '"/cancel"' in script
    assert 'method: "POST"' in script
    assert "showCancelledState();" in script
    assert 'finishResearchProgress("cancelled")' in script
    assert 'elements.activityTitle.textContent = cancelled ? "调研已中止"' in script


def test_frontend_uses_session_scoped_multi_turn_workspace():
    html = client.get("/").text
    script = client.get("/static/app.js").text
    assert 'class="search-band composer-dock"' in html
    assert 'class="query-summary"' in html
    assert 'class="assistant-message research-results"' in html
    assert 'id="conversation-thread"' in html
    assert "function archiveCurrentExchange()" in script
    assert 'fetch("/api/sessions"' in script
    assert 'session_id: activeSessionId' in script


def test_v23_is_a_two_pane_light_research_workspace():
    html = client.get("/").text
    css = client.get("/static/styles.css").text
    assert '<meta name="color-scheme" content="light">' in html
    assert 'class="dashboard research-shell"' in html
    assert 'class="control-sidebar workspace-sidebar"' in html
    assert 'class="research-artifacts evidence-sidebar"' in html
    assert 'class="app-header"' in html
    assert 'class="primary-nav"' in html
    assert 'class="search-examples"' in html
    assert 'class="chat-shell"' not in html
    assert 'composer-dock' in html
    assert 'class="user-message"' not in html
    assert "--bg: #ffffff" in css
    assert "grid-template-columns: 252px minmax(0, 1fr)" in css
    assert ".evidence-sidebar, .research-shell.workspace-page-mode .evidence-sidebar { display: none !important; }" in css
    assert ".control-sidebar.is-open" in css
    # v2.4: 抽屉机制保留,用于移动端左侧导航(证据面板已整体退役)
    assert "transform: translateX(-104%)" in css


def test_ambiguous_short_input_only_gets_clarification_before_first_turn():
    script = client.get("/static/app.js").text
    assert "function isAmbiguousResearchRequest(value)" in script
    assert "function showClarificationRequest(topic)" in script
    assert "我是 Academic Research Copilot 学术研究助手" in script
    assert "if (isAmbiguousResearchRequest(topic) && sessionTurnCount === 0)" in script


def test_phase2_routes_and_outline_have_dedicated_ui_contracts():
    html = client.get("/").text
    script = client.get("/static/app.js").text
    for route in ("paper_qa", "paper_compare", "report_follow_up"):
        assert route in script
    for element_id in ("session-id", "session-summary", "outline-list", "outline-empty"):
        assert f'id="{element_id}"' in html


def test_mobile_controls_use_drawer_breakpoint():
    html = client.get("/").text
    css = client.get("/static/styles.css").text
    assert 'id="btn-config-toggle"' in html
    assert 'id="sidebar-backdrop"' in html
    assert "@media (max-width: 860px)" in css
    assert ".control-sidebar.is-open" in css


def test_sidebar_collapse_works_on_desktop_and_restores_on_mobile():
    """v2.4.1: 桌面端可折叠侧栏;移动端抽屉不受折叠类影响。"""
    html = client.get("/").text
    css = client.get("/static/styles.css").text
    script = client.get("/static/app.js").text
    # 新对话按钮纳入 workspace 导航组,切换时高亮互斥
    assert 'id="btn-focus-topic" class="nav-item is-active" type="button" data-workspace="research"' in html
    # 桌面折叠:侧栏隐藏、主区占满
    assert ".research-shell.sidebar-collapsed .workspace-sidebar { display: none; }" in css
    assert "grid-template-columns: 0 minmax(0, 1fr)" in css
    # 移动端 860px 断点内恢复抽屉行为
    assert ".research-shell.sidebar-collapsed .workspace-sidebar { display: flex; }" in css
    # JS: 桌面走折叠切换,移动端走抽屉开合;跨断点清理残留
    assert "function isMobileLayout()" in script
    assert "toggleDesktopSidebar" in script
    assert "sidebar-collapsed" in script
    assert "if (!isMobileLayout()) closeSidebar();" in script


def test_conversation_has_visible_scrollbar_and_manual_scroll_lock():
    css = client.get("/static/styles.css").text
    script = client.get("/static/app.js").text
    assert "overflow-y: scroll" in css
    assert ".chat-scroll::-webkit-scrollbar" in css
    assert "scrollbar-color: #b9bcc2 transparent" in css
    assert "if (!force && !followLatest) return;" in script


def test_archived_conversation_preserves_full_markdown_and_table_layout():
    script = client.get("/static/app.js").text
    css = client.get("/static/styles.css").text
    assert 'node("div", "archived-report-content report-content")' in script
    assert 'buildReport(answer, false, content, { interactiveCitations: false });' in script
    assert "answer.length > 1400" not in script
    assert "function buildReport(text, animate, container, options)" in script
    assert ".archived-report-content .report-table-wrap" in css
    assert ".archived-report-content .report-table" in css
