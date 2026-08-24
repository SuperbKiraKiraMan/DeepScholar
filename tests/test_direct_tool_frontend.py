"""Direct-tool conversational rendering regression tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def test_direct_response_surface_and_thinking_state_are_rendered():
    html = client.get("/").text
    for element_id in (
        "direct-response",
        "direct-thinking",
        "direct-thinking-title",
        "direct-provider-notice",
        "direct-results",
        "direct-source-list",
        "direct-pagination",
        "btn-direct-more",
        "discovered-source-count",
        "analyzed-source-count",
    ):
        assert f'id="{element_id}"' in html
    assert 'aria-live="polite"' in html


def test_direct_mode_hides_deep_research_runtime_chrome():
    css = client.get("/static/styles.css").text
    assert ".assistant-message.is-direct-mode .deep-response" in css
    assert ".assistant-message.is-deep-mode .direct-response" in css
    assert "@keyframes thinking-pulse" in css


def test_frontend_switches_modes_from_controller_route():
    script = client.get("/static/app.js").text
    assert 'setExecutionDisplayMode("direct")' in script
    assert 'setExecutionDisplayMode("deep")' in script
    assert 'payload.execution_route === "direct_tool"' in script
    assert 'eventType === "source_found"' in script
    assert "upsertDirectSource(payload)" in script
    assert 'eventType === "tool_loop_fallback"' in script
    assert "showDirectProviderNotice(" in script


def test_direct_results_use_safe_dom_rendering_and_final_sources():
    script = client.get("/static/app.js").text
    assert "renderDirectSource(source, index, ordinalOffset)" in script
    assert "elements.directSourceList.appendChild" in script
    assert "innerHTML" not in script
    assert "renderDirectResults(sources, data.topic)" in script


def test_archived_direct_results_keep_paper_cards_across_turns():
    script = client.get("/static/app.js").text
    assert "function renderArchivedDirectResults(data, container, ordinalOffset)" in script
    assert 'resultData.execution_route === "direct_tool"' in script
    assert "appendArchivedExchange(lastResultData, currentPaperOrdinalStart)" in script
    assert 'node("div", "direct-source-list")' in script


def test_react_answer_markdown_is_rendered_in_live_and_archived_conversation():
    html = client.get("/").text
    script = client.get("/static/app.js").text
    css = client.get("/static/styles.css").text
    # 对话里新增了 ReAct 回复正文的容器（助手的自然语言开场 + 收尾），来源卡片保留为详细列表。
    assert 'id="direct-answer"' in html
    assert "elements.directAnswer" in script
    assert "function renderDirectAnswer(answer, container)" in script
    assert "renderDirectAnswer(data.answer, elements.directAnswer)" in script
    # 归档轮次同样展示自然语言回复正文，且跳过旧版"# 标题 + 编号列表"格式的迁移保护。
    assert 'node("div", "archived-direct-answer report-content")' in script
    assert "buildReport(answer, false, answerBox, { interactiveCitations: false })" in script
    assert "function isLegacyDirectAnswer(answer)" in script
    # 回复正文里的 [标题](url) 链接可点击；答案不再带 # 大标题回显用户问题。
    assert "node(\"a\", \"report-link\")" in script
    assert ".report-content .report-link" in css
    # 汇总行与回复同时存在时被隐藏，避免"整理 N 篇"一句话说两遍。
    assert "elements.directResultsSummary.hidden = hasReply && sources.length > 0" in script


def test_direct_paper_ordinals_continue_within_a_conversation():
    script = client.get("/static/app.js").text
    assert "let nextPaperOrdinal = 0;" in script
    assert "let currentPaperOrdinalStart = 0;" in script
    assert "currentPaperOrdinalStart = nextPaperOrdinal;" in script
    assert "nextPaperOrdinal = Math.max(nextPaperOrdinal, currentPaperOrdinalStart + sources.length);" in script


def test_direct_results_can_load_provider_pages_and_deep_research_shows_counts():
    script = client.get("/static/app.js").text
    assert '"/papers?offset=" + requestOffset + "&limit=20"' in script
    assert "loadMoreDirectPapers" in script
    assert "page.has_more" in script
    assert "data.discovered_source_count" in script
    assert "data.analyzed_source_count" in script


def test_source_found_sse_payload_contains_progressive_display_fields():
    runtime = (ROOT / "app/graph/runtime.py").read_text(encoding="utf-8")
    for field in (
        '"url": s.get("url", "")',
        '"authors": (s.get("authors", []) or [])[:8]',
        '"year": s.get("year")',
        '"provider": s.get("provider", "")',
        '"snippet": (s.get("snippet", "") or "")[:320]',
    ):
        assert field in runtime
