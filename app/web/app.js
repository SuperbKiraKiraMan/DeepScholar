/** Academic Research Copilot - framework-free conversational UI controller. */
(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const elements = {
    btnStart: $("#btn-start"), btnCancel: $("#btn-cancel"), btnQuery: $("#btn-query"),
    btnHistoryRefresh: $("#btn-history-refresh"), btnHistoryMore: $("#btn-history-more"),
    btnConfig: $("#btn-config-toggle"), btnSidebarClose: $("#btn-sidebar-close"),
    btnFocusTopic: $("#btn-focus-topic"), sidebar: $("#config-panel"), backdrop: $("#sidebar-backdrop"),
    topic: $("#topic"), backend: $("#backend"), agentMode: $("#agent_mode"),
    maxSources: $("#max_sources"), language: $("#language"), runEval: $("#run_eval"),
    queryRunId: $("#query-run-id"), currentRunId: $("#current-run-id"),
    historyList: $("#history-list"), historyEmpty: $("#history-empty"),
    runStatus: $("#run-status"), elapsed: $("#elapsed-time"), connection: $("#connection-badge"),
    workspaceTitle: $("#workspace-title"), workspaceSubtitle: $("#workspace-subtitle"),
    workspaceState: $("#workspace-state-text"), overviewIdle: $("#overview-idle"),
    overviewRunning: $("#overview-running"), activityTitle: $("#current-activity-title"),
    activityDetail: $("#current-activity-detail"), activityTime: $("#current-activity-time"),
    activityFeed: $("#activity-feed"), traceContainer: $("#trace-container"),
    traceEmpty: $("#trace-empty"), traceEvents: $("#trace-events"), traceCount: $("#trace-count"),
    dagContainer: $("#dag-container"), dagDisplay: $("#dag-display"),
    errorsContainer: $("#errors-container"), errorsList: $("#errors-list"),
    resultsEmpty: $("#results-empty"), reportSection: $("#report-section"),
    reportContent: $("#report-content"), sourcesSection: $("#sources-section"),
    sourcesTable: $("#sources-table"), sourcesEmpty: $("#sources-empty"),
    evidenceSection: $("#evidence-section"), evidenceList: $("#evidence-list"),
    evidenceEmpty: $("#evidence-empty"), citationSection: $("#citation-section"),
    citationTable: $("#citation-table"), citationEmpty: $("#citation-empty"),
    evalSection: $("#eval-section"), evalDisplay: $("#eval-display"), evalEmpty: $("#eval-empty"),
    observabilitySection: $("#observability-section"), observabilityDisplay: $("#observability-display"),
    observabilityEmpty: $("#observability-empty"),
    sourceLimitDialog: $("#source-limit-dialog"), sourceLimitMessage: $("#source-limit-message"),
    runtimeErrorDialog: $("#runtime-error-dialog"), runtimeErrorMessage: $("#runtime-error-message"),
    metricSources: $("#metric-sources"), metricEvidence: $("#metric-evidence"),
    metricCitations: $("#metric-citations"), metricRuntime: $("#metric-runtime"),
    chatScroll: $("#chat-scroll"), userMessage: $("#user-message"),
    userMessageText: $("#user-message-text"), assistantMessage: $("#assistant-message"),
    routeBadge: $("#route-badge"), directResponse: $("#direct-response"),
    directThinking: $("#direct-thinking"), directThinkingTitle: $("#direct-thinking-title"),
    directThinkingDetail: $("#direct-thinking-detail"), directResults: $("#direct-results"),
    directAnswer: $("#direct-answer"),
    directResultsTitle: $("#direct-results-title"), directResultsCount: $("#direct-results-count"),
    directResultsSummary: $("#direct-results-summary"), directSourceList: $("#direct-source-list"),
    directPagination: $("#direct-pagination"), btnDirectMore: $("#btn-direct-more"),
    directPaginationStatus: $("#direct-pagination-status"),
    directProviderNotice: $("#direct-provider-notice"),
    directProviderNoticeTitle: $("#direct-provider-notice-title"),
    directProviderNoticeDetail: $("#direct-provider-notice-detail"),
    deepResponse: $("#deep-response"), deepReport: $("#deep-report"),
    discoveredSourceCount: $("#discovered-source-count"),
    analyzedSourceCount: $("#analyzed-source-count"),
    researchArtifacts: $("#research-artifacts"), artifactSummary: $("#artifact-summary"),
    reportStateLabel: $("#report-state-label"), btnSkipTyping: $("#btn-skip-typing"),
    conversationThread: $("#conversation-thread"), sessionId: $("#session-id"),
    sessionTopic: $("#session-topic"), sessionSummary: $("#session-summary"),
    btnNewSessionTop: $("#btn-new-session-top"), outlineSection: $("#outline-section"),
    outlineList: $("#outline-list"), outlineEmpty: $("#outline-empty"),
    shell: $(".research-shell"), composerDock: $(".composer-dock"),
    libraryView: $("#library-view"), librarySearch: $("#library-search"),
    libraryCount: $("#library-count"), libraryList: $("#library-list"),
    libraryEmpty: $("#library-empty"), libraryDetail: $("#library-detail"),
    paperDetailQuery: $("#paper-detail-query"),
    paperDetailButton: $("#paper-detail-button"),
    paperDetailResult: $("#paper-detail-result"),
    evidenceLibraryView: $("#evidence-library-view"),
    evidenceLibrarySearch: $("#evidence-library-search"),
    evidenceLibraryCount: $("#evidence-library-count"),
    evidenceLibraryList: $("#evidence-library-list"),
    evidenceLibraryEmpty: $("#evidence-library-empty"),
    evidenceLibraryDetail: $("#evidence-library-detail"),
    reportsView: $("#reports-view"), reportsSearch: $("#reports-search"),
    reportsCount: $("#reports-count"), reportsList: $("#reports-list"),
    reportsEmpty: $("#reports-empty"), reportLibraryDetail: $("#report-library-detail")
  };

  const phaseOrder = ["plan", "search", "read", "analyze", "cite", "evaluate", "review"];
  let activeRunId = null;
  let activeEventSource = null;
  let elapsedInterval = null;
  let startTime = null;
  let submitting = false;
  let traceEventCount = 0;
  let sourceIds = new Set();
  let evidenceKeys = new Set();
  let completedChapterHeadings = new Set();
  let historyOffset = 0;
  let executionDisplayMode = "routing";
  let currentIntent = "";
  let currentResearchPhase = "";
  let reportTypingTimer = null;
  let pendingReportText = "";
  let reportTypingComplete = null;
  let directNextOffset = 0;
  let directHasMore = false;
  let directPageLoading = false;
  let currentDirectToolArgs = {};
  let currentRunStatus = "idle";
  let citationEvidenceTargets = new Map();
  // 重要步骤：统一会话存储键，避免把本地会话数据与其他应用混用。
  const SESSION_STORAGE_KEY = "academic_research_copilot_session_id";
  let activeSessionId = window.sessionStorage.getItem(SESSION_STORAGE_KEY) || "";
  let sessionTurnCount = 0;
  let lastResultData = null;
  // 记录当前会话已经展示过的论文数量，让“继续推荐”沿用上一轮的序号。
  let nextPaperOrdinal = 0;
  let currentPaperOrdinalStart = 0;
  let followLatest = true;
  let activeWorkspace = "research";
  let libraryOrigin = "all";
  let collectionSearchTimer = null;
  const workspaceCache = { library: null, evidence: null, reports: null };
  const historyPageSize = 5;

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }

  function updateSessionDisplay(context) {
    if (!context) return;
    activeSessionId = context.session_id || activeSessionId;
    sessionTurnCount = Number(context.turn_count || 0);
    if (activeSessionId) window.sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
    elements.sessionId.textContent = activeSessionId || "会话不可用";
    elements.sessionSummary.textContent = sessionTurnCount + " 轮对话" +
      (context.compaction_count ? " · 已压缩 " + context.compaction_count + " 次" : "") +
      (context.active_paper_id ? " · 已定位论文" : "") +
      (context.restored_from_session_id ? " · 已恢复历史 Session" : "");
  }

  function createSession() {
    return fetch("/api/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ttl_minutes: 30 })
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    }).then(function (context) {
      updateSessionDisplay(context);
      return context;
    });
  }

  function restoreSession(sessionId) {
    if (!sessionId) return Promise.reject(new Error("没有可恢复的历史 Session"));
    return fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ttl_minutes: 30 })
    }).then(function (response) {
      if (!response.ok) {
        return response.json().catch(function () { return {}; }).then(function (error) {
          throw new Error(error.detail && error.detail.message || error.detail || "HTTP " + response.status);
        });
      }
      return response.json();
    }).then(function (context) {
      updateSessionDisplay(context);
      return context;
    });
  }

  function ensureSession() {
    if (!activeSessionId) return createSession();
    return fetch("/api/sessions/" + encodeURIComponent(activeSessionId))
      .then(function (response) {
        if (response.status === 404 || response.status === 410) return restoreSession(activeSessionId);
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (context) { updateSessionDisplay(context); return context; });
  }

  function refreshSession() {
    if (!activeSessionId) return Promise.resolve();
    return fetch("/api/sessions/" + encodeURIComponent(activeSessionId))
      .then(function (response) { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); })
      .then(updateSessionDisplay)
      .catch(function () { return restoreSession(activeSessionId); });
  }

  function appendArchivedExchange(resultData, ordinalOffset) {
    if (!resultData || !elements.conversationThread) return;
    const answer = resultData.answer || resultData.final_report || resultData.draft_report || "";
    const turn = node("article", "archived-turn");
    turn.dataset.runId = resultData.run_id || "";
    const query = node("div", "archived-query", resultData.topic || "研究问题");
    const response = node("div", "archived-answer");
    const content = node("div", "archived-report-content report-content");
    response.append(node(
      "div", "archived-meta",
      routeLabel(resultData.intent, resultData.execution_route, resultData.selected_tools)
    ));
    // 关键步骤：归档多轮对话时保留直接论文推荐的结构化来源，避免卡片降级成纯文本。
    if (resultData.execution_route === "direct_tool" && Array.isArray(resultData.sources) && resultData.sources.length) {
      renderArchivedDirectResults(resultData, content, Number(ordinalOffset || 0));
    } else {
      // 深度报告仍按 Markdown 归档；归档内容不绑定当前轮次的证据跳转。
      content.classList.add("report-content");
      buildReport(answer, false, content, { interactiveCitations: false });
    }
    response.appendChild(content);
    turn.append(query, response);
    elements.conversationThread.appendChild(turn);
  }

  function archiveCurrentExchange() {
    if (!lastResultData || !elements.conversationThread) return;
    appendArchivedExchange(lastResultData, currentPaperOrdinalStart);
    lastResultData = null;
  }

  function recommendationRange(resultData, fallbackOffset) {
    const trace = Array.isArray(resultData && resultData.trace) ? resultData.trace : [];
    const event = trace.find(function (entry) {
      return entry && entry.event === "direct_reviewer_complete";
    }) || {};
    const startNumber = Number(
      resultData.recommendation_number_start ?? event.recommendation_number_start
    );
    const endNumber = Number(
      resultData.recommendation_number_end ?? event.recommendation_number_end
    );
    const count = Array.isArray(resultData.sources) ? resultData.sources.length : 0;
    const startOffset = Number.isFinite(startNumber) && startNumber >= 1
      ? startNumber - 1 : Number(fallbackOffset || 0);
    const endOffset = Number.isFinite(endNumber) && endNumber >= startOffset + 1
      ? endNumber : startOffset + count;
    return { startOffset: startOffset, endOffset: endOffset };
  }

  function deduplicateHistorySources(resultData, seenSourceKeys) {
    if (!resultData || !Array.isArray(resultData.sources)) return resultData;
    const copy = Object.assign({}, resultData);
    copy.sources = resultData.sources.filter(function (source) {
      const id = String(source.source_id || source.paper_id || "").trim().toLowerCase();
      const title = String(source.title || "").trim().toLowerCase().replace(/\s+/g, " ");
      const keys = [id ? "id:" + id : "", title ? "title:" + title : ""].filter(Boolean);
      if (keys.some(function (key) { return seenSourceKeys.has(key); })) return false;
      keys.forEach(function (key) { seenSourceKeys.add(key); });
      return true;
    });
    return copy;
  }

  function renderArchivedDirectResults(data, container, ordinalOffset) {
    const sources = Array.isArray(data.sources) ? data.sources : [];
    const results = node("section", "archived-direct-results");
    // 关键步骤：归档对话先展示助手的自然语言回复（与 live 一致），
    // 论文列表交给下方来源卡片，避免 markdown 正文与卡片两套样式并存。
    const answer = String(data.answer || "").trim();
    const showReply = Boolean(answer) && !isLegacyDirectAnswer(answer);
    if (showReply) {
      const answerBox = node("div", "archived-direct-answer report-content");
      buildReport(answer, false, answerBox, { interactiveCitations: false });
      results.appendChild(answerBox);
    }
    const heading = node("div", "direct-results-heading");
    const headingCopy = node("div");
    headingCopy.append(
      node("span", "eyebrow", "Research results"),
      node("h2", "", directIntentCopy(data.intent).title)
    );
    heading.append(headingCopy, node("span", "direct-results-count", sources.length + " sources"));
    results.appendChild(heading);
    // 汇总行只在没有回复正文时兜底展示（回复里已交代整理结果，避免重复）。
    if (!showReply) {
      results.appendChild(node(
        "p", "direct-results-summary",
        "已根据“" + (data.topic || "你的请求") + "”整理 " + sources.length + " 条可追溯学术来源。"
      ));
    }

    const list = node("div", "direct-source-list");
    sources.forEach(function (source, index) {
      list.appendChild(renderDirectSource(source, index, ordinalOffset));
    });
    results.appendChild(list);
    container.appendChild(results);
  }

  function renderOutline(outline) {
    elements.outlineList.replaceChildren();
    let sections = outline && Array.isArray(outline.sections) ? outline.sections : [];
    sections.forEach(function (section) {
      const title = typeof section === "string" ? section :
        section.title || section.heading || section.section_title || "未命名章节";
      const item = node("li", "", title);
      item.addEventListener("click", function () {
        elements.topic.value = "继续展开报告中的“" + title + "”部分";
        resizeComposer();
        elements.topic.focus();
      });
      elements.outlineList.appendChild(item);
    });
    elements.outlineSection.hidden = sections.length === 0;
    elements.outlineEmpty.hidden = sections.length > 0;
  }

  function clearDetail(container, title, copy) {
    container.replaceChildren();
    const placeholder = node("div", "detail-placeholder");
    placeholder.append(node("strong", "", title), node("p", "", copy));
    container.appendChild(placeholder);
  }

  function appendDetailSection(container, title, content, quote) {
    if (!content) return;
    const section = node("section", "detail-section");
    section.appendChild(node("h3", "", title));
    section.appendChild(node(quote ? "blockquote" : "p", "", content));
    container.appendChild(section);
  }

  function formatBytes(value) {
    const bytes = Number(value || 0);
    if (!bytes) return "—";
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  function selectCollectionItem(list, selected) {
    list.querySelectorAll(".collection-item").forEach(function (item) {
      item.classList.toggle("is-selected", item === selected);
    });
  }

  function showPaperDetail(paper, button) {
    selectCollectionItem(elements.libraryList, button);
    const detail = elements.libraryDetail;
    detail.replaceChildren();
    detail.append(node("span", "detail-kicker", (paper.origins || []).join(" + ") || "paper"), node("h2", "detail-title", paper.title || "Untitled paper"));
    const meta = node("div", "detail-meta");
    meta.append(node("span", "", (paper.authors || []).join(", ") || "作者未知"), node("span", "", paper.year || "年份未知"), node("span", "", paper.provider || paper.source_type || "来源未知"));
    detail.appendChild(meta);
    const stats = node("div", "detail-stats");
    [["本地分块", paper.chunk_count || "—"], ["被引", paper.cited_by_count ?? "—"], ["历史研究", (paper.seen_in_runs || []).length]].forEach(function (entry) {
      const card = node("div", "detail-stat"); card.append(node("span", "", entry[0]), node("strong", "", entry[1])); stats.appendChild(card);
    });
    detail.appendChild(stats);
    appendDetailSection(detail, "摘要 / 检索片段", paper.snippet || "暂无摘要。");
    appendDetailSection(detail, "DOI", paper.doi);
    appendDetailSection(detail, "本地索引", paper.source_path ? paper.source_path + "\nZotero key: " + (paper.zotero_storage_key || "—") + " · " + formatBytes(paper.size_bytes) + " · indexed " + (paper.indexed_at || "—") : "");
    appendDetailSection(detail, "出现于研究", (paper.seen_in_runs || []).join(", "));
    if (/^https?:\/\//i.test(paper.url || "")) {
      const link = node("a", "detail-link", "打开论文来源 ↗"); link.href = paper.url; link.target = "_blank"; link.rel = "noopener noreferrer"; detail.appendChild(link);
    }
  }

  function renderLibrary(data) {
    elements.libraryList.replaceChildren();
    const items = Array.isArray(data.items) ? data.items : [];
    elements.libraryCount.textContent = data.total + " 篇 · 本地 " + data.local_count + " · 搜索历史 " + data.searched_count;
    elements.libraryEmpty.hidden = items.length > 0;
    items.forEach(function (paper) {
      const button = node("button", "collection-item"); button.type = "button";
      button.append(node("span", "collection-item-title", paper.title || "Untitled paper"), node("span", "collection-item-meta", [(paper.authors || []).slice(0, 3).join(", ") || "作者未知", paper.year || "年份未知", paper.provider || "unknown"].join(" · ")));
      const origins = node("span", "origin-row"); (paper.origins || []).forEach(function (origin) { origins.appendChild(node("span", "origin-badge", origin === "local" ? "本地索引" : "搜索历史")); }); button.appendChild(origins);
      button.addEventListener("click", function () { showPaperDetail(paper, button); });
      elements.libraryList.appendChild(button);
    });
    clearDetail(elements.libraryDetail, "选择一篇论文", "查看作者、年份、来源、索引状态和历史研究记录。");
  }

  function renderPaperDetailResult(data) {
    const container = elements.paperDetailResult;
    container.replaceChildren();
    container.hidden = false;
    if (!data || !data.found) {
      const placeholder = node("div", "detail-placeholder");
      placeholder.append(
        node("strong", "", "未找到论文详情"),
        node("p", "", (data && data.error) || "本地库与在线数据源均未命中该标题。")
      );
      container.appendChild(placeholder);
      return;
    }
    const paper = data.paper || {};
    const isLocal = data.matched_local === true;
    const kicker = isLocal ? "本地命中 · Zotero"
      : (data.provider === "openalex" ? "OpenAlex" : "Semantic Scholar");
    container.append(
      node("span", "detail-kicker", kicker),
      node("h2", "detail-title", paper.title || data.query || "Untitled paper")
    );
    const meta = node("div", "detail-meta");
    meta.append(
      node("span", "", (paper.authors || []).join(", ") || "作者未知"),
      node("span", "", paper.year || "年份未知"),
      node("span", "", paper.venue || paper.provider || "来源未知")
    );
    container.appendChild(meta);
    const stats = node("div", "detail-stats");
    [["被引", paper.cited_by_count ?? "—"], ["参考文献", paper.reference_count ?? "—"]].forEach(function (entry) {
      const card = node("div", "detail-stat"); card.append(node("span", "", entry[0]), node("strong", "", entry[1])); stats.appendChild(card);
    });
    if (isLocal && data.local) {
      const local = data.local;
      [["本地分块", local.chunk_count ?? "—"], ["Zotero key", local.zotero_storage_key || "—"]].forEach(function (entry) {
        const card = node("div", "detail-stat"); card.append(node("span", "", entry[0]), node("strong", "", entry[1])); stats.appendChild(card);
      });
    }
    container.appendChild(stats);
    appendDetailSection(container, "摘要", data.abstract || paper.snippet || "暂无摘要。");
    appendDetailSection(container, "DOI", paper.doi);
    if (isLocal && data.local) {
      const local = data.local;
      appendDetailSection(
        container, "本地 PDF",
        local.source_path ? local.source_path + "\nZotero key: " + (local.zotero_storage_key || "—") + " · " + local.chunk_count + " chunks" : ""
      );
    }
    if (/^https?:\/\//i.test(paper.url || "")) {
      const link = node("a", "detail-link", "打开论文来源 ↗"); link.href = paper.url; link.target = "_blank"; link.rel = "noopener noreferrer"; container.appendChild(link);
    }
  }

  function queryPaperDetail() {
    const query = elements.paperDetailQuery.value.trim();
    if (!query) { elements.paperDetailQuery.focus(); return; }
    elements.paperDetailButton.disabled = true;
    fetch("/api/papers/detail?query=" + encodeURIComponent(query))
      .then(function (response) { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); })
      .then(renderPaperDetailResult)
      .catch(function (error) { renderPaperDetailResult({ found: false, error: error.message }); })
      .finally(function () { elements.paperDetailButton.disabled = false; });
  }

  function showEvidenceDetail(card, button) {
    selectCollectionItem(elements.evidenceLibraryList, button);
    const detail = elements.evidenceLibraryDetail; detail.replaceChildren();
    detail.append(node("span", "detail-kicker", "Evidence " + (card.evidence_id || "")), node("h2", "detail-title", card.claim || "Untitled claim"));
    const meta = node("div", "detail-meta"); meta.append(node("span", "", "置信度 " + (card.confidence ?? "—")), node("span", "", card.evidence_type || "primary_claim"), node("span", "", card.page_or_section || card.quote_location || "位置未知")); detail.appendChild(meta);
    appendDetailSection(detail, "原文证据", card.quote || card.evidence_quote || card.original_quote, true);
    appendDetailSection(detail, "来源论文", card.source_title + (card.source_year ? " (" + card.source_year + ")" : ""));
    appendDetailSection(detail, "对应研究", card.run_topic + "\nRun " + card.run_id);
    appendDetailSection(detail, "方法 / 数据集", [card.method, card.dataset || card.dataset_name].filter(Boolean).join("\n"));
    appendDetailSection(detail, "局限", card.limitation);
    if (/^https?:\/\//i.test(card.source_url || "")) { const link = node("a", "detail-link", "打开证据来源 ↗"); link.href = card.source_url; link.target = "_blank"; link.rel = "noopener noreferrer"; detail.appendChild(link); }
  }

  function renderEvidenceLibrary(data) {
    elements.evidenceLibraryList.replaceChildren();
    const items = Array.isArray(data.items) ? data.items : [];
    elements.evidenceLibraryCount.textContent = data.total + " 条证据";
    elements.evidenceLibraryEmpty.hidden = items.length > 0;
    items.slice(0, 500).forEach(function (card) {
      const button = node("button", "collection-item"); button.type = "button";
      button.append(node("span", "collection-item-title", card.claim || "Untitled claim"), node("span", "collection-item-meta", [card.source_title, card.source_year, "置信度 " + (card.confidence ?? "—")].filter(Boolean).join(" · ")), node("span", "collection-item-preview", card.quote || card.evidence_quote || "暂无原文片段"));
      button.addEventListener("click", function () { showEvidenceDetail(card, button); }); elements.evidenceLibraryList.appendChild(button);
    });
    clearDetail(elements.evidenceLibraryDetail, "选择一条证据", "查看完整引文、来源论文、置信度和对应研究。");
  }

  function renderDocument(container, text) {
    const body = node("div", "report-library-body");
    String(text || "").split(/\r?\n/).forEach(function (line) {
      const heading = line.trim().match(/^(#{1,3})\s+(.+)$/);
      if (heading) body.appendChild(node("h" + Math.min(3, heading[1].length), "", heading[2]));
      else if (line.trim()) body.appendChild(node("p", "", line.trim()));
    });
    container.appendChild(body);
  }

  function showReportDetail(report, button) {
    selectCollectionItem(elements.reportsList, button);
    const detail = elements.reportLibraryDetail; detail.replaceChildren();
    detail.append(node("span", "detail-kicker", "Saved research report"), node("h2", "detail-title", report.topic || "Untitled report"));
    const meta = node("div", "detail-meta"); meta.append(node("span", "", formatHistoryDate(report.created_at)), node("span", "", report.source_count + " 个来源"), node("span", "", report.evidence_count + " 条证据"), node("span", "", formatElapsed(report.total_latency_ms))); detail.appendChild(meta);
    const link = node("button", "detail-link", "在研究对话中打开"); link.type = "button"; link.addEventListener("click", function () { elements.queryRunId.value = report.run_id; setWorkspace("research"); queryRun(); }); detail.appendChild(link);
    appendDetailSection(detail, "完整报告", ""); renderDocument(detail, report.report);
  }

  function renderReports(data) {
    elements.reportsList.replaceChildren();
    const items = Array.isArray(data.items) ? data.items : [];
    elements.reportsCount.textContent = data.total + " 份报告"; elements.reportsEmpty.hidden = items.length > 0;
    items.forEach(function (report) { const button = node("button", "collection-item"); button.type = "button"; button.append(node("span", "collection-item-title", report.topic || "Untitled report"), node("span", "collection-item-meta", [formatHistoryDate(report.created_at), report.source_count + " sources", report.evidence_count + " evidence"].join(" · ")), node("span", "collection-item-preview", String(report.report || "").replace(/[#*_`]/g, " ").slice(0, 180))); button.addEventListener("click", function () { showReportDetail(report, button); }); elements.reportsList.appendChild(button); });
    clearDetail(elements.reportLibraryDetail, "选择一份报告", "完整报告会显示在这里。");
  }

  function loadWorkspaceCollection(name, query) {
    const endpoints = { library: "/api/library?origin=" + encodeURIComponent(libraryOrigin) + "&query=", evidence: "/api/evidence-library?query=", reports: "/api/reports?query=" };
    const renderers = { library: renderLibrary, evidence: renderEvidenceLibrary, reports: renderReports };
    return fetch(endpoints[name] + encodeURIComponent(query || "")).then(function (response) { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); }).then(function (data) { workspaceCache[name] = data; renderers[name](data); }).catch(function (error) { addError("Could not load " + name + ": " + error.message); });
  }

  function setWorkspace(name) {
    name = name === "library" ? "library" : "research";
    activeWorkspace = name;
    const research = name === "research";
    elements.shell.classList.toggle("workspace-page-mode", !research);
    elements.chatScroll.hidden = !research; elements.composerDock.hidden = !research;
    elements.researchArtifacts.hidden = true;
    elements.libraryView.hidden = name !== "library";
    $$("[data-workspace]").forEach(function (button) { button.classList.toggle("is-active", button.dataset.workspace === name); });
    const titles = { research: "新对话", library: "论文库" };
    elements.workspaceState.textContent = titles[name]; closeSidebar();
    if (name === "library") loadWorkspaceCollection(name, "");
  }

  function scrollConversation(force) {
    if (!elements.chatScroll) return;
    if (!force && !followLatest) return;
    window.requestAnimationFrame(function () {
      elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
    });
  }

  function resizeComposer() {
    elements.topic.style.height = "auto";
    elements.topic.style.height = Math.min(elements.topic.scrollHeight, 150) + "px";
  }

  function routeLabel(intent, route, tools) {
    const labels = {
      paper_recommendation: "Paper recommendation",
      recommend_more: "More recommendations",
      literature_search: "Literature search",
      paper_graph_lookup: "Paper graph",
      deep_research: "Deep research",
      paper_qa: "Paper Q&A",
      paper_compare: "Paper comparison",
      report_follow_up: "Report follow-up",
      research_from_session: "Research from session papers",
      conversation: "Conversation"
    };
    const tool = Array.isArray(tools) && tools.length ? tools[0].replace(/^mcp__[^_]+__/, "") : "";
    return labels[intent] || (route === "direct_tool" ? "Direct tool" : "Research DAG") || tool;
  }

  function directIntentCopy(intent) {
    const copy = {
      paper_recommendation: {
        thinking: "正在寻找相似论文",
        detail: "正在调用论文推荐能力，并整理可追溯的候选结果。",
        title: "为你推荐这些论文"
      },
      literature_search: {
        thinking: "正在检索相关论文",
        detail: "正在搜索学术索引，并按相关性整理结果。",
        title: "找到这些相关论文"
      },
      paper_graph_lookup: {
        thinking: "正在展开论文关系",
        detail: "正在查询引用与参考文献关系。",
        title: "论文关系查询结果"
      }
    };
    return copy[intent] || {
      thinking: "正在调用研究能力",
      detail: "正在获取并整理可追溯的学术结果。",
      title: "研究结果"
    };
  }

  function setExecutionDisplayMode(mode) {
    executionDisplayMode = mode;
    elements.assistantMessage.classList.remove("is-routing", "is-direct-mode", "is-deep-mode");
    elements.assistantMessage.classList.add(
      mode === "direct" ? "is-direct-mode" : mode === "deep" ? "is-deep-mode" : "is-routing"
    );
    elements.directResponse.hidden = mode === "deep";
  }

  function setDirectThinking(title, detail) {
    elements.directThinking.hidden = false;
    setText(elements.directThinkingTitle, title, "正在处理请求");
    setText(elements.directThinkingDetail, detail, "");
  }

  function showDirectProviderNotice(title, detail) {
    elements.directProviderNotice.hidden = false;
    setText(elements.directProviderNoticeTitle, title, "已切换备用数据源");
    setText(elements.directProviderNoticeDetail, detail, "");
  }

  function resetDirectResponse() {
    currentIntent = "";
    elements.directResults.hidden = true;
    elements.directAnswer.hidden = true;
    elements.directAnswer.replaceChildren();
    elements.directSourceList.replaceChildren();
    elements.directResultsCount.textContent = "0 sources";
    elements.directResultsSummary.textContent = "";
    elements.directResultsTitle.textContent = "正在查找论文";
    elements.directProviderNotice.hidden = true;
    elements.directProviderNoticeDetail.textContent = "";
    directNextOffset = 0;
    directHasMore = false;
    directPageLoading = false;
    currentDirectToolArgs = {};
    elements.directPagination.hidden = true;
    elements.btnDirectMore.disabled = false;
    elements.btnDirectMore.hidden = false;
    elements.btnDirectMore.textContent = "加载更多论文";
    elements.directPaginationStatus.textContent = "继续从学术数据源获取下一页";
    setDirectThinking("正在理解你的请求", "正在选择最合适的学术研究能力。");
    setExecutionDisplayMode("routing");
  }

  function isAmbiguousResearchRequest(value) {
    const compact = String(value || "").trim().replace(/\s+/g, "");
    return /^(继续|继续推荐|再推荐|再来一些|更多|还有吗|换一批|推荐一下|推荐论文|帮我查一下|研究一下|这个|那个|你好|在吗)[。！？!?.]*$/.test(compact);
  }

  function showClarificationRequest(topic) {
    resetRunView();
    activeRunId = null;
    elements.overviewIdle.hidden = true;
    elements.userMessage.hidden = false;
    elements.userMessageText.textContent = topic;
    elements.assistantMessage.hidden = false;
    setExecutionDisplayMode("direct");
    elements.directThinking.hidden = true;
    showDirectProviderNotice(
      "请补充你的研究主题",
      "我是 Academic Research Copilot 学术研究助手，可以检索论文、分析引用关系并完成深度调研。请重新输入一个完整问题，例如：推荐 5 篇关于多模态实体对齐的论文。"
    );
    elements.routeBadge.textContent = "需要更多信息";
    elements.workspaceTitle.textContent = "等待明确的研究问题";
    elements.workspaceSubtitle.textContent = "当前输入缺少可独立执行的主题或论文信息。";
    elements.workspaceState.textContent = "等待补充研究主题";
    elements.currentRunId.textContent = "Needs details";
    setStatus("idle");
    resetForm();
    scrollConversation();
  }

  function updateDirectCount(count) {
    elements.directResultsCount.textContent = count + (count === 1 ? " source" : " sources");
  }

  function updateDirectPagination(hasMore, nextOffset, total) {
    directHasMore = Boolean(hasMore);
    if (Number.isInteger(nextOffset) && nextOffset >= 0) directNextOffset = nextOffset;
    elements.directPagination.hidden = !directHasMore;
    if (total !== undefined && total !== null) {
      elements.directPaginationStatus.textContent = "共找到 " + total + " 篇，继续加载下一页";
    } else {
      elements.directPaginationStatus.textContent = "按需继续从学术数据源获取，不设页面总量上限";
    }
  }

  function renderDirectSource(source, index, ordinalOffset) {
    const article = node("article", "direct-source-item");
    article.dataset.sourceId = source.source_id || "";
    const offset = Number.isFinite(Number(ordinalOffset)) ? Number(ordinalOffset) : 0;
    const ordinal = node("span", "direct-source-index", String(offset + index + 1).padStart(2, "0"));
    const main = node("div", "direct-source-main");
    let title;
    if (/^https?:\/\//i.test(source.url || "")) {
      title = node("a", "direct-source-title", source.title || "Untitled source");
      title.href = source.url;
      title.target = "_blank";
      title.rel = "noopener noreferrer";
    } else {
      title = node("span", "direct-source-title", source.title || "Untitled source");
    }
    main.appendChild(title);

    const meta = node("div", "direct-source-meta");
    [
      source.year,
      source.venue,
      source.provider || source.source_type,
      source.cited_by_count !== undefined && source.cited_by_count !== null
        ? source.cited_by_count + " citations" : ""
    ].filter(Boolean).forEach((value) => meta.appendChild(node("span", "", value)));
    if (meta.childNodes.length) main.appendChild(meta);

    if (Array.isArray(source.authors) && source.authors.length) {
      const visibleAuthors = source.authors.slice(0, 4).join(", ");
      main.appendChild(node(
        "p", "direct-source-authors",
        visibleAuthors + (source.authors.length > 4 ? " 等" : "")
      ));
    }
    if (source.snippet) main.appendChild(node("p", "direct-source-snippet", source.snippet));
    article.append(ordinal, main);
    return article;
  }

  function upsertDirectSource(source) {
    if (!source || !source.source_id) return;
    elements.directResults.hidden = false;
    const copy = directIntentCopy(currentIntent);
    elements.directResultsTitle.textContent = copy.title;
    elements.directResultsSummary.textContent = "结果到达后立即展示，最终列表会以运行记录中的标准化来源为准。";
    const existing = Array.from(elements.directSourceList.children)
      .find((item) => item.dataset.sourceId === source.source_id);
    const index = existing
      ? Array.from(elements.directSourceList.children).indexOf(existing)
      : elements.directSourceList.children.length;
    const rendered = renderDirectSource(source, index, currentPaperOrdinalStart);
    if (existing) existing.replaceWith(rendered);
    else elements.directSourceList.appendChild(rendered);
    updateDirectCount(elements.directSourceList.children.length);
    scrollConversation();
  }

  function renderDirectResults(sources, topic) {
    elements.directSourceList.replaceChildren();
    // 关键步骤：本轮最终结果从当前会话序号起点继续编号，而不是从 01 重新开始。
    sources.forEach((source, index) => elements.directSourceList.appendChild(
      renderDirectSource(source, index, currentPaperOrdinalStart)
    ));
    nextPaperOrdinal = Math.max(nextPaperOrdinal, currentPaperOrdinalStart + sources.length);
    elements.directResults.hidden = false;
    elements.directThinking.hidden = true;
    elements.directResultsTitle.textContent = directIntentCopy(currentIntent).title;
    // 关键步骤：回复正文已交代整理结果时隐藏汇总行，避免一句意思说两遍；
    // 无回复或空结果时用汇总行兜底提示。
    const hasReply = !elements.directAnswer.hidden;
    elements.directResultsSummary.hidden = hasReply && sources.length > 0;
    if (!(hasReply && sources.length > 0)) {
      elements.directResultsSummary.textContent = sources.length
        ? "已根据“" + (topic || "你的请求") + "”整理 " + sources.length + " 条可追溯学术来源。"
        : "没有找到符合当前条件的论文，可以尝试补充主题、论文 ID 或 DOI。";
    }
    updateDirectCount(sources.length);
    directNextOffset = sources.length;
    const args = currentDirectToolArgs;
    const canPaginate = currentIntent === "paper_recommendation" ||
      (currentIntent === "paper_graph_lookup" && (args.relation === "citations" || args.relation === "references"));
    updateDirectPagination(canPaginate && sources.length > 0, directNextOffset, null);
  }

  function isLegacyDirectAnswer(answer) {
    // 迁移保护：旧版 direct 把论文列表直接写进 answer（# 标题 + 编号 [标题](url) 行），
    // 与来源卡片重复且缺少助手发言，渲染时跳过，仅保留下方卡片。
    return /^\s*#\s+\S/.test(String(answer || ""))
      || /\n?\d+\.\s*\[[^\]]+\]\([^)]*\)/.test(String(answer || ""));
  }

  function renderDirectAnswer(answer, container) {
    // 关键步骤：direct 的 answer 是助手口吻的自然语言回复（开场 + 收尾），
    // 在对话里作为助手正文渲染；论文条目交给下方来源卡片按编号展示。
    // 若是历史存档遗留的"# 标题 + 编号列表"格式则跳过，避免两套样式并存。
    if (!container) return;
    const text = String(answer || "").trim();
    if (!text || isLegacyDirectAnswer(text)) {
      container.hidden = true;
      container.replaceChildren();
      return;
    }
    container.hidden = false;
    buildReport(text, false, container, { interactiveCitations: false });
  }

  function loadMoreDirectPapers() {
    if (!activeRunId || !directHasMore || directPageLoading) return;
    directPageLoading = true;
    elements.btnDirectMore.disabled = true;
    elements.btnDirectMore.textContent = "正在加载…";
    const requestOffset = directNextOffset;
    fetch(
      "/api/research/runs/" + encodeURIComponent(activeRunId) +
      "/papers?offset=" + requestOffset + "&limit=20"
    )
      .then(function (response) {
        if (!response.ok) return response.json().then(function (error) {
          throw new Error(error.detail || "HTTP " + response.status);
        });
        return response.json();
      })
      .then(function (page) {
        const existingKeys = new Set(Array.from(elements.directSourceList.children).map(function (item) {
          return item.dataset.sourceId;
        }).filter(Boolean));
        (page.items || []).forEach(function (source) {
          const sourceKey = source.source_id || "";
          if (sourceKey && existingKeys.has(sourceKey)) return;
          elements.directSourceList.appendChild(
            renderDirectSource(
              source,
              elements.directSourceList.children.length,
              currentPaperOrdinalStart
            )
          );
          if (sourceKey) existingKeys.add(sourceKey);
        });
        updateDirectCount(elements.directSourceList.children.length);
        nextPaperOrdinal = Math.max(
          nextPaperOrdinal,
          currentPaperOrdinalStart + elements.directSourceList.children.length
        );
        updateDirectPagination(page.has_more, page.next_offset, page.total);
        if (!page.has_more) {
          elements.directPagination.hidden = false;
          elements.btnDirectMore.hidden = true;
          elements.directPaginationStatus.textContent = "已加载当前数据源返回的全部结果";
        }
        scrollConversation();
      })
      .catch(function (error) {
        addError("Could not load more papers: " + error.message);
        elements.directPaginationStatus.textContent = "加载失败，请稍后重试";
      })
      .finally(function () {
        directPageLoading = false;
        elements.btnDirectMore.disabled = false;
        elements.btnDirectMore.textContent = "加载更多论文";
      });
  }

  function setText(el, value, fallback) {
    if (el) el.textContent = value === undefined || value === null || value === "" ? (fallback || "") : String(value);
  }

  function showSourceLimitNotice(requested, adjusted) {
    const isMaximum = requested > adjusted;
    elements.sourceLimitMessage.textContent = isMaximum
      ? "You requested " + requested + " sources. This run supports at most " + adjusted + ", so the value was adjusted automatically."
      : "A research run needs at least " + adjusted + " source. The value was adjusted automatically.";

    if (elements.sourceLimitDialog && typeof elements.sourceLimitDialog.showModal === "function") {
      if (!elements.sourceLimitDialog.open) elements.sourceLimitDialog.showModal();
      return;
    }
    window.alert(elements.sourceLimitMessage.textContent);
  }

  function showRuntimeError(message) {
    const detail = String(message || "未知运行错误");
    if (elements.runtimeErrorMessage) elements.runtimeErrorMessage.textContent = detail;
    if (elements.runtimeErrorDialog && typeof elements.runtimeErrorDialog.showModal === "function") {
      if (!elements.runtimeErrorDialog.open) elements.runtimeErrorDialog.showModal();
      return;
    }
    window.alert("调研执行失败：" + detail);
  }

  function normalizeMaxSources(notify) {
    const input = elements.maxSources;
    const minimum = Number.parseInt(input.min, 10) || 1;
    const maximum = Number.parseInt(input.max, 10) || 20;
    const fallback = 5;
    const requested = Number(input.value);
    const wholeNumber = Number.isFinite(requested) ? Math.trunc(requested) : fallback;
    const adjusted = Math.min(maximum, Math.max(minimum, wholeNumber));

    input.value = String(adjusted);
    if (notify && Number.isFinite(requested) && requested !== adjusted) {
      showSourceLimitNotice(requested, adjusted);
    }
    return adjusted;
  }

  function safeStatus(status) {
    const value = String(status || "idle").toLowerCase();
    return /^[a-z_]+$/.test(value) ? value : "idle";
  }

  function historyStatusLabel(status) {
    const labels = {
      completed: "done",
      completed_with_warnings: "warnings",
      partial: "needs evidence",
      failed: "failed",
      cancelled: "cancelled",
    };
    const value = safeStatus(status);
    return labels[value] || value;
  }

  function setStatus(status) {
    const value = safeStatus(status);
    currentRunStatus = value;
    elements.runStatus.className = "status-tag status-" + value;
    elements.runStatus.replaceChildren(node("i"), document.createTextNode(value.replace(/_/g, " ")));
  }

  function setConnection(connected) {
    elements.connection.className = "connection-badge " + (connected ? "badge-connected" : "badge-offline");
    const dot = node("i");
    dot.id = "sse-status";
    elements.connection.replaceChildren(dot, node("span", "", connected ? "Connected" : "Offline"));
    elements.workspaceState.textContent = connected ? "Live event stream connected" : "Awaiting a run";
  }

  function formatElapsed(milliseconds) {
    const raw = Number(milliseconds || 0);
    if (raw > 0 && raw < 1000) return "<1s";
    const seconds = Math.max(0, Math.floor(raw / 1000));
    if (seconds < 60) return seconds + "s";
    return Math.floor(seconds / 60) + "m " + String(seconds % 60).padStart(2, "0") + "s";
  }

  function formatHistoryDate(value) {
    if (!value) return "Unknown date";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value).slice(0, 16);
    return date.toLocaleString([], {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
    });
  }

  function renderHistory(items, append) {
    if (!append) elements.historyList.replaceChildren();
    items.forEach(function (item) {
      const button = node("button", "history-item");
      button.type = "button";
      const historyTitle = item.conversation_title || item.topic || "Untitled research";
      button.setAttribute("aria-label", "Open conversation " + String(historyTitle || item.run_id));
      const heading = node("span", "history-item-heading");
      heading.append(
        node("strong", "", historyTitle),
        node("i", "history-status status-" + safeStatus(item.status), historyStatusLabel(item.status))
      );
      heading.lastElementChild.title = String(item.status || "unknown");
      const detail = [
        formatHistoryDate(item.created_at),
        Number(item.turn_count || 1) > 1 ? String(item.turn_count) + " turns" : "",
        String(item.conversation_source_count ?? item.source_count ?? 0) + " sources",
        formatElapsed(item.total_latency_ms || 0),
      ].filter(Boolean).join(" · ");
      button.append(heading, node("span", "history-item-meta", detail));
      button.addEventListener("click", function () {
        openHistoryItem(item);
      });
      elements.historyList.appendChild(button);
    });
  }

  function loadRunHistory(reset) {
    if (reset) historyOffset = 0;
    return fetch("/api/runs?limit=" + historyPageSize + "&offset=" + historyOffset)
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        const items = Array.isArray(data.runs) ? data.runs : [];
        renderHistory(items, historyOffset > 0);
        historyOffset += items.length;
        elements.historyEmpty.hidden = historyOffset > 0;
        elements.btnHistoryMore.hidden = historyOffset >= Number(data.total || 0);
      })
      .catch(function () {
        elements.historyEmpty.textContent = "History is temporarily unavailable.";
        elements.historyEmpty.hidden = false;
        elements.btnHistoryMore.hidden = true;
      });
  }

  function updateElapsed() {
    if (!startTime) return;
    const seconds = Math.floor((Date.now() - startTime) / 1000);
    const formatted = String(Math.floor(seconds / 60)).padStart(2, "0") + ":" + String(seconds % 60).padStart(2, "0");
    elements.elapsed.textContent = formatted;
    elements.activityTime.textContent = formatted;
  }

  function startElapsedTimer() {
    stopElapsedTimer();
    startTime = Date.now();
    updateElapsed();
    elapsedInterval = window.setInterval(updateElapsed, 1000);
  }

  function stopElapsedTimer() {
    if (elapsedInterval) window.clearInterval(elapsedInterval);
    elapsedInterval = null;
  }

  function setActiveTab(name) {
    $$(".tab-button").forEach(function (button) {
      const active = button.dataset.tab === name;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    $$(".tab-panel").forEach(function (panel) {
      const active = panel.dataset.panel === name;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
  }

  const phaseCopy = {
    plan: ["正在制定研究计划", "拆解问题并安排检索任务"],
    search: ["正在检索学术来源", "从学术索引中查找相关研究"],
    read: ["正在阅读与筛选资料", "检查元数据、相关性与来源质量"],
    analyze: ["正在提炼证据与观点", "对照研究发现并组织核心论点"],
    cite: ["正在校验引用关系", "核对论点、证据与来源之间的对应"],
    evaluate: ["正在执行质量检查", "检查覆盖度、可信度和研究局限"],
    review: ["正在生成并复核报告", "整理结构、措辞与最终回答"]
  };

  function resetResearchProgress() {
    currentResearchPhase = "";
    elements.activityFeed.replaceChildren();
    elements.overviewRunning.classList.remove("is-failed");
    elements.overviewRunning.hidden = false;
    elements.deepReport.hidden = true;
    elements.activityTitle.textContent = "正在准备深度调研";
    elements.activityDetail.textContent = "正在理解问题并选择研究路径。";
    elements.activityTime.textContent = "00:00";
  }

  function researchStepDetail(phase, payload) {
    if (phase === "search" && sourceIds.size) return "已找到 " + sourceIds.size + " 个候选来源";
    if (phase === "analyze" && evidenceKeys.size) return "已提炼 " + evidenceKeys.size + " 条证据";
    if (payload && payload.source_count !== undefined) return "已整理 " + payload.source_count + " 个来源";
    if (payload && payload.card_count !== undefined) return "已生成 " + payload.card_count + " 条证据";
    return phaseCopy[phase] ? phaseCopy[phase][1] : "研究任务正在执行";
  }

  function updateResearchProgress(phase, payload, timeText) {
    const nextIndex = phaseOrder.indexOf(phase);
    const currentIndex = phaseOrder.indexOf(currentResearchPhase);
    if (nextIndex < 0 || (currentIndex >= 0 && nextIndex < currentIndex)) return;

    const copy = phaseCopy[phase];
    elements.activityTitle.textContent = copy[0];
    elements.activityDetail.textContent = researchStepDetail(phase, payload);

    let item = elements.activityFeed.querySelector('[data-research-phase="' + phase + '"]');
    if (!item) {
      Array.from(elements.activityFeed.children).forEach(function (child) {
        child.classList.remove("is-active");
        child.classList.add("is-complete");
        child.querySelector("i").textContent = "✓";
      });
      item = node("div", "research-step-item is-active");
      item.dataset.researchPhase = phase;
      item.append(node("i"), node("strong", "", copy[0]), node("time", "", timeText));
      elements.activityFeed.appendChild(item);
      while (elements.activityFeed.children.length > 4) elements.activityFeed.firstElementChild.remove();
      currentResearchPhase = phase;
    } else {
      item.querySelector("strong").textContent = copy[0];
      item.querySelector("time").textContent = timeText;
    }
    scrollConversation();
  }

  function finishResearchProgress(status) {
    const cancelled = status === "cancelled";
    const failed = status === "failed";
    const partial = status === "partial";
    const active = elements.activityFeed.querySelector(".is-active");
    if (active) {
      active.classList.remove("is-active");
      active.classList.add(failed || cancelled || partial ? "is-error" : "is-complete");
      active.querySelector("i").textContent = failed || partial ? "!" : cancelled ? "×" : "✓";
    }
    if (failed || cancelled || partial) {
      elements.overviewRunning.classList.add("is-failed");
      elements.activityTitle.textContent = cancelled ? "调研已中止" : partial ? "报告需要补充证据" : "调研未能完成";
      elements.activityDetail.textContent = cancelled
        ? "任务已按你的要求停止，可以继续输入新的调研问题。"
        : partial
          ? "结果已保留，但尚未通过方法、数据集、局限与引用闭环的最低验收标准。"
          : "运行已停止，请在 Trace 中查看具体错误后重试。";
    }
  }

  function showCancelledState() {
    if (executionDisplayMode === "direct") {
      elements.directThinking.hidden = true;
      showDirectProviderNotice("任务已中止", "本次调研已停止，可以继续输入新的调研问题。");
    } else {
      setExecutionDisplayMode("deep");
      finishResearchProgress("cancelled");
    }
    elements.workspaceSubtitle.textContent = "本次调研已中止。";
    scrollConversation();
  }

  function phaseForEvent(eventType, payload) {
    const hint = [eventType, payload.node, payload.task_type, payload.task_id, payload.tool_name]
      .filter(Boolean).join(" ").toLowerCase();
    if (eventType === "intent_classified" || eventType === "plan_created" || hint.includes("planner")) return "plan";
    if (hint.includes("search") || eventType === "source_found" || eventType === "send_dispatch") return "search";
    if (hint.includes("reading") || hint.includes("metadata") || hint.includes("quality_scorer")) return "read";
    if (hint.includes("analysis") || hint.includes("evidence") || eventType === "evidence_created") return "analyze";
    if (hint.includes("citation") || eventType === "citation_checked") return "cite";
    if (hint.includes("eval") || eventType === "eval_finished") return "evaluate";
    if (hint.includes("review") || hint.includes("draft") || hint.includes("chapter") ||
        eventType === "outline_created" || eventType === "chapter_generated" || eventType === "run_finished") return "review";
    return "";
  }

  function eventLabel(eventType) {
    const labels = {
      run_started: "Run started", intent_classified: "Intent classified",
      plan_created: "Plan created", send_dispatch: "Task dispatched",
      worker_started: "Worker started", worker_finished: "Worker finished",
      function_call_started: "LLM decision", tool_started: "Tool started", tool_finished: "Tool finished",
      tool_args_rejected: "Arguments rejected", tool_loop_fallback: "Fallback activated",
      tool_loop_finished: "Tool loop complete", tool_loop_limit_reached: "Budget reached",
      source_found: "Source found", evidence_created: "Evidence created",
      citation_checked: "Citations checked", eval_finished: "Evaluation complete",
      direct_reviewer_complete: "Direct answer ready",
      outline_created: "Report outline ready", chapter_generated: "Chapter generated",
      merge_result: "Results merged", run_finished: "Run finished", error: "Runtime error"
    };
    return labels[eventType] || eventType.replace(/_/g, " ");
  }

  function describeEvent(eventType, payload) {
    const subject = payload.tool_name || payload.task_id || payload.node || payload.source_id || "Agent runtime";
    let detail = subject;
    if (payload.source_count !== undefined) detail += " · " + payload.source_count + " sources";
    if (payload.card_count !== undefined) detail += " · " + payload.card_count + " evidence cards";
    if (payload.latency_ms !== undefined) detail += " · " + formatElapsed(payload.latency_ms);
    if (payload.message) detail += " · " + String(payload.message).slice(0, 120);
    if (payload.error) detail += " · " + String(payload.error).slice(0, 120);
    return { title: eventLabel(eventType), detail: detail };
  }

  function eventTone(eventType) {
    if (eventType === "error" || eventType.includes("rejected")) return "te-type-error";
    if (eventType.includes("fallback") || eventType.includes("limit")) return "te-type-warning";
    if (eventType.includes("finished") || eventType.includes("checked") || eventType === "evidence_created") return "te-type-finish";
    return "te-type-start";
  }

  function addTraceEvent(eventType, payload) {
    if (eventType === "heartbeat") return;
    payload = payload || {};
    elements.traceEmpty.hidden = true;
    traceEventCount += 1;
    elements.traceCount.textContent = String(traceEventCount);

    const now = new Date();
    const timeText = String(now.getHours()).padStart(2, "0") + ":" +
      String(now.getMinutes()).padStart(2, "0") + ":" + String(now.getSeconds()).padStart(2, "0");
    const copy = describeEvent(eventType, payload);
    const row = node("div", "trace-event");
    row.append(
      node("span", "te-time", timeText),
      node("span", "te-type " + eventTone(eventType), eventLabel(eventType)),
      node("span", "te-msg", copy.detail),
      node("i", "trace-dot")
    );

    const details = node("details", "trace-details");
    details.append(node("summary", "", "Payload"));
    details.append(node("pre", "", JSON.stringify(payload, null, 2)));
    row.append(details);
    elements.traceEvents.appendChild(row);
    elements.traceContainer.scrollTop = elements.traceContainer.scrollHeight;
    const phase = phaseForEvent(eventType, payload);
    if (phase) updateResearchProgress(phase, payload, timeText);
  }

  function updateDAG(taskDag) {
    if (!taskDag || !Array.isArray(taskDag.tasks)) return;
    elements.dagContainer.hidden = false;
    elements.dagDisplay.replaceChildren();
    taskDag.tasks.forEach(function (task) {
      const card = node("article", "dag-display-item");
      const meta = node("div", "dag-task-meta");
      meta.append(node("span", "", task.task_id || "task"), node("span", "", task.task_type || "worker"));
      card.append(meta, node("p", "", task.description || "No description"));
      elements.dagDisplay.appendChild(card);
    });
  }

  function addError(message) {
    if (!message) return;
    elements.errorsContainer.hidden = false;
    elements.errorsList.appendChild(node("li", "", String(message)));
  }

  function tableHeader(labels) {
    const thead = node("thead");
    const row = node("tr");
    labels.forEach((label) => row.appendChild(node("th", "", label)));
    thead.appendChild(row);
    return thead;
  }

  function buildReport(text, animate, container, options) {
    const target = container || elements.reportContent;
    const interactiveCitations = !options || options.interactiveCitations !== false;
    target.replaceChildren();
    const lines = String(text || "").split(/\r?\n/);
    const queue = [];
    let list = null;
    let listType = "";

    function closeList() { list = null; listType = ""; }
    function appendText(parent, content) {
      const textNode = document.createTextNode(animate ? "" : content);
      parent.appendChild(textNode);
      if (animate && content) queue.push({ node: textNode, graphemes: Array.from(content), index: 0 });
    }
    function appendInline(parent, content) {
      const pattern = /(\[[^\]]+\]\([^)]*\)|\*\*[^*]+\*\*|`[^`]+`|\[(\d+)\])/g;
      let cursor = 0;
      let match;
      while ((match = pattern.exec(content)) !== null) {
        if (match.index > cursor) appendText(parent, content.slice(cursor, match.index));
        const token = match[0];
        // ReAct 的 direct 答案里推荐条目是 [标题](url) 链接，直接渲染成可点击超链接。
        const link = token.match(/^\[([^\]]+)\]\(([^)]*)\)$/);
        if (link) {
          const linkNode = node("a", "report-link");
          linkNode.textContent = link[1];
          linkNode.href = link[2];
          linkNode.target = "_blank";
          linkNode.rel = "noopener noreferrer";
          parent.appendChild(linkNode);
          cursor = match.index + token.length;
          continue;
        }
        let inlineNode;
        if (match[2] && interactiveCitations) {
          const citationNumber = Number(match[2]);
          inlineNode = node("button", "report-citation");
          inlineNode.type = "button";
          inlineNode.title = "查看对应原文证据";
          // 引用编号必须与正文一起进入动画队列，不能在正文前抢先显示。
          appendText(inlineNode, token);
          inlineNode.addEventListener("click", function () {
            const target = citationEvidenceTargets.get(citationNumber);
            setActiveTab(target !== undefined ? "evidence" : "sources");
            elements.researchArtifacts.hidden = false;
            const targetElement = target !== undefined
              ? document.getElementById("evidence-card-" + target)
              : elements.sourcesSection;
            if (targetElement) {
              targetElement.scrollIntoView({ behavior: "smooth", block: "center" });
              targetElement.classList.add("is-citation-target");
              window.setTimeout(() => targetElement.classList.remove("is-citation-target"), 1600);
            }
          });
        } else {
          inlineNode = node(token.startsWith("**") ? "strong" : "code");
          if (match[2]) {
            inlineNode = node("span", "report-citation is-static");
            appendText(inlineNode, token);
          } else {
            appendText(inlineNode, token.startsWith("**") ? token.slice(2, -2) : token.slice(1, -1));
          }
        }
        parent.appendChild(inlineNode);
        cursor = match.index + token.length;
      }
      if (cursor < content.length) appendText(parent, content.slice(cursor));
    }
    function tableCells(line) {
      return line.replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
    }
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex].trim();
      if (!line) { closeList(); continue; }
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(3, heading[1].length);
        const headingNode = node("h" + level);
        appendInline(headingNode, heading[2]);
        target.appendChild(headingNode);
        continue;
      }
      if (/^[-*_]{3,}$/.test(line)) {
        closeList();
        const divider = node("hr");
        target.appendChild(divider);
        if (animate) {
          divider.hidden = true;
          queue.push({ structural: true, revealNodes: [divider] });
        }
        continue;
      }
      const nextLine = lines[lineIndex + 1] ? lines[lineIndex + 1].trim() : "";
      if (line.includes("|") && /^\|?\s*:?-{3,}.*\|\s*$/.test(nextLine)) {
        closeList();
        const wrap = node("div", "report-table-wrap");
        const table = node("table", "report-table");
        const head = node("thead");
        const headRow = node("tr");
        tableCells(line).forEach(function (cell) {
          const th = node("th");
          appendInline(th, cell);
          headRow.appendChild(th);
        });
        head.appendChild(headRow);
        table.appendChild(head);
        const body = node("tbody");
        lineIndex += 2;
        while (lineIndex < lines.length && lines[lineIndex].trim().includes("|")) {
          const row = node("tr");
          tableCells(lines[lineIndex].trim()).forEach(function (cell) {
            const td = node("td");
            appendInline(td, cell);
            row.appendChild(td);
          });
          body.appendChild(row);
          lineIndex += 1;
        }
        lineIndex -= 1;
        table.appendChild(body);
        wrap.appendChild(table);
        target.appendChild(wrap);
        continue;
      }
      const bullet = line.match(/^[-*]\s+(.+)$/);
      if (bullet) {
        if (!list || listType !== "ul") {
          list = node("ul");
          listType = "ul";
          target.appendChild(list);
        }
        const item = node("li");
        appendInline(item, bullet[1]);
        list.appendChild(item);
        continue;
      }
      const ordered = line.match(/^\d+[.)]\s+(.+)$/);
      if (ordered) {
        if (!list || listType !== "ol") {
          list = node("ol");
          listType = "ol";
          target.appendChild(list);
        }
        const item = node("li");
        appendInline(item, ordered[1]);
        list.appendChild(item);
        continue;
      }
      const quote = line.match(/^>\s?(.+)$/);
      if (quote) {
        closeList();
        const blockquote = node("blockquote");
        appendInline(blockquote, quote[1]);
        target.appendChild(blockquote);
        continue;
      }
      closeList();
      const paragraph = node("p");
      appendInline(paragraph, line);
      target.appendChild(paragraph);
    }
    if (animate) {
      const blockSelector = "h1, h2, h3, p, li, blockquote, tr";
      const containerSelector = "ul, ol, .report-table-wrap";
      // 关键步骤：先隐藏尚未开始输出的 Markdown 块，避免列表圆点和表格骨架抢跑。
      target.querySelectorAll(blockSelector + ", " + containerSelector).forEach(function (element) {
        element.hidden = true;
      });
      queue.forEach(function (segment) {
        if (segment.structural || !segment.node || !segment.node.parentElement) return;
        const block = segment.node.parentElement.closest(blockSelector);
        const containerNode = block ? block.closest(containerSelector) : null;
        segment.revealNodes = [containerNode, block].filter(Boolean);
      });
    }
    return queue;
  }

  function finishReportTyping() {
    if (reportTypingTimer) window.clearTimeout(reportTypingTimer);
    reportTypingTimer = null;
    pendingReportText = "";
    elements.reportContent.classList.remove("is-typing");
    elements.btnSkipTyping.hidden = true;
    elements.reportStateLabel.textContent = currentRunStatus === "partial" ? "报告部分完成" : "报告已完成";
    elements.researchArtifacts.hidden = false;
    if (reportTypingComplete) {
      const complete = reportTypingComplete;
      reportTypingComplete = null;
      complete();
    }
    scrollConversation();
  }

  function stopReportTyping(revealAll) {
    if (reportTypingTimer) window.clearTimeout(reportTypingTimer);
    reportTypingTimer = null;
    if (revealAll && pendingReportText) {
      const report = pendingReportText;
      pendingReportText = "";
      buildReport(report, false);
      finishReportTyping();
      return;
    }
    pendingReportText = "";
    reportTypingComplete = null;
    elements.reportContent.classList.remove("is-typing");
    elements.btnSkipTyping.hidden = true;
  }

  function renderReport(text, animate, onComplete) {
    stopReportTyping(false);
    const shouldAnimate = Boolean(animate) &&
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const queue = buildReport(text, shouldAnimate);
    reportTypingComplete = typeof onComplete === "function" ? onComplete : null;

    if (!shouldAnimate || queue.length === 0) {
      finishReportTyping();
      return;
    }

    pendingReportText = String(text || "");
    elements.reportContent.classList.add("is-typing");
    elements.reportStateLabel.textContent = "正在生成回答";
    elements.btnSkipTyping.hidden = false;
    let segmentIndex = 0;
    let characterCount = 0;

    function typeNext() {
      const segment = queue[segmentIndex];
      if (!segment) {
        finishReportTyping();
        return;
      }
      (segment.revealNodes || []).forEach(function (element) {
        element.hidden = false;
      });
      if (segment.structural) {
        segmentIndex += 1;
        reportTypingTimer = window.setTimeout(typeNext, 7);
        return;
      }
      const character = segment.graphemes[segment.index];
      segment.node.data += character;
      segment.index += 1;
      characterCount += 1;
      if (segment.index >= segment.graphemes.length) segmentIndex += 1;
      if (characterCount % 24 === 0) scrollConversation();
      const punctuationPause = /[。！？；：.!?]/.test(character) ? 34 : 0;
      reportTypingTimer = window.setTimeout(typeNext, 7 + punctuationPause);
    }
    typeNext();
  }

  function renderSources(sources, analysisSelection) {
    elements.sourcesTable.replaceChildren();
    const reasons = analysisSelection && typeof analysisSelection.selection_reasons === "object"
      ? analysisSelection.selection_reasons : {};
    const showReason = Object.keys(reasons).length > 0;
    const headers = ["Source", "Provider", "Year", "Citations", "Quality"];
    if (showReason) headers.push("Why analyzed");
    elements.sourcesTable.appendChild(tableHeader(headers));
    const tbody = node("tbody");
    sources.forEach(function (source) {
      const row = node("tr");
      const titleCell = node("td", "source-title");
      if (/^https?:\/\//i.test(source.url || "")) {
        const link = node("a", "source-link", source.title || "Untitled source");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        titleCell.appendChild(link);
      } else {
        titleCell.textContent = source.title || "Untitled source";
      }
      row.append(
        titleCell,
        node("td", "", source.provider || source.source_type || "unknown"),
        node("td", "", source.year || "—"),
        node("td", "", source.cited_by_count !== undefined && source.cited_by_count !== null ? source.cited_by_count : "—"),
        node("td", "", source.quality_score !== undefined && source.quality_score !== null ? Number(source.quality_score).toFixed(2) : "—")
      );
      if (showReason) {
        row.appendChild(node("td", "selection-reason", reasons[String(source.source_id || "")] || "Coverage backfill"));
      }
      tbody.appendChild(row);
    });
    elements.sourcesTable.appendChild(tbody);
  }

  function renderEvidence(cards) {
    elements.evidenceList.replaceChildren();
    cards.forEach(function (card, index) {
      const article = node("article", "evidence-card");
      article.id = "evidence-card-" + index;
      article.appendChild(node("span", "ec-index", String(index + 1).padStart(2, "0")));
      article.appendChild(node("h4", "ec-claim", card.claim || "Untitled claim"));
      if (card.quote) article.appendChild(node("blockquote", "ec-quote", "“" + card.quote + "”"));
      const meta = node("div", "ec-meta");
      meta.append(
        node("span", "", "EVIDENCE " + String(card.evidence_id || "—")),
        node("span", "", "SOURCE " + String(card.source_id || "").slice(0, 8)),
        node("span", "", "TYPE " + String(card.evidence_type || "primary_claim")),
        node("span", "", "LOCATION " + String(card.page_or_section || card.quote_location || "—")),
        node("span", "", "CONFIDENCE " + (card.confidence !== undefined ? Number(card.confidence).toFixed(2) : "—"))
      );
      article.appendChild(meta);
      elements.evidenceList.appendChild(article);
    });
  }

  function renderCitations(results) {
    elements.citationTable.replaceChildren();
    elements.citationTable.appendChild(tableHeader(["Citation", "Source", "Status", "Issues"]));
    const tbody = node("tbody");
    results.forEach(function (result) {
      const row = node("tr");
      const statusCell = node("td");
      statusCell.appendChild(node("span", "table-status " + (result.is_valid ? "pass" : "fail"), result.is_valid ? "Verified" : "Failed"));
      row.append(
        node("td", "", result.citation_id !== undefined ? "[" + result.citation_id + "]" : "—"),
        node("td", "", String(result.source_id || "").slice(0, 8)),
        statusCell,
        node("td", "", (result.issues || []).join("; ") || "None")
      );
      tbody.appendChild(row);
    });
    elements.citationTable.appendChild(tbody);
  }

  function metricPass(value) {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value >= 0.5;
    return String(value).toLowerCase() === "pass";
  }

  function renderEval(metrics) {
    elements.evalDisplay.replaceChildren();
    Object.keys(metrics).forEach(function (key) {
      const pass = metricPass(metrics[key]);
      const row = node("div", "eval-metric " + (pass ? "pass" : "fail"));
      row.append(
        node("span", "eval-icon", pass ? "✓" : "!"),
        node("span", "eval-name", key.replace(/_/g, " ")),
        node("span", "eval-value", String(metrics[key]))
      );
      elements.evalDisplay.appendChild(row);
    });
  }

  function formatMetricValue(value, suffix) {
    if (value === undefined || value === null || value === "") return "Unavailable";
    return String(value) + (suffix || "");
  }

  function renderObservability(metrics) {
    elements.observabilityDisplay.replaceChildren();
    const run = metrics.run || {};
    const nodes = metrics.nodes || {};
    const workers = metrics.workers || {};
    const tools = metrics.tools || {};
    const llm = metrics.llm || {};
    const cards = [
      ["Run latency", formatElapsed(run.latency_ms || 0), "End-to-end"],
      ["Nodes", formatMetricValue(nodes.execution_count), formatMetricValue(nodes.error_count, " errors")],
      ["Workers", formatMetricValue(workers.execution_count), formatMetricValue(workers.partial_failure_count, " partial failures")],
      ["Tool calls", formatMetricValue(tools.call_count), formatMetricValue(tools.error_count, " errors")],
      ["LLM calls", formatMetricValue(llm.call_count), formatMetricValue(llm.total_tokens, " tokens")],
      ["Retries", formatMetricValue(run.retry_count), formatMetricValue(run.fallback_count, " fallbacks")],
    ];

    cards.forEach(function (item) {
      const card = node("div", "observability-card");
      card.append(
        node("span", "observability-label", item[0]),
        node("strong", "observability-value", item[1]),
        node("span", "observability-detail", item[2])
      );
      elements.observabilityDisplay.appendChild(card);
    });

    const slowest = node("div", "observability-slowest");
    const slowestNode = nodes.slowest;
    const slowestTool = tools.slowest;
    slowest.append(
      node("span", "observability-label", "Slowest operations"),
      node("span", "observability-detail",
        "Node: " + (slowestNode ? slowestNode.name + " · " + slowestNode.latency_ms + "ms" : "unavailable") +
        "  |  Tool: " + (slowestTool ? slowestTool.name + " · " + slowestTool.latency_ms + "ms" : "unavailable"))
    );
    elements.observabilityDisplay.appendChild(slowest);
  }

  function clearResults() {
    stopReportTyping(false);
    elements.resultsEmpty.hidden = false;
    elements.reportSection.hidden = true;
    elements.deepReport.hidden = true;
    elements.sourcesSection.hidden = true;
    elements.sourcesEmpty.hidden = false;
    elements.evidenceSection.hidden = true;
    elements.evidenceEmpty.hidden = false;
    elements.citationSection.hidden = true;
    elements.citationEmpty.hidden = false;
    elements.evalSection.hidden = true;
    elements.evalEmpty.hidden = false;
    elements.observabilitySection.hidden = true;
    elements.observabilityEmpty.hidden = false;
    elements.reportContent.replaceChildren();
    elements.sourcesTable.replaceChildren();
    elements.evidenceList.replaceChildren();
    elements.citationTable.replaceChildren();
    elements.evalDisplay.replaceChildren();
    elements.observabilityDisplay.replaceChildren();
    elements.outlineList.replaceChildren();
    elements.outlineSection.hidden = true;
    elements.outlineEmpty.hidden = false;
    elements.metricSources.textContent = "—";
    elements.metricEvidence.textContent = "—";
    elements.metricCitations.textContent = "—";
    elements.metricRuntime.textContent = "—";
  }

  function showResults(data, animateReport) {
    const report = data.answer || data.final_report || data.draft_report || "";
    const sources = Array.isArray(data.sources) ? data.sources : [];
    const evidence = Array.isArray(data.evidence_cards) ? data.evidence_cards : [];
    const citations = Array.isArray(data.citation_check_results) ? data.citation_check_results : [];
    const metrics = data.eval_metrics && typeof data.eval_metrics === "object" ? data.eval_metrics : {};
    const observability = data.observability_metrics && typeof data.observability_metrics === "object"
      ? data.observability_metrics : {};
    if (data.report_completion_ready !== undefined) {
      metrics.report_completion_ready = Boolean(data.report_completion_ready);
    }
    if (Array.isArray(data.report_completion_issues)) {
      data.report_completion_issues.forEach(function (issue) { addError("验收未通过：" + issue); });
    }
    currentRunStatus = safeStatus(data.status || "completed");
    citationEvidenceTargets = new Map();
    const sourceNumbers = new Map(sources.map((source, index) => [String(source.source_id || ""), index + 1]));
    evidence.forEach(function (card, index) {
      const number = sourceNumbers.get(String(card.source_id || ""));
      if (number !== undefined && !citationEvidenceTargets.has(number)) citationEvidenceTargets.set(number, index);
    });
    currentIntent = data.intent || currentIntent;
    if (data.execution_route === "direct_tool") {
      setExecutionDisplayMode("direct");
      currentDirectToolArgs = data.selected_tool_args && typeof data.selected_tool_args === "object"
        ? data.selected_tool_args : {};
      renderDirectAnswer(data.answer, elements.directAnswer);
      renderDirectResults(sources, data.topic);
      const providers = new Set(sources.map((source) => source.provider).filter(Boolean));
      const fallbackWarning = (data.warnings || []).find((warning) =>
        String(warning).toLowerCase().includes("fallback")
      );
      if (fallbackWarning) {
        showDirectProviderNotice(
          "主数据源暂时不可用",
          "本次已降级到 " + (Array.from(providers).join(", ") || "备用数据源") +
            "。详情：" + fallbackWarning
        );
      } else {
        elements.directProviderNotice.hidden = true;
      }
    } else {
      setExecutionDisplayMode("deep");
      finishResearchProgress(data.status || "completed");
    }

    elements.resultsEmpty.hidden = Boolean(report);
    elements.reportSection.hidden = !report;
    if (data.execution_route !== "direct_tool") {
      if (report) {
        elements.overviewRunning.hidden = true;
        elements.deepReport.hidden = false;
        renderReport(report, animateReport);
      } else {
        elements.deepReport.hidden = true;
        elements.overviewRunning.hidden = false;
      }
    }

    elements.sourcesSection.hidden = sources.length === 0;
    elements.sourcesEmpty.hidden = sources.length > 0;
    if (sources.length) renderSources(sources, data.analysis_selection || {});

    elements.evidenceSection.hidden = evidence.length === 0;
    elements.evidenceEmpty.hidden = evidence.length > 0;
    if (evidence.length) renderEvidence(evidence);

    elements.citationSection.hidden = citations.length === 0;
    elements.citationEmpty.hidden = citations.length > 0;
    if (citations.length) renderCitations(citations);

    elements.evalSection.hidden = Object.keys(metrics).length === 0;
    elements.evalEmpty.hidden = Object.keys(metrics).length > 0;
    if (Object.keys(metrics).length) renderEval(metrics);

    elements.observabilitySection.hidden = Object.keys(observability).length === 0;
    elements.observabilityEmpty.hidden = Object.keys(observability).length > 0;
    if (Object.keys(observability).length) renderObservability(observability);
    renderOutline(data.outline || {});

    const validCitations = citations.filter((item) => item.is_valid).length;
    elements.metricSources.textContent = String(sources.length);
    elements.metricEvidence.textContent = String(evidence.length);
    elements.metricCitations.textContent = citations.length ? Math.round(validCitations / citations.length * 100) + "%" : "—";
    const observedLatency = observability.run && observability.run.latency_ms;
    elements.metricRuntime.textContent = observedLatency || data.total_latency_ms
      ? formatElapsed(observedLatency || data.total_latency_ms) : elements.elapsed.textContent;
    elements.artifactSummary.textContent = sources.length + " 个来源 · " + evidence.length + " 条证据";
    const discoveredCount = Number.isFinite(Number(data.discovered_source_count))
      ? Number(data.discovered_source_count) : sources.length;
    const analyzedCount = Number.isFinite(Number(data.analyzed_source_count))
      ? Number(data.analyzed_source_count) : sources.length;
    elements.discoveredSourceCount.textContent = String(discoveredCount);
    elements.analyzedSourceCount.textContent = String(analyzedCount);
    setStatus(data.status || "completed");
    elements.overviewIdle.hidden = true;
    elements.userMessage.hidden = false;
    elements.assistantMessage.hidden = false;
    if (data.status) elements.workspaceState.textContent = "Saved run · " + String(data.status).replace(/_/g, " ");
    if (data.topic) {
      elements.userMessageText.textContent = data.topic;
      elements.workspaceTitle.textContent = data.topic;
      elements.workspaceSubtitle.textContent = data.execution_route === "direct_tool"
        ? "已通过所选学术能力完成请求。"
        : data.status === "failed"
          ? "调研未完成，可展开 Trace 查看错误。"
          : data.status === "partial"
            ? "报告尚未通过最低验收标准，可在质量页查看缺失项。"
            : "报告已完成，来源与证据可在下方展开查看。";
    }
    elements.routeBadge.textContent = routeLabel(
      data.intent, data.execution_route, data.selected_tools
    );
    if (data.session_id) {
      activeSessionId = data.session_id;
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
    }
    lastResultData = data;
    workspaceCache.library = null;
    workspaceCache.evidence = null;
    workspaceCache.reports = null;
    elements.sessionTopic.textContent = data.research_topic || data.topic || "当前研究";
    refreshSession();
    if (traceEventCount === 0 && Array.isArray(data.trace)) {
      data.trace.forEach(function (event) {
        if (event && event.event !== "heartbeat") addTraceEvent(event.event || "trace", event);
      });
    }
    scrollConversation();
  }

  function handleSSE(eventType, event) {
    try {
      const payload = JSON.parse(event.data || "{}");
      addTraceEvent(eventType, payload);
      if (eventType === "intent_classified") {
        currentIntent = payload.intent || "";
        if (payload.execution_route === "direct_tool") {
          const copy = directIntentCopy(currentIntent);
          setExecutionDisplayMode("direct");
          setDirectThinking(copy.thinking, copy.detail);
        } else {
          setExecutionDisplayMode("deep");
          if (["conversation", "paper_qa", "paper_compare", "report_follow_up"].includes(payload.execution_route)) {
            const strictFollowUp = ["paper_qa", "paper_compare", "report_follow_up"].includes(
              payload.conversation_operation || payload.intent
            );
            elements.activityTitle.textContent = strictFollowUp
              ? "正在处理会话追问"
              : "正在基于当前会话上下文作答";
            elements.activityDetail.textContent = "正在统一读取当前 Session 中的论文、证据、报告与对话历史。";
          }
        }
        elements.routeBadge.textContent = routeLabel(
          payload.intent, payload.execution_route, payload.selected_tools
        );
        elements.workspaceSubtitle.textContent = payload.execution_route === "direct_tool"
          ? "Controller selected a bounded research capability."
          : "已进入深度调研，将持续展示当前步骤。";
      }
      if (eventType === "plan_created") updateDAG(payload.task_dag || payload);
      if (eventType === "source_found" && payload.source_id) {
        sourceIds.add(payload.source_id);
        elements.metricSources.textContent = String(sourceIds.size);
        if (executionDisplayMode === "direct") {
          upsertDirectSource(payload);
          const copy = directIntentCopy(currentIntent);
          setDirectThinking(
            "已找到 " + sourceIds.size + " 篇，正在继续整理",
            copy.detail
          );
        } else if (executionDisplayMode === "deep") {
          elements.activityDetail.textContent = "已找到 " + sourceIds.size + " 个候选来源，正在继续检索。";
        }
      }
      if (eventType === "merge_result" && payload.merge_type === "search") {
        const discovered = Number(payload.unique_count);
        const analyzed = Number(payload.capped_count);
        if (Number.isFinite(discovered)) elements.discoveredSourceCount.textContent = String(discovered);
        if (Number.isFinite(analyzed)) elements.analyzedSourceCount.textContent = String(analyzed);
      }
      if (eventType === "merge_result" && payload.merge_type === "reading") {
        const analyzed = Number(payload.scored_count);
        if (Number.isFinite(analyzed)) elements.analyzedSourceCount.textContent = String(analyzed);
      }
      if (eventType === "worker_finished" && payload.event === "search_complete") {
        const discovered = Number(payload.discovered_count || payload.source_count);
        if (Number.isFinite(discovered)) elements.discoveredSourceCount.textContent = String(discovered);
      }
      if (eventType === "worker_finished" && payload.event === "reading_complete") {
        const analyzed = Number(payload.scored_count);
        if (Number.isFinite(analyzed)) elements.analyzedSourceCount.textContent = String(analyzed);
      }
      if (executionDisplayMode === "direct" && eventType === "tool_started") {
        const copy = directIntentCopy(currentIntent);
        setDirectThinking(copy.thinking, "学术工具正在查询数据源。");
      }
      if (executionDisplayMode === "direct" && eventType === "tool_loop_fallback") {
        showDirectProviderNotice(
          "主数据源暂时不可用",
          "正在从 " + (payload.from_tool || "首选工具") +
            " 切换到 " + (payload.to_tool || "备用工具") + "。"
        );
      }
      if (executionDisplayMode === "direct" && eventType === "tool_finished" &&
          payload.success === false) {
        showDirectProviderNotice(
          "Semantic Scholar 请求未完成",
          payload.error || "请求失败，系统将尝试备用学术数据源。"
        );
      }
      if (executionDisplayMode === "direct" &&
          ["tool_finished", "worker_finished", "direct_reviewer_complete"].includes(eventType)) {
        setDirectThinking("正在整理回答", "正在校对论文元数据并生成最终结果。");
      }
      if (eventType === "evidence_created") {
        const key = String(payload.source_id || "") + ":" + String(payload.claim || "");
        evidenceKeys.add(key);
        elements.metricEvidence.textContent = String(evidenceKeys.size);
        if (executionDisplayMode === "deep") {
          elements.activityDetail.textContent = "已提炼 " + evidenceKeys.size + " 条证据，正在交叉核对。";
        }
      }
      if (eventType === "chapter_generated") {
        completedChapterHeadings.add(String(payload.heading || payload.task_id || completedChapterHeadings.size + 1));
        if (executionDisplayMode === "deep") {
          elements.activityDetail.textContent = "已并行完成 " + completedChapterHeadings.size + " 个报告章节。";
        }
      }
      if (eventType === "error") {
        addError(payload.error || payload.message || "Unknown runtime error");
        if (executionDisplayMode === "deep") {
          elements.activityDetail.textContent = "某个研究任务遇到问题，系统正在尝试继续或安全结束。";
        }
      }
      return payload;
    } catch (error) {
      addError("Could not parse " + eventType + " event");
      return {};
    }
  }

  function connectSSE(runId) {
    disconnectSSE();
    setConnection(false);
    const stream = new EventSource("/api/research/stream/" + encodeURIComponent(runId));
    activeEventSource = stream;
    stream.onopen = function () { setConnection(true); };

    ["run_started", "intent_classified", "plan_created", "send_dispatch", "worker_started", "worker_finished",
      "function_call_started", "tool_started", "tool_finished", "tool_args_rejected",
      "tool_loop_fallback", "tool_loop_finished", "tool_loop_limit_reached", "citation_checked",
      "eval_finished", "draft_reviewer_complete", "final_reviewer_complete", "direct_reviewer_complete",
      "source_found", "evidence_created", "outline_created", "chapter_generated", "merge_result"].forEach(function (type) {
      stream.addEventListener(type, (event) => handleSSE(type, event));
    });

    stream.addEventListener("error", function (event) {
      if (event.data) {
        const payload = handleSSE("error", event);
        // Internal llm_failed trace events also use the SSE `error` name and
        // may be recoverable.  API/runtime terminal errors are distinguished by
        // error_type and must settle the UI because no run_finished follows.
        if (payload.error_type && payload.error_type !== "CancelledError") {
          const message = payload.message || payload.error || "Unknown runtime error";
          stopElapsedTimer();
          finishResearchProgress("failed");
          setStatus("failed");
          setConnection(false);
          disconnectSSE();
          elements.workspaceState.textContent = "Run failed";
          elements.workspaceSubtitle.textContent = "本次调研执行失败，可查看错误后重试。";
          showRuntimeError(message);
          fetchRunResult(runId, false, false).then(function () { loadRunHistory(true); });
          resetForm();
          return;
        }
      }
      setConnection(false);
    });

    stream.addEventListener("run_finished", function (event) {
      const payload = handleSSE("run_finished", event);
      stopElapsedTimer();
      finishResearchProgress(payload.status || "completed");
      setStatus(payload.status || "completed");
      setConnection(false);
      disconnectSSE();
      elements.workspaceState.textContent = "Run complete";
      fetchRunResult(runId, true, true).then(function () { loadRunHistory(true); });
      resetForm();
    });
  }

  function disconnectSSE() {
    if (activeEventSource) activeEventSource.close();
    activeEventSource = null;
  }

  function fetchRunResult(runId, openReport, animateReport) {
    return fetch("/api/runs/" + encodeURIComponent(runId))
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        if (data.error) throw new Error(data.error);
        showResults(data, Boolean(openReport && animateReport));
        return data;
      })
      .catch(function (error) { addError("Could not load run: " + error.message); });
  }

  function prepareHistoryView(runId) {
    setWorkspace("research");
    nextPaperOrdinal = 0;
    currentPaperOrdinalStart = 0;
    resetRunView();
    elements.conversationThread.replaceChildren();
    followLatest = true;
    activeRunId = runId;
    elements.currentRunId.textContent = runId;
    elements.overviewIdle.hidden = true;
    elements.assistantMessage.hidden = false;
    elements.overviewRunning.hidden = false;
    elements.workspaceTitle.textContent = "Loading research conversation";
    elements.workspaceSubtitle.textContent = "正在读取已保存的多轮对话与研究材料。";
    closeSidebar();
  }

  function openHistoryItem(item) {
    const runId = String(item.run_id || "");
    const sessionId = String(item.session_id || "");
    if (!runId) return;
    prepareHistoryView(runId);
    if (!sessionId) {
      fetchRunResult(runId, true, false);
      return;
    }
    fetch("/api/conversations/" + encodeURIComponent(sessionId) + "/runs")
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        const runs = Array.isArray(data.runs) ? data.runs : [];
        if (!runs.length) throw new Error("Conversation has no saved turns");
        // 关键步骤：前 N-1 轮恢复到历史区，最后一轮继续使用当前回答布局。
        let ordinalOffset = 0;
        const seenSourceKeys = new Set();
        const restoredRuns = runs.map(function (run) {
          if (run.execution_route !== "direct_tool") return run;
          return deduplicateHistorySources(run, seenSourceKeys);
        });
        restoredRuns.slice(0, -1).forEach(function (run) {
          const range = recommendationRange(run, ordinalOffset);
          appendArchivedExchange(run, range.startOffset);
          if (run.execution_route === "direct_tool") {
            ordinalOffset = Math.max(ordinalOffset, range.endOffset);
          }
        });
        const latest = restoredRuns[restoredRuns.length - 1];
        const latestRange = recommendationRange(latest, ordinalOffset);
        currentPaperOrdinalStart = latest.execution_route === "direct_tool"
          ? latestRange.startOffset : ordinalOffset;
        nextPaperOrdinal = Math.max(ordinalOffset, latestRange.endOffset);
        activeRunId = latest.run_id || runId;
        // 关键步骤：历史 Session 失效时使用后端返回的新恢复 Session，后续追问继续复用重建后的论文上下文。
        activeSessionId = data.session_id || sessionId;
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
        if (data.session) updateSessionDisplay(data.session);
        elements.currentRunId.textContent = activeRunId;
        showResults(latest, false);
        scrollConversation(true);
      })
      .catch(function (error) {
        addError("Could not load conversation: " + error.message);
        // 会话聚合读取失败时仍允许查看最新一轮，避免整个历史入口不可用。
        fetchRunResult(runId, true, false);
      });
  }

  function resetRunView() {
    disconnectSSE();
    stopElapsedTimer();
    traceEventCount = 0;
    sourceIds = new Set();
    evidenceKeys = new Set();
    completedChapterHeadings = new Set();
    elements.discoveredSourceCount.textContent = "0";
    elements.analyzedSourceCount.textContent = "0";
    elements.traceCount.textContent = "0";
    elements.traceEvents.replaceChildren();
    elements.traceEmpty.hidden = false;
    elements.dagDisplay.replaceChildren();
    elements.dagContainer.hidden = true;
    elements.errorsList.replaceChildren();
    elements.errorsContainer.hidden = true;
    resetResearchProgress();
    clearResults();
    elements.routeBadge.textContent = "Routing";
    resetDirectResponse();
    setActiveTab("sources");
  }

  // 是否处于移动端抽屉布局(与 CSS 的 860px 断点保持一致)
  function isMobileLayout() {
    return window.matchMedia("(max-width: 860px)").matches;
  }

  function closeSidebar() {
    elements.sidebar.classList.remove("is-open");
    elements.backdrop.hidden = true;
    elements.btnConfig.setAttribute("aria-expanded", "false");
  }

  function openSidebar() {
    elements.sidebar.classList.add("is-open");
    elements.backdrop.hidden = false;
    elements.btnConfig.setAttribute("aria-expanded", "true");
  }

  // 桌面端:折叠/展开左侧栏(整栏收起,主区占满;与移动端抽屉互不影响)
  function toggleDesktopSidebar() {
    const collapsed = elements.shell.classList.toggle("sidebar-collapsed");
    elements.btnConfig.setAttribute("aria-expanded", String(!collapsed));
  }

  function startResearch() {
    if (submitting) return;
    const topic = elements.topic.value.trim();
    if (!topic) { addError("Research question is required"); elements.topic.focus(); return; }
    const maxSources = normalizeMaxSources(true);
    if (isAmbiguousResearchRequest(topic) && sessionTurnCount === 0) {
      showClarificationRequest(topic);
      elements.topic.value = "";
      resizeComposer();
      elements.topic.focus();
      return;
    }
    submitting = true;
    followLatest = true;
    archiveCurrentExchange();
    resetRunView();
    // 归档上一轮后，以会话累计数量作为本轮论文卡片的序号起点。
    currentPaperOrdinalStart = nextPaperOrdinal;
    elements.overviewIdle.hidden = true;
    elements.userMessage.hidden = false;
    elements.userMessageText.textContent = topic;
    elements.assistantMessage.hidden = false;
    resetDirectResponse();
    resetResearchProgress();
    elements.overviewRunning.hidden = false;
    elements.workspaceTitle.textContent = topic;
    elements.workspaceSubtitle.textContent = "调研完成后将在这里生成完整报告。";
    elements.activityTitle.textContent = "正在准备深度调研";
    elements.activityDetail.textContent = "正在创建研究任务并初始化运行环境。";
    setActiveTab("sources");
    setStatus("queued");
    elements.btnStart.disabled = true;
    elements.btnCancel.disabled = false;
    elements.topic.value = "";
    resizeComposer();
    closeSidebar();
    scrollConversation();

    const body = {
      topic: topic,
      max_sources: maxSources,
      language: elements.language.value,
      mode: "quick",
      run_eval: elements.runEval.checked,
      agent_mode: elements.agentMode.value,
      session_id: activeSessionId
    };

    ensureSession().then(function () {
      body.session_id = activeSessionId;
      return fetch("/api/research/runs?backend=" + encodeURIComponent(elements.backend.value), {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
      });
    })
      .then(function (response) {
        if (!response.ok) return response.json().then((error) => { throw new Error(error.detail || "HTTP " + response.status); });
        return response.json();
      })
      .then(function (data) {
        activeRunId = data.run_id;
        if (data.session_id) {
          activeSessionId = data.session_id;
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, activeSessionId);
          elements.sessionId.textContent = activeSessionId;
        }
        elements.currentRunId.textContent = data.run_id;
        setStatus("running");
        startElapsedTimer();
        connectSSE(data.run_id);
      })
      .catch(function (error) {
        addError("Could not start research: " + error.message);
        setStatus("failed");
        setExecutionDisplayMode("deep");
        finishResearchProgress("failed");
        resetForm();
      });
  }

  function cancelResearch() {
    if (!activeRunId) return;
    elements.btnCancel.disabled = true;
    fetch("/api/research/runs/" + encodeURIComponent(activeRunId) + "/cancel", { method: "POST" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        disconnectSSE();
        stopElapsedTimer();
        setConnection(false);
        setStatus(data.status || "cancelled");
        elements.workspaceState.textContent = "Run cancelled";
        addTraceEvent("error", { message: "Run cancelled by user" });
        showCancelledState();
        activeRunId = null;
        resetForm();
      })
      .catch(function (error) {
        addError("Could not cancel run: " + error.message);
        elements.btnCancel.disabled = false;
      });
  }

  function queryRun() {
    const runId = elements.queryRunId.value.trim();
    if (!runId) return;
    prepareHistoryView(runId);
    fetchRunResult(runId, true, false);
  }

  function resetForm() {
    elements.btnStart.disabled = false;
    elements.btnCancel.disabled = true;
    submitting = false;
  }

  function startNewConversation() {
    setWorkspace("research");
    archiveCurrentExchange();
    nextPaperOrdinal = 0;
    currentPaperOrdinalStart = 0;
    resetRunView();
    resetForm();
    activeRunId = null;
    startTime = null;
    elements.currentRunId.textContent = "New search";
    elements.elapsed.textContent = "00:00";
    elements.activityTime.textContent = "00:00";
    elements.userMessage.hidden = true;
    elements.assistantMessage.hidden = true;
    elements.overviewIdle.hidden = false;
    elements.workspaceState.textContent = "等待提问";
    setStatus("idle");
    setConnection(false);
    elements.workspaceState.textContent = "等待提问";
    elements.topic.value = "";
    elements.conversationThread.replaceChildren();
    lastResultData = null;
    sessionTurnCount = 0;
    elements.sessionTopic.textContent = "新的研究问题";
    elements.sessionSummary.textContent = "正在创建新会话";
    activeSessionId = "";
    window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
    createSession().catch(function (error) { addError("Could not create session: " + error.message); });
    resizeComposer();
    closeSidebar();
    elements.topic.focus();
  }

  $$(".tab-button").forEach((button) => button.addEventListener("click", () => setActiveTab(button.dataset.tab)));
  elements.btnSkipTyping.addEventListener("click", function () { stopReportTyping(true); });
  elements.btnStart.addEventListener("click", startResearch);
  elements.btnDirectMore.addEventListener("click", loadMoreDirectPapers);
  elements.btnCancel.addEventListener("click", cancelResearch);
  elements.btnQuery.addEventListener("click", queryRun);
  elements.btnHistoryRefresh.addEventListener("click", function () { loadRunHistory(true); });
  elements.btnHistoryMore.addEventListener("click", function () { loadRunHistory(false); });
  elements.btnConfig.addEventListener("click", function () {
    // 移动端:抽屉开/关切换;桌面端:整栏折叠/展开切换
    if (isMobileLayout()) {
      if (elements.sidebar.classList.contains("is-open")) closeSidebar();
      else openSidebar();
    } else {
      toggleDesktopSidebar();
    }
  });
  elements.btnSidebarClose.addEventListener("click", function () {
    closeSidebar();                                    // 关闭移动端抽屉与遮罩
    elements.shell.classList.add("sidebar-collapsed"); // 桌面端收起侧栏
    elements.btnConfig.setAttribute("aria-expanded", "false");
  });
  elements.backdrop.addEventListener("click", closeSidebar);
  window.addEventListener("resize", function () {
    // 跨断点时清理移动端抽屉残留,避免遮罩挡住桌面界面
    if (!isMobileLayout()) closeSidebar();
  });
  elements.btnFocusTopic.addEventListener("click", startNewConversation);
  elements.btnNewSessionTop.addEventListener("click", startNewConversation);
  $$("[data-workspace]").forEach(function (button) {
    button.addEventListener("click", function () { setWorkspace(button.dataset.workspace); });
  });
  $$('[data-library-origin]').forEach(function (button) {
    button.addEventListener("click", function () {
      libraryOrigin = button.dataset.libraryOrigin || "all";
      $$('[data-library-origin]').forEach(function (item) { item.classList.toggle("is-active", item === button); });
      loadWorkspaceCollection("library", elements.librarySearch.value.trim());
    });
  });
  [
    [elements.librarySearch, "library"],
    [elements.evidenceLibrarySearch, "evidence"],
    [elements.reportsSearch, "reports"]
  ].filter(function (entry) { return Boolean(entry[0]); }).forEach(function (entry) {
    entry[0].addEventListener("input", function () {
      if (collectionSearchTimer) window.clearTimeout(collectionSearchTimer);
      collectionSearchTimer = window.setTimeout(function () {
        loadWorkspaceCollection(entry[1], entry[0].value.trim());
      }, 220);
    });
  });
  if (elements.paperDetailButton && elements.paperDetailQuery) {
    elements.paperDetailButton.addEventListener("click", queryPaperDetail);
    elements.paperDetailQuery.addEventListener("keydown", function (event) {
      if (event.key === "Enter") { event.preventDefault(); queryPaperDetail(); }
    });
  }
  elements.chatScroll.addEventListener("wheel", function (event) {
    if (event.deltaY < 0) followLatest = false;
  }, { passive: true });
  elements.chatScroll.addEventListener("scroll", function () {
    const distance = elements.chatScroll.scrollHeight - elements.chatScroll.scrollTop - elements.chatScroll.clientHeight;
    followLatest = distance < 80;
  }, { passive: true });
  elements.maxSources.addEventListener("input", function () {
    const raw = elements.maxSources.value.trim();
    const maximum = Number.parseInt(elements.maxSources.max, 10) || 20;
    if (raw !== "" && Number(raw) > maximum) normalizeMaxSources(true);
  });
  elements.maxSources.addEventListener("blur", function () { normalizeMaxSources(true); });
  elements.topic.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); startResearch(); }
  });
  elements.topic.addEventListener("input", resizeComposer);
  $$("[data-prompt]").forEach(function (button) {
    button.addEventListener("click", function () {
      elements.topic.value = button.dataset.prompt || "";
      resizeComposer();
      elements.topic.focus();
    });
  });
  document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeSidebar(); });

  (function init() {
    setConnection(false);
    setStatus("idle");
    setActiveTab("sources");
    clearResults();
    resizeComposer();
    loadRunHistory(true);
    ensureSession().catch(function (error) {
      elements.sessionId.textContent = "会话创建失败";
      addError("Could not initialize session: " + error.message);
    });
    const runId = new URLSearchParams(window.location.search).get("run_id");
    if (runId) { elements.queryRunId.value = runId; queryRun(); }
  })();
})();
