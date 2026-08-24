"""
app/api/routes.py

API 路由定义 —— 类比 Spring Boot 的 @RestController。

支持 loop 与 graph_send 两种执行后端。
支持异步运行、SSE 实时 Trace、取消和后台任务管理。
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    HealthResponse,
    ResearchRequest,
    ResearchRunResponse,
    PaperDetailResponse,
    PaperPageResponse,
    PaperSource,
    SessionCreateRequest,
    SessionResponse,
)
from app.services.orchestrator import Orchestrator
from app.services.run_store import run_store
from app.services.event_broker import event_broker
from app.core.config import config
from app.mcp.manager import mcp_manager
from app.services.session_store import (
    SessionContext,
    SessionExpiredError,
    SessionNotFoundError,
    session_store,
)
from app.services.context_compressor import context_compressor
from app.services.user_memory import user_memory_store
from app.services.workspace_catalog import workspace_catalog

router = APIRouter(tags=["research"])

orchestrator = Orchestrator()

VALID_BACKENDS = frozenset({"loop", "graph_send"})

# ---- TaskManager：追踪后台任务，支持取消 ----
_task_registry: Dict[str, asyncio.Task] = {}
_memory_tasks: set[asyncio.Task] = set()


def _register_task(run_id: str, task: asyncio.Task):
    _task_registry[run_id] = task
    task.add_done_callback(lambda _t: _task_registry.pop(run_id, None))


def _session_or_http_error(session_id: str, *, touch: bool = False) -> SessionContext:
    try:
        return session_store.get(session_id, touch=touch)
    except SessionExpiredError:
        raise HTTPException(
            status_code=410,
            detail={"error_code": "SESSION_EXPIRED", "message": f"Session expired: {session_id}"},
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"},
        )


def _restore_session_from_history(session_id: str, *, ttl_minutes: int | None = None) -> tuple[SessionContext, List[Dict[str, Any]], bool]:
    """从历史 runs 恢复 Session；返回 (当前 Session, runs, 是否新建恢复 Session)。"""
    runs = run_store.list_session_runs(session_id)
    if not runs:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "CONVERSATION_NOT_FOUND",
                "message": f"Conversation not found: {session_id}",
            },
        )
    try:
        return session_store.get(session_id, touch=True), runs, False
    except (SessionExpiredError, SessionNotFoundError):
        # 关键步骤：历史 Session 失效时创建带历史论文上下文的恢复 Session，而不是空 Session。
        restored = session_store.restore_from_runs(
            session_id,
            runs,
            ttl_minutes=ttl_minutes,
        )
        return restored, runs, True
def _public_session(context: SessionContext) -> Dict[str, Any]:
    payload = context.model_dump(exclude={"conversation_messages"})
    payload["recent_messages"] = [item.model_dump() for item in context.recent_messages]
    # 对外返回论文历史时统一移除全文和向量，避免最近一批绕过公开字段裁剪。
    for field in ("recommended_papers", "last_recommendation_batch"):
        payload[field] = [
            {
                key: value for key, value in paper.items()
                if key not in {"full_text", "text", "embedding"}
            }
            for paper in getattr(context, field)
        ]
    return payload

# ---- SessionTurn：处理用户输入,压缩上下文,加载相关论文 ----
async def _prepare_session_turn(
    query: str,
    session_id: str | None,
    agent_mode: str,
) -> Tuple[SessionContext, List[Dict[str, Any]], str, List[str]]:
    context = (
        _session_or_http_error(session_id, touch=True)
        if session_id else session_store.create()
    )
    messages = (
        [dict(item) for item in context.conversation_messages]
        or [
            {
                "role": item.role,
                "content": item.content_preview,
                "intent": item.intent,
                "tool_calls_summary": item.tool_calls_summary,
            }
            for item in context.recent_messages
        ]
    ) + [{"role": "user", "content": query}]
    # 通过四级压缩上下文(L3->L1->L2->L4)
    compaction = await context_compressor.compress(
        messages, context, allow_llm=(agent_mode == "llm")
    )
    # 压缩确实生效(任一 L3/L1/L2/L4 触发)时,才把压缩结果回写进 SessionContext
    if compaction.levels_applied:
        # 累计压缩次数,供前端展示与后续门控使用
        changes: Dict[str, Any] = {"compaction_count": context.compaction_count + 1}
        # 仅 L4 产出结构化摘要时才落盘:summary_so_far 供后续轮次复用,last_compaction_turn 供频次门控
        if compaction.summary is not None:
            changes.update({
                "summary_so_far": compaction.summary.summary,
                "last_compaction_turn": context.turn_count,
            })
        # 只更新 SessionContext 声明的字段,并重绑定 context 供本轮后续逻辑使用
        context = session_store.update(context.session_id, **changes)
    # 按当前问题加载相关的长期记忆条目(user/feedback/project/reference,最多 3 条)
    relevant = await user_memory_store.load_relevant(
        query, use_llm=(agent_mode == "llm")
    )
    # 把记忆索引 + 相关记忆拼成 <memory_index>/<relevant_memories> 提示块,注入 Controller 的 system prompt
    memory_prompt = user_memory_store.build_prompt(relevant)
    return context, compaction.messages, memory_prompt, compaction.levels_applied


def _report_sections(result: Dict[str, Any]) -> List[str]:
    outline = result.get("outline") or {}
    sections = [
        str(item.get("heading") or "").strip()
        for item in outline.get("sections", []) if isinstance(item, dict)
    ]
    if sections:
        return [item for item in sections if item]
    import re
    return re.findall(r"^#{1,4}\s+(.+?)\s*$", result.get("final_report", ""), re.M)[:20]


def _schedule_memory_maintenance(context: SessionContext, *, allow_llm: bool) -> None:
    # 门控:至少 3 轮对话且 LLM 可用才做记忆维护
    if context.turn_count < 3 or not allow_llm:
        return

    async def maintain(snapshot: SessionContext):
        # 先提取本轮新记忆,再整理(去重/合并)存量记忆
        await user_memory_store.extract(snapshot)
        consolidated = await user_memory_store.consolidate(snapshot)
        # 整理后存量仍 ≥10 条时,记录本轮为 consolidation 轮次(供频次门控复用)
        if len(consolidated) >= 10:
            try:
                session_store.update(
                    snapshot.session_id,
                    last_consolidation_turn=snapshot.turn_count,
                )
            except (SessionNotFoundError, SessionExpiredError):
                pass

    # 异步执行,不阻塞本轮响应;用 _memory_tasks 追踪避免悬挂
    task = asyncio.create_task(maintain(context.model_copy(deep=True)))
    _memory_tasks.add(task)
    task.add_done_callback(_memory_tasks.discard)


def _update_session_after_run(
    context: SessionContext,
    query: str,
    result: Dict[str, Any],
    conversation_messages: List[Dict[str, Any]] | None = None,
) -> SessionContext:
    """把本轮运行结果归档回 Session,供下一轮追问复用。"""
    session_id = context.session_id
    base_conversation_messages = [dict(item) for item in context.conversation_messages]
    sources = result.get("sources") or []
    # 研究类 intent：论文按稳定标识累积追加，供“再推荐”和后续指代复用。
    intent = result.get("intent")
    is_recommendation = intent in {"paper_recommendation", "recommend_more"}
    # 展示为“编号论文列表”的 intent 会刷新最近批次；批次让“第 N 篇 / 这篇”指代
    # 优先对齐屏幕上刚返回的那批论文（跨主题搜索时避免指回上一主题的论文）。
    list_intents = {
        "paper_recommendation", "recommend_more",
        "literature_search", "paper_graph_lookup",
    }
    update_batch = intent in list_intents
    if update_batch and sources:
        # 与 node_direct_reviewer 的展示一致：续接推荐沿用会话累计序号，新搜索/推荐从 1 重新编号。
        batch_start = (len(context.recommended_papers) + 1) if is_recommendation else 1
        session_store.update(session_id, last_recommendation_batch_start=batch_start)
    if is_recommendation or (sources and intent in {
        "literature_search", "paper_graph_lookup", "deep_research", "research_from_session",
    }):
        session_store.append_recommended_papers(
            session_id,
            sources,
            recommendation_topic=result.get("research_topic", "") if is_recommendation else "",
            update_last_batch=update_batch,
        )
    resolved_ids = result.get("resolved_paper_ids") or []
    # 短路轮解析出的论文 ID 写回 active paper 状态(首个为 active,其余进最近提及列表)
    if resolved_ids:
        for paper_id in reversed(resolved_ids[1:]):
            session_store.set_active_paper(session_id, paper_id)
        session_store.set_active_paper(session_id, resolved_ids[0])
    elif sources and update_batch:
        # 关键修复：纯搜索/推荐轮未解析具体论文时，把批次首篇设为 active paper，
        # 否则“这篇论文”在跨主题搜索后要么指向上一个主题的陈旧论文，要么无法解析。
        first = sources[0]
        first_id = str(first.get("source_id") or first.get("paper_id") or "")
        if first_id:
            session_store.set_active_paper(session_id, first_id)
    # 深度研究产出报告:记录章节 + run_id,供报告追问从 run_store 取报告
    if result.get("intent") in {"deep_research", "research_from_session"} and result.get("final_report"):
        session_store.set_report_sections(session_id, _report_sections(result), result.get("run_id"))
    assistant = result.get("answer") or result.get("final_report") or ""
    transcript = [dict(item) for item in (conversation_messages or [])]
    tool_payload = {
        "sources": result.get("sources") or [],
        "evidence_cards": result.get("evidence_cards") or [],
        "conversation_result": result.get("conversation_result") or {},
    }
    turn_tool_messages: List[Dict[str, Any]] = []
    # 有工具产出时,补成一条"run 级"tool_use↔tool_result 配对,保证压缩时配对不被拆散
    if any(tool_payload.values()):
        turn_tool_messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"run-{result.get('run_id', '')}", "name": result.get("route_name") or result.get("execution_route") or "research"}],
            },
            {
                "role": "tool",
                "tool_call_id": f"run-{result.get('run_id', '')}",
                "content": json.dumps(tool_payload, ensure_ascii=False, default=str),
            },
        ]
        transcript.extend(turn_tool_messages)
    transcript.append({"role": "assistant", "content": assistant})
    # 关键步骤：原子合并本轮 transcript，避免并发请求用旧快照覆盖先完成的轮次。
    session_store.merge_conversation_turn(
        session_id,
        base_messages=base_conversation_messages,
        prepared_messages=transcript,
        turn_messages=[
            {"role": "user", "content": query},
            *turn_tool_messages,
            {"role": "assistant", "content": assistant},
        ],
    )
    # 记录本轮对话摘要(截断 200 字符)并累加 turn_count
    context = session_store.record_turn(
        session_id,
        user_content=query,
        assistant_content=assistant,
        intent=result.get("intent", ""),
        tool_calls_summary=", ".join(result.get("selected_tools") or []),
    )
    # 本轮结束后异步触发记忆提取/整理
    _schedule_memory_maintenance(
        context, allow_llm=(str(result.get("agent_mode") or "") == "llm")
    )
    return context


# ================================================================
# GET /health
# ================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version=config.APP_VERSION)


@router.get("/api/mcp/tools")
async def list_mcp_tools():
    """Return MCP connection state and registered public tool names without secrets."""
    return mcp_manager.status()


# ================================================================
# 多轮对话 Session 接口
# ================================================================

@router.post("/api/sessions", response_model=SessionResponse, status_code=201)
async def create_session(req: SessionCreateRequest = SessionCreateRequest()):
    return _public_session(session_store.create(ttl_minutes=req.ttl_minutes))


@router.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    return _public_session(_session_or_http_error(session_id))


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_store.delete(session_id):
        raise HTTPException(
            status_code=404,
            detail={"error_code": "SESSION_NOT_FOUND", "message": f"Session not found: {session_id}"},
        )
    return {"session_id": session_id, "status": "deleted"}


@router.post("/api/sessions/{session_id}/restore", response_model=SessionResponse)
async def restore_session(session_id: str, req: SessionCreateRequest = SessionCreateRequest()):
    """恢复已过期或重启后丢失的 Session，并重建论文与 active paper 上下文。"""
    context, _, _ = _restore_session_from_history(
        session_id,
        ttl_minutes=req.ttl_minutes,
    )
    return _public_session(context)


# ================================================================
# POST /api/research/runs（异步）
# ================================================================

@router.post("/api/research/runs", status_code=202, response_model=ResearchRunResponse)
async def research_async(
    req: ResearchRequest,
    backend: str = Query(default="graph_send", description="Execution backend"),
):
    """创建异步研究任务。返回 HTTP 202。"""
    agent_mode = req.agent_mode or os.getenv("AGENT_MODE", "llm")
    session_context, conversation_messages, memory_prompt, _ = await _prepare_session_turn(
        req.topic, req.session_id, agent_mode
    )

    # ---- 校验 backend ----
    if backend not in VALID_BACKENDS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid backend: '{backend}'. Valid options: {', '.join(sorted(VALID_BACKENDS))}",
        )

    run_id = str(uuid.uuid4())[:8]

    # 创建 run
    run_store.create(topic=req.topic, run_id=run_id)
    run_store.update(
        run_id, status="queued", backend=backend, agent_mode=agent_mode,
        session_id=session_context.session_id,
    )

    # 初始化 event broker
    event_broker.init_run(run_id, topic=req.topic)

    # 发布 run_started
    event_broker.publish(run_id, "run_started", {
        "run_id": run_id, "topic": req.topic,
        "backend": backend, "agent_mode": agent_mode,
        "max_sources": req.max_sources,
    })

    # 后台启动研究任务
    task = asyncio.create_task(_run_research_background(
        run_id=run_id, topic=req.topic, max_sources=req.max_sources,
        language=req.language, mode=req.mode, run_eval=req.run_eval,
        backend=backend, agent_mode=agent_mode,
        session_context=session_context,
        conversation_messages=conversation_messages,
        memory_prompt=memory_prompt,
    ))
    _register_task(run_id, task)

    return ResearchRunResponse(
        run_id=run_id, status="queued", topic=req.topic,
        session_id=session_context.session_id,
    )


async def _run_research_background(
    run_id: str, topic: str, max_sources: int, language: str,
    mode: str, run_eval: bool, backend: str, agent_mode: str,
    session_context: SessionContext,
    conversation_messages: List[Dict[str, Any]],
    memory_prompt: str,
):
    """后台执行研究任务，通过 run_graph_async 实时发布 SSE 事件。"""
    try:
        if backend == "graph_send":
            from app.graph.runtime import run_graph_async
            result = await run_graph_async(
                topic=topic, max_sources=max_sources,
                language=language, mode=mode, run_eval=run_eval,
                agent_mode=agent_mode, run_id=run_id,
                session_context=session_context,
                conversation_messages=conversation_messages,
                memory_prompt=memory_prompt,
            )
        elif backend == "loop":
            result = await orchestrator.run(
                topic=topic, max_sources=max_sources,
                language=language, mode=mode, run_eval=run_eval,
                run_id=run_id,
            )
            event_broker.publish(run_id, "run_finished", {
                "run_id": run_id,
                "status": result.get("status", "completed"),
            })
            event_broker.finish_run(run_id)
        result["session_id"] = session_context.session_id
        result.setdefault("answer", result.get("final_report", ""))
        _update_session_after_run(
            session_context, topic, result,
            conversation_messages=conversation_messages,
        )
    except asyncio.CancelledError:
        run_store.update(run_id, status="cancelled")
        run_store.finish(run_id, "cancelled")
        event_broker.cancel_run(run_id)
        event_broker.publish(run_id, "error", {
            "message": "Run cancelled by user",
            "error_type": "CancelledError",
        })
        event_broker.finish_run(run_id)
    except Exception as e:
        # ---- 失败状态落库 ----
        run_store.update(run_id, status="failed", error=str(e)[:500])
        run_store.finish(run_id, "failed")
        event_broker.publish(run_id, "error", {
            "message": str(e)[:500],
            "error_type": type(e).__name__,
        })
        event_broker.finish_run(run_id, str(e)[:200])


# ================================================================
# POST /api/research/runs/{run_id}/cancel
# ================================================================

@router.post("/api/research/runs/{run_id}/cancel")
async def cancel_research(run_id: str):
    """取消正在运行的异步研究任务。"""
    task = _task_registry.get(run_id)
    if task is None:
        # 可能已经完成或不存在
        run_data = run_store.get(run_id)
        if run_data is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return {
            "run_id": run_id,
            "status": run_data.get("status", "unknown"),
            "message": "Run is not currently running; cannot cancel",
        }

    event_broker.cancel_run(run_id)
    task.cancel()
    run_store.update(run_id, status="cancelled")

    return {"run_id": run_id, "status": "cancelled", "message": "Run cancelled"}


# ================================================================
# GET /api/research/stream/{run_id}（SSE）
# ================================================================

@router.get("/api/research/stream/{run_id}")
async def research_stream(request: Request, run_id: str):
    """
    SSE 实时事件流。

    行为：
    - 首次连接：回放所有已有事件，然后订阅新事件
    - 重连（Last-Event-ID）：只回放未消费事件，然后订阅新事件
    - 已完成 run：回放全部事件，发送 run_finished，关闭
    - 不存在的 run：直接 404
    """
    # 检查 run 是否存在（RunStore + EventBroker 双重检查）
    run_data = run_store.get(run_id)
    if run_data is None and not event_broker.run_exists(run_id):
        return JSONResponse(
            status_code=404,
            content={"error": f"Run not found: {run_id}"},
        )

    last_event_id = request.headers.get("Last-Event-ID", "")
    is_finished_at_start = event_broker.is_finished(run_id)

    async def event_generator():
        # ---- 回放已有事件（首次连接为全部，重连只发送未消费事件） ----
        if is_finished_at_start:
            # 已完成 run：回放全部事件，然后发送 run_finished，关闭
            all_events = event_broker.get_all_events(run_id)
            for ev in all_events:
                yield {"id": ev["id"], "event": ev["event"], "data": ev["data"]}
            # Ensure run_finished is sent
            if not all_events or all_events[-1]["event"] != "run_finished":
                yield {
                    "event": "run_finished",
                    "data": json.dumps({
                        "run_id": run_id,
                        "status": str((run_data or {}).get("status") or "completed"),
                    }),
                }
            return

        # 运行中 run：回放已有事件
        replay_events = event_broker.get_unconsumed_events(run_id, last_event_id)
        for ev in replay_events:
            yield {"id": ev["id"], "event": ev["event"], "data": ev["data"]}

        # ---- 订阅后续实时事件 ----
        queue = event_broker.subscribe(run_id)
        if queue is None:
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Run {run_id} not available for streaming"}),
            }
            return

        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    # 等待新事件，超时 15s
                    event = await asyncio.wait_for(queue.get(), timeout=15.0) # 阻塞等待
                except asyncio.TimeoutError:
                    if event_broker.is_finished(run_id):
                        break
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"ts": int(time.time() * 1000)}),
                    }
                    continue

                if event is None:
                    break

                yield {"id": event["id"], "event": event["event"], "data": event["data"]}

                if event["event"] == "run_finished":
                    break
                if event["event"] == "error" and event_broker.is_finished(run_id):
                    break
        finally:
            event_broker.unsubscribe(run_id, queue)

    return EventSourceResponse(event_generator())


# ================================================================
# GET /api/runs/{run_id}
# ================================================================

@router.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """查询某次运行的详情。"""
    run_data = run_store.get(run_id)
    if run_data is None:
        return {"run_id": run_id, "error": f"Run not found: {run_id}"}
    return run_data


@router.get("/api/conversations/{session_id}/runs")
async def get_conversation_runs(session_id: str):
    """读取同一 Session 的全部运行，供前端恢复完整多轮对话。"""
    context, runs, restored = _restore_session_from_history(session_id)
    return {
        "session_id": context.session_id,
        "source_session_id": session_id,
        "restored": restored,
        "session": _public_session(context),
        "runs": runs,
        "turn_count": len(runs),
    }


@router.get("/api/research/runs/{run_id}/papers", response_model=PaperPageResponse)
async def get_more_papers(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
):
    """继续获取论文推荐或论文引用图谱结果，不设置应用级总数量上限。"""
    run_data = run_store.get(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    intent = str(run_data.get("intent") or "")
    if run_data.get("execution_route") != "direct_tool" or intent not in {
        "paper_recommendation", "paper_graph_lookup",
    }:
        raise HTTPException(
            status_code=409,
            detail="Pagination is available for paper recommendations and paper graph results only",
        )

    from app.tools.semantic_scholar_provider import SemanticScholarClient

    client = SemanticScholarClient()
    tool_args = run_data.get("selected_tool_args") or {}
    if intent == "paper_recommendation":
        result = await client.recommend_page(
            topic=str(tool_args.get("topic") or run_data.get("research_topic") or run_data.get("topic") or ""),
            offset=offset,
            limit=limit,
            positive_paper_ids=tool_args.get("positive_paper_ids"),
            negative_paper_ids=tool_args.get("negative_paper_ids"),
        )
    else:
        result = await client.paper_graph(
            paper_query=str(tool_args.get("paper_query") or run_data.get("research_topic") or ""),
            relation=str(tool_args.get("relation") or "details"),
            limit=limit,
            offset=offset,
        )

    if not result.success:
        raise HTTPException(status_code=502, detail=result.error or "Academic provider request failed")

    data = result.data if isinstance(result.data, dict) else {}
    items = data.get("results") or data.get("sources") or []
    next_offset = data.get("next_offset")
    return PaperPageResponse(
        run_id=run_id,
        intent=intent,
        items=[PaperSource(**item) for item in items],
        offset=offset,
        limit=limit,
        returned=len(items),
        total=data.get("total_found"),
        has_more=bool(data.get("has_more")),
        next_offset=next_offset,
    )


# ================================================================
# GET /api/runs
# ================================================================

@router.get("/api/runs")
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str = Query(default=None),
    group_by_session: bool = Query(
        default=True,
        description="Group multiple runs from the same conversation session into one history item",
    ),
):
    """分页列出历史摘要；最近报告默认按 Session 聚合为对话窗口。"""
    runs, total = run_store.list_runs_page(
        limit=limit, offset=offset, status=status,
        group_by_session=group_by_session,
    )
    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


# ================================================================
# Read-only workspace collections
# ================================================================

@router.get("/api/library")
async def list_library_papers(
    query: str = Query(default="", max_length=200),
    origin: str = Query(default="all", pattern="^(all|local|searched)$"),
):
    """List indexed local papers and deduplicated papers found by past runs."""
    return workspace_catalog.papers(query=query, origin=origin)


@router.get("/api/papers/detail", response_model=PaperDetailResponse)
async def get_paper_detail(
    query: str = Query(..., min_length=1, max_length=300),
):
    """按论文标题返回论文详情：本地 Zotero 库命中优先，否则 S2/OpenAlex 兜底。"""
    from app.services.paper_detail import resolve_paper_detail

    return await resolve_paper_detail(query)


@router.get("/api/evidence-library")
async def list_evidence_library(query: str = Query(default="", max_length=200)):
    """List historical evidence cards enriched with their source and run metadata."""
    return workspace_catalog.evidence(query=query)


@router.get("/api/reports")
async def list_research_reports(query: str = Query(default="", max_length=200)):
    """List persisted research reports with their source/evidence counts."""
    return workspace_catalog.reports(query=query)
