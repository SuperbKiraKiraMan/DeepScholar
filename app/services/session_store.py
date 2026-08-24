"""In-memory conversation sessions for multi-turn research."""

from __future__ import annotations

import threading
import time
import uuid
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _now_ms() -> int:
    return int(time.time() * 1000)


class SessionNotFoundError(KeyError):
    """The requested session has never existed (or was explicitly deleted)."""


class SessionExpiredError(KeyError):
    """The requested session existed but its inactivity TTL elapsed."""


class SessionMessage(BaseModel):
    role: str
    content_preview: str = Field(default="", max_length=200)
    intent: str = ""
    tool_calls_summary: str = ""
    timestamp_ms: int = Field(default_factory=_now_ms)


class SessionContext(BaseModel):
    session_id: str
    ttl_minutes: int = Field(default=30, ge=1)
    recommended_papers: List[Dict[str, Any]] = Field(default_factory=list)
    last_recommendation_batch: List[Dict[str, Any]] = Field(default_factory=list)
    # 最近批次在用户可见编号中的起始序号：新搜索/推荐从 1 重新编号，
    # “再推荐”沿用会话累计序号。引用解析据此把“第 N 篇”对齐到屏幕上的批次。
    last_recommendation_batch_start: int = Field(default=1)
    last_recommendation_topic: str = ""
    active_paper_id: Optional[str] = None
    active_report_id: Optional[str] = None
    last_intent: str = ""
    last_mentioned_paper_ids: List[str] = Field(default_factory=list)
    last_report_sections: List[str] = Field(default_factory=list)
    recent_messages: List[SessionMessage] = Field(default_factory=list)
    # Full model-facing transcript; recent_messages remains the small public preview window.
    conversation_messages: List[Dict[str, Any]] = Field(default_factory=list)
    compaction_count: int = 0
    summary_so_far: str = ""
    turn_count: int = 0
    last_compaction_turn: int = -10_000
    last_consolidation_turn: int = -10_000
    created_at_ms: int = Field(default_factory=_now_ms)
    updated_at_ms: int = Field(default_factory=_now_ms)
    expires_at_ms: int = 0
    restored_from_session_id: Optional[str] = None


class SessionStore:
    """Small thread-safe store; persistence can replace it without changing callers."""

    def __init__(self, default_ttl_minutes: int = 30):
        self.default_ttl_minutes = max(1, int(default_ttl_minutes))
        self._sessions: Dict[str, SessionContext] = {}
        self._expired_ids: set[str] = set()
        self._lock = threading.RLock()

    def create(
        self,
        session_id: Optional[str] = None,
        ttl_minutes: Optional[int] = None,
    ) -> SessionContext:
        with self._lock:
            sid = session_id or str(uuid.uuid4())
            ttl = max(1, int(ttl_minutes or self.default_ttl_minutes))
            now = _now_ms()
            context = SessionContext(
                session_id=sid,
                ttl_minutes=ttl,
                created_at_ms=now,
                updated_at_ms=now,
                expires_at_ms=now + ttl * 60_000,
            )
            self._sessions[sid] = context
            self._expired_ids.discard(sid)
            return context.model_copy(deep=True)

    def get(self, session_id: str, *, touch: bool = False) -> SessionContext:
        with self._lock:
            context = self._sessions.get(session_id)
            if context is None:
                if session_id in self._expired_ids:
                    raise SessionExpiredError(session_id)
                raise SessionNotFoundError(session_id)
            if self._is_expired(context):
                self._sessions.pop(session_id, None)
                self._expired_ids.add(session_id)
                raise SessionExpiredError(session_id)
            if touch:
                context = self._touch_locked(context)
            return context.model_copy(deep=True)

    def update(self, session_id: str, **changes: Any) -> SessionContext:
        with self._lock:
            context = self._require_active_locked(session_id)
            allowed = set(SessionContext.model_fields)
            payload = context.model_dump()
            payload.update({key: value for key, value in changes.items() if key in allowed})
            payload["updated_at_ms"] = _now_ms()
            payload["expires_at_ms"] = payload["updated_at_ms"] + int(payload["ttl_minutes"]) * 60_000
            updated = SessionContext(**payload)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    def touch(self, session_id: str) -> SessionContext:
        with self._lock:
            return self._touch_locked(self._require_active_locked(session_id)).model_copy(deep=True)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            self._expired_ids.discard(session_id)
            return self._sessions.pop(session_id, None) is not None

    def expire_stale(self, now_ms: Optional[int] = None) -> List[str]:
        with self._lock:
            now = int(now_ms or _now_ms())
            stale = [sid for sid, ctx in self._sessions.items() if ctx.expires_at_ms <= now]
            for sid in stale:
                self._sessions.pop(sid, None)
                self._expired_ids.add(sid)
            return stale

    def record_turn(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        intent: str = "",
        tool_calls_summary: str = "",
    ) -> SessionContext:
        with self._lock:
            context = self._require_active_locked(session_id)
            now = _now_ms()
            messages = list(context.recent_messages)
            messages.extend([
                SessionMessage(
                    role="user",
                    content_preview=(user_content or "")[:200],
                    intent=intent,
                    timestamp_ms=now,
                ),
                SessionMessage(
                    role="assistant",
                    content_preview=(assistant_content or "")[:200],
                    intent=intent,
                    tool_calls_summary=(tool_calls_summary or "")[:500],
                    timestamp_ms=now,
                ),
            ])
            # Ten turns = at most twenty role messages.
            context.recent_messages = messages[-20:]
            context.turn_count += 1
            context.last_intent = intent or context.last_intent
            self._touch_locked(context)
            return context.model_copy(deep=True)

    def merge_conversation_turn(
        self,
        session_id: str,
        *,
        base_messages: List[Dict[str, Any]],
        prepared_messages: List[Dict[str, Any]],
        turn_messages: List[Dict[str, Any]],
    ) -> SessionContext:
        """原子写入本轮完整消息，并在并发冲突时保留其他已完成轮次。"""
        with self._lock:
            context = self._require_active_locked(session_id)
            current = [dict(item) for item in context.conversation_messages]

            # 关键步骤：基线未变化时保留压缩结果；发生并发写入时只追加当前轮增量。
            if current == [dict(item) for item in base_messages]:
                context.conversation_messages = [dict(item) for item in prepared_messages]
            else:
                context.conversation_messages = current + [dict(item) for item in turn_messages]
            self._touch_locked(context)
            return context.model_copy(deep=True)

    def set_active_paper(self, session_id: str, paper_id: Optional[str]) -> SessionContext:
        with self._lock:
            context = self._require_active_locked(session_id)
            context.active_paper_id = paper_id or None
            if paper_id:
                ordered = [paper_id] + [
                    item for item in context.last_mentioned_paper_ids if item != paper_id
                ]
                context.last_mentioned_paper_ids = ordered[:10]
            self._touch_locked(context)
            return context.model_copy(deep=True)

    def set_recommended_papers(
        self,
        session_id: str,
        papers: List[Dict[str, Any]],
    ) -> SessionContext:
        """兼容旧调用：替换论文上下文，并把本批记录为最近推荐。"""
        with self._lock:
            context = self._require_active_locked(session_id)
            normalized = self._deduplicate_papers(papers or [])
            context.recommended_papers = normalized
            context.last_recommendation_batch = [dict(item) for item in normalized]
            valid_ids = {
                str(item.get("source_id") or item.get("paper_id") or "")
                for item in context.recommended_papers
            }
            if context.active_paper_id and context.active_paper_id not in valid_ids:
                context.active_paper_id = None
            self._touch_locked(context)
            return context.model_copy(deep=True)

    def append_recommended_papers(
        self,
        session_id: str,
        papers: List[Dict[str, Any]],
        *,
        recommendation_topic: str = "",
        update_last_batch: bool = True,
    ) -> SessionContext:
        """将本轮新论文去重后追加到会话历史，避免覆盖已有推荐。"""
        with self._lock:
            context = self._require_active_locked(session_id)
            existing = [dict(item) for item in context.recommended_papers]
            existing_keys = {self._paper_key(item) for item in existing}

            # 关键步骤：本批既要排除历史论文，也要消除工具本次返回的重复项。
            new_batch: List[Dict[str, Any]] = []
            for raw in papers or []:
                paper = dict(raw)
                key = self._paper_key(paper)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                new_batch.append(paper)

            context.recommended_papers = existing + new_batch
            if update_last_batch:
                context.last_recommendation_batch = [dict(item) for item in new_batch]
                if recommendation_topic.strip():
                    context.last_recommendation_topic = recommendation_topic.strip()
            self._touch_locked(context)
            return context.model_copy(deep=True)

    @classmethod
    def _deduplicate_papers(cls, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按稳定论文标识去重并保留首次出现顺序。"""
        seen: set[str] = set()
        result: List[Dict[str, Any]] = []
        for raw in papers:
            paper = dict(raw)
            key = cls._paper_key(paper)
            if key in seen:
                continue
            seen.add(key)
            result.append(paper)
        return result

    @staticmethod
    def _paper_key(paper: Dict[str, Any]) -> str:
        """生成跨提供商可复用的论文去重键。"""
        for field in ("doi", "semantic_scholar_id", "openalex_id", "paper_id", "source_id"):
            value = str(paper.get(field) or "").strip().lower()
            if value:
                return f"{field}:{value}"
        url = re.sub(r"^https?://(?:www\.)?", "", str(paper.get("url") or "").strip().lower()).rstrip("/")
        if url:
            return f"url:{url}"
        title = re.sub(r"\s+", " ", str(paper.get("title") or "").strip().lower())
        return f"title:{title}"

    def set_report_sections(
        self,
        session_id: str,
        sections: List[str],
        report_id: Optional[str] = None,
    ) -> SessionContext:
        with self._lock:
            context = self._require_active_locked(session_id)
            context.last_report_sections = list(dict.fromkeys(
                str(item).strip() for item in (sections or []) if str(item).strip()
            ))[:20]
            if report_id:
                context.active_report_id = report_id
            self._touch_locked(context)
            return context.model_copy(deep=True)

    def restore_from_runs(
        self,
        source_session_id: str,
        runs: List[Dict[str, Any]],
        *,
        ttl_minutes: Optional[int] = None,
    ) -> SessionContext:
        """从已持久化的 runs 创建带论文上下文的新恢复 Session。"""
        restored = self.create(ttl_minutes=ttl_minutes)
        restored_id = restored.session_id
        ordered_runs = sorted(
            (dict(run) for run in (runs or [])),
            key=lambda run: (str(run.get("created_at") or ""), str(run.get("run_id") or "")),
        )
        transcript: List[Dict[str, Any]] = []

        for run in ordered_runs:
            sources = run.get("sources") or []
            intent = str(run.get("intent") or "")
            is_research_run = intent in {
                "paper_recommendation", "recommend_more", "literature_search",
                "paper_graph_lookup", "deep_research", "research_from_session",
            }
            if is_research_run and isinstance(sources, list) and sources:
                # 关键步骤：按历史运行顺序追加论文，去重后保留首次出现位置，保证第 N 篇稳定。
                update_batch = intent in {
                    "paper_recommendation", "recommend_more",
                    "literature_search", "paper_graph_lookup",
                }
                if update_batch:
                    # 与实时运行一致：续接推荐沿用累计序号，新搜索/推荐从 1 编号。
                    pool = self._sessions[restored_id].recommended_papers
                    batch_start = (len(pool) + 1) if intent == "recommend_more" else 1
                    self.update(restored_id, last_recommendation_batch_start=batch_start)
                self.append_recommended_papers(
                    restored_id,
                    sources,
                    recommendation_topic=(
                        run.get("research_topic") or run.get("topic") or ""
                    ) if intent in {"paper_recommendation", "recommend_more"} else "",
                    update_last_batch=update_batch,
                )

            resolved_ids = [str(item) for item in (run.get("resolved_paper_ids") or []) if item]
            if resolved_ids:
                # 与正常运行归档保持一致：第一个解析结果作为当前 active paper。
                for paper_id in reversed(resolved_ids[1:]):
                    self.set_active_paper(restored_id, paper_id)
                self.set_active_paper(restored_id, resolved_ids[0])

            if intent in {"deep_research", "research_from_session"} and (
                run.get("final_report") or run.get("draft_report")
            ):
                self.set_report_sections(
                    restored_id,
                    self._run_report_sections(run),
                    run.get("run_id"),
                )

            query = str(run.get("topic") or "").strip()
            answer = str(
                run.get("answer") or run.get("final_report") or run.get("draft_report") or ""
            )
            if query or answer:
                transcript.append({"role": "user", "content": query})
                transcript.append({"role": "assistant", "content": answer})
                self.record_turn(
                    restored_id,
                    user_content=query,
                    assistant_content=answer,
                    intent=intent,
                    tool_calls_summary=", ".join(run.get("selected_tools") or []),
                )

        self.update(
            restored_id,
            conversation_messages=transcript,
            restored_from_session_id=source_session_id,
        )
        return self.get(restored_id)

    @staticmethod
    def _run_report_sections(run: Dict[str, Any]) -> List[str]:
        """提取历史报告章节，避免恢复 Session 依赖 API 层的解析函数。"""
        outline = run.get("outline") or {}
        if isinstance(outline, dict):
            sections = [
                str(item.get("heading") or item.get("title") or "").strip()
                for item in outline.get("sections", [])
                if isinstance(item, dict)
            ]
            sections = [item for item in sections if item]
            if sections:
                return sections[:20]
        report = str(run.get("final_report") or run.get("draft_report") or "")
        return re.findall(r"^#{1,4}\s+(.+?)\s*$", report, re.M)[:20]

    @staticmethod
    def _is_expired(context: SessionContext) -> bool:
        return bool(context.expires_at_ms and context.expires_at_ms <= _now_ms())

    def _require_active_locked(self, session_id: str) -> SessionContext:
        context = self._sessions.get(session_id)
        if context is None:
            if session_id in self._expired_ids:
                raise SessionExpiredError(session_id)
            raise SessionNotFoundError(session_id)
        if self._is_expired(context):
            self._sessions.pop(session_id, None)
            self._expired_ids.add(session_id)
            raise SessionExpiredError(session_id)
        return context

    @staticmethod
    def _touch_locked(context: SessionContext) -> SessionContext:
        now = _now_ms()
        context.updated_at_ms = now
        context.expires_at_ms = now + context.ttl_minutes * 60_000
        return context


session_store = SessionStore(int(os.getenv("SESSION_TTL_MINUTES", "30")))
