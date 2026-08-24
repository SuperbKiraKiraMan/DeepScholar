"""ContextLoad Worker 使用的只读基础设施资源适配器。"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from app.services.run_store import run_store
from app.services.session_store import SessionContext, session_store


class ContextResourceAdapter:
    """按 Controller 给出的稳定引用加载并裁剪会话资源。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._snapshots: Dict[str, SessionContext] = {}

    def bind_session_snapshot(self, context: SessionContext | None) -> None:
        """兼容直接调用 Runtime 的场景；API 会话仍优先从 SessionStore 读取。"""
        if context is None:
            return
        with self._lock:
            self._snapshots[context.session_id] = context.model_copy(deep=True)
            # 关键步骤：兼容缓存严格有界，避免长期运行时积累历史会话正文。
            while len(self._snapshots) > 128:
                self._snapshots.pop(next(iter(self._snapshots)))

    def load(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        resources: List[Dict[str, Any]] = []
        sessions: Dict[str, SessionContext] = {}

        def session_for(session_id: str) -> SessionContext | None:
            if not session_id:
                return None
            if session_id in sessions:
                return sessions[session_id]
            try:
                context = session_store.get(session_id)
            except Exception:
                with self._lock:
                    context = self._snapshots.get(session_id)
                    context = context.model_copy(deep=True) if context is not None else None
            if context is not None:
                sessions[session_id] = context
            return context

        for reference in references:
            kind = str(reference.get("resource_type") or "")
            if kind in {"paper", "report", "history"}:
                # 旧调用方已经提供显式资源时原样裁剪，保持迁移期 API 兼容。
                resources.append(self._clip_legacy(reference))
                continue
            session_id = str(reference.get("session_id") or "")
            context = session_for(session_id)
            if kind == "paper_ref" and context is not None:
                paper_id = str(reference.get("paper_id") or reference.get("source_id") or "")
                paper = next((
                    item for item in context.recommended_papers
                    if str(item.get("source_id") or item.get("paper_id") or "") == paper_id
                ), None)
                if paper is not None:
                    resources.append({**dict(paper), "resource_type": "paper"})
            elif kind == "report_ref":
                report_id = str(reference.get("report_id") or "")
                report = run_store.get(report_id) or {}
                if report:
                    resources.append({
                        "resource_type": "report",
                        "source_id": f"report:{report_id}",
                        "report_id": report_id,
                        "report_text": str(report.get("final_report") or report.get("draft_report") or "")[:80_000],
                        "evidence_cards": list(report.get("evidence_cards") or [])[:200],
                        "sources": list(report.get("sources") or [])[:100],
                        "resolved_section": reference.get("resolved_section"),
                    })
            elif kind == "history_ref" and context is not None:
                resources.append({
                    "resource_type": "history",
                    "source_id": f"history:{session_id}",
                    "history": [dict(item) for item in context.conversation_messages[-40:]],
                })
        return resources

    @staticmethod
    def _clip_legacy(resource: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(resource)
        if data.get("resource_type") == "report":
            data["report_text"] = str(data.get("report_text") or "")[:80_000]
            data["evidence_cards"] = list(data.get("evidence_cards") or [])[:200]
            data["sources"] = list(data.get("sources") or [])[:100]
        elif data.get("resource_type") == "history":
            data["history"] = list(data.get("history") or [])[-40:]
        return data


context_resource_adapter = ContextResourceAdapter()
