"""Agent-Harness-inspired, four-level conversation context compaction."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.llm.client import get_llm_client
from app.services.session_store import SessionContext


class CompactionConfig(BaseModel):
    tool_result_budget_bytes: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_TOOL_RESULT_BUDGET_BYTES", str(200 * 1024)))
    )
    single_tool_result_bytes: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_SINGLE_TOOL_RESULT_BYTES", str(30 * 1024)))
    )
    persisted_preview_chars: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_PERSISTED_PREVIEW_CHARS", "2000"))
    )
    max_messages: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_MAX_MESSAGES", "30"))
    )
    keep_head_messages: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_KEEP_HEAD_MESSAGES", "6"))
    )
    keep_tail_messages: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_KEEP_TAIL_MESSAGES", "24"))
    )
    recent_tool_results: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_RECENT_TOOL_RESULTS", "5"))
    )
    l4_token_threshold: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_L4_TOKEN_THRESHOLD", "8000"))
    )
    l4_min_turns: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_L4_MIN_TURNS", "10"))
    )
    l4_turn_gap: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_L4_TURN_GAP", "5"))
    )


class CompactionSummary(BaseModel):
    topic: str = ""
    key_papers_discussed: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    pending_questions: List[str] = Field(default_factory=list)
    user_constraints: List[str] = Field(default_factory=list)
    report_progress: str = ""
    summary: str = Field(default="", max_length=2400)


class CompactionResult(BaseModel):
    messages: List[Dict[str, Any]]
    levels_applied: List[str] = Field(default_factory=list)
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    persisted_outputs: List[str] = Field(default_factory=list)
    transcript_path: str = ""
    summary: Optional[CompactionSummary] = None


class ContextCompressor:
    """Apply deterministic L3/L1/L2 first and use an LLM only for L4."""

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        base_dir: Optional[Path | str] = None,
    ):
        self.config = config or CompactionConfig()
        self.base_dir = Path(base_dir or os.getenv("CONVERSATION_DATA_DIR", "."))

    async def compress(
        self,
        messages: List[Dict[str, Any]],
        session: SessionContext,
        *,
        llm_client=None,
        emergency: bool = False,
        allow_llm: bool = True,
    ) -> CompactionResult:
        original = [dict(message) for message in messages]
        working = [dict(message) for message in messages]
        levels: List[str] = []
        persisted: List[str] = []
        before = self.estimate_tokens(working)

        working, l3_paths, l3_changed = self._l3_tool_result_budget(
            working, session.session_id, force=emergency
        )
        if l3_changed:
            levels.append("L3_tool_result_budget")
            persisted.extend(l3_paths)

        working, l1_changed = self._l1_snip_compact(working, force=emergency)
        if l1_changed:
            levels.append("L1_snip_compact")

        working, l2_changed = self._l2_micro_compact(working, force=emergency)
        if l2_changed:
            levels.append("L2_micro_compact")

        transcript_path = ""
        summary = None
        eligible = (
            emergency
            or (
                session.turn_count >= self.config.l4_min_turns
                and session.turn_count - session.last_compaction_turn >= self.config.l4_turn_gap
                and self.estimate_tokens(working) > self.config.l4_token_threshold
            )
        )
        if eligible:
            transcript_path = self._persist_transcript(session.session_id, original)
            summary = await self._l4_summary(
                working,
                session=session,
                llm_client=(llm_client or get_llm_client()) if allow_llm else None,
            )
            working = [{
                "role": "user",
                "content": "<conversation_summary>\n" + summary.summary + "\n</conversation_summary>",
            }]
            levels.append("Emergency_reactive_compact" if emergency else "L4_compact_summary")

        return CompactionResult(
            messages=working,
            levels_applied=levels,
            estimated_tokens_before=before,
            estimated_tokens_after=self.estimate_tokens(working),
            persisted_outputs=persisted,
            transcript_path=transcript_path,
            summary=summary,
        )

    async def reactive_compact(
        self,
        messages: List[Dict[str, Any]],
        session: SessionContext,
        *,
        llm_client=None,
        allow_llm: bool = True,
    ) -> CompactionResult:
        return await self.compress(
            messages, session, llm_client=llm_client, emergency=True, allow_llm=allow_llm
        )

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        # A provider-independent conservative approximation for Chinese/English mixtures.
        chars = sum(len(ContextCompressor._content(message)) for message in messages)
        return max(0, (chars + 2) // 3 + len(messages) * 4)

    def _l3_tool_result_budget(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        *,
        force: bool,
    ) -> tuple[List[Dict[str, Any]], List[str], bool]:
        tool_indexes = [index for index, item in enumerate(messages) if self._is_tool_result(item)]
        total = sum(len(self._content(messages[index]).encode("utf-8")) for index in tool_indexes)
        if not force and total <= self.config.tool_result_budget_bytes:
            return messages, [], False

        output_dir = self.base_dir / ".task_outputs" / "sessions" / session_id
        paths: List[str] = []
        changed = False
        for sequence, index in enumerate(tool_indexes, start=1):
            content = self._content(messages[index])
            size = len(content.encode("utf-8"))
            if size <= self.config.single_tool_result_bytes and not force:
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"tool_result_{int(time.time() * 1000)}_{sequence}.txt"
            path.write_text(content, encoding="utf-8")
            preview = content[: self.config.persisted_preview_chars]
            messages[index] = {
                **messages[index],
                "content": (
                    f"<persisted-output path=\"{path}\" bytes=\"{size}\">\n"
                    f"{preview}\n</persisted-output>"
                ),
            }
            paths.append(str(path))
            changed = True
        return messages, paths, changed

    def _l1_snip_compact(
        self,
        messages: List[Dict[str, Any]],
        *,
        force: bool,
    ) -> tuple[List[Dict[str, Any]], bool]:
        if not force and len(messages) <= self.config.max_messages:
            return messages, False
        if len(messages) <= self.config.keep_head_messages + self.config.keep_tail_messages:
            return messages, False

        units = self._message_units(messages)
        head_target = self.config.keep_head_messages
        tail_target = self.config.keep_tail_messages
        head: List[Dict[str, Any]] = []
        tail: List[Dict[str, Any]] = []
        head_units = 0
        for unit in units:
            if len(head) + len(unit) > head_target:
                break
            head.extend(unit)
            head_units += 1
        tail_units = len(units)
        for unit in reversed(units[head_units:]):
            if len(tail) + len(unit) > tail_target:
                break
            tail = unit + tail
            tail_units -= 1
        removed = sum(len(unit) for unit in units[head_units:tail_units])
        if removed <= 0:
            return messages, False
        marker = {"role": "system", "content": f"[snipped {removed} messages]"}
        return head + [marker] + tail, True

    def _l2_micro_compact(
        self,
        messages: List[Dict[str, Any]],
        *,
        force: bool,
    ) -> tuple[List[Dict[str, Any]], bool]:
        indexes = [index for index, item in enumerate(messages) if self._is_tool_result(item)]
        if not force and len(indexes) <= self.config.recent_tool_results:
            return messages, False
        old = indexes[:-self.config.recent_tool_results] if self.config.recent_tool_results else indexes
        changed = False
        for index in old:
            content = self._content(messages[index])
            if len(content) <= 120:
                continue
            messages[index] = {
                **messages[index],
                "content": "[Earlier tool result compacted; use persisted output or rerun if needed.]",
            }
            changed = True
        return messages, changed

    async def _l4_summary(
        self,
        messages: List[Dict[str, Any]],
        *,
        session: SessionContext,
        llm_client,
    ) -> CompactionSummary:
        transcript = "\n".join(
            f"{item.get('role', 'unknown')}: {self._content(item)}" for item in messages
        )
        if llm_client is None:
            return self._fallback_summary(messages, session)
        result = await llm_client.generate_structured(
            system_prompt=_COMPACTION_SYSTEM,
            user_prompt=(
                f"Existing summary:\n{session.summary_so_far}\n\n"
                f"Conversation:\n{transcript[:120_000]}"
            ),
            output_schema=CompactionSummary,
            temperature=0.0,
        )
        if result.get("success"):
            data = result["data"]
            if data.summary.strip():
                return data
        return self._fallback_summary(messages, session)

    def _persist_transcript(self, session_id: str, messages: List[Dict[str, Any]]) -> str:
        directory = self.base_dir / ".transcripts"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{session_id}_{int(time.time() * 1000)}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in messages:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return str(path)

    @staticmethod
    def _fallback_summary(
        messages: List[Dict[str, Any]], session: SessionContext
    ) -> CompactionSummary:
        recent = [ContextCompressor._content(item) for item in messages[-8:]]
        text = "\n".join(part[:500] for part in recent if part).strip()
        combined = "\n".join(part for part in [session.summary_so_far, text] if part).strip()
        return CompactionSummary(
            topic=session.last_intent,
            key_papers_discussed=session.last_mentioned_paper_ids[:10],
            report_progress=(
                "Active report: " + session.active_report_id
                if session.active_report_id else ""
            ),
            summary=combined[-2400:] or "No durable conversation details were available.",
        )

    @staticmethod
    def _content(message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    @staticmethod
    def _is_tool_result(message: Dict[str, Any]) -> bool:
        return message.get("role") == "tool" or message.get("type") in {
            "tool_result", "function_result"
        }

    @classmethod
    def _message_units(cls, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group an assistant tool request with all immediately following results."""
        units: List[List[Dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            current = messages[index]
            unit = [current]
            has_call = bool(current.get("tool_calls")) or current.get("type") == "tool_use"
            cursor = index + 1
            if has_call:
                while cursor < len(messages) and cls._is_tool_result(messages[cursor]):
                    unit.append(messages[cursor])
                    cursor += 1
            units.append(unit)
            index = cursor if has_call else index + 1
        return units


_COMPACTION_SYSTEM = """You compact an academic research conversation without losing state.
Return JSON with topic, key_papers_discussed, key_findings, pending_questions,
user_constraints, report_progress, and summary. Preserve exact paper/report IDs,
decisions, evidence limitations, and unresolved questions. The summary must be no more
than 300 words. Never introduce facts not present in the transcript."""


context_compressor = ContextCompressor()
