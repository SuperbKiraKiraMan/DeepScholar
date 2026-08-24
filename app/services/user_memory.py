"""File-backed durable user memory with index, retrieval, extraction and consolidation."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm.client import get_llm_client
from app.services.session_store import SessionContext


MEMORY_TYPES = frozenset({"user", "feedback", "project", "reference"})


class MemoryEntry(BaseModel):
    name: str = Field(default_factory=lambda: f"memory-{uuid.uuid4().hex[:8]}")
    memory_type: str = "user"
    title: str
    body: str
    tags: List[str] = Field(default_factory=list)
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault("memory_type", data.get("type") or "user")
        data.setdefault("title", data.get("description") or data.get("name") or "Memory")
        data.setdefault("body", data.get("content") or data.get("description") or "")
        return data


class MemorySelection(BaseModel):
    names: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault(
            "names",
            data.get("selected_names") or data.get("selected_memory_names")
            or data.get("relevant_memories") or data.get("memories") or [],
        )
        if isinstance(data["names"], str):
            data["names"] = [data["names"]]
        if isinstance(data["names"], list):
            data["names"] = [
                (str(item.get("name") or "") if isinstance(item, dict) else str(item)).strip(" `")
                for item in data["names"]
            ]
        return data


class MemoryExtraction(BaseModel):
    entries: List[MemoryEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_shape(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.setdefault(
            "entries",
            data.get("new_memories") or data.get("memories") or data.get("consolidated_memories") or [],
        )
        return data


class UserMemoryStore:
    """MEMORY.md is a cheap always-on index; individual files load on demand."""

    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root or os.getenv("USER_MEMORY_DIR", ".memory"))
        self.index_path = self.root / "MEMORY.md"
        self.last_selection_result: Dict[str, Any] = {}

    def write(self, entry: MemoryEntry | Dict[str, Any]) -> MemoryEntry:
        value = entry if isinstance(entry, MemoryEntry) else MemoryEntry(**entry)
        if value.memory_type not in MEMORY_TYPES:
            raise ValueError(f"Unsupported memory type: {value.memory_type}")
        value.name = self._safe_name(value.name)
        existing = self._read_path(self.root / f"{value.name}.md")
        if existing:
            value.created_at_ms = existing.created_at_ms
        value.updated_at_ms = int(time.time() * 1000)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{value.name}.md").write_text(
            self._serialize(value), encoding="utf-8"
        )
        self._rebuild_index()
        return value

    def load_index(self) -> str:
        if not self.index_path.exists():
            return ""
        return "\n".join(self.index_path.read_text(encoding="utf-8").splitlines()[:100])

    async def load_relevant(
        self,
        query: str,
        llm_client=None,
        *,
        limit: int = 3,
        use_llm: bool = True,
    ) -> List[MemoryEntry]:
        entries = self._all_entries()
        if not entries:
            return []
        index = self.load_index()
        result = {"success": False, "error": "LLM memory selection disabled"}
        if use_llm:
            client = llm_client or get_llm_client()
            result = await client.generate_structured(
                system_prompt=(
                    "Select at most three memory names relevant to the current request. "
                    "Prefer durable preferences, feedback, and project constraints. "
                    "Return only names that appear in the index as JSON {\"names\": [\"exact-name\"]}."
                ),
                user_prompt=f"Request: {query}\n\nMemory index:\n{index}",
                output_schema=MemorySelection,
                temperature=0.0,
            )
        self.last_selection_result = {
            key: value for key, value in result.items() if key != "data"
        }
        if result.get("success"):
            self.last_selection_result["selected_names"] = result["data"].names
        by_name = {entry.name: entry for entry in entries}
        if result.get("success"):
            selected = [by_name[name] for name in result["data"].names if name in by_name]
            if selected:
                return selected[:limit]

        query_terms = self._terms(query)
        ranked = sorted(
            entries,
            key=lambda item: self._jaccard(
                query_terms, self._terms(" ".join([item.title, item.body, *item.tags]))
            ),
            reverse=True,
        )
        return [
            item for item in ranked
            if self._jaccard(
                query_terms, self._terms(" ".join([item.title, item.body, *item.tags]))
            ) > 0
        ][:limit]

    async def extract(
        self,
        session: SessionContext,
        *,
        llm_client=None,
    ) -> List[MemoryEntry]:
        if session.turn_count < 3 or not session.recent_messages:
            return []
        transcript = "\n".join(
            f"{message.role}: {message.content_preview}"
            for message in session.recent_messages[-10:]
        )
        client = llm_client or get_llm_client()
        result = await client.generate_structured(
            system_prompt=_MEMORY_EXTRACT_SYSTEM,
            user_prompt=transcript,
            output_schema=MemoryExtraction,
            temperature=0.0,
        )
        if not result.get("success"):
            return []
        existing = self._all_entries()
        stored: List[MemoryEntry] = []
        for entry in result["data"].entries:
            if entry.memory_type not in MEMORY_TYPES or self._looks_transient(entry.body):
                continue
            candidate_terms = self._terms(entry.title + " " + entry.body)
            duplicate = any(
                self._jaccard(
                    candidate_terms, self._terms(item.title + " " + item.body)
                ) > 0.6
                for item in [*existing, *stored]
            )
            if not duplicate:
                stored.append(self.write(entry))
        return stored

    async def consolidate(
        self,
        session: SessionContext,
        *,
        llm_client=None,
    ) -> List[MemoryEntry]:
        entries = self._all_entries()
        if len(entries) < 10 or session.turn_count - session.last_consolidation_turn < 5:
            return entries

        client = llm_client or get_llm_client()
        compact = [entry.model_dump() for entry in entries]
        result = await client.generate_structured(
            system_prompt=_MEMORY_CONSOLIDATE_SYSTEM,
            user_prompt=json.dumps(compact, ensure_ascii=False)[:100_000],
            output_schema=MemoryExtraction,
            temperature=0.0,
        )
        if result.get("success") and result["data"].entries:
            consolidated = result["data"].entries[:30]
        else:
            consolidated = self._deterministic_consolidate(entries)[:30]

        keep_names = {self._safe_name(entry.name) for entry in consolidated}
        for path in self.root.glob("*.md") if self.root.exists() else []:
            if path.name == "MEMORY.md":
                continue
            if path.stem not in keep_names:
                path.unlink(missing_ok=True)
        for entry in consolidated:
            self.write(entry)
        self._rebuild_index()
        return self._all_entries()

    def build_prompt(self, relevant: Iterable[MemoryEntry] = ()) -> str:
        index = self.load_index()
        details = "\n\n".join(
            f"## {entry.title}\nType: {entry.memory_type}\n{entry.body}"
            for entry in relevant
        )
        parts = []
        if index:
            parts.append(f"<memory_index>\n{index}\n</memory_index>")
        if details:
            parts.append(f"<relevant_memories>\n{details}\n</relevant_memories>")
        return "\n\n".join(parts)

    def _all_entries(self) -> List[MemoryEntry]:
        if not self.root.exists():
            return []
        values = [
            value for path in sorted(self.root.glob("*.md"))
            if path.name != "MEMORY.md" and (value := self._read_path(path)) is not None
        ]
        return values

    def _rebuild_index(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entries = sorted(
            self._all_entries(),
            key=lambda item: (item.memory_type != "user", -item.updated_at_ms),
        )[:99]
        lines = ["# User Memory Index"]
        for entry in entries:
            tags = ", ".join(entry.tags[:5])
            lines.append(
                f"- `{entry.name}` [{entry.memory_type}] {entry.title}"
                + (f" — {tags}" if tags else "")
            )
        self.index_path.write_text("\n".join(lines[:100]) + "\n", encoding="utf-8")

    @staticmethod
    def _serialize(entry: MemoryEntry) -> str:
        metadata = {
            "name": entry.name,
            "type": entry.memory_type,
            "title": entry.title,
            "tags": entry.tags,
            "created_at_ms": entry.created_at_ms,
            "updated_at_ms": entry.updated_at_ms,
        }
        return "---\n" + "\n".join(
            f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items()
        ) + "\n---\n\n" + entry.body.strip() + "\n"

    @staticmethod
    def _read_path(path: Path) -> Optional[MemoryEntry]:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n\n?(.*)$", text, re.S)
        if not match:
            return None
        metadata: Dict[str, Any] = {}
        for line in match.group(1).splitlines():
            key, separator, raw = line.partition(":")
            if not separator:
                continue
            try:
                metadata[key.strip()] = json.loads(raw.strip())
            except json.JSONDecodeError:
                metadata[key.strip()] = raw.strip()
        try:
            return MemoryEntry(
                name=metadata.get("name") or path.stem,
                memory_type=metadata.get("type") or "user",
                title=metadata.get("title") or path.stem,
                tags=metadata.get("tags") or [],
                created_at_ms=int(metadata.get("created_at_ms") or 0),
                updated_at_ms=int(metadata.get("updated_at_ms") or 0),
                body=match.group(2).strip(),
            )
        except Exception:
            return None

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name).strip()).strip("-")
        return safe[:80] or f"memory-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_+-]{2,}|[\u4e00-\u9fff]", text.lower()))

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left | right))

    @staticmethod
    def _looks_transient(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in (
            "用户询问了论文", "用户问了论文", "asked about paper",
            "本轮", "this turn", "current question",
        ))

    def _deterministic_consolidate(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        ordered = sorted(
            entries,
            key=lambda item: (item.memory_type != "user", -item.updated_at_ms),
        )
        kept: List[MemoryEntry] = []
        for entry in ordered:
            terms = self._terms(entry.title + " " + entry.body)
            if any(
                self._jaccard(terms, self._terms(item.title + " " + item.body)) > 0.6
                for item in kept
            ):
                continue
            kept.append(entry)
        return kept


_MEMORY_EXTRACT_SYSTEM = """Extract only durable memories that will improve future research conversations.
Allowed types: user (stable preferences), feedback (how the assistant should behave),
project (long-running research direction), reference (durable pointer). Do not store a
paper merely because it was discussed, current task state, one-off questions, secrets,
or guesses. Return JSON {"entries": [...]} with name, memory_type, title, body, tags."""

_MEMORY_CONSOLIDATE_SYSTEM = """Consolidate durable user memories. Merge duplicates,
resolve contradictions in favor of newer explicit feedback, remove obsolete/transient
items, preserve user preferences first, and return no more than 30 entries as JSON
{"entries": [...]}. Never add facts."""


user_memory_store = UserMemoryStore()
