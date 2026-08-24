"""Read-only catalog views for papers, evidence, and completed reports."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.services.run_store import RunStore, run_store


class WorkspaceCatalog:
    """Aggregate persisted research artifacts into UI-friendly collections."""

    def __init__(self, store: RunStore = run_store, local_index_path: str | None = None):
        self.store = store
        self.local_index_path = local_index_path or os.getenv(
            "LOCAL_PAPER_INDEX_PATH", "data/local_paper_index.sqlite3"
        )

    def papers(self, query: str = "", origin: str = "all") -> Dict[str, Any]:
        local = self._local_papers()
        searched = self._searched_papers(self._completed_runs())
        merged: Dict[str, Dict[str, Any]] = {}
        for item in [*local, *searched]:
            key = self._paper_key(item)
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["origins"] = list(item.get("origins") or [])
                merged[key]["seen_in_runs"] = list(item.get("seen_in_runs") or [])
                continue
            current = merged[key]
            for field, value in item.items():
                if field not in {"origins", "seen_in_runs"} and not current.get(field) and value:
                    current[field] = value
            current["origins"] = list(dict.fromkeys([
                *current.get("origins", []), *item.get("origins", [])
            ]))
            current["seen_in_runs"] = list(dict.fromkeys([
                *current.get("seen_in_runs", []), *item.get("seen_in_runs", [])
            ]))

        items = list(merged.values())
        if origin in {"local", "searched"}:
            items = [item for item in items if origin in item.get("origins", [])]
        needle = query.strip().casefold()
        if needle:
            items = [item for item in items if needle in " ".join([
                str(item.get("title") or ""),
                " ".join(item.get("authors") or []),
                str(item.get("doi") or ""),
                str(item.get("provider") or ""),
            ]).casefold()]
        items.sort(key=lambda item: (
            "local" not in item.get("origins", []),
            -(int(item.get("year") or 0)),
            str(item.get("title") or "").casefold(),
        ))
        return {
            "items": items,
            "total": len(items),
            "local_count": len(local),
            "searched_count": len(searched),
        }

    def evidence(self, query: str = "") -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        needle = query.strip().casefold()
        for run in self._completed_runs():
            sources = {
                str(item.get("source_id") or ""): item
                for item in run.get("sources") or [] if isinstance(item, dict)
            }
            for card in run.get("evidence_cards") or []:
                if not isinstance(card, dict):
                    continue
                source = sources.get(str(card.get("source_id") or ""), {})
                item = {
                    **card,
                    "run_id": run.get("run_id", ""),
                    "run_topic": run.get("topic", ""),
                    "created_at": run.get("created_at", ""),
                    "source_title": source.get("title") or "Unknown source",
                    "source_url": card.get("url") or source.get("url") or "",
                    "source_provider": source.get("provider") or source.get("source_type") or "",
                    "source_year": source.get("year"),
                    "source_authors": source.get("authors") or [],
                }
                haystack = " ".join(str(item.get(field) or "") for field in (
                    "claim", "quote", "source_title", "run_topic", "method", "dataset", "limitation"
                )).casefold()
                if not needle or needle in haystack:
                    items.append(item)
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"items": items, "total": len(items)}

    def reports(self, query: str = "") -> Dict[str, Any]:
        needle = query.strip().casefold()
        items = []
        for run in self._completed_runs():
            report = str(run.get("final_report") or run.get("draft_report") or "").strip()
            if not report:
                continue
            topic = str(run.get("topic") or "Untitled report")
            if needle and needle not in (topic + " " + report).casefold():
                continue
            items.append({
                "run_id": run.get("run_id", ""),
                "topic": topic,
                "status": run.get("status", ""),
                "created_at": run.get("created_at", ""),
                "finished_at": run.get("finished_at"),
                "source_count": len(run.get("sources") or []),
                "evidence_count": len(run.get("evidence_cards") or []),
                "total_latency_ms": run.get("total_latency_ms") or run.get("latency_ms") or 0,
                "report": report,
                "outline": run.get("outline") or {},
            })
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"items": items, "total": len(items)}

    def _completed_runs(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        offset = 0
        while offset < 500:
            page, total = self.store.list_runs_page(limit=100, offset=offset)
            summaries.extend(page)
            offset += len(page)
            if not page or offset >= total:
                break
        return [
            detail for summary in summaries
            if (detail := self.store.get(str(summary.get("run_id") or ""))) is not None
        ]

    def _local_papers(self) -> List[Dict[str, Any]]:
        path = Path(self.local_index_path)
        if not path.exists():
            return []
        try:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT d.*, COUNT(c.chunk_id) AS chunk_count
                FROM local_paper_documents d
                LEFT JOIN local_paper_chunks c ON c.source_path = d.source_path
                GROUP BY d.source_path
                ORDER BY d.indexed_at DESC
                """
            ).fetchall()
            connection.close()
        except (sqlite3.Error, OSError):
            return []
        items = []
        for row in rows:
            try:
                authors = json.loads(row["authors_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                authors = []
            items.append({
                "paper_id": row["paper_id"], "source_id": row["paper_id"],
                "title": row["title"], "authors": authors, "year": row["year"],
                "doi": row["doi"], "source_path": row["source_path"],
                "zotero_storage_key": row["zotero_storage_key"],
                "size_bytes": row["size_bytes"], "indexed_at": row["indexed_at"],
                "chunk_count": row["chunk_count"], "provider": "local_zotero",
                "source_type": "local", "origins": ["local"], "seen_in_runs": [],
            })
        return items

    @staticmethod
    def _searched_papers(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            run_id = str(run.get("run_id") or "")
            for source in run.get("sources") or []:
                if not isinstance(source, dict):
                    continue
                item = dict(source)
                item.pop("full_text", None)
                item["origins"] = ["searched"]
                item["seen_in_runs"] = [run_id] if run_id else []
                key = WorkspaceCatalog._paper_key(item)
                if key in merged:
                    merged[key]["seen_in_runs"] = list(dict.fromkeys([
                        *merged[key].get("seen_in_runs", []), *item["seen_in_runs"]
                    ]))
                else:
                    merged[key] = item
        return list(merged.values())

    @staticmethod
    def _paper_key(item: Dict[str, Any]) -> str:
        return str(
            item.get("doi") or item.get("semantic_scholar_id") or item.get("openalex_id")
            or item.get("paper_id") or item.get("source_id") or item.get("title") or "unknown"
        ).strip().casefold()


workspace_catalog = WorkspaceCatalog()
