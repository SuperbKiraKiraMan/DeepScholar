"""Adaptive, auditable selection of papers for full evidence analysis."""

import asyncio
import math
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from app.llm.client import get_llm_client
from app.llm.prompts import SOURCE_SELECTION_SYSTEM, SOURCE_SELECTION_USER
from app.llm.schemas import LLMSourceSelectionOutput


class AdaptiveSourceSelector:
    """Let the model choose corpus size, then enforce identifiers and coverage in code."""

    _FACET_CUES = OrderedDict((
        ("methods", ("method", "model", "framework", "approach", "architecture", "fusion", "alignment")),
        ("datasets", ("dataset", "benchmark", "corpus", "dbp15k", "dwy100k", "evaluation")),
        ("results", ("experiment", "result", "outperform", "hits@", "mrr", "accuracy")),
        ("limitations", ("limitation", "challenge", "robust", "missing", "noise", "bias", "scalab")),
        ("reviews", ("survey", "review", "taxonomy", "overview")),
        ("recent", ("llm", "large language model", "pretrain", "foundation model")),
    ))

    async def select(
        self,
        topic: str,
        ranked_sources: List[Dict[str, Any]],
        requested_count: int,
        agent_mode: str,
        allow_rule_fallback: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not ranked_sources:
            return [], {
                "mode": "rule", "analysis_count": 0, "candidate_count": 0,
                "selection_reasons": {}, "coverage_plan": {},
                "rationale": "No candidates were available.",
            }

        configured_guard = max(0, int(os.getenv("MAX_ANALYSIS_SOURCES", "0") or 0))
        # There is no product-level fixed analysis ceiling.  By default the model
        # may choose anywhere in the discovered candidate pool.  Operators can set
        # MAX_ANALYSIS_SOURCES as an explicit resource guard for a deployment.
        available_max = (
            min(len(ranked_sources), configured_guard)
            if configured_guard else len(ranked_sources)
        )
        requested = max(1, min(int(requested_count or 1), available_max))
        comprehensive = self._is_comprehensive_request(topic)
        minimum = min(available_max, max(requested, 12) if comprehensive else requested)
        suggested = min(
            available_max,
            max(minimum, int(math.ceil(math.sqrt(len(ranked_sources)) * 2)))
            if comprehensive else minimum,
        )
        fallback_ids, fallback_plan = self._stratified_ids(ranked_sources, suggested)
        by_id = {
            str(source.get("source_id")): source
            for source in ranked_sources if source.get("source_id")
        }

        result: Dict[str, Any] = {}
        # When the discovered pool already fits inside the user's preference,
        # there is no selection decision for an LLM to make.  Retain the whole
        # pool explicitly instead of entering the failure/fallback branch.
        if len(ranked_sources) <= requested:
            selected_ids = list(by_id)
            return [by_id[source_id] for source_id in selected_ids], {
                "mode": "all_candidates",
                "analysis_count": len(selected_ids),
                "candidate_count": len(ranked_sources),
                "requested_count": requested,
                "minimum_count": minimum,
                "available_maximum": available_max,
                "configured_resource_guard": configured_guard or None,
                "selected_source_ids": selected_ids,
                "selection_reasons": {
                    source_id: "Entire eligible candidate pool retained; no ranking cutoff was required."
                    for source_id in selected_ids
                },
                "coverage_plan": fallback_plan,
                "rationale": "All eligible candidates fit within the requested analysis budget.",
                "latency_ms": 0,
                "model": "",
                "usage": {},
            }
        if agent_mode == "llm" and len(ranked_sources) > requested:
            candidate_summary = "\n".join(
                self._candidate_line(index, source)
                for index, source in enumerate(ranked_sources[:80], start=1)
            )
            # Selecting from dozens of compact paper summaries is a larger
            # structured-output request than planner/tool decisions.  Give it
            # its own 90s budget instead of inheriting the old 45s ceiling.
            wall_timeout = max(10, int(os.getenv("LLM_SOURCE_SELECTION_TIMEOUT_SECONDS", "90")))
            attempts = max(1, int(os.getenv("LLM_SOURCE_SELECTION_ATTEMPTS", "2")))
            user_prompt = SOURCE_SELECTION_USER.format(
                topic=topic,
                candidate_count=len(ranked_sources),
                requested_count=requested,
                minimum_count=minimum,
                maximum_count=available_max,
                candidate_summary=candidate_summary,
            )
            for attempt in range(attempts):
                try:
                    result = await asyncio.wait_for(
                        get_llm_client().generate_structured(
                            system_prompt=SOURCE_SELECTION_SYSTEM,
                            user_prompt=user_prompt + (
                                "\nThis is a retry. Return only the exact requested JSON shape."
                                if attempt else ""
                            ),
                            output_schema=LLMSourceSelectionOutput,
                            timeout_seconds=wall_timeout,
                            max_retries=0,
                        ),
                        timeout=wall_timeout + 2,
                    )
                except asyncio.TimeoutError:
                    result = {
                        "success": False,
                        "error": f"Source selection timeout after {wall_timeout}s (attempt {attempt + 1}/{attempts})",
                        "latency_ms": wall_timeout * 1000,
                    }
                if result.get("success"):
                    break

        if result.get("success"):
            output: LLMSourceSelectionOutput = result["data"]
            count = max(minimum, min(int(output.analysis_count), available_max))
            selected_ids = []
            for source_id in output.selected_source_ids:
                source_id = str(source_id)
                if source_id in by_id and source_id not in selected_ids:
                    selected_ids.append(source_id)
                if len(selected_ids) >= count:
                    break
            for source_id in fallback_ids:
                if len(selected_ids) >= count:
                    break
                if source_id not in selected_ids:
                    selected_ids.append(source_id)
            reasons = {
                str(source_id): str(reason)
                for source_id, reason in output.selection_reasons.items()
                if str(source_id) in selected_ids
            }
            for source_id in selected_ids:
                reasons.setdefault(
                    source_id,
                    str(by_id[source_id].get("task_relevance_reason") or "Selected to complete facet coverage."),
                )
            selection = {
                "mode": "llm",
                "analysis_count": len(selected_ids),
                "candidate_count": len(ranked_sources),
                "requested_count": requested,
                "minimum_count": minimum,
                "available_maximum": available_max,
                "configured_resource_guard": configured_guard or None,
                "selected_source_ids": selected_ids,
                "selection_reasons": reasons,
                "coverage_plan": {
                    str(facet): [str(item) for item in ids if str(item) in selected_ids]
                    for facet, ids in (output.coverage_plan or fallback_plan).items()
                },
                "rationale": output.rationale,
                "latency_ms": result.get("latency_ms", 0),
                "model": result.get("model", ""),
                "usage": result.get("usage", {}),
            }
        else:
            if agent_mode == "llm" and not allow_rule_fallback:
                raise RuntimeError(
                    "LLM-only source selection failed after retries: "
                    + str(result.get("error") or "unknown error")
                )
            selected_ids = fallback_ids
            selection = {
                "mode": "rule",
                "analysis_count": len(selected_ids),
                "candidate_count": len(ranked_sources),
                "requested_count": requested,
                "minimum_count": minimum,
                "available_maximum": available_max,
                "configured_resource_guard": configured_guard or None,
                "selected_source_ids": selected_ids,
                "selection_reasons": {
                    source_id: str(
                        by_id[source_id].get("task_relevance_reason")
                        or "Deterministic stratified coverage selection"
                    )
                    for source_id in selected_ids
                },
                "coverage_plan": fallback_plan,
                "rationale": (
                    "Model selection was unavailable; the deterministic selector used "
                    "topic facets, evidence availability, influence, recency, and source diversity."
                ),
                "error": result.get("error", "") if result else "",
                "latency_ms": result.get("latency_ms", 0) if result else 0,
            }

        return [by_id[source_id] for source_id in selection["selected_source_ids"]], selection

    def _stratified_ids(
        self, sources: List[Dict[str, Any]], count: int,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        buckets: "OrderedDict[str, List[str]]" = OrderedDict(
            (facet, []) for facet in self._FACET_CUES
        )
        buckets["full_text"] = []
        buckets["influential"] = []
        for source in sources:
            source_id = str(source.get("source_id") or "")
            if not source_id:
                continue
            text = " ".join((
                str(source.get("title") or ""),
                str(source.get("snippet") or source.get("abstract") or ""),
            )).lower()
            for facet, cues in self._FACET_CUES.items():
                if any(cue in text for cue in cues):
                    buckets[facet].append(source_id)
            if str(source.get("full_text") or source.get("abstract") or "").strip():
                buckets["full_text"].append(source_id)
            if int(source.get("cited_by_count") or 0) > 20:
                buckets["influential"].append(source_id)

        selected: List[str] = []
        positions = {key: 0 for key in buckets}
        while len(selected) < count:
            added = False
            for facet, ids in buckets.items():
                while positions[facet] < len(ids) and ids[positions[facet]] in selected:
                    positions[facet] += 1
                if positions[facet] < len(ids) and len(selected) < count:
                    selected.append(ids[positions[facet]])
                    positions[facet] += 1
                    added = True
            if not added:
                break
        for source in sources:
            source_id = str(source.get("source_id") or "")
            if source_id and source_id not in selected:
                selected.append(source_id)
            if len(selected) >= count:
                break
        plan = {
            facet: [source_id for source_id in ids if source_id in selected]
            for facet, ids in buckets.items() if ids
        }
        return selected[:count], plan

    @classmethod
    def _is_comprehensive_request(cls, topic: str) -> bool:
        text = str(topic or "").lower()
        is_mmea = any(cue in text for cue in (
            "mmea", "multimodal entity alignment", "多模态实体对齐",
        ))
        asks_methods = any(cue in text for cue in ("主要方法", "methods", "method taxonomy"))
        asks_datasets = any(cue in text for cue in ("数据集", "datasets", "benchmarks"))
        asks_limits = any(cue in text for cue in ("研究局限", "limitations", "open problems"))
        return is_mmea or (asks_methods and asks_datasets and asks_limits)

    @staticmethod
    def _candidate_line(index: int, source: Dict[str, Any]) -> str:
        title = str(source.get("title") or "Untitled").replace("\n", " ")[:220]
        text = " ".join((
            title,
            str(source.get("abstract") or source.get("snippet") or "")[:500],
        )).lower()
        facets = ",".join(
            facet for facet, cues in AdaptiveSourceSelector._FACET_CUES.items()
            if any(cue in text for cue in cues)
        ) or "general"
        return (
            f"{index}. source_id={source.get('source_id', '')}; title={title}; "
            f"year={source.get('year', 'N/A')}; cited_by={source.get('cited_by_count', 0)}; "
            f"type={source.get('source_type', 'unknown')}; provider={source.get('provider', '')}; "
            f"has_text={bool(source.get('full_text') or source.get('abstract'))}; facets={facets}"
        )
