"""
app/agents/evaluator.py

Evaluator —— 规则评估器。12 项 Rule Metrics，全部确定性。

稳定性与质量门：
- task_success_rate: computed from worker_started/worker_finished trace events
- tool_error_rate: only counts actual tool execution events (not graph nodes)
- Configurable thresholds via class attributes
"""

import re
from typing import Any, Dict, List, Set, Tuple

from app.core.config import get_research_latency_ttl_ms


# Events that are NOT tool executions
_NON_TOOL_EVENTS: Set[str] = {
    "controller_start", "planner_complete", "send_dispatch",
    "merge_result", "search_complete", "reading_complete",
    "analysis_complete", "citation_complete", "outline_created", "chapter_generated", "draft_reviewer_complete",
    "evaluator_complete", "final_reviewer_complete",
    "llm_started", "llm_finished", "llm_failed", "llm_fallback",
    "function_call_started", "tool_selected", "tool_started",
    "tool_loop_finished", "tool_loop_limit_reached", "tool_loop_fallback",
    "tool_rejected", "tool_args_rejected", "tool_finished",
    "llm_function_call",
    "provider_fallback",
    "retrieval_observed", "retrieval_query_rewritten",
    "retrieval_source_switched", "retrieval_finished",
}
# Worker-level outcome events
_WORKER_EVENTS: Set[str] = {"worker_started", "worker_finished"}
_PSEUDO_TOOL_NAMES: Set[str] = {
    "", "orchestrator", "function_call_started", "tool_selected", "tool_started",
    "tool_finished", "tool_args_rejected", "tool_loop_finished",
    "tool_loop_limit_reached", "tool_loop_fallback", "tool_rejected",
    "llm_finish", "llm_function_call",
    "provider_fallback",
    "retrieval_observed", "retrieval_query_rewritten",
    "retrieval_source_switched", "retrieval_finished",
}


def _is_real_tool_completion(event: Dict[str, Any]) -> bool:
    """Identify one completed business-tool call in loop or graph trace format."""
    tool_name = str(event.get("tool_name", ""))
    if tool_name in _PSEUDO_TOOL_NAMES or tool_name.startswith(("tool_", "llm_")):
        return False
    event_type = event.get("event", "")
    if event_type == "tool_finished":
        return True
    return not event_type and "success" in event


class Evaluator:
    """规则评估器。Thresholds are class-level for fixture override."""

    MIN_SOURCES: int = 1
    MAX_TOOL_ERROR_RATE: float = 0.5
    # Class attribute remains patchable by the deterministic harness. Production
    # may override it with RESEARCH_LATENCY_TTL_SECONDS.
    MAX_LATENCY_MS: int = 180_000

    def evaluate(
        self,
        topic: str = "",
        draft_report: str = "",
        sources: List[Dict[str, Any]] = None,
        evidence_cards: List[Dict[str, Any]] = None,
        citation_check_results: List[Dict[str, Any]] = None,
        citation_summary: Dict[str, Any] = None,
        trace: List[Dict[str, Any]] = None,
        task_dag: Dict[str, Any] = None,
        total_latency_ms: int = 0,
    ) -> Dict[str, Any]:
        sources = sources or []
        evidence_cards = evidence_cards or []
        citation_check_results = citation_check_results or []
        citation_summary = citation_summary or {}
        trace = trace or []
        task_dag = task_dag or {}

        metrics = {}
        feedback = []

        # Metric 1
        metrics["no_fake_citation"] = self._check_no_fake_citation(citation_summary)
        if not metrics["no_fake_citation"]:
            feedback.append("Fake citations detected.")

        # Metric 2
        metrics["min_sources"] = len(sources) >= self.MIN_SOURCES
        if not metrics["min_sources"]:
            feedback.append(f"Too few sources: {len(sources)} < {self.MIN_SOURCES}.")

        # Metric 3
        metrics["citation_id_exists"] = self._check_citation_ids_exist(citation_check_results)
        if not metrics["citation_id_exists"]:
            feedback.append("Some citation IDs do not exist in source list.")

        # Metric 4
        metrics["source_url_valid"] = self._check_source_urls_valid(citation_check_results)
        if not metrics["source_url_valid"]:
            feedback.append("Some citation URLs do not match any source URL.")

        # Metric 5
        metrics["evidence_available"] = bool(evidence_cards)
        if not metrics["evidence_available"]:
            feedback.append("No evidence cards were extracted; report claims must be suppressed.")

        # Metric 6
        metrics["answer_not_empty"] = bool(draft_report and len(draft_report.strip()) > 50)
        if not metrics["answer_not_empty"]:
            feedback.append("Draft report is empty or too short.")

        # Metric 7 — real data from trace
        tsr = self._calc_task_success_rate(trace, task_dag)
        metrics["task_success_rate"] = tsr["rate"]
        if not tsr["passed"]:
            feedback.append(f"Task success rate: {tsr['rate']:.0%} ({tsr['success']}/{tsr['total']})")

        # Metric 8 — real data from trace
        ter = self._calc_tool_error_rate(trace)
        metrics["tool_error_rate"] = ter["passed"]
        if not ter["passed"]:
            feedback.append(f"Tool error rate: {ter['error_count']}/{ter['total']} = {ter['rate']:.1%}")

        # Metric 9 — end-to-end quality TTL. This does not relax tool/model timeouts.
        latency_threshold_ms = self._latency_threshold_ms()
        metrics["latency_under_threshold"] = total_latency_ms <= latency_threshold_ms
        if not metrics["latency_under_threshold"]:
            feedback.append(f"Latency {total_latency_ms}ms exceeds {latency_threshold_ms}ms.")

        latency_detail = self._calc_latency_detail(
            trace, total_latency_ms, latency_threshold_ms,
        )

        # Metric 10 — citations in conclusions/report must resolve to evidence-bearing sources.
        coverage_ok, coverage_detail = self._check_conclusion_evidence_coverage(
            draft_report, evidence_cards, sources, with_detail=True,
        )
        metrics["conclusion_evidence_coverage"] = coverage_ok
        if not coverage_ok:
            feedback.append("Some report/conclusion citations do not resolve to an Evidence Card.")

        # Metric 11 — adjacent chapters must add new information.
        duplication_ok, duplication_detail = self._check_chapter_duplication(
            draft_report, with_detail=True,
        )
        metrics["chapter_duplication"] = duplication_ok
        if not duplication_ok:
            feedback.append("Adjacent report chapters contain more than 50% bigram overlap.")

        # Metric 12 — long declarative prose must carry an explicit source citation.
        unsupported_ok, unsupported_detail = self._check_unsupported_expansion(
            draft_report, evidence_cards, with_detail=True,
        )
        metrics["unsupported_expansion"] = unsupported_ok
        if not unsupported_ok:
            feedback.append("Long declarative paragraphs without [N] citations were detected.")

        # Reader-facing content acceptance metrics. These complement execution
        # metrics and prevent a structurally valid but substantively empty report
        # from being marked complete.
        content_metrics, content_detail = self._check_content_acceptance(
            topic, draft_report, sources, evidence_cards,
        )
        if content_detail.get("comprehensive_request"):
            metrics.update(content_metrics)
            for name, passed in content_metrics.items():
                if not passed:
                    feedback.append(f"Content acceptance failed: {name}.")

        metric_passes = {
            **{name: bool(value) for name, value in metrics.items()},
            "task_success_rate": tsr["passed"],
        }
        passed_count = sum(1 for passed in metric_passes.values() if passed)
        total_count = len(metric_passes)

        return {
            "metrics": metrics,
            "metrics_detail": {
                "passed_count": passed_count,
                "total_count": total_count,
                "pass_rate": round(passed_count / max(total_count, 1), 4),
                "task_success_rate": tsr,
                "tool_error_rate": ter,
                "latency": latency_detail,
                "conclusion_evidence_coverage": coverage_detail,
                "chapter_duplication": duplication_detail,
                "unsupported_expansion": unsupported_detail,
                "content_acceptance": content_detail,
            },
            "feedback": feedback,
            "all_passed": all(metric_passes.values()),
        }

    def _check_content_acceptance(
        self,
        topic: str,
        report: str,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        lowered = str(report or "").lower()
        topic_text = str(topic or "").lower()
        comprehensive = (
            any(cue in topic_text for cue in ("mmea", "multimodal entity alignment", "多模态实体对齐"))
            or (
                any(cue in topic_text for cue in ("主要方法", "methods", "method taxonomy"))
                and any(cue in topic_text for cue in ("数据集", "datasets", "benchmarks"))
                and any(cue in topic_text for cue in ("研究局限", "limitations", "open problems"))
            )
        )
        section_specs = {
            "methods": ("主要方法", "method taxonomy"),
            "datasets": ("常用数据集", "common datasets"),
            "protocols": ("评估指标与实验协议", "metrics and experimental protocols"),
            "limitations": ("研究局限", "research limitations"),
        }
        sections = {
            name: self._section_text(report, cues)
            for name, cues in section_specs.items()
        }
        coverage = {
            name: len(set(re.findall(r"\[(\d+)\]", text)))
            for name, text in sections.items()
        }
        table_count = len(re.findall(
            r"^\|.+\|\s*$\n^\|[-:|\s]+\|\s*$", report or "", re.MULTILINE,
        ))
        strong_issues = []
        for sentence in re.split(r"(?<=[。！？.!?])\s*", report or ""):
            if not re.search(
                r"(?:首次|首个|显著|优于|突破|first|significant|outperform|breakthrough)",
                sentence, re.IGNORECASE,
            ):
                continue
            if not re.search(r"\[\d+\]", sentence):
                strong_issues.append("missing citation")
            if re.search(r"(?:显著|优于|outperform|significant)", sentence, re.IGNORECASE) and not re.search(r"\d", sentence):
                strong_issues.append("missing numeric context")

        evidence_types = [str(card.get("evidence_type") or "primary_claim") for card in evidence_cards]
        primary_count = sum(item.startswith("primary") for item in evidence_types)
        primary_ratio = primary_count / max(1, len(evidence_types))
        public_sources = [
            source for source in sources
            if str(source.get("url") or "").startswith(("http://", "https://"))
            or source.get("doi") or source.get("openalex_id") or source.get("semantic_scholar_id")
        ]
        reproducible_ratio = len(public_sources) / max(1, len(sources))
        metadata_complete = [
            source for source in sources
            if source.get("authors") and source.get("year")
            and (
                str(source.get("url") or "").startswith(("http://", "https://"))
                or source.get("doi") or source.get("openalex_id") or source.get("semantic_scholar_id")
            )
        ]
        metadata_ratio = len(metadata_complete) / max(1, len(sources))
        internal_terms = (
            "evidence card", "finalreviewer", "draft_reviewer", "agent trace",
            "latency_ms", "latency exceeded threshold", "slowest observed node",
            "规则通过率", "评估结果：", "逐章生成", "证据卡绑定", "总延迟超过阈值",
        )
        facet_source_counts = {
            "methods": len({card.get("source_id") for card in evidence_cards if card.get("method") and card.get("source_id")}),
            "datasets": len({card.get("source_id") for card in evidence_cards if card.get("dataset") and card.get("source_id")}),
            "protocols": len({
                card.get("source_id") for card in evidence_cards
                if card.get("source_id") and (card.get("metric") or card.get("experimental_setting"))
            }),
            "limitations": len({card.get("source_id") for card in evidence_cards if card.get("limitation") and card.get("source_id")}),
        }
        method_families = {
            family.strip()
            for card in evidence_cards
            for family in str(card.get("method_family") or "").split("；")
            if family.strip()
        }
        dataset_detail_sources = {
            card.get("source_id") for card in evidence_cards
            if card.get("source_id") and card.get("dataset") and any(card.get(field) for field in (
                "dataset_name", "graph_or_language_pair", "entity_count", "modalities",
                "missingness", "data_split", "seed_ratio",
            ))
        }
        metrics = {
            "no_citation_placeholders": "[citation needed]" not in lowered,
            "method_taxonomy_complete": (not comprehensive) or bool(
                sections["methods"] and facet_source_counts["methods"] >= 2
                and len(method_families) >= 2
            ),
            "dataset_coverage_complete": (not comprehensive) or bool(
                sections["datasets"] and facet_source_counts["datasets"] >= 2
                and len(dataset_detail_sources) >= 2
            ),
            "evaluation_protocol_complete": (not comprehensive) or bool(sections["protocols"] and facet_source_counts["protocols"] >= 2),
            "domain_limitations_complete": (not comprehensive) or bool(sections["limitations"] and facet_source_counts["limitations"] >= 2),
            "core_question_source_coverage": (not comprehensive) or all(
                coverage[name] >= 2 for name in ("methods", "datasets", "limitations")
            ),
            "comparison_tables_present": (not comprehensive) or table_count >= 2,
            "unsupported_strong_claims": not strong_issues,
            "no_internal_execution_content": not any(term in lowered for term in internal_terms),
            "no_local_file_urls": "file://" not in lowered,
            "primary_source_ratio": primary_ratio >= 0.5 if evidence_types else False,
            "reproducible_citation_ratio": reproducible_ratio >= 0.8 if sources else False,
            "reference_metadata_complete": metadata_ratio >= 0.8 if sources else False,
            "task_boundary_consistent": (not comprehensive) or not any(
                source.get("task_relevance") == "excluded"
                or source.get("research_task") not in (None, "", "multimodal_entity_alignment")
                for source in sources
            ),
        }
        return metrics, {
            "comprehensive_request": comprehensive,
            "section_source_counts": coverage,
            "structured_facet_source_counts": facet_source_counts,
            "method_family_count": len(method_families),
            "dataset_detail_source_count": len(dataset_detail_sources),
            "table_count": table_count,
            "strong_claim_issues": strong_issues,
            "primary_source_ratio": round(primary_ratio, 4),
            "reproducible_citation_ratio": round(reproducible_ratio, 4),
            "reference_metadata_ratio": round(metadata_ratio, 4),
        }

    # ---- Metric helpers ----

    def _latency_threshold_ms(self) -> int:
        """Use an explicit environment TTL when set; otherwise keep harness overrides."""
        import os

        if "RESEARCH_LATENCY_TTL_SECONDS" in os.environ:
            return get_research_latency_ttl_ms()
        return max(1, int(self.MAX_LATENCY_MS))

    def _calc_latency_detail(
        self,
        trace: List[Dict[str, Any]],
        total_latency_ms: int,
        threshold_ms: int,
    ) -> Dict[str, Any]:
        """
        Extract critical-path hints without summing parallel Send branches.

        A max duration is used for Search/Reading because their workers execute
        concurrently; summing them would overstate wall-clock stage latency.
        """
        node_events = [
            event for event in trace
            if event.get("event") == "node_observed"
        ]
        tool_events = [
            event for event in trace
            if _is_real_tool_completion(event)
        ]
        llm_events = [
            event for event in trace
            if event.get("event") in {"llm_finished", "llm_failed"}
        ]

        def duration(event: Dict[str, Any]) -> int:
            try:
                return max(0, int(event.get("latency_ms") or 0))
            except (TypeError, ValueError):
                return 0

        def slowest(events: List[Dict[str, Any]], *name_fields: str) -> Dict[str, Any]:
            if not events:
                return {}
            event = max(events, key=duration)
            name = next(
                (str(event.get(field)) for field in name_fields if event.get(field)),
                str(event.get("event") or "unknown"),
            )
            return {"name": name, "latency_ms": duration(event)}

        def stage_max(stage: str) -> int:
            candidates = [
                event for event in node_events
                if stage in str(event.get("observed_node") or "").lower()
            ]
            return max((duration(event) for event in candidates), default=0)

        return {
            "total_latency_ms": max(0, int(total_latency_ms or 0)),
            "threshold_ms": threshold_ms,
            "overage_ms": max(0, int(total_latency_ms or 0) - threshold_ms),
            "slowest_node": slowest(node_events, "observed_node", "node"),
            "slowest_tool": slowest(tool_events, "tool_name"),
            "slowest_llm": slowest(llm_events, "agent", "model"),
            "search_stage_max_ms": stage_max("search"),
            "reading_stage_max_ms": stage_max("reading"),
            "send_dispatch_count": sum(
                1 for event in trace if event.get("event") == "send_dispatch"
            ),
        }

    def _check_no_fake_citation(self, citation_summary: Dict[str, Any]) -> bool:
        if not citation_summary:
            return True
        return citation_summary.get("all_valid", False)

    def _check_citation_ids_exist(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True
        return all(r.get("id_exists", False) for r in results)

    def _check_source_urls_valid(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True
        return all(r.get("url_matches_source", False) for r in results)

    def _check_conclusion_evidence_coverage(
        self,
        report: str,
        evidence_cards: List[Dict[str, Any]],
        sources: List[Dict[str, Any]] = None,
        with_detail: bool = False,
    ):
        """Cross-check every program citation and the conclusion against evidence."""
        sources = sources or []
        evidence_source_ids = {card.get("source_id") for card in evidence_cards if card.get("source_id")}
        source_number = {
            index: source.get("source_id") for index, source in enumerate(sources, start=1)
        }
        claim_report = re.split(
            r"^##\s+(?:Sources|来源|Sources Appendix|来源附录|Citation Validation|引用校验)",
            report or "", maxsplit=1, flags=re.MULTILINE | re.IGNORECASE,
        )[0]
        refs = [int(value) for value in re.findall(r"\[(\d+)\]", claim_report)]
        mismatches = sorted({
            ref for ref in refs
            if ref not in source_number or source_number.get(ref) not in evidence_source_ids
        })
        conclusion = self._section_text(report, ("conclusion", "结论"))
        conclusion_refs = [int(value) for value in re.findall(r"\[(\d+)\]", conclusion)]
        conclusion_supported = not evidence_cards or bool(conclusion_refs)
        passed = not mismatches and conclusion_supported
        detail = {
            "citation_count": len(refs),
            "conclusion_citation_count": len(conclusion_refs),
            "mismatched_citation_ids": mismatches,
            "conclusion_has_evidence": conclusion_supported,
        }
        return (passed, detail) if with_detail else passed

    def _check_chapter_duplication(self, report: str, with_detail: bool = False):
        chapters = self._report_chapters(report)
        duplicate_pairs = []
        for (left_heading, left), (right_heading, right) in zip(chapters, chapters[1:]):
            score = self._bigram_jaccard(left, right)
            if score > 0.50:
                duplicate_pairs.append({
                    "left": left_heading, "right": right_heading,
                    "similarity": round(score, 4),
                })
        passed = not duplicate_pairs
        detail = {"adjacent_pairs_checked": max(0, len(chapters) - 1), "duplicate_pairs": duplicate_pairs}
        return (passed, detail) if with_detail else passed

    def _check_unsupported_expansion(
        self, report: str, evidence_cards: List[Dict[str, Any]], with_detail: bool = False,
    ):
        unsupported = []
        for paragraph in re.split(r"\n\s*\n", report or ""):
            clean = " ".join(line.strip() for line in paragraph.splitlines()).strip()
            if self._is_declarative_paragraph(clean) and not re.search(r"\[\d+\]", clean):
                unsupported.append(clean[:240])
        passed = not unsupported
        detail = {"unsupported_paragraph_count": len(unsupported), "unsupported_paragraphs": unsupported}
        return (passed, detail) if with_detail else passed

    @staticmethod
    def _is_declarative_paragraph(paragraph: str) -> bool:
        if len(paragraph) <= 80 or paragraph.startswith(("#", "|", ">", "- ", "*")):
            return False
        lowered = paragraph.lower()
        if any(label in lowered for label in (
            "guiding question", "核心问题", "citation validation", "引用校验",
            "fix summary", "修复摘要", "unresolved issues", "未解决",
        )):
            return False
        return True

    @staticmethod
    def _section_text(report: str, names: Tuple[str, ...]) -> str:
        pattern = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
        matches = list(pattern.finditer(report or ""))
        for index, match in enumerate(matches):
            if any(name in match.group(1).lower() for name in names):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
                return report[match.end():end]
        return ""

    @staticmethod
    def _report_chapters(report: str) -> List[Tuple[str, str]]:
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", report or "", re.MULTILINE))
        excluded = ("摘要", "summary", "conclusion", "结论", "source", "来源", "evidence gap", "证据空白", "fix", "修复")
        chapters = []
        for index, match in enumerate(matches):
            heading = match.group(1).strip()
            if any(term in heading.lower() for term in excluded):
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
            body = report[match.end():end].strip()
            if body:
                chapters.append((heading, body))
        return chapters

    @staticmethod
    def _bigram_jaccard(left: str, right: str) -> float:
        def grams(text: str) -> Set[str]:
            cjk = "".join(re.findall(r"[\u3400-\u9fff]", text))
            cjk_grams = {cjk[i:i + 2] for i in range(max(0, len(cjk) - 1))}
            words = re.findall(r"[a-z0-9]+", text.lower())
            word_grams = {f"{words[i]} {words[i + 1]}" for i in range(max(0, len(words) - 1))}
            return cjk_grams | word_grams
        left_grams, right_grams = grams(left), grams(right)
        if not left_grams or not right_grams:
            return 0.0
        return len(left_grams & right_grams) / len(left_grams | right_grams)

    def _calc_task_success_rate(
        self, trace: List[Dict[str, Any]], task_dag: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute task success rate from worker_started/worker_finished trace events.

        Returns {"rate": float, "passed": bool, "success": int, "total": int, "task_outcomes": dict}
        """
        tasks = task_dag.get("tasks", [])
        dag_task_ids = {t.get("task_id", "") for t in tasks}

        # Values are explicit: success, failed, or missing. Missing completion
        # must never be silently counted as success.
        task_outcomes: Dict[str, str] = {task_id: "missing" for task_id in dag_task_ids if task_id}
        started: Set[str] = set()
        worker_outcomes: Dict[str, str] = {}
        for ev in trace:
            if ev.get("event") in _WORKER_EVENTS:
                tid = ev.get("task_id", "")
                if not tid:
                    continue
                if ev.get("event") == "worker_started":
                    started.add(tid)
                    task_outcomes.setdefault(tid, "missing")
                elif ev.get("event") == "worker_finished":
                    worker_outcomes[tid] = (
                        "success" if ev.get("success", False) else "failed"
                    )

            if _is_real_tool_completion(ev):
                tid = ev.get("task_id", "")
                if not tid:
                    continue
                if not ev.get("success", False):
                    task_outcomes[tid] = "failed"
                elif task_outcomes.get(tid) != "failed":
                    task_outcomes[tid] = "success"

        # A worker_finished event is the authoritative task outcome. Internal tool
        # failures may be recoverable (for example, one graph-expansion provider
        # fails and the Worker completes from other retrieval results). Send nodes
        # currently prepend worker_finished before their detailed tool trace, so
        # applying this final outcome in a second pass also makes the calculation
        # independent of trace serialization order.
        task_outcomes.update(worker_outcomes)

        # Merge with DAG task ids
        all_task_ids = {tid for tid in dag_task_ids | set(task_outcomes.keys()) | started if tid}
        if not all_task_ids:
            return {
                "rate": 1.0, "passed": True, "success": 0, "failed": 0,
                "missing": 0, "total": 0, "task_outcomes": {},
            }

        total = len(all_task_ids)
        success_count = sum(1 for tid in all_task_ids if task_outcomes.get(tid) == "success")
        failed_count = sum(1 for tid in all_task_ids if task_outcomes.get(tid) == "failed")
        missing_count = total - success_count - failed_count
        rate = success_count / max(total, 1)
        return {
            "rate": rate,
            "passed": rate >= 1.0,
            "success": success_count,
            "failed": failed_count,
            "missing": missing_count,
            "total": total,
            "task_outcomes": task_outcomes,
        }

    def _calc_tool_error_rate(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute tool error rate from actual tool execution trace events only.

        Excludes: controller, planner, graph nodes, send_dispatch, merge, evaluator,
        function_call_started, tool_selected, etc.
        """
        tool_events = [event for event in trace if _is_real_tool_completion(event)]

        total = len(tool_events)
        if total == 0:
            return {"rate": 0.0, "passed": True, "error_count": 0, "total": 0}

        error_count = sum(1 for t in tool_events if not t.get("success", True))
        rate = error_count / total
        return {
            "rate": rate,
            "passed": rate <= self.MAX_TOOL_ERROR_RATE,
            "error_count": error_count,
            "total": total,
        }
