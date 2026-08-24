"""Deterministic, evidence-grounded fallback report generator.

The rule reviewer is used in offline mode and whenever the LLM reviewer fails.
It must therefore remain useful on its own: it answers the topic, groups verified
evidence into research themes, exposes gaps, and always closes with a conclusion.
"""

import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from app.agents.source_titles import display_paper_title, format_reference
from app.agents.report_style import academic_report_title, is_mmea_topic, report_keywords


class DraftReviewer:
    """Build a bounded academic synthesis without generating unsupported claims."""

    _THEMES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        (
            "Agent Process and Capability Evaluation",
            ("agent", "planning", "planner", "tool use", "tool-use", "trajectory",
             "action", "reasoning", "memory", "workflow", "environment"),
        ),
        (
            "Robustness, Reliability, and Safety",
            ("robust", "safety", "risk", "hallucination", "faithful", "reliab",
             "failure", "error", "adversarial", "attack"),
        ),
        (
            "Human and Model-based Assessment",
            ("human", "judge", "llm-as", "preference", "subjective", "annotator"),
        ),
        (
            "Efficiency and Operational Constraints",
            ("latency", "cost", "token", "efficien", "runtime", "compute", "budget"),
        ),
        (
            "Benchmarks, Metrics, and Evaluation Scope",
            ("benchmark", "metric", "taxonomy", "evaluation", "evaluate", "dataset",
             "task", "dimension", "score", "framework"),
        ),
        (
            "Reported Methods and Empirical Results",
            ("method", "result", "propose", "introduce", "improve", "achieve",
             "outperform", "experiment", "study"),
        ),
    )

    def review(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        citation_summary: Dict[str, Any] = None,
        language: str = "en",
        outline: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        if outline and outline.get("sections"):
            return self.assemble_report(
                topic, outline, sources, evidence_cards,
                citation_check_results, citation_summary, language,
            )
        if str(language or "").lower().replace("_", "-").startswith("zh"):
            return self._review_zh(
                topic=topic,
                sources=sources,
                evidence_cards=evidence_cards,
                citation_check_results=citation_check_results,
                citation_summary=citation_summary,
            )

        warnings: List[str] = []
        source_by_id = {source.get("source_id"): source for source in sources}
        source_number = {
            source.get("source_id"): index
            for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        verified_cards = self._verified_cards(evidence_cards, citation_check_results)
        verified_cards = self._deduplicate_cards(verified_cards)
        themes = self._group_by_theme(verified_cards)

        if not sources:
            warnings.append("No sources available for report")
        if not verified_cards:
            warnings.append("No verified evidence cards extracted from sources")

        lines = [f"# Research Report: {topic}", "", "## Executive Summary", ""]
        lines.append(self._executive_summary(topic, verified_cards, themes, source_number))

        lines.extend(["", "## Scope and Introduction", ""])
        lines.append(
            f"This bounded review covers {len(sources)} retrieved academic sources and "
            f"{len(verified_cards)} distinct citation-verified evidence statements. "
            "Statements below are limited to text exposed by the selected sources; paper "
            "titles and model background knowledge are not treated as evidence."
        )

        lines.extend(["", "## Thematic Synthesis", ""])
        if themes:
            for heading, cards in themes.items():
                refs = self._refs_for_cards(cards, source_number)
                lines.append(f"### {heading} {refs}".rstrip())
                lines.append(self._theme_summary(cards, source_by_id, source_number))
                lines.append("")
        else:
            lines.append(
                "No defensible thematic synthesis can be produced because no extracted "
                "statement passed deterministic citation validation."
            )
            lines.append("")

        lines.extend(["## Evidence-backed Key Findings", ""])
        if verified_cards:
            for card in verified_cards[:10]:
                sid = card.get("source_id", "")
                source = source_by_id.get(sid, {})
                ref = f"[{source_number[sid]}]" if sid in source_number else "[source unavailable]"
                lines.append(f"- **{ref} {display_paper_title(source)}**: "
                             f"{card.get('claim', '').strip()}")
        else:
            lines.append("- No verified finding is available for this run.")

        lines.extend(["", "## Research Gaps and Interpretation", ""])
        gap_lines = self._research_gaps(sources, verified_cards, themes)
        lines.extend(f"- {gap}" for gap in gap_lines)

        lines.extend(["", "## Practical Takeaways", ""])
        if themes:
            lines.append(
                "- Treat the themes above as complementary evaluation dimensions rather "
                "than as interchangeable single scores."
            )
            lines.append(
                "- Preserve claim-to-source provenance when comparing methods, and report "
                "coverage gaps separately from negative empirical results."
            )
            lines.append(
                "- Use the cited passages as the boundary of the present conclusion; claims "
                "requiring full experimental tables or implementation details need a deeper source read."
            )
        else:
            lines.append("- Retrieve sources with usable text before drawing a substantive conclusion.")

        lines.extend(["", "## Limitations", ""])
        providers = sorted({source.get("provider", "unknown") for source in sources}) or ["none"]
        missing_text = sum(1 for source in sources if not source.get("full_text", "").strip())
        lines.append(f"1. The evidence base is limited to {', '.join(providers)} search results.")
        lines.append(
            f"2. {missing_text}/{len(sources)} retrieved sources did not expose abstract or "
            "full text and therefore could not support findings."
        )
        lines.append(
            "3. Rule-based extraction favors explicit standalone statements and can miss "
            "claims distributed across paragraphs, figures, or tables."
        )
        lines.append(
            "4. Citation checks establish ID, URL, and exact-quote provenance; they do not "
            "by themselves prove semantic entailment or methodological quality."
        )

        lines.extend(["", "## Conclusion", ""])
        lines.append(self._conclusion(topic, verified_cards, themes, source_number))

        lines.extend(["", "## Sources Appendix", ""])
        if sources:
            lines.extend(["| # | Title | Type | Year | Quality |", "|---|-------|------|------|---------|"])
            for index, source in enumerate(sources, start=1):
                title = display_paper_title(source).replace("|", "\\|")
                lines.append(
                    f"| [{index}] | {title} | {source.get('source_type', 'unknown')} | "
                    f"{source.get('year', 'N/A')} | {float(source.get('quality_score', 0.0) or 0.0):.2f} |"
                )
        else:
            lines.append("No sources were retrieved.")

        lines.extend(["", "## Citation Validation Audit", ""])
        if citation_summary:
            total = citation_summary.get("total_checked", 0)
            valid = citation_summary.get("valid_count", 0)
            invalid = citation_summary.get("invalid_count", 0)
            lines.append(f"Checked {total} evidence bindings: {valid} valid, {invalid} invalid.")
            if invalid:
                warnings.append(f"{invalid}/{total} citations failed validation")
        else:
            lines.append("Citation validation was not performed.")
            warnings.append("Citation validation not performed")

        return {"draft_report": "\n".join(lines), "warnings": warnings}

    def generate_chapter(
        self,
        section: Dict[str, Any],
        evidence_cards: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        language: str = "zh",
        source_number: Dict[str, int] = None,
    ) -> str:
        """Generate one bounded chapter from only its pre-assigned cards."""
        zh = str(language or "").lower().replace("_", "-").startswith("zh")
        source_number = source_number or {
            source.get("source_id"): index for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        heading = section.get("heading", "未命名章节" if zh else "Untitled Chapter")
        lines = [f"## {heading}", ""]
        if not evidence_cards:
            lines.append(
                "现有文献未提供足以回答本节问题的可核验信息，本节不作推断。"
                if zh else
                "The available literature does not provide verifiable information sufficient to answer this section."
            )
            return "\n".join(lines)

        source_by_id = {source.get("source_id"): source for source in sources}
        source_count = len({card.get("source_id") for card in evidence_cards if card.get("source_id")})
        if source_count < 2:
            lines.append(
                "由于只有一个来源覆盖该问题，下述内容是单来源观察，不能视为跨研究共识。"
                if zh else
                "Only one source covers this question, so the observations below are not cross-study consensus."
            )
        lowered_heading = str(heading).lower()
        if any(term in lowered_heading for term in ("局限", "limitation", "open problem")):
            if source_count < 2:
                lines.append(
                    "目前只有一篇独立论文提供该领域局限证据，不能据此形成领域级结论；本节保留为空白并等待补充交叉来源。"
                    if zh else
                    "Only one independent paper provides limitation evidence, so no field-level limitation conclusion is reported."
                )
                return "\n".join(lines)
            lines.extend(self._limitation_synthesis(evidence_cards, source_number, zh))
            return "\n".join(lines)
        if any(term in lowered_heading for term in ("方法", "method taxonomy")):
            lines.extend(self._method_comparison_table(
                evidence_cards, source_by_id, source_number, zh,
            ))
            lines.append("")
        if any(term in lowered_heading for term in ("数据集", "dataset")):
            lines.extend(self._dataset_comparison_table(
                evidence_cards, source_by_id, source_number, zh,
            ))
            lines.append("")
        for card in evidence_cards[:8]:
            sid = card.get("source_id", "")
            ref = f"[{source_number[sid]}]" if sid in source_number else ""
            if zh:
                lines.append(self._bounded_chinese_card_summary(card, lowered_heading, ref))
                continue
            title = display_paper_title(source_by_id.get(sid, {"title": sid or "Unknown"}))
            claim = card.get("claim", "").strip()
            evidence_type = card.get("evidence_type", "primary_claim")
            qualifier = (
                "二手概述" if evidence_type in {"secondary_summary", "review"} else "原始研究"
            ) if zh else (
                "secondary account" if evidence_type in {"secondary_summary", "review"} else "primary study"
            )
            lines.append(f"- **{title}**（{qualifier}）：{claim} {ref}".rstrip())
            if card.get("key_results"):
                lines.append(f"  - {'关键结果' if zh else 'Key result'}：{card['key_results']} {ref}".rstrip())
            if card.get("limitation"):
                lines.append(f"  - {'原文局限' if zh else 'Reported limitation'}：{card['limitation']} {ref}".rstrip())
        return "\n".join(lines)

    @staticmethod
    def _limitation_synthesis(
        cards: List[Dict[str, Any]], source_number: Dict[str, int], zh: bool,
    ) -> List[str]:
        """Group limitations by domain mechanism instead of narrating one paper."""
        dimensions = (
            ("模态缺失、噪声与质量不一致", "Missing/noisy modalities and quality mismatch", ("missing", "noise", "noisy", "image", "visual", "modality", "模态", "图像", "噪声")),
            ("监督信号与伪种子依赖", "Dependence on supervision and pseudo seeds", ("seed", "pseudo", "label", "supervision", "种子", "伪标签", "标注")),
            ("跨语言、跨图谱与跨领域泛化", "Cross-language, graph, and domain generalization", ("general", "domain", "distribution", "language", "transfer", "泛化", "领域", "分布", "跨语言")),
            ("训练成本、推理成本与可扩展性", "Training/inference cost and scalability", ("cost", "scalab", "compute", "memory", "large-scale", "成本", "扩展", "计算", "显存")),
            ("基准覆盖、真实场景与可复现性", "Benchmark scope, real-world validity, and reproducibility", ("benchmark", "dataset", "real-world", "reproduc", "dynamic", "基准", "数据集", "真实", "复现", "动态")),
        )
        grouped: Dict[str, List[Dict[str, Any]]] = {zh_name if zh else en_name: [] for zh_name, en_name, _ in dimensions}
        other = "其他有文献支持的开放问题" if zh else "Other evidence-backed open problems"
        grouped[other] = []
        for card in cards:
            limitation = str(card.get("limitation") or "").strip()
            if not limitation:
                continue
            lowered = limitation.lower()
            matched = False
            for zh_name, en_name, cues in dimensions:
                if any(cue in lowered for cue in cues):
                    grouped[zh_name if zh else en_name].append(card)
                    matched = True
            if not matched:
                grouped[other].append(card)

        lines = []
        for dimension, dimension_cards in grouped.items():
            unique_sources = list(dict.fromkeys(
                card.get("source_id") for card in dimension_cards if card.get("source_id")
            ))
            if not unique_sources:
                continue
            refs = "".join(
                f"[{source_number[source_id]}]" for source_id in unique_sources[:2]
                if source_id in source_number
            )
            qualifier = (
                "多篇研究共同涉及" if len(unique_sources) >= 2 else "单篇研究报告，尚不能视为共识"
            ) if zh else (
                "reported across multiple studies" if len(unique_sources) >= 2 else "reported by one study, not field consensus"
            )
            if zh:
                lines.append(
                    f"- **{dimension}**（{qualifier}）：现有证据确认该问题已被相关研究明确讨论；"
                    f"不同数据与设置下的影响程度仍需可比实验进一步验证 {refs}。".rstrip()
                )
            else:
                excerpts = " ".join(
                    str(card.get("limitation") or "").strip() for card in dimension_cards[:2]
                )
                lines.append(f"- **{dimension}**（{qualifier}）：{excerpts} {refs}".rstrip())
        return lines or [
            "现有来源未提供可归纳的领域局限。" if zh else
            "The available sources do not expose limitations that can be synthesized."
        ]

    def assemble_report(
        self,
        topic: str,
        outline: Dict[str, Any],
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        citation_summary: Dict[str, Any],
        language: str,
    ) -> Dict[str, Any]:
        """Assemble title/summary, isolated chapters, conclusion, and appendices."""
        zh = str(language or "").lower().replace("_", "-").startswith("zh")
        verified = self._deduplicate_cards(self._verified_cards(evidence_cards, citation_check_results))
        by_id = {card.get("evidence_id"): card for card in verified}
        source_number = {
            source.get("source_id"): index for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        covered_headings = [
            str(section.get("heading") or "")
            for section in outline.get("sections", [])
            if section.get("assigned_evidence_ids")
        ]
        refs = self._refs_for_cards(verified, source_number, limit=2) if verified else ""
        lead = (
            f"现有文献可支持对{'、'.join(covered_headings[:5]) or '所提问题'}的有界回答 {refs}。"
            if zh else
            f"The available literature supports bounded answers on {', '.join(covered_headings[:5]) or 'the requested questions'} {refs}."
        )
        lines = [
            f"# {academic_report_title(topic, language)}", "",
            f"## {'摘要' if zh else 'Abstract'}", "",
            lead, "",
            f"**{'关键词' if zh else 'Keywords'}**：{report_keywords(topic, covered_headings, language)}", "",
        ]
        strict_structure = is_mmea_topic(topic)
        if not strict_structure:
            lines.extend([
                f"## {'范围与证据基础' if zh else 'Scope and Evidence Base'}", "",
                (f"本报告围绕“{topic}”整理可追溯的学术研究，并区分跨来源发现与单篇论文观察 {refs}。"
                 if zh else
                 f"This report surveys traceable academic work on {topic} and distinguishes cross-source findings from single-paper observations {refs}."), "",
                f"## {'主题综合' if zh else 'Thematic Synthesis'}", "",
            ])
        chapter_timings = []
        for section in outline.get("sections", []):
            ids = set(section.get("assigned_evidence_ids") or [])
            chapter_cards = [by_id[item] for item in ids if item in by_id]
            source_ids = set(section.get("assigned_source_ids") or [])
            chapter_sources = [source for source in sources if source.get("source_id") in source_ids]
            chapter = self.generate_chapter(
                section, chapter_cards, chapter_sources, language, source_number=source_number,
            )
            lines.extend([chapter, ""])
            chapter_timings.append({"heading": section.get("heading", ""), "latency_ms": 0, "mode": "rule"})

        lines.extend([f"## {'结论' if zh else 'Conclusion'}", ""])
        if verified:
            lines.append(
                (f"围绕“{topic}”，现有研究呈现了上述方法、数据与实验结论；"
                 f"跨来源重复出现的发现较稳健，单来源观察仍需进一步验证 {refs}。" if zh else
                 f"For {topic}, the literature supports the method, data, and experimental findings above; "
                 f"cross-source findings are stronger than single-source observations {refs}.")
            )
        else:
            lines.append("没有通过校验的证据，无法形成实质结论。" if zh else "No verified evidence is available for a substantive conclusion.")

        lines.extend(["", f"## {'参考文献与证据追踪' if zh else 'References and Evidence Traceability'}", ""])
        evidence_types_by_source: Dict[str, set] = {}
        for card in verified:
            evidence_types_by_source.setdefault(str(card.get("source_id") or ""), set()).add(
                str(card.get("evidence_type") or "primary_claim")
            )
        for index, source in enumerate(sources, start=1):
            lines.append(format_reference(
                index, source, evidence_types= evidence_types_by_source.get(
                    str(source.get("source_id") or ""), set()
                ),
            ))
        warnings = list(outline.get("evidence_gaps") or [])
        invalid = int((citation_summary or {}).get("invalid_count", 0) or 0)
        total = int((citation_summary or {}).get("total_checked", 0) or 0)
        if invalid:
            warnings.append(
                f"{invalid}/{total} 条引用未通过校验" if zh else
                f"{invalid}/{total} citations failed validation"
            )
        return {"draft_report": "\n".join(lines), "warnings": warnings, "chapter_timings": chapter_timings}

    @staticmethod
    def _method_comparison_table(
        cards: List[Dict[str, Any]], source_by_id: Dict[str, Dict[str, Any]],
        source_number: Dict[str, int], zh: bool,
    ) -> List[str]:
        grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for card in cards:
            source_id = str(card.get("source_id") or "")
            if source_id:
                grouped.setdefault(source_id, []).append(card)
        records = []
        for source_id, source_cards in grouped.items():
            source = source_by_id.get(source_id, {})
            ref = f"[{source_number[source_id]}]" if source_id in source_number else ""
            records.append({
                "论文": f"论文{ref}" if zh else f"{display_paper_title(source)} {ref}",
                "年份": str(source.get("year") or ""),
                "技术路线": DraftReviewer._first_concise(source_cards, ("method_family",), zh),
                "核心方法": DraftReviewer._first_concise(source_cards, ("method",), zh),
                "模态": DraftReviewer._first_concise(source_cards, ("modalities",), zh),
                "数据集": DraftReviewer._first_concise(source_cards, ("dataset_name",), zh),
                "评价指标": DraftReviewer._first_concise(source_cards, ("metric",), zh),
            })
        if len(records) < 2:
            return []
        optional = ["年份", "技术路线", "核心方法", "模态", "数据集", "评价指标"]
        threshold = max(2, (len(records) + 2) // 3)
        selected = [key for key in optional if sum(bool(row.get(key)) for row in records) >= threshold]
        if not selected:
            return []
        headers = ["论文", *selected]
        if not zh:
            labels = {"论文": "Paper", "年份": "Year", "技术路线": "Method Family", "核心方法": "Method", "模态": "Modalities", "数据集": "Dataset", "评价指标": "Metric"}
            display_headers = [labels[item] for item in headers]
        else:
            display_headers = headers
        rows = ["| " + " | ".join(display_headers) + " |", "|" + "---|" * len(headers)]
        for record in records:
            rows.append("| " + " | ".join(record.get(key) or "—" for key in headers) + " |")
        return rows

    @staticmethod
    def _dataset_comparison_table(
        cards: List[Dict[str, Any]], source_by_id: Dict[str, Dict[str, Any]],
        source_number: Dict[str, int], zh: bool,
    ) -> List[str]:
        grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for card in cards:
            dataset = DraftReviewer._concise_field(card.get("dataset_name"), zh)
            if not dataset:
                continue
            grouped.setdefault(dataset, []).append(card)
        records = []
        for dataset, dataset_cards in grouped.items():
            source_ids = list(dict.fromkeys(str(card.get("source_id") or "") for card in dataset_cards))
            refs = "".join(
                f"[{source_number[source_id]}]" for source_id in source_ids[:2]
                if source_id in source_number
            )
            records.append({
                "数据集": dataset,
                "图谱或语言对": DraftReviewer._first_concise(dataset_cards, ("graph_or_language_pair",), zh),
                "规模": DraftReviewer._first_concise(dataset_cards, ("entity_count",), zh),
                "模态": DraftReviewer._first_concise(dataset_cards, ("modalities",), zh),
                "缺失情况": DraftReviewer._first_concise(dataset_cards, ("missingness",), zh),
                "划分或种子比例": DraftReviewer._first_concise(dataset_cards, ("data_split", "seed_ratio"), zh),
                "代表文献": refs,
            })
        if len(records) < 2:
            return []
        optional = ["图谱或语言对", "规模", "模态", "缺失情况", "划分或种子比例"]
        threshold = max(2, (len(records) + 2) // 3)
        selected = [key for key in optional if sum(bool(row.get(key)) for row in records) >= threshold]
        if not selected:
            return []
        headers = ["数据集", *selected, "代表文献"]
        labels = {
            "数据集": "Dataset", "图谱或语言对": "Graph/Language Pair", "规模": "Scale",
            "模态": "Modalities", "缺失情况": "Missingness", "划分或种子比例": "Split/Seed Ratio",
            "代表文献": "Representative Papers",
        }
        display_headers = headers if zh else [labels[item] for item in headers]
        rows = ["| " + " | ".join(display_headers) + " |", "|" + "---|" * len(headers)]
        for record in records:
            rows.append("| " + " | ".join(record.get(key) or "—" for key in headers) + " |")
        return rows

    @staticmethod
    def _concise_field(value: Any, zh: bool) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "/")
        if not text or len(text) > 90 or text.count(".") > 1:
            return ""
        if zh and len(re.findall(r"[A-Za-z]", text)) > 70 and not re.search(r"\d|[-@]", text):
            return ""
        return text

    @staticmethod
    def _first_concise(cards: List[Dict[str, Any]], keys: Tuple[str, ...], zh: bool) -> str:
        for card in cards:
            for key in keys:
                value = DraftReviewer._concise_field(card.get(key), zh)
                if value:
                    return value
        return ""

    @staticmethod
    def _bounded_chinese_card_summary(card: Dict[str, Any], heading: str, ref: str) -> str:
        dataset = DraftReviewer._concise_field(card.get("dataset_name"), True)
        metric = DraftReviewer._concise_field(card.get("metric"), True)
        seed = DraftReviewer._concise_field(card.get("seed_ratio") or card.get("data_split"), True)
        details = []
        if "数据集" in heading and dataset:
            details.append(f"使用 {dataset}")
        if any(cue in heading for cue in ("评价", "评估", "实验")) and metric:
            details.append(f"采用 {metric} 进行评价")
        if any(cue in heading for cue in ("评价", "评估", "实验")) and seed:
            details.append(f"实验划分或种子设置为 {seed}")
        if details:
            return f"- 论文{ref}{'，'.join(details)}；其余比较条件在当前证据中未完整报告。"
        return (
            f"- 论文{ref}为本章问题提供了直接证据，但现有结构化信息不足以支持更细致的中文概括，"
            "因此不对英文原文作机械直译。"
        )

    @staticmethod
    def _table_value(value: Any, fallback: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip() or fallback
        return text[:180].replace("|", "/")

    def _review_zh(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        citation_summary: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """中文确定性回退；原始论文标题和证据陈述保持原文，避免无依据翻译。"""
        warnings: List[str] = []
        source_by_id = {source.get("source_id"): source for source in sources}
        source_number = {
            source.get("source_id"): index
            for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        verified_cards = self._deduplicate_cards(
            self._verified_cards(evidence_cards, citation_check_results)
        )
        themes = self._group_by_theme(verified_cards)
        theme_names = {
            "Agent Process and Capability Evaluation": "智能体过程与能力评估",
            "Robustness, Reliability, and Safety": "鲁棒性、可靠性与安全",
            "Human and Model-based Assessment": "人工与模型评审",
            "Efficiency and Operational Constraints": "效率与运行约束",
            "Benchmarks, Metrics, and Evaluation Scope": "基准、指标与评估范围",
            "Reported Methods and Empirical Results": "方法与实证结果",
            "Other Supported Observations": "其他有证据支持的观察",
        }

        if not sources:
            warnings.append("没有可用于报告的来源")
        if not verified_cards:
            warnings.append("没有从来源中提取到通过引用校验的证据卡")

        lines = [f"# 调研报告：{topic}", "", "## 执行摘要", ""]
        if verified_cards:
            named_themes = [theme_names.get(name, name) for name in list(themes)[:4]]
            lead = " ".join(
                (
                    f"{card.get('claim', '').strip()} "
                    f"[{source_number[card.get('source_id')]}]"
                )
                if card.get("source_id") in source_number
                else card.get("claim", "").strip()
                for card in verified_cards[:3]
            )
            lines.append(
                f"围绕 **{topic}**，现有可核验证据主要分布在"
                f"{'、'.join(named_themes)}。以下原始证据界定了本次报告能够直接支持的"
                f"结论范围：{lead}。超出这些证据的判断仍需补充可访问的全文材料。"
            )
        else:
            lines.append(
                f"本次运行无法可靠回答 **{topic}**，因为没有证据陈述通过引用校验。"
                "论文标题本身不能作为研究结论的证据。"
            )

        lines.extend(["", "## 范围与证据基础", ""])
        lines.append(
            f"本次有界调研共纳入 {len(sources)} 个学术来源，其中形成 "
            f"{len(verified_cards)} 条去重且通过引用校验的证据陈述。报告仅使用来源"
            "实际暴露的文本；论文标题与模型背景知识均不被视为证据。"
        )

        lines.extend(["", "## 主题综合", ""])
        if themes:
            for heading, cards in themes.items():
                refs = self._refs_for_cards(cards, source_number)
                lines.append(f"### {theme_names.get(heading, heading)} {refs}".rstrip())
                source_count = len({
                    card.get("source_id") for card in cards if card.get("source_id")
                })
                statements = []
                for card in cards[:3]:
                    sid = card.get("source_id", "")
                    ref = f"[{source_number[sid]}]" if sid in source_number else ""
                    statements.append(f"{card.get('claim', '').strip()} {ref}".strip())
                lines.append(
                    f"来自 {source_count} 个来源的证据支持以下有界综合。"
                    "为避免引入未经验证的翻译，证据陈述保留来源原文："
                    + " ".join(statements)
                )
                lines.append("")
        else:
            lines.extend([
                "由于没有任何提取陈述通过确定性引用校验，当前不能形成可靠的主题综合。",
                "",
            ])

        lines.extend(["## 有证据支持的关键发现", ""])
        if verified_cards:
            for card in verified_cards[:10]:
                sid = card.get("source_id", "")
                source = source_by_id.get(sid, {})
                ref = f"[{source_number[sid]}]" if sid in source_number else "[来源不可用]"
                lines.append(
                    f"- **{ref} {display_paper_title(source)}**："
                    f"{card.get('claim', '').strip()}"
                )
        else:
            lines.append("- 本次运行没有可核验的研究发现。")

        represented = {
            card.get("source_id") for card in verified_cards if card.get("source_id")
        }
        lines.extend(["", "## 研究空白与解释边界", ""])
        limitations = [card.get("limitation", "").strip() for card in verified_cards
                       if card.get("limitation")]
        if limitations:
            lines.append("- 来源明确报告的局限仍待解决：" + " ".join(limitations[:2]))
        if len(represented) < len(sources):
            lines.append(
                f"- 仅有 {len(represented)}/{len(sources)} 个来源贡献了通过校验的证据，"
                "说明检索覆盖范围大于实际证据覆盖范围。"
            )
        if len(themes) < 2:
            lines.append("- 证据涉及的独立主题不足两个，目前尚不能建立稳健的分类或方法比较。")
        if not limitations and len(represented) == len(sources) and len(themes) >= 2:
            lines.append(
                "- 当前摘要可支持主题概览，但不足以对基准分数、实现细节或因果效果"
                "进行受控比较。"
            )

        lines.extend(["", "## 局限性", ""])
        providers = sorted({source.get("provider", "unknown") for source in sources}) or ["无"]
        missing_text = sum(1 for source in sources if not source.get("full_text", "").strip())
        lines.extend([
            f"1. 证据范围受限于以下检索来源：{', '.join(providers)}。",
            f"2. {missing_text}/{len(sources)} 个来源未暴露摘要或全文，因而不能支持研究发现。",
            "3. 规则抽取偏向显式、独立的陈述，可能遗漏分散在段落、图表中的主张。",
            "4. 引用校验能够确认 ID、URL 与原文片段的溯源关系，但不能单独证明"
            "语义蕴含或方法质量。",
        ])

        lines.extend(["", "## 结论", ""])
        if verified_cards:
            refs = self._refs_for_cards(verified_cards, source_number)
            named_themes = [theme_names.get(name, name) for name in list(themes)[:4]]
            lines.append(
                f"在本次获取的证据范围内，**{topic}** 涉及"
                f"{'、'.join(named_themes)}等多个维度 {refs}。这些证据支持采用结构化、"
                "保留来源链路的分析方式，但现有覆盖空白不足以支持关于通用基准或方法"
                "优越性的更强结论。"
            )
        else:
            lines.append(
                f"本次运行无法对 **{topic}** 给出有证据支持的结论；后续检索应优先"
                "选择具有可访问摘要或全文的来源。"
            )

        lines.extend(["", "## 来源附录", ""])
        if sources:
            lines.extend(["| # | 标题 | 类型 | 年份 | 质量分 |", "|---|------|------|------|--------|"])
            for index, source in enumerate(sources, start=1):
                title = display_paper_title(source).replace("|", "\\|")
                lines.append(
                    f"| [{index}] | {title} | {source.get('source_type', 'unknown')} | "
                    f"{source.get('year', 'N/A')} | "
                    f"{float(source.get('quality_score', 0.0) or 0.0):.2f} |"
                )
        else:
            lines.append("未检索到来源。")

        lines.extend(["", "## 引用校验审计", ""])
        if citation_summary:
            total = citation_summary.get("total_checked", 0)
            valid = citation_summary.get("valid_count", 0)
            invalid = citation_summary.get("invalid_count", 0)
            lines.append(f"共检查 {total} 条证据绑定：{valid} 条有效，{invalid} 条无效。")
            if invalid:
                warnings.append(f"{invalid}/{total} 条引用未通过校验")
        else:
            lines.append("未执行引用校验。")
            warnings.append("未执行引用校验")

        return {"draft_report": "\n".join(lines), "warnings": warnings}

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _verified_cards(
        self,
        cards: List[Dict[str, Any]],
        checks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_ids = {
            check.get("citation_id")
            for check in checks
            if check.get("citation_id") is not None and check.get("is_valid", False)
        }
        return [dict(card) for index, card in enumerate(cards, start=1) if index in valid_ids]

    def _deduplicate_cards(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for card in cards:
            key = self._normalize(card.get("claim", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(card)
        return unique

    def _group_by_theme(self, cards: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
        grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for card in cards:
            text = " ".join(
                str(card.get(field, "")) for field in ("claim", "method", "limitation")
            ).lower()
            heading = "Other Supported Observations"
            for candidate, keywords in self._THEMES:
                if any(keyword in text for keyword in keywords):
                    heading = candidate
                    break
            grouped.setdefault(heading, []).append(card)
        return grouped

    @staticmethod
    def _refs_for_cards(
        cards: List[Dict[str, Any]], source_number: Dict[str, int], limit: int = None,
    ) -> str:
        numbers = sorted({source_number[c.get("source_id")] for c in cards if c.get("source_id") in source_number})
        if limit is not None:
            numbers = numbers[:max(0, int(limit))]
        return "".join(f"[{number}]" for number in numbers)

    def _theme_summary(
        self,
        cards: List[Dict[str, Any]],
        source_by_id: Dict[str, Dict[str, Any]],
        source_number: Dict[str, int],
    ) -> str:
        statements = []
        for card in cards[:3]:
            sid = card.get("source_id", "")
            ref = f"[{source_number[sid]}]" if sid in source_number else ""
            statements.append(f"{card.get('claim', '').strip()} {ref}".strip())
        source_count = len({card.get("source_id") for card in cards if card.get("source_id")})
        prefix = (
            f"Evidence from {source_count} source{'s' if source_count != 1 else ''} "
            "supports the following bounded synthesis: "
        )
        return prefix + " ".join(statements)

    def _executive_summary(
        self,
        topic: str,
        cards: List[Dict[str, Any]],
        themes: "OrderedDict[str, List[Dict[str, Any]]]",
        source_number: Dict[str, int],
    ) -> str:
        if not cards:
            return (
                f"The current run cannot answer **{topic}** reliably because no evidence "
                "statement passed citation validation. No evidence-backed findings can be "
                "reported. Retrieved titles alone are insufficient "
                "for a research conclusion."
            )
        theme_names = list(themes)[:4]
        lead_cards = cards[:3]
        lead = " ".join(
            f"{card.get('claim', '').strip()} "
            f"[{source_number[card.get('source_id')]}]"
            if card.get("source_id") in source_number else card.get("claim", "").strip()
            for card in lead_cards
        )
        return (
            f"For **{topic}**, the available evidence clusters around "
            f"{', '.join(theme_names)}. {lead} These statements define what this run can "
            "support directly; broader claims require additional full-text evidence."
        )

    @staticmethod
    def _research_gaps(
        sources: List[Dict[str, Any]],
        cards: List[Dict[str, Any]],
        themes: "OrderedDict[str, List[Dict[str, Any]]]",
    ) -> List[str]:
        gaps = []
        limitation_cards = [card for card in cards if card.get("limitation")]
        if limitation_cards:
            gaps.append(
                "Source-reported limitations remain unresolved: "
                + " ".join(card["limitation"].strip() for card in limitation_cards[:2])
            )
        represented_sources = {card.get("source_id") for card in cards if card.get("source_id")}
        if len(represented_sources) < len(sources):
            gaps.append(
                f"Only {len(represented_sources)}/{len(sources)} sources contributed verified "
                "evidence, so source discovery is broader than evidential coverage."
            )
        if len(themes) < 2:
            gaps.append(
                "The extracted evidence covers fewer than two distinct themes; a robust "
                "taxonomy or method comparison cannot yet be established."
            )
        if not gaps:
            gaps.append(
                "The available abstracts support a thematic overview but not a controlled "
                "comparison of benchmark scores, implementation details, or causal effects."
            )
        return gaps

    def _conclusion(
        self,
        topic: str,
        cards: List[Dict[str, Any]],
        themes: "OrderedDict[str, List[Dict[str, Any]]]",
        source_number: Dict[str, int],
    ) -> str:
        if not cards:
            return (
                f"No evidence-backed conclusion about **{topic}** is possible in this run. "
                "A follow-up search should prioritize sources with accessible abstracts or full text."
            )
        refs = self._refs_for_cards(cards, source_number)
        return (
            f"Within the retrieved evidence, **{topic}** is best understood as a multi-dimensional "
            f"problem spanning {', '.join(list(themes)[:4])} {refs}. The evidence supports a "
            "structured, provenance-preserving evaluation approach, while the identified coverage "
            "gaps prevent stronger claims about universal benchmarks or method superiority."
        )
