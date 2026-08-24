"""
app/agents/final_reviewer.py

Final Reviewer —— 最终报告修正器。

根据 Evaluator 的 feedback 对 draft_report 做**实质修复**。

核心原则（工程诚信）：
- 能删除的假内容直接删除，不贴标签假装修好了
- 需要重新研究才能解决的在报告末尾列入 UNRESOLVED ISSUES
-- 暂时做不到的在 README 和项目文档中诚实记录

支持的操作：
  1. 删除假引用标记：报告中 [N] 对应 is_valid=False → 移除 [N]，相关句子也删除
  2. 清理引用表中的无效条目：Citation Validation section 中的 failed citation → 移除
  3. URL 强制覆盖：URL 不匹配 → 用 PaperSource 中的权威 URL 替换报告中的错误 URL
  4. 补全过短报告：answer_not_empty 失败 → 从 evidence_cards 追加内容
  5. 评估横幅：关键指标失败 → 标题下插入警告
  6. UNRESOLVED ISSUES：当前无法解决的问题 → 诚实列出

和 Reflection 的关系：
- Final Reviewer 是 Reflection 闭环的执行端
- 不是"看一眼然后写个备注"——是做实际修改
- 做不了的诚实说做不了
"""

import re
from typing import Any, Dict, List, Set


class FinalReviewer:
    """
    Final Reviewer —— 实质修复 draft_report，不是追加注解。
    """

    def review(
        self,
        draft_report: str = "",
        eval_metrics: Dict[str, Any] = None,
        eval_feedback: List[str] = None,
        citation_check_results: List[Dict[str, Any]] = None,
        evidence_cards: List[Dict[str, Any]] = None,
        sources: List[Dict[str, Any]] = None,
        warnings: List[str] = None,
        language: str = "en",
        topic: str = "",
        outline: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        对 draft_report 做实质修复。

        返回：{"final_report": str, "fixes_applied": List[str],
               "unresolved_issues": List[str], "warnings": List[str]}
        """
        eval_metrics = eval_metrics or {}
        eval_feedback = eval_feedback or []
        citation_check_results = citation_check_results or []
        evidence_cards = evidence_cards or []
        sources = sources or []
        warnings = warnings or []
        outline = outline or {}
        zh = str(language or "").lower().replace("_", "-").startswith("zh")

        fixes_applied: List[str] = []
        unresolved_issues: List[str] = []
        unsupported_paragraphs: List[str] = []
        merged_sections: List[str] = []
        citation_repairs: List[str] = []

        report = draft_report
        metrics = eval_metrics.get("metrics", eval_metrics)
        metrics_detail = eval_metrics.get("metrics_detail", {})

        # ================================================================
        # 修复 1：删除假引用（不是标记，是删除）
        # ================================================================
        invalid_ids = self._collect_invalid_ids(citation_check_results)
        if invalid_ids:
            report, removed_count = self._delete_fake_citations(report, invalid_ids)
            if removed_count > 0:
                fixes_applied.append(
                    (
                        f"已从报告正文删除 {removed_count} 个无效引用标记"
                        f"（ID：{sorted(invalid_ids)}）"
                        if zh else
                        f"Deleted {removed_count} fake citation marker(s) "
                        f"(IDs: {sorted(invalid_ids)}) from report text"
                    ).replace("[", "#").replace("]", "")
                )

        # ================================================================
        # 修复 2：清理引用校验 section 中的无效条目
        # ================================================================
        if invalid_ids:
            report, removed_rows = self._clean_citation_section(report, invalid_ids)
            if removed_rows > 0:
                fixes_applied.append(
                    f"已删除 {removed_rows} 行无效引用记录"
                    if zh else
                    f"Removed {removed_rows} invalid citation table row(s)"
                )

        # ================================================================
        # 修复 3：URL 强制覆盖（用 PaperSource 的权威 URL 替换错误 URL）
        # ================================================================
        if not metrics.get("source_url_valid", True) and sources:
            url_mismatches = [
                r for r in citation_check_results
                if not r.get("url_matches_source", False) and r.get("id_exists", False)
            ]
            if url_mismatches:
                report, urls_fixed = self._fix_urls_from_sources(
                    report, url_mismatches, sources
                )
                if urls_fixed > 0:
                    fixes_applied.append(
                        f"已使用权威来源 URL 修正 {urls_fixed} 个引用链接"
                        if zh else
                        f"Corrected {urls_fixed} citation URL(s) using authoritative source URLs"
                    )

        # ================================================================
        # 修复 4：报告过短 → 从 evidence_cards 追加有效证据
        # ================================================================
        if not metrics.get("answer_not_empty", True):
            # 只用 is_valid=True 的 citation 对应的 evidence
            valid_source_ids = self._collect_valid_source_ids(citation_check_results)
            valid_cards = [
                c for c in evidence_cards
                if c.get("source_id") in valid_source_ids or not valid_source_ids
            ]
            if valid_cards:
                report, added = self._expand_thin_report(
                    report, valid_cards, language=language
                )
                fixes_applied.append(
                    f"已追加 {added} 条证据卡以补充过短报告"
                    if zh else
                    f"Appended {added} evidence card(s) to expand thin report"
                )

        # ================================================================
        # 修复 5：长声明段落缺少引用 → 绑定最相近证据，否则删除
        # ================================================================
        if not metrics.get("unsupported_expansion", True):
            if topic:
                report, unsupported_paragraphs, repaired_count = self._resolve_unsupported_paragraphs(
                    report, evidence_cards, sources,
                )
            else:
                report, unsupported_paragraphs = self._mark_unsupported_paragraphs_legacy(report)
                repaired_count = 0
            if unsupported_paragraphs:
                fixes_applied.append(
                    f"已处理 {len(unsupported_paragraphs)} 个缺少引用的声明性段落："
                    f"绑定 {repaired_count} 个，删除 {len(unsupported_paragraphs) - repaired_count} 个"
                    if zh else
                    f"Resolved {len(unsupported_paragraphs)} unsupported paragraph(s): "
                    f"bound {repaired_count}, removed {len(unsupported_paragraphs) - repaired_count}"
                )

        # ================================================================
        # 修复 6：相邻章节高度重复 → 合并（修复阈值 70%）
        # ================================================================
        if not metrics.get("chapter_duplication", True):
            report, merged_sections = self._merge_duplicate_chapters(report)
            if merged_sections:
                fixes_applied.append(
                    f"已合并 {len(merged_sections)} 组高度重复章节"
                    if zh else f"Merged {len(merged_sections)} highly duplicated chapter pair(s)"
                )

        # ================================================================
        # 修复 7：[N] 与实际来源/证据绑定不一致 → 基于段落证据相似度修正
        # ================================================================
        if not metrics.get("conclusion_evidence_coverage", True):
            mismatch_ids = set(
                metrics_detail.get("conclusion_evidence_coverage", {}).get(
                    "mismatched_citation_ids", []
                )
            )
            report, citation_repairs = self._repair_citation_mismatches(
                report, mismatch_ids, evidence_cards, sources,
            )
            if citation_repairs:
                fixes_applied.append(
                    f"已修正 {len(citation_repairs)} 个引用编号与来源绑定"
                    if zh else f"Repaired {len(citation_repairs)} citation/source binding(s)"
                )

        # ================================================================
        # 不可自动修复的问题 → UNRESOLVED ISSUES
        # ================================================================
        unresolved_issues = self._collect_unresolved(
            metrics, metrics_detail, language=language
        )

        if topic:
            # Formal report and internal audit are separate artifacts.  No Reviewer,
            # metric, latency, trace, or repair text is appended to reader-facing prose.
            report = self._strip_internal_sections(report)
            report = self._remove_internal_paragraphs(report)
            report, removed_markers = self._remove_citation_needed(report)
            if removed_markers:
                fixes_applied.append(
                    f"已删除 {removed_markers} 个无法闭环的引用占位声明"
                    if zh else f"Removed {removed_markers} unresolved citation placeholder statement(s)"
                )
            report = re.sub(r"file://[^\s\)\]]+", "", report, flags=re.IGNORECASE)
            report = self._clean_reader_citations(report)
        else:
            banners = self._build_warning_banners(
                metrics, citation_check_results, language=language,
            )
            if banners:
                report = self._insert_banners_after_title(report, banners)
            report = self._append_fix_summary(
                report, fixes_applied, unresolved_issues, eval_metrics,
                unsupported_paragraphs=unsupported_paragraphs,
                merged_sections=merged_sections,
                citation_repairs=citation_repairs,
                language=language,
            )

        completion_issues = self._completion_gate(
            report, topic, sources, evidence_cards, outline,
        )

        # 最终警告合并
        all_warnings = list(warnings)
        if unresolved_issues:
            all_warnings.append(
                (
                    f"本次有界调研仍有 {len(unresolved_issues)} 个问题无法自动修复"
                    if zh else
                    f"{len(unresolved_issues)} issue(s) could not be auto-resolved "
                    "during this bounded research run"
                )
            )

        return {
            "final_report": report,
            "fixes_applied": fixes_applied,
            "unresolved_issues": unresolved_issues,
            "warnings": all_warnings,
            "completion_ready": not completion_issues,
            "completion_issues": completion_issues,
            "strict_completion_required": self._requires_strict_completion(topic),
        }

    @staticmethod
    def _clean_reader_citations(report: str) -> str:
        """Keep citations readable without exposing internal evidence identifiers."""
        cleaned = re.sub(
            r"\(?\b(?:s2:[a-f0-9]+|W\d+|[^\s()]+):e\d+\)?", "", str(report or ""),
        )
        cleaned = re.sub(
            r"^(#{1,6}\s+.*?)(?:\s*\[\d+\]){1,}\s*$", r"\1",
            cleaned, flags=re.MULTILINE,
        )

        def limit_cluster(match: re.Match) -> str:
            numbers = list(dict.fromkeys(re.findall(r"\d+", match.group(0))))[:2]
            return "".join(f"[{number}]" for number in numbers)

        cleaned = re.sub(r"(?:\s*\[\d+\]){3,}", limit_cluster, cleaned)
        # Models commonly place a citation immediately after Chinese sentence
        # punctuation ("结论。 [3]").  Move it before the punctuation so the
        # statement-level verifier and the reader both see an unambiguous bond.
        cleaned = re.sub(
            r"([。！？.!?])\s*((?:\[\d+\]){1,2})(?=\s|$)", r"\2\1", cleaned,
        )
        cleaned = re.sub(r"[ \t]+([，。；：,.!?])", r"\1", cleaned)
        return cleaned

    def _resolve_unsupported_paragraphs(
        self,
        report: str,
        evidence_cards: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> tuple:
        """Bind an unsupported paragraph to verified evidence or remove it."""
        marked = []
        repaired = 0
        source_number = {
            source.get("source_id"): index for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        candidates = [
            (card, source_number.get(card.get("source_id")))
            for card in evidence_cards if source_number.get(card.get("source_id"))
        ]
        blocks = re.split(r"(\n\s*\n)", report)
        for index in range(0, len(blocks), 2):
            block = blocks[index]
            clean = " ".join(line.strip() for line in block.splitlines()).strip()
            if len(clean) <= 80 or re.search(r"\[\d+\]", clean):
                continue
            if clean.startswith(("#", "|", ">", "- ", "*")):
                continue
            lowered = clean.lower()
            if any(term in lowered for term in (
                "guiding question", "核心问题", "citation validation", "引用校验",
                "fixes applied", "修复摘要", "unresolved", "未解决",
            )):
                continue
            tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", clean.lower()))
            scored = []
            for card, number in candidates:
                evidence_text = " ".join(str(card.get(field) or "") for field in (
                    "claim", "method", "dataset", "metric", "result", "limitation",
                ))
                card_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", evidence_text.lower()))
                score = len(tokens & card_tokens) / max(1, len(tokens | card_tokens))
                scored.append((score, number))
            score, number = max(scored, default=(0.0, None))
            marked.append(f"paragraph-{len(marked) + 1}")
            if number and score >= 0.04:
                blocks[index] = block.rstrip() + f" [{number}]"
                repaired += 1
            else:
                blocks[index] = ""
                if index + 1 < len(blocks):
                    blocks[index + 1] = ""
        return "".join(blocks), marked, repaired

    def _mark_unsupported_paragraphs(self, report: str) -> tuple:
        """Backward-compatible wrapper; placeholders are no longer emitted."""
        cleaned, paragraph_ids, _ = self._resolve_unsupported_paragraphs(report, [], [])
        return cleaned, paragraph_ids

    @staticmethod
    def _mark_unsupported_paragraphs_legacy(report: str) -> tuple:
        """Retain the pre-formal-report behavior for direct unit-level callers."""
        marked = []
        blocks = re.split(r"(\n\s*\n)", report)
        for index in range(0, len(blocks), 2):
            block = blocks[index]
            clean = " ".join(line.strip() for line in block.splitlines()).strip()
            if len(clean) <= 80 or re.search(r"\[\d+\]", clean):
                continue
            if clean.startswith(("#", "|", ">", "- ", "*")):
                continue
            blocks[index] = block.rstrip() + " [citation needed]"
            marked.append(clean[:200])
        return "".join(blocks), marked

    def _merge_duplicate_chapters(self, report: str) -> tuple:
        """Merge adjacent substantive H2 chapters only when Jaccard is above 70%."""
        from app.agents.evaluator import Evaluator

        pattern = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
        matches = list(pattern.finditer(report))
        if len(matches) < 2:
            return report, []
        prefix = report[:matches[0].start()]
        sections = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
            sections.append([match.group(1).strip(), report[match.end():end].strip()])
        excluded = ("summary", "摘要", "conclusion", "结论", "source", "来源", "evidence gap", "证据空白")
        merged = []
        output = []
        index = 0
        while index < len(sections):
            heading, body = sections[index]
            if index + 1 < len(sections):
                next_heading, next_body = sections[index + 1]
                substantive = not any(term in heading.lower() or term in next_heading.lower() for term in excluded)
                similarity = Evaluator._bigram_jaccard(body, next_body) if substantive else 0.0
                if similarity > 0.70:
                    unique_lines = list(dict.fromkeys(
                        line for line in (body + "\n" + next_body).splitlines() if line.strip()
                    ))
                    output.append((f"{heading} / {next_heading}", "\n".join(unique_lines)))
                    merged.append(f"{heading} + {next_heading} ({similarity:.0%})")
                    index += 2
                    continue
            output.append((heading, body))
            index += 1
        rebuilt = prefix.rstrip() + "\n\n" + "\n\n".join(
            f"## {heading}\n\n{body}" for heading, body in output
        )
        return rebuilt.strip(), merged

    def _repair_citation_mismatches(
        self,
        report: str,
        mismatch_ids: Set[int],
        evidence_cards: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> tuple:
        """Replace invalid source numbers with the closest evidence-bearing source."""
        if not mismatch_ids:
            return report, []
        source_number = {
            source.get("source_id"): index for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        candidates = [
            (card, source_number.get(card.get("source_id")))
            for card in evidence_cards if source_number.get(card.get("source_id"))
        ]
        repairs = []
        lines = report.splitlines()
        for line_index, line in enumerate(lines):
            bad = {int(value) for value in re.findall(r"\[(\d+)\]", line)} & mismatch_ids
            if not bad or not candidates:
                continue
            line_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", line.lower()))
            scored = []
            for card, number in candidates:
                card_tokens = set(re.findall(
                    r"[\w\u4e00-\u9fff]+", str(card.get("claim", "")).lower()
                ))
                score = len(line_tokens & card_tokens) / max(1, len(line_tokens | card_tokens))
                scored.append((score, number))
            score, replacement = max(scored)
            for old in bad:
                if score > 0:
                    line = line.replace(f"[{old}]", f"[{replacement}]")
                    repairs.append(f"[{old}] -> [{replacement}]")
                else:
                    line = ""
                    repairs.append(f"[{old}] -> removed unsupported statement")
            lines[line_index] = line
        return "\n".join(lines), repairs

    @staticmethod
    def _strip_internal_sections(report: str) -> str:
        """Remove legacy audit/debug sections from the formal report."""
        internal = re.compile(
            r"(?:FinalReviewer|Fixes Applied|修复摘要|已应用的修复|"
            r"Unresolved Evidence and Execution Issues|未解决的证据与执行问题|"
            r"Citation Validation Audit|引用校验审计|Supplementary Evidence|补充证据)",
            re.IGNORECASE,
        )
        lines = report.splitlines()
        output = []
        skip_level = None
        for line in lines:
            heading = re.match(r"^(#{2,3})\s+(.+)$", line.strip())
            if heading:
                level = len(heading.group(1))
                if internal.search(heading.group(2)):
                    skip_level = level
                    continue
                if skip_level is not None and level <= skip_level:
                    skip_level = None
            if skip_level is None:
                output.append(line)
        return "\n".join(output).strip()

    @staticmethod
    def _remove_internal_paragraphs(report: str) -> str:
        terms = re.compile(
            r"(?:Evidence Cards?|FinalReviewer|draft_reviewer|Agent Trace|"
            r"latency|规则通过率|评估结果：|逐章生成|证据卡绑定|token budget)",
            re.IGNORECASE,
        )
        blocks = re.split(r"(\n\s*\n)", report)
        for index in range(0, len(blocks), 2):
            if terms.search(blocks[index]):
                blocks[index] = ""
                if index + 1 < len(blocks):
                    blocks[index + 1] = ""
        return "".join(blocks).strip()

    @staticmethod
    def _remove_citation_needed(report: str) -> tuple[str, int]:
        """Delete statements containing an unresolved citation placeholder."""
        removed = 0
        output = []
        for line in report.splitlines():
            if re.search(r"\[citation needed\]", line, re.IGNORECASE):
                removed += 1
                continue
            output.append(line)
        return "\n".join(output), removed

    @classmethod
    def _completion_gate(
        cls,
        report: str,
        topic: str,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        outline: Dict[str, Any],
    ) -> List[str]:
        """Hard reader-facing acceptance gate; failures produce a partial run."""
        issues = []
        lowered = report.lower()
        if "[citation needed]" in lowered:
            issues.append("最终报告仍包含 [citation needed]")
        if "file://" in lowered:
            issues.append("最终报告包含本地 file:// 路径")
        internal_terms = (
            "evidence card", "finalreviewer", "draft_reviewer", "agent trace",
            "latency_ms", "latency exceeded threshold", "slowest observed node",
            "规则通过率", "评估结果：", "逐章生成", "证据卡绑定", "总延迟超过阈值",
        )
        if any(term in lowered for term in internal_terms):
            issues.append("正式报告混入 Agent 执行或评估信息")

        topic_text = str(topic or "").lower()
        comprehensive = (
            any(cue in topic_text for cue in ("mmea", "multimodal entity alignment", "多模态实体对齐"))
            or (
                any(cue in topic_text for cue in ("主要方法", "methods", "method taxonomy"))
                and any(cue in topic_text for cue in ("数据集", "datasets", "benchmarks"))
                and any(cue in topic_text for cue in ("研究局限", "limitations", "open problems"))
            )
        )
        if comprehensive:
            off_task = [
                source for source in sources
                if source.get("task_relevance") == "excluded"
                or source.get("research_task") not in (None, "", "multimodal_entity_alignment")
            ]
            if off_task:
                issues.append(f"存在 {len(off_task)} 篇跨任务来源进入正式报告")
            excluded_report_tasks = (
                "多模态实体链接", "知识图谱补全", "情感分类", "人类偏好对齐",
                "多模态检索增强生成", "课堂话语", "时间序列预测", "文本属性图",
                "multimodal entity linking", "knowledge graph completion",
                "sentiment classification", "human preference", "multimodal rag",
                "classroom discourse", "time series forecasting", "text-attributed graph",
            )
            scope_checked_blocks = []
            for block in re.split(r"(?<=[。！？.!?])\s*|\n\s*\n", report):
                block_lower = block.lower()
                mentions_adjacent = any(cue in block_lower for cue in excluded_report_tasks)
                explicit_exclusion = bool(re.search(
                    r"(?:不(?:扩展到|延伸至|包括|包含|属于|涉及|等同于)|排除|区别于|边界(?:限定|排除)|"
                    r"exclude|not include|distinct from|outside (?:the )?scope)",
                    block_lower,
                    re.IGNORECASE,
                )) or (
                    bool(re.search(
                        r"(?:任务边界|相邻任务|调研范围|研究范围|task boundary|adjacent task|research scope)",
                        block_lower,
                        re.IGNORECASE,
                    ))
                    and bool(re.search(r"(?:不|排除|区别|exclude|outside)", block_lower, re.IGNORECASE))
                )
                if not (mentions_adjacent and explicit_exclusion):
                    scope_checked_blocks.append(block_lower)
            if any(
                cue in "\n".join(scope_checked_blocks)
                for cue in excluded_report_tasks
            ):
                issues.append("正式报告混入多模态实体对齐之外的相邻研究任务")
            required = {
                "方法": ("## 方法", "主要方法", "method taxonomy"),
                "数据集": ("## 数据集", "常用数据集", "common datasets"),
                "评价指标": ("## 评价指标", "评估指标与实验协议", "metrics and experimental protocols"),
                "实验": ("## 实验", "代表性实验结果", "representative experimental results"),
                "局限": ("## 局限", "研究局限与开放问题", "research limitations and open problems"),
                "结论": ("## 结论", "## conclusion"),
            }
            for label, cues in required.items():
                if not any(cue in lowered for cue in cues):
                    issues.append(f"缺少核心章节：{label}")
            for label, cues in {
                "方法": ("方法", "method taxonomy"),
                "数据集": ("数据集", "common datasets"),
                "局限": ("局限", "research limitations"),
            }.items():
                section = cls._find_section(report, cues)
                refs = set(re.findall(r"\[(\d+)\]", section))
                if len(refs) < 2:
                    issues.append(f"{label}章节少于两个独立来源")
            facet_sources = {
                "主要方法": {card.get("source_id") for card in evidence_cards if card.get("method")},
                "数据集": {card.get("source_id") for card in evidence_cards if card.get("dataset")},
                "评估协议": {
                    card.get("source_id") for card in evidence_cards
                    if card.get("metric") or card.get("experimental_setting")
                },
                "研究局限": {card.get("source_id") for card in evidence_cards if card.get("limitation")},
            }
            for label, source_ids in facet_sources.items():
                if len({item for item in source_ids if item}) < 2:
                    issues.append(f"{label}缺少两篇独立论文的结构化证据")
            method_families = {
                family.strip()
                for card in evidence_cards
                for family in str(card.get("method_family") or "").split("；")
                if family.strip()
            }
            if len(method_families) < 2:
                issues.append("主要方法未形成至少两个按技术机制划分的方法家族")
            dataset_detail_sources = {
                card.get("source_id") for card in evidence_cards
                if card.get("source_id") and card.get("dataset") and any(card.get(field) for field in (
                    "dataset_name", "graph_or_language_pair", "entity_count", "modalities",
                    "missingness", "data_split", "seed_ratio",
                ))
            }
            if len(dataset_detail_sources) < 2:
                issues.append("数据集章节缺少两篇独立来源的规模、模态或划分细节")

        # Citations are intentionally paragraph-level to avoid the unreadable
        # citation clusters reported by users.  Validate the same unit that the
        # chapter prompt requires instead of demanding one marker per sentence.
        for paragraph in re.split(r"\n\s*\n", report):
            if paragraph.lstrip().startswith("#"):
                continue
            if not re.search(r"(?:首次|首个|显著|优于|突破|first|significant|outperform|breakthrough)", paragraph, re.IGNORECASE):
                continue
            if re.search(
                r"(?:作者(?:报告|称|指出)|研究报告(?:称|指出|报告)|仅报告|"
                r"未给出|没有给出|无法(?:独立)?验证|"
                r"不能(?:独立)?验证|只能(?:作为|视为)|证据不足|"
                r"reported by the authors|not independently verified|"
                r"insufficient evidence)",
                paragraph,
                re.IGNORECASE,
            ):
                continue
            if not re.search(r"\[\d+\]", paragraph):
                issues.append("存在无声明级引用的强结论")
                break
            strong_performance = r"(?:显著|优于|outperform|significant)"
            performance_context = r"(?:性能|准确|指标|得分|结果|提升|performance|accuracy|metric|score|result)"
            performance = re.search(
                rf"(?:{strong_performance}.{{0,36}}{performance_context}|"
                rf"{performance_context}.{{0,36}}{strong_performance})",
                paragraph,
                re.IGNORECASE | re.DOTALL,
            )
            prose_without_refs = re.sub(r"\[\d+\]", "", paragraph)
            if performance and not re.search(r"\d", prose_without_refs):
                issues.append("性能强结论缺少数值或提升幅度")
                break
        if not sources or not evidence_cards:
            issues.append("缺少可追溯来源或证据")
        if comprehensive:
            referenced_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", report)}
            incomplete_metadata = []
            for number in referenced_numbers:
                if not 1 <= number <= len(sources):
                    continue
                source = sources[number - 1]
                has_identifier = bool(
                    str(source.get("url") or "").startswith(("http://", "https://"))
                    or source.get("doi") or source.get("openalex_id") or source.get("semantic_scholar_id")
                )
                if not source.get("authors") or not source.get("year") or not has_identifier:
                    incomplete_metadata.append(number)
            if incomplete_metadata:
                issues.append(f"参考文献元数据不完整：{sorted(set(incomplete_metadata))}")
        return list(dict.fromkeys(issues))

    @staticmethod
    def _requires_strict_completion(topic: str) -> bool:
        text = str(topic or "").lower()
        return (
            any(cue in text for cue in ("mmea", "multimodal entity alignment", "多模态实体对齐"))
            or (
                any(cue in text for cue in ("主要方法", "methods", "method taxonomy"))
                and any(cue in text for cue in ("数据集", "datasets", "benchmarks"))
                and any(cue in text for cue in ("研究局限", "limitations", "open problems"))
            )
        )

    @staticmethod
    def _find_section(report: str, cues: tuple[str, ...]) -> str:
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", report, re.MULTILINE))
        for index, match in enumerate(matches):
            if any(cue in match.group(1).lower() for cue in cues):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
                return report[match.end():end]
        return ""

    # ================================================================
    # 修复 1：删除假引用标记
    # ================================================================

    def _collect_invalid_ids(
        self, citation_check_results: List[Dict[str, Any]]
    ) -> Set[int]:
        """收集所有 is_valid=False 的 citation id。"""
        return {
            r.get("citation_id", 0)
            for r in citation_check_results
            if not r.get("is_valid", False) and r.get("citation_id", 0) > 0
        }

    def _collect_valid_source_ids(
        self, citation_check_results: List[Dict[str, Any]]
    ) -> Set[str]:
        """收集所有 is_valid=True 的 source_id。"""
        return {
            r.get("source_id", "")
            for r in citation_check_results
            if r.get("is_valid", False)
        }

    def _delete_fake_citations(
        self, report: str, invalid_ids: Set[int]
    ) -> tuple:
        """
        删除报告中无效引用标记。

        策略：
        1. 删除独立的 [N] 标记
        2. 如果一句话里所有引用都是假的，删除整句
        3. 如果一句话里有真有假，只删除假的 [N]

        返回：(修改后的报告, 删除的标记数)
        """
        removed_count = 0

        # Step 1: 逐句处理
        lines = report.split("\n")
        new_lines = []

        for line in lines:
            # 找到所有 [N]
            refs = list(re.finditer(r'\[(\d+)\]', line))
            if not refs:
                new_lines.append(line)
                continue

            # 判断每个引用是否有效
            all_invalid = all(
                int(m.group(1)) in invalid_ids for m in refs
            )

            if all_invalid and len(refs) >= 1:
                # 整句只有假引用 → 删除整句
                removed_count += len(refs)
                continue  # 跳过这一行

            # 部分假引用 → 只删除假的 [N]
            new_line = line
            for m in reversed(refs):  # 从后往前替换，避免下标偏移
                if int(m.group(1)) in invalid_ids:
                    new_line = new_line[:m.start()] + new_line[m.end():]
                    removed_count += 1

            # 清理多余空格
            new_line = re.sub(r'  +', ' ', new_line).strip()
            if new_line:
                new_lines.append(new_line)

        return "\n".join(new_lines), removed_count

    # ================================================================
    # 修复 2：清理引用 section 中的无效行
    # ================================================================

    def _clean_citation_section(
        self, report: str, invalid_ids: Set[int]
    ) -> tuple:
        """
        在 Citation Validation / Sources Retrieved 等 section 中，
        删除引用无效来源的表格行和列表项。
        """
        removed = 0
        lines = report.split("\n")
        new_lines = []
        in_citation_section = False

        for line in lines:
            # 检测是否进入了引用相关的 section
            if line.startswith("## ") and any(
                kw in line.lower()
                for kw in ("citation", "source", "reference")
            ):
                in_citation_section = True
            elif line.startswith("## "):
                in_citation_section = False

            # 在引用 section 中，检查是否包含无效引用 ID
            if in_citation_section:
                refs = re.findall(r'\[(\d+)\]', line)
                if refs and all(int(r) in invalid_ids for r in refs):
                    removed += 1
                    continue  # 删除这行

            new_lines.append(line)

        return "\n".join(new_lines), removed

    # ================================================================
    # 修复 3：URL 强制覆盖（确定性修复，不需要 LLM）
    # ================================================================

    def _fix_urls_from_sources(
        self,
        report: str,
        url_mismatches: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> tuple:
        """
        用 PaperSource 中的权威 URL 覆盖报告中错误的 URL。

        策略：对于每个 url_matches_source=False 但 id_exists=True 的引用，
        从 sources 中找到对应 source_id 的正确 URL，替换报告中出现的错误 URL。

        这是确定性修复——PaperSource.url 是权威来源，不需要 LLM 判断。
        """
        # 构建 source_id → correct_url 索引
        source_urls: Dict[str, str] = {}
        for s in sources:
            sid = s.get("source_id", "")
            url = s.get("url", "").strip().rstrip("/")
            if sid and url:
                source_urls[sid] = url

        fixed_count = 0
        result = report

        for mismatch in url_mismatches:
            source_id = mismatch.get("source_id", "")
            correct_url = source_urls.get(source_id, "")
            if not correct_url:
                continue

            # 从 citation_check_result 中找出当前的错误 URL（如果 mismatch 里有记录的话）
            # mismatch 中没有存错误的 URL，我们需要从报告中找
            # 策略：找报告中与该 source_id 关联的错误 URL 模式

            # 更直接的做法：扫描报告中所有 URL，如果某个 URL 不在 sources 中，
            # 且在某个已知 source_id 的上下文中，就替换为正确 URL
            # 简化：对每个 url_mismatch，找报告中含 source_id 的行，替换其中的 URL

            # 先尝试最简单的：如果正确 URL 已经在报告中，不需要修复
            if correct_url in result:
                continue

            # 对包含该 source_id 上下文的行，替换 URL
            # 匹配报告中常见的 URL 格式
            for wrong_url_pattern in [
                rf'\(https?://[^\s\)]+\)',   # markdown link: (url)
                rf'https?://[^\s\)\]]+',      # bare URL
            ]:
                # 找到报告中所有 URL
                found_urls = re.findall(wrong_url_pattern, result)
                for found in found_urls:
                    found_clean = found.strip().rstrip("/")
                    # 如果这个 URL 不在任何 source 中，它可能是错误的
                    if found_clean not in source_urls.values():
                        # 替换为正确 URL
                        result = result.replace(found, correct_url)
                        fixed_count += 1
                        break  # 每个 mismatch 只修一次
                else:
                    continue
                break

        return result, fixed_count

    # ================================================================
    # 修复 4：补全过短报告
    # ================================================================

    def _expand_thin_report(
        self,
        report: str,
        evidence_cards: List[Dict[str, Any]],
        language: str = "en",
    ) -> tuple:
        """追加有效证据卡的内容。"""
        if not evidence_cards:
            return report, 0

        zh = str(language or "").lower().replace("_", "-").startswith("zh")
        lines = (
            [
                "", "---", "", "## 补充证据", "",
                "_由于初始报告低于最低内容阈值，系统自动追加以下证据。_", "",
            ]
            if zh else
            [
                "", "---", "", "## Supplementary Evidence", "",
                "_The following evidence was automatically appended because the "
                "initial report was below the minimum content threshold._", "",
            ]
        )

        added = 0
        for i, card in enumerate(evidence_cards[:5]):
            claim = card.get("claim", "")
            method = card.get("method", "")
            limitation = card.get("limitation", "")
            confidence = card.get("confidence", 1.0)

            lines.append(
                f"### 证据 [{i + 1}]（置信度：{confidence:.2f}）"
                if zh else
                f"### Evidence [{i + 1}] (confidence: {confidence:.2f})"
            )
            lines.append(f"- **{'主张' if zh else 'Claim'}**: {claim}")
            if method:
                lines.append(f"- **{'方法' if zh else 'Method'}**: {method}")
            if limitation:
                lines.append(f"- **{'局限' if zh else 'Limitation'}**: {limitation}")
            lines.append("")
            added += 1

        return report.rstrip() + "\n" + "\n".join(lines), added

    # ================================================================
    # 修复 4：警告横幅
    # ================================================================

    def _build_warning_banners(
        self,
        metrics: Dict[str, Any],
        citation_check_results: List[Dict[str, Any]],
        language: str = "en",
    ) -> List[str]:
        """构建警告横幅。"""
        banners = []
        zh = str(language or "").lower().replace("_", "-").startswith("zh")

        if not metrics.get("no_fake_citation", True):
            invalid_count = sum(
                1 for r in citation_check_results if not r.get("is_valid", False)
            )
            banners.append(
                (
                    f"> ⚠️ **引用警告**：评估器检测到 {invalid_count} 个无效引用，"
                    "已将其从报告中**删除**。使用前请核验其余引用。"
                    if zh else
                    f"> ⚠️ **Citation Warning**: {invalid_count} fake citation(s) were "
                    f"detected by the Evaluator. These have been **removed** from the report. "
                    f"Please verify remaining citations before use."
                )
            )

        if not metrics.get("min_sources", True):
            banners.append(
                "> ⚠️ **覆盖范围警告**：来源数量低于最低阈值，可能需要扩大检索范围。"
                "详见下方“未解决的证据与执行问题”。"
                if zh else
                "> ⚠️ **Coverage Warning**: The number of sources is below the minimum "
                "threshold. A broader search may be needed to ensure comprehensive coverage. "
                "See UNRESOLVED ISSUES below."
            )

        if not metrics.get("evidence_available", True):
            banners.append(
                "> **证据警告**：没有通过引用校验的证据卡。报告不包含可验证的研究发现，"
                "应视为证据不足的结果。"
                if zh else
                "> **Evidence Warning**: No citation-verified EvidenceCards were available. "
                "The report contains no research findings and should be treated as an "
                "insufficient-evidence result."
            )

        if not metrics.get("tool_error_rate", True):
            banners.append(
                "> ⚠️ **可靠性警告**：调研过程中出现工具错误，部分结果可能不完整。"
                "详见下方“未解决的证据与执行问题”。"
                if zh else
                "> ⚠️ **Reliability Warning**: Tool errors occurred during research. "
                "Some results may be incomplete. See UNRESOLVED ISSUES below."
            )

        return banners

    def _insert_banners_after_title(self, report: str, banners: List[str]) -> str:
        """在标题后插入横幅。"""
        if not banners:
            return report

        banner_block = "\n".join(banners)
        lines = report.split("\n")
        insert_idx = 0
        found_title = False

        for i, line in enumerate(lines):
            if line.startswith("# ") and not found_title:
                found_title = True
                insert_idx = i + 1
                while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                    insert_idx += 1
                break

        if not found_title:
            return banner_block + "\n\n" + report

        result_lines = (
            lines[:insert_idx] + [""] + [banner_block] + [""] + lines[insert_idx:]
        )
        return "\n".join(result_lines)

    # ================================================================
    # UNRESOLVED ISSUES
    # ================================================================

    def _collect_unresolved(
        self,
        metrics: Dict[str, Any],
        metrics_detail: Dict[str, Any] = None,
        language: str = "en",
    ) -> List[str]:
        """
        收集当前有界研究运行无法自动修复的问题。

        这些信息会进入最终报告，因此只描述事实、影响和可执行建议，
        不暴露项目内部开发编号。
        """
        issues = []
        metrics_detail = metrics_detail or {}
        zh = str(language or "").lower().replace("_", "-").startswith("zh")

        if not metrics.get("min_sources", True):
            issues.append(
                "[未解决] 来源数量不足：报告可能无法充分覆盖主题；请使用更聚焦的概念、"
                "同义词或更高的来源上限重新运行检索。"
                if zh else
                "[UNRESOLVED] min_sources failed: too few sources. "
                "The report may not cover the topic adequately; rerun the search "
                "with narrower concepts, synonyms, or a broader source limit."
            )

        if not metrics.get("evidence_available", True):
            issues.append(
                "[未解决] 缺少可用证据：检索来源未形成通过引用校验的证据。分析 Worker "
                "已进行有界重试；剩余空白需要带摘要或可访问全文的来源。"
                if zh else
                "[UNRESOLVED] evidence_available failed: retrieved sources did not yield "
                "citation-verified evidence. The Analysis Worker retried with deterministic "
                "coverage; remaining gaps require sources with abstracts or accessible full text."
            )

        if not metrics.get("citation_id_exists", True):
            issues.append(
                "[未解决] 部分引用 ID 不对应任何来源。无效 ID 已从报告删除，但生成阶段"
                "出现引用幻觉的根因仍保留在 Trace 中供检查。"
                if zh else
                "[UNRESOLVED] citation_id_exists failed: some citation IDs "
                "do not correspond to any source. Fake IDs have been deleted "
                "from the report, but the underlying cause (LLM hallucination "
                "during generation) remains recorded for review."
            )

        # source_url_valid is now handled deterministically by _fix_urls_from_sources
        # (PaperSource.url is authoritative — no LLM needed).
        # If it still fails after fix, that means the source_id itself doesn't exist,
        # which is already covered by citation_id_exists.

        if not metrics.get("task_success_rate", True) or metrics.get("task_success_rate", 1.0) < 1.0:
            issues.append(
                "[未解决] 任务成功率低于 100%：并非所有任务都成功完成。依赖失败任务的"
                "发现可能不完整；请检查执行 Trace 并重新运行受影响的调研任务。"
                if zh else
                "[UNRESOLVED] task_success_rate < 100%: not all tasks completed "
                "successfully. Findings that depended on failed tasks may be incomplete; "
                "inspect the execution trace and rerun the affected research tasks."
            )

        if not metrics.get("tool_error_rate", True):
            issues.append(
                "[未解决] 工具错误率超过阈值。本次运行的重试与回退均受预算限制；若错误"
                "持续出现，请检查提供方可用性、凭据、配额与工具参数。"
                if zh else
                "[UNRESOLVED] tool_error_rate exceeded threshold. Retrying "
                "and fallback were bounded for this run. Persistent errors require "
                "checking provider availability, credentials, quotas, and tool inputs."
            )

        if not metrics.get("latency_under_threshold", True):
            latency = metrics_detail.get("latency", {})
            slowest = latency.get("slowest_node") or {}
            slowest_hint = ""
            if slowest.get("name") and slowest.get("latency_ms") is not None:
                slowest_hint = (
                    f" Slowest observed node: {slowest['name']} "
                    f"({slowest['latency_ms']}ms)."
                )
            if zh:
                slowest_hint_zh = ""
                if slowest.get("name") and slowest.get("latency_ms") is not None:
                    slowest_hint_zh = (
                        f" 最慢节点：{slowest['name']}（{slowest['latency_ms']} 毫秒）。"
                    )
                issues.append(
                    "[未解决] 总延迟超过阈值。" + slowest_hint_zh
                    + " 请检查 Trace 中的慢速工具或模型调用；适当降低来源与证据预算，"
                    "保留单工具超时，并使用 Send API 并行执行相互独立的任务。"
                )
            else:
                issues.append(
                    "[UNRESOLVED] latency exceeded threshold."
                    + slowest_hint
                    + " Check the trace for slow tools "
                    "or model calls; reduce source and evidence budgets, preserve per-tool "
                    "timeouts, and parallelize independent tasks with Send API."
                )

        return issues

    # ================================================================
    # 修复摘要
    # ================================================================

    def _append_fix_summary(
        self,
        report: str,
        fixes_applied: List[str],
        unresolved_issues: List[str],
        metrics: Dict[str, Any],
        unsupported_paragraphs: List[str] = None,
        merged_sections: List[str] = None,
        citation_repairs: List[str] = None,
        language: str = "en",
    ) -> str:
        """在报告末尾添加诚实的修复摘要。"""
        detail = metrics.get("metrics_detail", {})
        unsupported_paragraphs = unsupported_paragraphs or []
        merged_sections = merged_sections or []
        citation_repairs = citation_repairs or []
        passed = detail.get("passed_count", 0)
        total = detail.get("total_count", 0)
        zh = str(language or "").lower().replace("_", "-").startswith("zh")

        lines = ["", "---", ""]

        if fixes_applied:
            lines.append("### FinalReviewer 已应用的修复" if zh else "### Fixes Applied by FinalReviewer")
            for fix in fixes_applied:
                lines.append(f"- ✅ {fix}")
            lines.append("")

        if unsupported_paragraphs:
            lines.append("### 缺少引用的段落" if zh else "### Unsupported Paragraphs")
            lines.extend(f"- {item}" for item in unsupported_paragraphs)
            lines.append("")
        if merged_sections:
            lines.append("### 已合并章节" if zh else "### Merged Sections")
            lines.extend(f"- {item}" for item in merged_sections)
            lines.append("")
        if citation_repairs:
            lines.append("### 引用编号修复" if zh else "### Citation Binding Repairs")
            lines.extend(f"- {item}" for item in citation_repairs)
            lines.append("")

        if unresolved_issues:
            lines.append(
                "### 未解决的证据与执行问题"
                if zh else
                "### Unresolved Evidence and Execution Issues"
            )
            lines.append("")
            lines.append(
                "_这些问题由评估器在有界重试与确定性修复后识别。保留它们是为了避免"
                "报告夸大证据质量。_"
                if zh else
                "_These issues were identified by the Evaluator after bounded retry and "
                "deterministic repair. They are retained so the report does not overstate "
                "its evidence quality._"
            )
            lines.append("")
            for issue in unresolved_issues:
                lines.append(f"- {issue}")
            lines.append("")

        # 简短评估摘要
        if total > 0:
            lines.append(
                (
                    f"*评估结果：{passed}/{total} 项规则指标通过；已应用 "
                    f"{len(fixes_applied)} 项修复，仍有 {len(unresolved_issues)} 个问题未解决。*"
                )
                if zh else
                (
                    f"*Evaluation: {passed}/{total} rule metrics passed. "
                    f"{len(fixes_applied)} fix(es) applied, "
                    f"{len(unresolved_issues)} issue(s) unresolved.*"
                )
            )

        return report.rstrip() + "\n\n" + "\n".join(lines)
