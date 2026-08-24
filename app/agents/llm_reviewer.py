"""
app/agents/llm_reviewer.py

LLMDraftReviewer —— DeepSeek V4 Flash 驱动的章节生成与审核器。

职责:
- 按 ReportOutline 的章节计划,用 LLM 逐章生成正文、findings 和标题翻译
- 三道引用闸(声明/正文标记/规范化)拦截 LLM 的引用幻觉:
  有卡章节必须标且标记必须落在分配给本章的卡上;
  无卡章节(证据缺口章)禁止声明或正文出现任何引用 ID
- 跨源章节(方法/数据集/局限等)强制双源综合,防单源偏信
- 双语标题翻译只保留与来源严格匹配的条目
- 语言门:中文模式要求正文含足够中文字符,防语言漂移
- LLM 失败 → 回退模板 DraftReviewer(规则版章节)

约束:
- LLM 只能使用提供的 verified PaperSource 和 EvidenceCard
- citation ID 由程序统一分配,LLM 不得自由生成
- URL 必须来自 PaperSource
- 无 EvidenceCard 支持的 claim 不进入正式报告
- 生成后再次执行 CitationCheckTool 和 Rule Eval
- LLM 失败 → 回退模板 DraftReviewer
"""

import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from app.agents.draft_reviewer import DraftReviewer
from app.agents.report_style import academic_report_title, is_mmea_topic, report_keywords
from app.agents.source_titles import (
    display_paper_title,
    format_reference,
    is_chinese_title,
    validated_title_translations,
)
from app.llm.client import get_llm_client
from app.llm.prompts import (
    CHAPTER_SYSTEM,
    CHAPTER_SYSTEM_ZH,
    CHAPTER_USER,
    CHAPTER_USER_ZH,
)
from app.llm.schemas import LLMChapterOutput


class LLMDraftReviewer:
    """Evidence-bound chapter writer and deterministic report assembler."""

    def __init__(self):
        self._rule_reviewer = DraftReviewer()
        self._llm = get_llm_client()

    async def review(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        citation_summary: Dict[str, Any] = None,
        language: str = "en",
        outline: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """兼容入口：用于未使用 Send 机制的串行大纲路径。"""
        if outline and outline.get("sections"):
            return await self.assemble_report(
                topic, outline, sources, evidence_cards,
                citation_check_results, citation_summary or {}, language,
            )
        raise RuntimeError(
            "LLMDraftReviewer.review 仅支持已废弃的串行大纲路径；"
            "请改用 generate_chapter 或 graph_send 的章节工作节点。"
        )
    # review 方法说明：
    # 作为旧版串行流程的兼容入口存在，仅当传入的 outline 包含 sections 时，
    # 直接调用 assemble_report 生成完整报告；否则抛出异常，引导调用方迁移到
    # 新的章节化生成路径（generate_chapter 或 graph_send 的章节工作节点）。
    async def generate_chapter(
        # 生成单个章节正文：严格墙钟预算 + 证据隔离，LLM / 规则双路径。
        self,
        section: Dict[str, Any],
        evidence_cards: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        language: str = "zh",
        source_number: Dict[str, int] = None,
        allow_rule_fallback: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate one chapter with a physically isolated evidence context."""
        # 本章没有任何通过校验的证据 → 直接走规则版生成"证据缺口(gap)"章节
        if not evidence_cards and allow_rule_fallback:
            chapter = self._rule_reviewer.generate_chapter(
                section, [], sources, language, source_number=source_number,
            )
            return chapter, {"success": True, "skipped": True, "latency_ms": 0, "mode": "gap"}
        heading_text = str(section.get("heading") or "").lower()
        # "局限/limitation" 类章节必须跨至少两个独立来源综合,来源不足时回退规则版
        if any(term in heading_text for term in ("局限", "limitation", "open problem")):
            source_count = len({
                card.get("source_id") for card in evidence_cards if card.get("source_id")
            })
            if source_count < 2 and allow_rule_fallback:
                chapter = self._rule_reviewer.generate_chapter(
                    section, evidence_cards, sources, language, source_number=source_number,
                )
                return chapter, {
                    "success": True, "skipped": True, "latency_ms": 0, "mode": "gap",
                    "error": "Cross-source limitation synthesis requires two independent sources",
                }

        source_number = source_number or {
            source.get("source_id"): index for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        # 章节生成:证据隔离 + LLM 调用 + 硬校验封装为一次 attempt;校验失败且可修复时,
        # 带明确修正指令重调一次 LLM(有界 repair),复检通过才用,否则才硬失败/规则降级。
        first = await self._generate_chapter_attempt(
            section, evidence_cards, sources, language, source_number, allow_rule_fallback,
        )
        if not first["success"]:
            if first["error_type"] == "llm_generation":
                if not allow_rule_fallback:
                    raise RuntimeError(
                        "LLM-only chapter generation failed: "
                        + str(first["result"].get("error") or "unknown error")
                    )
                return self._rule_reviewer.generate_chapter(
                    section, evidence_cards, sources, language, source_number=source_number,
                ), first["result"]
            # 仅对可修复的校验错误重试一次;结构错误(证据绑定等)不重试。
            retry_instruction = self._chapter_repair_instruction(first["error"])
            if retry_instruction and not section.get("_retry_instruction"):
                repaired_section = dict(section)
                repaired_section["_retry_instruction"] = retry_instruction
                second = await self._generate_chapter_attempt(
                    repaired_section, evidence_cards, sources, language, source_number,
                    allow_rule_fallback,
                )
                if second["success"]:
                    first = second
                    first["result"]["repair"] = {"attempted": True, "success": True}
                else:
                    first["result"]["repair"] = {
                        "attempted": True,
                        "success": False,
                        "error": second["error"] or first["error"],
                    }
            if not first["success"]:
                if not allow_rule_fallback:
                    raise RuntimeError(
                        "LLM-only chapter validation failed: "
                        + str(first["result"].get("error") or first["error"] or "unknown error")
                    )
                return self._rule_reviewer.generate_chapter(
                    section, evidence_cards, sources, language, source_number=source_number,
                ), first["result"]

        # 将正文中的证据标记替换为读者可见的 [n] 引用编号,再拼成章节 markdown
        cited_synthesis = self._attach_statement_citations(
            first["output"].synthesis, first["used_ids"], first["by_id"], source_number,
        )
        chapter = "\n".join([
            f"## {section.get('heading') or first['output'].heading}".rstrip(), "",
            cited_synthesis,
        ])
        return chapter, first["result"]

    async def _generate_chapter_attempt(
        self,
        section: Dict[str, Any],
        evidence_cards: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        language: str,
        source_number: Dict[str, int],
        allow_rule_fallback: bool,
    ) -> Dict[str, Any]:
        """单次章节生成:证据隔离 → 构建 prompt → 调 LLM → 硬校验。

        返回 dict,失败时携带 error_type("llm_generation"|"validation")与 error,
        供 generate_chapter 决定重试(可修复的校验错误)还是硬失败(结构错误/生成失败)。
        """
        # 章节物理隔离:只保留本节证据卡,并把来源列表裁剪到这些卡引用的源,防止 LLM 串用其他章节上下文
        evidence_limit = max(2, int(os.getenv("LLM_CHAPTER_EVIDENCE_LIMIT", "4")))
        evidence_cards = self._select_diverse_evidence(evidence_cards, evidence_limit)
        selected_source_ids = {
            str(card.get("source_id") or "") for card in evidence_cards if card.get("source_id")
        }
        sources = [
            source for source in sources
            if str(source.get("source_id") or "") in selected_source_ids
        ]
        source_summary = "\n".join(
            f"source_id={source.get('source_id', '')}; title={self._clip_text(source.get('title'), 300)}; "
            f"authors={', '.join(source.get('authors', [])[:4]) if isinstance(source.get('authors'), list) else source.get('authors', '')}; "
            f"year={source.get('year', 'N/A')}; venue={source.get('venue', '')}; "
            f"source_type={source.get('source_type', 'unknown')}; doi={source.get('doi', '')}"
            for source in sources
        ) or "(none)"
        evidence_summary = "\n\n".join(
            f"[{card.get('evidence_id', '')}] source_id={card.get('source_id', '')}\n"
            f"claim={self._clip_text(card.get('claim'), 350)}\n"
            f"method={self._clip_text(card.get('method'), 200)}\n"
            f"method_family={self._clip_text(card.get('method_family'), 120)}\n"
            f"dataset={self._clip_text(card.get('dataset'), 250)}\n"
            f"dataset_name={self._clip_text(card.get('dataset_name'), 100)}\n"
            f"graph_or_language_pair={self._clip_text(card.get('graph_or_language_pair'), 150)}\n"
            f"entity_count={self._clip_text(card.get('entity_count'), 100)}\n"
            f"modalities={self._clip_text(card.get('modalities'), 150)}\n"
            f"missingness={self._clip_text(card.get('missingness'), 150)}\n"
            f"data_split={self._clip_text(card.get('data_split'), 120)}\n"
            f"seed_ratio={self._clip_text(card.get('seed_ratio'), 120)}\n"
            f"research_task={card.get('research_task', '')}\n"
            f"dataset_task_consistent={bool(card.get('dataset_task_consistent', True))}\n"
            f"metric={self._clip_text(card.get('metric'), 150)}\n"
            f"result={self._clip_text(card.get('result') or card.get('key_results'), 220)}\n"
            f"baseline={self._clip_text(card.get('baseline'), 150)}\n"
            f"experimental_setting={self._clip_text(card.get('experimental_setting'), 200)}\n"
            f"limitations={self._clip_text(card.get('limitation'), 200)}\n"
            f"evidence_type={card.get('evidence_type', 'primary_claim')}"
            for card in evidence_cards
        )
        zh = self._is_chinese(language)
        heading_text = str(section.get("heading") or "").lower()
        system_prompt = CHAPTER_SYSTEM_ZH if zh else CHAPTER_SYSTEM
        user_prompt = (
            CHAPTER_USER_ZH.format(
                heading=section.get("heading", ""),
                guiding_question=section.get("guiding_question", ""),
                source_summary=source_summary,
                evidence_summary=evidence_summary,
            )
            if zh else
            CHAPTER_USER.format(
                heading=section.get("heading", ""),
                guiding_question=section.get("guiding_question", ""),
                output_language="English",
                source_summary=source_summary,
                evidence_summary=evidence_summary,
            )
        )
        if not evidence_cards:
            user_prompt += (
                "\n本章没有通过校验的证据。仍需由 LLM 输出简短中文证据缺口说明，"
                "evidence_ids 和 findings 必须为空，正文不得添加证据标记。"
                if zh else
                "\nNo verified evidence is available. Write a short evidence-gap chapter; "
                "evidence_ids and findings must be empty and no evidence markers may be used."
            )
        retry_instruction = str(section.get("_retry_instruction") or "").strip()
        if retry_instruction:
            user_prompt += "\n\n严格修正要求：" + retry_instruction
        # 单章节一次 LLM 调用(不重试),超时/预算走独立 env 配置;repair 由 generate_chapter 兜底
        result = await self._llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=LLMChapterOutput,
            timeout_seconds=int(os.getenv("LLM_CHAPTER_TIMEOUT_SECONDS", "120")),
            max_retries=0,
        )
        # LLM 生成失败:由调用方决定抛错还是回退规则版章节
        if not result.get("success"):
            return {
                "success": False,
                "error_type": "llm_generation",
                "error": str(result.get("error") or "unknown error"),
                "result": result,
            }

        output: LLMChapterOutput = result["data"]
        # 校验论文标题翻译:只保留与来源严格匹配的条目
        title_translations = validated_title_translations(
            output.source_title_translations, sources,
        )
        result["source_title_translations"] = title_translations
        if self._is_chinese(language):
            result["missing_title_translation_source_ids"] = [
                str(source.get("source_id") or "")
                for source in sources
                if source.get("source_id")
                and not is_chinese_title(source.get("title"))
                and str(source.get("source_id")) not in title_translations
                and display_paper_title(source) == str(source.get("title") or "Unknown").strip()
            ]
        by_id = {card.get("evidence_id"): card for card in evidence_cards}
        # 正文 [[e:...]] 标记才是"实际用了哪些证据"的权威记录;evidence_ids 数组仅作辅助声明
        declared_ids = list(dict.fromkeys(output.evidence_ids))
        raw_marker_ids = self._extract_marker_ids(output.synthesis)
        marker_ids = self._normalize_marker_ids(raw_marker_ids, by_id)
        # 段落标记准确记录了正文中实际引用了哪些证据。模型偶尔会漏掉
        # evidence_ids 列表里某个本应存在的标记；只有当这个标记确实对应
        # 某张章节卡片时，才把它补回声明中。凡是对应不上的未知 ID 一律报错。
        used_ids = list(dict.fromkeys(marker_ids))
        cross_source_required = any(
            cue in heading_text
            for cue in ("方法", "数据集", "局限", "method taxonomy", "datasets", "limitations")
        )
        unknown_markers = [item for item in marker_ids if item not in by_id]
        # 硬校验:有证据时必须标注且不得用未知 ID,跨源章节须达两源;证据缺口章节禁止虚构 ID
        validation_error: str = ""
        # 语言门:中文模式要求正文含足够中文字符,防止 LLM 输出语言漂移
        if zh and not self._text_has_sufficient_chinese(output.synthesis):
            validation_error = "Chapter output language is not Simplified Chinese"
        if evidence_cards:
            if not marker_ids:
                validation_error = "Chapter omitted paragraph-level evidence markers"
            elif unknown_markers:
                validation_error = (
                    f"Chapter used unknown paragraph-level evidence marker(s): {unknown_markers[:3]}"
                )
            elif (
                cross_source_required
                and (
                    not allow_rule_fallback
                    or len({card.get("source_id") for card in evidence_cards if card.get("source_id")}) >= 2
                )
                and len({by_id[item].get("source_id") for item in used_ids if item in by_id}) < 2
            ):
                validation_error = "Chapter failed the two-source synthesis gate"
        elif declared_ids or marker_ids:
            validation_error = "Evidence-gap chapter invented Evidence Card IDs"
        # 逐条校验 finding:绑定证据必须存在,且 source_id 与证据来源一致
        for finding in output.findings:
            bound_ids = finding.bound_evidence_ids()
            if not bound_ids or any(item not in by_id for item in bound_ids):
                validation_error = "Finding used an unverified Evidence Card"
                break
            if not any(by_id[item].get("source_id") == finding.source_id for item in bound_ids):
                # source_id remains the primary source for backward compatibility;
                # evidence_ids may additionally bind corroborating sources.
                validation_error = "Finding source/evidence binding mismatch"
                break
        if validation_error:
            result.update(success=False, error=validation_error)
            return {
                "success": False,
                "error_type": "validation",
                "error": validation_error,
                "result": result,
                "output": output,
                "by_id": by_id,
                "used_ids": used_ids,
            }
        return {
            "success": True,
            "error_type": None,
            "error": None,
            "result": result,
            "output": output,
            "by_id": by_id,
            "used_ids": used_ids,
        }

    @staticmethod
    def _chapter_repair_instruction(error: str) -> str:
        """把可修复的章节校验错误翻译成给 LLM 的修正指令;不可修复(结构错误)返回空串。"""
        message = str(error or "")
        if "two-source synthesis gate" in message:
            return (
                "本章被判定为跨来源综合章节，但正文只引用了单一来源的证据。"
                "必须同时引用至少 2 个不同来源的证据卡标记 [[e:evidence_id]]，"
                "并显式对比、综合这些来源之间的观点与差异。"
            )
        if "omitted paragraph-level evidence markers" in message:
            return (
                "正文中每一处基于证据的陈述之后，都必须紧跟对应的证据标记 "
                "[[e:evidence_id]]，不得遗漏。"
            )
        if "unknown paragraph-level evidence marker" in message:
            return (
                "正文只能引用给定证据卡列表中真实存在的 evidence_id；"
                "任何出现在 [[e:...]] 里的 ID 都必须能在卡片列表中找到，不得使用列表外的 ID。"
            )
        if "invented Evidence Card IDs" in message:
            return (
                "本章没有任何可用证据：evidence_ids 与 findings 必须为空，"
                "正文不得包含任何 [[e:...]] 标记，只输出证据缺口说明。"
            )
        if "language is not Simplified Chinese" in message:
            return "正文必须使用简体中文撰写，不得混用其他语言。"
        return ""

    async def assemble_report(
        self,
        topic: str,
        outline: Dict[str, Any],
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        citation_summary: Dict[str, Any],
        language: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate chapters sequentially and assemble a citation-bounded report."""
        # 引用校验过滤:只保留通过 CitationCheck 的证据卡(列表位置序号即 citation_id)
        citation_by_id = {
            item.get("citation_id"): item for item in citation_check_results
            if item.get("citation_id") is not None
        }
        verified = []
        for index, raw in enumerate(evidence_cards, start=1):
            if not citation_by_id.get(index, {}).get("is_valid", False):
                continue
            card = dict(raw)
            card["evidence_id"] = card.get("evidence_id") or f"{card.get('source_id', 'unknown')}:e{index}"
            verified.append(card)
        by_id = {card["evidence_id"]: card for card in verified}
        source_number = {
            source.get("source_id"): index for index, source in enumerate(sources, start=1)
            if source.get("source_id")
        }
        # 按 outline 逐节生成:取 plan 的 assigned_evidence_ids / assigned_source_ids 作为本节上下文
        generated = []
        for index, section in enumerate(outline.get("sections", [])):
            ids = set(section.get("assigned_evidence_ids") or [])
            chapter_cards = [by_id[item] for item in ids if item in by_id]
            source_ids = set(section.get("assigned_source_ids") or [])
            chapter_sources = [source for source in sources if source.get("source_id") in source_ids]
            # 调用章节生成器,获取章节内容与校验结果
            chapter, result = await self.generate_chapter(
                section, chapter_cards, chapter_sources, language, source_number=source_number,
            )
            generated.append({
                "index": index,
                "section": section,
                "chapter": chapter,
                "result": result,
            })

        return self.assemble_generated_chapters(
            topic, outline, sources, evidence_cards, citation_check_results,
            citation_summary, language, generated,
        )

    def assemble_generated_chapters(
        self,
        topic: str,
        outline: Dict[str, Any],
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        citation_summary: Dict[str, Any],
        language: str,
        generated: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Merge independently generated chapter results in outline order."""
        citation_by_id = {
            item.get("citation_id"): item for item in citation_check_results
            if item.get("citation_id") is not None
        }
        verified = []
        for index, raw in enumerate(evidence_cards, start=1):
            if not citation_by_id.get(index, {}).get("is_valid", False):
                continue
            card = dict(raw)
            card["evidence_id"] = card.get("evidence_id") or f"{card.get('source_id', 'unknown')}:e{index}"
            verified.append(card)

        zh = self._is_chinese(language)
        chapter_blocks: List[str] = []
        chapter_results: List[Dict[str, Any]] = []
        warnings: List[str] = []
        title_translations: Dict[str, str] = {}
        summary_parts: List[str] = []
        # 按 outline 顺序合并各章节,同时收集标题翻译、逐章模式/耗时与章节首句(供摘要)
        for item in sorted(generated, key=lambda value: int(value.get("index", 0))):
            section = item.get("section") or {}
            chapter = str(item.get("chapter") or "")
            result = item.get("result") or {}
            chapter_blocks.extend([chapter, ""])
            title_translations.update(result.get("source_title_translations") or {})
            chapter_mode = result.get("mode") or (
                "gap" if result.get("skipped") else
                "llm" if result.get("success") else "rule"
            )
            chapter_results.append({
                "heading": section.get("heading", ""), "success": bool(result.get("success")),
                "latency_ms": result.get("latency_ms", 0), "model": result.get("model", ""),
                "usage": result.get("usage", {}), "error": result.get("error", ""),
                "mode": chapter_mode,
            })
            if not result.get("success"):
                warnings.append(f"Chapter fallback ({section.get('heading', '')}): {result.get('error', 'unknown error')}")
            elif result.get("missing_title_translation_source_ids"):
                missing = ", ".join(result["missing_title_translation_source_ids"])
                warnings.append(f"Paper-title translation missing for source_id: {missing}")
            summary_sentence = self._first_reader_sentence(chapter)
            if summary_sentence:
                summary_parts.append(summary_sentence)

        # 摘要 lead:优先用已带引用的章节首句拼接,否则按章节标题兜底
        if summary_parts:
            lead = " ".join(summary_parts[:3])
        elif zh:
            headings = "、".join(
                f"“{section.get('heading', '')}”" for section in outline.get("sections", [])[:4]
            )
            lead = f"报告围绕{headings or '已绑定问题'}展开；具体结论及证据边界见各章节。"
        else:
            headings = ", ".join(
                str(section.get("heading", "")) for section in outline.get("sections", [])[:4]
            )
            lead = f"The report addresses {headings or 'the evidence-bound questions'}; findings and evidence boundaries appear in each chapter."

        headings = [str(section.get("heading") or "") for section in outline.get("sections", [])]
        # 组装报告骨架:标题 / 摘要 / 关键词;MMEA 主题走严格结构,否则补"范围+主题综合"标题
        lines = [
            f"# {academic_report_title(topic, language)}", "",
            f"## {'报告摘要' if zh else 'Abstract'}", "",
            lead, "",
            f"**{'关键词' if zh else 'Keywords'}**：{report_keywords(topic, headings, language)}", "",
        ]
        strict_structure = is_mmea_topic(topic)
        if not strict_structure:
            lead_refs = "".join(f"[{item}]" for item in list(dict.fromkeys(
                re.findall(r"\[(\d+)\]", lead)
            ))[:2])
            lines.extend([
                f"## {'范围与证据基础' if zh else 'Scope and Evidence Base'}", "",
                (
                    f"本报告围绕“{topic}”整理可追溯研究，并仅使用已绑定来源形成结论 {lead_refs}。"
                    if zh else
                    f"This report surveys traceable work on {topic} and limits conclusions to bound sources {lead_refs}."
                ), "",
                f"## {'主题综合' if zh else 'Thematic Synthesis'}", "",
            ])
        lines.extend(chapter_blocks)

        lines.extend([
            f"## {'结论' if zh else 'Conclusion'}", "",
            ((f"综合现有研究，{lead} 尚未由可比实验充分回答的部分应视为开放问题。" if zh else
              f"Across the available research, {lead} Questions not resolved by comparable experiments remain open.")), "",
        ])
        # outline 显式声明的证据缺口并入 warnings,对用户透明展示
        gaps = outline.get("evidence_gaps") or []
        if gaps:
            warnings.extend(gaps)
        lines.extend([f"## {'参考文献与证据追踪' if zh else 'References and Evidence Traceability'}", ""])
        # 参考文献按来源统一编号,并标注每个来源被提取出的证据类型
        evidence_types_by_source: Dict[str, set] = {}
        for card in verified:
            evidence_types_by_source.setdefault(str(card.get("source_id") or ""), set()).add(
                str(card.get("evidence_type") or "primary_claim")
            )
        lines.extend(
            format_reference(
                index, source, title_translations,
                evidence_types_by_source.get(str(source.get("source_id") or ""), set()),
            )
            for index, source in enumerate(sources, start=1)
        )
        # 汇总指标:任一 LLM 章节成功即整体成功;存在规则回退则标记 partial_fallback
        successful_llm = sum(1 for item in chapter_results if item["mode"] == "llm")
        aggregate = {
            "success": successful_llm > 0 or not verified,
            "partial_fallback": any(item["mode"] == "rule" for item in chapter_results),
            "latency_ms": sum(int(item.get("latency_ms") or 0) for item in chapter_results),
            "chapter_results": chapter_results,
            "usage": {},
            "error": "" if successful_llm else "All evidence-backed chapters used rule fallback",
        }
        return {"draft_report": "\n".join(lines), "warnings": warnings, "chapter_timings": chapter_results}, aggregate

    @staticmethod
    def _first_reader_sentence(chapter: str) -> str:
        """Return one already-cited reader-facing sentence for the executive summary."""
        for raw in str(chapter or "").splitlines():
            line = raw.strip().lstrip("- ")
            if not line or line.startswith("#") or line.startswith("|"):
                continue
            sentence = re.split(r"(?<=[。！？.!?])\s*", line, maxsplit=1)[0].strip()
            if sentence:
                return sentence
        return ""

    @staticmethod
    def _attach_statement_citations(
        synthesis: str,
        used_ids: List[str],
        evidence_by_id: Dict[str, Dict[str, Any]],
        source_number: Dict[str, int],
    ) -> str:
        """Convert exact evidence markers to at most two reader-facing source refs."""
        allowed = set(used_ids)

        def replace_marker(match: re.Match) -> str:
            parsed_ids = [
                item.strip() for item in re.split(r"[,;，；、\s]+", match.group(1))
                if item.strip()
            ]
            evidence_ids = [
                item for item in LLMDraftReviewer._normalize_marker_ids(
                    parsed_ids, evidence_by_id,
                )
                if item in allowed and item in evidence_by_id
            ]
            numbers = []
            for evidence_id in evidence_ids:
                source_id = evidence_by_id[evidence_id].get("source_id")
                number = source_number.get(source_id)
                if number and number not in numbers:
                    numbers.append(number)
                if len(numbers) >= 2:
                    break
            return "".join(f"[{number}]" for number in numbers)

        output = re.sub(r"\[\[e:([^\]]+)\]\]", replace_marker, str(synthesis or ""))
        # Internal evidence IDs are never part of the reader-facing citation format.
        output = re.sub(r"\(?\b(?:s2:[a-f0-9]+|W\d+|[^\s()]+):e\d+\)?", "", output)
        output = re.sub(r"(?:\[(\d+)\])(?:\s*\[\1\])+", r"[\1]", output)
        output = re.sub(r"[ \t]+([，。；：,.!?])", r"\1", output)
        return output

    @staticmethod
    def _extract_marker_ids(synthesis: str) -> List[str]:
        """Parse exact paragraph evidence markers with common list separators."""
        groups = re.findall(r"\[\[e:([^\]]+)\]\]", str(synthesis or ""))
        return list(dict.fromkeys(
            item.strip()
            for group in groups
            for item in re.split(r"[,;，；、\s]+", group)
            if item.strip()
        ))

    @staticmethod
    def _normalize_marker_ids(
        marker_ids: List[str], evidence_by_id: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Restore a dropped provider prefix only on a unique exact suffix match."""
        valid_ids = [str(item) for item in evidence_by_id]
        normalized = []
        for raw in marker_ids:
            marker = str(raw or "").strip()
            if marker in evidence_by_id:
                normalized.append(marker)
                continue
            matches = [
                candidate for candidate in valid_ids
                if candidate.endswith(f":{marker}")
            ]
            normalized.append(matches[0] if len(matches) == 1 else marker)
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _clip_text(value: Any, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        clipped = text[:limit]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return clipped.rstrip(" ,;:")

    @staticmethod
    def _is_chinese(language: str) -> bool:
        return str(language or "").lower().replace("_", "-").startswith("zh")

    @staticmethod
    def _text_has_sufficient_chinese(text: str) -> bool:
        value = str(text or "")
        cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
        latin_count = len(re.findall(r"[A-Za-z]", value))
        return cjk_count >= 24 and cjk_count >= latin_count * 0.10

    @staticmethod
    def _select_diverse_evidence(
        cards: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Remove repeated claims and round-robin sources into the prompt budget."""
        buckets: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        seen_claims = set()
        for card in cards:
            claim_key = re.sub(
                r"[^a-z0-9]+", " ", str(card.get("claim", "")).lower()
            ).strip()
            if not claim_key or claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
            buckets.setdefault(card.get("source_id", "unknown"), []).append(card)

        for bucket in buckets.values():
            bucket.sort(key=lambda card: float(card.get("confidence", 0.0) or 0.0), reverse=True)

        selected = []
        while len(selected) < limit and any(buckets.values()):
            for bucket in buckets.values():
                if bucket and len(selected) < limit:
                    selected.append(bucket.pop(0))
        return selected
