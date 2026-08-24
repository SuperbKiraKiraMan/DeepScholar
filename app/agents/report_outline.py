"""生成受证据约束的报告大纲。"""

import asyncio
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.agents.draft_reviewer import DraftReviewer
from app.llm.client import get_llm_client
from app.llm.prompts import (
    OUTLINE_REPAIR_SYSTEM,
    OUTLINE_REPAIR_USER,
    OUTLINE_SYSTEM,
    OUTLINE_USER,
)
from app.llm.schemas import LLMOutlinePlan

# 用户请求侧线索：只有用户问题明确命中这些词，对应研究维度才成为"必需内容"。
# 命中判断使用小写子串匹配；"评价方法/评测方法/评估方法"这类组合词先按 metrics
# 消解，避免误触发 methods 维度。
REQUEST_DIMENSION_CUES: Dict[str, Tuple[str, ...]] = {
    "methods": (
        "主要方法", "方法", "方法家族", "methods", "method", "taxonomy",
        "技术路线", "approach", "机制",
    ),
    "datasets": ("数据集", "基准", "语料", "dataset", "datasets", "benchmark", "corpus"),
    "metrics": (
        "评价", "评估", "评测", "指标", "评价标准", "评测指标",
        "evaluation", "metric", "metrics", "protocol",
    ),
    "results": (
        "实验", "结果", "性能", "比较", "效果", "experiment", "experiments",
        "results", "performance", "comparison",
    ),
    "limitations": (
        "局限", "研究空白", "不足", "挑战", "limitations", "limitation",
        "open problem", "open problems", "future work",
    ),
    "scope": ("研究范围", "范围界定", "术语", "边界", "定义", "scope", "terminology"),
}

# 章节覆盖侧线索：用于判断"heading + guiding_question"是否覆盖某个维度，
# 比请求侧更宽容，允许一个自然章节标题覆盖多个研究维度。
SECTION_DIMENSION_CUES: Dict[str, Tuple[str, ...]] = {
    "scope": (
        "引言", "introduction", "范围", "scope", "定义", "术语", "terminology",
        "综述", "背景", "overview", "background",
    ),
    "methods": (
        "方法", "method", "机制", "mechanism", "技术", "technique", "模型",
        "model", "taxonomy", "家族", "family", "approach", "算法", "algorithm",
    ),
    "datasets": ("数据集", "dataset", "基准", "benchmark", "语料", "corpus"),
    "metrics": (
        "评价", "评估", "评测", "指标", "评价标准", "evaluation", "metric",
        "protocol", "实验设置", "experimental setting", "measure",
    ),
    "results": (
        "实验", "结果", "experiment", "result", "性能", "performance",
        "比较", "comparison", "基线", "baseline", "效果",
    ),
    "limitations": (
        "局限", "limitation", "不足", "挑战", "challenge", "空白", "gap",
        "open problem", "future", "风险",
    ),
}

# 维度的稳定展示顺序
_DIMENSION_ORDER = ("scope", "methods", "datasets", "metrics", "results", "limitations")


def _dump(model: Any) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


class ReportOutlineGenerator:
    """生成大纲,并在代码层面强制所有证据/来源绑定。"""

    def __init__(self) -> None:
        self._rule_reviewer = DraftReviewer()

    async def generate(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        evidence_cards: List[Dict[str, Any]],
        citation_check_results: List[Dict[str, Any]],
        language: str = "zh",
        agent_mode: str = "rule",
        allow_rule_fallback: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 先做引用校验过滤 + 去重,得到"可信证据卡",并补齐缺省 evidence_id
        verified = self._rule_reviewer._deduplicate_cards(
            self._rule_reviewer._verified_cards(evidence_cards, citation_check_results)
        )
        for index, card in enumerate(verified, start=1):
            card["evidence_id"] = card.get("evidence_id") or f"{card.get('source_id', 'unknown')}:e{index}"

        # 规则版大纲预先算好作兜底:规则模式直接返回;LLM 模式若无证据则回退或抛错
        fallback = self._rule_outline(topic, sources, verified, language)
        if agent_mode != "llm":
            return fallback, {"success": True, "mode": "rule", "skipped": not verified}
        if not verified:
            if not allow_rule_fallback:
                raise RuntimeError("LLM-only outline cannot run without verified evidence")
            return fallback, {"success": False, "mode": "rule", "skipped": True}

        # 精选代表性证据卡喂给 LLM;用紧凑别名 E1/E2…、P1/P2… 缩短 prompt,避免暴露完整 ID
        prompt_cards = self._select_outline_evidence(
            verified, max(6, int(os.getenv("LLM_OUTLINE_EVIDENCE_LIMIT", "10"))),
        )
        evidence_aliases = {
            f"E{index}": str(card["evidence_id"])
            for index, card in enumerate(prompt_cards, start=1)
        }
        source_aliases: Dict[str, str] = {}
        evidence_summary_parts = []
        for index, card in enumerate(prompt_cards, start=1):
            source_id = str(card.get("source_id") or "")
            source_aliases.setdefault(source_id, f"P{len(source_aliases) + 1}")
            evidence_summary_parts.append(
            f"[E{index}] source={source_aliases[source_id]}\n"
            f"claim={card.get('claim', '')[:180]}\n"
            f"method={card.get('method', '')[:100]}\n"
            f"key_results={card.get('key_results', '')[:120]}\n"
            f"dataset={card.get('dataset', '')[:100]}\n"
            f"metric={card.get('metric', '')[:80]}\n"
            f"baseline={card.get('baseline', '')[:80]}\n"
            f"experimental_setting={card.get('experimental_setting', '')[:100]}\n"
            f"limitations={card.get('limitation', '')[:120]}\n"
            f"evidence_type={card.get('evidence_type', 'primary_claim')}\n"
            f"relevance={card.get('relevance_to_topic', '')[:100]}"
            )
        evidence_summary = "\n\n".join(evidence_summary_parts)
        # 研究维度由用户请求驱动：只有用户明确要求的维度才成为必需内容
        required_dimensions = self._required_dimensions(topic)
        required_sections = self._required_sections(topic, language)
        # 动态大纲要产出标题/引导问题/证据分配;生产中发现原先 60s 墙钟预算常误杀有效请求,故超时与重试次数独立配置
        wall_timeout = max(10, int(os.getenv("LLM_OUTLINE_TIMEOUT_SECONDS", "90")))
        attempts = max(1, int(os.getenv("LLM_OUTLINE_ATTEMPTS", "2")))
        # 把"必需章节"清单拼进 prompt,确保 LLM 不遗漏研究框架关键部分
        required_text = "\n".join(
            f"- {heading}: {question}" for heading, question in required_sections
        )
        system_prompt = OUTLINE_SYSTEM
        user_prompt = OUTLINE_USER.format(
            topic=topic,
            output_language="Simplified Chinese" if self._is_zh(language) else "English",
            source_count=len(sources),
            evidence_count=len(prompt_cards),
            evidence_summary=evidence_summary,
            required_sections=(
                required_text
                or "- Derive coverage directly from the user's explicit sub-questions."
            ),
        )
        output_schema = LLMOutlinePlan
        retry_instruction = "\nRetry: return only the exact compact outline JSON object."
        result: Dict[str, Any] = {}
        # 带墙钟超时的重试循环:超时/失败按 attempts 重试,全失败则回退规则版(LLM-only 抛错)
        for attempt in range(attempts):
            try:
                result = await asyncio.wait_for(
                    get_llm_client().generate_structured(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt + (
                            retry_instruction if attempt else ""
                        ),
                        output_schema=output_schema,
                        timeout_seconds=wall_timeout,
                        max_retries=0,
                    ),
                    timeout=wall_timeout + 2,
                )
            except asyncio.TimeoutError:
                result = {
                    "success": False,
                    "error": f"Outline timeout after {wall_timeout}s (attempt {attempt + 1}/{attempts})",
                    "latency_ms": wall_timeout * 1000,
                }
            if result.get("success"):
                break
        if not result.get("success"):
            if not allow_rule_fallback:
                raise RuntimeError(
                    "LLM-only outline failed after retries: "
                    + str(result.get("error") or "unknown error")
                )
            return fallback, result

        # 规范化模型输出(别名→真实 ID、补来源绑定),再做硬校验
        outline = self._normalize_model_outline(
            _dump(result["data"]), prompt_cards, evidence_aliases, source_aliases,
        )
        error = self._validate_outline(outline, verified, required_dimensions)
        # 维度缺失类错误先做一次有界 LLM Repair:保留已有合法章节与证据绑定,
        # 只要求补齐用户明确要求的研究维度;结构性错误(证据绑定/数量)不修复
        if error and required_dimensions and error.startswith(
            "required research dimensions missing"
        ):
            missing = [
                dimension for dimension in self._missing_dimensions(
                    outline, required_dimensions,
                )
            ]
            repaired, repair_info = await self._repair_outline(
                topic=topic,
                outline=outline,
                prompt_cards=prompt_cards,
                evidence_aliases=evidence_aliases,
                source_aliases=source_aliases,
                missing_dimensions=missing,
                evidence_summary=evidence_summary,
                wall_timeout=wall_timeout,
            )
            result["repair"] = repair_info
            if repaired is not None:
                repair_error = self._repair_preserves_existing(outline, repaired)
                if not repair_error:
                    repair_error = self._validate_outline(
                        repaired, verified, required_dimensions,
                    )
                if not repair_error:
                    outline = repaired
                    error = ""
                else:
                    # Repair 调用成功但复检仍未通过:以最终复检结果为准
                    error = repair_error
                    repair_info.update(success=False, error=repair_error)
        if error:
            result.update(success=False, error=f"Outline validation failed: {error}")
            if not allow_rule_fallback:
                raise RuntimeError("LLM-only outline validation failed: " + error)
            return fallback, result
        outline["topic"] = topic
        outline["evidence_card_count"] = len(verified)
        outline["source_count"] = len({c.get("source_id") for c in verified if c.get("source_id")})
        return outline, result

    def _rule_outline(
        self, topic: str, sources: List[Dict[str, Any]], cards: List[Dict[str, Any]], language: str,
    ) -> Dict[str, Any]:
        zh = self._is_zh(language)
        required = self._required_sections(topic, language)
        required_dimensions = self._required_dimensions(topic)
        # 关键步骤：明确的“仅局限”窄问题保持单章；其它单维请求沿用证据主题宽度契约。
        if len(required) >= 2 or required_dimensions == ["limitations"]:
            # 有必需章节框架:按 facet 给每张卡打分,分配到最相关章节(最多 2 个)
            section_defs = [
                (heading, question, self._facet_score_key(heading))
                for heading, question in required
            ]
            assignments = {heading: [] for heading, _, _ in section_defs}
            for card in cards:
                scored = sorted(
                    (
                        (self._card_facet_score(card, facet), position, heading)
                        for position, (heading, _, facet) in enumerate(section_defs)
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
                chosen = [item for item in scored if item[0] > 0][:2]
                if not chosen:
                    chosen = [(1, 0, section_defs[0][0])]
                for _, _, heading in chosen:
                    assignments[heading].append(card)

            # 每个章节只携带分配给它的证据卡;空章节显式记录为证据缺口
            sections = []
            for heading, question, _ in section_defs:
                section_cards = assignments[heading]
                sections.append({
                    "heading": heading,
                    "guiding_question": question,
                    "assigned_evidence_ids": list(dict.fromkeys(
                        card["evidence_id"] for card in section_cards
                    )),
                    "assigned_source_ids": list(dict.fromkeys(
                        card.get("source_id", "") for card in section_cards if card.get("source_id")
                    )),
                    "estimated_length": "long" if heading in {
                        "方法", "数据集", "局限",
                        "Main Method Taxonomy", "Common Datasets", "Research Limitations and Open Problems",
                    } else "medium",
                })
            gaps = [
                (f"章节“{section['heading']}”缺少可核验证据。" if zh
                 else f"Chapter '{section['heading']}' lacks verified evidence.")
                for section in sections if not section["assigned_evidence_ids"]
            ]
            return {
                "topic": topic,
                "evidence_card_count": len(cards),
                "source_count": len({c.get("source_id") for c in cards if c.get("source_id")}),
                "sections": sections,
                "cross_cutting_themes": [],
                "evidence_gaps": gaps,
            }

        # 无必需框架时:按证据卡主题分组生成章节
        grouped = self._rule_reviewer._group_by_theme(cards)
        translations = {
            "Agent Process and Capability Evaluation": "智能体过程与能力评估",
            "Robustness, Reliability, and Safety": "鲁棒性、可靠性与安全",
            "Human and Model-based Assessment": "人工与模型评审",
            "Efficiency and Operational Constraints": "效率与运行约束",
            "Benchmarks, Metrics, and Evaluation Scope": "基准、指标与评估范围",
            "Reported Methods and Empirical Results": "方法与实证结果",
            "Other Supported Observations": "其他有证据支持的观察",
        }
        sections = []
        for heading, section_cards in list(grouped.items())[:6]:
            localized = translations.get(heading, heading) if zh else heading
            sections.append({
                "heading": localized,
                "guiding_question": (
                    f"现有证据如何回答“{localized}”这一问题？" if zh
                    else f"What does the available evidence establish about {heading.lower()}?"
                ),
                "assigned_evidence_ids": [c["evidence_id"] for c in section_cards],
                "assigned_source_ids": list(dict.fromkeys(
                    c.get("source_id", "") for c in section_cards if c.get("source_id")
                )),
                "estimated_length": "medium",
            })

        # 即使证据稀疏也要显式保留空章节:空章节暴露证据缺口,章节写作者不得用先验知识填充
        defaults = (
            ["研究范围与定义", "方法与证据比较", "局限与研究空白"] if zh
            else ["Scope and Definitions", "Methods and Evidence Comparison", "Limitations and Gaps"]
        )
        for heading in defaults:
            if len(sections) >= 3:
                break
            sections.append({
                "heading": heading,
                "guiding_question": (
                    f"现有证据能否可靠回答“{heading}”？" if zh
                    else f"Can the available evidence reliably answer the question of {heading.lower()}?"
                ),
                "assigned_evidence_ids": [],
                "assigned_source_ids": [],
                "estimated_length": "short",
            })
        gaps = [
            (f"章节“{section['heading']}”缺少可核验证据。" if zh
             else f"Chapter '{section['heading']}' lacks verified evidence.")
            for section in sections if not section["assigned_evidence_ids"]
        ]
        return {
            "topic": topic,
            "evidence_card_count": len(cards),
            "source_count": len({c.get("source_id") for c in cards if c.get("source_id")}),
            "sections": sections,
            "cross_cutting_themes": [],
            "evidence_gaps": gaps,
        }

    @classmethod
    def _required_dimensions(cls, topic: str) -> List[str]:
        """从用户请求中提取明确要求的研究维度。

        只有用户问题命中某维度的请求侧线索时,该维度才是必需内容;
        "评价方法/评测方法/评估方法"等组合词先按 metrics 消解,
        避免误把 metrics 请求识别成 methods 请求。
        """
        text = str(topic or "").lower()
        full_report_cues = (
            "完整报告", "完整调研", "全面调研", "系统性调研",
            "complete report", "comprehensive report", "systematic review",
        )
        if any(cue in text for cue in full_report_cues):
            # 关键步骤：完整框架只由用户显式范围标志触发，不再按研究领域关键词暗中扩张。
            return list(_DIMENSION_ORDER)
        # 组合词消解:评价/评估/评测/衡量 + 方法 属于 metrics 语境
        methods_text = re.sub(r"(评价|评估|评测|衡量)\s*方法", "", text)
        required = []
        for dimension in _DIMENSION_ORDER:
            cues = REQUEST_DIMENSION_CUES[dimension]
            source_text = methods_text if dimension == "methods" else text
            if any(cue in source_text for cue in cues):
                required.append(dimension)
        return required

    @staticmethod
    def _missing_dimensions(
        outline: Dict[str, Any], required_dimensions: List[str],
    ) -> List[str]:
        """返回大纲(heading + guiding_question)未覆盖的必需维度。

        一个章节允许覆盖多个维度(如"数据集与评测指标"),
        每个维度只需被任意一个章节命中即可。
        """
        sections = outline.get("sections") or []
        combined = " ".join(
            f"{section.get('heading', '')} {section.get('guiding_question', '')}"
            for section in sections
        ).lower()
        return [
            dimension for dimension in required_dimensions
            if not any(
                cue in combined for cue in SECTION_DIMENSION_CUES[dimension]
            )
        ]

    async def _repair_outline(
        self,
        *,
        topic: str,
        outline: Dict[str, Any],
        prompt_cards: List[Dict[str, Any]],
        evidence_aliases: Dict[str, str],
        source_aliases: Dict[str, str],
        missing_dimensions: List[str],
        evidence_summary: str,
        wall_timeout: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """对缺失必需维度的大纲执行一次有界 LLM Repair。

        Repair Prompt 要求保留全部已有合法章节(标题/引导问题/证据绑定),
        只补齐缺失维度;返回结果仍要走与首次生成完全相同的归一化与硬校验。
        任何失败都返回 (None, info),由调用方按 fallback/LLM-only 规则处置。
        """
        # 只把章节决策(标题/引导问题/证据别名)送回模型,绑定由运行时重推导
        compact_sections = [
            {
                "heading": section.get("heading", ""),
                "guiding_question": section.get("guiding_question", ""),
                "assigned_evidence_ids": section.get("assigned_evidence_ids", []),
            }
            for section in (outline.get("sections") or [])[:8]
        ]
        user_prompt = OUTLINE_REPAIR_USER.format(
            topic=topic,
            missing_dimensions=", ".join(missing_dimensions),
            outline_json=json.dumps(
                {
                    "sections": compact_sections,
                    "evidence_gaps": list(outline.get("evidence_gaps") or [])[:10],
                },
                ensure_ascii=False,
            ),
            evidence_summary=evidence_summary,
        )
        try:
            repair_result = await asyncio.wait_for(
                get_llm_client().generate_structured(
                    system_prompt=OUTLINE_REPAIR_SYSTEM,
                    user_prompt=user_prompt,
                    output_schema=LLMOutlinePlan,
                    timeout_seconds=wall_timeout,
                    max_retries=0,
                ),
                timeout=wall_timeout + 2,
            )
        except asyncio.TimeoutError:
            return None, {
                "attempted": True, "success": False,
                "missing_dimensions": missing_dimensions,
                "error": f"repair timeout after {wall_timeout}s",
            }
        if not repair_result.get("success"):
            return None, {
                "attempted": True, "success": False,
                "missing_dimensions": missing_dimensions,
                "error": str(repair_result.get("error") or "unknown repair error")[:200],
            }
        repaired = self._normalize_model_outline(
            _dump(repair_result["data"]), prompt_cards, evidence_aliases, source_aliases,
        )
        return repaired, {
            "attempted": True, "success": True,
            "missing_dimensions": missing_dimensions,
            "latency_ms": repair_result.get("latency_ms", 0),
            "model": repair_result.get("model", ""),
            "usage": repair_result.get("usage", {}),
        }

    @staticmethod
    def _repair_preserves_existing(original: Dict[str, Any], repaired: Dict[str, Any]) -> str:
        """Repair 只能补缺，不能删改已经合法的章节和证据绑定。"""
        keys = (
            "heading", "guiding_question", "assigned_evidence_ids", "assigned_source_ids",
        )

        def signature(section: Dict[str, Any]) -> tuple:
            return tuple(
                tuple(section.get(key) or []) if key.endswith("_ids")
                else str(section.get(key) or "")
                for key in keys
            )

        original_signatures = [signature(section) for section in original.get("sections") or []]
        repaired_signatures = [signature(section) for section in repaired.get("sections") or []]
        cursor = 0
        for expected in original_signatures:
            try:
                cursor = repaired_signatures.index(expected, cursor) + 1
            except ValueError:
                return "outline repair rewrote or removed a validated section"
        return ""

    @staticmethod
    def _validate_outline(
        outline: Dict[str, Any], cards: List[Dict[str, Any]],
        required_dimensions: Optional[List[str]] = None,
    ) -> str:
        sections = outline.get("sections") or []
        minimum = 1 if list(required_dimensions or []) == ["limitations"] else 2
        # 窄问题允许单章，其余报告仍保持 2~8 章的结构边界。
        if not minimum <= len(sections) <= 8:
            return f"section count must be between {minimum} and 8"
        # 必需维度覆盖检查:只检查用户明确要求的研究维度,
        # 同时检查 heading 与 guiding_question,不要求标题与预设模板一致
        if required_dimensions:
            missing = ReportOutlineGenerator._missing_dimensions(outline, required_dimensions)
            if missing:
                return f"required research dimensions missing: {missing}"
        # 逐章节校验证据绑定:证据必须存在、来源必须在 assigned_source_ids 内,且同一证据最多复用 2 次
        by_id = {card.get("evidence_id"): card for card in cards}
        assignments: Counter = Counter()
        for section in sections:
            evidence_ids = list(dict.fromkeys(section.get("assigned_evidence_ids") or []))
            source_ids = set(section.get("assigned_source_ids") or [])
            for evidence_id in evidence_ids:
                card = by_id.get(evidence_id)
                if not card:
                    return f"unknown evidence_id {evidence_id}"
                if card.get("source_id") not in source_ids:
                    return f"source binding missing for {evidence_id}"
                assignments[evidence_id] += 1
        overused = [key for key, count in assignments.items() if count > 2]
        return f"evidence assigned to more than two sections: {overused}" if overused else ""

    @staticmethod
    def _normalize_model_outline(
        outline: Dict[str, Any],
        cards: List[Dict[str, Any]],
        evidence_aliases: Dict[str, str],
        source_aliases: Dict[str, str],
    ) -> Dict[str, Any]:
        """修复无害的标识符漂移,而不凭空编造大纲内容。

        prompt 有意暴露紧凑的 E/P 别名;模型可能返回这些别名(或省略来源绑定),
        而运行时校验期望规范的原始 ID。只解析无歧义的别名,并从所选证据卡推导
        来源绑定;标题与引导问题仍由模型决定。
        """
        result = dict(outline or {})
        by_evidence_id = {
            str(card.get("evidence_id") or ""): card
            for card in cards if card.get("evidence_id")
        }
        evidence_lookup = {
            str(alias).upper(): str(evidence_id)
            for alias, evidence_id in (evidence_aliases or {}).items()
        }
        source_lookup = {
            str(alias).upper(): str(source_id)
            for source_id, alias in (source_aliases or {}).items()
        }
        usage: Counter = Counter()
        normalized_sections = []
        for raw_section in list(result.get("sections") or [])[:8]:
            section = dict(raw_section or {})
            evidence_ids = []
            for raw_id in section.get("assigned_evidence_ids") or []:
                candidate = str(raw_id or "").strip().strip("[]（）() ")
                # 规范 ID 形如 s2:abc:e1(以 e1 结尾);只允许整值作为 E 别名,子串匹配会把章节错误绑定到别的论文
                alias_match = re.fullmatch(r"E\d+", candidate, re.IGNORECASE)
                if alias_match:
                    candidate = evidence_lookup.get(
                        alias_match.group(0).upper(), candidate,
                    )
                if candidate in by_evidence_id and usage[candidate] < 2:
                    evidence_ids.append(candidate)
                    usage[candidate] += 1
            evidence_ids = list(dict.fromkeys(evidence_ids))

            # 来源绑定一律从被选中的证据卡推导,不信任模型自填的来源
            source_ids = [
                str(by_evidence_id[evidence_id].get("source_id") or "")
                for evidence_id in evidence_ids
                if by_evidence_id[evidence_id].get("source_id")
            ]
            # 保留证据缺口章节的显式"仅来源"分配,同时解析紧凑的 P 别名
            for raw_id in section.get("assigned_source_ids") or []:
                candidate = str(raw_id or "").strip().strip("[]（）() ")
                candidate = source_lookup.get(candidate.upper(), candidate)
                if candidate and candidate in source_aliases:
                    source_ids.append(candidate)
            section["assigned_evidence_ids"] = evidence_ids
            section["assigned_source_ids"] = list(dict.fromkeys(source_ids))
            section.setdefault("estimated_length", "medium")
            normalized_sections.append(section)
        result["sections"] = normalized_sections
        return result

    # 每个研究维度的规范章节框架(heading, guiding_question),
    # 仅当用户明确要求该维度时才会进入必需章节列表
    _DIMENSION_SECTIONS_ZH = {
        "scope": ("引言", "研究对象如何定义，其边界以及与相邻任务的区别是什么？"),
        "methods": ("方法", "主要技术机制可分为哪些方法家族，各自优势、局限和适用条件是什么？"),
        "datasets": ("数据集", "常用数据集的规模、构成、划分与已知偏差分别是什么？"),
        "metrics": ("评价指标", "常用指标与实验协议如何影响不同研究之间的可比性？"),
        "results": ("实验", "哪些带数据集、指标、基线和数值的结果可以可靠比较？"),
        "limitations": ("局限", "该领域在数据、方法、泛化和扩展性方面有哪些局限？"),
    }
    _DIMENSION_SECTIONS_EN = {
        "scope": ("Terminology and Scope", "How is the research object defined and bounded?"),
        "methods": ("Main Method Taxonomy", "Which mechanism-based method families dominate, with what trade-offs?"),
        "datasets": ("Common Datasets", "Which datasets are used, with what scale, splits, and biases?"),
        "metrics": ("Metrics and Experimental Protocols", "How do metrics and protocols affect comparability?"),
        "results": ("Representative Experimental Results", "Which results include comparable datasets, metrics, and baselines?"),
        "limitations": ("Research Limitations and Open Problems", "What data, method, and generalization limitations remain?"),
    }

    @classmethod
    def _required_sections(cls, topic: str, language: str) -> List[Tuple[str, str]]:
        """按用户明确要求的研究维度生成规范章节框架。

        用户没有要求的维度不再强加章节;没有任何必需维度时返回空列表,
        由规则大纲按证据主题分组、LLM 大纲按子问题自行组织。
        """
        required = cls._required_dimensions(topic)
        if not required:
            return []
        table = (
            cls._DIMENSION_SECTIONS_ZH if cls._is_zh(language)
            else cls._DIMENSION_SECTIONS_EN
        )
        return [table[dimension] for dimension in required]

    @staticmethod
    def _facet_score_key(heading: str) -> str:
        # 将章节标题归入"方法/数据集/指标/结果/局限"五类 facet,未知归 scope
        lowered = heading.lower()
        for facet, cues in {
            "methods": ("方法", "method"), "datasets": ("数据集", "dataset"),
            "metrics": ("评估", "metric", "protocol"), "results": ("结果", "result"),
            "limitations": ("局限", "limitation", "open problem"),
        }.items():
            if any(cue in lowered for cue in cues):
                return facet
        return "scope"

    @staticmethod
    def _card_facet_score(card: Dict[str, Any], facet: str) -> int:
        # 证据卡与 facet 的相关度:非空相关字段每个 +2 分,命中关键词每个 +1 分
        fields = {
            "methods": ("method", "claim"),
            "datasets": ("dataset", "experimental_setting", "claim"),
            "metrics": ("metric", "experimental_setting", "claim"),
            "results": ("result", "key_results", "baseline", "claim"),
            "limitations": ("limitation", "claim"),
            "scope": ("claim", "relevance_to_topic"),
        }[facet]
        cues = {
            "methods": ("method", "model", "framework", "fusion", "alignment"),
            "datasets": ("dataset", "benchmark", "corpus", "graph", "language pair"),
            "metrics": ("hits@", "mrr", "metric", "split", "seed", "candidate"),
            "results": ("outperform", "improv", "result", "%", "baseline"),
            "limitations": ("limit", "challenge", "missing", "noise", "bias", "cost", "scalab"),
            "scope": (),
        }[facet]
        values = [str(card.get(field) or "") for field in fields]
        score = sum(2 for value in values if value.strip())
        text = " ".join(values).lower()
        return score + sum(1 for cue in cues if cue in text)

    @staticmethod
    def _is_zh(language: str) -> bool:
        return str(language or "").lower().replace("_", "-").startswith("zh")

    @staticmethod
    def _select_outline_evidence(cards: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """让 outline prompt 保持精简,同时覆盖每个报告维度(facet)。"""
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        facets = (
            ("method", "method_family"),
            ("dataset", "dataset_name"),
            ("metric", "experimental_setting"),
            ("result", "key_results", "baseline"),
            ("limitation",),
        )
        # 第一优先:每个 facet 选一张最具代表性的卡,保证大纲覆盖全部维度
        for fields in facets:
            candidates = sorted(
                cards,
                key=lambda card: (
                    -sum(bool(str(card.get(field) or "").strip()) for field in fields),
                    -float(card.get("confidence") or 0),
                ),
            )
            card = next(
                (item for item in candidates if any(item.get(field) for field in fields)),
                None,
            )
            if card and card.get("evidence_id") not in selected_ids:
                selected.append(card)
                selected_ids.add(card.get("evidence_id"))
        # 第二优先:按来源轮询补足,避免单一来源霸屏,直到达到 limit
        seen_sources = set()
        for card in cards:
            source_id = str(card.get("source_id") or "")
            evidence_id = card.get("evidence_id")
            if source_id in seen_sources or evidence_id in selected_ids:
                continue
            selected.append(card)
            selected_ids.add(evidence_id)
            seen_sources.add(source_id)
            if len(selected) >= limit:
                return selected[:limit]
        # 最后:仍有空位时按序补足
        for card in cards:
            if card.get("evidence_id") not in selected_ids:
                selected.append(card)
            if len(selected) >= limit:
                break
        return selected[:limit]

    @classmethod
    def _outline_from_assignments(
        cls,
        decision: Dict[str, Any],
        required: List[Tuple[str, str]],
        cards: List[Dict[str, Any]],
        language: str,
        evidence_aliases: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """把模型紧凑的证据分配展开为安全的大纲。"""
        by_id = {str(card.get("evidence_id") or ""): card for card in cards}
        aliases = {str(key): str(value) for key, value in (evidence_aliases or {}).items()}
        raw_assignments = decision.get("assignments") or {}
        usage: Counter = Counter()
        sections = []
        gaps = list(decision.get("evidence_gaps") or [])
        zh = ReportOutlineGenerator._is_zh(language)
        for heading, question in required:
            evidence_ids = []
            assigned_values = raw_assignments.get(heading)
            if assigned_values is None:
                assigned_values = next((
                    values for raw_heading, values in raw_assignments.items()
                    if str(raw_heading or "").strip().startswith(heading)
                    or heading in str(raw_heading or "")
                ), [])
            for raw_id in assigned_values or []:
                raw_text = str(raw_id or "").strip().strip("[]（）() ")
                alias_match = re.fullmatch(r"E\d+", raw_text, re.IGNORECASE)
                if alias_match:
                    raw_text = alias_match.group(0).upper()
                evidence_id = aliases.get(raw_text, raw_text)
                if evidence_id not in by_id:
                    # 仅当模型给出的是来源 ID 时容忍:映射到该来源第一个仍可用的证据卡
                    evidence_id = next((
                        item_id for item_id, card in by_id.items()
                        if str(card.get("source_id") or "") == raw_text
                    ), evidence_id)
                if evidence_id not in by_id or usage[evidence_id] >= 2:
                    continue
                evidence_ids.append(evidence_id)
                usage[evidence_id] += 1

            # 模型决定语义分配,但有界后置条件防止同一章节误收同一篇论文的两张卡;
            # 它不编造证据或文字,仅在有独立来源时从已展示的 prompt 卡中补足分配
            facet = cls._facet_score_key(heading)
            if facet in {"methods", "datasets", "limitations"}:
                distinct_ids = []
                seen_section_sources = set()
                for evidence_id in evidence_ids:
                    source_id = str(by_id[evidence_id].get("source_id") or "")
                    if not source_id or source_id in seen_section_sources:
                        usage[evidence_id] -= 1
                        continue
                    distinct_ids.append(evidence_id)
                    seen_section_sources.add(source_id)
                evidence_ids = distinct_ids
                candidates = sorted(
                    by_id.items(),
                    key=lambda item: (
                        -cls._card_facet_score(item[1], facet),
                        -float(item[1].get("confidence") or 0),
                    ),
                )
                for candidate_id, card in candidates:
                    source_id = str(card.get("source_id") or "")
                    if (
                        len(seen_section_sources) >= 2
                        or not source_id
                        or source_id in seen_section_sources
                        or usage[candidate_id] >= 2
                        or cls._card_facet_score(card, facet) <= 0
                    ):
                        continue
                    evidence_ids.append(candidate_id)
                    seen_section_sources.add(source_id)
                    usage[candidate_id] += 1
            source_ids = list(dict.fromkeys(
                str(by_id[evidence_id].get("source_id") or "")
                for evidence_id in evidence_ids
                if by_id[evidence_id].get("source_id")
            ))
            if not evidence_ids:
                gaps.append(
                    f"章节“{heading}”缺少可核验证据。" if zh
                    else f"Chapter '{heading}' lacks verified evidence."
                )
            sections.append({
                "heading": heading,
                "guiding_question": question,
                "assigned_evidence_ids": evidence_ids,
                "assigned_source_ids": source_ids,
                "estimated_length": "long" if heading in {
                    "方法", "数据集", "局限", "Main Method Taxonomy",
                    "Common Datasets", "Research Limitations and Open Problems",
                } else "medium",
            })
        return {
            "sections": sections,
            "cross_cutting_themes": [],
            "evidence_gaps": list(dict.fromkeys(str(item) for item in gaps if item)),
        }
