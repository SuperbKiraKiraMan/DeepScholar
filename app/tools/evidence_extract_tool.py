"""
app/tools/evidence_extract_tool.py

EvidenceExtractTool —— 证据抽取工具。

从来源的 full_text 中抽取结构化证据卡（EvidenceCard）。

当前实现策略：
- 优先用规则（从 full_text 中按 section 标题解析）
- 抽取 claim、quote、method、limitation
- confidence 来自 SourceQualityScorer 的评分
- 每条 EvidenceCard 绑定 source_id

在 Agent 调用链中的位置：
Analysis Worker -> EvidenceExtractTool -> CitationCheckTool

与 RAG 的本质区别：
- RAG 检索返回的是 chunk（被动切分的文本片段）
- EvidenceExtractTool 返回的是 EvidenceCard（主动抽取的结构化证据）
- EvidenceCard 有 claim/quote 分离、source_id 绑定、confidence 评分
- CitationCheckTool 可以对 EvidenceCard 做规则校验
"""

import re
from typing import Any, Dict, List, Optional

from app.tools.base import BaseTool, ToolResult


_TRACEABLE_TEXT_FIELDS = ("full_text", "abstract", "text", "quote", "snippet")


def _stable_source_identity(source: Any) -> Optional[str]:
    """Return a stable source identity or locator when one is present."""
    if not isinstance(source, dict):
        return None
    for key in (
        "paper_id", "source_id", "doi", "url", "openalex_id",
        "semantic_scholar_id", "source_path",
    ):
        value = str(source.get(key) or "").strip()
        if value:
            return value
    return None


def available_source_identity(source: Any) -> Optional[str]:
    """Return a stable identity only for a titled, downstream-usable paper."""
    if not isinstance(source, dict) or not str(source.get("title") or "").strip():
        return None
    return _stable_source_identity(source)


def traceable_source_text(source: Any) -> str:
    """
    Return evidence-bearing text tied to an identifiable paper source.

    Title-only OpenAlex fallbacks are deliberately rejected even when copied into
    ``snippet``. Metadata fields, identifiers, URLs, counts, and graph edges never
    become evidence text.
    """
    if _stable_source_identity(source) is None:
        return ""
    title = str(source.get("title") or "").strip()
    for field in _TRACEABLE_TEXT_FIELDS:
        value = source.get(field)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text == title:
            continue
        return text
    return ""


def is_evidence_eligible_source(source: Any) -> bool:
    """Whether the existing EvidenceCard pipeline may consume this source."""
    return bool(traceable_source_text(source))


class EvidenceExtractTool(BaseTool):
    """
    证据抽取工具。

    从 full_text 中提取结构化证据，返回 EvidenceCard 列表。
    默认使用规则解析；可选接入 LLM 做更精准的抽取。

    抽取策略：
    1. 按 section 标题分割 full_text
    2. 在 Abstract / Key Findings 中提取 claim
    3. 在 Method 中提取 method
    4. 在 Limitations 中提取 limitation
    5. 每个抽取结果截取对应的原文片段作为 quote

    """

    # 段落分隔模式
    # 匹配 section 标题行。两种格式都支持：
    #   1. "Method:\ncontent..."（标题后换行）
    #   2. "Method: content..."（标题后直接跟内容，同一行）
    SECTION_PATTERN = re.compile(
        r'(?:^|\n)((?:Abstract|Methods?|Key Findings?|Results?|'
        r'Limitations?|Conclusion|Introduction|Related Work|Datasets?|Data|'
        r'Experimental Setup|Experiments?|Evaluation|Discussion)(?:\s*:)?)',
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "evidence_extract"

    @property
    def description(self) -> str:
        return (
            "Extract structured evidence cards from source full_text. "
            "Each evidence card contains a claim, supporting quote, source_id, "
            "and confidence score. Uses rule-based section parsing by default; "
            "LLM-based extraction available in later phases."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "object",
                    "description": "Paper source with full_text, source_id, url, and quality_score",
                },
            },
            "required": ["source"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        """从来源文本中抽取证据卡。"""
        source = kwargs.get("source", {})
        topic = str(kwargs.get("topic") or source.get("_research_topic") or "").strip()
        if not isinstance(source, dict) or not source:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="No source provided for evidence extraction",
            )

        source_id = source.get("source_id", "")
        url = source.get("url", "")
        full_text = traceable_source_text(source)
        has_full_text = bool(str(source.get("full_text") or "").strip())
        quality_score = source.get("quality_score", 0.5)

        if not full_text:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=(
                    f"No full_text or other traceable text in source "
                    f"{source_id or '?'}; metadata-only records cannot produce "
                    "EvidenceCards"
                ),
            )

        # 解析 full_text 中的 section
        sections = self._parse_sections(full_text)

        evidence_cards = []

        def make_card(claim: Dict[str, str], confidence: float, location: str) -> Dict[str, Any]:
            quote = claim["quote"]
            evidence_type = self._evidence_type(source, location, quote)
            return {
                "claim": claim["text"],
                "quote": quote,
                "source_id": source_id,
                "url": url,
                "confidence": round(confidence, 4),
                "method": "",
                "method_family": "",
                "dataset": "",
                "dataset_name": "",
                "graph_or_language_pair": "",
                "entity_count": "",
                "modalities": "",
                "missingness": "",
                "data_split": "",
                "seed_ratio": "",
                "metric": "",
                "result": "",
                "baseline": "",
                "experimental_setting": "",
                "limitation": "",
                "key_results": self._extract_key_results(quote),
                # Only claim full-text provenance when the source really exposed it.
                "original_quote": self._clip_text(quote, 300, add_ellipsis=False) if has_full_text else "",
                "quote_location": location if has_full_text else "abstract_only",
                "evidence_quote": self._clip_text(quote, 700, add_ellipsis=False),
                "page_or_section": location if has_full_text else "abstract_only",
                "evidence_type": evidence_type,
                "relevance_to_topic": self._explain_relevance(topic, claim["text"]),
                "research_task": source.get("research_task", ""),
                "task_relevance": source.get("task_relevance", ""),
                "dataset_task_consistent": True,
            }

        # 1. 从 Abstract 提取 claim
        abstract_text = sections.get("abstract", "")
        if abstract_text:
            # 继续抽取，并通过_extract_claims_from_abstract 函数过滤无效 claim
            claims = self._extract_claims_from_abstract(abstract_text)
            for claim in claims:
                evidence_cards.append(make_card(claim, quality_score, "Abstract"))

        # 如果没解析到 Abstract section，用整个文本的前 3 句作为 claim
        if not evidence_cards and full_text:
            sentences = self._split_sentences(full_text)
            main_claim = " ".join(sentences[:3])
            if main_claim:
                evidence_cards.append(make_card({
                    "text": self._clip_text(main_claim, 420, add_ellipsis=True),
                    "quote": self._clip_text(main_claim, 700, add_ellipsis=False),
                }, quality_score, "Full text (opening passage)"))

        # 2. 从 Key Findings 提取更多 claim
        findings_text = sections.get("key findings", "") or sections.get("results", "")
        if findings_text:
            findings_claims = self._extract_claims_from_abstract(findings_text)
            for claim in findings_claims:
                evidence_cards.append(make_card(
                    claim, quality_score * 0.95,
                    "Key Findings" if sections.get("key findings") else "Results",
                ))

        # 3. 提取 method
        method_text = sections.get("method", "") or sections.get("methods", "")
        if not method_text:
            # Public scholarly APIs frequently expose only an abstract. Recover
            # explicit method sentences from that abstract instead of treating
            # every abstract-only paper as having no method evidence.
            method_cues = re.compile(
                r"\b(?:we (?:propose|present|introduce|develop)|this (?:paper|work) "
                r"(?:proposes|presents|introduces)|method|framework|model|approach|"
                r"architecture|fusion|embedding|encoder|attention|pseudo[- ]seed|"
                r"contrastive|siamese|graph neural)\b",
                re.IGNORECASE,
            )
            method_sentences = [
                sentence for sentence in self._split_sentences(abstract_text or full_text)
                if method_cues.search(sentence)
            ]
            method_text = " ".join(method_sentences[:2])
        if method_text:
            method_sentences = self._split_sentences(method_text)
            method = self._clip_text(
                " ".join(method_sentences[:3]), 500, add_ellipsis=True
            )
            # 将 method 附加到所有已有 evidence card
            for card in evidence_cards:
                card["method"] = method
                card["method_family"] = self._classify_method_families(method)

        # 4. 提取 limitation
        limitation_text = sections.get("limitations", "") or self._find_author_limitations(full_text)
        if limitation_text:
            limitation_sentences = self._split_sentences(limitation_text)
            limitation = self._clip_text(
                " ".join(limitation_sentences[:3]), 500, add_ellipsis=True
            )
            for card in evidence_cards:
                card["limitation"] = limitation

        # A numerical/statistical result in the Results section is stronger than a
        # generic result sentence; share it with cards that otherwise lack one.
        section_key_results = self._extract_key_results(findings_text)
        if section_key_results:
            for card in evidence_cards:
                card["key_results"] = card.get("key_results") or section_key_results

        # Preserve the experimental context needed by downstream comparison
        # tables. These are exact or clipped source sentences, not inferred facts.
        explicit_dataset_text = " ".join(filter(None, [
            sections.get("dataset", ""), sections.get("datasets", ""), sections.get("data", ""),
        ]))
        dataset = self._extract_context(
            " ".join(filter(None, [explicit_dataset_text, full_text])),
            r"\b(?:dataset|benchmark|corpus|knowledge graph|language pair|DBP15K|DWY100K|FB15K|WN18)\b",
        )
        metric = self._extract_context(
            full_text,
            r"\b(?:Hits?\s*@\s*(?:1|5|10)|MRR|mean reciprocal rank|accuracy|precision|recall|F1)\b",
        )
        baseline = self._extract_context(
            full_text,
            r"\b(?:baseline|compared with|comparison with|outperform(?:s|ed)?|versus|vs\.)\b",
        )
        experimental_setting = self._extract_context(
            " ".join(filter(None, [
                sections.get("experimental setup", ""), sections.get("experiments", ""),
                sections.get("experiment", ""), sections.get("evaluation", ""), full_text,
            ])),
            r"\b(?:train(?:ing)?|validation|test|split|seed(?: alignment)?|candidate set|ratio|\d+\s*%)\b",
        )
        dataset_name = self._extract_dataset_names(dataset)
        graph_or_language_pair = self._extract_context(
            dataset,
            r"\b(?:knowledge graph pair|graph pair|language pair|cross[- ]lingual|"
            r"DBpedia|Wikidata|YAGO|Freebase|English|Chinese|French|German|Japanese|"
            r"EN[-_/](?:ZH|FR|DE|JA)|ZH[-_/]EN|FR[-_/]EN|DE[-_/]EN)\b",
        )
        entity_count = self._extract_context(
            dataset,
            r"\b(?:\d[\d,.]*\s*(?:k|m|million|thousand)?\s*(?:aligned\s+)?"
            r"entit(?:y|ies)|entity pairs?|alignment pairs?)\b",
        )
        modalities = self._extract_context(
            dataset,
            r"\b(?:visual|image|text(?:ual)?|attribute|relation|structure|multimodal|modality|modalities)\b",
        )
        missingness = self._extract_context(
            dataset,
            r"\b(?:missing|incomplete|absent|noise|noisy|coverage|sparse|unavailable)\b",
        )
        data_split = self._extract_context(
            experimental_setting,
            r"\b(?:train(?:ing)?|validation|dev|test)\b.*\b(?:split|set|ratio|\d+\s*%)\b",
        )
        seed_ratio = self._extract_context(
            experimental_setting,
            r"(?:\b(?:seed(?: alignment)?|supervision|training pairs?|aligned pairs?)\b.*(?:\d+\s*%|ratio)|"
            r"(?:\d+\s*%|ratio).*\b(?:seed(?: alignment)?|supervision|training pairs?|aligned pairs?)\b)",
        )
        dataset_task_consistent = True
        dataset_scope_text = explicit_dataset_text or dataset
        if (
            self._is_mmea_topic(topic) and dataset_scope_text
            and not self._is_mmea_dataset_context(dataset_scope_text)
        ):
            dataset_task_consistent = False
            dataset = ""
            dataset_name = ""
            graph_or_language_pair = ""
            entity_count = ""
            modalities = ""
            missingness = ""
            data_split = ""
            seed_ratio = ""
        for card in evidence_cards:
            card["dataset"] = dataset
            card["dataset_name"] = dataset_name
            card["graph_or_language_pair"] = graph_or_language_pair
            card["entity_count"] = entity_count
            card["modalities"] = modalities
            card["missingness"] = missingness
            card["data_split"] = data_split
            card["seed_ratio"] = seed_ratio
            card["dataset_task_consistent"] = dataset_task_consistent
            card["metric"] = metric
            card["result"] = card.get("key_results", "")
            card["baseline"] = baseline
            card["experimental_setting"] = experimental_setting
            if card.get("result") and card.get("evidence_type") == "primary_claim":
                card["evidence_type"] = "primary_result"

        # Stable IDs are assigned by the program, never by the LLM.  Reviewer
        # findings must reference one of these IDs before they can enter a report.
        for index, card in enumerate(evidence_cards, start=1):
            card["evidence_id"] = f"{source_id}:e{index}"

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "evidence_cards": evidence_cards,
                "source_id": source_id,
                "card_count": len(evidence_cards),
            },
        )

    # ---- 内部方法 ----

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """按 section 标题解析 full_text，返回 {section_name: content} 字典。"""
        sections: Dict[str, str] = {}
        # 找所有 section 边界
        matches = list(self.SECTION_PATTERN.finditer(text))
        if not matches:
            # 没有明确的 section 标题，把全部文本当作 abstract
            sections["abstract"] = text
            return sections

        for i, match in enumerate(matches):
            section_name = match.group(1).strip().rstrip(":").lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections[section_name] = content

        return sections

    def _extract_claims_from_abstract(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取 claims。每句话可能是一个 claim。"""
        sentences = self._split_sentences(text)
        candidates = []
        claim_cues = (
            "we propose", "we present", "we introduce", "we develop",
            "we demonstrate", "we show", "we find", "our results",
            "results show", "outperform", "improve", "achieve", "evaluation",
            "benchmark", "metric", "taxonomy", "framework", "limitation",
        )
        generic_preambles = (
            "this survey provides", "this paper presents a comprehensive review",
            "this paper provides an overview", "the rise of", "continue to play a vital role",
            "has opened new frontiers", "is increasingly critical",
        )
        seen = set()
        for index, sent in enumerate(sentences):
            sent = sent.strip()
            # 过滤太短或明显不是 claim 的句子
            if len(sent) < 30:
                continue
            if sent.startswith(("http", "Figure", "Table", "See ")):
                continue
            lower = sent.lower()
            normalized = re.sub(r"[^a-z0-9]+", " ", lower).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            # 根据线索、数字、长度和通用前缀计算分数，选择分数最高的前3句作为 claim。
            cue_score = sum(2 for cue in claim_cues if cue in lower)
            detail_score = 1 if any(ch.isdigit() for ch in sent) else 0
            length_score = 1 if 60 <= len(sent) <= 500 else 0
            generic_penalty = 3 if any(cue in lower for cue in generic_preambles) else 0
            candidates.append(
                (cue_score + detail_score + length_score - generic_penalty, index, sent)
            )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "text": self._clip_text(sent, 420, add_ellipsis=True),
                # Quotes must remain exact substrings for CitationCheckTool.
                "quote": self._clip_text(sent, 700, add_ellipsis=False),
            }
            for _, _, sent in candidates[:3]
        ]

    def _extract_key_results(self, text: str) -> str:
        """Prefer explicit numerical, comparison, and significance statements."""
        result_cues = re.compile(
            r"(?:\d+(?:\.\d+)?\s*%|p\s*[<=>]\s*0?\.\d+|statistically significant|"
            r"outperform(?:s|ed)?|improv(?:e|es|ed|ement)|increase[sd]?|decrease[sd]?|"
            r"higher than|lower than)", re.IGNORECASE,
        )
        matches = [sentence for sentence in self._split_sentences(text) if result_cues.search(sentence)]
        return self._clip_text(" ".join(matches[:2]), 500, add_ellipsis=True) if matches else ""

    def _find_author_limitations(self, text: str) -> str:
        """Recover explicit task limitations/challenges when no section was parsed."""
        cues = re.compile(
            r"\b(?:our (?:study|work) (?:is|was) limited|we acknowledge|a limitation|"
            r"limitations? (?:include|of)|future work|remains? (?:limited|unclear))\b",
            re.IGNORECASE,
        )
        domain_cues = re.compile(
            r"\b(?:challenge[sd]?|suffer(?:s|ed)? from|sensitive to|rel(?:y|ies) on|"
            r"depend(?:s|ence) on|missing modalit(?:y|ies)|modality missing|noisy (?:image|text|attribute)|"
            r"sparse (?:graph|topology|label)|distribution shift|domain shift|"
            r"scalab(?:ility|le)|computational(?:ly)? (?:cost|expensive)|high (?:training|inference) cost|"
            r"low[- ]resource|bias(?:ed)?|poor generalization|lack(?:s|ing)? (?:robustness|real-world))\b",
            re.IGNORECASE,
        )
        sentences = [
            sentence for sentence in self._split_sentences(text)
            if cues.search(sentence) or domain_cues.search(sentence)
        ]
        return " ".join(sentences[:3])

    def _extract_context(self, text: str, cue_pattern: str) -> str:
        """Return up to two exact source sentences matching an evidence facet."""
        cue = re.compile(cue_pattern, re.IGNORECASE)
        matches = [sentence for sentence in self._split_sentences(text) if cue.search(sentence)]
        return self._clip_text(" ".join(matches[:2]), 700, add_ellipsis=True) if matches else ""

    @staticmethod
    def _extract_dataset_names(text: str) -> str:
        """Return only dataset names explicitly present in the source passage."""
        known = re.findall(
            r"\b(?:DBP15K(?:[-_/][A-Z]{2,3})?|DWY100K(?:[-_/][A-Z]{2,3})?|"
            r"FB15K(?:-DB15K|-YAGO15K|-237)?|FBDB15K|FBYG15K|EMMEAD|"
            r"WN18(?:RR)?|OpenEA|WK3l(?:-\d+k)?|ICEWS\d*|YAGO\d*)\b",
            str(text or ""), re.IGNORECASE,
        )
        return ", ".join(dict.fromkeys(item.strip() for item in known))

    @staticmethod
    def _classify_method_families(text: str) -> str:
        """Classify only mechanisms explicitly mentioned by the source method text."""
        lowered = str(text or "").lower()
        families = []
        cues = (
            ("结构、关系与属性表示", ("graph", "structure", "relation", "attribute", "gnn", "attention")),
            ("多模态特征编码与融合", ("multimodal", "visual", "image", "text", "fusion", "modality")),
            ("联合表示与对齐目标", ("joint representation", "alignment loss", "contrastive", "matching objective")),
            ("半监督或无监督对齐", ("semi-supervised", "unsupervised", "self-supervised")),
            ("伪标签或伪种子增强", ("pseudo-label", "pseudo label", "pseudo-seed", "pseudo seed", "bootstrapping")),
            ("预训练模型或大语言模型", ("pre-trained", "pretrained", "language model", "llm", "foundation model")),
        )
        for family, family_cues in cues:
            if any(cue in lowered for cue in family_cues):
                families.append(family)
        return "；".join(families)

    @staticmethod
    def _is_mmea_topic(topic: str) -> bool:
        lowered = str(topic or "").lower()
        return any(cue in lowered for cue in (
            "mmea", "multimodal entity alignment", "multi-modal entity alignment", "多模态实体对齐",
        ))

    @staticmethod
    def _is_mmea_dataset_context(text: str) -> bool:
        """Reject datasets that reveal an adjacent primary task."""
        lowered = str(text or "").lower()
        excluded = (
            "twitter-2015", "twitter-2017", "cora", "pubmed", "wikics",
            "wikidiverse", "richpediamel", "wikimel", "wn18", "fb15k-237",
            "sentiment classification", "entity linking", "knowledge graph completion",
            "text-attributed graph",
        )
        if any(cue in lowered for cue in excluded):
            return False
        allowed = (
            "dbp15k", "dwy100k", "fb15k-db15k", "fbdb15k", "fb15k-yago15k",
            "fbyg15k", "emmead", "entity alignment dataset", "entity alignment benchmark",
            "knowledge graph pair", "language pair", "aligned entity pair", "mmea benchmark",
        )
        return any(cue in lowered for cue in allowed)

    @staticmethod
    def _evidence_type(source: Dict[str, Any], location: str, quote: str) -> str:
        title = str(source.get("title") or "").lower()
        source_type = str(source.get("source_type") or "").lower()
        if "related work" in str(location).lower():
            return "secondary_summary"
        if source_type in {"survey", "review"} or any(term in title for term in ("survey", "review", "taxonomy")):
            return "review"
        if re.search(r"\b(?:outperform|improv|achiev|\d+(?:\.\d+)?\s*%)", quote, re.IGNORECASE):
            return "primary_result"
        return "primary_claim"

    @staticmethod
    def _explain_relevance(topic: str, claim: str) -> str:
        """Give a compact, deterministic topic relation instead of an opaque score."""
        if not topic:
            return "Supports the report topic through its extracted research claim."
        tokens = {
            token for token in re.findall(r"[\w\u4e00-\u9fff]+", topic.lower())
            if len(token) > 1
        }
        claim_lower = claim.lower()
        matched = [token for token in sorted(tokens) if token in claim_lower]
        if matched:
            return f"Directly relates to the topic through: {', '.join(matched[:5])}."
        return f"Provides evidence for one aspect of the research question: {topic[:120]}."

    @staticmethod
    def _clip_text(text: str, limit: int, add_ellipsis: bool) -> str:
        """Clip at a word boundary; exact quotes never receive synthetic text."""
        clean = text.strip()
        if len(clean) <= limit:
            return clean
        clipped = clean[:limit]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        clipped = clipped.rstrip(" ,;:")
        return f"{clipped}..." if add_ellipsis else clipped

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """简单分句。"""
        # OpenAlex inverted abstracts occasionally reconstruct as "end.Next"
        # without whitespace. Split before the next uppercase sentence too.
        raw = re.split(r'(?<=[.!?])(?:\s+|(?=[A-Z]))', text)
        return [s.strip() for s in raw if s.strip() and len(s.strip()) > 10]
