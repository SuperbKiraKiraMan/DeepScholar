"""
tests/test_evidence_extract.py

EvidenceExtractTool 测试 —— Phase 1A。
"""

import pytest
from app.tools.evidence_extract_tool import EvidenceExtractTool


# 测试用 source
_FAKE_SOURCE = {
    "source_id": "test001",
    "url": "https://arxiv.org/abs/2405.12345",
    "title": "RAG Evaluation: A Comprehensive Survey",
    "full_text": (
        "Title: RAG Evaluation: A Comprehensive Survey\n\n"
        "Abstract: This paper surveys evaluation methods for RAG systems. "
        "We find that retrieval quality is the dominant factor for faithful generation. "
        "The study compares RAGAS, TruLens, and DeepEval frameworks.\n\n"
        "Method: We reviewed 87 papers from 2020-2024. We propose a taxonomy "
        "with 12 evaluation dimensions. Each metric is mapped to one or more dimensions.\n\n"
        "Key Findings: (1) Retrieval quality explains 65% of faithfulness variance. "
        "(2) Self-reflection RAG reduces hallucination by 23%. "
        "(3) Citation accuracy is under-evaluated in current frameworks.\n\n"
        "Limitations: The survey focuses on English-language systems. "
        "The taxonomy may not cover emerging architectures. "
        "Comparisons are based on reported results, not controlled experiments."
    ),
    "quality_score": 0.85,
}


class TestEvidenceExtractTool:
    """测试 EvidenceExtractTool。"""

    def setup_method(self):
        self.tool = EvidenceExtractTool()

    @pytest.mark.asyncio
    async def test_extract_from_full_text(self):
        """从 full_text 中抽取证据卡。"""
        result = await self.tool.run(source=_FAKE_SOURCE)

        assert result.success is True
        data = result.data
        assert data["source_id"] == "test001"
        assert data["card_count"] >= 1
        cards = data["evidence_cards"]
        assert len(cards) >= 1

    @pytest.mark.asyncio
    async def test_evidence_cards_have_required_fields(self):
        """每张证据卡包含必填字段。"""
        result = await self.tool.run(source=_FAKE_SOURCE)

        for card in result.data["evidence_cards"]:
            assert card["evidence_id"].startswith("test001:e")
            assert "claim" in card
            assert len(card["claim"]) > 0
            assert "quote" in card
            assert len(card["quote"]) > 0
            assert "source_id" in card
            assert card["source_id"] == "test001"
            assert "url" in card
            assert card["url"] == _FAKE_SOURCE["url"]
            assert "confidence" in card
            assert 0.0 <= card["confidence"] <= 1.0
            assert "method" in card
            assert "limitation" in card
            assert "method_family" in card
            assert "dataset_name" in card
            assert "graph_or_language_pair" in card
            assert "entity_count" in card
            assert "modalities" in card
            assert "missingness" in card
            assert "data_split" in card
            assert "seed_ratio" in card

    @pytest.mark.asyncio
    async def test_extracts_structured_mmea_dataset_protocol_fields(self):
        source = {
            "source_id": "MMEA1",
            "url": "https://example.org/mmea1",
            "title": "Multimodal Entity Alignment",
            "quality_score": 0.9,
            "full_text": (
                "Abstract: We propose a multimodal entity alignment model on DBP15K.\n"
                "Methods: Our graph encoder combines structural, relation, visual image, and textual features.\n"
                "Datasets: DBP15K EN-ZH contains 15,000 aligned entity pairs with visual and textual modalities; some images are missing.\n"
                "Experimental Setup: We use 20% seed alignment pairs for training and separate validation and test sets.\n"
                "Results: The model reaches Hits@1 of 81.2% compared with Baseline A at 78.0%.\n"
                "Limitations: Missing images and noisy attributes reduce performance."
            ),
        }

        result = await self.tool.run(source=source)
        card = result.data["evidence_cards"][0]

        assert card["dataset_name"].lower() == "dbp15k"
        assert "EN-ZH" in card["graph_or_language_pair"]
        assert "15,000" in card["entity_count"]
        assert "visual" in card["modalities"].lower()
        assert "missing" in card["missingness"].lower()
        assert "20%" in card["seed_ratio"]
        assert card["method_family"]

    @pytest.mark.asyncio
    async def test_openalex_abstract_shape_produces_ranked_exact_evidence(self):
        """A reconstructed OpenAlex abstract has no section headers but is still usable."""
        source = {
            "source_id": "W123",
            "url": "https://openalex.org/W123",
            "title": "Evaluation of Retrieval-Augmented Generation",
            "full_text": (
                "Retrieval augmented generation is increasingly used in language systems. "
                "We propose a benchmark covering retrieval quality, faithfulness, and citation accuracy. "
                "Results show that retrieval errors account for 42 percent of unsupported answers. "
                "The evaluation is limited to English-language datasets."
            ),
            "quality_score": 0.82,
            "provider": "openalex",
        }
        result = await self.tool.run(source=source)
        cards = result.data["evidence_cards"]

        assert result.success is True
        assert cards
        assert cards[0]["evidence_id"] == "W123:e1"
        assert cards[0]["quote"] in source["full_text"]
        assert any("42 percent" in card["claim"] for card in cards)

    @pytest.mark.asyncio
    async def test_openalex_reconstructed_abstract_splits_without_spaces(self):
        source = {
            "source_id": "W456",
            "url": "https://openalex.org/W456",
            "full_text": (
                "Agent evaluation remains underdeveloped.This survey proposes a taxonomy "
                "of behavior, capability, reliability, and safety.We identify realistic "
                "benchmarks as an open research direction."
            ),
            "quality_score": 0.8,
        }

        result = await self.tool.run(source=source)
        cards = result.data["evidence_cards"]

        assert len(cards) >= 2
        assert all("." not in card["claim"][:-1] for card in cards)

    @pytest.mark.asyncio
    async def test_confidence_matches_quality_score(self):
        """confidence 来自 source 的 quality_score。"""
        result = await self.tool.run(source=_FAKE_SOURCE)

        for card in result.data["evidence_cards"]:
            # Abstract claims 的 confidence = quality_score
            # Key Findings claims 的 confidence = quality_score * 0.95
            # 都在 quality_score 的合理范围内
            assert 0.75 <= card["confidence"] <= 0.86, (
                f"confidence {card['confidence']} out of expected range [0.75, 0.86]"
            )

    @pytest.mark.asyncio
    async def test_method_extracted(self):
        """Method section 被抽取。"""
        result = await self.tool.run(source=_FAKE_SOURCE)

        cards = result.data["evidence_cards"]
        assert len(cards) >= 1
        # 至少有一张卡有 method
        methods = [c["method"] for c in cards if c["method"]]
        assert len(methods) >= 1, "Should extract method from Method section"
        assert "reviewed" in methods[0].lower() or "87" in methods[0]

    @pytest.mark.asyncio
    async def test_limitation_extracted(self):
        """Limitations section 被抽取。"""
        result = await self.tool.run(source=_FAKE_SOURCE)

        cards = result.data["evidence_cards"]
        limitations = [c["limitation"] for c in cards if c["limitation"]]
        assert len(limitations) >= 1, "Should extract limitation from Limitations section"

    @pytest.mark.asyncio
    async def test_empty_source_returns_error(self):
        """空 source 返回错误。"""
        result = await self.tool.run(source={})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_no_full_text_returns_error(self):
        """没有 full_text 返回错误。"""
        result = await self.tool.run(source={
            "source_id": "test",
            "url": "http://x.com",
            "full_text": "",
        })
        assert result.success is False
        assert "no full_text" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fallback_on_text_without_sections(self):
        """无 section 标题的文本也能抽取（fallback 逻辑）。"""
        source = {
            "source_id": "plain001",
            "url": "http://plain.com",
            "full_text": (
                "This is a simple research paper about RAG evaluation. "
                "It proposes new metrics for measuring faithfulness and relevancy. "
                "The results show that retrieval quality is important. "
                "However, the study has limitations in scope and language coverage."
            ),
            "quality_score": 0.6,
        }
        result = await self.tool.run(source=source)

        assert result.success is True
        cards = result.data["evidence_cards"]
        assert len(cards) >= 1
        # 无 section 时也应优先保留信息量更高的具体方法/结果句。
        assert "metrics" in cards[0]["claim"] or "results" in cards[0]["claim"].lower()

    @pytest.mark.asyncio
    async def test_quote_is_from_full_text(self):
        """quote 内容来自 full_text。"""
        result = await self.tool.run(source=_FAKE_SOURCE)

        for card in result.data["evidence_cards"]:
            # quote 应该是 full_text 的子串（允许大小写差异）
            quote_lower = card["quote"].lower()
            full_lower = _FAKE_SOURCE["full_text"].lower()
            # 至少前 50 个字符在 full_text 中
            assert quote_lower[:50] in full_lower, (
                f"Quote not found in full_text: '{card['quote'][:80]}...'"
            )
