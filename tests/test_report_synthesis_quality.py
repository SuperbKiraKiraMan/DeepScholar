"""Regression tests for substantive, evidence-grounded report synthesis."""

import pytest

from app.agents.draft_reviewer import DraftReviewer
from app.graph.runtime import _dedup_sources
from app.tools.evidence_extract_tool import EvidenceExtractTool


def _checks(count):
    return [
        {"citation_id": index, "is_valid": True, "source_id": f"s{index}"}
        for index in range(1, count + 1)
    ]


def test_rule_fallback_is_a_research_synthesis_not_pipeline_narration():
    sources = [
        {"source_id": "s1", "title": "Agent Evaluation Survey", "year": 2025,
         "source_type": "paper", "quality_score": 0.8, "provider": "openalex",
         "full_text": "Agent benchmarks evaluate planning and tool use."},
        {"source_id": "s2", "title": "Robust Agent Evaluation", "year": 2024,
         "source_type": "paper", "quality_score": 0.75, "provider": "openalex",
         "full_text": "Robustness tests expose tool errors and unsafe actions."},
    ]
    cards = [
        {"source_id": "s1", "claim": "Agent benchmarks should evaluate planning and tool-use trajectories.",
         "quote": "Agent benchmarks evaluate planning and tool use.", "confidence": 0.8},
        {"source_id": "s2", "claim": "Robustness tests expose tool errors and unsafe actions.",
         "quote": "Robustness tests expose tool errors and unsafe actions.", "confidence": 0.75},
    ]

    report = DraftReviewer().review(
        "LLM Agent evaluation methods", sources, cards, _checks(2),
        {"total_checked": 2, "valid_count": 2, "invalid_count": 0, "all_valid": True},
    )["draft_report"]

    for heading in (
        "Executive Summary", "Thematic Synthesis", "Research Gaps",
        "Practical Takeaways", "Limitations", "Conclusion",
    ):
        assert heading in report
    assert "planned multiple academic searches" not in report.lower()
    assert "normalized and ranked" not in report.lower()
    assert "planning and tool-use trajectories" in report
    assert len(report) > 1800


def test_rule_fallback_deduplicates_repeated_evidence_claims():
    claim = "Evaluation should cover task performance and social risk."
    report = DraftReviewer().review(
        "LLM evaluation",
        [{"source_id": "s1", "title": "Survey", "quality_score": 0.7}],
        [
            {"source_id": "s1", "claim": claim, "quote": claim},
            {"source_id": "s1", "claim": claim, "quote": claim},
        ],
        _checks(2),
        {"total_checked": 2, "valid_count": 2, "invalid_count": 0},
    )["draft_report"]

    assert report.count(f"- **[1] Survey**: {claim}") == 1


def test_exact_duplicate_titles_are_merged_across_openalex_records():
    sources = [
        {"source_id": "W1", "url": "https://doi.org/10.1/preprint",
         "title": "A Survey on Evaluation of Large Language Models", "quality_score": 0.58},
        {"source_id": "W2", "url": "https://doi.org/10.1/published",
         "title": "A Survey on Evaluation of Large Language Models", "quality_score": 0.72,
         "full_text": "A longer published abstract."},
    ]

    result = _dedup_sources(sources)

    assert len(result) == 1
    assert result[0]["source_id"] == "W2"


@pytest.mark.asyncio
async def test_long_evidence_is_clipped_at_words_and_quote_remains_exact():
    long_sentence = "We present " + "meaningful evaluation evidence " * 40 + "for agent systems."
    source = {
        "source_id": "long1", "url": "https://example.test/long",
        "full_text": long_sentence, "quality_score": 0.8,
    }

    result = await EvidenceExtractTool().run(source=source)
    card = result.data["evidence_cards"][0]

    assert card["quote"] in long_sentence
    assert not card["quote"].endswith("...")
    assert card["claim"].endswith("...")
    assert card["claim"][:-3].split()[-1] in {"meaningful", "evaluation", "evidence"}

