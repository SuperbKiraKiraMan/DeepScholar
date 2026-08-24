"""Regression coverage for MMEA task boundaries, citations, and Send chapters."""

import asyncio

import pytest

from app.agents.final_reviewer import FinalReviewer
from app.agents.llm_reviewer import LLMDraftReviewer
from app.agents.report_outline import ReportOutlineGenerator
from app.agents.task_relevance import filter_sources_for_task
from app.graph.runtime import (
    node_chapter_writer_send,
    node_merge_chapter_results,
    send_to_chapter_writer,
)
from app.tools.evidence_extract_tool import EvidenceExtractTool
from app.llm.schemas import LLMOutlinePlan, ReportOutline


def test_report_outline_accepts_missing_model_topic_because_runtime_owns_it():
    outline = ReportOutline(sections=[])

    assert outline.topic == ""


def test_outline_plan_normalizes_object_themes_and_evidence_gaps():
    plan = LLMOutlinePlan(
        sections=[],
        cross_cutting_themes=[
            {"theme": "多模态信息融合", "description": "跨章节主题"},
            {"name": "种子监督依赖"},
            "鲁棒性",
        ],
        evidence_gaps=[
            {"question": "缺失模态下能否稳定泛化？", "severity": "high"},
            {"description": "跨数据集比较证据不足。"},
        ],
    )

    assert plan.cross_cutting_themes == ["多模态信息融合", "种子监督依赖", "鲁棒性"]
    assert plan.evidence_gaps == ["缺失模态下能否稳定泛化？", "跨数据集比较证据不足。"]


def test_dynamic_outline_accepts_seven_reader_facing_chapters():
    sections = [
        {
            "heading": f"核心议题{i}",
            "guiding_question": f"议题{i}的关键变化是什么？",
            "assigned_evidence_ids": [],
            "assigned_source_ids": [],
        }
        for i in range(1, 8)
    ]

    assert ReportOutlineGenerator._validate_outline({"sections": sections}, []) == ""


def test_outline_normalizer_resolves_prompt_aliases_and_source_bindings():
    cards = [
        {"evidence_id": "s2:abc:e1", "source_id": "s2:abc"},
        {"evidence_id": "s2:def:e1", "source_id": "s2:def"},
    ]
    outline = {
        "sections": [{
            "heading": "能力演化的关键转折",
            "guiding_question": "智能体能力如何发生阶段性变化？",
            "assigned_evidence_ids": ["[E1]", "E2"],
            "assigned_source_ids": ["P1", "P2"],
        }],
    }

    normalized = ReportOutlineGenerator._normalize_model_outline(
        outline,
        cards,
        {"E1": "s2:abc:e1", "E2": "s2:def:e1"},
        {"s2:abc": "P1", "s2:def": "P2"},
    )

    section = normalized["sections"][0]
    assert section["heading"] == "能力演化的关键转折"
    assert section["assigned_evidence_ids"] == ["s2:abc:e1", "s2:def:e1"]
    assert section["assigned_source_ids"] == ["s2:abc", "s2:def"]


def test_outline_normalizer_does_not_mistake_canonical_evidence_suffix_for_alias():
    cards = [
        {"evidence_id": "s2:first:e1", "source_id": "s2:first"},
        {"evidence_id": "s2:second:e1", "source_id": "s2:second"},
    ]
    outline = {"sections": [{
        "heading": "证据脉络",
        "guiding_question": "不同来源分别支持什么结论？",
        "assigned_evidence_ids": ["s2:second:e1"],
        "assigned_source_ids": [],
    }]}

    normalized = ReportOutlineGenerator._normalize_model_outline(
        outline, cards, {"E1": "s2:first:e1", "E2": "s2:second:e1"},
        {"s2:first": "P1", "s2:second": "P2"},
    )

    assert normalized["sections"][0]["assigned_evidence_ids"] == ["s2:second:e1"]
    assert normalized["sections"][0]["assigned_source_ids"] == ["s2:second"]


def test_mmea_boundary_rejects_adjacent_keyword_overlap_tasks():
    sources = [
        {"source_id": "core1", "title": "Multi-modal Knowledge Graph Entity Alignment: A Survey"},
        {"source_id": "core2", "title": "Not All Imputations are Trustworthy: A Multimodal Entity Alignment Framework"},
        {"source_id": "sentiment", "title": "Target-Entity Sentiment Classification with Image-Text Multimodal Entity Alignment"},
        {"source_id": "mel", "title": "Multi-Element Interaction for Multimodal Entity Linking"},
        {"source_id": "mkgc", "title": "Diffusion Transformer for Multimodal Knowledge Graph Completion"},
        {"source_id": "rag", "title": "A Survey on Multimodal Retrieval-Augmented Generation"},
        {"source_id": "generic", "title": "Multimodal Alignment and Fusion: A Survey"},
        {"source_id": "classroom", "title": "Multimodal Classroom Discourse Collaboration"},
    ]

    eligible, audit = filter_sources_for_task("调研多模态实体对齐", sources)

    assert {item["source_id"] for item in eligible} == {"core1", "core2"}
    assert audit["eligible_count"] == 2
    assert audit["rejected_count"] == 6


def test_reader_citations_remove_heading_clusters_and_internal_evidence_ids():
    report = (
        "## 常用数据集 [14][13][18][7]\n\n"
        "DBP15K 是常用基准（s2:abcdef:e2）。 [14][13][18][7][15]\n"
    )

    cleaned = FinalReviewer._clean_reader_citations(report)

    assert cleaned.splitlines()[0] == "## 常用数据集"
    assert "s2:abcdef:e2" not in cleaned
    assert "[14][13]" in cleaned
    assert "[18]" not in cleaned


def test_reader_citation_after_sentence_punctuation_is_bound_to_the_claim():
    cleaned = FinalReviewer._clean_reader_citations("作者报告该方法优于基线。 [3]\n")

    assert cleaned == "作者报告该方法优于基线[3]。\n"


def test_evidence_markers_become_at_most_two_source_citations():
    synthesis = "该结论由两项研究共同支持。 [[e:S1:e1,S2:e1,S3:e1]]"
    cards = {
        "S1:e1": {"source_id": "S1"},
        "S2:e1": {"source_id": "S2"},
        "S3:e1": {"source_id": "S3"},
    }

    output = LLMDraftReviewer._attach_statement_citations(
        synthesis, list(cards), cards, {"S1": 1, "S2": 2, "S3": 3},
    )

    assert output.endswith("[1][2]")
    assert "[3]" not in output
    assert "S1:e1" not in output


def test_evidence_marker_parser_accepts_chinese_list_separator():
    markers = LLMDraftReviewer._extract_marker_ids(
        "跨来源结论。 [[e:S1:e1、S2:e1]]"
    )

    assert markers == ["S1:e1", "S2:e1"]


def test_evidence_marker_normalizer_restores_only_unique_provider_prefix():
    cards = {
        "s2:abc123:e1": {"source_id": "s2:abc123"},
        "s2:def456:e1": {"source_id": "s2:def456"},
    }

    normalized = LLMDraftReviewer._normalize_marker_ids(
        ["abc123:e1", "unknown:e1"], cards,
    )

    assert normalized == ["s2:abc123:e1", "unknown:e1"]


@pytest.mark.asyncio
async def test_mmea_evidence_rejects_cross_task_dataset_context():
    source = {
        "source_id": "S1", "title": "A Multimodal Entity Alignment Survey",
        "url": "https://example.org/1", "quality_score": 0.8,
        "full_text": (
            "Abstract: We survey multimodal entity alignment methods.\n"
            "Datasets: Twitter-2015 and Twitter-2017 are used for target-entity sentiment classification."
        ),
    }

    result = await EvidenceExtractTool().run(source=source, topic="调研多模态实体对齐")
    card = result.data["evidence_cards"][0]

    assert card["dataset"] == ""
    assert card["dataset_task_consistent"] is False


@pytest.mark.asyncio
async def test_llm_outline_keeps_dynamic_titles_and_recovers_exact_recent_failures(monkeypatch):
    from app.llm.client import FakeLLMClient, reset_llm_client
    import app.llm.client as client_mod

    headings = [
        "从规则链到自主循环", "工具使用成为能力杠杆", "记忆架构的持续演进",
        "规划与反思走向闭环", "多智能体协作边界", "评测范式的重新定义", "安全约束与未来方向",
    ]
    monkeypatch.delenv("LLM_OUTLINE_TIMEOUT_SECONDS", raising=False)
    fake = FakeLLMClient(responses=[{
        # Deliberately omit topic, return seven sections, and use E/P-style
        # prompt aliases: these are the three production drift cases.
        "sections": [
            {
                "heading": heading,
                "guiding_question": f"{heading}体现了怎样的关键变化？",
                "assigned_evidence_ids": [f"E{index}"] if index <= 2 else [],
            }
            for index, heading in enumerate(headings, start=1)
        ],
        "evidence_gaps": [],
    }])
    client_mod._global_client = fake
    cards = [
        {"evidence_id": "s2:first:e1", "source_id": "s2:first", "claim": "Agent planning changed through tool use."},
        {"evidence_id": "s2:second:e1", "source_id": "s2:second", "claim": "Agent memory enables longer autonomous loops."},
    ]
    try:
        outline, result = await ReportOutlineGenerator().generate(
            "生成关于 AI Agent Evolution 的深度调研报告",
            [{"source_id": "s2:first"}, {"source_id": "s2:second"}],
            cards,
            [{"citation_id": 1, "is_valid": True}, {"citation_id": 2, "is_valid": True}],
            language="zh", agent_mode="llm", allow_rule_fallback=False,
        )
    finally:
        reset_llm_client()

    assert result["success"] is True
    assert fake.calls[0]["timeout_seconds"] == 90
    assert outline["topic"] == "生成关于 AI Agent Evolution 的深度调研报告"
    assert [section["heading"] for section in outline["sections"]] == headings
    assert outline["sections"][1]["assigned_evidence_ids"] == ["s2:second:e1"]


@pytest.mark.asyncio
async def test_outline_fans_out_chapters_with_send_and_merges_in_outline_order():
    sources = [
        {"source_id": "S1", "title": "Paper One", "url": "https://example.org/1"},
        {"source_id": "S2", "title": "Paper Two", "url": "https://example.org/2"},
    ]
    cards = [
        {"evidence_id": "S1:e1", "source_id": "S1", "claim": "FIRST_CHAPTER evidence"},
        {"evidence_id": "S2:e1", "source_id": "S2", "claim": "SECOND_CHAPTER evidence"},
    ]
    sections = [
        {"heading": "第一章", "guiding_question": "一？", "assigned_evidence_ids": ["S1:e1"], "assigned_source_ids": ["S1"]},
        {"heading": "第二章", "guiding_question": "二？", "assigned_evidence_ids": ["S2:e1"], "assigned_source_ids": ["S2"]},
    ]
    state = {
        "topic": "测试主题", "language": "zh", "agent_mode": "rule",
        "outline": {"sections": sections, "evidence_gaps": []},
        "sources": sources, "evidence_cards": cards,
    }

    sends = send_to_chapter_writer(state)
    assert [send.node for send in sends] == ["chapter_writer_send", "chapter_writer_send"]

    outputs = await asyncio.gather(*(node_chapter_writer_send(send.arg) for send in reversed(sends)))
    bucket = [item for output in outputs for item in output["_chapter_bucket"]]
    merged = await node_merge_chapter_results({
        **state,
        "_chapter_bucket": bucket,
        "citation_check_results": [
            {"citation_id": 1, "is_valid": True},
            {"citation_id": 2, "is_valid": True},
        ],
        "citation_summary": {"total_checked": 2, "valid_count": 2, "invalid_count": 0},
    })

    report = merged["draft_report"]
    assert report.index("## 第一章") < report.index("## 第二章")
    first = report.split("## 第一章", 1)[1].split("## 第二章", 1)[0]
    second = report.split("## 第二章", 1)[1].split("## 结论", 1)[0]
    assert "论文[1]" in first and "论文[2]" not in first
    assert "论文[2]" in second and "论文[1]" not in second
    assert merged["trace"][0]["merge_type"] == "chapters"
