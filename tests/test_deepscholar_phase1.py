"""Evidence, outline, chapter isolation, and final checks."""

import pytest

from app.agents.draft_reviewer import DraftReviewer
from app.agents.evaluator import Evaluator
from app.agents.final_reviewer import FinalReviewer
from app.agents.report_outline import ReportOutlineGenerator
from app.agents.source_selector import AdaptiveSourceSelector
from app.agents.source_titles import display_paper_title, validated_title_translations
from app.llm.schemas import LLMChapterOutput, LLMFinding
from app.tools.evidence_extract_tool import EvidenceExtractTool


@pytest.mark.asyncio
async def test_full_text_evidence_has_quote_location_and_key_result():
    source = {
        "source_id": "S1", "title": "A Study", "url": "https://example.org/s1",
        "quality_score": 0.9,
        "full_text": (
            "Abstract: We evaluate a retrieval agent on multiple academic tasks and report robust gains.\n"
            "Method: We compare the agent against two controlled baselines.\n"
            "Results: The agent improves citation accuracy by 18.5% and significantly outperforms the baseline.\n"
            "Limitations: Our study is limited to English-language benchmarks."
        ),
    }
    result = await EvidenceExtractTool().run(source=source, topic="retrieval agent evaluation")
    assert result.success
    cards = result.data["evidence_cards"]
    assert cards
    assert all(0 < len(card["original_quote"]) <= 300 for card in cards)
    assert all(card["quote_location"] != "abstract_only" for card in cards)
    assert any("18.5%" in card["key_results"] for card in cards)
    assert all("English-language" in card["limitation"] for card in cards)


@pytest.mark.asyncio
async def test_abstract_only_evidence_does_not_claim_full_text_quote():
    source = {
        "source_id": "S2", "title": "Abstract Study", "url": "https://example.org/s2",
        "abstract": "We demonstrate that an evidence-bound agent improves retrieval reliability across three tasks.",
    }
    result = await EvidenceExtractTool().run(source=source, topic="retrieval reliability")
    card = result.data["evidence_cards"][0]
    assert card["original_quote"] == ""
    assert card["quote_location"] == "abstract_only"


def test_llm_finding_keeps_single_id_and_supports_multiple_ids():
    old = LLMFinding(claim="A sufficiently long grounded claim", source_id="S1", evidence_id="E1")
    new = LLMFinding(
        claim="A sufficiently long corroborated claim", source_id="S1",
        evidence_id="E1", evidence_ids=["E1", "E2"],
    )
    assert old.bound_evidence_ids() == ["E1"]
    assert new.bound_evidence_ids() == ["E1", "E2"]


def test_chapter_schema_normalizes_common_provider_keys_without_weakening_validation():
    output = LLMChapterOutput(
        chapter_title="主要方法",
        content="现有证据表明，不同方法分别利用结构信息和多模态信息完成实体对应关系建模。",
        used_evidence_ids=["S1:e1", "S2:e1"],
        findings=[],
        title_translations={"S1": "多模态实体对齐综述"},
    )
    assert output.heading == "主要方法"
    assert output.synthesis.startswith("现有证据表明")
    assert output.evidence_ids == ["S1:e1", "S2:e1"]
    assert output.source_title_translations == {"S1": "多模态实体对齐综述"}

    with pytest.raises(Exception):
        LLMChapterOutput(chapter_title="主要方法", content="太短")


def test_chapter_schema_ignores_malformed_optional_findings_but_keeps_grounded_prose():
    output = LLMChapterOutput(
        heading="方法",
        synthesis="现有研究分别采用结构引导和特征融合机制，并由给定证据支持这一有界比较。 [[e:S1:e1,S2:e1]]",
        evidence_ids=["S1:e1", "S2:e1"],
        findings=[
            "模型错误地把摘要写进了附加字段",
            {"aspect": "融合机制", "summary": "该字典不是正式 LLMFinding"},
        ],
    )
    assert output.findings == []
    assert output.evidence_ids == ["S1:e1", "S2:e1"]


def test_paper_title_display_preserves_chinese_and_formats_valid_translation():
    english = {"source_id": "S1", "title": "Multimodal Entity Alignment: A Survey"}
    chinese = {"source_id": "S2", "title": "多模态实体对齐研究综述"}
    translations = validated_title_translations(
        {"S1": "多模态实体对齐：一项综述", "unknown": "无效绑定"},
        [english, chinese],
    )
    assert translations == {"S1": "多模态实体对齐：一项综述"}
    assert display_paper_title(english, translations) == (
        "多模态实体对齐：一项综述（Multimodal Entity Alignment: A Survey）"
    )
    assert display_paper_title(chinese, translations) == "多模态实体对齐研究综述"
    assert display_paper_title(english, {"S1": "English only"}) == english["title"]


@pytest.mark.asyncio
async def test_rule_outline_is_bounded_and_program_bound():
    cards = [
        {"evidence_id": "S1:e1", "source_id": "S1", "claim": "A benchmark metric evaluates agent task success."},
        {"evidence_id": "S2:e1", "source_id": "S2", "claim": "Robustness evaluation measures failure and safety risk."},
    ]
    checks = [
        {"citation_id": 1, "is_valid": True},
        {"citation_id": 2, "is_valid": True},
    ]
    outline, result = await ReportOutlineGenerator().generate(
        "LLM Agent evaluation", [{"source_id": "S1"}, {"source_id": "S2"}],
        cards, checks, language="zh", agent_mode="rule",
    )
    assert result["mode"] == "rule"
    assert 3 <= len(outline["sections"]) <= 6
    assignments = [item for section in outline["sections"] for item in section["assigned_evidence_ids"]]
    assert all(assignments.count(item) <= 2 for item in set(assignments))
    assert outline["evidence_gaps"]


def test_rule_chapter_generation_isolates_assigned_evidence():
    sources = [
        {"source_id": "S1", "title": "Paper One", "url": "https://example.org/1"},
        {"source_id": "S2", "title": "Paper Two", "url": "https://example.org/2"},
    ]
    cards = [
        {"evidence_id": "S1:e1", "source_id": "S1", "claim": "UNIQUE_CHAPTER_ONE finding", "dataset_name": "DATASET_ONE", "confidence": 0.9},
        {"evidence_id": "S2:e1", "source_id": "S2", "claim": "UNIQUE_CHAPTER_TWO finding", "dataset_name": "DATASET_TWO", "confidence": 0.9},
    ]
    outline = {
        "sections": [
            {"heading": "第一章", "guiding_question": "问题一？", "assigned_evidence_ids": ["S1:e1"], "assigned_source_ids": ["S1"]},
            {"heading": "第二章", "guiding_question": "问题二？", "assigned_evidence_ids": ["S2:e1"], "assigned_source_ids": ["S2"]},
            {"heading": "证据空白", "guiding_question": "问题三？", "assigned_evidence_ids": [], "assigned_source_ids": []},
        ],
        "evidence_gaps": ["问题三缺少证据"],
    }
    result = DraftReviewer().review(
        "测试", sources, cards,
        [{"citation_id": 1, "is_valid": True}, {"citation_id": 2, "is_valid": True}],
        language="zh", outline=outline,
    )
    first = result["draft_report"].split("## 第一章", 1)[1].split("## 第二章", 1)[0]
    assert "论文[1]" in first
    assert "论文[2]" not in first
    assert len(result["chapter_timings"]) == 3


@pytest.mark.asyncio
async def test_llm_phase1_accepts_alias_shape_and_renders_bilingual_english_title():
    from app.agents.llm_reviewer import LLMDraftReviewer
    from app.llm.client import FakeLLMClient, reset_llm_client
    import app.llm.client as client_mod

    source = {
        "source_id": "S1",
        "title": "Multimodal Entity Alignment: A Survey",
        "url": "https://example.org/s1",
        "year": 2025,
    }
    card = {
        "evidence_id": "S1:e1",
        "source_id": "S1",
        "claim": "The survey categorizes multimodal entity alignment methods.",
        "confidence": 0.9,
    }
    fake = FakeLLMClient(responses=[{
        "chapter_title": "主要方法分类",
        "content": "现有证据将多模态实体对齐方法划分为若干类别，并说明该分类目前只由一项综述性来源直接支持，不能外推为领域共识。 [[e:S1:e1]]",
        "used_evidence_ids": ["S1:e1"],
        "findings": [],
        "title_translations": {"S1": "多模态实体对齐：一项综述"},
    }])
    client_mod._global_client = fake
    try:
        report, llm_result = await LLMDraftReviewer().review(
            topic="多模态实体对齐",
            sources=[source],
            evidence_cards=[card],
            citation_check_results=[{"citation_id": 1, "is_valid": True}],
            citation_summary={"total_checked": 1, "valid_count": 1, "invalid_count": 0},
            language="zh",
            outline={
                "sections": [{
                    "heading": "主要方法分类",
                    "guiding_question": "现有证据如何划分主要方法？",
                    "assigned_evidence_ids": ["S1:e1"],
                    "assigned_source_ids": ["S1"],
                }],
                "evidence_gaps": [],
            },
        )
        assert llm_result["partial_fallback"] is False
        assert report["chapter_timings"][0]["mode"] == "llm"
        assert "多模态实体对齐：一项综述（Multimodal Entity Alignment: A Survey）" in report["draft_report"]
        assert "核心证据显示：The survey" not in report["draft_report"]
        assert "正文必须使用简体中文" in fake.calls[0]["system_prompt"]
    finally:
        reset_llm_client()


@pytest.mark.asyncio
async def test_chapter_uses_valid_prose_marker_when_auxiliary_ids_drift():
    from app.agents.llm_reviewer import LLMDraftReviewer
    from app.llm.client import FakeLLMClient, reset_llm_client
    import app.llm.client as client_mod

    source = {"source_id": "S1", "title": "MMEA Study", "year": 2025}
    card = {
        "evidence_id": "S1:e1", "source_id": "S1",
        "claim": "The study defines the multimodal entity alignment task.",
    }
    client_mod._global_client = FakeLLMClient(responses=[{
        "heading": "引言",
        "synthesis": "现有证据将多模态实体对齐界定为跨异构多模态知识图谱识别等价实体的任务，并将结论限制在该来源所描述的边界内。 [[e:S1:e1]]",
        "evidence_ids": ["provider-alias-that-is-not-in-prose"],
        "findings": [],
        "source_title_translations": {"S1": "多模态实体对齐研究"},
    }])
    try:
        chapter, result = await LLMDraftReviewer().generate_chapter(
            {"heading": "引言", "guiding_question": "任务边界是什么？"},
            [card], [source], language="zh", source_number={"S1": 1},
            allow_rule_fallback=False,
        )
        assert result["success"] is True
        assert "[1]" in chapter
        assert "provider-alias" not in chapter
    finally:
        reset_llm_client()


def test_final_quality_checks_and_repairs_are_recorded():
    long_unsupported = "This unsupported declarative paragraph repeats a broad claim without any source marker and is intentionally longer than eighty characters."
    report = (
        f"# Report\n\n## Chapter A\n\n{long_unsupported}\n\n"
        f"## Chapter B\n\n{long_unsupported}\n\n## Conclusion\n\nA conclusion uses an invalid source [9]."
    )
    sources = [{"source_id": "S1", "title": "Paper", "url": "https://example.org/1"}]
    cards = [{"source_id": "S1", "claim": "A conclusion uses a source", "evidence_id": "S1:e1"}]
    evaluation = Evaluator().evaluate(
        draft_report=report, sources=sources, evidence_cards=cards,
    )
    assert evaluation["metrics"]["chapter_duplication"] is False
    assert evaluation["metrics"]["unsupported_expansion"] is False
    assert evaluation["metrics"]["conclusion_evidence_coverage"] is False

    reviewed = FinalReviewer().review(
        draft_report=report,
        eval_metrics={"metrics": evaluation["metrics"], "metrics_detail": evaluation["metrics_detail"]},
        evidence_cards=cards, sources=sources, language="zh",
    )
    assert "[citation needed]" in reviewed["final_report"]
    assert "已合并章节" in reviewed["final_report"]
    assert "引用编号修复" in reviewed["final_report"]


@pytest.mark.asyncio
async def test_mmea_outline_preserves_required_research_chapters():
    cards = [
        {
            "evidence_id": "S1:e1", "source_id": "S1",
            "claim": "A multimodal alignment method is evaluated on DBP15K.",
            "method": "Cross-modal fusion", "dataset": "DBP15K",
            "metric": "Hits@1", "result": "Hits@1 is 80%", "baseline": "Baseline A",
            "experimental_setting": "20% seed alignment", "limitation": "Missing images reduce performance",
        },
        {
            "evidence_id": "S2:e1", "source_id": "S2",
            "claim": "Another method evaluates noisy multimodal inputs.",
            "method": "Modality-aware encoder", "dataset": "DWY100K",
            "metric": "MRR", "result": "MRR is 0.75", "baseline": "Baseline B",
            "experimental_setting": "30% seed alignment", "limitation": "Training cost grows with graph size",
        },
    ]
    outline, _ = await ReportOutlineGenerator().generate(
        "完整调研 MMEA 的范围、主要方法、数据集、评价指标、实验结果与研究局限",
        [{"source_id": "S1"}, {"source_id": "S2"}], cards,
        [{"citation_id": 1, "is_valid": True}, {"citation_id": 2, "is_valid": True}],
        language="zh", agent_mode="rule",
    )
    assert [section["heading"] for section in outline["sections"]] == [
        "引言", "方法", "数据集", "评价指标", "实验", "局限",
    ]


def test_narrow_mmea_request_does_not_expand_to_full_report():
    assert ReportOutlineGenerator._required_sections("调研 MMEA 的研究局限", "zh") == [
        ("局限", "该领域在数据、方法、泛化和扩展性方面有哪些局限？"),
    ]


def test_llm_outline_assignments_are_supplemented_with_independent_sources():
    cards = [
        {
            "evidence_id": "S1:e1", "source_id": "S1", "claim": "Method one.",
            "method": "Fusion method", "dataset": "DBP15K",
            "limitation": "Missing images are a limitation.", "confidence": 0.9,
        },
        {
            "evidence_id": "S1:e2", "source_id": "S1", "claim": "Method one result.",
            "method": "Fusion encoder", "dataset": "DBP15K",
            "limitation": "Visual noise is a challenge.", "confidence": 0.8,
        },
        {
            "evidence_id": "S2:e1", "source_id": "S2", "claim": "Method two.",
            "method": "Contrastive method", "dataset": "DWY100K",
            "limitation": "Training cost limits scale.", "confidence": 0.85,
        },
        {
            "evidence_id": "S2:e2", "source_id": "S2", "claim": "Method two result.",
            "method": "Graph encoder", "dataset": "DWY100K",
            "limitation": "Domain transfer remains limited.", "confidence": 0.75,
        },
    ]
    required = ReportOutlineGenerator._required_sections(
        "调研多模态实体对齐的主要方法、数据集与研究局限", "zh",
    )
    decision = {
        "assignments": {
            "方法": ["E1", "E2"],
            "数据集": ["E1", "E2"],
            "局限": ["E1", "E2"],
        },
        "evidence_gaps": [],
    }

    outline = ReportOutlineGenerator._outline_from_assignments(
        decision, required, cards, "zh",
        {"E1": "S1:e1", "E2": "S1:e2", "E3": "S2:e1", "E4": "S2:e2"},
    )

    by_heading = {section["heading"]: section for section in outline["sections"]}
    for heading in ("方法", "数据集", "局限"):
        assert set(by_heading[heading]["assigned_source_ids"]) == {"S1", "S2"}


@pytest.mark.asyncio
async def test_model_decides_adaptive_analysis_count_for_comprehensive_request():
    from app.llm.client import FakeLLMClient, reset_llm_client
    import app.llm.client as client_mod

    sources = [
        {
            "source_id": f"S{i}", "title": f"MMEA method dataset limitation study {i}",
            "abstract": "Method benchmark dataset results and limitations.",
            "year": 2020 + i % 6, "cited_by_count": i * 3,
        }
        for i in range(20)
    ]
    selected_ids = [f"S{i}" for i in range(14)]
    client_mod._global_client = FakeLLMClient(responses=[{
        "analysis_count": 14,
        "selected_source_ids": selected_ids,
        "selection_reasons": {source_id: "facet coverage" for source_id in selected_ids},
        "coverage_plan": {"methods": selected_ids[:7], "datasets": selected_ids[7:]},
        "rationale": "Fourteen papers cover independent method, dataset, result, and limitation facets.",
    }])
    try:
        selected, decision = await AdaptiveSourceSelector().select(
            "调研 MMEA 的主要方法、数据集与研究局限", sources, 5, "llm",
        )
        assert decision["mode"] == "llm"
        assert decision["analysis_count"] == 14
        assert len(selected) == 14
        assert decision["minimum_count"] == 12
    finally:
        reset_llm_client()


@pytest.mark.asyncio
async def test_llm_only_selector_keeps_entire_pool_when_no_cutoff_is_needed():
    sources = [
        {"source_id": "S1", "title": "MMEA method", "abstract": "A method."},
        {"source_id": "S2", "title": "MMEA dataset", "abstract": "A benchmark."},
    ]

    selected, decision = await AdaptiveSourceSelector().select(
        "调研多模态实体对齐", sources, requested_count=20,
        agent_mode="llm", allow_rule_fallback=False,
    )

    assert [source["source_id"] for source in selected] == ["S1", "S2"]
    assert decision["mode"] == "all_candidates"
    assert decision["analysis_count"] == 2


@pytest.mark.asyncio
async def test_source_selector_uses_independent_ninety_second_default(monkeypatch):
    from app.llm.client import FakeLLMClient, reset_llm_client
    import app.llm.client as client_mod

    monkeypatch.delenv("LLM_SOURCE_SELECTION_TIMEOUT_SECONDS", raising=False)
    fake = FakeLLMClient(responses=[{
        "analysis_count": 1,
        "selected_source_ids": ["S1"],
        "selection_reasons": {"S1": "best match"},
        "coverage_plan": {"methods": ["S1"]},
        "rationale": "The selected paper best matches the request.",
    }])
    client_mod._global_client = fake
    try:
        await AdaptiveSourceSelector().select(
            "agent evolution",
            [
                {"source_id": "S1", "title": "Agent Evolution One"},
                {"source_id": "S2", "title": "Agent Evolution Two"},
            ],
            requested_count=1,
            agent_mode="llm",
            allow_rule_fallback=False,
        )
    finally:
        reset_llm_client()

    assert fake.calls[0]["timeout_seconds"] == 90


def test_formal_report_gate_removes_internal_audit_and_closes_citations():
    from app.agents.final_reviewer import FinalReviewer

    report = """# MMEA 调研报告

## 执行摘要

现有研究形成了多类技术路线 [1][2]。

## 主要方法分类

两项原始研究采用不同融合机制 [1][2]。

| 方法 | 数据集 |
|---|---|
| A [1] | DBP15K |

## 常用数据集

两项研究分别使用 DBP15K 与 DWY100K [1][2]。

| 数据集 | 划分 |
|---|---|
| DBP15K [1] | 20% seeds |

## 评估指标与实验协议

常用协议报告 Hits@1 与 MRR，并区分种子比例 [1][2]。

## 代表性实验结果

可比结果必须同时说明数据集、指标、数值与基线 [1][2]。

## 研究局限与开放问题

模态缺失与扩展成本仍是两项研究共同涉及的问题 [1][2]。

FinalReviewer latency_ms=1234，规则通过率 10/12。

## 结论

当前研究以多模态融合方法为主，但数据缺失与规模扩展仍未解决 [1][2]。

## 参考文献与证据追踪

- [1] Paper A
- [2] Paper B
"""
    sources = [
        {"source_id": "S1", "title": "Paper A", "url": "https://example.org/a", "authors": ["A"], "year": 2024},
        {"source_id": "S2", "title": "Paper B", "url": "https://example.org/b", "authors": ["B"], "year": 2025},
    ]
    cards = [
        {
            "source_id": "S1", "method": "Fusion A",
            "method_family": "多模态特征编码与融合", "dataset": "DBP15K",
            "dataset_name": "DBP15K", "graph_or_language_pair": "EN-ZH",
            "metric": "Hits@1", "experimental_setting": "20% seeds",
            "seed_ratio": "20% seeds", "limitation": "Missing modalities",
        },
        {
            "source_id": "S2", "method": "Fusion B",
            "method_family": "预训练模型或大语言模型", "dataset": "DWY100K",
            "dataset_name": "DWY100K", "entity_count": "100K entities",
            "metric": "MRR", "experimental_setting": "30% seeds",
            "seed_ratio": "30% seeds", "limitation": "Scaling cost",
        },
    ]
    result = FinalReviewer().review(
        draft_report=report,
        eval_metrics={"metrics": {"unsupported_expansion": True, "chapter_duplication": True}},
        sources=sources, evidence_cards=cards, language="zh",
        topic="调研 MMEA 的主要方法、数据集与研究局限",
    )
    assert result["completion_ready"] is True
    assert "FinalReviewer" not in result["final_report"]
    assert "latency_ms" not in result["final_report"]
    assert "[citation needed]" not in result["final_report"]


def test_completion_gate_allows_explicit_adjacent_task_exclusions_and_paragraph_citations():
    base = """# 多模态实体对齐研究综述

## 方法

该研究报告首次使用自动搜索减少显著的人力设计开销。进一步说明见下句[1][2]。

## 数据集

两项研究使用不同数据集[1][2]。

## 评价指标

两项研究讨论 Hits@1 与 MRR[1][2]。

## 实验

现有证据不足以支持定量性能比较[1][2]。

## 局限

融合开销与数据缺失是两项研究分别报告的局限[1][2]。

## 结论

在任务边界上，本报告不包括知识图谱补全、情感分类或多模态 RAG 等相邻任务[1]。
"""
    sources = [
        {"source_id": "S1", "authors": ["A"], "year": 2024, "url": "https://example.org/1"},
        {"source_id": "S2", "authors": ["B"], "year": 2025, "url": "https://example.org/2"},
    ]
    cards = [
        {
            "source_id": "S1", "method": "Fusion", "method_family": "融合",
            "dataset": "DBP15K", "dataset_name": "DBP15K", "modalities": "text,image",
            "metric": "Hits@1", "limitation": "cost",
        },
        {
            "source_id": "S2", "method": "Search", "method_family": "搜索",
            "dataset": "DWY100K", "dataset_name": "DWY100K", "entity_count": "100K",
            "metric": "MRR", "limitation": "missingness",
        },
    ]

    issues = FinalReviewer._completion_gate(
        base, "调研多模态实体对齐", sources, cards, {},
    )

    assert "正式报告混入多模态实体对齐之外的相邻研究任务" not in issues
    assert "存在无声明级引用的强结论" not in issues
    assert "性能强结论缺少数值或提升幅度" not in issues


def test_source_selection_schema_has_no_fixed_fifty_paper_ceiling():
    from app.llm.schemas import LLMSourceSelectionOutput

    output = LLMSourceSelectionOutput(
        analysis_count=75,
        selected_source_ids=[f"S{i}" for i in range(75)],
    )
    assert output.analysis_count == 75


def test_chinese_mmea_report_uses_core_journal_structure_and_no_english_claim_dump():
    reviewer = DraftReviewer()
    source = {"source_id": "S1", "title": "An English MMEA Paper", "year": 2025}
    card = {
        "evidence_id": "S1:e1", "source_id": "S1",
        "claim": "This long English evidence sentence must not be copied into the Chinese fallback report.",
        "dataset_name": "DBP15K", "metric": "Hits@1",
    }
    outline = {
        "sections": [
            {
                "heading": heading, "guiding_question": f"{heading}是什么？",
                "assigned_evidence_ids": ["S1:e1"], "assigned_source_ids": ["S1"],
            }
            for heading in ("引言", "方法", "数据集", "评价指标", "实验", "局限")
        ],
        "evidence_gaps": [],
    }
    result = reviewer.review(
        "生成关于 Multimodal entity alignment 的调研报告", [source], [card],
        [{"citation_id": 1, "is_valid": True}], language="zh", outline=outline,
    )["draft_report"]

    assert result.startswith("# 多模态实体对齐研究综述")
    assert "## 摘要" in result and "**关键词**：多模态实体对齐" in result
    for heading in ("引言", "方法", "数据集", "评价指标", "实验", "局限", "结论"):
        assert f"## {heading}" in result
    body = result.split("## 参考文献与证据追踪", 1)[0]
    assert "This long English evidence sentence" not in body


def test_rule_comparison_tables_select_supported_columns_and_skip_sparse_tables():
    sources = {
        "S1": {"source_id": "S1", "title": "Paper One", "year": 2024},
        "S2": {"source_id": "S2", "title": "Paper Two", "year": 2025},
    }
    method_cards = [
        {"source_id": "S1", "method_family": "特征级融合", "metric": "Hits@1"},
        {"source_id": "S2", "method_family": "决策级融合", "metric": "MRR"},
    ]
    table = DraftReviewer._method_comparison_table(method_cards, sources, {"S1": 1, "S2": 2}, True)
    assert table[0] == "| 论文 | 年份 | 技术路线 | 评价指标 |"
    assert "未报告" not in "\n".join(table)

    sparse_dataset = [{"source_id": "S1", "dataset_name": "DBP15K"}]
    assert DraftReviewer._dataset_comparison_table(
        sparse_dataset, sources, {"S1": 1, "S2": 2}, True,
    ) == []
