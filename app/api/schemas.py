"""
app/api/schemas.py

Pydantic 请求/响应模型 —— 类比 Spring Boot 的 Request DTO / Response VO。

定义 Agent 主链路的核心数据结构。
- PaperSource：学术来源（搜索结果的标准化表示）
- EvidenceCard：证据卡（claim + quote + source_id + confidence）
- CitationCheckResult：引用校验结果
- ResearchRequest / ResearchResponse：API 契约
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ================================================================
# 核心领域对象（Agent MVP 主链路数据结构）
# ================================================================


class PaperSource(BaseModel):
    """
    学术来源 —— Agent MVP 的核心数据单元。

    这是搜索结果的标准化表示，不同于 RAG chunk：
    - RAG chunk：文档被自动切分后的文本片段，没有结构
    - PaperSource：结构化的学术来源记录，包含完整元数据

    每个 PaperSource 有唯一 source_id（不可变），后续所有引用、评分、
    证据卡都通过 source_id 关联到它。
    """
    source_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="来源唯一标识（不可变，不依赖数组下标）",
    )
    title: str = Field(..., description="来源标题")
    url: str = Field(..., description="来源链接")
    snippet: str = Field(default="", description="摘要/片段")
    full_text: str = Field(default="", description="完整文本或 fixture 文本")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    year: Optional[int] = Field(default=None, description="发表年份")
    venue: str = Field(default="", description="发表 venue（期刊/会议）")
    source_type: str = Field(default="unknown", description="来源类型: paper/benchmark/tool/blog")
    quality_score: Optional[float] = Field(default=None, description="来源质量评分 [0, 1]")
    provider: str = Field(default="", description="搜索数据提供方")
    openalex_id: Optional[str] = Field(default=None, description="OpenAlex Work ID")
    semantic_scholar_id: Optional[str] = Field(
        default=None, description="Semantic Scholar paperId"
    )
    corpus_id: Optional[int] = Field(default=None, description="Semantic Scholar corpusId")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    cited_by_count: Optional[int] = Field(default=None, description="OpenAlex 被引数")
    reference_count: Optional[int] = Field(default=None, description="参考文献数量")
    is_oa: Optional[bool] = Field(default=None, description="是否开放获取")
    oa_status: Optional[str] = Field(default=None, description="OpenAlex OA 状态")
    content_url: Optional[str] = Field(default=None, description="OpenAlex Content API URL")
    has_content: Dict[str, Any] = Field(default_factory=dict, description="OpenAlex 全文可用性")
    content_source: Optional[str] = Field(default=None, description="full_text 的数据来源")
    publication_date: Optional[str] = Field(default=None, description="发布日期")
    research_task: str = Field(default="", description="规范化后的论文主要研究任务")
    task_relevance: str = Field(default="", description="相对当前问题的任务级相关性")
    task_relevance_reason: str = Field(default="", description="任务边界判定理由")


class LocalPaperDetail(BaseModel):
    """本地 Zotero 论文的论文级详情（PDF 来源等 RAG 特有字段）。"""

    paper_id: str = ""
    source_path: str = ""
    zotero_storage_key: str = ""
    chunk_count: int = 0
    size_bytes: int = 0
    indexed_at: str = ""
    seen_in_runs: List[str] = Field(default_factory=list)
    snippet: str = ""


class PaperDetailResponse(BaseModel):
    """论文详情检索响应：本地优先命中，否则 S2/OpenAlex 在线兜底。"""

    query: str
    found: bool = False
    provider: str = "none"  # local_zotero | semantic_scholar | openalex
    resolved_via: str = "none"  # local | online
    matched_local: bool = False
    abstract: str = ""
    paper: Optional[PaperSource] = None
    local: Optional[LocalPaperDetail] = None
    error: str = ""


class EvidenceCard(BaseModel):
    """
    证据卡 —— Agent MVP 的可信度核心。

    每条 EvidenceCard 代表一个"主张-证据"对：
    - claim：从 source 中提取的研究主张
    - quote：支撑该主张的原文引用片段
    - source_id：指向 PaperSource 的唯一标识
    - confidence：该证据的置信度

    与 RAG chunk 的本质区别：
    - RAG chunk 是"被动检索"的文本片段，谁都可以放进上下文
    - EvidenceCard 是"主动抽取"的结构化证据，有来源绑定和置信度
    - CitationCheckTool 可以对 EvidenceCard 做规则校验，chunk 做不到
    """
    evidence_id: str = Field(default="", description="稳定的证据标识，用于 Reviewer 绑定")
    claim: str = Field(..., description="研究主张/结论陈述")
    quote: str = Field(..., description="支撑该主张的原文引用片段")
    source_id: str = Field(..., description="对应 PaperSource 的 source_id")
    url: str = Field(..., description="来源链接")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="证据置信度 [0, 1]")
    method: str = Field(default="", description="涉及的研究方法（如有）")
    dataset: str = Field(default="", description="数据集、图谱/语言对及规模信息")
    dataset_name: str = Field(default="", description="明确出现的数据集或基准名称")
    graph_or_language_pair: str = Field(default="", description="知识图谱对或语言对")
    entity_count: str = Field(default="", description="实体规模及对齐实体规模原文")
    modalities: str = Field(default="", description="数据集包含的视觉、文本、属性或结构模态")
    missingness: str = Field(default="", description="模态缺失、噪声或不完整性说明")
    data_split: str = Field(default="", description="训练、验证、测试划分")
    seed_ratio: str = Field(default="", description="种子对齐比例或监督比例")
    metric: str = Field(default="", description="评估指标，如 Hits@1、Hits@10、MRR")
    result: str = Field(default="", description="包含数值和实验条件的结果陈述")
    baseline: str = Field(default="", description="性能结论对应的比较基线")
    experimental_setting: str = Field(default="", description="种子比例、划分、候选空间等实验设置")
    limitation: str = Field(default="", description="局限性（如有）")
    key_results: str = Field(default="", description="优先保留数值、对比或统计结果")
    original_quote: str = Field(
        default="", max_length=300, description="不超过 300 字符的可追溯原文摘录"
    )
    quote_location: str = Field(default="", description="原文所在章节；仅摘要时为 abstract_only")
    evidence_quote: str = Field(default="", description="支撑该证据卡的原文片段")
    page_or_section: str = Field(default="", description="原文页码或章节位置")
    evidence_type: str = Field(
        default="primary_claim",
        description="primary_claim | primary_result | secondary_summary | review",
    )
    relevance_to_topic: str = Field(default="", description="该证据与本次研究主题的关系")
    method_family: str = Field(default="", description="按技术机制归纳的方法家族，可包含多个类别")
    research_task: str = Field(default="", description="该证据所属的规范化学术任务")
    task_relevance: str = Field(default="", description="core | background | excluded")
    dataset_task_consistent: bool = Field(default=True, description="数据集是否属于当前规范化研究任务")


class CitationCheckResult(BaseModel):
    """
    引用校验结果。

    CitationCheckTool 对每条引用逐项检查后输出的结构化结果。
    所有检查均为规则判断，不依赖 LLM。
    """
    citation_id: int = Field(..., description="被检查的引用编号")
    source_id: str = Field(..., description="对应的 PaperSource source_id")
    id_exists: bool = Field(default=True, description="引用编号是否在 source list 中存在")
    url_matches_source: bool = Field(default=True, description="URL 是否来自 source list")
    quote_found_in_source: bool = Field(default=True, description="quote 是否能在 source full_text 中找到")
    is_valid: bool = Field(default=True, description="综合判断：是否通过所有校验")
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")


class SourceMatrixEntry(BaseModel):
    """
    论文/来源矩阵中的一行。

    用于前端展示和最终报告中的来源对比表。
    """
    source_id: str = Field(..., description="来源唯一标识")
    title: str = Field(..., description="来源标题")
    authors: str = Field(default="", description="作者（缩写字符串）")
    year: Optional[int] = Field(default=None, description="年份")
    venue: str = Field(default="", description="发表 venue")
    source_type: str = Field(default="unknown", description="来源类型")
    quality_score: float = Field(default=0.0, description="质量评分")
    key_contribution: str = Field(default="", description="对本次研究的关键贡献")


# ================================================================
# API Request / Response（类比 Spring Boot 的 DTO / VO）
# ================================================================


class ResearchRequest(BaseModel):
    """
    研究请求。

    类比：Java Controller 方法的 @RequestBody ResearchRequestDto
    """
    topic: str = Field(..., description="研究主题", examples=["LLM Agent evaluation methods"])
    language: str = Field(default="zh", description="输出语言", examples=["zh", "en"])
    max_sources: int = Field(
        default=5, ge=1, le=50,
        description="期望分析来源数；深度调研会由模型按问题复杂度在安全上限内调整",
    )
    mode: str = Field(default="quick", description="调研模式: quick | deep")
    session_id: Optional[str] = Field(default=None, description="多轮研究会话 ID；省略时自动创建")
    run_eval: bool = Field(default=True, description="是否运行质量评估")
    agent_mode: Optional[str] = Field(
        default=None,
        description="Agent mode. The product UI uses LLM-only execution; rule remains an internal test utility.",
    )


class ResearchResponse(BaseModel):
    """
    研究响应。

    类比：Java Controller 返回给前端的 ResearchResponseVO
    """
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="本次运行唯一标识",
    )
    session_id: str = Field(default="", description="多轮研究会话 ID")
    topic: str = Field(..., description="研究主题")
    research_topic: str = Field(default="", description="Controller 解析后的核心研究主题")
    intent: str = Field(default="deep_research", description="Controller 识别的请求意图")
    execution_route: str = Field(
        default="full_research",
        description="执行路径: direct_tool | full_research",
    )
    selected_tools: List[str] = Field(default_factory=list, description="Controller 选择的工具")
    selected_tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="直达工具的可信参数；用于继续分页获取论文",
    )
    intent_confidence: float = Field(default=0.0, description="意图分类置信度")
    is_follow_up: bool = Field(default=False, description="是否为对已有论文/报告的追问")
    reference_expression: str = Field(default="", description="用户原始指代表达")
    resolved_paper_ids: List[str] = Field(default_factory=list, description="解析出的论文 ID")
    seed_paper_ids: List[str] = Field(default_factory=list, description="续接调研注入 Research DAG 的会话论文 ID")
    resolved_section: Optional[str] = Field(default=None, description="解析出的报告章节")
    clarification_message: str = Field(default="", description="需要用户补充的会话澄清信息")
    route_name: str = Field(default="full_research", description="实际执行路由")
    answer: str = Field(default="", description="本轮直接回答；新研究任务通常等于 final_report")
    conversation_result: Dict[str, Any] = Field(default_factory=dict, description="追问节点结构化输出")
    status: str = Field(
        default="running",
        description="run 状态: queued | running | completed | completed_with_warnings | partial | failed | cancelled",
    )
    final_report: str = Field(default="", description="最终研究报告")
    draft_report: str = Field(default="", description="草稿报告")
    sources: List[PaperSource] = Field(default_factory=list, description="来源列表")
    discovered_source_count: int = Field(default=0, ge=0, description="检索阶段发现的候选论文数")
    analyzed_source_count: int = Field(default=0, ge=0, description="进入分析流程的论文数")
    analysis_selection: Dict[str, Any] = Field(
        default_factory=dict, description="分析预算、入选论文及分层覆盖理由（生成详情）",
    )
    evidence_cards: List[EvidenceCard] = Field(default_factory=list, description="证据卡列表")
    outline: Dict[str, Any] = Field(default_factory=dict, description="证据预绑定的报告大纲")
    report_completion_ready: bool = Field(default=False, description="是否通过正式报告硬性验收")
    report_completion_issues: List[str] = Field(default_factory=list, description="未通过的正式报告验收项")
    citation_check_results: List[CitationCheckResult] = Field(default_factory=list, description="引用校验结果")
    source_matrix: List[SourceMatrixEntry] = Field(default_factory=list, description="来源矩阵")
    eval_metrics: Dict[str, Any] = Field(default_factory=dict, description="评估指标")
    observability_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Run、Node、Worker、Tool 与 LLM 的运行遥测聚合",
    )
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="执行 trace")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="创建时间",
    )


class ResearchRunResponse(BaseModel):
    """
    异步研究运行响应（202 Accepted）。
    """
    run_id: str = Field(..., description="任务唯一标识")
    status: str = Field(default="queued", description="任务状态: queued | running")
    topic: str = Field(..., description="研究主题")
    session_id: str = Field(default="", description="多轮研究会话 ID")


class SessionCreateRequest(BaseModel):
    ttl_minutes: int = Field(default=30, ge=1, le=1440)


class SessionResponse(BaseModel):
    session_id: str
    ttl_minutes: int
    recommended_papers: List[Dict[str, Any]] = Field(default_factory=list)
    last_recommendation_batch: List[Dict[str, Any]] = Field(default_factory=list)
    last_recommendation_topic: str = ""
    active_paper_id: Optional[str] = None
    active_report_id: Optional[str] = None
    last_intent: str = ""
    last_mentioned_paper_ids: List[str] = Field(default_factory=list)
    last_report_sections: List[str] = Field(default_factory=list)
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    compaction_count: int = 0
    summary_so_far: str = ""
    turn_count: int = 0
    created_at_ms: int
    updated_at_ms: int
    expires_at_ms: int
    restored_from_session_id: Optional[str] = None


class PaperPageResponse(BaseModel):
    """论文推荐与引用图谱的增量分页响应。"""

    run_id: str
    intent: str
    items: List[PaperSource] = Field(default_factory=list)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1)
    returned: int = Field(default=0, ge=0)
    total: Optional[int] = Field(default=None, ge=0)
    has_more: bool = False
    next_offset: Optional[int] = Field(default=None, ge=0)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "1.3.0"
