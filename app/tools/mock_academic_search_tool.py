"""
app/tools/mock_academic_search_tool.py

MockAcademicSearchTool —— 离线学术搜索工具。

Agent MVP 的搜索入口。内置 7 篇带完整学术字段的 mock 论文数据，
每条记录包含 full_text fixture，确保离线 demo 时：
- EvidenceExtractTool 有文本可抽取
- CitationCheckTool 有内容可校验
- 不会因为网络失败导致 citation/evidence 为空

类比 Spring Boot：
- MockAcademicSearchTool ≈ @Service("mockAcademicSearchService")
- 通过 SEARCH_PROVIDER=academic 环境变量激活
- 后续可替换为 ArxivSearchTool / OpenAlexSearchTool 等真实实现
"""

import re
import uuid
from typing import Any, Dict, List

from app.tools.base import BaseTool, ToolResult

# ================================================================
# Mock 学术论文数据库（7 篇，含 full_text fixture）
# ================================================================
# 设计原则：
# 1. 每条记录有完整的学术元数据（authors/year/venue/source_type）
# 2. full_text 足够详细，使 EvidenceExtractTool 能提取 claim/method/limitation
# 3. 覆盖 paper/benchmark/tool 等不同类型
# 4. 内容聚焦 RAG/Agent 评估这一主题方向

_MOCK_PAPERS: List[Dict[str, Any]] = [
    {
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "url": "https://arxiv.org/abs/2309.15217",
        "snippet": "We introduce RAGAS, a framework for automated evaluation of RAG systems. "
                   "The framework proposes metrics for faithfulness, answer relevancy, "
                   "context recall, and context precision without requiring human annotations.",
        "full_text": (
            "Title: RAGAS: Automated Evaluation of Retrieval Augmented Generation\n\n"
            "Authors: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert\n\n"
            "Abstract: We introduce RAGAS (Retrieval Augmented Generation Assessment), "
            "a framework for automated reference-free evaluation of RAG systems. "
            "RAGAS proposes four key metrics: faithfulness (whether the answer is "
            "grounded in the retrieved context), answer relevancy (whether the answer "
            "addresses the question), context recall (whether all relevant information "
            "was retrieved), and context precision (whether retrieved context is relevant). "
            "These metrics use LLM-as-judge with carefully designed prompts.\n\n"
            "Method: The framework decomposes RAG evaluation into component-level metrics. "
            "Faithfulness is measured by decomposing the answer into atomic claims and "
            "checking each against the retrieved context. Answer relevancy uses question "
            "generation from the answer and computes similarity with the original question. "
            "Context recall identifies key sentences in the ground truth and checks if "
            "they appear in the retrieved context.\n\n"
            "Limitations: The framework relies on LLM judgments which may have biases. "
            "It does not evaluate citation accuracy directly. Ground truth is needed for "
            "context recall. The approach has not been validated on non-English languages."
        ),
        "authors": ["Shahul Es", "Jithin James", "Luis Espinosa-Anke", "Steven Schockaert"],
        "year": 2023,
        "venue": "arXiv preprint (arXiv:2309.15217)",
        "source_type": "paper",
    },
    {
        "title": "Evaluating RAG Pipelines: A Comprehensive Survey of Metrics and Methods",
        "url": "https://arxiv.org/abs/2405.12345",
        "snippet": "This paper surveys evaluation methods for Retrieval-Augmented Generation systems, "
                   "covering retrieval quality, answer faithfulness, answer relevance, and citation accuracy.",
        "full_text": (
            "Title: Evaluating RAG Pipelines: A Comprehensive Survey of Metrics and Methods\n\n"
            "Authors: Michael Chen, Sarah Johnson, David Smith\n\n"
            "Abstract: This paper presents a comprehensive survey of evaluation methods "
            "for Retrieval-Augmented Generation (RAG) systems. We taxonomize evaluation "
            "along three dimensions: retrieval quality (how well the retriever finds "
            "relevant documents), generation quality (how faithful and relevant the "
            "generated answer is), and end-to-end pipeline quality (how the components "
            "interact). We compare frameworks including RAGAS, TruLens, DeepEval, and "
            "ARES across these dimensions.\n\n"
            "Method: We systematically reviewed 87 papers from 2020-2024. We propose "
            "a unified taxonomy with 12 evaluation dimensions and map each existing "
            "metric to one or more dimensions. We also identify gaps in current "
            "evaluation practices.\n\n"
            "Key Findings: (1) Retrieval quality is the dominant factor for faithful "
            "generation—improving retrieval consistently outperforms improving generation. "
            "(2) Citation accuracy is under-evaluated: most frameworks focus on content "
            "quality, not whether citations actually support claims. (3) Human evaluation "
            "remains the gold standard but is expensive; automated metrics correlate "
            "moderately (r=0.6-0.8) with human judgments.\n\n"
            "Limitations: The survey focuses on English-language systems. The taxonomy "
            "may not cover emerging RAG architectures. The comparison of frameworks "
            "is based on reported results, not controlled experiments."
        ),
        "authors": ["Michael Chen", "Sarah Johnson", "David Smith"],
        "year": 2024,
        "venue": "Proceedings of ACL 2024",
        "source_type": "paper",
    },
    {
        "title": "DeepEval: Unit Testing for LLM Applications",
        "url": "https://docs.confident-ai.com/",
        "snippet": "DeepEval provides metrics like groundedness, answer relevancy, contextual recall, "
                   "contextual precision, and hallucination detection for RAG evaluation.",
        "full_text": (
            "Title: DeepEval: Unit Testing for LLM Applications\n\n"
            "Authors: Confident AI Team\n\n"
            "Abstract: DeepEval is an open-source evaluation framework for LLM applications "
            "that treats evaluation like unit testing. It provides 20+ metrics including "
            "groundedness, answer relevancy, contextual recall, contextual precision, "
            "hallucination detection, toxicity, and bias. DeepEval integrates directly "
            "with pytest, enabling CI/CD evaluation pipelines.\n\n"
            "Method: DeepEval implements each metric as a composable test case. Metrics "
            "can use deterministic scoring (e.g., exact match, cosine similarity), "
            "LLM-as-judge (using GPT-4 or other models), or hybrid approaches. Results "
            "are stored in a local SQLite database for tracking over time.\n\n"
            "Limitations: LLM-as-judge metrics depend on the quality of the judge model. "
            "Some metrics require reference answers. The framework's scoring can vary "
            "across judge model versions. Not all metrics are suitable for real-time "
            "evaluation in production systems."
        ),
        "authors": ["Confident AI Team"],
        "year": 2024,
        "venue": "Confident AI (Open Source)",
        "source_type": "tool",
    },
    {
        "title": "TruLens: Evaluate and Track LLM Applications",
        "url": "https://www.trulens.org/",
        "snippet": "TruLens offers feedback functions for RAG evaluation including groundedness, "
                   "relevance, and custom metrics. It supports the RAG triad.",
        "full_text": (
            "Title: TruLens: Evaluate and Track LLM Applications\n\n"
            "Authors: TruEra Team\n\n"
            "Abstract: TruLens is an open-source observability and evaluation framework "
            "for LLM applications. It introduces the 'RAG Triad' of evaluation: "
            "context relevance (is the retrieved context relevant to the query?), "
            "groundedness (is the answer supported by the context?), and answer relevance "
            "(does the answer address the query?). TruLens provides both feedback functions "
            "and a visualization dashboard.\n\n"
            "Method: Each feedback function uses a chain-of-thought approach: the LLM "
            "is first asked to reason about the evaluation criteria, then produce a score. "
            "TruLens supports both human feedback and automated LLM feedback within the "
            "same evaluation pipeline.\n\n"
            "Limitations: The RAG Triad does not explicitly evaluate citation correctness. "
            "Chain-of-thought evaluation increases latency and cost. The dashboard is "
            "designed for development, not production monitoring."
        ),
        "authors": ["TruEra Team"],
        "year": 2024,
        "venue": "TruEra (Open Source)",
        "source_type": "tool",
    },
    {
        "title": "Faithfulness and Factuality in Retrieval-Augmented Generation: A Benchmark Study",
        "url": "https://arxiv.org/abs/2310.12345",
        "snippet": "A benchmark study evaluating faithfulness and factuality across 5 RAG architectures. "
                   "Finds that retrieval quality is the dominant factor for faithful generation.",
        "full_text": (
            "Title: Faithfulness and Factuality in Retrieval-Augmented Generation: A Benchmark Study\n\n"
            "Authors: Yunfan Zhang, Rui Wang, Hai Zhao\n\n"
            "Abstract: We present a systematic benchmark study of faithfulness and factuality "
            "across five RAG architectures. Using a dataset of 2,000 questions spanning "
            "biomedical, legal, and technical domains, we compare vanilla RAG, RAG with "
            "re-ranking, RAG with query rewriting, RAG with self-reflection, and a "
            "commercial RAG system.\n\n"
            "Method: We introduce two new metrics: Factual Precision (proportion of "
            "generated statements that are factually correct) and Factual Recall "
            "(proportion of source facts that appear in the answer). We use both "
            "LLM-as-judge and human annotators for evaluation.\n\n"
            "Key Findings: (1) Retrieval quality drives 65% of the variance in "
            "faithfulness scores. (2) Self-reflection RAG reduces hallucination by "
            "23% compared to vanilla RAG. (3) Query rewriting improves retrieval "
            "recall by 18% but can introduce semantic drift. (4) No single architecture "
            "dominates across all domains.\n\n"
            "Limitations: The study uses English-only data. The commercial RAG system "
            "version is fixed and may not reflect current capabilities. Human evaluation "
            "used 3 annotators per sample (moderate agreement, Fleiss' kappa=0.65)."
        ),
        "authors": ["Yunfan Zhang", "Rui Wang", "Hai Zhao"],
        "year": 2023,
        "venue": "EMNLP 2023",
        "source_type": "paper",
    },
    {
        "title": "RIO: A Benchmark for Retrieval-Augmented Generation Evaluation",
        "url": "https://github.com/example/rio-benchmark",
        "snippet": "RIO is a benchmark for evaluating retrieval-augmented generation across multiple "
                   "dimensions including retrieval precision, answer quality, and citation accuracy.",
        "full_text": (
            "Title: RIO: A Benchmark for Retrieval-Augmented Generation Evaluation\n\n"
            "Authors: RIO Benchmark Contributors\n\n"
            "Abstract: RIO (Retrieval-Integrated Output) is an open benchmark for "
            "evaluating RAG systems. It includes 500 test queries across 5 domains "
            "(biomedicine, law, technology, finance, education), each with a curated "
            "document corpus of 10,000 documents, ground truth answers, and citation "
            "annotations that mark which documents support which claims.\n\n"
            "Method: RIO evaluates three dimensions: (1) Retrieval Precision/Recall "
            "against the curated corpus, (2) Answer Quality using both automated metrics "
            "and human evaluation rubrics, and (3) Citation Accuracy by checking whether "
            "cited document spans actually support the associated claims.\n\n"
            "Key Features: RIO is the first benchmark to include citation-level ground "
            "truth, enabling precise measurement of citation hallucination. It provides "
            "a leaderboard and standardized evaluation scripts.\n\n"
            "Limitations: The 10,000-document corpus per domain is moderate in size. "
            "Ground truth is limited to English. The benchmark does not cover "
            "multi-modal or multi-turn RAG scenarios."
        ),
        "authors": ["RIO Benchmark Contributors"],
        "year": 2024,
        "venue": "GitHub (Open Source)",
        "source_type": "benchmark",
    },
    {
        "title": "Contextual Relevancy in RAG: Beyond Keyword Matching",
        "url": "https://arxiv.org/abs/2402.56789",
        "snippet": "Proposes a novel metric for contextual relevancy in RAG systems "
                   "that goes beyond traditional keyword matching, introducing semantic overlap "
                   "and information coverage as key dimensions.",
        "full_text": (
            "Title: Contextual Relevancy in RAG: Beyond Keyword Matching\n\n"
            "Authors: Lin Zhang, Wei Chen\n\n"
            "Abstract: Traditional RAG evaluation relies heavily on keyword overlap "
            "metrics like BLEU and ROUGE, which fail to capture semantic relevance. "
            "We propose Semantic Context Overlap (SCO), a metric that uses sentence "
            "embeddings to measure the semantic similarity between retrieved context "
            "and the ideal context for answering a query.\n\n"
            "Method: SCO computes the maximum cosine similarity between each sentence "
            "in the retrieved context and any sentence in a reference context. It also "
            "introduces Information Coverage (IC), which measures what fraction of "
            "the reference context's semantic clusters are covered by retrieved context. "
            "Both metrics use off-the-shelf sentence transformers without fine-tuning.\n\n"
            "Results: SCO achieves Spearman correlation of 0.78 with human judgments "
            "of context relevancy, compared to 0.52 for BLEU and 0.61 for ROUGE-L. "
            "IC provides complementary information about coverage breadth.\n\n"
            "Limitations: SCO requires a reference context, which may not be available "
            "in production settings. Performance depends on the quality of the sentence "
            "embedding model. The metric has only been validated on English and Chinese."
        ),
        "authors": ["Lin Zhang", "Wei Chen"],
        "year": 2024,
        "venue": "NAACL 2024 Findings",
        "source_type": "paper",
    },
]


class MockAcademicSearchTool(BaseTool):
    """
    离线学术搜索工具。

    Agent MVP 的搜索入口。完全离线运行，返回带 full_text 的 PaperSource 列表。

    搜索策略：
    1. 从 query 中提取英文关键词
    2. 在 mock 论文库的 title + snippet 中匹配关键词
    3. 未匹配时返回全部 mock 数据作为兜底
    4. 每条结果分配唯一的 source_id（UUID 前 8 位），不依赖数组下标

    为什么 source_id 用 UUID 而不是数组下标：
    - 下标会因排序、过滤而改变，导致下游引用错位
    - UUID 是不可变标识，SourceQualityScorer 通过 source_id 绑定分数
    - CitationCheckTool 通过 source_id 校验引用来源
    """

    @property
    def name(self) -> str:
        return "mock_academic_search"

    @property
    def description(self) -> str:
        return (
            "Search for academic papers, benchmarks, and tools related to a research topic. "
            "Returns structured PaperSource objects with title, url, snippet, full_text, "
            "authors, year, venue, and source_type. Completely offline (mock data)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research topic or search query, e.g. 'RAG evaluation methods'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5, max 7)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def _arun(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "").strip()
        max_results = min(kwargs.get("max_results", 5), len(_MOCK_PAPERS))

        if not query:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Search query is empty",
            )

        # 关键词提取
        query_lower = query.lower()
        keywords = re.findall(r'[a-zA-Z]{2,}', query_lower)

        # 关键词匹配
        matched = []
        for paper in _MOCK_PAPERS:
            text = (paper["title"] + " " + paper["snippet"]).lower()
            if any(kw in text for kw in keywords):
                matched.append(paper)

        # 无匹配时兜底：返回全部
        if not matched:
            matched = list(_MOCK_PAPERS)

        # 截断 + 分配 source_id
        results = []
        for paper in matched[:max_results]:
            source = {
                "source_id": str(uuid.uuid4())[:8],
                "title": paper["title"],
                "url": paper["url"],
                "snippet": paper["snippet"],
                "full_text": paper["full_text"],
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "source_type": paper.get("source_type", "unknown"),
            }
            results.append(source)

        return ToolResult(
            success=True,
            tool_name=self.name,
            data={
                "results": results,
                "query": query,
                "total_found": len(results),
                "provider": "mock_academic",
            },
        )
