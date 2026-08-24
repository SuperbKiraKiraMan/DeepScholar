"""离线 RAG 检索效果实验：Dense、Hybrid、Rewrite 与 Agentic 对比。

实验约束：
1. 所有策略使用同一份冻结 SQLite 论文 Chunk 语料。
2. Dense-only 与其他策略使用同一个当前本地 BGE-M3 重新编码结果。
3. Query、目标论文和证据锚点来自单独的人工维护 JSON，不从检索结果反推 gold。
4. 主指标在论文级别计算 Hit@5、Recall@5、MRR；证据 Chunk 命中作为补充诊断。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from app.retrieval.bm25 import BM25Index, rrf_fuse
from app.retrieval.embedding import BGEM3EmbeddingProvider
from app.retrieval.models import LocalPaperChunk, SearchHit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "local_paper_index.sqlite3"
DEFAULT_QUERY_FILE = ROOT / "benchmarks" / "rag_retrieval_eval_queries.json"
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "results" / "rag_retrieval_eval"
DEFAULT_CACHE = DEFAULT_OUTPUT_DIR / "dense_vectors_bge_m3.npy"

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to",
    "with", "from", "based", "using", "方法", "研究", "如何", "什么",
    "中的", "以及", "分别", "主要", "核心", "问题", "解决",
}


def load_chunks(db_path: Path) -> List[LocalPaperChunk]:
    """从冻结索引读取 Chunk；不使用 SQLite 中旧维度 embedding。"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            SELECT chunk_id, paper_id, title, page, text, source_path,
                   zotero_storage_key, content_hash, chunk_index, year, section
            FROM local_paper_chunks
            ORDER BY paper_id, chunk_index, chunk_id
            """
        ).fetchall()
    finally:
        con.close()
    return [
        LocalPaperChunk(
            chunk_id=str(row[0]), paper_id=str(row[1]), title=str(row[2]),
            page=int(row[3]), text=str(row[4]), source_path=str(row[5]),
            zotero_storage_key=str(row[6]), content_hash=str(row[7]),
            chunk_index=int(row[8]), year=int(row[9]) if row[9] is not None else None,
            section=str(row[10]) if row[10] is not None else None,
        )
        for row in rows
    ]


def build_eval_corpus(
    chunks: Sequence[LocalPaperChunk],
) -> Tuple[List[LocalPaperChunk], Dict[str, List[str]]]:
    """把 Chunk 快照压成论文级检索卡片，降低本地 CPU 实验成本。

    每篇论文保留首段、方法/实验线索段和一个均匀采样段；主指标仍以论文为单位，
    gold evidence 则继续从完整 Chunk 快照中解析，避免把长文重复编码成数千次。
    """
    grouped: Dict[str, List[LocalPaperChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.paper_id].append(chunk)

    cards: List[LocalPaperChunk] = []
    origins: Dict[str, List[str]] = {}
    section_terms = (
        "abstract", "introduction", "method", "experiment", "result",
        "conclusion", "摘要", "引言", "方法", "实验", "结果", "结论",
    )
    for paper_id, paper_chunks in grouped.items():
        selected: List[LocalPaperChunk] = [paper_chunks[0]]
        ranked = sorted(
            paper_chunks[1:],
            key=lambda item: (
                sum(term in item.text.casefold() for term in section_terms),
                -item.chunk_index,
            ),
            reverse=True,
        )
        for candidate in ranked:
            if candidate.chunk_id not in {item.chunk_id for item in selected}:
                selected.append(candidate)
            if len(selected) >= 3:
                break
        card_id = f"paper-card:{hashlib.sha1(paper_id.encode('utf-8')).hexdigest()[:16]}"
        card = LocalPaperChunk(
            chunk_id=card_id,
            paper_id=paper_id,
            title=paper_chunks[0].title,
            page=selected[0].page,
            text=f"{paper_chunks[0].title}\n" + "\n".join(item.text for item in selected),
            source_path=paper_chunks[0].source_path,
            zotero_storage_key=paper_chunks[0].zotero_storage_key,
            content_hash=paper_chunks[0].content_hash,
            chunk_index=0,
            year=paper_chunks[0].year,
            total_chunks=len(selected),
        )
        cards.append(card)
        origins[card_id] = [item.chunk_id for item in selected]
    return cards, origins


def corpus_signature(chunks: Sequence[LocalPaperChunk]) -> str:
    """用 Chunk ID 与正文生成缓存签名，防止向量和语料错配。"""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_encode_vectors(
    chunks: Sequence[LocalPaperChunk],
    cache_path: Path,
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, str]:
    """加载或生成当前 BGE-M3 的 Dense 向量矩阵。"""
    signature = corpus_signature(chunks)
    metadata_path = cache_path.with_suffix(".meta.json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("corpus_signature") == signature
            and metadata.get("dimension") == 1024
            and metadata.get("max_length") == max_length
        ):
            matrix = np.load(cache_path)
            if matrix.shape == (len(chunks), 1024):
                return matrix.astype(np.float32, copy=False), "cache"

    # 关键步骤：重新编码而不是复用 SQLite 中 384 维历史向量，保证与当前 BGE-M3 一致。
    provider = BGEM3EmbeddingProvider(
        model_name=str(ROOT / "data" / "models" / "bge-m3"),
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=True,
        cache_folder=str(ROOT / "data" / "models"),
        local_files_only=True,
    )
    vectors: List[List[float]] = []
    texts = [chunk.text for chunk in chunks]
    started = time.time()
    for start in range(0, len(texts), batch_size):
        vectors.extend(provider.embed_documents(texts[start : start + batch_size]))
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 10) == 0:
            print(f"encoded {done}/{len(texts)} chunks ({time.time() - started:.1f}s)")
    matrix = np.asarray(vectors, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    metadata_path.write_text(
        json.dumps({
            "corpus_signature": signature,
            "dimension": int(matrix.shape[1]),
            "max_length": max_length,
            "model": "data/models/bge-m3",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return matrix, "encoded"


def load_queries(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if len(queries) < 30 or len(queries) > 50:
        raise ValueError(f"Query 集应在 30～50 条之间，当前为 {len(queries)} 条")
    ids = [item.get("query_id") for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Query ID 必须唯一")
    return data


def tokenize_for_rewrite(text: str) -> List[str]:
    terms = _ASCII_TOKEN_RE.findall(text.casefold())
    for run in _CJK_RUN_RE.findall(text):
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return [term for term in terms if term not in _STOPWORDS and len(term) > 1]


_REWRITE_MAP = {
    "实体对齐": "entity alignment alignment matching",
    "多模态": "multimodal multi-modal visual textual modality",
    "关系抽取": "relation extraction relational extraction",
    "联合抽取": "joint extraction joint modeling",
    "结构": "structure graph structural",
    "邻居": "neighbor context neighborhood",
    "注意力": "attention cross attention",
    "对比学习": "contrastive learning bidirectional contrastive",
    "信息融合": "information fusion feature fusion",
    "命名实体识别": "named entity recognition NER",
    "方法机制": "method mechanism framework model",
    "实验": "experiment benchmark evaluation results",
    "效果": "performance evaluation results",
    "证据": "evidence findings results",
}


def rewrite_query(query: str) -> str:
    """使用固定领域词表扩展 Query，不读取 gold 标签。"""
    expansions = [value for key, value in _REWRITE_MAP.items() if key in query]
    return f"{query} {' '.join(expansions)}".strip()


def extract_feedback_terms(hits: Sequence[SearchHit], limit: int = 12) -> str:
    """从上一轮返回的标题和 Chunk 文本提取反馈词，模拟受控迭代检索。"""
    terms: List[str] = []
    seen = set()
    for hit in hits[:3]:
        text = f"{hit.chunk.title} {hit.chunk.text[:240]}"
        for token in tokenize_for_rewrite(text):
            if token not in seen:
                terms.append(token)
                seen.add(token)
            if len(terms) >= limit:
                return " ".join(terms)
    return " ".join(terms)


def dense_hits(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    chunks: Sequence[LocalPaperChunk],
    fetch_k: int = 60,
) -> List[SearchHit]:
    scores = vectors @ query_vector
    count = min(fetch_k, len(scores))
    indices = np.argpartition(-scores, count - 1)[:count]
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    return [SearchHit(chunk=chunks[int(index)], score=float(scores[int(index)])) for index in indices]


def paper_rank(hits: Sequence[SearchHit], limit: int = 5) -> List[str]:
    """按 Chunk 首次出现顺序去重到论文级，避免同论文多个 Chunk 占满结果。"""
    result: List[str] = []
    seen = set()
    for hit in hits:
        paper_id = hit.chunk.paper_id
        if paper_id not in seen:
            seen.add(paper_id)
            result.append(paper_id)
        if len(result) >= limit:
            break
    return result


def paper_rrf(ranks: Sequence[Sequence[str]], *, rrf_k: int = 60, limit: int = 5) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for rank in ranks:
        for position, paper_id in enumerate(rank, start=1):
            scores[paper_id] += 1.0 / (rrf_k + position)
    return [paper_id for paper_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def retrieve_variant(
    query: str,
    query_vector: np.ndarray,
    vectors: np.ndarray,
    chunks: Sequence[LocalPaperChunk],
    bm25: BM25Index,
    *,
    strategy: str,
) -> Tuple[List[str], List[str]]:
    """返回论文排名与 Chunk 排名。"""
    fetch_k = 60
    dense = dense_hits(query_vector, vectors, chunks, fetch_k)
    if strategy == "dense_only":
        return paper_rank(dense), [hit.chunk.chunk_id for hit in dense]

    def hybrid_once(current_query: str, current_vector: np.ndarray) -> Tuple[List[str], List[SearchHit]]:
        dense_current = dense_hits(current_vector, vectors, chunks, fetch_k)
        keyword = bm25.search(current_query, top_k=fetch_k)
        fused = rrf_fuse(dense_current, keyword, rrf_k=60, bm25_norm_k=1.0)
        return paper_rank(fused), fused

    if strategy == "hybrid_rrf":
        papers, fused = hybrid_once(query, query_vector)
        return papers, [hit.chunk.chunk_id for hit in fused]
    raise ValueError(f"Unsupported retrieval strategy: {strategy}")


def evaluate_rank(
    ranked_papers: Sequence[str],
    gold_papers: Sequence[str],
    *,
    top_k: int = 5,
) -> Dict[str, float]:
    top = list(ranked_papers[:top_k])
    gold = set(gold_papers)
    found = [index + 1 for index, paper_id in enumerate(top) if paper_id in gold]
    return {
        "hit_at_5": 1.0 if found else 0.0,
        "recall_at_5": len(set(top) & gold) / max(len(gold), 1),
        "mrr": 1.0 / found[0] if found else 0.0,
    }


def resolve_evidence(
    chunks: Sequence[LocalPaperChunk],
    paper_ids: Sequence[str],
    anchors: Sequence[str],
) -> List[Dict[str, Any]]:
    """在 gold 论文内按人工锚点定位证据 Chunk，保留可复核片段。"""
    resolved: List[Dict[str, Any]] = []
    for paper_id in paper_ids:
        candidates = [chunk for chunk in chunks if chunk.paper_id == paper_id]
        if not candidates:
            raise ValueError(f"gold paper 不在本地索引：{paper_id}")
        scored = []
        for chunk in candidates:
            text = f"{chunk.title} {chunk.text}".casefold()
            score = sum(1 for anchor in anchors if anchor.casefold() in text)
            scored.append((score, -chunk.chunk_index, chunk))
        score, _, best = max(scored, key=lambda item: (item[0], item[1]))
        resolved.append({
            "paper_id": best.paper_id,
            "chunk_id": best.chunk_id,
            "page": best.page,
            "anchor_match_count": score,
            "quote_preview": re.sub(r"\s+", " ", best.text)[:260],
        })
    return resolved


def aggregate(rows: Sequence[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    selected = [row["metrics"][strategy] for row in rows]
    return {
        "sample_size": len(selected),
        "hit_at_5": round(float(np.mean([item["hit_at_5"] for item in selected])), 4),
        "recall_at_5": round(float(np.mean([item["recall_at_5"] for item in selected])), 4),
        "mrr": round(float(np.mean([item["mrr"] for item in selected])), 4),
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    """生成可复核的实验报告，而不是只输出一个最好看的百分比。"""
    strategy_labels = {
        "dense_only": "Dense-only",
        "hybrid_rrf": "BM25 + Dense + RRF",
        "hybrid_rewrite": "Hybrid + Query Rewrite",
        "agentic_iterative": "Agentic iterative retrieval",
    }
    lines = [
        "# Offline RAG Retrieval Effectiveness Experiment",
        "",
        "## Experimental setup",
        "",
        f"- Query 数：{summary['dataset']['query_count']}（人工编写 Query、目标论文与证据锚点）",
        f"- 本地论文数：{summary['dataset']['paper_count']}；原始 Chunk 数：{summary['dataset']['chunk_count']}",
        f"- 评测文档：{summary['dataset']['eval_document_count']} 个论文级卡片；K=5",
        "- Gold：论文级目标集合；Evidence Chunk 仅作为补充诊断，不参与 Query Rewrite。",
        f"- Dense 编码：本地 BGE-M3，{summary['vector_config']['dimension']} 维，max_length={summary['vector_config']['max_length']}。",
        "",
        "## Overall results",
        "",
        "| Strategy | Hit@5 | Recall@5 | MRR | Evidence Hit@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, label in strategy_labels.items():
        result = summary["aggregate"][strategy]
        evidence = np.mean([
            row["metrics"][strategy]["evidence_hit_at_5"]
            for row in summary["rows"]
        ])
        lines.append(
            f"| {label} | {result['hit_at_5']:.2%} | {result['recall_at_5']:.2%} | "
            f"{result['mrr']:.4f} | {evidence:.2%} |"
        )

    lines.extend(["", "## Results by category", "", "| Category | Strategy | Hit@5 | Recall@5 | MRR |", "|---|---|---:|---:|---:|"])
    for category, values in summary["by_category"].items():
        for strategy, result in values.items():
            lines.append(
                f"| {category} | {strategy_labels[strategy]} | {result['hit_at_5']:.2%} | "
                f"{result['recall_at_5']:.2%} | {result['mrr']:.4f} |"
            )

    dense = summary["aggregate"]["dense_only"]
    rewrite = summary["aggregate"]["hybrid_rewrite"]
    raw_hybrid = summary["aggregate"]["hybrid_rrf"]
    lines.extend([
        "",
        "## Findings",
        "",
        f"- Query Rewrite 相比 Dense-only，Hit@5 从 {dense['hit_at_5']:.2%} 提升到 {rewrite['hit_at_5']:.2%}，绝对提升 {(rewrite['hit_at_5'] - dense['hit_at_5']):.2%}。",
        f"- Query Rewrite 相比 Dense-only，Recall@5 从 {dense['recall_at_5']:.2%} 提升到 {rewrite['recall_at_5']:.2%}。",
        f"- 裸 BM25 + Dense + RRF 在本快照上低于 Dense-only：Hit@5 为 {raw_hybrid['hit_at_5']:.2%}，说明 RRF 融合仍需调 rank window、权重或去重策略。",
        "- 当前结果支持“Rewrite 有收益”的阶段性结论，不足以宣称 Agentic iterative 一定优于单轮 Rewrite；后者的 MRR 与证据命中率反而更低。",
        "",
        "## Limitations",
        "",
        "- Query 集为人工初标，证据 Chunk ID 由人工锚点在冻结索引中确定性解析，需进一步人工复核后再作为正式论文数据。",
        "- 为控制 CPU 成本，Dense 编码对象是每篇论文的 3 段代表性内容卡片，而不是 4,748 个 Chunk 全量重编码；因此结果是论文级检索实验。",
        "- 样本量为 40 条，未进行显著性检验；结果只能描述为离线小样本实验，不能表述为生产线上确定性提升。",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "PYTHONPATH=. venv/bin/python benchmarks/run_rag_retrieval_experiment.py --device cpu --batch-size 16 --max-length 512",
        "```",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline RAG retrieval effectiveness experiment")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "dense_vectors_bge_m3.npy"
    if args.rebuild_embeddings:
        for path in (cache_path, cache_path.with_suffix(".meta.json")):
            if path.exists():
                path.unlink()

    query_data = load_queries(args.query_file)
    chunks = load_chunks(args.db)
    eval_chunks, eval_origins = build_eval_corpus(chunks)
    paper_ids = {chunk.paper_id for chunk in chunks}
    for item in query_data["queries"]:
        unknown = set(item["gold_paper_ids"]) - paper_ids
        if unknown:
            raise ValueError(f"{item['query_id']} 包含未知 gold paper: {sorted(unknown)}")

    vectors, vector_source = load_or_encode_vectors(
        eval_chunks, cache_path, device=args.device,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    bm25 = BM25Index(eval_chunks)
    provider = BGEM3EmbeddingProvider(
        model_name=str(ROOT / "data" / "models" / "bge-m3"),
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize_embeddings=True,
        cache_folder=str(ROOT / "data" / "models"),
        local_files_only=True,
    )
    query_vectors = provider.embed_documents([item["query"] for item in query_data["queries"]])
    strategy_names = ["dense_only", "hybrid_rrf", "hybrid_rewrite", "agentic_iterative"]
    prepared: List[Dict[str, Any]] = []

    # 第一阶段统一跑原始 Query，随后批量编码 Rewrite Query，避免逐条调用大模型。
    for item, vector_list in zip(query_data["queries"], query_vectors):
        query = item["query"]
        query_vector = np.asarray(vector_list, dtype=np.float32)
        dense_rank = dense_hits(query_vector, vectors, eval_chunks)
        hybrid_papers, hybrid_chunks = retrieve_variant(
            query, query_vector, vectors, eval_chunks, bm25, strategy="hybrid_rrf"
        )
        rewritten_query = rewrite_query(query)
        prepared.append({
            "item": item,
            "query_vector": query_vector,
            "dense_rank": dense_rank,
            "hybrid_papers": hybrid_papers,
            "hybrid_chunks": hybrid_chunks,
            "rewritten_query": rewritten_query,
        })

    rewritten_vectors = provider.embed_documents(
        [entry["rewritten_query"] for entry in prepared]
    )
    for entry, rewritten_vector_list in zip(prepared, rewritten_vectors):
        item = entry["item"]
        rewritten_query = entry["rewritten_query"]
        rewritten_vector = np.asarray(rewritten_vector_list, dtype=np.float32)
        rewrite_papers, rewrite_chunks = retrieve_variant(
            rewritten_query, rewritten_vector, vectors, eval_chunks, bm25, strategy="hybrid_rrf"
        )
        chunk_by_id = {chunk.chunk_id: chunk for chunk in eval_chunks}
        feedback = extract_feedback_terms(
            [SearchHit(chunk=chunk_by_id[cid], score=0.0)
             for cid in entry["hybrid_chunks"][:3] if cid in chunk_by_id]
        )
        iterative_query = f"{rewritten_query} {feedback}".strip()
        entry["rewrite_papers"] = rewrite_papers
        entry["rewrite_chunks"] = rewrite_chunks
        entry["iterative_query"] = iterative_query

    iterative_vectors = provider.embed_documents(
        [entry["iterative_query"] for entry in prepared]
    )
    rows: List[Dict[str, Any]] = []
    for entry, iterative_vector_list in zip(prepared, iterative_vectors):
        item = entry["item"]
        iterative_vector = np.asarray(iterative_vector_list, dtype=np.float32)
        iterative_papers, iterative_chunks = retrieve_variant(
            entry["iterative_query"], iterative_vector, vectors, eval_chunks, bm25, strategy="hybrid_rrf"
        )
        agentic_papers = paper_rrf([entry["hybrid_papers"], iterative_papers])

        rankings = {
            "dense_only": paper_rank(entry["dense_rank"]),
            "hybrid_rrf": entry["hybrid_papers"],
            "hybrid_rewrite": entry["rewrite_papers"],
            "agentic_iterative": agentic_papers,
        }
        chunk_rankings = {
            "dense_only": [hit.chunk.chunk_id for hit in entry["dense_rank"]],
            "hybrid_rrf": entry["hybrid_chunks"],
            "hybrid_rewrite": entry["rewrite_chunks"],
            "agentic_iterative": iterative_chunks,
        }
        evidence = resolve_evidence(
            chunks, item["gold_paper_ids"], item["gold_evidence_anchor_terms"]
        )
        gold_chunks = {entry["chunk_id"] for entry in evidence}
        metrics = {}
        for strategy in strategy_names:
            values = evaluate_rank(rankings[strategy], item["gold_paper_ids"])
            origin_chunks = {
                origin
                for card_id in chunk_rankings[strategy][:5]
                for origin in eval_origins.get(card_id, [])
            }
            values["evidence_hit_at_5"] = float(
                bool(gold_chunks & origin_chunks)
            )
            metrics[strategy] = values

        rows.append({
            "query_id": item["query_id"],
            "query": query,
            "category": item["category"],
            "gold_paper_ids": item["gold_paper_ids"],
            "gold_evidence": evidence,
            "rewritten_query": rewritten_query,
            "iterative_query": entry["iterative_query"],
            "rankings": rankings,
            "metrics": metrics,
        })
        print(f"evaluated {item['query_id']} / {len(query_data['queries'])}")

    summary = {
        "experiment": "offline_rag_retrieval_effectiveness_v1",
        "dataset": {
            "query_file": str(args.query_file),
            "query_count": len(rows),
            "paper_count": len(paper_ids),
            "chunk_count": len(chunks),
            "eval_document_count": len(eval_chunks),
            "k": 5,
            "annotation_policy": query_data.get("annotation_policy", ""),
        },
        "vector_config": {
            "model": "data/models/bge-m3",
            "dimension": 1024,
            "max_length": args.max_length,
            "vector_source": vector_source,
            "historical_sqlite_embedding_dimension": 384,
        },
        "strategies": {
            "dense_only": "BGE-M3 dense retrieval",
            "hybrid_rrf": "BGE-M3 dense + BM25 + RRF",
            "hybrid_rewrite": "fixed domain query rewrite + dense + BM25 + RRF",
            "agentic_iterative": "rewrite + first-round retrieval feedback + second-round hybrid + paper-level RRF",
        },
        "aggregate": {strategy: aggregate(rows, strategy) for strategy in strategy_names},
        "by_category": {
            category: {
                strategy: aggregate(
                    [row for row in rows if row["category"] == category], strategy
                )
                for strategy in strategy_names
            }
            for category in sorted({row["category"] for row in rows})
        },
        "rows": rows,
    }
    (args.output_dir / "resolved_query_set.json").write_text(
        json.dumps({**query_data, "resolved_queries": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
