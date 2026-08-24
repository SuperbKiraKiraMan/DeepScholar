"""使用已配置的真实 LLM 执行 live smoke test。

This test uses synthetic paper text so it exercises conversation grounding without
depending on a search provider. It never prints or persists API credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The desktop test environment may define an empty variable, which makes the
# default dotenv loader intentionally keep that empty value. Fill only missing
# LLM settings from the local file and never print them.
from dotenv import dotenv_values
_local_env = dotenv_values(PROJECT_ROOT / ".env")
for _name in ("AGENT_MODE", "LLM_PROVIDER", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
    if not os.environ.get(_name) and _local_env.get(_name):
        os.environ[_name] = str(_local_env[_name])

from fastapi.testclient import TestClient

from app.core.config import config  # Loads the local .env before constructing the LLM client.
from app.llm.client import get_llm_client, load_llm_config, reset_llm_client
from app.main import app
from app.services.context_compressor import CompactionConfig, ContextCompressor
from app.services.run_store import run_store
from app.services.session_store import SessionStore, session_store
from app.services.user_memory import MemoryEntry, UserMemoryStore


PAPERS = [
    {
        "source_id": "live-p1",
        "title": "Dense Retrieval Study",
        "url": "https://example.org/live-p1",
        "snippet": "A controlled study of dense retrieval for research question answering.",
        "full_text": (
            "Abstract. We study dense retrieval for research question answering. "
            "Methods. The system encodes queries and passages with separate transformer encoders, "
            "retrieves the top five passages by cosine similarity, and passes only those passages "
            "to the answer generator. Experiments use 1,200 questions split into 800 training, "
            "200 validation, and 200 test examples. Results. Dense retrieval achieved 78.4 percent "
            "answer accuracy. Limitations. The evaluation uses one English-domain dataset and does "
            "not measure robustness to temporal distribution shift."
        ),
    },
    {
        "source_id": "live-p2",
        "title": "Sparse Retrieval Baseline",
        "url": "https://example.org/live-p2",
        "snippet": "A sparse BM25 baseline evaluated on the same research QA task.",
        "full_text": (
            "Abstract. We evaluate sparse retrieval for research question answering. "
            "Methods. The system applies BM25 over paragraph text and sends the top ten passages "
            "to a constrained answer generator. Experiments use 1,200 English questions with the "
            "same 800, 200, and 200 split. Results. Sparse retrieval achieved 72.1 percent answer "
            "accuracy but required less indexing memory. Limitations. The study does not evaluate "
            "multilingual questions or hybrid dense-sparse retrieval."
        ),
    },
]


def _run_live(client, session_id: str, topic: str) -> dict:
    """POST /api/research/runs 启动异步 run 并轮询到完成，返回 run_store 完整记录。"""
    resp = client.post(
        "/api/research/runs?backend=graph_send",
        json={"topic": topic, "agent_mode": "llm", "session_id": session_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]
    deadline = time.time() + 120
    data = None
    while time.time() < deadline:
        data = run_store.get(run_id)
        if data and data.get("status") not in {"queued", "running", "started"}:
            return data
        time.sleep(0.2)
    raise SystemExit(f"Live run {run_id} did not finish; last: {data}")


def _successful_models(payload: dict) -> list[str]:
    return [
        str(item.get("model") or "")
        for item in payload.get("trace", [])
        if item.get("event") == "llm_finished" and item.get("success")
    ]


async def _exercise_compaction_and_memory(client):
    with tempfile.TemporaryDirectory(prefix="academic-copilot-live-") as temp:
        root = Path(temp)
        compressor = ContextCompressor(CompactionConfig(
            tool_result_budget_bytes=500,
            single_tool_result_bytes=200,
            persisted_preview_chars=80,
            max_messages=12,
            keep_head_messages=2,
            keep_tail_messages=8,
            recent_tool_results=2,
            l4_token_threshold=20,
            l4_min_turns=1,
            l4_turn_gap=1,
        ), root)
        compact_session = SessionStore().create("live-compaction")
        compact_session.turn_count = 12
        messages = []
        for index in range(9):
            messages.extend([
                {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{index}"}]},
                {"role": "tool", "tool_call_id": f"c{index}", "content": (f"result {index} " * 80)},
            ])
        compaction = await compressor.compress(messages, compact_session, llm_client=client)
        assert compaction.summary and compaction.summary.summary
        assert compaction.levels_applied == [
            "L3_tool_result_budget", "L1_snip_compact", "L2_micro_compact", "L4_compact_summary"
        ]
        assert Path(compaction.transcript_path).exists()

        memory = UserMemoryStore(root / ".memory")
        memory.write(MemoryEntry(
            name="answer-style", memory_type="user", title="Research answer style",
            body="The user prefers concise Chinese answers with exact evidence quotations.",
            tags=["Chinese", "citations"],
        ))
        memory.write(MemoryEntry(
            name="unrelated-travel", memory_type="project", title="Travel planning",
            body="A future trip should prioritize rail travel.", tags=["travel"],
        ))
        relevant = await memory.load_relevant(
            "请用中文简洁回答，并给出准确原文引用", client
        )
        assert any(item.name == "answer-style" for item in relevant), json.dumps(
            memory.last_selection_result, ensure_ascii=False, default=str
        )

        memory_sessions = SessionStore()
        memory_session = memory_sessions.create("live-memory")
        for _ in range(3):
            memory_session = memory_sessions.record_turn(
                memory_session.session_id,
                user_content="请跨会话记住：我偏好简洁中文回答，并附准确原文引用。",
                assistant_content="已了解。",
                intent="paper_qa",
            )
        extracted = await memory.extract(memory_session, llm_client=client)
        return {
            "compaction_levels": compaction.levels_applied,
            "compaction_summary": compaction.summary.model_dump(),
            "memory_selected": [item.name for item in relevant],
            "memory_extracted": [item.model_dump() for item in extracted],
        }


def main() -> None:
    reset_llm_client()
    llm_config = load_llm_config()
    if not llm_config.api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not configured; live test cannot run")
    llm = get_llm_client()
    if not llm.is_available:
        raise SystemExit("Real LLM client is unavailable")

    context = session_store.create()
    session_store.set_recommended_papers(context.session_id, PAPERS)
    session_store.set_active_paper(context.session_id, "live-p1")

    report_id = run_store.create("Dense versus sparse retrieval", session_id=context.session_id)
    report = (
        "# Retrieval Comparison Report\n\n"
        "## Method comparison\nDense retrieval uses separate transformer encoders and top-five cosine retrieval [1]. "
        "Sparse retrieval uses BM25 and top-ten passage retrieval [2].\n\n"
        "## Evidence gaps\nThe supplied studies do not evaluate multilingual robustness or a hybrid retriever."
    )
    run_store.update(
        report_id,
        status="completed",
        final_report=report,
        sources=PAPERS,
        evidence_cards=[
            {"evidence_id": "E1", "source_id": "live-p1", "claim": "Dense method", "quote": "top five passages by cosine similarity"},
            {"evidence_id": "E2", "source_id": "live-p2", "claim": "Sparse method", "quote": "applies BM25 over paragraph text"},
        ],
    )
    session_store.set_report_sections(
        context.session_id, ["Method comparison", "Evidence gaps"], report_id
    )

    api = TestClient(app)
    qa = _run_live(api, context.session_id, "这篇论文使用了什么检索方法？请引用原文。")
    compare = _run_live(api, context.session_id, "对比第一篇和第二篇的方法、结果与局限。")
    follow_up = _run_live(api, context.session_id, "展开 Evidence gaps 章节，并说明证据边界。")

    assert qa["intent"] == "paper_qa" and qa["conversation_result"]["supporting_quotes"], json.dumps({
        "intent": qa.get("intent"), "result": qa.get("conversation_result"),
        "llm_events": [item for item in qa.get("trace", []) if item.get("event", "").startswith("llm")],
    }, ensure_ascii=False)
    assert compare["intent"] == "paper_compare" and len(compare["conversation_result"]["comparison_dimensions"]) >= 5, json.dumps({
        "result": compare.get("conversation_result"),
        "llm_events": [item for item in compare.get("trace", []) if item.get("event", "").startswith("llm")],
    }, ensure_ascii=False)
    assert follow_up["intent"] == "report_follow_up" and follow_up["answer"], json.dumps(follow_up.get("conversation_result"), ensure_ascii=False)
    route_models = {
        "paper_qa": _successful_models(qa),
        "paper_compare": _successful_models(compare),
        "report_follow_up": _successful_models(follow_up),
    }
    assert all(route_models.values()), json.dumps(route_models, ensure_ascii=False)
    models = route_models["paper_qa"] + route_models["paper_compare"] + route_models["report_follow_up"]
    assert models and all("fake" not in model.lower() and model != "rule" for model in models)

    extra = asyncio.run(_exercise_compaction_and_memory(llm))
    output = {
        "tested_at_ms": int(time.time() * 1000),
        "provider": llm_config.provider,
        "configured_model": llm_config.model,
        "successful_trace_models": models,
        "route_models": route_models,
        "session_id": context.session_id,
        "turns": {
            "paper_qa": {
                "route": qa["route_name"],
                "resolved_paper_ids": qa["resolved_paper_ids"],
                "answer": qa["answer"],
                "supporting_quotes": qa["conversation_result"]["supporting_quotes"],
            },
            "paper_compare": {
                "route": compare["route_name"],
                "resolved_paper_ids": compare["resolved_paper_ids"],
                "summary": compare["conversation_result"]["summary"],
                "dimension_count": len(compare["conversation_result"]["comparison_dimensions"]),
            },
            "report_follow_up": {
                "route": follow_up["route_name"],
                "resolved_section": follow_up["resolved_section"],
                "answer": follow_up["answer"],
            },
        },
        **extra,
    }
    output_path = Path("harness/reports/phase2_live_llm_latest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "model": llm_config.model,
        "successful_llm_calls": len(models) + 3,
        "routes": [qa["route_name"], compare["route_name"], follow_up["route_name"]],
        "compaction_levels": extra["compaction_levels"],
        "memory_selected": extra["memory_selected"],
        "report": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
