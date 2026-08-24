<div align="center">
  <img src="app/web/assets/academic-copilot-logo.png" alt="Academic Research Copilot" width="220">
  <h1>Academic Research Copilot</h1>
  <p>An Agent-first research assistant for scholarly search, evidence extraction, and traceable reports</p>
  <p><a href="README.md">简体中文</a> · English</p>
  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
    <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/Runtime-LangGraph-1C3C3C" alt="LangGraph"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2F80ED" alt="Apache 2.0"></a>
  </p>
</div>

## Overview

Academic Research Copilot turns a research question into an observable, reviewable workflow: classify intent, plan work, retrieve scholarly sources, extract source-bound Evidence Cards, check citation consistency, and generate a cited research report. It also provides multi-turn sessions, progressive SSE events, run history, and optional local-paper retrieval.

This is more than a “retrieve chunks and call an LLM” demo. The source, evidence, citation, and report stages have explicit contracts. The LLM handles semantic decisions and prose generation; deterministic code owns source identifiers, tool allowlists, budgets, timeouts, citation checks, and terminal status.

> The project is intended for local research workflows, teaching, and extension. It is a single-process, single-instance application by default; it does not guarantee factual correctness and does not include authentication, multi-tenancy, or a production SLA.

## What it provides

- **Three request paths**: direct paper search/recommendation/graph lookup, contextual conversation, and full research reports.
- **Four Agent responsibilities**: Controller routes intent, Planner builds a task DAG, Worker executes isolated tools, and Reviewer performs quality checks with bounded repair.
- **Two execution modes**: Rule Mode runs deterministic offline fixtures; LLM Mode connects to a DeepSeek-compatible endpoint while preserving programmatic safety gates.
- **Traceable evidence**: PaperSource → EvidenceCard → CitationCheckResult → report sections, with stable source_id binding.
- **Bounded parallelism**: LangGraph StateGraph + Send fan-out/fan-in for search, reading, and chapter tasks.
- **Multiple providers**: Mock, the built-in academic provider, OpenAlex, Semantic Scholar, and optional Zotero + BGE-M3 + BM25 + Qdrant retrieval.
- **Operational visibility**: background runs, progressive SSE events, cancellation, warnings, retry/fallback, and completed-run SQLite snapshots.
- **MCP integration**: external MCP servers can be enabled through explicit configuration and tool allowlists; disabled by default.


### Full research lifecycle

1. The Controller chooses direct_tool, conversation, or full_research from the request and Session.
2. The Planner produces a dependency-aware WorkPlan.
3. Workers execute inside role-specific tool allowlists; independent WorkItems fan out through Send.
4. Reducers deduplicate results and create stable PaperSource records.
5. Evidence extraction only creates cards from traceable text.
6. Citation Check validates citation IDs, source_id, URLs, and quoted text.
7. The Reviewer chooses pass, one bounded repair, one bounded re-plan, or failure.
8. The API streams progress over SSE and persists completed-run snapshots for later queries.

## Rule Mode and LLM Mode

| Mode | Use case | Dependencies |
| --- | --- | --- |
| Rule | Offline tests, deterministic regression, demos without API keys | No network when SEARCH_PROVIDER=mock |
| LLM | Complex intent, dynamic planning, controlled tool loops, and natural-language writing | DEEPSEEK_API_KEY or a compatible endpoint |

backend=graph_send is the default LangGraph parallel backend; backend=loop remains available for compatibility and comparison. Tool permissions, call budgets, source deduplication, and citation checks remain program-controlled in both modes.

## Quick start

### Local development

Python 3.12 or newer is recommended:

~~~bash
git clone <your-repository-url>
cd academic_research_copilot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
~~~

The example configuration uses Rule Mode and the Mock Provider and contains no real credentials. Open:

- Dashboard: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

### Start a research run

~~~bash
curl -X POST "http://127.0.0.1:8000/api/research/runs?backend=graph_send" \
  -H "Content-Type: application/json" \
  -d '{"topic":"large language model agent evaluation","language":"en","max_sources":5,"mode":"quick","agent_mode":"rule"}'
~~~

The endpoint returns HTTP 202 and a run_id. Replace the placeholder below:

~~~bash
curl -N "http://127.0.0.1:8000/api/research/stream/<run_id>"
curl "http://127.0.0.1:8000/api/runs/<run_id>"
~~~

### Docker Compose

~~~bash
cp .env.example .env
docker compose up --build
~~~

Compose starts FastAPI and an optional Qdrant service. Zotero, model caches, and Qdrant collections are only needed when local RAG is enabled. The host Zotero directory is mounted read-only through ZOTERO_STORAGE_PATH; databases and model weights are not source files.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health | Service status and version |
| GET | /api/mcp/tools | MCP status and public tool names |
| POST | /api/sessions | Create a multi-turn session |
| GET / DELETE | /api/sessions/{session_id} | Read or delete a session |
| POST | /api/research/runs?backend=graph_send | Create an asynchronous run |
| POST | /api/research/runs/{run_id}/cancel | Cancel a running task |
| GET | /api/research/stream/{run_id} | Subscribe to SSE events |
| GET | /api/runs/{run_id} | Read an active run or snapshot |
| GET | /api/runs | Paginated run history |
| GET | /api/library | List the paper workspace |
| GET | /api/papers/detail?query=... | Resolve paper details |
| GET | /api/evidence-library | List historical evidence cards |
| GET | /api/reports | List persisted reports |

See Swagger UI or app/api/schemas.py for complete request and response schemas.

## Configuration

Secrets are read from environment variables only. Copy .env.example and adjust what you need:

| Variable | Description |
| --- | --- |
| AGENT_MODE | rule or llm |
| SEARCH_PROVIDER | mock, academic, or openalex |
| DEEPSEEK_API_KEY | Optional local-only key for LLM Mode |
| DEEPSEEK_BASE_URL / DEEPSEEK_MODEL | Compatible endpoint and model |
| OPENALEX_API_KEY | Optional OpenAlex key |
| SEMANTIC_SCHOLAR_API_KEY | Optional Semantic Scholar key |
| ZOTERO_STORAGE_PATH | Read-only local Zotero storage |
| LOCAL_RAG_ENABLED | Enable local-paper retrieval |
| QDRANT_URL / QDRANT_API_KEY | Qdrant endpoint and optional key |
| SQLITE_DB_PATH | Completed-run snapshot path |
| MCP_ENABLED / MCP_CONFIG_PATH | Enable MCP and select its config |

Never commit .env, mcp_servers.json, databases, model weights, paper PDFs, .memory/, .task_outputs/, .transcripts/, or raw JSONL evaluation output.

## Local-paper retrieval

Local RAG is optional. The data path is Zotero PDF → text-quality checks → page-aware chunks → BGE-M3 embeddings + BM25 → Qdrant → an allowlisted retrieval tool. The indexer does not modify the Zotero directory; use a new collection as a staging target when changing schemas.

Start with a read-only discovery:

~~~bash
ZOTERO_STORAGE_PATH=/path/to/zotero/storage \
python scripts/index_zotero_papers.py --discover-only
~~~

See [docs/本地论文数据.md](docs/本地论文数据.md) for indexing and migration boundaries.

## Tests and benchmarks

The test suite forces Rule Mode, the Mock Provider, and offline settings through tests/conftest.py:

~~~bash
python -m pytest tests -m "not openalex_live and not semantic_scholar_live" -q
~~~

Live smoke tests are opt-in and require user-provided credentials:

~~~bash
RUN_OPENALEX_LIVE=true python -m pytest -m openalex_live -v
RUN_SEMANTIC_SCHOLAR_LIVE=true python -m pytest -m semantic_scholar_live -v
~~~

The Agent Harness and scheduling benchmark use deterministic fixtures:

~~~bash
PYTHONPATH=. python -m harness.run_suite
python benchmarks/serial_vs_send_benchmark.py --limit 3
~~~

Generated outputs are ignored by Git. Benchmark samples, synthetic delays, and provider behavior are not production performance claims; see [benchmarks/README.md](benchmarks/README.md).

## Repository layout

~~~text
app/
├── agents/          Controller, Planner, Worker, Reviewer, and conversation agents
├── api/             FastAPI routes and Pydantic schemas
├── graph/           LangGraph StateGraph, Send, and conditional routing
├── llm/             LLM client, protocols, and prompt templates
├── mcp/             MCP lifecycle, adapters, and security configuration
├── retrieval/       Zotero, PDF, BM25, embeddings, and Qdrant
├── services/        Sessions, runs, memory, SSE, and workspace queries
├── storage/         Completed-run repositories
└── web/             Framework-free static dashboard
benchmarks/          Reproducible offline benchmark code
docs/                Architecture, deployment, and local-data notes
harness/             Fault injection and deterministic acceptance tools
scripts/             Indexing and live-smoke commands
tests/               Unit, contract, and integration tests
~~~

## Security and boundaries

- The LLM does not assign source_id values, decide whether a quote exists, or bypass tool allowlists.
- Citation checks validate internal consistency; they are not independent fact verification.
- MCP is disabled by default and still requires explicit server and tool allowlists.
- Local papers, full text, memory, databases, and model caches are user data and must stay outside the repository or be ignored.
- If you suspect a credential leak, revoke and rotate it immediately, then follow [SECURITY.md](SECURITY.md).

## Roadmap

- Add authentication, tenant isolation, and an external task queue for longer-lived service deployments.
- Add optional external database and event-store adapters.
- Add more public scholarly providers and pluggable evaluation protocols.
- Improve report formats, human-confirmation steps, and local indexing tools.

The roadmap is directional and does not promise release dates or features.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md) first. Run the offline tests before submitting changes, and make sure no credentials, paper full text, databases, or generated runtime artifacts are included.

## License

This project is released under the [Apache License 2.0](LICENSE). External data sources, models, and APIs remain subject to their own terms and licenses.
