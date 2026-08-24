"""
app/core/config.py

应用配置 —— 类比 Spring Boot 的 application.yml + @ConfigurationProperties。

所有 provider 选择通过环境变量控制，改 .env 即可切换，不写死在代码里。
"""

import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """应用全局配置。"""

    APP_NAME: str = "Academic Research Copilot"
    APP_VERSION: str = "1.3.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ---- LLM 配置 ----
    LLM_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ---- 搜索 Provider ----
    # 可选值: "mock" | "academic" | "tavily" | "openalex"
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "academic")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ---- OpenAlex Provider ----
    OPENALEX_API_KEY: str = os.getenv("OPENALEX_API_KEY", "")
    OPENALEX_BASE_URL: str = os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    OPENALEX_CONTENT_BASE_URL: str = os.getenv("OPENALEX_CONTENT_BASE_URL", "https://content.openalex.org")
    OPENALEX_TIMEOUT_SECONDS: int = int(os.getenv("OPENALEX_TIMEOUT_SECONDS", "15"))
    OPENALEX_MAX_RETRIES: int = int(os.getenv("OPENALEX_MAX_RETRIES", "2"))
    OPENALEX_FALLBACK_TO_MOCK: bool = os.getenv("OPENALEX_FALLBACK_TO_MOCK", "false").lower() == "true"
    OPENALEX_CONTENT_MODE: str = os.getenv("OPENALEX_CONTENT_MODE", "abstract")
    OPENALEX_MAX_CONTENT_FETCHES: int = int(os.getenv("OPENALEX_MAX_CONTENT_FETCHES", "3"))
    OPENALEX_MAX_TEXT_CHARS: int = int(os.getenv("OPENALEX_MAX_TEXT_CHARS", "30000"))

    # ---- Zotero Local Paper Retrieval ----
    # 重要步骤：发布默认值使用通用目录，避免绑定开发者本机路径。
    ZOTERO_STORAGE_PATH: str = os.getenv(
        "ZOTERO_STORAGE_PATH",
        "./data/zotero",
    )
    LOCAL_RAG_ENABLED: bool = (
        os.getenv("LOCAL_RAG_ENABLED", "true").lower() == "true"
    )
    # 混合检索：向量通道 + BM25 关键词通道，RRF 融合排序。
    LOCAL_RAG_HYBRID_ENABLED: bool = (
        os.getenv("LOCAL_RAG_HYBRID_ENABLED", "true").lower() == "true"
    )
    LOCAL_RAG_EMBEDDING_PROVIDER: str = os.getenv(
        "LOCAL_RAG_EMBEDDING_PROVIDER",
        "bge-m3",
    )
    BGE_M3_MODEL: str = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
    BGE_M3_DEVICE: str = os.getenv("BGE_M3_DEVICE", "auto")
    BGE_M3_BATCH_SIZE: int = int(os.getenv("BGE_M3_BATCH_SIZE", "16"))
    BGE_M3_MAX_LENGTH: int = int(os.getenv("BGE_M3_MAX_LENGTH", "8192"))
    BGE_M3_NORMALIZE_EMBEDDINGS: bool = (
        os.getenv("BGE_M3_NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    )
    BGE_M3_CACHE_DIR: str = os.getenv("BGE_M3_CACHE_DIR", "")
    BGE_M3_REVISION: str = os.getenv(
        "BGE_M3_REVISION",
        "5617a9f61b028005a4858fdac845db406aefb181",
    )
    BGE_M3_LOCAL_FILES_ONLY: bool = (
        os.getenv("BGE_M3_LOCAL_FILES_ONLY", "false").lower() == "true"
    )
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv(
        "QDRANT_COLLECTION",
        "local_papers_bge_m3_v4",
    )
    QDRANT_TIMEOUT_SECONDS: float = float(
        os.getenv("QDRANT_TIMEOUT_SECONDS", "30")
    )
    QDRANT_PREFER_GRPC: bool = (
        os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    )
    QDRANT_BATCH_SIZE: int = int(os.getenv("QDRANT_BATCH_SIZE", "64"))
    LOCAL_RAG_INDEX_SCHEMA_VERSION: str = os.getenv(
        "LOCAL_RAG_INDEX_SCHEMA_VERSION",
        "v4-section-quality-reference-boundary-late-signal",
    )
    LOCAL_RAG_CHUNK_SIZE: int = int(os.getenv("LOCAL_RAG_CHUNK_SIZE", "900"))
    LOCAL_RAG_CHUNK_OVERLAP: int = int(
        os.getenv("LOCAL_RAG_CHUNK_OVERLAP", "120")
    )
    LOCAL_RAG_MIN_SCORE: float = float(
        os.getenv("LOCAL_RAG_MIN_SCORE", "0.25")
    )

    # ---- Semantic Scholar ----
    SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    SEMANTIC_SCHOLAR_GRAPH_BASE_URL: str = os.getenv(
        "SEMANTIC_SCHOLAR_GRAPH_BASE_URL",
        "https://api.semanticscholar.org/graph/v1",
    )
    SEMANTIC_SCHOLAR_RECOMMENDATIONS_BASE_URL: str = os.getenv(
        "SEMANTIC_SCHOLAR_RECOMMENDATIONS_BASE_URL",
        "https://api.semanticscholar.org/recommendations/v1",
    )
    SEMANTIC_SCHOLAR_TIMEOUT_SECONDS: int = int(
        os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", "20")
    )
    SEMANTIC_SCHOLAR_MAX_RETRIES: int = int(
        os.getenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "2")
    )
    SEMANTIC_SCHOLAR_RATE_LIMIT_RETRIES: int = int(
        os.getenv("SEMANTIC_SCHOLAR_RATE_LIMIT_RETRIES", "6")
    )
    SEMANTIC_SCHOLAR_MAX_RETRY_WAIT_SECONDS: float = float(
        os.getenv("SEMANTIC_SCHOLAR_MAX_RETRY_WAIT_SECONDS", "45")
    )

# 全局单例
config = AppConfig()


# ---- OpenAlex 配置函数（支持测试替换） ----

def get_search_provider() -> str:
    """返回当前搜索 provider（支持测试 monkeypatch）。"""
    return os.getenv("SEARCH_PROVIDER", "academic")


def get_openalex_api_key() -> str:
    """返回 OpenAlex API key（仅从环境变量读取，不允许 LLM args 传入）。"""
    return os.getenv("OPENALEX_API_KEY", "")


def get_openalex_base_url() -> str:
    return os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org")


def get_openalex_content_base_url() -> str:
    return os.getenv("OPENALEX_CONTENT_BASE_URL", "https://content.openalex.org")


def get_openalex_timeout() -> int:
    return int(os.getenv("OPENALEX_TIMEOUT_SECONDS", "15"))


def get_openalex_max_retries() -> int:
    return int(os.getenv("OPENALEX_MAX_RETRIES", "2"))


def get_openalex_fallback_to_mock() -> bool:
    return os.getenv("OPENALEX_FALLBACK_TO_MOCK", "false").lower() == "true"


def get_openalex_content_mode() -> str:
    return os.getenv("OPENALEX_CONTENT_MODE", "abstract")


def get_openalex_max_content_fetches() -> int:
    return int(os.getenv("OPENALEX_MAX_CONTENT_FETCHES", "3"))


def get_openalex_max_text_chars() -> int:
    return int(os.getenv("OPENALEX_MAX_TEXT_CHARS", "30000"))


def get_research_latency_ttl_ms() -> int:
    """
    返回一次完整深度调研的质量延迟窗口。

    这是 Evaluator 的端到端延迟阈值，不会改变任何单工具 timeout。
    默认 180 秒，为并行 Search/Reading 加长报告 Reviewer 留出合理窗口。
    """
    try:
        seconds = float(os.getenv("RESEARCH_LATENCY_TTL_SECONDS", "180"))
    except (TypeError, ValueError):
        seconds = 180.0
    return max(1_000, int(seconds * 1_000))


# ---- Zotero Local Paper Retrieval 配置函数 ----

def get_zotero_storage_path() -> str:
    return os.getenv(
        "ZOTERO_STORAGE_PATH",
        "./data/zotero",
    )


def get_local_rag_enabled() -> bool:
    return os.getenv("LOCAL_RAG_ENABLED", "true").lower() == "true"


def get_local_rag_hybrid_enabled() -> bool:
    return os.getenv("LOCAL_RAG_HYBRID_ENABLED", "true").lower() == "true"


def get_local_rag_embedding_provider() -> str:
    return os.getenv("LOCAL_RAG_EMBEDDING_PROVIDER", "bge-m3")


def get_bge_m3_model() -> str:
    return os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")


def get_bge_m3_device() -> str:
    return os.getenv("BGE_M3_DEVICE", "auto")


def get_bge_m3_batch_size() -> int:
    return max(1, int(os.getenv("BGE_M3_BATCH_SIZE", "16")))


def get_bge_m3_max_length() -> int:
    return max(128, int(os.getenv("BGE_M3_MAX_LENGTH", "8192")))


def get_bge_m3_normalize_embeddings() -> bool:
    return os.getenv("BGE_M3_NORMALIZE_EMBEDDINGS", "true").lower() == "true"


def get_bge_m3_cache_dir() -> str:
    return os.getenv("BGE_M3_CACHE_DIR", "")


def get_bge_m3_revision() -> str:
    return os.getenv(
        "BGE_M3_REVISION",
        "5617a9f61b028005a4858fdac845db406aefb181",
    )


def get_bge_m3_local_files_only() -> bool:
    return os.getenv("BGE_M3_LOCAL_FILES_ONLY", "false").lower() == "true"


def get_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")


def get_qdrant_api_key() -> str:
    """Qdrant API key 只从环境变量读取，不进入 Tool 参数或 Trace。"""
    return os.getenv("QDRANT_API_KEY", "")


def get_qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION", "local_papers_bge_m3_v4")


def get_qdrant_timeout() -> float:
    return max(1.0, float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30")))


def get_qdrant_prefer_grpc() -> bool:
    return os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"


def get_qdrant_batch_size() -> int:
    return max(1, int(os.getenv("QDRANT_BATCH_SIZE", "64")))


def get_local_rag_index_schema_version() -> str:
    return os.getenv(
        "LOCAL_RAG_INDEX_SCHEMA_VERSION",
        "v4-section-quality-reference-boundary-late-signal",
    )


def get_local_rag_chunk_size() -> int:
    return max(100, int(os.getenv("LOCAL_RAG_CHUNK_SIZE", "900")))


def get_local_rag_chunk_overlap() -> int:
    chunk_size = get_local_rag_chunk_size()
    return max(
        0,
        min(
            int(os.getenv("LOCAL_RAG_CHUNK_OVERLAP", "120")),
            chunk_size - 1,
        ),
    )


def get_local_rag_min_score() -> float:
    return max(0.0, min(float(os.getenv("LOCAL_RAG_MIN_SCORE", "0.25")), 1.0))


# ---- Semantic Scholar 配置函数（支持测试替换） ----

def get_semantic_scholar_api_key() -> str:
    """API key 仅从环境变量读取，不进入 Tool args、Trace 或 LLM 上下文。"""
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


def get_semantic_scholar_graph_base_url() -> str:
    return os.getenv(
        "SEMANTIC_SCHOLAR_GRAPH_BASE_URL",
        "https://api.semanticscholar.org/graph/v1",
    ).rstrip("/")


def get_semantic_scholar_recommendations_base_url() -> str:
    return os.getenv(
        "SEMANTIC_SCHOLAR_RECOMMENDATIONS_BASE_URL",
        "https://api.semanticscholar.org/recommendations/v1",
    ).rstrip("/")


def get_semantic_scholar_timeout() -> int:
    return int(os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", "20"))


def get_semantic_scholar_max_retries() -> int:
    return max(0, int(os.getenv("SEMANTIC_SCHOLAR_MAX_RETRIES", "2")))


def get_semantic_scholar_rate_limit_retries() -> int:
    """429 使用独立预算，避免与短暂 5xx 共用过小的重试次数。"""
    return max(0, int(os.getenv("SEMANTIC_SCHOLAR_RATE_LIMIT_RETRIES", "6")))


def get_semantic_scholar_max_retry_wait() -> float:
    """单次 Semantic Scholar 调用允许用于退避休眠的总秒数。"""
    return max(
        0.0,
        float(os.getenv("SEMANTIC_SCHOLAR_MAX_RETRY_WAIT_SECONDS", "45")),
    )
