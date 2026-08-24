"""本地论文检索使用的向量模型抽象与 BGE-M3 实现。"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EmbeddingProvider(ABC):
    """隔离上层检索逻辑与具体向量模型。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """编码单条检索查询。"""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """批量编码待索引文档。"""

    def embed(self, text: str) -> List[float]:
        """兼容旧调用方，单条文本默认按查询编码。"""
        return self.embed_query(text)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """兼容旧调用方，批量文本默认按文档编码。"""
        return self.embed_documents(texts)


_MODEL_CACHE: Dict[Tuple[str, str, str, str, bool], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _load_sentence_transformer(
    *,
    model_name: str,
    device: str,
    cache_folder: str,
    revision: str,
    local_files_only: bool,
) -> Any:
    """按进程复用模型，避免每次 Tool 调用重复加载数 GB 权重。"""
    cache_key = (
        model_name,
        device,
        cache_folder,
        revision,
        local_files_only,
    )
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers，无法加载 BGE-M3；"
                "请先安装 requirements.txt 中的生产依赖"
            ) from exc

        kwargs: Dict[str, Any] = {
            "model_name_or_path": model_name,
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        if device and device != "auto":
            kwargs["device"] = device
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        if revision and not Path(model_name).is_dir():
            kwargs["revision"] = revision

        model = SentenceTransformer(**kwargs)
        _MODEL_CACHE[cache_key] = model
        return model


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """使用 BAAI/bge-m3 生成归一化稠密向量。"""

    DEFAULT_MODEL_NAME = "BAAI/bge-m3"
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 8192,
        normalize_embeddings: bool = True,
        cache_folder: str = "",
        revision: str = "",
        local_files_only: bool = False,
    ):
        if not model_name.strip():
            raise ValueError("BGE-M3 模型名称不能为空")
        if batch_size <= 0:
            raise ValueError("BGE-M3 batch_size 必须大于 0")
        if max_length <= 0:
            raise ValueError("BGE-M3 max_length 必须大于 0")

        raw_model_name = model_name.strip()
        local_model_path = Path(raw_model_name).expanduser()
        if local_model_path.is_dir():
            self.model_name = str(local_model_path.resolve())
        else:
            looks_like_path = (
                raw_model_name.startswith(("/", "./", "../", "~"))
                or "\\" in raw_model_name
            )
            if local_files_only and looks_like_path:
                raise ValueError(
                    f"BGE-M3 本地模型目录不存在：{local_model_path}"
                )
            self.model_name = raw_model_name
        self.device = device.strip().lower() or "auto"
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = cache_folder.strip()
        self.revision = revision.strip()
        self.local_files_only = local_files_only
        self._model: Optional[Any] = None

    @property
    def dimension(self) -> int:
        return self.DEFAULT_DIMENSION

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self.dimension
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        normalized_texts = [str(text).strip() for text in texts]
        if not normalized_texts:
            return []

        model = self._get_model()
        with _INFERENCE_LOCK:
            vectors = model.encode(
                normalized_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

        result = [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]
        for vector in result:
            if len(vector) != self.dimension:
                raise RuntimeError(
                    f"BGE-M3 返回了 {len(vector)} 维向量，"
                    f"但 Qdrant collection 需要 {self.dimension} 维"
                )
        return result

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _load_sentence_transformer(
                model_name=self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            current_max_length = getattr(self._model, "max_seq_length", None)
            if current_max_length is None:
                self._model.max_seq_length = self.max_length
            else:
                self._model.max_seq_length = min(
                    int(current_max_length),
                    self.max_length,
                )

            dimension_getter = getattr(
                self._model,
                "get_embedding_dimension",
                None,
            )
            if dimension_getter is None:
                dimension_getter = (
                    self._model.get_sentence_embedding_dimension
                )
            model_dimension = dimension_getter()
            if model_dimension and int(model_dimension) != self.dimension:
                raise RuntimeError(
                    f"模型 {self.model_name} 的向量维度为 {model_dimension}，"
                    f"预期为 {self.dimension}"
                )
        return self._model
