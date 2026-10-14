"""Lazy local sentence-transformers embeddings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .errors import EmbeddingError


def _cached_model_path(model_name: str) -> tuple[str, bool]:
    if Path(model_name).exists():
        return model_name, True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached_modules = try_to_load_from_cache(model_name, "modules.json")
    except ImportError:
        return model_name, False
    if isinstance(cached_modules, str):
        return str(Path(cached_modules).parent), True
    return model_name, False


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer

        model_path, local_files_only = _cached_model_path(model_name)
        return SentenceTransformer(
            model_path, device="cpu", local_files_only=local_files_only
        )
    except Exception as exc:
        raise EmbeddingError(
            "The local embedding model could not be loaded. Install the requirements and "
            "allow the first-run model download, then retry."
        ) from exc


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts or any(not text.strip() for text in texts):
        raise EmbeddingError("Cannot embed an empty text batch.")
    try:
        vectors = _load_model(model_name).encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("The local embedding model failed while encoding text.") from exc
