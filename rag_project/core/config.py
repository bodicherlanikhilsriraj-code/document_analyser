"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://127.0.0.1:11434"
    default_model: str = "Qwen2.5:3b"
    request_timeout: int = 180
    temperature: float = 0.2
    max_output_tokens: int = 1024
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 180
    retrieval_count: int = 5
    relevance_threshold: float = 0.15
    max_upload_mb: int = 25
    data_dir: Path = PROJECT_ROOT / "data"

    @classmethod
    def from_env(cls) -> "Settings":
        env_file = PROJECT_ROOT / ".env"
        if env_file.is_file():
            try:
                from dotenv import load_dotenv
            except ImportError as exc:
                from .errors import ConfigurationError

                raise ConfigurationError(
                    "Install the project requirements to load the local .env configuration."
                ) from exc

            load_dotenv(env_file, override=False)

        root = Path(os.getenv("STUDYPILOT_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
        try:
            timeout = _bounded_int("OLLAMA_TIMEOUT_SECONDS", 180, 5, 900)
            output_tokens = _bounded_int("OLLAMA_MAX_OUTPUT_TOKENS", 1024, 64, 8192)
            chunk_size = _bounded_int("CHUNK_SIZE", 1200, 200, 6000)
            chunk_overlap = _bounded_int("CHUNK_OVERLAP", 180, 0, 1500)
            retrieval_count = _bounded_int("RETRIEVAL_COUNT", 5, 1, 20)
            max_upload_mb = _bounded_int("MAX_UPLOAD_MB", 25, 1, 100)
            temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
            threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.15"))
        except ValueError as exc:
            from .errors import ConfigurationError

            raise ConfigurationError(str(exc)) from exc

        if not 0 <= temperature <= 2:
            from .errors import ConfigurationError

            raise ConfigurationError("OLLAMA_TEMPERATURE must be between 0 and 2.")
        if not -1 <= threshold <= 1:
            from .errors import ConfigurationError

            raise ConfigurationError("RELEVANCE_THRESHOLD must be between -1 and 1.")
        if chunk_overlap >= chunk_size:
            from .errors import ConfigurationError

            raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        return cls(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            default_model=os.getenv("OLLAMA_MODEL", "Qwen2.5:3b"),
            request_timeout=timeout,
            temperature=temperature,
            max_output_tokens=output_tokens,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_count=retrieval_count,
            relevance_threshold=threshold,
            max_upload_mb=max_upload_mb,
            data_dir=root.resolve(),
        )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def metadata_db(self) -> Path:
        return self.data_dir / "documents.sqlite3"
