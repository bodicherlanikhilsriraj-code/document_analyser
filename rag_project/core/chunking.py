"""Deterministic, page-aware text chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .documents import ExtractedPage

MAX_CHUNKS_PER_DOCUMENT = 10_000


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    page_number: int | None
    start_offset: int
    text: str


def split_pages(
    document_id: str,
    pages: list[ExtractedPage | None] | None,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[TextChunk]:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id cannot be empty.")
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size.")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for page in pages or []:
        if page is None:
            continue
        text = (page.text or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(
                    f"The document exceeds the {MAX_CHUNKS_PER_DOCUMENT:,}-chunk limit. "
                    "Increase the chunk size or use a smaller file."
                )

            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                identity = (
                    f"{document_id}:{page.page_number}:{start}:{piece}".encode("utf-8")
                )
                chunks.append(
                    TextChunk(
                        chunk_id=hashlib.sha256(identity).hexdigest(),
                        document_id=document_id,
                        page_number=page.page_number,
                        start_offset=start,
                        text=piece,
                    )
                )

            if end == len(text):
                break
            start += step
    return chunks
