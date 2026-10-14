"""Persistent local vector storage and source-aware retrieval."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from .chunking import TextChunk
from .config import Settings
from .documents import document_hash, read_stored_document
from .embeddings import embed_texts
from .errors import DocumentError, RetrievalError, StudyPilotError

logger = logging.getLogger(__name__)
MAX_DOCUMENT_CHUNKS = 10_000
VECTOR_BATCH_SIZE = 64


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    file_hash: str
    stored_path: str
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class Passage:
    source_id: str
    document_id: str
    filename: str
    page_number: int | None
    text: str
    relevance: float


class DocumentStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_sqlite()
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            self._collection = self._client.get_or_create_collection(
                "studypilot_documents",
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": settings.embedding_model,
                },
            )
            collection_model = (self._collection.metadata or {}).get("embedding_model")
            if collection_model and collection_model != settings.embedding_model:
                raise RetrievalError(
                    "The embedding model setting differs from the one used by the existing "
                    "index. Restore the original EMBEDDING_MODEL or remove the local index "
                    "from Settings before changing models."
                )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                "The local vector database could not be opened. Check the data directory "
                "permissions and ChromaDB installation."
            ) from exc

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.settings.metadata_db, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_sqlite(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    stored_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )

    def list_documents(self) -> list[DocumentRecord]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC"
                ).fetchall()
            return [DocumentRecord(**dict(row)) for row in rows]
        except sqlite3.Error as exc:
            raise RetrievalError("The document library could not be loaded.") from exc

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def index_document(
        self, filename: str, content: bytes, document_id: str, chunks: list[TextChunk]
    ) -> tuple[DocumentRecord, bool]:
        with self._lock:
            return self._index_document(filename, content, document_id, chunks)

    def _index_document(
        self, filename: str, content: bytes, document_id: str, chunks: list[TextChunk]
    ) -> tuple[DocumentRecord, bool]:
        digest = document_hash(content)
        safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not safe_filename or document_id != digest:
            raise DocumentError("The document name or identifier is invalid.")
        if Path(safe_filename).suffix.lower() not in {".pdf", ".docx", ".txt"}:
            raise DocumentError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")
        duplicate = self.find_by_hash(digest)
        if duplicate:
            return duplicate, False
        if not chunks:
            raise DocumentError("This file did not produce any indexable text.")
        if len(chunks) > MAX_DOCUMENT_CHUNKS:
            raise DocumentError(
                f"This file creates more than {MAX_DOCUMENT_CHUNKS:,} chunks. Use a larger "
                "chunk size or a smaller file before indexing."
            )
        if any(chunk.document_id != document_id for chunk in chunks):
            raise DocumentError("The generated chunks do not belong to this document.")

        suffix = Path(safe_filename).suffix.lower()
        stored_path = self.settings.upload_dir / f"{document_id}{suffix}"
        try:
            stored_path.write_bytes(content)
            self._collection.delete(where={"document_id": document_id})
            for offset in range(0, len(chunks), VECTOR_BATCH_SIZE):
                batch = chunks[offset : offset + VECTOR_BATCH_SIZE]
                vectors = embed_texts(
                    [chunk.text for chunk in batch], self.settings.embedding_model
                )
                self._collection.add(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    embeddings=vectors,
                    metadatas=[
                        {
                            "document_id": document_id,
                            "filename": safe_filename,
                            "page_number": chunk.page_number or 0,
                            "start_offset": chunk.start_offset,
                        }
                        for chunk in batch
                    ],
                )
            created = datetime.now(timezone.utc).isoformat()
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO documents
                       (document_id, filename, file_hash, stored_path, chunk_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        safe_filename,
                        digest,
                        str(stored_path),
                        len(chunks),
                        created,
                    ),
                )
            return (
                DocumentRecord(
                    document_id, safe_filename, digest, str(stored_path), len(chunks), created
                ),
                True,
            )
        except Exception as exc:
            try:
                self._collection.delete(where={"document_id": document_id})
            except Exception:
                logger.warning("Could not remove partial vector entries for document %s.", document_id)
            try:
                if stored_path.exists():
                    stored_path.unlink()
            except OSError:
                logger.warning("Could not remove partial upload for document %s.", document_id)
            if isinstance(exc, StudyPilotError):
                raise
            raise RetrievalError("The document could not be saved to the local index.") from exc

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._delete_document(document_id)

    def _delete_document(self, document_id: str) -> None:
        record = next(
            (item for item in self.list_documents() if item.document_id == document_id), None
        )
        if record is None:
            raise DocumentError("That document is no longer in the library.")
        try:
            self._collection.delete(where={"document_id": document_id})
            path = Path(record.stored_path)
            if path.exists() and path.parent.resolve() == self.settings.upload_dir.resolve():
                path.unlink()
            with self._connect() as connection:
                connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        except Exception as exc:
            raise RetrievalError(
                "The document could not be fully removed from local storage."
            ) from exc

    def reindex_document(
        self,
        document_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentRecord:
        with self._lock:
            return self._reindex_document(document_id, chunk_size, chunk_overlap)

    def _reindex_document(
        self,
        document_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentRecord:
        record = next(
            (item for item in self.list_documents() if item.document_id == document_id), None
        )
        if record is None:
            raise DocumentError("That document is no longer in the library.")
        from .chunking import split_pages
        from .documents import extract_document

        source_path = Path(record.stored_path)
        content = read_stored_document(
            source_path,
            self.settings.upload_dir,
            self.settings.max_upload_mb * 1024 * 1024,
        )
        pages = extract_document(
            record.filename, content, self.settings.max_upload_mb * 1024 * 1024
        )
        chunks = split_pages(
            document_id,
            pages,
            self.settings.chunk_size if chunk_size is None else chunk_size,
            self.settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        )
        if len(chunks) > MAX_DOCUMENT_CHUNKS:
            raise DocumentError(
                f"This file creates more than {MAX_DOCUMENT_CHUNKS:,} chunks. Increase "
                "chunk size before re-indexing."
            )
        try:
            existing = self._collection.get(
                where={"document_id": document_id}, include=["metadatas"]
            )
            existing_ids = set(existing.get("ids") or [])
            new_ids = {chunk.chunk_id for chunk in chunks}
            batches: list[tuple[list[TextChunk], list[list[float]]]] = []
            for offset in range(0, len(chunks), VECTOR_BATCH_SIZE):
                batch = chunks[offset : offset + VECTOR_BATCH_SIZE]
                vectors = embed_texts(
                    [chunk.text for chunk in batch], self.settings.embedding_model
                )
                batches.append((batch, vectors))

            try:
                for batch, vectors in batches:
                    self._collection.upsert(
                        ids=[chunk.chunk_id for chunk in batch],
                        documents=[chunk.text for chunk in batch],
                        embeddings=vectors,
                        metadatas=[
                            {
                                "document_id": document_id,
                                "filename": record.filename,
                                "page_number": chunk.page_number or 0,
                                "start_offset": chunk.start_offset,
                            }
                            for chunk in batch
                        ],
                    )
                stale_ids = list(existing_ids - new_ids)
                for offset in range(0, len(stale_ids), VECTOR_BATCH_SIZE):
                    self._collection.delete(
                        ids=stale_ids[offset : offset + VECTOR_BATCH_SIZE]
                    )
            except Exception:
                new_only_ids = new_ids - existing_ids
                if new_only_ids:
                    try:
                        self._collection.delete(ids=list(new_only_ids))
                    except Exception:
                        logger.warning(
                            "Could not remove partial replacement vectors for document %s.",
                            document_id,
                        )
                raise

            with self._connect() as connection:
                connection.execute(
                    "UPDATE documents SET chunk_count = ? WHERE document_id = ?",
                    (len(chunks), document_id),
                )
            return DocumentRecord(
                record.document_id,
                record.filename,
                record.file_hash,
                record.stored_path,
                len(chunks),
                record.created_at,
            )
        except Exception as exc:
            if isinstance(exc, StudyPilotError):
                raise
            raise RetrievalError(
                "Re-indexing failed. The document remains in the library; retry the operation."
            ) from exc

    def chunk_count(self) -> int:
        with self._lock:
            return self._chunk_count()

    def _chunk_count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise RetrievalError("The vector index could not be counted.") from exc

    def search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        count: int | None = None,
        threshold: float | None = None,
    ) -> list[Passage]:
        with self._lock:
            return self._search(
                query,
                document_ids=document_ids,
                count=count,
                threshold=threshold,
            )

    def _search(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        count: int | None = None,
        threshold: float | None = None,
    ) -> list[Passage]:
        if not query.strip():
            raise DocumentError("Enter a question or topic to search your notes.")
        if len(query) > 2000:
            raise DocumentError("Search text must be 2,000 characters or fewer.")
        if count is not None and not 1 <= count <= 20:
            raise DocumentError("Choose between 1 and 20 retrieved passages.")
        if threshold is not None and not -1 <= threshold <= 1:
            raise DocumentError("Relevance threshold must be between -1 and 1.")
        if document_ids is not None and not document_ids:
            return []
        total = self.chunk_count()
        if total == 0:
            return []
        query_vector = embed_texts([query], self.settings.embedding_model)[0]
        where = {"document_id": {"$in": document_ids}} if document_ids is not None else None
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(
                    self.settings.retrieval_count if count is None else count, total
                ),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            passages: list[Passage] = []
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            min_score = (
                self.settings.relevance_threshold if threshold is None else threshold
            )
            for index, (text, metadata, distance) in enumerate(
                zip(documents, metadatas, distances)
            ):
                relevance = 1 - float(distance)
                if relevance < min_score:
                    continue
                passages.append(
                    Passage(
                        source_id=f"S{index + 1}",
                        document_id=str(metadata["document_id"]),
                        filename=str(metadata["filename"]),
                        page_number=int(metadata.get("page_number", 0)) or None,
                        text=str(text),
                        relevance=relevance,
                    )
                )
            return passages
        except DocumentError:
            raise
        except Exception as exc:
            raise RetrievalError(
                "The local search failed. The embedding model and vector index may need "
                "to be reinstalled or re-indexed."
            ) from exc
