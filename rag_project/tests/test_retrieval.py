import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.chunking import TextChunk
from core.config import Settings
from core.errors import DocumentError, RetrievalError
from core.documents import document_hash
from core.retrieval import DocumentStore


@unittest.skipUnless(importlib.util.find_spec("chromadb"), "ChromaDB is not installed")
class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = Settings(data_dir=Path(self.temporary_directory.name), embedding_model="test")
        self.store = DocumentStore(self.settings)
        self.embedding_patch = patch(
            "core.retrieval.embed_texts",
            side_effect=lambda texts, model: [[1.0, 0.0, 0.0] for _ in texts],
        )
        self.embedding_patch.start()
        self.content = b"local study text"
        self.document_id = document_hash(self.content)
        self.chunks = [
            TextChunk("chunk-one", self.document_id, 1, 0, "local study passage")
        ]

    def tearDown(self):
        self.embedding_patch.stop()
        self.store._client.close()
        self.temporary_directory.cleanup()

    def test_add_search_duplicate_and_delete_use_isolated_database(self):
        self.assertEqual(self.store.search("local study"), [])
        record, created = self.store.index_document(
            "notes.txt", self.content, self.document_id, self.chunks
        )
        self.assertTrue(created)
        self.assertEqual(record.chunk_count, 1)
        results = self.store.search("local study", threshold=0)
        self.assertEqual(results[0].filename, "notes.txt")
        duplicate, was_created = self.store.index_document(
            "renamed.txt", self.content, self.document_id, self.chunks
        )
        self.assertFalse(was_created)
        self.assertEqual(duplicate.document_id, self.document_id)
        self.store.delete_document(self.document_id)
        self.assertEqual(self.store.chunk_count(), 0)
        self.assertEqual(self.store.search("local study"), [])

    def test_unknown_document_cannot_be_deleted(self):
        with self.assertRaises(DocumentError):
            self.store.delete_document("missing")

    def test_reindex_replaces_chunks_only_after_embeddings_succeed(self):
        self.store.index_document("notes.txt", self.content, self.document_id, self.chunks)

        with patch(
            "core.retrieval.embed_texts",
            side_effect=RuntimeError("embedding unavailable"),
        ):
            with self.assertRaisesRegex(RetrievalError, "Re-indexing failed"):
                self.store.reindex_document(self.document_id, chunk_size=5, chunk_overlap=0)

        self.assertEqual(self.store.chunk_count(), 1)
        self.assertEqual(
            [passage.text for passage in self.store.search("local study", threshold=0)],
            ["local study passage"],
        )

        updated = self.store.reindex_document(
            self.document_id, chunk_size=5, chunk_overlap=0
        )
        self.assertEqual(updated.chunk_count, 4)
        self.assertEqual(self.store.chunk_count(), 4)


if __name__ == "__main__":
    unittest.main()
