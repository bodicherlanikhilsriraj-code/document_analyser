import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.errors import DocumentError
from core.retrieval import DocumentRecord, Passage
from core.study_tools import summarize_document


class FakeStore:
    def __init__(self, record, settings, passages=None):
        self.record = record
        self.settings = settings
        self.passages = passages or []
        self.search_calls = []

    def list_documents(self):
        return [self.record]

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return self.passages


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, model, prompt, **kwargs):
        self.prompts.append(prompt)
        return f"summary-{len(self.prompts)}"


class StudyToolTests(unittest.TestCase):
    def test_full_document_summary_uses_hierarchical_batches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_directory = Path(temporary_directory) / "data"
            settings = Settings(
                data_dir=data_directory,
                chunk_size=40,
                chunk_overlap=5,
            )
            settings.upload_dir.mkdir(parents=True)
            source_path = settings.upload_dir / "notes.txt"
            source_path.write_text(
                "A useful study fact. " * 18,
                encoding="utf-8",
            )
            record = DocumentRecord(
                "doc-id", "notes.txt", "hash", str(source_path), 1, "2026-10-09"
            )
            llm = FakeLLM()
            summary = summarize_document(
                "doc-id", FakeStore(record, settings), llm, "model"
            )
        self.assertTrue(summary.startswith("summary-"))
        self.assertGreater(len(llm.prompts), 2)
        self.assertIn("Exam revision points", llm.prompts[-1])

    def test_topic_summary_uses_only_retrieved_passages(self):
        passage = Passage(
            "S1",
            "doc-id",
            "notes.txt",
            4,
            "Mitosis separates duplicated chromosomes.",
            0.8,
        )
        record = DocumentRecord("doc-id", "notes.txt", "hash", "unused", 1, "date")
        store = FakeStore(record, Settings(), [passage])
        llm = FakeLLM()
        summarize_document(
            "doc-id",
            store,
            llm,
            "model",
            topic="mitosis",
            retrieval_count=3,
            relevance_threshold=0.2,
        )
        self.assertEqual(store.search_calls[0][0], "mitosis")
        self.assertEqual(store.search_calls[0][1]["document_ids"], ["doc-id"])
        self.assertIn('"page_number": 4', llm.prompts[0])

    def test_topic_without_retrieved_evidence_is_rejected(self):
        record = DocumentRecord("doc-id", "notes.txt", "hash", "unused", 1, "date")
        with self.assertRaises(DocumentError):
            summarize_document(
                "doc-id", FakeStore(record, Settings()), FakeLLM(), "model", topic="missing topic"
            )


if __name__ == "__main__":
    unittest.main()
