import unittest

from core.rag import INSUFFICIENT_EVIDENCE, answer_question
from core.retrieval import Passage


class FakeStore:
    def __init__(self, passages):
        self.passages = passages

    def search(self, query, **kwargs):
        return self.passages


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return self.response


class RAGTests(unittest.TestCase):
    def test_empty_retrieval_returns_insufficient_evidence_without_generation(self):
        llm = FakeLLM("should not run")
        result = answer_question("What is photosynthesis?", FakeStore([]), llm, "model")
        self.assertEqual(result.answer, INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.grounded)
        self.assertEqual(llm.calls, 0)

    def test_citations_are_limited_to_passages_actually_retrieved(self):
        passage = Passage("S1", "doc", "biology.txt", 2, "Chlorophyll absorbs light.", 0.8)
        result = answer_question(
            "What absorbs light?",
            FakeStore([passage]),
            FakeLLM("Chlorophyll absorbs light [S1]. Unsupported claim [S99]."),
            "model",
        )
        self.assertIn("[S1]", result.answer)
        self.assertNotIn("[S99]", result.answer)
        self.assertEqual(result.sources, (passage,))
        self.assertTrue(result.grounded)

    def test_adds_source_citation_when_model_omits_one(self):
        passage = Passage(
            "S1",
            "doc",
            "rag-notes.txt",
            None,
            "RAG stands for Retrieval-Augmented Generation.",
            0.8,
        )
        result = answer_question(
            "What does RAG stand for?",
            FakeStore([passage]),
            FakeLLM("RAG stands for Retrieval-Augmented Generation."),
            "model",
        )
        self.assertTrue(result.answer.endswith("[S1]"))


if __name__ == "__main__":
    unittest.main()
