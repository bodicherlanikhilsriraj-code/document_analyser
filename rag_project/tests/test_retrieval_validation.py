import threading
import unittest

from core.errors import DocumentError
from core.retrieval import DocumentStore


class RetrievalValidationTests(unittest.TestCase):
    def setUp(self):
        self.store = object.__new__(DocumentStore)
        self.store._lock = threading.RLock()

    def test_empty_document_selection_does_not_search_all_documents(self):
        self.assertEqual(self.store.search("photosynthesis", document_ids=[]), [])

    def test_rejects_empty_and_overlong_queries_without_database_access(self):
        with self.assertRaises(DocumentError):
            self.store.search(" ")
        with self.assertRaises(DocumentError):
            self.store.search("x" * 2001)


if __name__ == "__main__":
    unittest.main()
