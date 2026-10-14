import unittest

from core.chunking import split_pages
from core.documents import ExtractedPage


class ChunkingTests(unittest.TestCase):
    def test_chunks_overlap_preserve_page_and_are_deterministic(self):
        page = ExtractedPage(3, "abcdefghijklmnopqrst")
        chunks = split_pages("doc-id", [page], chunk_size=10, overlap=3)
        self.assertEqual([chunk.text for chunk in chunks], ["abcdefghij", "hijklmnopq", "opqrst"])
        self.assertTrue(all(chunk.page_number == 3 for chunk in chunks))
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks],
            [chunk.chunk_id for chunk in split_pages("doc-id", [page], 10, 3)],
        )

    def test_rejects_invalid_overlap_and_empty_document_id(self):
        with self.assertRaises(ValueError):
            split_pages("id", [ExtractedPage(None, "text")], 10, 10)
        with self.assertRaises(ValueError):
            split_pages("", [], 10, 1)

    def test_ignores_blank_pages_and_enforces_chunk_cap(self):
        self.assertEqual(
            split_pages(
                "doc-id",
                [None, ExtractedPage(1, ""), ExtractedPage(2, "   ")],
                10,
                3,
            ),
            [],
        )
        self.assertEqual(split_pages("doc-id", None, 10, 3), [])

        text = "a" * (10_001 * 10)
        with self.assertRaises(ValueError):
            split_pages("doc-id", [ExtractedPage(1, text)], chunk_size=10, overlap=3)


if __name__ == "__main__":
    unittest.main()
