import sys
import types
import unittest
from unittest.mock import patch

from core.errors import WebSearchError
from core.web_search import _validated_result, search_web


class WebSearchTests(unittest.TestCase):
    def test_rejects_invalid_and_local_urls(self):
        self.assertIsNone(_validated_result({"title": "bad", "href": "javascript:alert(1)"}))
        self.assertIsNone(_validated_result({"title": "local", "href": "http://127.0.0.1/a"}))
        self.assertIsNone(_validated_result({"title": "credential", "href": "https://u:p@example.com"}))

    def test_returns_only_valid_http_results(self):
        self.assertEqual(
            _validated_result(
                {"title": "Source", "body": "Summary", "href": "https://example.com/article"}
            ).domain,
            "example.com",
        )

    def test_provider_failure_is_reported(self):
        fake_module = types.ModuleType("ddgs")

        class BrokenDDGS:
            def __init__(self, timeout):
                pass

            def text(self, query, max_results):
                raise OSError("network down")

        fake_module.DDGS = BrokenDDGS
        with patch.dict(sys.modules, {"ddgs": fake_module}):
            with self.assertRaises(WebSearchError):
                search_web("study topic")


if __name__ == "__main__":
    unittest.main()
