import unittest

import requests

from core.errors import OllamaModelMissingError, OllamaTimeoutError
from core.llm import OllamaClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, models=("llama3.2:3b",), response=None, post_error=None):
        self.models = models
        self.response = response or {"message": {"content": "Generated answer"}}
        self.post_error = post_error
        self.post_json = None

    def get(self, url, timeout):
        if isinstance(self.post_error, requests.ConnectionError):
            raise self.post_error
        return FakeResponse({"models": [{"name": name} for name in self.models]})

    def post(self, url, json, timeout):
        self.post_json = json
        if self.post_error:
            raise self.post_error
        return FakeResponse(self.response)


class LLMTests(unittest.TestCase):
    def test_generation_uses_installed_model(self):
        client = OllamaClient("http://localhost:11434")
        client._session = FakeSession()
        self.assertEqual(client.generate("llama3.2:3b", "hello"), "Generated answer")

    def test_generation_passes_optional_json_schema_to_ollama(self):
        client = OllamaClient("http://localhost:11434")
        client._session = FakeSession()
        schema = {"type": "object"}
        client.generate("llama3.2:3b", "hello", format=schema)
        self.assertEqual(client._session.post_json["format"], schema)

    def test_missing_model_and_generation_timeout_are_distinct(self):
        client = OllamaClient("http://localhost:11434")
        client._session = FakeSession()
        with self.assertRaises(OllamaModelMissingError):
            client.generate("qwen2.5:3b", "hello")
        client._session = FakeSession(post_error=requests.Timeout())
        with self.assertRaises(OllamaTimeoutError):
            client.generate("llama3.2:3b", "hello")

    def test_unavailable_service_is_reported_without_generation(self):
        from core.errors import OllamaUnavailableError

        client = OllamaClient("http://localhost:11434")
        client._session = FakeSession(post_error=requests.ConnectionError())
        with self.assertRaises(OllamaUnavailableError):
            client.generate("llama3.2:3b", "hello")


if __name__ == "__main__":
    unittest.main()
