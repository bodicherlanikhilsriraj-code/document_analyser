"""Small Ollama HTTP client with explicit service/model/timeout errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .errors import (
    GenerationError,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    models: tuple[str, ...] = ()
    detail: str = ""


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def status(self) -> OllamaStatus:
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags", timeout=min(self.timeout, 3)
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                return OllamaStatus(False, (), "Ollama returned an invalid model-list response.")
            models = tuple(
                str(model["name"])
                for model in payload.get("models", [])
                if isinstance(model, dict) and model.get("name")
            )
            return OllamaStatus(True, models, "Ollama is responding.")
        except (requests.RequestException, ValueError):
            return OllamaStatus(False, (), "Ollama is not reachable at the configured address.")

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        history: list[dict[str, str]] | None = None,
        format: str | dict[str, Any] | None = None,
    ) -> str:
        status = self.status()
        if not status.available:
            raise OllamaUnavailableError(
                "Ollama is unavailable. Start the Ollama app and check its local API address."
            )
        if model not in status.models:
            raise OllamaModelMissingError(
                f"The selected model '{model}' is not installed in Ollama. Choose an installed "
                "model or explicitly pull it with Ollama."
            )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for item in (history or [])[-6:]:
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content[-4000:]})
        messages.append({"role": "user", "content": prompt})
        request_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if format is not None:
            request_body["format"] = format

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=request_body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except requests.Timeout as exc:
            raise OllamaTimeoutError(
                f"Model generation timed out after {self.timeout} seconds. Try a smaller "
                "request or increase the timeout in Settings."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                "Could not complete the request to Ollama. Check that the service is running."
            ) from exc
        except ValueError as exc:
            raise GenerationError("Ollama returned an unreadable response.") from exc

        content = payload.get("message", {}).get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Ollama returned an empty response.")
        return content.strip()
