"""Encoding helpers for clearly separating untrusted source data from prompt instructions."""

from __future__ import annotations

import json
from typing import Any


def encode_untrusted_data(value: Any) -> str:
    """Serialize source data as JSON and escape characters that can forge HTML-like tags."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
