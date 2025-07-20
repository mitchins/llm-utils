from __future__ import annotations
import json
from httpx import Response


def decode_json(response: Response):
    """Return JSON body or raise ValueError."""
    try:
        return response.json()
    except Exception as exc:
        raise ValueError("Invalid JSON response") from exc
