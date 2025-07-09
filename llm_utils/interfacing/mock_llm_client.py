from __future__ import annotations

import json
from pathlib import Path

from .base_client import BaseLLMClient
from .llm_request import LLMError


class LLMRequestMismatchError(LLMError):
    """Raised when no mock response matches the given request."""


class MockLLMClient(BaseLLMClient):
    """LLM client that returns predefined responses for testing."""

    def __init__(self, data: list[dict] | str | Path, **kwargs):
        """Initialize the mock client.

        Parameters
        ----------
        data: list[dict] | str | Path
            Either a list of response records or path to a JSON file containing
            such a list. Each record must include ``response`` and all request
            fields used for lookup (e.g. ``model``, ``system_prompt``,
            ``user_prompt``/``prompt``, ``temperature``, etc.).
        kwargs: any
            Additional parameters forwarded to :class:`BaseLLMClient`.
        """
        super().__init__(**kwargs)

        if isinstance(data, (str, Path)):
            records = json.loads(Path(data).read_text())
        else:
            records = data

        self._lookup: dict[str, str] = {}
        for rec in records:
            key = self._make_key(
                prompt=rec.get("user_prompt") or rec.get("prompt"),
                temperature=rec.get("temperature"),
                max_tokens=rec.get("max_tokens"),
                repetition_penalty=rec.get("repetition_penalty"),
                model=rec.get("model"),
                system_prompt=rec.get("system_prompt"),
            )
            self._lookup[key] = rec["response"]

    def _make_key(self, prompt: str, temperature, max_tokens, repetition_penalty, model=None, system_prompt=None) -> str:
        data = {
            "model": model if model is not None else self.model,
            "system_prompt": system_prompt if system_prompt is not None else self.system_prompt,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else self.repetition_penalty,
        }
        return json.dumps(data, sort_keys=True)

    def chat(self, prompt: str, temperature=None, max_tokens=None, repetition_penalty=None, stream: bool = False) -> str:
        if stream:
            raise NotImplementedError("Streaming is not supported in MockLLMClient.")

        key = self._make_key(prompt, temperature, max_tokens, repetition_penalty)
        try:
            return self._lookup[key]
        except KeyError:
            raise LLMRequestMismatchError(f"No mock response for request: {key}")
