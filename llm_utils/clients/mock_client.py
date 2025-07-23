import json
from typing import Iterable, Mapping, Tuple, Union, List, Dict, Callable, Optional

from .base import BaseLLMClient, LLMError

class MockError(LLMError):
    """Generic mock exception for tests."""
    def __init__(self, message: str):
        super().__init__(message)


class MockLLMClient(BaseLLMClient):
    """A simple mock client that returns predefined responses.

    Responses can be provided as a path to a JSON file or as a list of
    dictionaries. Each dictionary should contain at least ``prompt`` and
    ``response`` keys. ``model``, ``system`` and ``temperature`` default to the
    values supplied when calling :meth:`generate` or during initialisation.

    Example JSON structure::

        [
            {
                "model": "dummy",
                "system": "You are helpful",
                "prompt": "Hello",
                "temperature": 0.0,
                "response": "Hi there"
            }
        ]
    """

    def _parse_responses(self, responses):
        """Normalize responses: load JSON, default to list, or single Exception."""
        # Load from JSON file path
        if isinstance(responses, str):
            with open(responses, "r", encoding="utf-8") as fh:
                return json.load(fh)
        # Default to empty list
        if responses is None:
            return []
        # Single-exception sentinel
        if isinstance(responses, list) and responses and isinstance(responses[0], Exception):
            return responses[0]
        # Otherwise assume iterable of mappings
        return list(responses)

    def __init__(
        self,
        responses: Union[str, Iterable[Mapping[str, Union[str, float]]]] = None,
        model_name: str | None = None,
        on_request: Optional[Callable] = None,
    ) -> None:
        super().__init__(model_name)
        parsed = self._parse_responses(responses)
        if isinstance(parsed, Exception):
            self._only_error = parsed
            self._responses = {}
        else:
            self._only_error = None
            self._responses = self._build_response_map(parsed)
        self._on_request = on_request

    def _build_response_map(self, entries):
        """Build a dict mapping request keys to responses."""
        resp_map = {}
        for entry in entries:  # type: ignore
            model = entry.get("model", self.model)
            system = entry.get("system", "")
            prompt = entry.get("prompt")
            temperature = float(entry.get("temperature", 0.0))
            resp = entry.get("response")
            if prompt is None or resp is None:
                raise ValueError("Each entry must include 'prompt' and 'response'")
            key = self._make_key(model, system, prompt, temperature)
            resp_map[key] = resp
        return resp_map

    @staticmethod
    def _make_key(model: str, system: str, prompt: str, temperature: float) -> Tuple[str, str, str, float]:
        return model, system, prompt, temperature

    def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> str:
        # If initialized with a single exception, raise it immediately
        if getattr(self, "_only_error", None):
            raise self._only_error
        if self._on_request:
            override = self._on_request(prompt=prompt, system=system, temperature=temperature, images=images)
            if override is not None:
                return override
        key = self._make_key(self.model, system, prompt, temperature)
        if key not in self._responses:
            raise LLMError(f"No mock response for request: {key}")
        response = self._responses[key]
        if isinstance(response, Exception):
            raise response
        return response

    def prompt(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> str:
        """
        Alias for generate, plus supports callback override.
        """
        return self.generate(prompt, system=system, temperature=temperature, images=images)
