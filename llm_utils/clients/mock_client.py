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
        self._on_request = on_request
        self.set_responses(responses)

    def _build_response_map(self, entries):
        """Build a dict mapping request keys to responses."""
        resp_map = {}
        for entry in entries:  # type: ignore
            model = entry.get("model", None)
            system = entry.get("system", None)
            prompt = entry.get("prompt")
            temperature = entry.get("temperature", None)
            if temperature is not None:
                temperature = float(temperature)
            resp = entry.get("response")
            if resp is None:
                raise ValueError("Each entry must include 'response'")
            key = self._make_key(model, system, prompt, temperature)
            resp_map[key] = resp
        return resp_map

    def set_responses(self, entries: Union[List[Mapping[str, Union[str, float]]], Exception, None]) -> None:
        """
        Configure the mock: if given an Exception, raise it on generate;
        otherwise treat entries as a list of mappings with 'prompt' and 'response'.
        """
        parsed = self._parse_responses(entries)
        if isinstance(parsed, Exception):
            self._only_error = parsed
            self._response_map = {}
        else:
            self._only_error = None
            self._response_map = self._build_response_map(parsed)

    @staticmethod
    def _make_key(model: str, system: str, prompt: str, temperature: float) -> Tuple[str, str, str, float]:
        return model, system, prompt, temperature

    def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
        reasoning: bool | None = None,
    ) -> str:
        # If initialized with a single exception, raise it immediately
        if getattr(self, "_only_error", None):
            raise self._only_error
        if self._on_request:
            override = self._on_request(prompt=prompt, system=system, temperature=temperature, images=images)
            # If the override is an exception, re-raise it to mimic real LLM errors
            if isinstance(override, Exception):
                raise override
            return override
        key = self._make_key(self.model, system, prompt, temperature)
        # Wildcard match: None in stored key matches any value
        for stored_key, stored_resp in self._response_map.items():
            if all(
                sk is None or sk == k
                for sk, k in zip(stored_key, key)
            ):
                response = stored_resp
                break
        else:
            raise LLMError(f"No mock response for request: {key}")
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
