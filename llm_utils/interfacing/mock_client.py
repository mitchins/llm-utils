import json
from typing import Iterable, Mapping, Tuple, Union, List, Dict, Callable, Optional

from .base_client import BaseLLMClient, LLMError


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

    def __init__(
        self,
        responses: Union[str, Iterable[Mapping[str, Union[str, float]]]],
        model_name: str | None = None,
        on_request: Optional[Callable] = None,
    ) -> None:
        super().__init__(model_name)
        if isinstance(responses, str):
            with open(responses, "r", encoding="utf-8") as fh:
                responses = json.load(fh)
        self._responses: Dict[Tuple[str, str, str, float], str] = {}
        for entry in responses:  # type: ignore[assignment]
            model = entry.get("model", self.model)
            system = entry.get("system", "")
            prompt = entry.get("prompt")
            temperature = float(entry.get("temperature", 0.0))
            response = entry.get("response")
            if prompt is None or response is None:
                raise ValueError("Each entry must include 'prompt' and 'response'")
            key = self._make_key(model, system, prompt, temperature)
            self._responses[key] = response
        self._on_request = on_request

    @staticmethod
    def _make_key(model: str, system: str, prompt: str, temperature: float) -> Tuple[str, str, str, float]:
        return model, system, prompt, temperature

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0, images: List[str] | None = None) -> str:
        if self._on_request:
            override = self._on_request(prompt=prompt, system=system, temperature=temperature)
            if override is not None:
                return override
        key = self._make_key(self.model, system, prompt, temperature)
        if key not in self._responses:
            raise LLMError(f"No mock response for request: {key}")
        return self._responses[key]

    def prompt(self, prompt: str, system: str = "", temperature: float = 0.0, images: List[str] | None = None) -> str:
        """
        Alias for generate, plus supports callback override.
        """
        return self.generate(prompt, system=system, temperature=temperature, images=images)
