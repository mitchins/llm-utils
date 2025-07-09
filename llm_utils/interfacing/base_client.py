from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract interface for language model clients."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: int = 60, system_prompt: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 1024,
                 repetition_penalty: float = 1.1):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.system_prompt = system_prompt or "You are a helpful and concise assistant."
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty

    @abstractmethod
    def chat(self, prompt: str, temperature: float | None = None,
             max_tokens: int | None = None,
             repetition_penalty: float | None = None, stream: bool = False) -> str:
        """Return the LLM response for the given prompt."""
        raise NotImplementedError
