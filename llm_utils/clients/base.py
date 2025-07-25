import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for all LLM clients."""


class RateLimitExceeded(LLMError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, details: str = "Rate limit exceeded"):
        self.details = details
        super().__init__(details)


class NoValidAPIKeysError(RateLimitExceeded):
    """Exception raised when all API keys are invalid or exhausted."""


class BaseLLMClient(ABC):
    def __init__(self, model_name=None, system_prompt=None, max_retries: int = 1, retry_interval: int = 5):
        """
        Abstract base class for LLM clients.

        Args:
            model_name (str, optional): Name of the model. Defaults to None.
            max_retries (int): Maximum number of retries on rate limit exceeded errors.
            retry_interval (int): Seconds to wait between retries.
        """
        self.model = model_name
        self.system_prompt = system_prompt or "You are a helpful and concise assistant."
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
        reasoning: bool | None = None,
    ) -> str:
        """
        (Internal) Generate a response from the language model.

        This method is responsible for the core logic of making a single request
        to the LLM. It should be implemented by subclasses and is not intended to
        be called directly by users.

        Args:
            prompt (str): The prompt text.
            system (str, optional): System message or instruction. Defaults to "".
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            images (list[str] | None, optional): Optional list of base64 encoded
                images to include with the prompt.

        Returns:
            str: The generated response.

        Raises:
            RateLimitExceeded: If the API rate limit is exceeded.
        """
        pass

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
        reasoning: bool | None = None,
    ) -> str:
        """
        Generate a response from the language model with automatic retry logic.

        This method wraps the internal `_generate` method, adding a retry loop
        that handles `RateLimitExceeded` errors. If a request fails due to rate
        limiting, it will be retried up to `max_retries` times, with a pause of
        `retry_interval` seconds between each attempt.

        Args:
            prompt (str): The prompt text.
            system (str, optional): System message or instruction. Defaults to "".
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            images (list[str] | None, optional): Optional list of base64 encoded
                images to include with the prompt.

        Returns:
            str: The generated response.
        """
        retries_left = self.max_retries
        while True:
            try:
                return self._generate(prompt, system=system, temperature=temperature, images=images, reasoning=reasoning)
            except RateLimitExceeded as e:
                if retries_left > 0:
                    logger.warning(f"Rate limit exceeded. Retrying in {self.retry_interval} seconds... ({retries_left} retries left)")
                    time.sleep(self.retry_interval)
                    retries_left -= 1
                else:
                    raise e
