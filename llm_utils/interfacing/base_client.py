from abc import ABC, abstractmethod


class LLMError(Exception):
    """Base exception for all LLM clients."""


class BaseLLMClient(ABC):
    def __init__(self, model_name=None, system_prompt=None):
        """
        Abstract base class for LLM clients.

        Args:
            model_name (str, optional): Name of the model. Defaults to None.
        """
        self.model = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."

    @abstractmethod
    def generate(self, prompt, system="", temperature=0.0) -> str:
        """
        Generate a response from the language model.

        Args:
            prompt (str): The prompt text.
            system (str, optional): System message or instruction. Defaults to "".
            temperature (float, optional): Sampling temperature. Defaults to 0.0.

        Returns:
            str: The generated response.
        """
        pass
