from enum import Enum
from llm_utils.clients.google_genai_client import GoogleLLMClient
from llm_utils.clients.openai_client import OpenAILikeLLMClient
from llm_utils.clients.base import BaseLLMClient


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    LOCAL = "local"


def create_client(provider: LLMProvider, model_name: str = None, **kwargs) -> BaseLLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider (str): The provider name ('gemini', 'openai', 'local').
        model_name (str, optional): The model name to use.
        **kwargs: Additional keyword arguments passed to the client.

    Returns:
        BaseLLMClient: An instance of a class implementing BaseLLMClient.
    """
    if provider == LLMProvider.GEMINI:
        return GoogleLLMClient(model=model_name, **kwargs)
    elif provider in {LLMProvider.OPENAI, LLMProvider.LOCAL}:
        return OpenAILikeLLMClient(model=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")