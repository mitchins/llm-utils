import pytest
from unittest.mock import patch, MagicMock
from llm_utils.clients.factory import create_client, LLMProvider
from llm_utils.clients.google_genai_client import GoogleLLMClient
from llm_utils.clients.openai_client import OpenAILikeLLMClient


def test_create_gemini_client():
    """Test factory creates GoogleLLMClient for gemini provider."""
    with patch('llm_utils.clients.factory.GoogleLLMClient') as MockGoogleClient:
        mock_instance = MagicMock()
        MockGoogleClient.return_value = mock_instance
        
        client = create_client(LLMProvider.GEMINI, model_name="gemini-pro", api_key="test-key")
        
        MockGoogleClient.assert_called_once_with(model="gemini-pro", api_key="test-key")
        assert client == mock_instance


def test_create_openai_client():
    """Test factory creates OpenAILikeLLMClient for openai provider."""
    with patch('llm_utils.clients.factory.OpenAILikeLLMClient') as MockOpenAIClient:
        mock_instance = MagicMock()
        MockOpenAIClient.return_value = mock_instance
        
        client = create_client(LLMProvider.OPENAI, model_name="gpt-4", base_url="https://api.openai.com/v1")
        
        MockOpenAIClient.assert_called_once_with(model="gpt-4", base_url="https://api.openai.com/v1")
        assert client == mock_instance


def test_create_local_client():
    """Test factory creates OpenAILikeLLMClient for local provider."""
    with patch('llm_utils.clients.factory.OpenAILikeLLMClient') as MockOpenAIClient:
        mock_instance = MagicMock()
        MockOpenAIClient.return_value = mock_instance
        
        client = create_client(LLMProvider.LOCAL, model_name="local-model", base_url="http://localhost:1234/v1")
        
        MockOpenAIClient.assert_called_once_with(model="local-model", base_url="http://localhost:1234/v1")
        assert client == mock_instance


def test_create_client_unknown_provider():
    """Test factory raises ValueError for unknown provider."""
    with pytest.raises(ValueError, match="Unknown LLM provider: unknown"):
        create_client("unknown", model_name="test-model")


def test_create_client_with_additional_kwargs():
    """Test factory passes additional kwargs to client constructor."""
    with patch('llm_utils.clients.factory.GoogleLLMClient') as MockGoogleClient:
        mock_instance = MagicMock()
        MockGoogleClient.return_value = mock_instance
        
        client = create_client(
            LLMProvider.GEMINI, 
            model_name="gemini-pro", 
            api_key="test-key",
            max_retries=3,
            retry_interval=10,
            timeout=120
        )
        
        MockGoogleClient.assert_called_once_with(
            model="gemini-pro", 
            api_key="test-key",
            max_retries=3,
            retry_interval=10,
            timeout=120
        )
        assert client == mock_instance


def test_enum_string_equivalence():
    """Test that LLMProvider enum works with string values."""
    assert LLMProvider.GEMINI == "gemini"
    assert LLMProvider.OPENAI == "openai" 
    assert LLMProvider.LOCAL == "local"