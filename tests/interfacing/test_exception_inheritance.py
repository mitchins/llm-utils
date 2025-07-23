import pytest
from llm_utils.clients.base import LLMError, RateLimitExceeded
from llm_utils.clients.mock_client import MockError
from llm_utils.clients.google_genai_client import GeminiContentBlockedException, GeminiTokenLimitException
from llm_utils.clients.openai_client import (
    LLMTimeoutError, LLMConnectionError, LLMModelNotFoundError, 
    LLMAccessDeniedError, LLMInvalidRequestError, LLMUnexpectedResponseError
)


def test_base_exceptions_inherit_correctly():
    """Test that base exceptions inherit from proper base classes."""
    assert issubclass(LLMError, Exception)
    assert issubclass(RateLimitExceeded, LLMError)


def test_mock_error_inherits_from_llm_error():
    """Test that MockError inherits from LLMError."""
    assert issubclass(MockError, LLMError)
    
    error = MockError("test message")
    assert isinstance(error, LLMError)
    assert isinstance(error, Exception)
    assert str(error) == "test message"


def test_gemini_exceptions_inherit_from_llm_error():
    """Test that Gemini-specific exceptions inherit from LLMError."""
    assert issubclass(GeminiContentBlockedException, LLMError)
    assert issubclass(GeminiTokenLimitException, LLMError)
    
    # Test instantiation
    content_error = GeminiContentBlockedException("Content blocked")
    assert isinstance(content_error, LLMError)
    assert isinstance(content_error, Exception)
    
    token_error = GeminiTokenLimitException("Token limit exceeded")
    assert isinstance(token_error, LLMError)
    assert isinstance(token_error, Exception)


def test_openai_exceptions_inherit_from_llm_error():
    """Test that OpenAI-specific exceptions inherit from LLMError."""
    openai_exceptions = [
        LLMTimeoutError, LLMConnectionError, LLMModelNotFoundError,
        LLMAccessDeniedError, LLMInvalidRequestError, LLMUnexpectedResponseError
    ]
    
    for exception_class in openai_exceptions:
        assert issubclass(exception_class, LLMError), f"{exception_class.__name__} should inherit from LLMError"
        
        # Test instantiation
        error = exception_class("test message")
        assert isinstance(error, LLMError)
        assert isinstance(error, Exception)
        assert str(error) == "test message"


def test_exception_hierarchy_allows_catching_all_llm_errors():
    """Test that all LLM errors can be caught with LLMError base class."""
    exceptions_to_test = [
        RateLimitExceeded("Rate limit"),
        MockError("Mock error"),
        GeminiContentBlockedException("Content blocked"),
        GeminiTokenLimitException("Token limit"),
        LLMTimeoutError("Timeout"),
        LLMConnectionError("Connection failed"),
        LLMModelNotFoundError("Model not found"),
        LLMAccessDeniedError("Access denied"),
        LLMInvalidRequestError("Invalid request"),
        LLMUnexpectedResponseError("Unexpected response")
    ]
    
    for error in exceptions_to_test:
        try:
            raise error
        except LLMError:
            # Should be caught here
            pass
        except Exception:
            pytest.fail(f"{type(error).__name__} was not caught by LLMError base class")