import base64
import types
import pytest
from unittest.mock import Mock, patch
import google.generativeai as genai
from llm_utils.clients.google_genai_client import GoogleLLMClient
from llm_utils.clients.base import RateLimitExceeded

IMG_B64 = "AAA="


class DummyResponse:
    def __init__(self):
        self.parts = [types.SimpleNamespace(text="ok")]
        self.text = "ok"
        self.prompt_feedback = None


class DummyModel:
    def __init__(self, model_name=None, system_instruction=None, tools=None):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.tools = tools

    def generate_content(self, contents, generation_config=None, safety_settings=None):
        DummyModel.captured = {
            "contents": contents,
            "generation_config": generation_config,
            "safety_settings": safety_settings,
            "tools": self.tools,
        }
        return DummyResponse()


class RateLimitError(Exception):
    """Mock rate limit error for testing."""
    def __init__(self, message="Rate limit exceeded"):
        self.status_code = 429
        super().__init__(message)


class RateLimitErrorString(Exception):
    """Mock rate limit error with string detection for testing."""
    def __init__(self, message="API rate limit exceeded"):
        super().__init__(message)


def test_generate_with_images(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda model_name, system_instruction=None, tools=None: DummyModel(model_name, system_instruction, tools))

    client = GoogleLLMClient(model="gemini", api_key="key")
    result = client.generate("Describe", images=[IMG_B64])

    assert result == "ok"
    captured = DummyModel.captured
    assert isinstance(captured["contents"], list)
    assert captured["contents"][0] == "Describe"
    assert base64.b64decode(IMG_B64) == captured["contents"][1]["inline_data"]["data"]


def test_reasoning_parameter(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda model_name, system_instruction=None, tools=None: DummyModel(model_name, system_instruction, tools))

    client = GoogleLLMClient(model="gemini", api_key="key")

    # Test with reasoning=False
    client.generate("test prompt", reasoning=False)
    assert DummyModel.captured["tools"] == []

    # Test with reasoning=True
    client.generate("test prompt", reasoning=True)
    assert DummyModel.captured["tools"] is None

    # Test with reasoning=None
    client.generate("test prompt", reasoning=None)
    assert DummyModel.captured["tools"] is None


def test_rate_limit_exception_with_status_code(monkeypatch):
    """Test that 429 status code raises RateLimitExceeded."""
    monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
    
    def mock_model_factory(model_name, system_instruction=None, tools=None):
        mock_model = DummyModel(model_name, system_instruction, tools)
        mock_model.generate_content = lambda *args, **kwargs: (_ for _ in ()).throw(RateLimitError("Rate limit exceeded"))
        return mock_model
    
    monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
    
    client = GoogleLLMClient(model="gemini", api_key="key")
    
    with pytest.raises(RateLimitExceeded) as exc_info:
        client.generate("test prompt")
    
    assert "Google API rate limit exceeded" in str(exc_info.value)


def test_rate_limit_exception_with_string_detection(monkeypatch):
    """Test that rate limit string in error message raises RateLimitExceeded."""
    monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
    
    def mock_model_factory(model_name, system_instruction=None, tools=None):
        mock_model = DummyModel(model_name, system_instruction, tools)
        mock_model.generate_content = lambda *args, **kwargs: (_ for _ in ()).throw(RateLimitErrorString("API rate limit exceeded"))
        return mock_model
    
    monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
    
    client = GoogleLLMClient(model="gemini", api_key="key")
    
    with pytest.raises(RateLimitExceeded) as exc_info:
        client.generate("test prompt")
    
    assert "Google API rate limit exceeded" in str(exc_info.value)


def test_rate_limit_exception_with_429_string(monkeypatch):
    """Test that '429' in error message raises RateLimitExceeded."""
    monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
    
    def mock_model_factory(model_name, system_instruction=None, tools=None):
        mock_model = DummyModel(model_name, system_instruction, tools)
        mock_model.generate_content = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("HTTP 429 Error"))
        return mock_model
    
    monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
    
    client = GoogleLLMClient(model="gemini", api_key="key")
    
    with pytest.raises(RateLimitExceeded) as exc_info:
        client.generate("test prompt")
    
    assert "Google API rate limit exceeded" in str(exc_info.value)