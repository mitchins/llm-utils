import base64
import types
import pytest
from unittest.mock import Mock, patch
import google.genai as genai
import google.genai.errors
from llm_utils.clients.google_genai_client import GoogleLLMClient
from llm_utils.clients.base import RateLimitExceeded

# Stub the new genai.Client to return a DummyModel-like interface
import types as _types
class DummyClient:
    def __init__(self, api_key=None, error=None):
        self.api_key = api_key
        self.error = error
        self.models = _types.SimpleNamespace(generate_content=self._wrapped_generate)
    def _wrapped_generate(self, model=None, contents=None, config=None, **kwargs):
        if self.error:
            raise self.error
        return DummyModel(model, config.system_instruction if hasattr(config, 'system_instruction') else None, None).generate_content(
            model, contents, generation_config=config, safety_settings=None
        )
monkeypatch_client = lambda api_key=None: DummyClient(api_key)

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

    def generate_content(self, model=None, contents=None, generation_config=None, tools=None, **kwargs):
        DummyModel.captured = {
            "model": model,
            "contents": contents,
            "generation_config": generation_config,
            "tools": tools,
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
    monkeypatch.setattr(genai, "Client", monkeypatch_client)

    client = GoogleLLMClient(model="gemini", api_key="key")
    result = client.generate("Describe", images=[IMG_B64])

    assert result == "ok"
    captured = DummyModel.captured
    assert isinstance(captured["contents"], list)
    assert captured["contents"][0] == "Describe"
    assert base64.b64decode(IMG_B64) == captured["contents"][1]["inline_data"]["data"]


def test_reasoning_parameter(monkeypatch):
    monkeypatch.setattr(genai, "Client", monkeypatch_client)

    client = GoogleLLMClient(model="gemini", api_key="key")

    # Test with reasoning=False - should pass tools=None to disable reasoning
    client.generate("test prompt", reasoning=False)
    assert DummyModel.captured["tools"] is None

    # Test with reasoning=True - should not pass tools parameter (reasoning enabled)
    client.generate("test prompt", reasoning=True)
    assert DummyModel.captured["tools"] is None

    # Test with reasoning=None - should not pass tools parameter (default behavior)
    client.generate("test prompt", reasoning=None)
    assert DummyModel.captured["tools"] is None


def test_rate_limit_exception_with_status_code(monkeypatch):
    """Test that 429 status code raises RateLimitExceeded."""
    # Patch genai.Client to return DummyClient that always raises RateLimitError
    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key=None: DummyClient(api_key, error=RateLimitError("Rate limit exceeded"))
    )

    client = GoogleLLMClient(model="gemini", api_key="key")

    with pytest.raises(RateLimitExceeded) as exc_info:
        client.generate("test prompt")

    assert "Google API rate limit exceeded" in str(exc_info.value)


def test_rate_limit_exception_with_string_detection(monkeypatch):
    """Test that rate limit string in error message raises RateLimitExceeded."""
    # Patch genai.Client to return DummyClient that always raises RateLimitErrorString
    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key=None: DummyClient(api_key, error=RateLimitErrorString("API rate limit exceeded"))
    )

    client = GoogleLLMClient(model="gemini", api_key="key")

    with pytest.raises(RateLimitExceeded) as exc_info:
        client.generate("test prompt")

    assert "Google API rate limit exceeded" in str(exc_info.value)


def test_rate_limit_exception_with_429_string(monkeypatch):
    """Test that '429' in error message raises RateLimitExceeded."""
    # Patch genai.Client to return DummyClient that always raises Exception("HTTP 429 Error")
    monkeypatch.setattr(
        genai,
        "Client",
        lambda api_key=None: DummyClient(api_key, error=Exception("HTTP 429 Error"))
    )

    client = GoogleLLMClient(model="gemini", api_key="key")

    with pytest.raises(RateLimitExceeded) as exc_info:
        client.generate("test prompt")

    assert "Google API rate limit exceeded" in str(exc_info.value)