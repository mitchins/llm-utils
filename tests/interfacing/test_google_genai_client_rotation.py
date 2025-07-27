import base64
import types
import pytest
from unittest.mock import Mock, patch
import google.genai as genai
from llm_utils.clients.google_genai_client import GoogleLLMClient
from llm_utils.clients.base import RateLimitExceeded


class DummyResponse:
    def __init__(self):
        self.parts = [types.SimpleNamespace(text="ok")]
        self.text = "ok"
        self.prompt_feedback = None


# Stub the new genai.Client to return a DummyModel-like interface
class DummyClient:
    def __init__(self, api_key=None, error=None):
        self.api_key = api_key
        self.error = error
        self.models = types.SimpleNamespace(generate_content=self._wrapped_generate)
    
    def _wrapped_generate(self, model=None, contents=None, config=None, tools=None, **kwargs):
        if self.error:
            raise self.error
        # Capture call details for test verification
        DummyClient.captured = {
            "model": model,
            "contents": contents,
            "config": config,
            "tools": tools,
        }
        return DummyResponse()

monkeypatch_client = lambda api_key=None, error=None: DummyClient(api_key, error)


class RateLimitError(Exception):
    """Mock rate limit error for testing."""
    def __init__(self, message="Rate limit exceeded"):
        self.status_code = 429
        super().__init__(message)


class TestKeyRotationIntegration:
    """Test key rotation functionality in GoogleLLMClient."""
    
    def test_single_key_mode_backward_compatibility(self, monkeypatch):
        """Test that single key mode works as before."""
        monkeypatch.setattr(genai, "Client", monkeypatch_client)
        
        client = GoogleLLMClient(model="gemini", api_key="single_key")
        result = client.generate("test")
        
        assert result == "ok"
        assert client._keys == ["single_key"]
        assert client._key_rotation_manager is None
    
    def test_multiple_keys_initialization(self, monkeypatch):
        """Test initialization with multiple keys enables rotation."""
        monkeypatch.setattr(genai, "Client", monkeypatch_client)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        
        assert client._keys == ["key1", "key2", "key3"]
        assert client._key_rotation_manager is not None
        assert client._key_rotation_manager.available_keys == ["key1", "key2", "key3"]
    
    def test_empty_key_list_raises_error(self, monkeypatch):
        """Test that empty key list raises appropriate error."""
        monkeypatch.setattr(genai, "Client", monkeypatch_client)
        
        with pytest.raises(ValueError, match="API key list cannot be empty"):
            GoogleLLMClient(model="gemini", api_key=[])
    
    def test_invalid_key_type_raises_error(self, monkeypatch):
        """Test that invalid key type raises appropriate error."""
        monkeypatch.setattr(genai, "Client", monkeypatch_client)
        
        with pytest.raises(ValueError, match="api_key must be a string or list of strings"):
            GoogleLLMClient(model="gemini", api_key=123)
    
    def test_key_rotation_on_rate_limit(self, monkeypatch):
        """Test that rate limit triggers key rotation."""
        client_calls = []
        
        call_count = 0
        def mock_client_factory(api_key=None):
            client_calls.append(api_key)
            
            def mock_generate_content(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # First two calls fail with rate limit, third succeeds
                if call_count <= 2:
                    raise RateLimitError("Rate limit exceeded")
                return DummyResponse()
            
            # Create a client that will fail on the first two generate calls
            client = DummyClient(api_key, error=None)
            client.models.generate_content = mock_generate_content
            return client
        
        monkeypatch.setattr(genai, "Client", mock_client_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        result = client.generate("test prompt")
        
        assert result == "ok"
        # Should have created clients for all three keys (one for init, two more for rotation)
        assert len(client_calls) >= 3
        assert "key1" in client_calls
        assert "key2" in client_calls  
        assert "key3" in client_calls
    
    def test_all_keys_exhausted_raises_rate_limit_exceeded(self, monkeypatch):
        """Test that exhausting all keys raises RateLimitExceeded."""
        def mock_client_factory(api_key=None):
            return DummyClient(api_key, error=RateLimitError("Rate limit exceeded"))
        
        monkeypatch.setattr(genai, "Client", mock_client_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2"])
        
        with pytest.raises(RateLimitExceeded) as exc_info:
            client.generate("test prompt")
        
        assert "All 2 API keys exhausted" in str(exc_info.value)
    
    def test_non_rate_limit_error_no_rotation(self, monkeypatch):
        """Test that non-rate-limit errors don't trigger rotation."""
        client_calls = []
        
        def mock_client_factory(api_key=None):
            client_calls.append(api_key)
            return DummyClient(api_key, error=ValueError("Some other error"))
        
        monkeypatch.setattr(genai, "Client", mock_client_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        
        with pytest.raises(ValueError):
            client.generate("test prompt")
        
        # Should only create client twice: once for init, once for generate call (no rotation)
        assert len(client_calls) == 2
        assert client_calls[0] == "key1"  # Initial client creation
        assert client_calls[1] == "key1"  # Client recreation for generate call
    
    def test_successful_key_remembered(self, monkeypatch):
        """Test that successful key is remembered for next request."""
        client_calls = []
        
        call_count = 0
        def mock_client_factory(api_key=None):
            client_calls.append(api_key)
            # key1 fails, key2 succeeds
            if api_key == "key1":
                return DummyClient(api_key, error=RateLimitError("Rate limit exceeded"))
            else:
                return DummyClient(api_key, error=None)
        
        monkeypatch.setattr(genai, "Client", mock_client_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        
        # First request: key1 fails, key2 succeeds
        result1 = client.generate("test prompt 1")
        assert result1 == "ok"
        
        # Reset call tracking
        initial_calls = len(client_calls)
        
        # Second request: should start with key2 (the successful one)
        result2 = client.generate("test prompt 2")
        assert result2 == "ok"
        # Should only create one more client (for the successful key)
        assert len(client_calls) == initial_calls + 1

    def test_resource_exhausted_triggers_rotation(self, monkeypatch):
        """ResourceExhausted exception variant should trigger rotation."""
        from llm_utils.clients.google_genai_client import ResourceExhausted

        call_count = 0
        def mock_client_factory(api_key=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt raises ResourceExhausted (429 equivalent)
                return DummyClient(api_key, error=ResourceExhausted("RESOURCE_EXHAUSTED: Too Many Requests"))
            return DummyClient(api_key, error=None)

        monkeypatch.setattr(genai, "Client", mock_client_factory)

        client = GoogleLLMClient(model="gemini", api_key=["keyA", "keyB"])
        result = client.generate("prompt")
        assert result == "ok"
        assert call_count == 2  # one fail + one success

    def test_retry_error_unwrapped_triggers_rotation(self, monkeypatch):
        """Wrapped RetryError.last_exc should be detected as rate-limit."""
        from llm_utils.clients.google_genai_client import RetryError, ResourceExhausted

        # Build a fake RetryError that wraps ResourceExhausted
        last_exc = ResourceExhausted("RESOURCE_EXHAUSTED: Too Many Requests")
        retry_err = RetryError("wrapped", last_exc)
        retry_err.last_exc = last_exc  # type: ignore[attr-defined]

        call_count = 0
        def mock_client_factory(api_key=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DummyClient(api_key, error=retry_err)
            return DummyClient(api_key, error=None)

        monkeypatch.setattr(genai, "Client", mock_client_factory)

        client = GoogleLLMClient(model="gemini", api_key=["keyX", "keyY"])
        result = client.generate("prompt")
        assert result == "ok"
        assert call_count == 2  # rotated to second key

    def test_rotation_logging_message(self, monkeypatch, caplog):
        """Verify that 'Rotating API key X/Y' log appears after a rotation."""
        call_count = 0
        def mock_client_factory(api_key=None):
            def mock_generate_content(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError("429 Too Many Requests")
                return DummyResponse()
            
            # Create a client that will fail on the first generate call
            client = DummyClient(api_key, error=None)
            client.models.generate_content = mock_generate_content
            return client

        monkeypatch.setattr(genai, "Client", mock_client_factory)

        from llm_utils.clients import google_genai_client as client_mod
        caplog.set_level("INFO", logger=client_mod.logger.name)

        client = GoogleLLMClient(model="gemini", api_key=["k1", "k2"])
        result = client.generate("prompt")
        assert result == "ok"
        # After rotation we should see info log with 'Rotating API key'
        assert any("Rotating API key 2/" in rec.message for rec in caplog.records)