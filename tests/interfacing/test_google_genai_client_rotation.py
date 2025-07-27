import base64
import types
import pytest
from unittest.mock import Mock, patch
import google.generativeai as genai
from llm_utils.clients.google_genai_client import GoogleLLMClient
from llm_utils.clients.base import RateLimitExceeded


class DummyResponse:
    def __init__(self):
        self.parts = [types.SimpleNamespace(text="ok")]
        self.text = "ok"
        self.prompt_feedback = None


class DummyModel:
    def __init__(self, model_name=None, system_instruction=None):
        self.model_name = model_name
        self.system_instruction = system_instruction

    def generate_content(self, contents, generation_config=None, safety_settings=None):
        DummyModel.captured = {
            "contents": contents,
            "generation_config": generation_config,
            "safety_settings": safety_settings,
        }
        return DummyResponse()


class RateLimitError(Exception):
    """Mock rate limit error for testing."""
    def __init__(self, message="Rate limit exceeded"):
        self.status_code = 429
        super().__init__(message)


class TestKeyRotationIntegration:
    """Test key rotation functionality in GoogleLLMClient."""
    
    def test_single_key_mode_backward_compatibility(self, monkeypatch):
        """Test that single key mode works as before."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
        monkeypatch.setattr(genai, "GenerativeModel", lambda model_name, system_instruction=None, tools=None: DummyModel(model_name, system_instruction))
        
        client = GoogleLLMClient(model="gemini", api_key="single_key")
        result = client.generate("test")
        
        assert result == "ok"
        assert client._keys == ["single_key"]
        assert client._key_rotation_manager is None
    
    def test_multiple_keys_initialization(self, monkeypatch):
        """Test initialization with multiple keys enables rotation."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        
        assert client._keys == ["key1", "key2", "key3"]
        assert client._key_rotation_manager is not None
        assert client._key_rotation_manager.available_keys == ["key1", "key2", "key3"]
    
    def test_empty_key_list_raises_error(self, monkeypatch):
        """Test that empty key list raises appropriate error."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
        
        with pytest.raises(ValueError, match="API key list cannot be empty"):
            GoogleLLMClient(model="gemini", api_key=[])
    
    def test_invalid_key_type_raises_error(self, monkeypatch):
        """Test that invalid key type raises appropriate error."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
        
        with pytest.raises(ValueError, match="api_key must be a string or list of strings"):
            GoogleLLMClient(model="gemini", api_key=123)
    
    def test_key_rotation_on_rate_limit(self, monkeypatch):
        """Test that rate limit triggers key rotation."""
        configure_calls = []
        
        def mock_configure(api_key):
            configure_calls.append(api_key)
        
        monkeypatch.setattr(genai, "configure", mock_configure)
        
        call_count = 0
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)
            
            def generate_content_with_rotation(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # First two calls fail with rate limit, third succeeds
                if call_count <= 2:
                    raise RateLimitError("Rate limit exceeded")
                return DummyResponse()
            
            mock_model.generate_content = generate_content_with_rotation
            return mock_model
        
        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        result = client.generate("test prompt")
        
        assert result == "ok"
        assert call_count == 3  # Three attempts made
        # Should have configured all three keys
        assert len(configure_calls) >= 3
        assert "key1" in configure_calls
        assert "key2" in configure_calls  
        assert "key3" in configure_calls
        # Ensure the last configure call corresponds to the key that finally succeeded
        assert configure_calls[-1] == "key3"
    
    def test_all_keys_exhausted_raises_rate_limit_exceeded(self, monkeypatch):
        """Test that exhausting all keys raises RateLimitExceeded."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
        
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)
            mock_model.generate_content = lambda *args, **kwargs: (_ for _ in ()).throw(RateLimitError("Rate limit exceeded"))
            return mock_model
        
        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2"])
        
        with pytest.raises(RateLimitExceeded) as exc_info:
            client.generate("test prompt")
        
        assert "All 2 API keys exhausted" in str(exc_info.value)
    
    def test_non_rate_limit_error_no_rotation(self, monkeypatch):
        """Test that non-rate-limit errors don't trigger rotation."""
        configure_calls = []
        
        def mock_configure(api_key):
            configure_calls.append(api_key)
        
        monkeypatch.setattr(genai, "configure", mock_configure)
        
        call_count = 0
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)
            
            def generate_content_with_error(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise ValueError("Some other error")
            
            mock_model.generate_content = generate_content_with_error
            return mock_model
        
        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        
        with pytest.raises(ValueError):
            client.generate("test prompt")
        
        # Should only attempt once, no rotation
        assert call_count == 1
    
    def test_successful_key_remembered(self, monkeypatch):
        """Test that successful key is remembered for next request."""
        configure_calls = []
        
        def mock_configure(api_key):
            configure_calls.append(api_key)
        
        monkeypatch.setattr(genai, "configure", mock_configure)
        
        call_count = 0
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)
            
            def generate_content_selective(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # key1 fails, key2 succeeds
                current_key = configure_calls[-1] if configure_calls else None
                if current_key == "key1":
                    raise RateLimitError("Rate limit exceeded")
                return DummyResponse()
            
            mock_model.generate_content = generate_content_selective
            return mock_model
        
        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)
        
        client = GoogleLLMClient(model="gemini", api_key=["key1", "key2", "key3"])
        
        # First request: key1 fails, key2 succeeds
        result1 = client.generate("test prompt 1")
        assert result1 == "ok"
        
        # Reset call tracking
        configure_calls.clear()
        call_count = 0
        
        # Second request: should start with key2 (the successful one)
        result2 = client.generate("test prompt 2")
        assert result2 == "ok"
        assert call_count == 1  # Should succeed immediately
        assert configure_calls[0] == "key2"  # Should start with key2

    def test_resource_exhausted_triggers_rotation(self, monkeypatch):
        """ResourceExhausted exception variant should trigger rotation."""
        from llm_utils.clients.google_genai_client import ResourceExhausted

        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)

        call_count = 0
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)

            def generate_content_with_exhausted(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First attempt raises ResourceExhausted (429 equivalent)
                    raise ResourceExhausted("RESOURCE_EXHAUSTED: Too Many Requests")
                return DummyResponse()

            mock_model.generate_content = generate_content_with_exhausted
            return mock_model

        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)

        client = GoogleLLMClient(model="gemini", api_key=["keyA", "keyB"])
        result = client.generate("prompt")
        assert result == "ok"
        assert call_count == 2  # one fail + one success

    def test_retry_error_unwrapped_triggers_rotation(self, monkeypatch):
        """Wrapped RetryError.last_exc should be detected as rate-limit."""
        from llm_utils.clients.google_genai_client import RetryError, ResourceExhausted

        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)

        # Build a fake RetryError that wraps ResourceExhausted
        last_exc = ResourceExhausted("RESOURCE_EXHAUSTED: Too Many Requests")
        # RetryError signature can be (message, cause); pass last_exc as cause for real google impl,
        # for stub it is harmless.
        retry_err = RetryError("wrapped", last_exc)
        retry_err.last_exc = last_exc  # type: ignore[attr-defined]

        call_count = 0
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)

            def generate_content_with_retry(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise retry_err
                return DummyResponse()

            mock_model.generate_content = generate_content_with_retry
            return mock_model

        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)

        client = GoogleLLMClient(model="gemini", api_key=["keyX", "keyY"])
        result = client.generate("prompt")
        assert result == "ok"
        assert call_count == 2  # rotated to second key

    def test_rotation_logging_message(self, monkeypatch, caplog):
        """Verify that 'Rotating API key X/Y' log appears after a rotation."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)

        call_count = 0
        def mock_model_factory(model_name, system_instruction=None, tools=None):
            mock_model = DummyModel(model_name, system_instruction)

            def generate_content_rotate(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError("429 Too Many Requests")
                return DummyResponse()

            mock_model.generate_content = generate_content_rotate
            return mock_model

        monkeypatch.setattr(genai, "GenerativeModel", mock_model_factory)

        from llm_utils.clients import google_genai_client as client_mod
        caplog.set_level("INFO", logger=client_mod.logger.name)

        client = GoogleLLMClient(model="gemini", api_key=["k1", "k2"])
        result = client.generate("prompt")
        assert result == "ok"
        # After rotation we should see info log with 'Rotating API key'
        assert any("Rotating API key 2/" in rec.message for rec in caplog.records)