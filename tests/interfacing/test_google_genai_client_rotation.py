import base64
import types
import pytest
from unittest.mock import Mock, patch
import google.generativeai as genai
from llm_utils.interfacing.google_genai_client import GoogleLLMClient
from llm_utils.interfacing.base_client import RateLimitExceeded


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
        monkeypatch.setattr(genai, "GenerativeModel", lambda model_name, system_instruction=None: DummyModel(model_name, system_instruction))
        
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
        def mock_model_factory(model_name, system_instruction=None):
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
    
    def test_all_keys_exhausted_raises_rate_limit_exceeded(self, monkeypatch):
        """Test that exhausting all keys raises RateLimitExceeded."""
        monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
        
        def mock_model_factory(model_name, system_instruction=None):
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
        def mock_model_factory(model_name, system_instruction=None):
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
        def mock_model_factory(model_name, system_instruction=None):
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