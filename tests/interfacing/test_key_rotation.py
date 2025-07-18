import pytest
from unittest.mock import Mock, call
from llm_utils.interfacing.key_rotation import (
    KeyRotationManager, 
    CircularRotationStrategy,
    KeyRotationStrategy
)
from llm_utils.interfacing.base_client import RateLimitExceeded


class MockStrategy(KeyRotationStrategy):
    """Mock strategy for testing."""
    
    def __init__(self):
        self.reset_calls = 0
        self.get_next_calls = []
    
    def get_next_key(self, keys, current_key=None):
        self.get_next_calls.append((keys, current_key))
        if len(self.get_next_calls) == 1:
            return keys[0] if keys else None
        elif len(self.get_next_calls) == 2:
            return keys[1] if len(keys) > 1 else None
        return None
    
    def reset(self):
        self.reset_calls += 1


class TestCircularRotationStrategy:
    
    def test_single_key(self):
        strategy = CircularRotationStrategy()
        keys = ["key1"]
        
        assert strategy.get_next_key(keys) == "key1"
        assert strategy.get_next_key(keys) is None  # Exhausted
    
    def test_multiple_keys_circular(self):
        strategy = CircularRotationStrategy()
        keys = ["key1", "key2", "key3"]
        
        assert strategy.get_next_key(keys) == "key1"
        assert strategy.get_next_key(keys) == "key2"
        assert strategy.get_next_key(keys) == "key3"
        assert strategy.get_next_key(keys) is None  # All exhausted
    
    def test_reset_allows_reuse(self):
        strategy = CircularRotationStrategy()
        keys = ["key1", "key2"]
        
        # First cycle
        assert strategy.get_next_key(keys) == "key1"
        assert strategy.get_next_key(keys) == "key2"
        assert strategy.get_next_key(keys) is None
        
        # Reset and try again
        strategy.reset()
        assert strategy.get_next_key(keys) == "key1"
    
    def test_remember_last_working_key(self):
        strategy = CircularRotationStrategy()
        keys = ["key1", "key2", "key3"]
        
        # Mark key2 as working
        strategy.mark_key_working(keys, "key2")
        strategy.reset()
        
        # Should start with key2 next time and go sequentially through the array
        assert strategy.get_next_key(keys) == "key2"  # Start with working key
        assert strategy.get_next_key(keys) == "key3"  # Next in array
        assert strategy.get_next_key(keys) == "key1"  # Wrap around to first
    
    def test_empty_keys(self):
        strategy = CircularRotationStrategy()
        assert strategy.get_next_key([]) is None


class TestKeyRotationManager:
    
    def test_init_with_empty_keys_raises_error(self):
        with pytest.raises(ValueError, match="At least one API key must be provided"):
            KeyRotationManager([])
    
    def test_init_with_single_key(self):
        manager = KeyRotationManager(["key1"])
        assert manager.available_keys == ["key1"]
        assert manager.get_initial_key() == "key1"
    
    def test_init_with_multiple_keys(self):
        manager = KeyRotationManager(["key1", "key2", "key3"])
        assert manager.available_keys == ["key1", "key2", "key3"]
    
    def test_get_initial_key(self):
        manager = KeyRotationManager(["key1", "key2"])
        assert manager.get_initial_key() == "key1"
        assert manager.current_key == "key1"
    
    def test_get_next_key(self):
        manager = KeyRotationManager(["key1", "key2", "key3"])
        manager.get_initial_key()  # Start rotation
        
        assert manager.get_next_key() == "key2"
        assert manager.current_key == "key2"
        assert manager.get_next_key() == "key3"
        assert manager.current_key == "key3"
        assert manager.get_next_key() is None  # Exhausted
    
    def test_mark_key_success(self):
        strategy = CircularRotationStrategy()
        manager = KeyRotationManager(["key1", "key2", "key3"], strategy)
        
        manager.mark_key_success("key2")
        manager.reset_rotation()
        
        # Should start with key2 next time
        assert manager.get_initial_key() == "key2"
    
    def test_reset_rotation(self):
        manager = KeyRotationManager(["key1", "key2"])
        manager.get_initial_key()
        manager.get_next_key()  # Move to key2
        
        manager.reset_rotation()
        assert manager.get_initial_key() == "key1"  # Back to start
    
    def test_custom_strategy(self):
        strategy = MockStrategy()
        manager = KeyRotationManager(["key1", "key2"], strategy)
        
        manager.get_initial_key()
        assert strategy.reset_calls == 1
        assert len(strategy.get_next_calls) == 1
    
    def test_keys_immutability(self):
        original_keys = ["key1", "key2"]
        manager = KeyRotationManager(original_keys)
        
        # Modify original list
        original_keys.append("key3")
        
        # Manager should be unaffected
        assert manager.available_keys == ["key1", "key2"]
        
        # Returned list should be a copy
        returned_keys = manager.available_keys
        returned_keys.append("key4")
        assert manager.available_keys == ["key1", "key2"]


class TestExecuteWithRotation:
    
    def test_successful_first_key(self):
        manager = KeyRotationManager(["key1", "key2"])
        
        def mock_operation(key):
            return f"success with {key}"
        
        def is_rate_limit(e):
            return False
        
        result = manager.execute_with_rotation(mock_operation, is_rate_limit)
        assert result == "success with key1"
        assert manager.current_key == "key1"
    
    def test_rotation_on_rate_limit(self):
        manager = KeyRotationManager(["key1", "key2", "key3"])
        call_count = 0
        
        def mock_operation(key):
            nonlocal call_count
            call_count += 1
            if key == "key1":
                raise Exception("429 Rate limit exceeded")
            elif key == "key2":
                raise Exception("Rate limit hit")
            return f"success with {key}"
        
        def is_rate_limit(e):
            return "rate limit" in str(e).lower() or "429" in str(e)
        
        result = manager.execute_with_rotation(mock_operation, is_rate_limit)
        assert result == "success with key3"
        assert call_count == 3
    
    def test_all_keys_exhausted_raises_rate_limit_exceeded(self):
        manager = KeyRotationManager(["key1", "key2"])
        
        def mock_operation(key):
            raise Exception("429 Rate limit exceeded")
        
        def is_rate_limit(e):
            return "429" in str(e)
        
        with pytest.raises(RateLimitExceeded) as exc_info:
            manager.execute_with_rotation(mock_operation, is_rate_limit)
        
        assert "All 2 API keys exhausted" in str(exc_info.value)
    
    def test_non_rate_limit_error_not_rotated(self):
        manager = KeyRotationManager(["key1", "key2"])
        call_count = 0
        
        def mock_operation(key):
            nonlocal call_count
            call_count += 1
            raise ValueError("Some other error")
        
        def is_rate_limit(e):
            return isinstance(e, RateLimitExceeded)
        
        with pytest.raises(ValueError):
            manager.execute_with_rotation(mock_operation, is_rate_limit)
        
        # Should only call once, no rotation
        assert call_count == 1
    
    def test_remembers_successful_key(self):
        manager = KeyRotationManager(["key1", "key2", "key3"])
        
        def mock_operation(key):
            if key == "key1":
                raise Exception("Rate limit exceeded")
            return f"success with {key}"
        
        def is_rate_limit(e):
            return "rate limit" in str(e).lower()
        
        # First call succeeds with key2
        result = manager.execute_with_rotation(mock_operation, is_rate_limit)
        assert result == "success with key2"
        
        # Reset and verify key2 is tried first next time
        manager.reset_rotation()
        assert manager.get_initial_key() == "key2"