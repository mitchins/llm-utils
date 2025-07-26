# Stub google.api_core.exceptions if not installed
import sys
import types
if 'google' not in sys.modules:
    google = types.ModuleType('google')
    api_core = types.ModuleType('google.api_core')
    exceptions = types.ModuleType('google.api_core.exceptions')
    class InvalidArgument(Exception):
        pass
    exceptions.InvalidArgument = InvalidArgument
    api_core.exceptions = exceptions
    google.api_core = api_core
    sys.modules['google'] = google
    sys.modules['google.api_core'] = api_core
    sys.modules['google.api_core.exceptions'] = exceptions
import pytest
from unittest.mock import Mock, call
from llm_utils.clients.key_rotation import (
    KeyRotationManager,
    CircularRotationStrategy,
    KeyRotationStrategy
)
from llm_utils.clients.base import RateLimitExceeded, NoValidAPIKeysError
from google.api_core.exceptions import InvalidArgument



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


    def test_invalid_key_then_success_on_second_key(self):
        # First key invalid, second key works
        manager = KeyRotationManager(["badkey", "goodkey"])

        def mock_operation(key):
            if key == "badkey":
                raise InvalidArgument("API key not valid. Please pass a valid API key.")
            return f"success with {key}"

        def is_rate_limit(e):
            return False

        result = manager.execute_with_rotation(mock_operation, is_rate_limit)
        assert result == "success with goodkey"
        # badkey should be removed
        assert manager.available_keys == ["goodkey"]

    def test_all_invalid_keys_exhausted_raises_no_valid_keys_error(self):
        manager = KeyRotationManager(["onlykey"])

        def mock_operation(key):
            raise InvalidArgument("API key not valid. Please pass a valid API key.")

        def is_rate_limit(e):
            return False

        with pytest.raises(NoValidAPIKeysError) as exc_info:
            manager.execute_with_rotation(mock_operation, is_rate_limit)
        assert "All 1 API keys invalid" in str(exc_info.value)


# --- Additional Tests ---


class TestKeyRemovalClearsLastIndex:
    def test_remove_key_clears_last_working_index(self):
        # Setup manager with three keys and a circular strategy
        manager = KeyRotationManager(["k1", "k2", "k3"])
        # Prime rotation and mark "k2" as working
        first = manager.get_initial_key()
        assert first == "k1"
        # Simulate success on "k2"
        manager.mark_key_success("k2")
        # Ensure strategy would start with "k2"
        manager.reset_rotation()
        assert manager.get_initial_key() == "k2"
        # Remove "k2" and verify last_working_index is cleared internally
        manager.remove_key("k2")
        strategy = manager._strategy
        # After removal, last_working_index should be None
        assert getattr(strategy, "_last_working_index") is None
        # Now initial key should default to first available key "k1"
        assert manager.get_initial_key() == "k1"


# --- Additional Tests ---

class TestMixedInvalidAndRateLimit:
    def test_mixed_invalid_and_rate_limit(self):
        """
        badkey1 -> InvalidArgument
        badkey2 -> InvalidArgument
        ratelimited -> 429 rate limit
        goodkey -> success
        """
        keys = ["badkey1", "badkey2", "ratelimited", "goodkey"]
        manager = KeyRotationManager(keys)

        def mock_operation(key):
            if key.startswith("badkey"):
                # Simulate permanently invalid keys
                raise InvalidArgument("API key not valid")
            if key == "ratelimited":
                raise Exception("429 Too Many Requests")
            return f"success with {key}"

        # Simple detector that only cares about the google 429 string
        def is_rate_limit(e):
            return "429" in str(e)

        result = manager.execute_with_rotation(mock_operation, is_rate_limit)
        assert result == "success with goodkey"
        # Invalid keys should be removed, ratelimited key retained
        assert manager.available_keys == ["ratelimited", "goodkey"]


class TestSuccessfulKeyPersistsAcrossExecutions:
    def test_successful_key_prioritized_after_previous_success(self):
        """
        First run:
          key1 -> 429
          key2 -> success
        Second run should start with key2 immediately.
        """
        manager = KeyRotationManager(["key1", "key2"])

        call_log = []

        def operation_first_run(key):
            call_log.append(key)
            if key == "key1":
                raise Exception("429 rate limit")
            return f"success with {key}"

        def is_rate_limit(e):
            return "429" in str(e)

        # First execution: should rotate to key2
        result1 = manager.execute_with_rotation(operation_first_run, is_rate_limit)
        assert result1 == "success with key2"
        assert call_log == ["key1", "key2"]

        # Reset external call log
        call_log.clear()

        # Second execution: should start immediately with key2 (no rotation)
        def operation_second_run(key):
            call_log.append(key)
            return f"success again with {key}"

        result2 = manager.execute_with_rotation(operation_second_run, is_rate_limit)
        assert result2 == "success again with key2"
        # Only one call because key2 tried first
        assert call_log == ["key2"]

