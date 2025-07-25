"""
API Key rotation management for LLM clients.

This module provides a SOLID-compliant key rotation system that:
- Follows Single Responsibility Principle: KeyRotationManager only handles key rotation
- Follows Open/Closed Principle: Extensible via KeyRotationStrategy interface
- Follows Liskov Substitution Principle: Any KeyRotationStrategy can be substituted
- Follows Interface Segregation Principle: Clean, focused interfaces
- Follows Dependency Inversion Principle: Depends on abstractions, not concretions
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Iterator, Callable, Any
import logging

from google.api_core.exceptions import InvalidArgument

logger = logging.getLogger(__name__)


class KeyRotationStrategy(ABC):
    """Abstract strategy for key rotation behavior."""
    
    @abstractmethod
    def get_next_key(self, keys: List[str], current_key: Optional[str] = None) -> Optional[str]:
        """Get the next key to try."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state."""
        pass


class CircularRotationStrategy(KeyRotationStrategy):
    """Circular rotation strategy that remembers the last working key."""
    
    def __init__(self):
        self._last_working_index: Optional[int] = None
        self._attempted_indices: set = set()
        self._current_start_index: Optional[int] = None
    
    def get_next_key(self, keys: List[str], current_key: Optional[str] = None) -> Optional[str]:
        if not keys:
            return None
        
        # Set the starting index for this rotation cycle
        if self._current_start_index is None:
            if self._last_working_index is not None:
                self._current_start_index = self._last_working_index
            else:
                self._current_start_index = 0
        
        # Try all keys starting from the preferred position
        for i in range(len(keys)):
            index = (self._current_start_index + i) % len(keys)
            if index not in self._attempted_indices:
                self._attempted_indices.add(index)
                return keys[index]
        
        return None
    
    def mark_key_working(self, keys: List[str], working_key: str) -> None:
        """Mark a key as working for future prioritization."""
        try:
            self._last_working_index = keys.index(working_key)
        except ValueError:
            # Key not in list, ignore
            pass
    
    def reset(self) -> None:
        """Reset attempted indices for a new rotation cycle."""
        self._attempted_indices.clear()
        self._current_start_index = None


class KeyRotationManager:
    """
    Manages API key rotation with configurable strategy.
    
    Features:
    - Remembers last working key for efficiency
    - Exhaustive retry before giving up
    - Configurable rotation strategy
    - Thread-safe design considerations
    """
    
    def __init__(self, 
                 keys: List[str], 
                 strategy: Optional[KeyRotationStrategy] = None):
        if not keys:
            raise ValueError("At least one API key must be provided")
        
        self._keys = list(keys)  # Copy to avoid external mutation
        self._strategy = strategy or CircularRotationStrategy()
        self._current_key: Optional[str] = None
        
    @property
    def current_key(self) -> Optional[str]:
        """Get the current active key."""
        return self._current_key
    
    @property
    def available_keys(self) -> List[str]:
        """Get a copy of available keys."""
        return list(self._keys)
    
    def get_initial_key(self) -> str:
        """Get the initial key to start with."""
        self._strategy.reset()
        key = self._strategy.get_next_key(self._keys)
        if key is None:
            raise ValueError("No keys available")
        self._current_key = key
        return key
    
    def get_next_key(self) -> Optional[str]:
        """
        Get the next key in rotation.
        
        Returns:
            Next key to try, or None if all keys exhausted.
        """
        key = self._strategy.get_next_key(self._keys, self._current_key)
        if key is not None:
            self._current_key = key
        return key
    
    def mark_key_success(self, key: str) -> None:
        """Mark a key as successfully working."""
        if isinstance(self._strategy, CircularRotationStrategy):
            self._strategy.mark_key_working(self._keys, key)

    def remove_key(self, key: str) -> None:
        """Remove a permanently invalid key and reset rotation."""
        try:
            self._keys.remove(key)
            logger.warning(f"Removed invalid API key: {key[:8]}...")
        except ValueError:
            pass
        self.reset_rotation()
    
    def reset_rotation(self) -> None:
        """Reset rotation state for a fresh cycle."""
        self._strategy.reset()
    
    def execute_with_rotation(self, 
                            operation: Callable[[str], Any], 
                            is_rate_limit_error: Callable[[Exception], bool]) -> Any:
        """
        Execute an operation with automatic key rotation on rate limit errors.
        
        Args:
            operation: Function that takes an API key and returns a result
            is_rate_limit_error: Function that determines if an exception is a rate limit error
            
        Returns:
            Result of the successful operation
            
        Raises:
            RateLimitExceeded: If all keys are exhausted
            Any other exception from the operation
        """
        from .base import RateLimitExceeded
        from google.api_core.exceptions import InvalidArgument

        # Reset for fresh attempt
        total_keys = len(self._keys)
        self.reset_rotation()

        # Try initial key
        current_key = self.get_initial_key()
        last_rate_limit_error = None

        while current_key is not None:
            try:
                logger.debug(f"Attempting operation with key: {current_key[:8]}...")
                result = operation(current_key)

                # Success! Remember this key for next time
                self.mark_key_success(current_key)
                logger.debug(f"Operation succeeded with key: {current_key[:8]}...")
                return result

            except Exception as e:
                # Handle invalid API key errors by removing key
                if isinstance(e, InvalidArgument) or 'API key not valid' in str(e):
                    logger.error(f"Invalid API key {current_key[:8]}..., removing from rotation")
                    self.remove_key(current_key)
                    last_rate_limit_error = e
                    current_key = self.get_next_key()
                    continue
                # Handle rate limit errors
                if is_rate_limit_error(e):
                    logger.debug(f"Rate limit hit with key: {current_key[:8]}..., trying next key")
                    last_rate_limit_error = e
                    current_key = self.get_next_key()
                    continue
                # Non-retryable error
                raise

        # All keys exhausted
        logger.warning("All API keys exhausted due to rate limiting")
        if last_rate_limit_error:
            # Distinguish between invalid keys and rate-limit exhaustion
            if isinstance(last_rate_limit_error, InvalidArgument) or 'API key not valid' in str(last_rate_limit_error):
                from .base import NoValidAPIKeysError
                raise NoValidAPIKeysError(f"All {total_keys} API keys invalid: {last_rate_limit_error}")
            else:
                raise RateLimitExceeded(f"All {total_keys} API keys exhausted: {last_rate_limit_error}")
        else:
            raise RateLimitExceeded(f"All {total_keys} API keys exhausted")