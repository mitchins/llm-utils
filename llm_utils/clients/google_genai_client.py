import logging
import base64
import threading
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
from llm_utils.clients.base import BaseLLMClient, LLMError, RateLimitExceeded
from llm_utils.clients.key_rotation import KeyRotationManager

logger = logging.getLogger(__name__)


def _extract_enum_value(obj, fallback_value: str = "UNKNOWN") -> str:
    """Extract string value from enum or return fallback."""
    if obj is None:
        return fallback_value
    if hasattr(obj, 'name'):
        return str(obj.name)
    return str(obj)



try:
    from google import genai
    from google.genai import types
    from google.genai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    logger.warning("google-genai library is not installed. Some features may be unavailable.")

# Gracefully handle missing google.api_core in test environments
try:
    from google.api_core.exceptions import ResourceExhausted, RetryError
except ImportError:  # fall back to dummy types
    class ResourceExhausted(Exception):
        pass
    class RetryError(Exception):
        last_exc: Exception | None = None

# genai uses global config; protect it in multithreaded scenarios
_genai_lock = threading.Lock()


class GeminiContentBlockedException(LLMError):
    """Raised when Gemini blocks content due to safety filters."""
    
    def __init__(self, message: str, response: 'GeminiResponse' = None):
        """
        Initialize with a clear message and optional detailed response for debugging.
        
        Args:
            message (str): Clear, user-friendly explanation of why content was blocked
            response (GeminiResponse, optional): Full response object for detailed analysis
        """
        super().__init__(message)
        self.response = response


class GeminiTokenLimitException(LLMError):
    """Raised when Gemini hits token limits during generation."""
    
    def __init__(self, message: str, response: 'GeminiResponse' = None):
        """
        Initialize with a clear message and optional detailed response for debugging.
        
        Args:
            message (str): Clear, user-friendly explanation of the token limit issue
            response (GeminiResponse, optional): Full response object with usage data
        """
        super().__init__(message)
        self.response = response


@dataclass
class GeminiUsageMetadata:
    """Usage statistics from Gemini API response."""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    @classmethod
    def from_response(cls, usage_metadata) -> 'GeminiUsageMetadata':
        """Create from Gemini response usage_metadata."""
        if usage_metadata is None:
            return cls()
        return cls(
            input_tokens=getattr(usage_metadata, 'prompt_token_count', None),
            output_tokens=getattr(usage_metadata, 'candidates_token_count', None),
            total_tokens=getattr(usage_metadata, 'total_token_count', None)
        )


@dataclass 
class GeminiSafetyRating:
    """Safety rating for a specific harm category."""
    category: str
    probability: str
    blocked: bool = False
    
    @classmethod
    def from_response(cls, rating) -> 'GeminiSafetyRating':
        """Create from Gemini response safety rating."""
        return cls(
            category=_extract_enum_value(rating.category),
            probability=_extract_enum_value(rating.probability),
            blocked=getattr(rating, 'blocked', False)
        )


@dataclass
class GeminiCandidate:
    """Candidate response from Gemini."""
    content: str
    finish_reason: Optional[str] = None
    safety_ratings: List[GeminiSafetyRating] = None
    index: int = 0
    
    def __post_init__(self):
        if self.safety_ratings is None:
            self.safety_ratings = []
    
    @classmethod
    def from_response(cls, candidate, index: int = 0) -> 'GeminiCandidate':
        """Create from Gemini response candidate."""
        content = ""
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                content = candidate.content.parts[0].text if candidate.content.parts[0].text else ""
        
        finish_reason = None
        if hasattr(candidate, 'finish_reason'):
            # Handle both enum and raw values
            if hasattr(candidate.finish_reason, 'name'):
                finish_reason = _extract_enum_value(candidate.finish_reason)
            else:
                # Convert integer codes to names
                reason_code = str(candidate.finish_reason)
                finish_reason = {
                    "1": "STOP",
                    "2": "MAX_TOKENS", 
                    "3": "SAFETY"
                }.get(reason_code, str(candidate.finish_reason))
        
        safety_ratings = []
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            safety_ratings = [GeminiSafetyRating.from_response(rating) for rating in candidate.safety_ratings]
        
        return cls(
            content=content,
            finish_reason=finish_reason,
            safety_ratings=safety_ratings,
            index=index
        )


@dataclass
class GeminiResponse:
    """Comprehensive response from Google Gemini API with all metadata."""
    text: str
    candidates: List[GeminiCandidate]
    usage_metadata: GeminiUsageMetadata
    prompt_feedback: Optional[Dict[str, Any]] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []
    
    @classmethod
    def from_response(cls, response) -> 'GeminiResponse':
        """Create comprehensive response from Gemini API response."""
        # Extract primary text safely (avoid response.text accessor when no parts)
        text = ""
        try:
            if hasattr(response, 'parts') and response.parts:
                text = response.parts[0].text if response.parts[0].text else ""
            elif hasattr(response, 'text'):
                text = response.text or ""
        except (ValueError, AttributeError):
            # Handle cases where response.text accessor fails due to no parts
            text = ""
        
        # Extract candidates with all metadata
        candidates = []
        if hasattr(response, 'candidates') and response.candidates:
            candidates = [GeminiCandidate.from_response(candidate, i) 
                         for i, candidate in enumerate(response.candidates)]
        
        # Extract usage metadata
        usage_metadata = GeminiUsageMetadata.from_response(
            getattr(response, 'usage_metadata', None)
        )
        
        # Extract prompt feedback
        prompt_feedback = None
        blocked = False
        block_reason = None
        
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            prompt_feedback = {
                'block_reason': getattr(response.prompt_feedback, 'block_reason', None),
                'safety_ratings': []
            }
            
            if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                blocked = True
                block_reason = _extract_enum_value(response.prompt_feedback.block_reason)
            
            if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings:
                prompt_feedback['safety_ratings'] = [
                    GeminiSafetyRating.from_response(rating) 
                    for rating in response.prompt_feedback.safety_ratings
                ]
        
        return cls(
            text=text,
            candidates=candidates,
            usage_metadata=usage_metadata,
            prompt_feedback=prompt_feedback,
            blocked=blocked,
            block_reason=block_reason
        )

class GoogleLLMClient(BaseLLMClient):
    """A client for Google's Generative AI models (Gemini) with built-in resilience features.

    This client provides robust error handling and high availability through:
    
    **API Key Management**:
    - Requires explicit API key provision (no environment variable fallbacks)
    - Supports single API key or multiple keys for rotation
    - Automatic key rotation on rate limit errors (429 status codes)
    
    **Error Handling & Retries**:
    - Automatic retries with configurable intervals
    - Rate limit detection and handling
    - Proper exception propagation instead of silent failures
    
    **Key Rotation Behavior**:
    When multiple API keys are provided:
    1. Starts with the first key in the list
    2. On rate limit error, immediately rotates to the next available key
    3. If all keys are exhausted, waits `retry_interval` seconds
    4. Retries the entire key rotation cycle up to `max_retries` times
    5. Raises RateLimitExceeded if all retries are exhausted
    
    **Token Limits**:
    - max_output_tokens controls RESPONSE length only (default: 65,535 tokens)
    - Input context window is separate and model-dependent:
      * Gemini 2.5 Pro: 1,048,576 input tokens (1M+)
      * Gemini 1.5 Pro/Flash: 1,000,000+ input tokens
      * Gemini 1.0 Pro: ~32,000 input tokens
    - Large inputs (70K+ tokens) are fine if using modern Gemini models
    
    **Thread Safety**: 
    Each instance maintains its own key rotation state. Multiple instances
    can safely use the same or overlapping key sets without interference.
    
    Example:
        >>> # Single key usage
        >>> client = GoogleLLMClient(model="gemini-2.5-pro", api_key="your-key")
        >>> response = client.generate("Hello world")
        
        >>> # Multiple keys for rotation with custom output limit
        >>> client = GoogleLLMClient(
        ...     model="gemini-2.5-pro", 
        ...     api_key=["key1", "key2", "key3"],
        ...     max_output_tokens=32000,  # Limit response size, not input
        ...     max_retries=2,
        ...     retry_interval=10
        ... )
        >>> response = client.generate("Very long input prompt...")
    """

    def __init__(self,
                 model: str,
                 api_key: Union[str, List[str]],
                 timeout: int = 60,
                 max_output_tokens: Optional[int] = None,
                 max_retries: int = 1,
                 retry_interval: int = 5,
                 **kwargs):
        """
        Initializes the GoogleLLMClient.

        Args:
            model (str): The name of the Gemini model to use (e.g., "gemini-pro").
            api_key (Union[str, List[str]]): A single API key or a list of API keys
                for rotation. This parameter is required and must be explicitly provided.
                When multiple keys are provided, the client will automatically rotate
                through them on rate limit errors to maximize uptime.
            timeout (int): Request timeout in seconds. Defaults to 60.
            max_output_tokens (int): The maximum number of tokens to generate in the response.
                This controls OUTPUT length only, not input context window. Defaults to 65535
                (Gemini 2.5 Pro's maximum). The model's input context window is separate and
                much larger (1M+ tokens for Gemini 2.5 Pro).
            max_retries (int): The maximum number of times to retry the request
                after all API keys have been exhausted. Defaults to 1.
            retry_interval (int): The number of seconds to wait between retries. Defaults to 5.
            
        Raises:
            ValueError: If api_key is None, empty, or not a string/list of strings.
            ImportError: If the google-generativeai library is not installed.
        """
        super().__init__(model, max_retries=max_retries, retry_interval=retry_interval)
        
        # Validate api_key parameter - no environment variable fallbacks
        if api_key is None:
            raise ValueError(
                "api_key parameter is required. Please provide either a single API key "
                "(string) or multiple API keys (list) for rotation. Environment variable "
                "fallbacks have been removed for security reasons."
            )
        
        if isinstance(api_key, str):
            # Single key mode - convert to list for consistency
            self._keys = [api_key]
            self._key_rotation_manager = None
        elif isinstance(api_key, list):
            if not api_key:
                raise ValueError("API key list cannot be empty")
            self._keys = api_key
            self._key_rotation_manager = KeyRotationManager(api_key)
        else:
            raise ValueError("api_key must be a string or list of strings")
        
        # Set initial key
        current_key = self._keys[0] if not self._key_rotation_manager else self._key_rotation_manager.get_initial_key()
        # Initialize the GenAI client
        self._client = genai.Client(api_key=current_key)
        
        self.timeout = timeout
        # Determine default max_output_tokens from model capabilities if not explicitly provided
        if max_output_tokens is None:
            if self.model.startswith("gemini-2.0"):
                self.max_output_tokens = 8192
            elif self.model.startswith("gemini-2.5"):
                self.max_output_tokens = 65535
            else:
                # Fallback default for unknown or future models
                self.max_output_tokens = 65535
        else:
            # Honor user-provided value (must not exceed model limits)
            self.max_output_tokens = max_output_tokens
    
    def _get_user_friendly_block_message(self, response: 'GeminiResponse') -> str:
        """Convert technical block reasons into clear, actionable user messages."""
        block_reason = response.block_reason
        
        # Build base message
        if not block_reason:
            base_message = "Content generation was blocked (no specific reason provided)"
        else:
            # Convert technical reasons to user-friendly messages
            reason_lower = str(block_reason).lower()
            
            if "safety" in reason_lower or "harm" in reason_lower:
                base_message = "Content was blocked due to safety guidelines"
            elif "recitation" in reason_lower:
                base_message = "Content was blocked due to potential copyright concerns"
            elif "other" in reason_lower:
                base_message = "Content was blocked for policy reasons"
            elif "prohibited" in reason_lower:
                base_message = "Content violates usage policies"
            else:
                # Fallback for unknown reasons
                base_message = f"Content was blocked: {block_reason}"
        
        # Add input token count if available
        if response.usage_metadata and response.usage_metadata.input_tokens is not None:
            base_message += f" (input: {response.usage_metadata.input_tokens} tokens)"
        
        return base_message
    
    def _get_token_limit_message(self, response: 'GeminiResponse') -> str:
        """Create clear message for token limit scenarios."""
        base_message = "Response truncated due to token limit"
        
        # Add token usage information if available
        if response.usage_metadata:
            if response.usage_metadata.input_tokens is not None:
                base_message += f" (input: {response.usage_metadata.input_tokens} tokens"
                if response.usage_metadata.output_tokens is not None:
                    base_message += f", output: {response.usage_metadata.output_tokens} tokens"
                if response.usage_metadata.total_tokens is not None:
                    base_message += f", total: {response.usage_metadata.total_tokens} tokens"
                base_message += ")"
        
        base_message += " - consider reducing input size or increasing max_tokens"
        return base_message

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate limit error (robust detection)."""
        # Unwrap google RetryError to inspect the underlying cause
        if isinstance(exc, RetryError) and getattr(exc, "last_exc", None):
            exc = exc.last_exc

        msg = str(exc).lower()
        return (
            getattr(exc, "status_code", None) == 429
            or getattr(exc, "code", None) == 429  # httpx / urllib style
            or isinstance(exc, ResourceExhausted)
            or "429" in msg
            or "too many requests" in msg
            or "rate limit" in msg
        )

    def _execute_generation(self, api_key: str, prompt: str, system: str, 
                          temperature: float, images: list[str] | None) -> str:
        """Execute generation and return text only (backward compatibility)."""
        response = self._execute_generation_detailed(api_key, prompt, system, temperature, images)
        return response.text
    
    def _execute_generation_detailed(self, api_key: str, prompt: str, system: str, 
                                   temperature: float, images: list[str] | None, reasoning: bool | None) -> GeminiResponse:
        """Execute generation and return comprehensive response with all metadata."""
        # Rotate API key by recreating the GenAI client under lock
        with _genai_lock:
            self._client = genai.Client(api_key=api_key)

        # Build generation configuration with safety settings included
        generation_config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=self.max_output_tokens,
            safety_settings=[
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )

        # Prepare contents for generation
        contents: List[Union[str, Dict[str, Any]]] = [prompt]
        if images:
            for img in images:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64decode(img),
                    }
                })

        # Handle reasoning parameter: tools=None disables reasoning, tools=[] still uses reasoning
        tools_param = None
        if reasoning is False:
            tools_param = None  # Explicitly disable reasoning
        elif reasoning is True or reasoning is None:
            # Default behavior (reasoning enabled) - don't pass tools parameter
            pass

        # Call the new GenAI client to generate content
        kwargs = {
            "model": self.model,
            "contents": contents,
            "config": generation_config,
        }
        if tools_param is not None:
            kwargs["tools"] = tools_param
            
        response = self._client.models.generate_content(**kwargs)

        # Create comprehensive response object
        gemini_response = GeminiResponse.from_response(response)
        
        # Check for different types of issues
        if gemini_response.blocked:
            # Content was blocked due to safety filters
            message = self._get_user_friendly_block_message(gemini_response)
            raise GeminiContentBlockedException(message, gemini_response)
        elif not gemini_response.text:
            # No text returned - check if it's due to token limits
            if (gemini_response.candidates and 
                len(gemini_response.candidates) > 0 and
                gemini_response.candidates[0].finish_reason == "MAX_TOKENS"):
                message = self._get_token_limit_message(gemini_response)
                raise GeminiTokenLimitException(message, gemini_response)
            
            # Some other reason for no text
            message = self._get_user_friendly_block_message(gemini_response)
            raise GeminiContentBlockedException(message, gemini_response)
            
        return gemini_response

    def generate_detailed(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
        reasoning: bool | None = None,
    ) -> GeminiResponse:
        """Generate response with comprehensive metadata including usage, safety ratings, and finish reasons.
        
        Args:
            prompt (str): The prompt text.
            system (str, optional): System message or instruction. Defaults to "".
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            images (list[str] | None, optional): Optional list of base64 encoded images.
            
        Returns:
            GeminiResponse: Comprehensive response with text, usage stats, safety ratings, etc.
            
        Raises:
            ValueError: If prompt is empty or model name is not set.
            RateLimitExceeded: If rate limits are exceeded across all API keys.
            Exception: If generation fails or content is blocked.
        """
        if not prompt or len(prompt) == 0:
            raise ValueError("Prompt must not be empty for GoogleLLMClient.")
        if not self.model:
            raise ValueError("Model name must be set for GoogleLLMClient.")

        # If we have key rotation enabled, use it
        if self._key_rotation_manager:
            total_keys = len(self._keys)
            attempt = 0          # 1‑based index within the current cycle
            cycle = 0            # how many full key‑set passes we’ve done
            def operation_with_logging(key):
                nonlocal attempt
                nonlocal cycle
                attempt += 1
                if attempt > total_keys:
                    cycle += 1
                    attempt = 1  # reset for the new cycle
                prefix = f"{key[:8]}…" if key else "UNKNOWN"
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Cycle {cycle} attempt {attempt}/{total_keys} using API key {prefix}"
                        f"{' (rotating after rate‑limit)' if cycle or attempt > 1 else ''}"
                    )
                if cycle or attempt > 1:
                    logger.info(f"Rotating API key {attempt}/{total_keys}")
                try:
                    return self._execute_generation_detailed(
                        key, prompt, system, temperature, images, reasoning
                    )
                except Exception as e:
                    if self._is_rate_limit_error(e):
                        logger.debug(f"GenAI rate-limit error with key {prefix}: {e}", exc_info=True)
                    raise
            return self._key_rotation_manager.execute_with_rotation(
                operation=operation_with_logging,
                is_rate_limit_error=self._is_rate_limit_error
            )
        
        # Single key mode
        try:
            return self._execute_generation_detailed(self._keys[0], prompt, system, temperature, images, reasoning)
        except Exception as e:
            # Check for rate limit errors (HTTP 429) and log full exception
            if self._is_rate_limit_error(e):
                logger.debug(f"GenAI rate-limit error (single-key): {e}", exc_info=True)
                raise RateLimitExceeded(f"Google API rate limit exceeded: {e}")
            raise e
    
    def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
        reasoning: bool | None = None,
    ) -> str:
        """Generate text response (backward compatibility). Use generate_detailed() for full metadata."""
        response = self.generate_detailed(prompt, system, temperature, images, reasoning)
        return response.text
