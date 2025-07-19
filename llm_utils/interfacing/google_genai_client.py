import os
import logging
import base64
from typing import Union, List
from llm_utils.interfacing.base_client import BaseLLMClient, RateLimitExceeded
from llm_utils.interfacing.key_rotation import KeyRotationManager

try:
    import google.generativeai as genai
    from google.generativeai import types
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    # Warn only
    logger = logging.getLogger(__name__)
    logger.warning("Google Generative AI library is not installed. Some features may be unavailable.")

logger = logging.getLogger(__name__)

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
    
    **Thread Safety**: 
    Each instance maintains its own key rotation state. Multiple instances
    can safely use the same or overlapping key sets without interference.
    
    Example:
        >>> # Single key usage
        >>> client = GoogleLLMClient(model="gemini-pro", api_key="your-key")
        >>> response = client.generate("Hello world")
        
        >>> # Multiple keys for rotation
        >>> client = GoogleLLMClient(
        ...     model="gemini-pro", 
        ...     api_key=["key1", "key2", "key3"],
        ...     max_retries=2,
        ...     retry_interval=10
        ... )
        >>> response = client.generate("Hello world")
    """

    def __init__(self,
                 model: str,
                 api_key: Union[str, List[str]],
                 timeout: int = 60,
                 max_output_tokens: int = 4096,
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
            max_output_tokens (int): The maximum number of tokens to generate. Defaults to 4096.
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
        genai.configure(api_key=current_key)
        
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        return (hasattr(exception, 'status_code') and exception.status_code == 429) or \
               "429" in str(exception) or "rate limit" in str(exception).lower()

    def _execute_generation(self, api_key: str, prompt: str, system: str, 
                          temperature: float, images: list[str] | None) -> str:
        """Execute the actual generation with a specific API key."""
        # Configure the API key for this request
        genai.configure(api_key=api_key)
        
        model_instance = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system if system else None,
        )

        generation_config = types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_output_tokens,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        contents = [prompt]
        if images:
            for img in images:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64decode(img),
                    }
                })

        response = model_instance.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        if response.parts:
            return response.text
        else:
            finish_reason = "Unknown"
            if response.prompt_feedback:
                finish_reason = response.prompt_feedback.block_reason
            return f"Response was blocked due to: {finish_reason}"

    def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
    ) -> str:
        if not prompt or len(prompt) == 0:
            raise ValueError("Prompt must not be empty for GoogleLLMClient.")
        if not self.model:
            raise ValueError("Model name must be set for GoogleLLMClient.")

        # If we have key rotation enabled, use it
        if self._key_rotation_manager:
            return self._key_rotation_manager.execute_with_rotation(
                operation=lambda key: self._execute_generation(key, prompt, system, temperature, images),
                is_rate_limit_error=self._is_rate_limit_error
            )
        
        # Single key mode - original behavior with improved error handling
        try:
            return self._execute_generation(self._keys[0], prompt, system, temperature, images)
        except Exception as e:
            # Check for rate limit errors (HTTP 429)
            if self._is_rate_limit_error(e):
                raise RateLimitExceeded(f"Google API rate limit exceeded: {e}")
            raise Exception(f"An error occurred during generation: {e}")
