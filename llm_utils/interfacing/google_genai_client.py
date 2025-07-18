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
    """A client for Google's Generative AI models (Gemini)."""

    def __init__(self,
                 model=None,
                 api_key: Union[str, List[str]] = None,
                 timeout: int = 60,
                 max_output_tokens: int = 4096,
                 **kwargs):
        super().__init__(model)
        
        # Handle both single key and key rotation
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use the Google LLM client.")
        
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

    def generate(
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
            return f"An error occurred during generation: {e}"
