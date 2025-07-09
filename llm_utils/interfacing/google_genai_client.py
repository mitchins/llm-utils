

import os
import logging
from llm_utils.interfacing.base_client import BaseLLMClient

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

    def __init__(self, model_name=None):
        super().__init__(model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        else:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use the Google LLM client.")

    def generate(self, prompt, system="", temperature=0.0) -> str:
        if not self.model_name:
            raise ValueError("Model name must be set for GoogleLLMClient.")

        try:
            model_instance = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system if system else None,
            )

            generation_config = types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4096,
            )

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = model_instance.generate_content(
                contents=prompt,
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
        except Exception as e:
            return f"An error occurred during generation: {e}"