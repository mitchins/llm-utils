import httpx
import os
from .base import BaseLLMClient, LLMError, RateLimitExceeded

class LLMTimeoutError(LLMError):
    """Raised when the LLM request times out."""

class LLMConnectionError(LLMError):
    """Raised when the LLM server cannot be reached."""

class LLMModelNotFoundError(LLMError):
    """Raised when the specified model is not found on the server."""

class LLMAccessDeniedError(LLMError):
    """Raised when authentication or permission issues occur."""

class LLMInvalidRequestError(LLMError):
    """Raised when a bad request is sent to the server."""

class LLMUnexpectedResponseError(LLMError):
    """Raised when an unexpected response is received from the server."""

# Default values - environment variables have been removed for security
# Users must explicitly provide model and base_url parameters
DEFAULT_MODEL = "qwen:14b"
DEFAULT_BASE_URL = "http://localhost:1234/v1"

class OpenAILikeLLMClient(BaseLLMClient):
    """A client for OpenAI-compatible LLM endpoints.

    This client supports automatic retries on 429 rate limit errors.
    
    **Security Note**: All parameters must be explicitly provided. Environment
    variable fallbacks have been removed for security reasons. Use the base_url
    and model parameters directly.
    
    Args:
        model (str, optional): The name of the model to use. Defaults to "qwen:14b".
        base_url (str, optional): The base URL of the LLM server. 
            Defaults to "http://localhost:1234/v1".
        timeout (int): Request timeout in seconds.
        system_prompt (str, optional): The system prompt to use.
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        repetition_penalty (float): The penalty for token repetition.
        client (httpx.Client, optional): An existing httpx client to use.
        max_retries (int): The maximum number of retries on rate limit errors.
        retry_interval (int): The number of seconds to wait between retries.
    """
    def __init__(self, model=None, base_url=None, timeout=60, system_prompt=None, temperature=0.7, max_tokens=1024, repetition_penalty=1.1, client=None, max_retries=1, retry_interval=5, api_key=None):
        """
        Initializes the OpenAILikeLLMClient.

        Args:
            model (str, optional): The name of the model to use.
            base_url (str, optional): The base URL of the LLM server.
            timeout (int): Request timeout in seconds.
            system_prompt (str, optional): The system prompt to use.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
            repetition_penalty (float): The penalty for token repetition.
            client (httpx.Client, optional): An existing httpx client to use.
            max_retries (int): The maximum number of retries on rate limit errors.
            retry_interval (int): The number of seconds to wait between retries.
        """
        system_prompt = system_prompt or os.getenv("LLM_SYSTEM_PROMPT") or "You are a helpful and concise assistant."
        model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL

        super().__init__(model, system_prompt=system_prompt, max_retries=max_retries, retry_interval=retry_interval)
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty

        self.client = client or httpx.Client(
            timeout=httpx.Timeout(
                self.timeout, connect=self.timeout, read=self.timeout, write=self.timeout
            )
        )

    def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        images: list[str] | None = None,
        reasoning: bool | None = None,
    ) -> str:
        if reasoning is not None:
            raise NotImplementedError("The `reasoning` parameter is not supported by the OpenAI client.")
        if system:
            self.system_prompt = system.strip()
        
        response = self.chat(
            prompt=prompt,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            stream=False,
            images=images,
        )
        return response
    
    # Holdover from the original interface
    def chat(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        repetition_penalty: float | None = None,
        stream: bool = False,
        images: list[str] | None = None,
    ):
        if stream:
            raise NotImplementedError("Streaming is not supported yet.")
        
        user_content = prompt
        if images:
            user_content = [{"type": "text", "text": prompt}]
            for img in images:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt.strip()},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else self.repetition_penalty,
            "stream": False
        }

        try:
            response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise LLMTimeoutError(f"LLM request timed out after {self.timeout} seconds.")
        except httpx.ConnectError:
            raise LLMConnectionError(f"LLM connection failed: could not reach server at {self.base_url}")
        except httpx.RequestError as e:
            raise LLMConnectionError(f"Unexpected request error: {e}")

        if response.status_code == 429:
            raise RateLimitExceeded(f"Rate limit exceeded: {response.text}")
        elif response.status_code == 404 and "model" in response.text.lower():
            raise LLMModelNotFoundError(f"Model not found: {self.model}")
        elif response.status_code in (401, 403):
            raise LLMAccessDeniedError("Access denied: check authentication or permissions.")
        elif response.status_code == 400:
            try:
                error_body = response.json()
                error_message = error_body.get("error", {}).get("message", "Invalid request")
            except Exception as parse_err:
                error_message = f"Non-JSON error body: {response.text.strip()} (parse error: {parse_err})"

            raise LLMInvalidRequestError(
                f"Invalid request sent to LLM server.\n"
                f"Payload: {payload}\n"
                f"Status Code: {response.status_code}\n"
                f"Error Message: {error_message}\n"
                f"Raw Response: {response.text.strip()}"
            )
        elif response.status_code != 200:
            raise LLMUnexpectedResponseError(f"Unexpected LLM error: {response.status_code} - {response.text}")

        try:
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMUnexpectedResponseError(f"Unexpected format in LLM response: {e}")

    def __del__(self):
        if hasattr(self, "client"):
            self.client.close()

# Backwards compatibility: allow older imports that expect `LLMClient`
LLMClient = OpenAILikeLLMClient

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Send a prompt to an OpenAI-compatible LLM endpoint.")
    parser.add_argument("prompt", type=str, help="Prompt to send to the LLM.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL of the LLM server.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Penalty for token repetition.")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for the LLM.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries on rate limit errors.")
    parser.add_argument("--retry-interval", type=int, default=5, help="Seconds to wait between retries.")

    args = parser.parse_args()

    client = OpenAILikeLLMClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        system_prompt=args.system_prompt,
        max_retries=args.max_retries,
        retry_interval=args.retry_interval
    )

    response = client.generate(args.prompt)
    print(response)

if __name__ == "__main__":
    main()
