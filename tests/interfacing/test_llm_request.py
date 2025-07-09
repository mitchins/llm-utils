import time
import httpx
import json
import pytest
import os
from llm_utils.interfacing.llm_request import OpenAILikeLLMClient
from llm_utils.interfacing.llm_request import (
    LLMTimeoutError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMAccessDeniedError,
    LLMInvalidRequestError,
    LLMUnexpectedResponseError
)


# Automatically isolate environment variables and patch httpx.Client for all tests in this module
@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    monkeypatch.delenv("LLM_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    # Do not override httpx.Client if all tests inject their own transport
    yield
    
class TestLLMClient:

    def test_default_values(self, monkeypatch):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "mock"}}]}))
        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))
        assert client.system_prompt == "You are a helpful and concise assistant."
        assert client.model == "qwen:14b"
        assert client.base_url == "http://localhost:1234/v1"

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("LLM_SYSTEM_PROMPT", "Test prompt.")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("LLM_BASE_URL", "http://test-server:8000")

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "mock"}}]}))

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        assert client.system_prompt == "Test prompt."
        assert client.model == "test-model"
        assert client.base_url == "http://test-server:8000"

    def test_init_override(self, monkeypatch):
        monkeypatch.setenv("LLM_SYSTEM_PROMPT", "Env prompt.")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_BASE_URL", "http://env-server")

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "mock"}}]}))

        client = OpenAILikeLLMClient(
            system_prompt="Manual prompt.",
            model="manual-model",
            base_url="http://manual-server",
            client=httpx.Client(transport=transport)
        )

        assert client.system_prompt == "Manual prompt."
        assert client.model == "manual-model"
        assert client.base_url == "http://manual-server"
        
    def test_chat_completion_happy_path(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured["body"] = body
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "Mocked response"}}]}
            )

        transport = httpx.MockTransport(handler)

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        prompt_text = "Tell me a joke."
        result = client.chat(prompt_text)

        assert result == "Mocked response"

        # Check the outgoing request content
        assert "messages" in captured["body"]
        assert captured["body"]["messages"][0]["role"] == "system"
        assert captured["body"]["messages"][1]["role"] == "user"
        assert captured["body"]["messages"][1]["content"] == prompt_text
        assert captured["body"]["model"] == "qwen:14b"


    def test_chat_completion_timeout(self, monkeypatch):
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Request timed out", request=request)

        transport = httpx.MockTransport(timeout_handler)

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        with pytest.raises(LLMTimeoutError):
            client.chat("Trigger timeout")
    
    def test_chat_completion_connection_error(self, monkeypatch):
        def broken_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection failed", request=request)

        transport = httpx.MockTransport(broken_handler)

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        with pytest.raises(LLMConnectionError):
            client.chat("Trigger connection error")

    def test_chat_completion_model_not_found(self, monkeypatch):
        def model_not_found_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=404,
                json={"error": {"message": 'model "gibberish-model-name" not found, try pulling it first'}}
            )

        transport = httpx.MockTransport(model_not_found_handler)

        client = OpenAILikeLLMClient(model="gibberish-model-name", client=httpx.Client(transport=transport))

        with pytest.raises(LLMModelNotFoundError):
            client.chat("Test with missing model")
            
    def test_chat_completion_access_denied(self, monkeypatch):
        def access_denied_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=403,
                json={"error": {"message": "Access denied: invalid API key"}}
            )

        transport = httpx.MockTransport(access_denied_handler)

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        with pytest.raises(LLMAccessDeniedError):
            client.chat("Trigger access denied")
            
    def test_chat_completion_invalid_request(self, monkeypatch):
        def invalid_request_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=400,
                json={"error": {"message": "Invalid request: malformed input"}}
            )

        transport = httpx.MockTransport(invalid_request_handler)

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        with pytest.raises(LLMInvalidRequestError):
            client.chat("Trigger invalid request")

    def test_chat_completion_unexpected_response(self, monkeypatch):
        def server_error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=500,
                json={"error": {"message": "Internal Server Error"}}
            )

        transport = httpx.MockTransport(server_error_handler)

        client = OpenAILikeLLMClient(client=httpx.Client(transport=transport))

        with pytest.raises(LLMUnexpectedResponseError) as exc_info:
            client.chat("Trigger internal server error")
        assert "Internal Server Error" in str(exc_info.value)