import json
import pytest
from llm_utils.interfacing.mock_llm_client import MockLLMClient, LLMRequestMismatchError


def make_record(prompt: str, response: str, model: str = "qwen:14b", system_prompt: str = "You are a helpful and concise assistant.", temperature: float = 0.7, max_tokens: int = 1024, repetition_penalty: float = 1.1):
    return {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
        "response": response,
    }


def test_match_from_data():
    data = [make_record("Tell me a joke", "Mocked joke")]
    client = MockLLMClient(data, model="qwen:14b")
    assert client.chat("Tell me a joke") == "Mocked joke"


def test_match_from_file(tmp_path):
    data = [make_record("Ping", "Pong")]
    p = tmp_path / "mocks.json"
    p.write_text(json.dumps(data))
    client = MockLLMClient(str(p), model="qwen:14b")
    assert client.chat("Ping") == "Pong"


def test_unmatched_request_raises():
    data = [make_record("A", "B")]
    client = MockLLMClient(data, model="qwen:14b")
    with pytest.raises(LLMRequestMismatchError):
        client.chat("Something else")
