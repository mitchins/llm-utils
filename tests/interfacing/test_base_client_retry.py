import pytest
import time
from unittest.mock import MagicMock, patch
from llm_utils.clients.base import BaseLLMClient, RateLimitExceeded

# A concrete implementation of the abstract BaseLLMClient for testing purposes
class ConcreteLLMClient(BaseLLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generate_mock = MagicMock()

    def _generate(self, prompt: str, system: str = "", temperature: float = 0.0, images: list[str] | None = None) -> str:
        return self._generate_mock(prompt, system=system, temperature=temperature, images=images)

def test_generate_no_retry_on_success():
    client = ConcreteLLMClient(max_retries=3, retry_interval=1)
    client._generate_mock.return_value = "Success"

    response = client.generate("test prompt")

    assert response == "Success"
    client._generate_mock.assert_called_once()

@patch("time.sleep", return_value=None)
def test_generate_retries_on_rate_limit_exceeded(mock_sleep):
    client = ConcreteLLMClient(max_retries=3, retry_interval=1)
    client._generate_mock.side_effect = [
        RateLimitExceeded("Rate limit exceeded"),
        RateLimitExceeded("Rate limit exceeded"),
        "Success"
    ]

    response = client.generate("test prompt")

    assert response == "Success"
    assert client._generate_mock.call_count == 3
    assert mock_sleep.call_count == 2

@patch("time.sleep", return_value=None)
def test_generate_raises_error_after_max_retries(mock_sleep):
    client = ConcreteLLMClient(max_retries=2, retry_interval=1)
    client._generate_mock.side_effect = [
        RateLimitExceeded("Rate limit exceeded"),
        RateLimitExceeded("Rate limit exceeded"),
        RateLimitExceeded("Rate limit exceeded"),
        "Success"  # This should not be reached
    ]

    with pytest.raises(RateLimitExceeded):
        client.generate("test prompt")

    assert client._generate_mock.call_count == 3
    assert mock_sleep.call_count == 2

@patch("time.sleep", return_value=None)
def test_retry_interval_is_respected(mock_sleep):
    retry_interval = 5
    client = ConcreteLLMClient(max_retries=1, retry_interval=retry_interval)
    client._generate_mock.side_effect = [RateLimitExceeded("Rate limit exceeded"), "Success"]

    client.generate("test prompt")

    mock_sleep.assert_called_once_with(retry_interval)
