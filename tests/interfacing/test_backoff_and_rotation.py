import pytest
from unittest.mock import patch, MagicMock
from llm_utils.clients.google_genai_client import GoogleLLMClient, RateLimitExceeded

@pytest.fixture
def mock_genai_in_client(monkeypatch):
    """Mocks the genai library inside the google_genai_client module."""
    mock_genai = MagicMock()
    # Patch 'genai' in the module where it is imported and used
    monkeypatch.setattr("llm_utils.clients.google_genai_client.genai", mock_genai)
    return mock_genai

def test_key_rotation_with_backoff_and_retry(mock_genai_in_client):
    """
    Test that when all keys are exhausted, the client backs off and retries the key rotation.
    """
    api_keys = ["key1", "key2"]
    # Configure client to retry once after a 1-second backoff
    client = GoogleLLMClient(model="gemini-pro", api_key=api_keys, max_retries=1, retry_interval=1)

    # Mock the key rotation manager to always indicate that all keys have been exhausted.
    # This will be raised twice: once on the initial attempt, and once on the retry.
    client._key_rotation_manager.execute_with_rotation = MagicMock(
        side_effect=[
            RateLimitExceeded("All keys failed on first attempt"),
            RateLimitExceeded("All keys failed on second attempt"),
        ]
    )

    with patch("time.sleep", return_value=None) as mock_sleep:
        # We expect the final RateLimitExceeded to be raised after all retries are exhausted.
        with pytest.raises(RateLimitExceeded) as exc_info:
            client.generate("test prompt")

        # The generate method should call the key rotation manager, which fails.
        # It then sleeps, and calls it again, which also fails.
        # So, two calls to the key rotation manager.
        assert client._key_rotation_manager.execute_with_rotation.call_count == 2

        # It should sleep once between the two attempts.
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_once_with(1)

        # The final exception should be the one from the last attempt.
        assert "All keys failed on second attempt" in str(exc_info.value)
