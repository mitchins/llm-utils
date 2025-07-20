# Agent Guidelines

This document provides guidance for agents (both human and automated) on how to interact with and contribute to the `mc-llm-utils` repository.

## Core Repository Structure

- **`llm_utils/clients/`**: LLM client implementations, including the base client with retry logic and the Google Gemini client with key rotation.
- **`llm_utils/training/`**: Scripts for model training and data preparation.
- **`tests/`**: Unit and integration tests. New features should be accompanied by corresponding tests.

## Key Feature: Resilient LLM Clients

A primary feature of this library is its resilience to rate limiting. This is achieved through two mechanisms:

1.  **Backoff and Retry**: The `BaseLLMClient` automatically retries requests that fail with a 429 status code. This is configured with the `max_retries` and `retry_interval` parameters.
2.  **API Key Rotation**: The `GoogleLLMClient` can be initialized with a list of API keys. If a request fails due to a rate limit, the client will automatically switch to the next available key.

These features work together to ensure that requests are handled as robustly as possible.

## Testing Guidelines

- **Full Test Suite**: To run the complete test suite, including tests for training and data utilities, install the `training` and `test` extras:

  ```bash
  pip install -e .[training,test]
  pip install evaluate
  pytest
  ```

- **Testing Resiliency Features**: To test the backoff and key rotation features, see the following test files:
  - `tests/interfacing/test_base_client_retry.py`
  - `tests/interfacing/test_google_genai_client_rotation.py`
  - `tests/interfacing/test_backoff_and_rotation.py`

## Development Workflow

1.  **Create a Virtual Environment**: Use a virtual environment to manage dependencies.
2.  **Install in Editable Mode**: Install the package in editable mode with the `test` and `training` extras.
3.  **Add New Features and Tests**: When adding a new feature, ensure that it is accompanied by corresponding tests.
4.  **Run Tests**: Run `pytest` to ensure that all tests pass.
5.  **Update Documentation**: Update the `README.md` and any other relevant documentation to reflect the changes.

