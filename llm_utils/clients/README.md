# LLM Interfacing

This directory contains the core clients for interacting with Large Language Models (LLMs). The design emphasizes resilience, flexibility, and ease of use.

## Core Components

- **`base.py`**: Defines the `BaseLLMClient`, an abstract base class that provides the foundation for all LLM clients in this library. It includes a built-in retry mechanism to handle rate limit errors, which is inherited by all subclasses.

- **`google_genai_client.py`**: Implements the `GoogleLLMClient` for interacting with Google's Gemini models. This client features a robust key rotation system that works in tandem with the base client's retry logic to maximize uptime.

- **`openai_client.py`**: Contains the `OpenAILikeLLMClient` for sending requests to any OpenAI-compatible endpoint. It also provides a command-line interface for quick, ad-hoc requests.

- **`key_rotation.py`**: Implements the `KeyRotationManager`, which is used by the `GoogleLLMClient` to cycle through a list of API keys.

- **`mock_client.py`**: Provides a `MockLLMClient` for use in testing. It can be pre-loaded with a set of expected responses, allowing you to test your application without making live API calls.

## Usage

For detailed usage examples, please refer to the main `README.md` file in the root of the repository.
