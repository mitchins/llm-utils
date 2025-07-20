# mc-llm-utils

Lightweight utilities for LLM data parsing, training, and inference, with a focus on resilience and ease of use.

## Features

- **Resilient LLM Clients**: Built-in support for rate limit handling, including:
  - **Automatic Retries**: Automatically retries requests on 429 errors with a configurable backoff interval.
  - **API Key Rotation**: Seamlessly rotates through a list of API keys to maximize uptime and avoid rate limits.
  - **Explicit Security**: No silent environment variable dependencies - all API keys must be explicitly provided.
- **Flexible Training Scripts**: Fine-tune common models like T5 and BERT with scripts that expose key hyperparameters.
- **Data Preparation Tools**: A collection of scripts to analyze, sample, and prepare your datasets for training.
- **OpenAI & Google Gemini Support**: Out-of-the-box clients for both OpenAI-compatible endpoints and Google's Generative AI.

## Installation

Install the library and its dependencies using pip. Optional extras are available for training, testing, and Google Gemini support.

```bash
# Minimal installation
pip install mc-llm-utils

# With training dependencies
pip install mc-llm-utils[training]

# With Google Gemini support
pip install mc-llm-utils[google]

# For development and testing
pip install -e .[training,test]
```

## Usage

### Command-Line Interface

The `llm-request` command provides a simple way to query any OpenAI-compatible LLM endpoint. It now supports automatic retries on rate limit errors.

```bash
llm-request "Translate to French: Hello, world!" \
  --model my-model \
  --base-url http://localhost:8000 \
  --max-retries 3 \
  --retry-interval 10
```

### Google Gemini Client

The `GoogleLLMClient` provides access to Google's Gemini models and includes built-in support for API key rotation and backoff.

**Security Note**: API keys must be explicitly provided as parameters. Environment variable fallbacks have been removed for security reasons.

#### Single API Key Usage

```python
from llm_utils.clients.google_genai_client import GoogleLLMClient

# Single API key
client = GoogleLLMClient(
    model="gemini-pro", 
    api_key="your-api-key-here",
    max_retries=2, 
    retry_interval=5
)

response = client.generate("What is the capital of France?")
print(response)
```

#### API Key Rotation for High Availability

For production environments, provide multiple API keys to enable automatic rotation on rate limits:

```python
from llm_utils.clients.google_genai_client import (
    GoogleLLMClient, 
    GeminiContentBlockedException, 
    GeminiTokenLimitException
)

# Multiple API keys for rotation
client = GoogleLLMClient(
    model="gemini-pro",
    api_key=["key1", "key2", "key3"],  # Rotates through keys on rate limits
    max_retries=2,
    retry_interval=5
)

try:
    # Basic text response (backward compatibility)
    response = client.generate("What is the capital of France?")
    print(response)  # Just the text
    
    # Detailed response with full metadata
    detailed_response = client.generate_detailed("What is the capital of France?")
    print(f"Text: {detailed_response.text}")
    print(f"Input tokens: {detailed_response.usage_metadata.input_tokens}")
    print(f"Output tokens: {detailed_response.usage_metadata.output_tokens}")
    print(f"Finish reason: {detailed_response.candidates[0].finish_reason}")
    
except GeminiContentBlockedException as e:
    print(f"Content blocked: {e}")  # Clear message with token count
    # Example: "Content was blocked due to safety guidelines (input: 150 tokens)"
    
except GeminiTokenLimitException as e:
    print(f"Token limit exceeded: {e}")  # Clear message with usage details
    # Example: "Response truncated due to token limit (input: 2000 tokens, output: 4096 tokens, total: 6096 tokens) - consider reducing input size or increasing max_tokens"
    
# Full details available in e.response for both exception types if needed for debugging
```

**Key Rotation Behavior**: When rate limits are encountered, the client automatically rotates to the next available API key. If all keys are exhausted, it waits `retry_interval` seconds before retrying the entire rotation cycle.

**Response Metadata**: Use `generate_detailed()` to access comprehensive response information including:
- **Usage Statistics**: Token counts for cost tracking and optimization
- **Safety Ratings**: Per-category safety assessments 
- **Finish Reasons**: Why generation stopped (STOP, MAX_TOKENS, SAFETY, etc.)
- **Multiple Candidates**: Access all response candidates with their metadata

**Important: Token Limits vs Context Windows**:
- `max_output_tokens` controls **response length only** (default: 65,535 tokens)
- **Input context window** is separate and much larger:
  - Gemini 2.5 Pro: 1,048,576 input tokens (1M+)
  - Gemini 1.5 Pro/Flash: 1,000,000+ input tokens
  - Gemini 1.0 Pro: ~32,000 input tokens
- Large inputs (70K+ tokens) are supported by modern Gemini models

## Contributing

Contributions are welcome! To get started:

1.  **Install Dependencies**: For running the full test suite, you'll need the `training` and `test` extras.

    ```bash
    pip install -e .[training,test]
    pip install evaluate
    ```

2.  **Run Tests**: Use `pytest` to run the test suite.

    ```bash
    pytest
    ```

## Core Components

The repository is organized into the following key areas:

- **`llm_utils/clients/`**: Contains the `BaseLLMClient`, `GoogleLLMClient`, and `OpenAILikeLLMClient` for interacting with LLMs.
- **`llm_utils/training/`**: Includes scripts for fine-tuning models like T5 and BERT.
- **`llm_utils/data/`**: Provides tools for dataset analysis and preparation.
- **`tests/`**: Contains unit and integration tests for the library.

