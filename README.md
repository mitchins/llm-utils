# llm-utils
Training and interface scripts for LLMs that are general purpose. The library
contains helpers for preparing datasets, running model training and sending
requests to OpenAI-compatible endpoints.

## Installation

Clone the repository and install in editable mode so the command line scripts
are available:

```bash
pip install -e .
```

## Key scripts

`train-bert` trains a BERT style classifier or regressor. The script exposes
many command line arguments but the minimal invocation looks like:

```bash
train-bert --data-path training_data.jsonl --output-dir bert_model
```

`llm-request` provides a simple interface to query a running LLM server:

```bash
llm-request "Hello" --model my-model --base-url http://localhost:8000
```

## Running the tests

The test suite uses **pytest**. Install the optional test dependencies and run:

```bash
pytest
```


## MockLLMClient

`MockLLMClient` is a lightweight implementation of `BaseLLMClient` useful for unit tests. It loads predefined
responses either from a list of dictionaries or from a JSON file. Each entry describes the request and the
expected response.

Example JSON structure:

```json
[
  {
    "model": "test-model",
    "system": "Assistant",
    "prompt": "Hello",
    "temperature": 0.0,
    "response": "Hi there"
  }
]
```

Usage:

```python
from llm_utils.interfacing.mock_client import MockLLMClient

responses = [
    {
        "model": "test-model",
        "system": "Assistant",
        "prompt": "Hello",
        "temperature": 0.0,
        "response": "Hi there"
    }
]

client = MockLLMClient(responses, model_name="test-model")
text = client.generate("Hello", system="Assistant")
print(text)  # "Hi there"
```
