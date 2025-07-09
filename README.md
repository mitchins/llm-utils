# llm-utils
Training and interface scripts for llms that are general purpose

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
