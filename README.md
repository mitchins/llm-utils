# llm-utils
Training and interface scripts for llms that are general purpose

## MockLLMClient

`MockLLMClient` is a lightweight testing client that returns predefined
responses instead of contacting a real model. It can be initialised with a list
of records or a path to a JSON file:

```python
from llm_utils.interfacing import MockLLMClient

records = [
    {
        "model": "qwen:14b",
        "system_prompt": "You are a helpful and concise assistant.",
        "user_prompt": "Hello",
        "temperature": 0.7,
        "max_tokens": 1024,
        "repetition_penalty": 1.1,
        "response": "Hi there!"
    }
]

client = MockLLMClient(records)
print(client.chat("Hello"))  # -> "Hi there!"
```

Every field used in a request must match a record; otherwise an
`LLMRequestMismatchError` is raised.
