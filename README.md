# mc-llm-utils
Training and interface scripts for LLMs that are general purpose. The library
contains helpers for preparing datasets, running model training and sending
requests to OpenAI-compatible endpoints.

## Capabilities

The repository is organised by feature area so you can quickly locate the
relevant utilities:

- **Data utilities** (`llm_utils/data/`)
  - `analyze_token_lengths.py` – check average token counts
  - `intersect_dataset.py` – find the overlap between datasets
  - `semantic_sample.py` – draw a semantically diverse subset
- **Training scripts** (`llm_utils/training/`)
  - `train_bert.py` and `train_bert_ner.py` – fine-tune BERT models
  - `train_t5.py` – fine-tune T5 style models
  - `prepare_dataset.py` – convert common formats
- **Inference tools** (`llm_utils/inference/`)
  - `run_bert.py` and `run_t5.py` – load a trained model for prediction
- **Interfacing helpers** (`llm_utils/interfacing/`)
  - `llm_request.py` – simple command line client
  - `google_genai_client.py` – Google Gemini implementation
  - `mock_client.py` – client for tests
  - `client_factory.py` and `base_client.py`
- **Output parsing** (`llm_utils/llm_parsers.py`)
  - Utilities to extract structured responses from models

Additional scripts such as `semantic_sample.py` at the repository root and the
`tests/` directory demonstrate typical usage of these tools.

## Installation

Clone the repository and install in editable mode so the command line scripts
are available. By default only the lightweight dependencies used by
`llm-request` are installed. Extra groups are provided for training, Google
Gemini support and test utilities.

```bash
pip install -e .            # minimal install with httpx only

# Install training extras
pip install -e .[training]

# Install Google Gemini extras
pip install -e .[google]

# Install test extras
pip install -e .[test]
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

The test suite uses **pytest**. Install the training and test extras so
heavy libraries like `datasets` are available:

```bash
pip install -e .[training,test]
pip install evaluate
pytest
```

## Google Gemini client

The `GoogleLLMClient` allows interfacing with Google's Generative AI models.
Install the optional extras and set `GEMINI_API_KEY`:

```bash
pip install -e .[google]
export GEMINI_API_KEY=your-key-here
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
