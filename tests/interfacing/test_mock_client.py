import json
import pytest
from llm_utils.interfacing.mock_client import MockLLMClient
from llm_utils.interfacing.base_client import LLMError

IMG_B64 = "AAA="


def test_generate_from_list():
    responses = [
        {
            "model": "m1",
            "system": "sys",
            "prompt": "hi",
            "temperature": 0.0,
            "response": "hello"
        }
    ]
    client = MockLLMClient(responses, model_name="m1")
    assert client.generate("hi", system="sys", temperature=0.0, images=[IMG_B64]) == "hello"


def test_generate_from_file(tmp_path):
    data = [
        {"model": "m2", "system": "", "prompt": "ping", "temperature": 0.5, "response": "pong"}
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    client = MockLLMClient(str(path), model_name="m2")
    assert client.generate("ping", temperature=0.5) == "pong"


def test_unmatched_request_raises():
    client = MockLLMClient([], model_name="none")
    with pytest.raises(LLMError):
        client.generate("something")
