import json
import pytest
from llm_utils.clients.mock_client import MockLLMClient, MockError
from llm_utils.clients.base import LLMError

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


# Additional tests
def test_exception_in_responses_raises():
    # If the sole response is an exception, generate should raise it immediately
    err = MockError("boom")
    client = MockLLMClient([err], model_name="m")
    with pytest.raises(MockError) as excinfo:
        client.generate("any", system="", temperature=0.0)
    assert str(excinfo.value) == "boom"

def test_prompt_alias_raises_exception():
    # prompt() alias should also raise the exception
    err = MockError("kaboom")
    client = MockLLMClient([err], model_name="m")
    with pytest.raises(MockError) as excinfo:
        client.prompt("any", system="", temperature=0.0)
    assert str(excinfo.value) == "kaboom"

def test_on_request_override_and_none_fallback():
    # on_request override takes precedence
    def cb(prompt, system, temperature, images=None):
        return "override"
    client = MockLLMClient([], model_name="m", on_request=cb)
    assert client.generate("x", system="s", temperature=1.0) == "override"
    # prompt alias also works
    assert client.prompt("y", system="t", temperature=2.0) == "override"
    # if on_request returns None, falls back to mapping
    data = [{"model":"m", "system":"t", "prompt":"z", "temperature":2.0, "response":"mapped"}]
    client2 = MockLLMClient(data, model_name="m", on_request=lambda *args, **kwargs: None)
    assert client2.generate("z", system="t", temperature=2.0) == "mapped"
