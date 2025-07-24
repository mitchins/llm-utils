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


def test_reasoning_not_implemented():
    client = MockLLMClient([], model_name="none")
    with pytest.raises(NotImplementedError):
        client.generate("something", reasoning=True)


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

def test_wildcard_prompt_only():
    # Entry only specifies prompt and response; model, system, temperature should act as wildcards
    responses = [
        {"prompt": "hello", "response": "world"}
    ]
    client = MockLLMClient(responses, model_name="any-model")
    # Any system and temperature should match
    assert client.generate("hello", system="sys1", temperature=0.1) == "world"
    assert client.generate("hello", system="different", temperature=0.9) == "world"

def test_wildcard_system_only():
    # Entry specifies system and prompt; model and temperature wildcards
    responses = [
        {"system": "SYS", "prompt": "foo", "response": "bar"}
    ]
    client = MockLLMClient(responses, model_name="mdl")
    assert client.generate("foo", system="SYS", temperature=0.0) == "bar"
    # Different model still matches
    client2 = MockLLMClient(responses, model_name="other")
    assert client2.generate("foo", system="SYS", temperature=1.0) == "bar"

def test_wildcard_temperature_only():
    # Entry specifies prompt and temperature; model and system wildcards
    responses = [
        {"prompt": "baz", "temperature": 0.5, "response": "qux"}
    ]
    client = MockLLMClient(responses, model_name="mdl", on_request=None)
    # Matching temperature
    assert client.generate("baz", system="", temperature=0.5) == "qux"
    # Different system and model but same temperature should match
    client3 = MockLLMClient(responses, model_name="x", on_request=None)
    assert client3.generate("baz", system="any", temperature=0.5) == "qux"

def test_no_match_raises_with_wildcards():
    # Entry wildcard on prompt only, but wrong prompt should raise
    responses = [
        {"prompt": "exact", "response": "value"}
    ]
    client = MockLLMClient(responses, model_name="mdl")
    with pytest.raises(LLMError):
        client.generate("other", system="SYS", temperature=0.5)

def test_wildcard_model_only():
    # Entry specifies only model and response; other fields act as wildcards
    responses = [
        {"model": "MOD", "response": "ok"}
    ]
    # Matching model
    client = MockLLMClient(responses, model_name="MOD")
    assert client.generate("any", system="sys", temperature=0.3) == "ok"
    # Non-matching model should raise
    client2 = MockLLMClient(responses, model_name="OTHER")
    with pytest.raises(LLMError):
        client2.generate("any", system="sys", temperature=0.3)