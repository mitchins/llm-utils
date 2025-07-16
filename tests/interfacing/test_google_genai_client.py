import base64
import types
import pytest
import google.generativeai as genai
from llm_utils.interfacing.google_genai_client import GoogleLLMClient

IMG_B64 = "AAA="


class DummyResponse:
    def __init__(self):
        self.parts = [types.SimpleNamespace(text="ok")]
        self.text = "ok"
        self.prompt_feedback = None


class DummyModel:
    def __init__(self, model_name=None, system_instruction=None):
        self.model_name = model_name
        self.system_instruction = system_instruction

    def generate_content(self, contents, generation_config=None, safety_settings=None):
        DummyModel.captured = {
            "contents": contents,
            "generation_config": generation_config,
            "safety_settings": safety_settings,
        }
        return DummyResponse()


def test_generate_with_images(monkeypatch):
    monkeypatch.setattr(genai, "configure", lambda api_key=None: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda model_name, system_instruction=None: DummyModel(model_name, system_instruction))

    client = GoogleLLMClient(model="gemini", api_key="key")
    result = client.generate("Describe", images=[IMG_B64])

    assert result == "ok"
    captured = DummyModel.captured
    assert isinstance(captured["contents"], list)
    assert captured["contents"][0] == "Describe"
    assert base64.b64decode(IMG_B64) == captured["contents"][1]["inline_data"]["data"]

