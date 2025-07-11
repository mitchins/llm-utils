import sys
import types
import importlib
import pytest


@pytest.fixture
def process_batch(monkeypatch):
    dummy_datasets = types.SimpleNamespace(
        load_dataset=lambda *a, **k: None,
        Dataset=lambda *a, **k: None,
        concatenate_datasets=lambda *a, **k: None,
    )
    dummy_transformers = types.SimpleNamespace(
        AutoTokenizer=lambda *a, **k: None,
        PreTrainedTokenizer=object,
    )
    dummy_tqdm = types.SimpleNamespace(tqdm=lambda x, **k: x)
    monkeypatch.setitem(sys.modules, "datasets", dummy_datasets)
    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)
    monkeypatch.setitem(sys.modules, "tqdm", dummy_tqdm)
    mod = importlib.import_module("llm_utils.training.prepare_dataset")
    return mod.process_and_filter_batch


class DummyTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id

    def __call__(self, text, truncation=False, add_special_tokens=False):
        tokens = text.split()
        return {"input_ids": list(range(len(tokens))), "attention_mask": [1] * len(tokens)}


def test_filter_min_length(process_batch):
    batch = {"input": ["one two", "one two three"], "output": ["a b", "c d e"]}
    tokenizer = DummyTokenizer()
    result = process_batch(batch, tokenizer, min_length=3)
    assert len(result) == 1
    assert result[0]["input_ids"] == [0, 1, 2]
    assert result[0]["labels"] == [0, 1, 2]


def test_filter_max_length(process_batch):
    batch = {"input": ["one two three", "one"], "output": ["a b c", "b"]}
    tokenizer = DummyTokenizer()
    result = process_batch(batch, tokenizer, max_length=1)
    assert len(result) == 1
    assert result[0]["input_ids"] == [0]
    assert result[0]["labels"] == [0]


def test_eos_added_in_t5_mode(process_batch):
    batch = {"input": ["hi"], "output": ["there"]}
    tokenizer = DummyTokenizer(eos_token_id=99)
    result = process_batch(batch, tokenizer, mode="T5")
    assert result[0]["labels"] == [0, 99]


def test_no_eos_in_gpt_mode(process_batch):
    batch = {"input": ["hi"], "output": ["there"]}
    tokenizer = DummyTokenizer(eos_token_id=99)
    result = process_batch(batch, tokenizer, mode="GPT")
    assert result[0]["labels"] == [0]
