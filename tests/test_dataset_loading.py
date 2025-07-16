import importlib
import types
import sys


def make_mod(monkeypatch):
    dummy_ds = types.SimpleNamespace(from_dict=lambda d: d,
                                     load_from_disk=lambda p: f"disk:{p}")
    def fake_load_dataset(fmt, data_files=None, split='train'):
        return f"{fmt}:{data_files}:{split}"
    datasets_mod = types.SimpleNamespace(load_dataset=fake_load_dataset,
                                         Dataset=dummy_ds,
                                         DatasetDict=dict)
    monkeypatch.setitem(sys.modules, 'datasets', datasets_mod)
    return importlib.reload(importlib.import_module('llm_utils.data.dataset_loading'))


def test_load_json(monkeypatch):
    mod = make_mod(monkeypatch)
    assert mod.load_dataset_auto('data.jsonl') == 'json:data.jsonl:train'


def test_load_csv(monkeypatch):
    mod = make_mod(monkeypatch)
    assert mod.load_dataset_auto('file.csv') == 'csv:file.csv:train'


def test_load_disk(monkeypatch, tmp_path):
    mod = make_mod(monkeypatch)
    d = tmp_path / 'ds'
    d.mkdir()
    assert mod.load_dataset_auto(str(d)) == f'disk:{d}'
