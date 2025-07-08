import sys
import importlib
import types


def test_analyze_token_lengths(monkeypatch, capsys):
    dummy_data = [
        {'input_ids': [1, 2, 3, 4]},
        {'input_ids': [5, 6]}
    ]
    dummy_datasets = types.SimpleNamespace(load_from_disk=lambda path: dummy_data)
    dummy_transformers = types.SimpleNamespace(
        BertTokenizerFast=types.SimpleNamespace(from_pretrained=lambda name: None)
    )
    monkeypatch.setitem(sys.modules, 'datasets', dummy_datasets)
    monkeypatch.setitem(sys.modules, 'transformers', dummy_transformers)
    mod = importlib.import_module('llm_utils.data.analyze_token_lengths')
    lengths = mod.analyze_token_lengths('ignored')
    captured = capsys.readouterr()
    assert lengths == [4, 2]
    assert 'Max tokens: 4' in captured.out
    assert 'Min tokens: 2' in captured.out
