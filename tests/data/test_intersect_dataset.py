import argparse
import importlib
import json


def test_load_texts(tmp_path):
    p = tmp_path / 'data.jsonl'
    p.write_text('{"text": "A"}\n{"text": "B"}\n{"text": "A"}\n', encoding='utf-8')
    mod = importlib.import_module('llm_utils.data.intersect_dataset')
    result = mod.load_texts(p)
    assert result == {'A', 'B'}


def test_intersect_dataset_main(tmp_path, monkeypatch, capsys):
    file1 = tmp_path / 'f1.jsonl'
    file2 = tmp_path / 'f2.jsonl'
    out = tmp_path / 'out.jsonl'
    file1.write_text('{"text": "A"}\n{"text": "B"}\n', encoding='utf-8')
    file2.write_text('{"text": "B"}\n{"text": "C"}\n', encoding='utf-8')
    args = argparse.Namespace(file1=file1, file2=file2, output=out)
    mod = importlib.import_module('llm_utils.data.intersect_dataset')
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', lambda self: args)
    mod.main()
    captured = capsys.readouterr()
    assert 'overlapping text entries' in captured.out
    saved = out.read_text().strip()
    assert json.loads(saved)['text'] == 'B'
