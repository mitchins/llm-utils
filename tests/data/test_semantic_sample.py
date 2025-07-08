import sys
import importlib
import types
import numpy as np
import pytest


@pytest.fixture
def semantic_sample_module(monkeypatch):
    dummy_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
        set_default_device=lambda dev: None,
        float16='float16'
    )
    dummy_st = types.SimpleNamespace(SentenceTransformer=lambda model: None)
    dummy_tqdm = types.SimpleNamespace(tqdm=lambda x, **k: x)
    dummy_sklearn_decomp = types.SimpleNamespace(PCA=lambda *a, **k: None)
    dummy_sklearn = types.ModuleType('sklearn')
    dummy_sklearn.decomposition = dummy_sklearn_decomp
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', dummy_st)
    monkeypatch.setitem(sys.modules, 'tqdm', dummy_tqdm)
    monkeypatch.setitem(sys.modules, 'sklearn', dummy_sklearn)
    monkeypatch.setitem(sys.modules, 'sklearn.decomposition', dummy_sklearn_decomp)
    mod = importlib.import_module('llm_utils.data.semantic_sample')
    return mod


def test_fingerprint(semantic_sample_module):
    mod = semantic_sample_module
    vec = np.array([1, -2, 3])
    assert mod.fingerprint(vec) == 5


def test_holdout_split(semantic_sample_module):
    mod = semantic_sample_module
    rows = [
        ('t1', '', 'A'),
        ('t2', '', 'A'),
        ('t3', '', 'B'),
        ('t4', '', 'B'),
    ]
    main, holdout = mod.holdout_split(rows, 0.5)
    assert len(main) == 2
    assert len(holdout) == 2
    all_rows = sorted(main + holdout)
    assert sorted(rows) == all_rows
