import sys
import types
import os
import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")
from datasets import Dataset

# Stub minimal torch modules so train_t5 can be imported without real torch
torch_mod = types.ModuleType("torch")
torch_mod.distributed = types.SimpleNamespace(is_initialized=lambda: False, get_rank=lambda: 0, is_available=lambda: False)
tensorboard_mod = types.ModuleType("torch.utils.tensorboard")
tensorboard_mod.SummaryWriter = lambda *a, **k: None
torch_utils_mod = types.ModuleType("torch.utils")
torch_utils_mod.tensorboard = tensorboard_mod
torch_mod.utils = torch_utils_mod
torch_mod.__spec__ = types.SimpleNamespace()
torch_utils_mod.__spec__ = types.SimpleNamespace()
tensorboard_mod.__spec__ = types.SimpleNamespace()
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.distributed", torch_mod.distributed)
sys.modules.setdefault("torch.utils", torch_utils_mod)
sys.modules.setdefault("torch.utils.tensorboard", tensorboard_mod)
sys.modules.setdefault("pynvml", types.SimpleNamespace())
sys.modules.setdefault("pynvml", types.SimpleNamespace())

# Patch out heavy dependencies
@pytest.fixture(autouse=True)
def stub_hf(monkeypatch):
    class DummyTok:
        def __init__(self):
            self.model_max_length = None
            self.pad_token_id = 0

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def __call__(self, text, truncation=False, padding=None, max_length=None):
            if isinstance(text, list):
                length = len(text[0].split())
            else:
                length = len(str(text).split())
            ids = list(range(length))
            return {"input_ids": ids, "attention_mask": [1]*len(ids)}

    class DummyModel:
        pass

    def fake_model_pretrained(*a, **k):
        m = DummyModel()
        m.config = types.SimpleNamespace(use_cache=False)
        return m

    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.Generator = type("Generator", (), {})
    torch_mod.nn = types.SimpleNamespace(Module=object)

    monkeypatch.setattr(train_t5, "AutoTokenizer", DummyTok)
    monkeypatch.setattr(train_t5.AutoModelForSeq2SeqLM, "from_pretrained", fake_model_pretrained)
    monkeypatch.setattr(train_t5.evaluate, "load", lambda *a, **k: types.SimpleNamespace(compute=lambda **kw: {}))
    # Simplify dataset splitting
    monkeypatch.setattr(Dataset, "train_test_split", lambda self, *a, **k: {"train": self, "test": self})
    monkeypatch.setattr(Dataset, "cast_column", lambda self, *a, **k: self)
    monkeypatch.setattr(train_t5.state, "is_tokenized", True)
# Ensure package is importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from llm_utils.training import train_t5

class DummyTrainer:
    def __init__(self, *, train_dataset, eval_dataset, data_collator, args, **kwargs):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.args = args
        self.trained = False
    def train(self):
        self.trained = True


def test_parse_args_defaults():
    args = train_t5.parse_args(["--task-name", "foo", "--train-dataset-dir", "x.csv"])
    assert args.max_input_length == 512
    assert args.task_name == "foo"


def test_tokenizer_override(monkeypatch):
    class FakeTok:
        def __init__(self):
            self.model_max_length = None
        def __call__(self, text, truncation=False, padding=None, max_length=None):
            ids = [0]
            return {"input_ids": ids, "attention_mask": [1], "labels": ids}
    fake = FakeTok()
    monkeypatch.setattr(train_t5.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: fake))
    args = train_t5.parse_args(["--task-name", "t", "--max-input-length", "50", "--train-dataset-dir", "d.csv", "--validation-size", "0"])
    monkeypatch.setattr(train_t5, "load_dataset", lambda *a, **k: Dataset.from_dict({"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]}))
    state, model, collator, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    assert state.tokenizer is fake
    assert collator.max_length == 50
    assert trainer.data_collator.padding == "max_length"
    assert state.tokenizer.model_max_length == 50


def test_csv_loading(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("input,output\nhello,world\n")
    monkeypatch.setattr(train_t5, "load_dataset", lambda *a, **k: Dataset.from_dict({"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]}))
    args = train_t5.parse_args(["--task-name", "t", "--train-dataset-dir", str(csv), "--validation-size", "0"])
    state, _, _, _ = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    assert len(state.train_dataset) == 1


def test_collator_padding(monkeypatch):
    monkeypatch.setattr(train_t5, "load_dataset", lambda *a, **k: Dataset.from_dict({"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]}))
    args = train_t5.parse_args(["--task-name", "t", "--max-input-length", "20", "--train-dataset-dir", "d.csv", "--validation-size", "0"])
    state, model, collator, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    dummy = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1]}
    batch = collator([dummy])
    assert batch["input_ids"].shape[1] == 20


def test_integration_smoke(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("input,output\na,b\nc,d\ne,f\ng,h\ni,j\n")
    monkeypatch.setattr(train_t5, "load_dataset", lambda *a, **k: Dataset.from_dict({"input_ids": [[0],[0]], "attention_mask": [[1],[1]], "labels": [[0],[0]]}))
    args = train_t5.parse_args(["--task-name", "t", "--train-dataset-dir", str(csv), "--max-input-length", "50", "--validation-size", "0"])
    _, _, _, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    train_t5.run_training(trainer)
    assert trainer.trained
