import sys
import types
import os
import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")
from transformers.trainer_utils import TrainOutput
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
    import types as _types
    class DummyTok:
        class DummyTensor(list):
            @property
            def shape(self):
                return (len(self), len(self[0]) if self else 0)

        def __init__(self):
            self.model_max_length = None
            self.pad_token_id = 0
            self.padding_side = "right"
            self.truncation_side = "right"

        @staticmethod
        def from_pretrained(*a, **k):
            # Return a new tokenizer instance
            return DummyTok()

        def __call__(self, text, truncation=False, padding=None, max_length=None):
            if isinstance(text, list):
                length = len(text[0].split())
            else:
                length = len(str(text).split())
            ids = list(range(length))
            return {"input_ids": ids, "attention_mask": [1]*len(ids)}

        def pad(self, batch, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
            # Handle dict-of-lists (e.g., padding labels)
            if isinstance(batch, dict):
                padded = {}
                for key, items in batch.items():
                    max_len = max_length or max(len(i) for i in items)
                    padded[key] = DummyTok.DummyTensor(
                        [[0] * (max_len - len(i)) + i for i in items]
                    )
                return padded

            # Handle list-of-dicts (features)
            padded = {}
            # Collect all keys present in any example
            keys = set().union(*(example.keys() for example in batch))
            for key in keys:
                items = [example.get(key, []) for example in batch]
                max_len = max_length or max(len(i) for i in items)
                padded[key] = DummyTok.DummyTensor(
                    [[0] * (max_len - len(i)) + i for i in items]
                )
            return padded

    class DummyModel:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=False)
        def gradient_checkpointing_enable(self):
            pass

    def fake_model_pretrained(*a, **k):
        return DummyModel()

    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.Generator = type("Generator", (), {})
    torch_mod.nn = types.SimpleNamespace(Module=object)

    monkeypatch.setattr(train_t5, "AutoTokenizer", _types.SimpleNamespace(from_pretrained=DummyTok.from_pretrained))
    monkeypatch.setattr(train_t5, "AutoModelForSeq2SeqLM", _types.SimpleNamespace(from_pretrained=fake_model_pretrained))
    monkeypatch.setattr(train_t5.evaluate, "load", lambda *a, **k: types.SimpleNamespace(compute=lambda **kw: {}))
    monkeypatch.setattr(train_t5, "determine_batch_size", lambda *a, **k: 1)
    monkeypatch.setattr(train_t5, "Seq2SeqTrainingArguments", lambda **kw: types.SimpleNamespace(**kw))
    # Simplify dataset splitting
    monkeypatch.setattr(Dataset, "train_test_split", lambda self, *a, **k: {"train": self, "test": self})
    monkeypatch.setattr(Dataset, "cast_column", lambda self, *a, **k: self)
    monkeypatch.setattr(train_t5.state, "is_tokenized", True)
def stub_dataset(monkeypatch, data):
    from llm_utils.training import train_t5
    from datasets import Dataset
    monkeypatch.setattr(train_t5, "load_dataset_auto", lambda *a, **k: Dataset.from_dict(data))
    import llm_utils.data.dataset_loading as dl
    monkeypatch.setattr(dl, "load_dataset_auto", lambda *a, **k: Dataset.from_dict(data))
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
        self.state = types.SimpleNamespace(best_model_checkpoint="dummy_checkpoint")
    def train(self):
        self.trained = True
        # Return metrics with an "epoch" key for compatibility with run_training
        return TrainOutput(global_step=0, training_loss=0.0, metrics={"epoch": self.args.num_train_epochs})


def test_parse_args_defaults():
    args = train_t5.parse_args(["--task-name", "foo", "--train-dataset-dir", "x.csv"])
    assert args.max_input_length is None
    assert args.task_name == "foo"


def test_tokenizer_override(monkeypatch):
    class FakeTok:
        def __init__(self, model_max_length):
            self.model_max_length = model_max_length
        def __call__(self, text, truncation=False, padding=None, max_length=None):
            ids = [0]
            return {"input_ids": ids, "attention_mask": [1], "labels": ids}
    args = train_t5.parse_args(["--task-name", "t", "--max-input-length", "50", "--train-dataset-dir", "d.csv", "--validation-size", "0"])
    fake = FakeTok(model_max_length=args.max_input_length)
    import types as _types
    monkeypatch.setattr(train_t5, "AutoTokenizer", _types.SimpleNamespace(from_pretrained=lambda *a, **k: fake))
    stub_dataset(monkeypatch, {"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]})
    state, model, collator, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    assert state.tokenizer is fake
    assert collator.padding == "longest"
    assert collator.max_length is None
    assert state.tokenizer.model_max_length == 50


def test_csv_loading(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("input,output\nhello,world\n")
    stub_dataset(monkeypatch, {"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]})
    args = train_t5.parse_args(["--task-name", "t", "--train-dataset-dir", str(csv), "--validation-size", "0"])
    state, _, _, _ = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    assert len(state.train_dataset) == 1


def test_collator_padding(monkeypatch):
    stub_dataset(monkeypatch, {"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]})
    args = train_t5.parse_args(["--task-name", "t", "--max-input-length", "20", "--train-dataset-dir", "d.csv", "--validation-size", "0"])
    state, model, collator, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    dummy = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1]}
    batch = collator([dummy])
    # ...
    assert batch["input_ids"].shape[1] == len(dummy["input_ids"])


def test_integration_smoke(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("input,output\na,b\nc,d\ne,f\ng,h\ni,j\n")
    stub_dataset(monkeypatch, {"input_ids": [[0],[0]], "attention_mask": [[1],[1]], "labels": [[0],[0]]})
    args = train_t5.parse_args(["--task-name", "t", "--train-dataset-dir", str(csv), "--max-input-length", "50", "--validation-size", "0"])
    _, _, _, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    train_t5.run_training(trainer)
    assert trainer.trained

def test_collator_pad_behavior(monkeypatch):
    stub_dataset(monkeypatch, {"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "labels": [[1]]})
    args = train_t5.parse_args(["--task-name", "t", "--max-input-length", "5", "--train-dataset-dir", "d.csv", "--validation-size", "0"])
    state, model, collator, trainer = train_t5.build_pipeline(args, trainer_cls=DummyTrainer)
    dummy_batch = [{"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [1, 2, 3]}]
    batch = collator(dummy_batch)
    assert "input_ids" in batch
    assert batch["input_ids"][0] == dummy_batch[0]["input_ids"]
def test_log_length_histogram(monkeypatch):
    # Prepare a fake dataset with varying lengths
    fake_data = [
        {"input_ids": [0]},
        {"input_ids": list(range(5))},
        {"input_ids": list(range(10))},
    ]
    # Capture logged messages
    logs = []
    monkeypatch.setattr(train_t5, "rank_logger", lambda level, msg: logs.append(msg))
    # Call utility
    train_t5.log_length_histogram(fake_data, max_bins=2)
    # Confirm histogram header and counts
    assert any("Input length histogram" in m for m in logs)
    # Expect one bin line ending with ": 2" and one ending with ": 1"
    assert any(m.strip().endswith(": 2") for m in logs)
    assert any(m.strip().endswith(": 1") for m in logs)


# Test versioned directory naming in save_model
def test_versioned_save(tmp_path):
    from llm_utils.training.train_t5 import RankZeroOnlySaveTrainer
    import types, os

    # Prepare a list to capture which dirs get passed to save_pretrained
    saved_model_dirs = []
    saved_tokenizer_dirs = []

    base = tmp_path / "model"

    class FakeTrainer(RankZeroOnlySaveTrainer):
        def __init__(self):
            # args with initial output_dir
            self.args = types.SimpleNamespace(local_rank=-1, output_dir=str(base))
            # Fake model.module or model with save_pretrained that records calls
            def fake_state_dict():
                return {}
            def fake_save_pretrained(out, **kwargs):
                saved_model_dirs.append(out)
            dummy_module = types.SimpleNamespace(
                state_dict=fake_state_dict,
                save_pretrained=fake_save_pretrained
            )
            # Expose module so save_model picks it up if needed
            self.model = types.SimpleNamespace(
                module=dummy_module,
                save_pretrained=fake_save_pretrained,
                state_dict=fake_state_dict
            )
            # Fake tokenizer
            self.tokenizer = types.SimpleNamespace(
                save_pretrained=lambda out: saved_tokenizer_dirs.append(out)
            )

    trainer = FakeTrainer()

    # First save → base dir
    trainer.save_model()
    # Second save → base-v1
    trainer.save_model()

    assert saved_model_dirs[0] == str(base)
    assert saved_model_dirs[1] == f"{base}-v1"
    # And tokenizer saved the same places
    assert saved_tokenizer_dirs == saved_model_dirs
def test_additional_special_tokens(monkeypatch):
    # Simulate dataset and tokenizer with no special tokens by default
    from llm_utils.training import train_t5
    args = train_t5.parse_args([
        "--task-name", "specialtok",
        "--train-dataset-dir", "dummy.csv",
        "--validation-size", "0",
        "--additional-special-tokens", "<MASK>,<FOO>"
    ])
    # Patch tokenizer to track added special tokens
    class TrackSpecialTok:
        def __init__(self):
            self.added = []
            self.model_max_length = 42
            self.pad_token_id = 0
        @staticmethod
        def from_pretrained(*a, **k):
            return TrackSpecialTok()
        def add_special_tokens(self, d):
            toks = d.get("additional_special_tokens", [])
            self.added.extend(toks)
            return len(toks)
        def __call__(self, text, truncation=False, padding=None, max_length=None):
            return {"input_ids": [0], "attention_mask": [1], "labels": [0]}
        def __len__(self):
            # Simulate a vocab size, e.g., base 100 plus any added special tokens
            return 100 + len(self.added)
    import types as _types
    import types
    monkeypatch.setattr(train_t5, "AutoTokenizer", _types.SimpleNamespace(from_pretrained=TrackSpecialTok.from_pretrained))
    monkeypatch.setattr(train_t5, "AutoModelForSeq2SeqLM", _types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(config=types.SimpleNamespace(use_cache=False), resize_token_embeddings=lambda n: None, gradient_checkpointing_enable=lambda : None)))
    monkeypatch.setattr(train_t5, "determine_batch_size", lambda *a, **k: 1)
    # Patch dataset loading
    from datasets import Dataset
    monkeypatch.setattr(
        train_t5,
        "load_dataset_auto",
        lambda *args, **kwargs: Dataset.from_dict({
            "input_ids": [[0]],
            "attention_mask": [[1]],
            "labels": [[0]]
        })
    )
    # Stub out log_length_histogram before build_pipeline
    monkeypatch.setattr(train_t5, "log_length_histogram", lambda *args, **kwargs: None)
    # Build pipeline
    state, model, collator, trainer = train_t5.build_pipeline(args, trainer_cls=lambda **kwargs: None)
    # Check that the special tokens were registered
    assert "<MASK>" in state.tokenizer.added
    assert "<FOO>" in state.tokenizer.added