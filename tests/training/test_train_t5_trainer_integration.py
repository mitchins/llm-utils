# File: tests/training/test_train_t5_trainer_integration.py

import os
import csv
import pytest
from llm_utils.training.train_t5 import parse_args, build_pipeline, state

class CaptureTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, processing_class, data_collator, callbacks, compute_metrics):
        # Capture for assertions
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = processing_class

    def train(self):
        # No training performed
        return

@pytest.fixture(autouse=True)
def reset_state():
    # Reset global state
    state.__init__()
    yield

@pytest.fixture
def mask_csv(tmp_path):
    # Ten-row CSV with <MASK> in various positions, alternating patterns
    file = tmp_path / "mask_data.csv"
    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        for i in range(10):
            if i % 2 == 0:
                writer.writerow({"input": "Hello <MASK> world", "output": "<MASK> test"})
            else:
                writer.writerow({"input": "Another <MASK> here", "output": "Result <MASK>"})
    return str(file)

def test_trainer_sees_atomic_mask(mask_csv):
    args = parse_args([
        "--task-name",            "trainer_mask",
        "--train-dataset-dir",    mask_csv,
        "--model-checkpoint",     "t5-small",
        "--input-col",            "input",
        "--target-col",           "output",
        "--additional-special-tokens", "<MASK>"
    ])
    state_obj, model, collator, trainer = build_pipeline(args, trainer_cls=CaptureTrainer)

    # 1) Tokenizer should treat <MASK> as one atomic token among SentencePiece tokens
    tok = state_obj.tokenizer
    tokens = tok.tokenize("Hello <MASK> world")
    # The mask token should appear as an atomic token among SentencePiece tokens
    assert "<MASK>" in tokens
    # Ensure it is not split
    assert not any(tok in ["MASK", "<", ">"] for tok in tokens)

    # 2) The processed train_dataset inside the trainer must contain mask_id
    mask_id = tok.convert_tokens_to_ids("<MASK>")
    for ex in trainer.train_dataset:
        assert mask_id in ex["input_ids"]
        assert mask_id in ex["labels"]

    # 3) Ensure input_ids and labels are integer lists
    sample = trainer.train_dataset[0]
    assert isinstance(sample["input_ids"], list)
    assert all(isinstance(i, int) for i in sample["input_ids"])
    assert isinstance(sample["labels"], list)
    assert all(isinstance(i, int) for i in sample["labels"])