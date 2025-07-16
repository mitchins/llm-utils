import os
from datasets import load_dataset, Dataset, DatasetDict

def load_dataset_auto(path: str, split: str = "train"):
    """Load a dataset from a local path or HF hub."""
    if path.startswith("hf:"):
        ds = load_dataset(path[3:])
        return ds[split] if isinstance(ds, DatasetDict) else ds
    ext = os.path.splitext(path)[1].lower()
    if ext in {".json", ".jsonl"}:
        return load_dataset("json", data_files=path, split=split)
    if ext == ".csv":
        return load_dataset("csv", data_files=path, split=split)
    if os.path.isdir(path):
        return Dataset.load_from_disk(path)
    raise ValueError(f"Unsupported dataset path: {path}")
