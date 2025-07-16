import json
from pathlib import Path
import pandas as pd
import math
from transformers import AutoConfig
import pynvml

def load_and_filter_dataframe(data_path: Path, label_field: str = "label") -> pd.DataFrame:
    rows = []
    skipped = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                item = json.loads(line)
                rows.append(item)
            except json.JSONDecodeError:
                skipped += 1
    df = pd.DataFrame(rows)
    # drop invalid labels
    df = df[df[label_field].notna()]
    return df

def calculate_eval_size(total_size: int, eval_ratio: float = 0.1) -> int:
    """Calculate evaluation set size given total dataset size and evaluation ratio."""
    return max(1, int(total_size * eval_ratio))

def determine_batch_size(model_checkpoint: str, no_batching: bool, total_vram_gb: float = None) -> int:
    """
    Determine batch size based on model size and available VRAM.
    If total_vram_gb is not provided, attempts to detect VRAM automatically (defaults to 16GB if detection fails).
    """
    if total_vram_gb is None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            total_vram_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
            pynvml.nvmlShutdown()
        except Exception:
            total_vram_gb = 16.0
    if no_batching:
        return 1
    # Load model config
    config = AutoConfig.from_pretrained(model_checkpoint)
    hidden = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    intermediate = getattr(config, "intermediate_size", hidden * 4 if hidden else None)
    # Approximate param count for transformer blocks
    if hidden and layers and intermediate:
        # self-attention + feed-forward weights
        approx_params = layers * (3 * hidden * hidden + 2 * hidden * intermediate)
    else:
        approx_params = 1e8  # fallback reference
    # Scaling factors
    ref_params = 1e8  # reference 100M params
    ref_vram = 24.0   # reference 24gb VRAM
    # Model-specific override: use fixed scale for deBERTa
    model_name = model_checkpoint.lower()
    # For classifiers, we've always just used items of length 512.
    if "deberta" in model_name:
        param_scale = 1.25 # Last tested at 1.1, 87.0% VRAM
    # Model-specific override: use fixed batch size for t5-small
    elif 't5-small' in model_name:
        param_scale = 3.1  # With 3.2 it hits 99% usage after a 300 step delay
    elif 't5-large' in model_name:
        param_scale = 0.25
    elif 'distilbert' in model_name:
        param_scale = 6.1
    elif 'qwen2.5-0.5b' in model_name:
        param_scale = 0.33
    else:
        param_scale = math.sqrt(ref_params / approx_params)
    vram_scale = total_vram_gb / ref_vram
    base_batch = 24
    batch_size = int(base_batch * param_scale * vram_scale)
    # Cap batch size for very large models to avoid OOM (e.g., DeBERTa, T5-large)
    if hidden and hidden > 1024:
        # do not exceed base_batch for large hidden sizes
        batch_size = min(batch_size, base_batch)
    return max(1, batch_size)


def calculate_dynamic_eval_steps(
    train_dataset_size: int,
    batch_size: int,
    preferred_fraction: float = 1 / 3,
    max_steps: int = 10000,
) -> int:
    """Return evaluation interval based on dataset size and batch size.

    This helper is shared by multiple training scripts to keep evaluation
    frequency consistent across models.
    """
    steps_per_epoch = train_dataset_size // batch_size
    proposed_steps = int(steps_per_epoch * preferred_fraction)
    return min(steps_per_epoch, max(proposed_steps, 1), max_steps)
