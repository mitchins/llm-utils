import logging
import math
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# training/utils.py
def calculate_dynamic_eval_steps(
    train_dataset_size: int,
    batch_size: int,
    preferred_fraction: float = 1 / 3,
    max_steps: int = 10000,
) -> int:
    """Compute evaluation interval based on dataset size and batch size.

    Uses a simple heuristic of evaluating roughly ``preferred_fraction`` of an
    epoch, capped by ``max_steps``. ``batch_size`` must be positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    steps_per_epoch = math.ceil(train_dataset_size / batch_size)
    proposed_steps = int(steps_per_epoch * preferred_fraction)
    return min(steps_per_epoch, max(proposed_steps, 1), max_steps)
