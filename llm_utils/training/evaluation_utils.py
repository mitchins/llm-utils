import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# training/utils.py
def calculate_dynamic_eval_steps(train_dataset_size: int, batch_size: int, preferred_fraction: float = 1/3, max_steps: int = 10000) -> int:
    steps_per_epoch = train_dataset_size // batch_size
    proposed_steps = int(steps_per_epoch * preferred_fraction)
    return min(steps_per_epoch, max(proposed_steps, 1), max_steps)
