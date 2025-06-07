import os
import logging
from transformers import TrainerCallback
import torch
import csv

logger = logging.getLogger(__name__)


# General-purpose callback: logs all metrics in logs dict at epoch-level using a SummaryWriter
class EpochNormalizedLogger(TrainerCallback):
    """
    Logs any metric in `logs` to a SummaryWriter, using epoch as x-axis,
    with 'epoch/' prefix for all metrics.
    Closes the writer at training end.
    Model/task-agnostic.
    """
    def __init__(self, writer):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        epoch = getattr(state, "epoch", None)
        if epoch is None:
            # Can't log without epoch info
            return
        for key, value in logs.items():
            # Only log scalars
            if isinstance(value, (int, float)):
                tag = f"epoch/{key}"
                try:
                    self.writer.add_scalar(tag, value, epoch)
                except Exception:
                    # Don't break training if logging fails for a metric
                    logger.warning(f"Failed to log {tag}={value} at epoch={epoch}")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self.writer.close()
        except Exception:
            logger.warning("EpochNormalizedLogger: failed to close SummaryWriter.")


class MemoryUsageLogger(TrainerCallback):
    """
    Logs model parameter count, input size, batch size, and peak VRAM usage
    to 'memory_usage.csv' on the first training step.
    """
    def __init__(self, model, checkpoint: str, batch_size: int, input_size: int, csv_path: str = "memory_usage.csv"):
        self.model = model
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.input_size = input_size
        self.csv_path = csv_path

    def on_step_end(self, args, state, control, **kwargs):
        # Only once, at the first step
        if state.global_step != 1:
            return
        # Parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        # Peak VRAM usage
        peak_bytes = torch.cuda.max_memory_allocated()
        vram_gb = peak_bytes / (1024**3)
        # Prepare CSV
        header = ["model_checkpoint", "param_count", "input_size", "batch_size", "vram_gb"]
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                self.checkpoint,
                param_count,
                self.input_size,
                self.batch_size,
                round(vram_gb, 3)
            ])

# ManualEarlyStopCallback: allows manual early stopping via file signal
class ManualEarlyStopCallback(TrainerCallback):
    """
    Allows manual early stopping during training by checking for a file signal.
    Creates a simple way to stop training cleanly via 'touch stop_training.txt'.
    """
    def __init__(self, stop_file="stop_training.txt"):
        self.stop_file = stop_file

    def on_evaluate(self, args, state, control, **kwargs):
        if os.path.exists(self.stop_file):
            logger.info(f"🛑 Manual stop signal detected ({self.stop_file}). Ending training early.")
            try:
                os.remove(self.stop_file)
                logger.info(f"🧹 Removed stop file {self.stop_file}.")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete stop file {self.stop_file}: {e}")
            control.should_training_stop = True
        return control