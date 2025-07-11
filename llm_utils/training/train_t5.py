import json
import numpy as np
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, Value
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import DataCollatorWithPadding
import logging
import os
import psutil
from transformers import EarlyStoppingCallback
from .callbacks import EpochNormalizedLogger, MemoryUsageLogger
from .utils import determine_batch_size
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
import time
from transformers import Seq2SeqTrainer as HFSeq2SeqTrainer, TrainerCallback
from typing import Optional
import math
from datetime import datetime

logger = logging.getLogger(__name__)

# === Utility helpers ===
def get_world_size_safe():
    import torch.distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

# Helper function to tokenize datasets
def tokenize_dataset(dataset, preprocess_fn, args, desc):
    return dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=args.threads,
        remove_columns=[args.input_col, args.target_col],
        desc=desc,
    )

def rank_logger(level, message):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        prefix = f"[rank{rank}] "
    elif "RANK" in os.environ:
        prefix = f"[rank{os.environ['RANK']}] "
    else:
        prefix = ""
    getattr(logger, level)(f"{prefix}{message}")

# === Default hyperparameter constants ===
DEFAULT_WARMUP_STEPS = 500
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_TOTAL_EPOCHS = 30
DEFAULT_VALIDATION_SIZE = 0.10
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_TARGET_LENGTH = 128

# === Training state ===
class TrainingState:
    def __init__(self):
        self.is_tokenized = False  # Whether the dataset has been tokenized
        self.train_dataset = None  # For training
        self.test_dataset = None  # For ongoing evaluation during training
        self.validation_dataset = None  # For post-training validation
        self.args_cli = None
        self.tokenizer = None

# Create a global instance of the state
state = TrainingState()

class PredictionShapeLoggerCallback(TrainerCallback):
    def on_prediction_step(self, args, state, control, **kwargs):
        rank_logger("debug", f"Available kwargs: {list(kwargs.keys())}")
        if 'model' in kwargs:
            model = kwargs['model']
            rank_logger("debug", f"Model: {type(model)}")
        if 'inputs' in kwargs:
            inputs = kwargs['inputs']
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    if hasattr(value, 'shape'):
                        pass
            elif hasattr(inputs, 'shape'):
                pass
        if state:
            step = getattr(state, 'prediction_step', None)
            rank_logger("debug", f"Current step: {step if step is not None else 'N/A'}")
        if args:
            rank_logger("debug", f"Batch size: {args.per_device_eval_batch_size}")
            if hasattr(args, 'generation_max_length'):
                rank_logger("info", f"Max generation length: {args.generation_max_length}")        

# === Trainer subclass: only save on rank 0 ===
class RankZeroOnlySaveTrainer(HFSeq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.args.predict_with_generate and not prediction_loss_only:
            self._start_time = time.time()
            rank_logger("debug", "🔁 Generation started...")
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if self.args.predict_with_generate and not prediction_loss_only:
            duration = time.time() - self._start_time
            rank_logger("debug", f"✅ Generation complete in {duration:.2f}s")
        return outputs
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = True):
        if getattr(self.args, "local_rank", -1) not in [-1, 0]:
            rank_logger("info", f"[rank{getattr(self.args, 'local_rank', -1)}] Skipping save_model.")
            return
        super().save_model(output_dir, _internal_call)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default min_delta values for early stopping
DEFAULT_MIN_DELTAS = {
    "combined": 0.01,
    "rougeL": 0.005
}

def default_stopping_delta(args):
    """
    Returns the default minimum delta for early stopping based on whether METEOR is enabled.
    """
    if args.calculate_meteor:
        return DEFAULT_MIN_DELTAS["combined"]
    else:
        return DEFAULT_MIN_DELTAS["rougeL"]

# Global counters for input length filtering
total_examples = 0
dropped_examples = 0

parser = argparse.ArgumentParser(
    description="Generic seq2seq trainer for tasks like summarization, translation, paraphrasing"
)
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--threads", type=int, default=1, help="Number of worker threads for dataset.map (default: 1)")
parser.add_argument("--eval_steps", type=int, default=None, help="Force evaluation every N steps")
parser.add_argument(
    "--eval_strategy",
    choices=["auto", "epoch"],
    default="auto",
    help="Use dynamic step-based checkpointing ('auto') or eval/save every epoch ('epoch').",
)
parser.add_argument(
    "--gradient-accumulation-steps",
    type=int,
    default=1,
    help="Number of steps to accumulate gradients before optimizing (default: 1 = no accumulation)"
)
parser.add_argument("--allow-empty-output", action="store_true", help="Permit empty strings in the target/output column")
parser.add_argument(
    "--batch-size", type=int, default=None,
    help="Override auto-scaled per-device batch size; set to 1 for effectively no batching"
)
parser.add_argument(
    "--output-dir", type=str, default=None,
    help="Output directory for model and encoder (default: <task_name>_model)"
)
parser.add_argument("--clean", action="store_true", help="Remove output_dir before training if it exists")
parser.add_argument("--total-epochs", type=int, default=DEFAULT_TOTAL_EPOCHS, help=f"Total number of training epochs (default: {DEFAULT_TOTAL_EPOCHS})")
parser.add_argument("--model-checkpoint", type=str, default="t5-small", help="HuggingFace model checkpoint to use")
parser.add_argument("--bf16", action="store_true", help="Enable mixed precision (BF16) training (default: FP32/full precision)")
parser.add_argument("--warm-up-steps", type=int, default=DEFAULT_WARMUP_STEPS, help=f"Number of warm-up steps for learning rate scheduler (set 0 for no warm-up, default: {DEFAULT_WARMUP_STEPS})")
parser.add_argument(
    "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
    help=f"Learning rate for optimizer (default: {DEFAULT_LEARNING_RATE})"
)
parser.add_argument(
    "--lr-scheduler-type",
    type=str,
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial"],
    default="cosine",
    help="Type of learning rate scheduler (default: cosine)"
)
parser.add_argument(
    "--early-stopping-patience",
    type=int,
    default=DEFAULT_EARLY_STOPPING_PATIENCE,
    help=f"Number of evaluations with no improvement before early stopping (default: {DEFAULT_EARLY_STOPPING_PATIENCE})"
)
parser.add_argument(
    "--min-delta", type=float, default=None,
    help=f"Minimum absolute improvement to reset early-stopping patience (default: {DEFAULT_MIN_DELTAS['combined']} if METEOR enabled, else {DEFAULT_MIN_DELTAS['rougeL']})"
)
parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE, help=f"Percentage of the dataset to use as validation set (default: {DEFAULT_VALIDATION_SIZE})")
parser.add_argument("--stratify-length", action="store_true", help="Stratify validation split by length of target output (token count or JSON element count)")
parser.add_argument("--calculate-meteor", action="store_true", help="Enable calculation of METEOR metric during evaluation")
parser.add_argument("--raw-metrics", action="store_true", help="Output ROUGE and METEOR as raw 0–1 values instead of percentages")
parser.add_argument(
    "--fields",
    type=str,
    default=None,
    help="Structured output fields and metric types, e.g. 'name:exact,type:f1,summary:rouge'"
)
parser.add_argument(
    "--task-name", type=str, required=True,
    help="Short name for this task (e.g. 'summary', 'translation', 'paraphrase'). Used to name output dirs, runs, etc."
)
parser.add_argument(
    "--input-col", type=str, default="input",
    help="Name of the source text field in your dataset"
)
parser.add_argument(
    "--target-col", type=str, default="output",
    help="Name of the target text field"
)
parser.add_argument(
    "--max-input-length", type=int, default=DEFAULT_MAX_INPUT_LENGTH,
    help=f"Maximum input sequence length (default: {DEFAULT_MAX_INPUT_LENGTH})"
)
parser.add_argument(
    "--max-target-length", type=int, default=DEFAULT_MAX_TARGET_LENGTH,
    help=f"Maximum target sequence length (default: {DEFAULT_MAX_TARGET_LENGTH})"
)

# DeepSpeed argument
parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON file (enables DeepSpeed training)")

# Optimizer selection: allow disabling Adafactor
parser.add_argument("--disable-adafactor", action="store_true", help="Use AdamW instead of Adafactor (default: Adafactor)")

# TODO: remove the 'dir' part from the naming here
# New CLI arguments for HuggingFace datasets from disk
parser.add_argument("--train-dataset-dir", type=str, help="Path to a HuggingFace dataset directory for training (saved with save_to_disk())")
parser.add_argument("--eval-dataset-dir", type=str, help="Optional path to evaluation dataset directory (if not provided, splits train)")

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def report_memory():
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    rank_logger("info", f"🧠 Current memory usage: {mem:.2f} MB")

# === Preprocessing ===
def preprocess_t5(example):
    # Extract raw text for inputs and targets
    inputs = example[state.args_cli.input_col]
    targets = example[state.args_cli.target_col]

    # Ensure Python str types for tokenizer compatibility
    if isinstance(inputs, (list, tuple)):
        inputs = [str(x) for x in inputs]
    else:
        inputs = str(inputs)
    if isinstance(targets, (list, tuple)):
        targets = [str(x) for x in targets]
    else:
        targets = str(targets)

    # Tokenize inputs and targets
    model_inputs = state.tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=state.args_cli.max_input_length
    )
    labels = state.tokenizer(
        targets,
        truncation=True,
        padding="max_length",
        max_length=state.args_cli.max_target_length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === Argument parsing ===
def parse_args(argv=None):
    """Parse command line arguments."""
    return parser.parse_args(argv)


# === Pipeline construction ===
def build_pipeline(args, trainer_cls=RankZeroOnlySaveTrainer):
    state.args_cli = args
    # Set up logging level
    logging.basicConfig(level=logging.DEBUG if state.args_cli.debug else logging.INFO)

    rank_logger("info", "🚦 Process start: initializing training script.")
    rank_logger("info", "🚀 Starting T5 training script...")
    if not state.args_cli.train_dataset_dir:
        parser.error("You must provide --train-dataset-dir.")
    rank_logger("info", f"🧠 Using model checkpoint: {state.args_cli.model_checkpoint}")
    rank_logger("info", "🧮 Initializing dataset preprocessing and tokenization pipeline...")

    if state.args_cli.output_dir is None:
        state.args_cli.output_dir = f"{state.args_cli.task_name}_model"

    # ----------------- TOKENIZER LOAD (must occur before tokenization) -----------------
    model_checkpoint = state.args_cli.model_checkpoint
    rank_logger("info", "🔤 Loading tokenizer...")
    state.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    rank_logger("info", "✅ Tokenizer loaded.")

    # Only load from disk (HuggingFace datasets), or from CSV/JSON if specified
    # Load train dataset: handle .jsonl, .csv, or dataset dir
    if state.args_cli.train_dataset_dir.endswith(".jsonl"):
        rank_logger("info", "📄 Detected JSONL format for training dataset.")
        state.train_dataset = load_dataset("json", data_files=state.args_cli.train_dataset_dir, split="train")
    elif state.args_cli.train_dataset_dir.endswith(".csv"):
        rank_logger("info", "📄 Detected CSV format for training dataset.")
        state.train_dataset = load_dataset("csv", data_files=state.args_cli.train_dataset_dir, split="train")
        # Ensure CSV columns are Python str
        state.train_dataset = state.train_dataset.cast_column(state.args_cli.input_col, Value("string"))
        state.train_dataset = state.train_dataset.cast_column(state.args_cli.target_col, Value("string"))
    else:
        rank_logger("info", "📂 Loading HuggingFace dataset from disk...")
        state.train_dataset = Dataset.load_from_disk(state.args_cli.train_dataset_dir)

    if state.args_cli.eval_dataset_dir:
        eval_path = Path(state.args_cli.eval_dataset_dir)
        if eval_path.suffix == ".csv":
            import pandas as pd
            rank_logger("info", "📄 Detected CSV format for validation dataset.")
            df = pd.read_csv(eval_path)
            state.validation_dataset = Dataset.from_pandas(df)
            # Ensure CSV columns are Python str
            state.validation_dataset = state.validation_dataset.cast_column(state.args_cli.input_col, Value("string"))
            state.validation_dataset = state.validation_dataset.cast_column(state.args_cli.target_col, Value("string"))
        elif eval_path.suffix == ".json":
            import pandas as pd
            rank_logger("info", "📄 Detected JSON format for validation dataset.")
            df = pd.read_json(eval_path)
            state.validation_dataset = Dataset.from_pandas(df)
        else:
            state.validation_dataset = Dataset.load_from_disk(state.args_cli.eval_dataset_dir)
        rank_logger("info", f"✅ Loaded train ({len(state.train_dataset)}), validation ({len(state.validation_dataset)}) from disk or file.")
    else:
        total_examples = len(state.train_dataset)
        eval_size = int(total_examples * state.args_cli.validation_size)
        rank_logger("info", f"📊 Splitting train into train/test: total {total_examples}, test size {eval_size} ({(eval_size/total_examples)*100:.2f}%)")
        split = state.train_dataset.train_test_split(test_size=eval_size/total_examples, seed=42)
        state.train_dataset = split["train"]
        state.test_dataset = split["test"]

    # Check if dataset is already tokenized
    state.is_tokenized = all(
        col in state.train_dataset.column_names for col in ["input_ids", "attention_mask"]
    )
    # Sanity check: no empty input or output rows
    if not state.is_tokenized:
        num_empty_input = sum([str(x).strip() == "" for x in state.train_dataset[state.args_cli.input_col]])
        num_empty_output = sum([str(x).strip() == "" for x in state.train_dataset[state.args_cli.target_col]])
        if num_empty_input > 0:
            raise ValueError(f"❌ Found {num_empty_input} empty input rows in '{state.args_cli.input_col}' — these must be removed or filled.")
        if num_empty_output > 0 and not state.args_cli.allow_empty_output:
            raise ValueError(f"❌ Found {num_empty_output} empty output rows in '{state.args_cli.target_col}'. Use --allow-empty-output to bypass.")
    else:
        rank_logger("info", "⚡ Dataset is pre-tokenized — skipping empty input/output validation.")

    # Warn if user requests a long context but model is not a known long-context model
    if state.args_cli.max_input_length > 2048 and not any(s in model_checkpoint.lower() for s in ["longt5", "long-t5", "tglobal"]):
        rank_logger("warning", f"⚠️ Specified max_input_length={state.args_cli.max_input_length} but model '{model_checkpoint}' may not support long contexts. Proceed with caution.")

    rank_logger("info", "🧠 Loading model...")
    # Prepare model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    rank_logger("info", "✅ Model loaded.")
    # Suppress warning about use_cache with gradient checkpointing
    model.config.use_cache = False

    rouge_metric = evaluate.load("rouge")
    if state.args_cli.calculate_meteor:
        meteor_metric = evaluate.load("meteor")

    def compute_structured_metrics(pred, fields_config):
        predictions = pred.predictions
        labels = pred.label_ids

        decoded_preds = state.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = state.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Attempt to parse to JSON
        structured_preds = []
        structured_labels = []
        for p, l in zip(decoded_preds, decoded_labels):
            try:
                structured_preds.append(json.loads(p))
                structured_labels.append(json.loads(l))
            except json.JSONDecodeError:
                structured_preds.append({})
                structured_labels.append({})

        from collections import defaultdict
        if not structured_preds or not structured_labels:
            rank_logger("warning", "⚠️ Empty or malformed structured predictions; returning empty metric set.")
            return {}

        metrics = defaultdict(list)

        for field, metric_type in fields_config.items():
            for pred_item, label_item in zip(structured_preds, structured_labels):
                pred_val = pred_item.get(field, "").strip()
                label_val = label_item.get(field, "").strip()

                if metric_type == "exact":
                    metrics[f"{field}_exact"].append(1.0 if pred_val == label_val else 0.0)
                elif metric_type == "f1":
                    pred_tokens = set(pred_val.lower().split())
                    label_tokens = set(label_val.lower().split())
                    tp = len(pred_tokens & label_tokens)
                    precision = tp / len(pred_tokens) if pred_tokens else 0
                    recall = tp / len(label_tokens) if label_tokens else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                    metrics[f"{field}_f1"].append(f1)
                elif metric_type == "rouge":
                    result = rouge_metric.compute(predictions=[pred_val], references=[label_val], use_stemmer=True)
                    val = result['rougeL']['fmeasure'] if isinstance(result['rougeL'], dict) else result['rougeL']
                    metrics[f"{field}_rougeL"].append(val)
                else:
                    metrics[f"{field}_unknown"].append(0.0)

        final_metrics = {}
        for k, vals in metrics.items():
            final_metrics[k] = sum(vals) / len(vals)

        return final_metrics

    def compute_metrics(eval_pred):
        """Compute evaluation metrics (ROUGE, METEOR if enabled, and combined) for predictions and labels."""
        from evaluate import load as load_metric

        rouge = load_metric("rouge")
        meteor = load_metric("meteor") if state.args_cli.calculate_meteor else None

        predictions, labels = eval_pred
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        labels_ = np.where(labels != -100, labels, state.tokenizer.pad_token_id)
        predictions_ = np.where(predictions > state.tokenizer.vocab_size, state.tokenizer.pad_token_id, predictions)
        predictions_ = np.clip(predictions_, 0, state.tokenizer.vocab_size)
        labels_ = np.where(labels_ > state.tokenizer.vocab_size, state.tokenizer.pad_token_id, labels_)
        labels_ = np.clip(labels_, 0, state.tokenizer.vocab_size)

        decoded_preds = state.tokenizer.batch_decode(predictions_, skip_special_tokens=True)
        decoded_labels = state.tokenizer.batch_decode(labels_, skip_special_tokens=True)

        import nltk
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        metrics = {
            "eval_rouge1": result["rouge1"].mid.fmeasure if hasattr(result["rouge1"], "mid") else result["rouge1"],
            "eval_rouge2": result["rouge2"].mid.fmeasure if hasattr(result["rouge2"], "mid") else result["rouge2"],
            "eval_rougeL": result["rougeL"].mid.fmeasure if hasattr(result["rougeL"], "mid") else result["rougeL"],
            "eval_rougeLsum": result["rougeLsum"].mid.fmeasure if hasattr(result["rougeLsum"], "mid") else result["rougeLsum"],
        }

        if state.args_cli.calculate_meteor:
            meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            metrics["eval_meteor"] = meteor_result["meteor"]
            metrics["eval_combined"] = 0.5 * metrics["eval_rougeL"] + 0.5 * metrics["eval_meteor"]
        return metrics

    rank_logger("info", "🪄 Starting dataset tokenization...")
    report_memory()
    if not state.is_tokenized:
        rank_logger("info", "🔄 Dataset not yet tokenized — applying preprocessing...")
        state.train_dataset = tokenize_dataset(state.train_dataset, preprocess_t5, state.args_cli, "🧠 Tokenizing train set")
        state.test_dataset = tokenize_dataset(state.test_dataset, preprocess_t5, state.args_cli, "🧠 Tokenizing test set")
        if state.args_cli.eval_dataset_dir:
            state.validation_dataset = tokenize_dataset(state.validation_dataset, preprocess_t5, state.args_cli, "🧠 Tokenizing validation set")
    else:
        rank_logger("info", "⚡ Detected pre-tokenized dataset — skipping tokenization.")
    report_memory()

    state.train_dataset = state.train_dataset.filter(lambda x: len(x["input_ids"]) <= state.args_cli.max_input_length)
    if state.args_cli.eval_dataset_dir:
        state.validation_dataset = state.validation_dataset.filter(lambda x: len(x["input_ids"]) <= state.args_cli.max_input_length)
    else:
        state.test_dataset = state.test_dataset.filter(lambda x: len(x["input_ids"]) <= state.args_cli.max_input_length)
    if state.args_cli.eval_dataset_dir:
        rank_logger("info", f"✅ Tokenization complete: train {len(state.train_dataset):,} examples, validation {len(state.validation_dataset):,} examples")
    else:
        rank_logger("info", f"✅ Tokenization complete: train {len(state.train_dataset):,} examples, test {len(state.test_dataset):,} examples")

    state.train_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    if state.args_cli.eval_dataset_dir:
        state.validation_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    else:
        state.test_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])

    model_name = state.args_cli.model_checkpoint.split("/")[-1]
    dataset_name = Path(state.args_cli.train_dataset_dir).stem

    if state.args_cli.batch_size is not None:
        base_batch_size = state.args_cli.batch_size
    else:
        base_batch_size = determine_batch_size(state.args_cli.model_checkpoint, False)
    rank_logger("info", f"📦 Auto-scaled batch size: using batch size {base_batch_size}")

    effective_batch_size = state.args_cli.batch_size * state.args_cli.gradient_accumulation_steps * get_world_size_safe()
    if state.args_cli.eval_strategy == "epoch":
        eval_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
        save_steps = None
        rank_logger("info", "🔁 Using epoch-based evaluation and checkpointing (--eval_strategy=epoch).")
    else:
        steps_per_epoch = math.ceil(len(state.train_dataset) / effective_batch_size)
        eval_steps = max(min(steps_per_epoch, 10_000) // 3, 1)
        save_steps = eval_steps
        eval_strategy = "steps"
        save_strategy = "steps"
        rank_logger("info", f"🔢 Using dynamic step-based evaluation/checkpointing every {eval_steps} steps (--eval_strategy=auto).")
    if state.args_cli.eval_steps is not None and state.args_cli.eval_strategy != "epoch":
        eval_steps = state.args_cli.eval_steps
        save_steps = state.args_cli.eval_steps
        rank_logger("info", f"📏 Overriding eval/save steps: {eval_steps}")

    if state.args_cli.fields:
        rank_logger("info", f"🧪 Structured metric evaluation active: {state.args_cli.fields}")

    optim_type = "adamw_hf" if state.args_cli.disable_adafactor else "adafactor"

    training_args = Seq2SeqTrainingArguments(
        run_name=f"{state.args_cli.task_name}-{model_name}-{dataset_name}-bs{base_batch_size}-lr{state.args_cli.learning_rate}-ws{state.args_cli.warm_up_steps}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{DEFAULT_MAX_TARGET_LENGTH}-{DEFAULT_MAX_INPUT_LENGTH}",
        logging_dir=f"logs/{state.args_cli.task_name}-{model_name}-{dataset_name}-bs{base_batch_size}-lr{state.args_cli.learning_rate}-ws{state.args_cli.warm_up_steps}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{DEFAULT_MAX_TARGET_LENGTH}-{DEFAULT_MAX_INPUT_LENGTH}",
        output_dir=state.args_cli.output_dir,
        per_device_train_batch_size=base_batch_size,
        per_device_eval_batch_size=base_batch_size,
        num_train_epochs=state.args_cli.total_epochs,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=15,
        logging_steps=50,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="combined" if state.args_cli.calculate_meteor else "rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        bf16=state.args_cli.bf16,
        learning_rate=state.args_cli.learning_rate,
        warmup_steps=state.args_cli.warm_up_steps,
        lr_scheduler_type=state.args_cli.lr_scheduler_type,
        gradient_accumulation_steps=state.args_cli.gradient_accumulation_steps,
        deepspeed=state.args_cli.deepspeed,
        optim=optim_type,
        generation_max_length=min(state.args_cli.max_target_length, 256),
        generation_num_beams=1,
    )

    writer = SummaryWriter(log_dir=str(training_args.logging_dir))

    rank_logger("info", "🏋️ Beginning training loop...")
    model.gradient_checkpointing_enable()

    collator = DataCollatorWithPadding(tokenizer=state.tokenizer, padding="max_length", max_length=state.args_cli.max_input_length)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=state.train_dataset,
        eval_dataset=state.test_dataset,
        processing_class=state.tokenizer,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=state.args_cli.early_stopping_patience,
                early_stopping_threshold=state.args_cli.min_delta if state.args_cli.min_delta is not None else default_stopping_delta(state.args_cli),
            ),
            EpochNormalizedLogger(writer),
            MemoryUsageLogger(model, state.args_cli.model_checkpoint, base_batch_size, input_size=512),
        ],
        compute_metrics=compute_metrics,
    )

    return state, model, collator, trainer


# === Training execution ===
def run_training(trainer):
    trainer.train()


def main(argv=None):
    args = parse_args(argv)
    _, _, _, trainer = build_pipeline(args)
    run_training(trainer)

# === Entrypoint ===
if __name__ == "__main__":
    main()
