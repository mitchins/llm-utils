import json
import numpy as np  # Ensure this import is at the top
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoConfig
from transformers import DataCollatorWithPadding
import logging
import os
from datetime import datetime
from tqdm.auto import tqdm
import psutil

# === Default hyperparameter constants ===
DEFAULT_WARMUP_STEPS = 500
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_TOTAL_EPOCHS = 30
DEFAULT_VALIDATION_SIZE = 0.10
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_MIN_DELTA = 0.1
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_TARGET_LENGTH = 128

from transformers import EarlyStoppingCallback
from .callbacks import EpochNormalizedLogger, MemoryUsageLogger, ManualEarlyStopCallback
from .evaluation_utils import calculate_dynamic_eval_steps
from .utils import load_and_filter_dataframe, determine_batch_size
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global counters for input length filtering
total_examples = 0
dropped_examples = 0

parser = argparse.ArgumentParser(
    description="Generic seq2seq trainer for tasks like summarization, translation, paraphrasing"
)
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--threads", type=int, default=1, help="Number of worker threads for dataset.map (default: 1)")
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
parser.add_argument("--enable-fp16", action="store_true", help="Enable mixed precision (FP16) training (default: FP32/full precision)")
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
    "--min-delta", type=float, default=DEFAULT_MIN_DELTA,
    help=f"Minimum absolute improvement on the monitored metric to reset early-stopping patience (default: {DEFAULT_MIN_DELTA})"
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

# New CLI arguments for HuggingFace datasets from disk
parser.add_argument("--train-dataset-dir", type=str, help="Path to a HuggingFace dataset directory for training (saved with save_to_disk())")
parser.add_argument("--eval-dataset-dir", type=str, help="Optional path to evaluation dataset directory (if not provided, splits train)")

def main():
    args_cli = parser.parse_args()

    # Logging at the start of the script
    logging.basicConfig(level=logging.DEBUG if args_cli.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("🚦 Process start: initializing training script.")
    logger.info("🚀 Starting T5 training script...")
    if not args_cli.train_dataset_dir:
        parser.error("You must provide --train-dataset-dir.")
    logger.info(f"🧠 Using model checkpoint: {args_cli.model_checkpoint}")
    logger.info("🧮 Initializing dataset preprocessing and tokenization pipeline...")

    if args_cli.output_dir is None:
        args_cli.output_dir = f"{args_cli.task_name}_model"

    # ----------------- TOKENIZER LOAD (must occur before tokenization) -----------------
    model_checkpoint = args_cli.model_checkpoint
    logger.info("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    logger.info("✅ Tokenizer loaded.")

    # Only load from disk (HuggingFace datasets), or from CSV/JSON if specified
    from pathlib import Path
    train_path = Path(args_cli.train_dataset_dir)
    if train_path.suffix == ".csv":
        import pandas as pd
        logger.info("📄 Detected CSV format for training dataset.")
        df = pd.read_csv(train_path)
        train_dataset = Dataset.from_pandas(df)
    elif train_path.suffix == ".json":
        import pandas as pd
        logger.info("📄 Detected JSON format for training dataset.")
        df = pd.read_json(train_path)
        train_dataset = Dataset.from_pandas(df)
    else:
        logger.info("📂 Loading HuggingFace dataset from disk...")
        train_dataset = Dataset.load_from_disk(args_cli.train_dataset_dir)

    if args_cli.eval_dataset_dir:
        eval_path = Path(args_cli.eval_dataset_dir)
        if eval_path.suffix == ".csv":
            import pandas as pd
            logger.info("📄 Detected CSV format for evaluation dataset.")
            df = pd.read_csv(eval_path)
            val_dataset = Dataset.from_pandas(df)
        elif eval_path.suffix == ".json":
            import pandas as pd
            logger.info("📄 Detected JSON format for evaluation dataset.")
            df = pd.read_json(eval_path)
            val_dataset = Dataset.from_pandas(df)
        else:
            val_dataset = Dataset.load_from_disk(args_cli.eval_dataset_dir)
        logger.info(f"✅ Loaded train ({len(train_dataset)}), eval ({len(val_dataset)}) from disk or file.")
    else:
        total_examples = len(train_dataset)
        eval_size = int(total_examples * args_cli.validation_size)
        logger.info(f"📊 Splitting train into train/val: total {total_examples}, val size {eval_size} ({(eval_size/total_examples)*100:.2f}%)")
        split = train_dataset.train_test_split(test_size=eval_size/total_examples, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]

    # Check if dataset is already tokenized
    is_tokenized = all(
        col in train_dataset.column_names for col in ["input_ids", "attention_mask"]
    )
    # Sanity check: no empty input or output rows
    if not is_tokenized:
        num_empty_input = sum([str(x).strip() == "" for x in train_dataset[args_cli.input_col]])
        num_empty_output = sum([str(x).strip() == "" for x in train_dataset[args_cli.target_col]])
        if num_empty_input > 0:
            raise ValueError(f"❌ Found {num_empty_input} empty input rows in '{args_cli.input_col}' — these must be removed or filled.")
        if num_empty_output > 0 and not args_cli.allow_empty_output:
            raise ValueError(f"❌ Found {num_empty_output} empty output rows in '{args_cli.target_col}'. Use --allow-empty-output to bypass.")
    else:
        logger.info("⚡ Dataset is pre-tokenized — skipping empty input/output validation.")

    # Warn if user requests a long context but model is not a known long-context model
    if args_cli.max_input_length > 2048 and not any(s in model_checkpoint.lower() for s in ["longt5", "long-t5", "tglobal"]):
        logger.warning(f"⚠️ Specified max_input_length={args_cli.max_input_length} but model '{model_checkpoint}' may not support long contexts. Proceed with caution.")

    logger.info("🧠 Loading model...")
    # Prepare model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    logger.info("✅ Model loaded.")
    # Suppress warning about use_cache with gradient checkpointing
    model.config.use_cache = False

    rouge_metric = evaluate.load("rouge")
    if args_cli.calculate_meteor:
        meteor_metric = evaluate.load("meteor")


    def compute_structured_metrics(pred, fields_config):
        predictions = pred.predictions
        labels = pred.label_ids

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Attempt to parse to JSON
        import json
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
            logger.warning("⚠️ Empty or malformed structured predictions; returning empty metric set.")
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

    def compute_metrics(pred):
        import numpy as np
        rank = int(os.environ.get("RANK", "0"))
        if args_cli.debug:
            logger.info(f"🧪 compute_metrics invoked on rank {rank}")
        if args_cli.fields:
            field_specs = dict(item.split(":") for item in args_cli.fields.split(","))
            return compute_structured_metrics(pred, field_specs)

        predictions = pred.predictions
        labels = pred.label_ids

        # Handle case where predictions are logits instead of token IDs
        if getattr(predictions, "ndim", None) == 3:  # shape: [batch, seq_len, vocab]
            predictions = np.argmax(predictions, axis=-1)

        # Clamp predictions to valid token ID range
        vocab_size = tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)

        try:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        except OverflowError as e:
            logger.error("❌ Overflow during decoding. Saving raw predictions to 'debug_predictions.npy'.")
            np.save("debug_predictions.npy", predictions)
            raise e

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        scale = 1.0 if args_cli.raw_metrics else 100.0

        rouge1 = result['rouge1']['fmeasure'] * scale if isinstance(result['rouge1'], dict) else result['rouge1'] * scale
        rouge2 = result['rouge2']['fmeasure'] * scale if isinstance(result['rouge2'], dict) else result['rouge2'] * scale
        rougeL = result['rougeL']['fmeasure'] * scale if isinstance(result['rougeL'], dict) else result['rougeL'] * scale

        metrics = {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
        }

        if args_cli.calculate_meteor:
            meteor_score = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)["meteor"] * scale
            metrics["meteor"] = meteor_score
            metrics["combined"] = 0.3 * rouge2 + 0.2 * rougeL + 0.5 * meteor_score

        return metrics

    def report_memory():
        mem = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(f"🧠 Current memory usage: {mem:.2f} MB")

    def preprocess_t5(example):
        model_inputs = tokenizer(
            example[args_cli.input_col],
            truncation=True,
            padding="max_length",
            max_length=args_cli.max_input_length
        )
        labels = tokenizer(
            example[args_cli.target_col],
            truncation=True,
            padding="max_length",
            max_length=args_cli.max_target_length
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize train and val splits after splitting
    logger.info("🪄 Starting dataset tokenization...")
    report_memory()
    if not is_tokenized:
        logger.info("🔄 Dataset not yet tokenized — applying preprocessing...")
        train_dataset = train_dataset.map(
            preprocess_t5,
            batched=True,
            num_proc=args_cli.threads,
            remove_columns=[args_cli.input_col, args_cli.target_col],
            desc="🧠 Tokenizing train set",
        )
        val_dataset = val_dataset.map(
            preprocess_t5,
            batched=True,
            num_proc=args_cli.threads,
            remove_columns=[args_cli.input_col, args_cli.target_col],
            desc="🧠 Tokenizing val set",
        )
    else:
        logger.info("⚡ Detected pre-tokenized dataset — skipping tokenization.")
    report_memory()
    # Filter by input length
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) <= args_cli.max_input_length)
    val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) <= args_cli.max_input_length)
    logger.info(f"✅ Tokenization complete: train {len(train_dataset):,} examples, val {len(val_dataset):,} examples")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model_name = args_cli.model_checkpoint.split("/")[-1]
    dataset_name = Path(args_cli.train_dataset_dir).stem

    # Determine batch size: user override or auto-scaled
    if args_cli.batch_size is not None:
        base_batch_size = args_cli.batch_size
    else:
        base_batch_size = determine_batch_size(args_cli.model_checkpoint, False)
    logger.info(f"📦 Auto-scaled batch size: using batch size {base_batch_size}")

    # Estimate dynamic evaluation steps: prefer 1/3 epoch, clamp at 10k or 1 epoch
    steps_per_epoch = len(train_dataset) // base_batch_size
    # Prefer evaluating every 1/3 of an epoch, but clamp properly
    third_epoch_steps = steps_per_epoch // 3
    if third_epoch_steps >= 10000:
        dynamic_eval_steps = third_epoch_steps
    else:
        dynamic_eval_steps = min(steps_per_epoch, 10000)
    print(f"🔢 Dynamically setting eval/save every {dynamic_eval_steps} steps (⅓ epoch preferred, clamped at 10k or 1 epoch).")

    if args_cli.fields:
        logger.info(f"🧪 Structured metric evaluation active: {args_cli.fields}")

    # Determine optimizer: Adafactor by default, AdamW if --disable-adafactor is set
    optim_type = "adamw_hf" if args_cli.disable_adafactor else "adafactor"

    args = Seq2SeqTrainingArguments(
        run_name=f"{args_cli.task_name}-{model_name}-{dataset_name}-bs{base_batch_size}-lr{args_cli.learning_rate}-ws{args_cli.warm_up_steps}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        logging_dir=f"logs/{args_cli.task_name}-{model_name}-{dataset_name}-bs{base_batch_size}-lr{args_cli.learning_rate}-ws{args_cli.warm_up_steps}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=base_batch_size,
        per_device_eval_batch_size=base_batch_size,
        num_train_epochs=args_cli.total_epochs,
        eval_strategy="steps",
        eval_steps=dynamic_eval_steps,
        save_strategy="steps",
        save_steps=dynamic_eval_steps,
        save_total_limit=15,
        logging_steps=50,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="combined" if args_cli.calculate_meteor else "rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        fp16=args_cli.enable_fp16,
        learning_rate=args_cli.learning_rate,
        warmup_steps=args_cli.warm_up_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
        deepspeed=args_cli.deepspeed,
        optim=optim_type,
    )

    # The run_name variable below is now redundant since it's incorporated above; remove if not used elsewhere.
    # run_name = f"{model_name}-{dataset_name}-bs{base_batch_size}-lr{args.learning_rate}-ws{args.warmup_steps}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    writer = SummaryWriter(log_dir=str(args.logging_dir))

    logger.info("🏋️ Beginning training loop...")
    # Enable gradient checkpointing to save memory (especially helpful with DeepSpeed)
    model.gradient_checkpointing_enable()
    from transformers import DataCollatorForSeq2Seq
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args_cli.early_stopping_patience,
                early_stopping_threshold=args_cli.min_delta
            ),
            EpochNormalizedLogger(writer),
            MemoryUsageLogger(model, args_cli.model_checkpoint, base_batch_size, input_size=512)
        ],
        compute_metrics=compute_metrics if int(os.environ.get("RANK", "0")) == 0 else None,
    )

    trainer.train()

    writer.close()

    # (Filtering log now occurs before split)

    # Save final model to versioned path (main process only)
    is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    if is_main_process:
        logger.info(f"🌟 Best model loaded from: {trainer.state.best_model_checkpoint}")
        root = Path(args_cli.output_dir)
        i = 1
        while Path(f"{root}-v{i}").exists():
            i += 1
        final_path = Path(f"{root}-v{i}")
        final_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"✅ Saved final model to {final_path}")

if __name__ == "__main__":
    main()
