import json
import argparse
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from torch.nn.functional import softmax
import numpy as np
import logging
logger = logging.getLogger(__name__)
import os
from datetime import datetime
import shutil
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
from torch.utils.tensorboard import SummaryWriter
from .callbacks import EpochNormalizedLogger, MemoryUsageLogger, ManualEarlyStopCallback
from .evaluation_utils import calculate_dynamic_eval_steps
from .utils import load_and_filter_dataframe, determine_batch_size
import warnings
from transformers import logging as hf_logging
import copy
import urllib3
urllib3.disable_warnings()

# 1. Leave Python FutureWarnings enabled (the default), so you still see deprecation notices:
warnings.filterwarnings("default", category=FutureWarning, module="transformers")

# 2. Silence all HF logging below WARNING:
hf_logging.set_verbosity_warning()

# Reduce urllib3 log spam
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Reduce filelock log spam (Triton autotune)
logging.getLogger("filelock").setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

grad_log = []


# Classification Trainer
class ClassificationTrainer(Trainer):
    def __init__(self, *args, args_cli=None, label_encoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.args_cli = args_cli
        self.label_encoder = label_encoder

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Ensure DistilBERT doesn't receive unexpected arguments like 'labels'
        model_input_keys = ["input_ids", "attention_mask", "token_type_ids"]
        inputs_clean = {k: v for k, v in inputs.items() if k in model_input_keys}
        outputs = model(**inputs_clean)
        logits = outputs.logits
        labels = inputs["labels"].long()

        # --- Dynamic class weighting for weak/strong classes (removed)
        class_weights = None

        # Compute individual sample loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=logits.device) if class_weights is not None else None,
            reduction='none'
        )

        # --- Focal loss integration (removed) ---
        loss = loss_fct(logits, labels)

        # Apply penalties for a specific class if configured and no other loss mods active
        if getattr(self.args_cli, "penalize_class_index", None) is not None:
            preds = torch.argmax(logits, dim=1)
            class_idx = self.args_cli.penalize_class_index
            penalty_mask = (labels == class_idx) | (preds == class_idx)
            fn_mask = (labels == class_idx) & (preds != class_idx)

            weights = torch.ones_like(loss)
            fn_weights = torch.where(fn_mask, torch.tensor(3.0, device=weights.device), weights)
            penalty_weights = torch.where(penalty_mask, torch.tensor(2.0, device=weights.device), torch.tensor(1.0, device=weights.device))
            weights = torch.max(fn_weights, penalty_weights)
            weighted_loss = (loss * weights).mean()

            # Additional penalties for entropy and softmax margin if class matches
            mask = labels == class_idx
            probs = softmax(logits, dim=1)
            if mask.any():
                selected_probs = probs[mask]
                selected_probs = torch.clamp(selected_probs, min=1e-8)
                entropy = -(selected_probs * torch.log(selected_probs)).sum(dim=1)
                if torch.isfinite(entropy).all():
                    entropy_penalty = entropy.mean()
                    weighted_loss += 0.75 * entropy_penalty.detach()
                topk = torch.topk(logits[mask], 2, dim=1)
                margin = topk.values[:, 0] - topk.values[:, 1]
                competition_penalty = (1.0 - torch.tanh(margin)).mean()
                weighted_loss += 0.4 * competition_penalty.detach()
        else:
            weighted_loss = loss.mean()

        if num_items_in_batch is not None:
            weighted_loss = weighted_loss / num_items_in_batch

        # Diagnostic block for NaN/inf losses
        if not torch.isfinite(weighted_loss):
            print("🛑 Non-finite weighted_loss detected!")
            print("Logits:", logits.detach().cpu())
            print("Labels:", labels.detach().cpu())
            print("Loss before weighting:", loss.detach().cpu())
            print("Weights:", weights.detach().cpu() if 'weights' in locals() else 'N/A')
            print("Entropy penalty:", entropy_penalty.item() if 'entropy_penalty' in locals() else 'N/A')
            print("Competition penalty:", competition_penalty.item() if 'competition_penalty' in locals() else 'N/A')
            raise ValueError("Non-finite weighted_loss encountered.")

        # if num_items_in_batch is None:
        #     print(f"End of loss: {return_outputs}: {(weighted_loss, logits, labels) if return_outputs else weighted_loss}")
        return (weighted_loss, {"predictions": logits, "label_ids": labels}) if return_outputs else weighted_loss

# Regression Trainer
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("regression_labels")
        outputs = model(**inputs)
        loss = outputs.regression_loss

        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        return (loss, outputs.regression_scores, labels) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-batching", action="store_true", help="Disable batching to reduce VRAM usage")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--output-dir", type=str, default="bert_model", help="Output directory for model and encoder")
    parser.add_argument("--clean", action="store_true", help="Remove output_dir before training if it exists")
    parser.add_argument("--total-epochs", type=int, default=15, help="Total number of training epochs (default: 15)")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification", help="Specify which head to train")
    parser.add_argument("--model-checkpoint", type=str, default="microsoft/deberta-v3-base", help="HuggingFace model checkpoint to use")
    parser.add_argument("--data-path", type=str, default="training_data.jsonl", help="Path to training data JSONL file")
    parser.add_argument("--balance-class-ratio", type=float, default=None,
                        help="Balance tone classes such that no class exceeds the smallest by more than this ratio (e.g., 1.0 = equal)")
    # New arguments:
    # New evaluation-path argument
    parser.add_argument("--evaluation-path", type=str, default=None, help="Optional evaluation dataset path for post-training scoring")
    # New argument to penalize a specific class
    parser.add_argument("--penalize-class", type=str, default=None,
        help="Class label (e.g. 'Neutral') to apply extra penalties during loss computation.")
    # New argument for label field
    parser.add_argument("--label-field", type=str, default="label", help="Name of the label column in the input dataset (default: 'label')")
    # New argument for label list
    parser.add_argument("--label-list", type=str, default=None,
                        help="Optional comma-separated list of labels to clamp to. If not provided, all seen labels are used.")
    # New argument for text field
    parser.add_argument("--text-field", type=str, default="text", help="Name of the input text field (default: 'text')")
    parser.add_argument("--alternate_field", type=str, default=None)
    parser.add_argument("--logging-dir", type=str, default="logs", help="Directory for TensorBoard logs")
    parser.add_argument("--stratify", action="store_true", help="Enable stratified split by label for eval set")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default is auto-detected)")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable BF16 mixed precision training (default: FP32)")
    args = parser.parse_args()
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # --- Unified data loading logic using HuggingFace datasets ---
    from datasets import load_dataset, Dataset
    import os
    # Unified loading: supports both HuggingFace and local files
    if args.data_path.startswith("hf:"):
        hf_path = args.data_path[3:]  # Strip the hf: prefix
        logger.info(f"📃 Loading HuggingFace dataset: {hf_path}")
        dataset = load_dataset(hf_path)
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            parser.error(f"❌ Provided data path does not exist: {data_path}")
        if len(sys.argv) == 1:
            parser.print_help()
            exit(0)
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
        if args.data_path.endswith(".json") or args.data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=args.data_path)
        elif args.data_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=args.data_path)
        else:
            raise ValueError(f"Unsupported file type for data_path: {args.data_path}")
    # Always assign train_dataset from the 'train' split (or entire dataset if only one split)
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        # If it's a single split, use as train
        train_dataset = dataset
    eval_dataset = None
    # If an evaluation path is provided, load it
    if args.evaluation_path:
        if args.evaluation_path.startswith("hf:"):
            hf_eval_path = args.evaluation_path[3:]
            logger.info(f"📃 Loading HuggingFace eval dataset: {hf_eval_path}")
            eval_dataset = load_dataset(hf_eval_path)
            if "train" in eval_dataset:
                eval_dataset = eval_dataset["train"]
        else:
            eval_path = Path(args.evaluation_path)
            if not eval_path.exists():
                parser.error(f"❌ Provided evaluation path does not exist: {eval_path}")
            if eval_path.suffix in [".json", ".jsonl"]:
                eval_dataset = load_dataset("json", data_files=str(eval_path))["train"]
            elif eval_path.suffix == ".csv":
                eval_dataset = load_dataset("csv", data_files=str(eval_path))["train"]
            else:
                raise ValueError(f"Unsupported evaluation file type: {eval_path.suffix}")

    # --- Dataset statistics logging ---
    from collections import Counter
    def log_dataset_stats(name, dataset, label_field):
        if dataset is None:
            return
        label_counts = Counter(dataset[label_field])
        logger.info(f"📦 {name} set: {len(dataset)} examples")
        logger.info(f"🧮 {name} label distribution: {dict(label_counts)}")

    # --- Label list logic ---
    # Get label list from argument or from dataset
    label_list = None
    if args.label_list:
        label_list = [lbl.strip() for lbl in args.label_list.split(",")]
        logger.info(f"🔖 Using label list from --label-list: {label_list}")
    else:
        # Get unique labels from train_dataset
        label_list = sorted(set(train_dataset.unique(args.label_field)))
        logger.info(f"🔖 No --label-list provided, using all observed labels: {label_list}")
    # --- Validate label list is not empty
    if not label_list:
        logger.error("❌ No valid labels found in dataset after filtering.")
        sys.exit(1)
    # Optionally balance classes (classification only)
    if args.balance_class_ratio and args.task == "classification":
        logger.info(f"🎯 Balancing classes using ratio {args.balance_class_ratio:.2f}")
        # Filter to label_list
        train_dataset = train_dataset.filter(lambda x: x[args.label_field] in label_list)
        # Count per class
        from collections import Counter
        counts = Counter(train_dataset[args.label_field])
        min_count = min(counts.values())
        max_allowed = int(min_count * args.balance_class_ratio)
        # For each label, keep only up to max_allowed
        def keep_limited(example, counts_so_far={}):
            l = example[args.label_field]
            if l not in counts_so_far:
                counts_so_far[l] = 0
            if counts_so_far[l] < max_allowed:
                counts_so_far[l] += 1
                return True
            return False
        filtered_indices = []
        counts_so_far = {}
        for idx, l in enumerate(train_dataset[args.label_field]):
            if l not in counts_so_far:
                counts_so_far[l] = 0
            if counts_so_far[l] < max_allowed:
                filtered_indices.append(idx)
                counts_so_far[l] += 1
        train_dataset = train_dataset.select(filtered_indices)
        # Log final per-class counts
        final_counts = Counter(train_dataset[args.label_field])
        logger.info(f"   Minimum class size = {min_count}")
        logger.info(f"   Class caps enforced at {max_allowed} per label")
        logger.info(f"   Final label distribution:")
        for tone in label_list:
            logger.info(f"     {tone:<12}: {final_counts.get(tone, 0)}")
        logger.info(f"   Total training samples: {len(train_dataset)}")
    # Log stats for raw train/eval datasets before encoding or filtering
    log_dataset_stats("Train", train_dataset, args.label_field)
    log_dataset_stats("Eval", eval_dataset, args.label_field)

    # Encode labels as classification label
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        logger.info(f"🧹 Removing existing model output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / f"{args.label_field}_label_encoder.json"
    # Internal column name for encoded labels
    label_column_internal = "label_index"
    label_encoder = LabelEncoder()
    # Filter to label_list
    train_dataset = train_dataset.filter(lambda x: x[args.label_field] in label_list)
    label_encoder.fit(label_list)
    # Add label_index column using map
    class2idx = {c: i for i, c in enumerate(label_encoder.classes_)}
    def add_label_index(example):
        example[label_column_internal] = int(class2idx[example[args.label_field]])
        return example
    train_dataset = train_dataset.map(add_label_index)
    # --- Penalize class logic ---
    penalize_class_index = None
    if args.penalize_class:
        if args.penalize_class not in label_encoder.classes_:
            raise ValueError(f"❌ Specified --penalize-class '{args.penalize_class}' not in known classes: {list(label_encoder.classes_)}")
        penalize_class_index = list(label_encoder.classes_).index(args.penalize_class)
        logger.info(f"⚖️ Will apply special loss handling to class: {args.penalize_class} (index {penalize_class_index})")
    args.penalize_class_index = penalize_class_index
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump({"classes": [int(x) if isinstance(x, (np.integer,)) else x for x in label_encoder.classes_]}, f)
    # --- Sanity check ---
    if len(train_dataset) == 0:
        logger.error("❌ No data remaining after label filtering. Cannot proceed with training.")
        sys.exit(1)
    # --- Split train/eval if needed ---
    if eval_dataset is None:
        split = train_dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
        logger.info("📊 Using HuggingFace random split for eval set.")
    else:
        # Filter eval_dataset to label_list, add label_index
        eval_dataset = eval_dataset.filter(lambda x: x[args.label_field] in label_list)
        eval_dataset = eval_dataset.map(add_label_index)
        val_dataset = eval_dataset

    # Log stats for validation set if present
    log_dataset_stats("Validation", val_dataset, args.label_field)
    # Optionally save eval set (for reference)
    eval_jsonl_path = output_dir / "last_eval_set.jsonl"
    val_dataset.to_json(str(eval_jsonl_path))
    logger.info(f"📝 Saved eval dataset used during training to: {eval_jsonl_path}")

    # Tokenizer
    model_checkpoint = args.model_checkpoint
    # Use DebertaV2Tokenizer only if "deberta" in model_checkpoint, else use AutoTokenizer with use_fast logic
    if "deberta" in model_checkpoint.lower():
        print(f"🔍 Forcing slow tokenizer for {model_checkpoint}")
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_checkpoint, use_fast=False)
    else:
        use_fast = not any(name in model_checkpoint.lower() for name in ["xlnet", "t5", "bart"])
        print(f"🔍 Loading tokenizer for {model_checkpoint} | use_fast={use_fast}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=use_fast
        )
    def tokenize_fn(example):
        max_length = 1024 if "deberta" in model_checkpoint.lower() else 512
        return tokenizer(
            example[args.text_field],
            example[args.alternate_field] if args.alternate_field is not None else None,
            truncation=True,
            max_length=max_length
        )
    # --- Optional holdout evaluation set ---
    monitor_eval_loader = None
    if args.evaluation_path:
        logger.info(f"🧪 Loading holdout monitoring set: {args.evaluation_path}")
        eval_path = args.evaluation_path
        if eval_path.startswith("hf:"):
            hf_eval_path = eval_path[3:]
            holdout_dataset = load_dataset(hf_eval_path)
            if "train" in holdout_dataset:
                holdout_dataset = holdout_dataset["train"]
        else:
            eval_path = Path(eval_path)
            if eval_path.suffix in [".jsonl", ".json"]:
                holdout_dataset = load_dataset("json", data_files=str(eval_path))["train"]
            elif eval_path.suffix == ".csv":
                holdout_dataset = load_dataset("csv", data_files=str(eval_path))["train"]
            else:
                raise ValueError(f"Unsupported evaluation file type: {eval_path.suffix}")
        holdout_dataset = holdout_dataset.filter(lambda x: x[args.label_field] in label_list)
        holdout_dataset = holdout_dataset.map(add_label_index)
        tokenized_holdout = holdout_dataset.map(tokenize_fn, batched=True)
        tokenized_holdout = tokenized_holdout.rename_column(label_column_internal, "labels")
        tokenized_holdout.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        from torch.utils.data import DataLoader
        monitor_eval_loader = DataLoader(tokenized_holdout, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))
    # Tokenize both splits
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)
    if args.task == "classification":
        tokenized_train = tokenized_train.rename_column(label_column_internal, "labels")
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized_train = tokenized_train.shuffle(seed=42)
        tokenized_val = tokenized_val.rename_column(label_column_internal, "labels")
        tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        def format_labels(example):
            example["labels"] = example[args.label_field]
            example[label_column_internal] = int(example[label_column_internal])
            example[label_column_internal] = np.int64(example[label_column_internal])
            return example
        tokenized_train = tokenized_train.map(format_labels)
        tokenized_train = tokenized_train.rename_column("labels", "regression_labels")
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "regression_labels", label_column_internal])
        tokenized_train = tokenized_train.shuffle(seed=42)
        tokenized_val = tokenized_val.map(format_labels)
        tokenized_val = tokenized_val.rename_column("labels", "regression_labels")
        tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "regression_labels", label_column_internal])

    from transformers import AutoConfig, AutoModelForSequenceClassification
    config = AutoConfig.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

    base_lr = 5e-5
    # Setup parameter groups
    optimizer_grouped_parameters = [{"params": model.parameters(), "lr": base_lr}]
    model_name = args.model_checkpoint.split("/")[-1]
    dataset_name = Path(args.data_path).stem
    ratio_tag = f"r{args.balance_class_ratio}"
    run_name = f"{model_name}-{dataset_name}-{ratio_tag}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    #
    # Determine batch size from shared utility
    base_batch_size = args.batch_size or determine_batch_size(args.model_checkpoint, args.no_batching)
    logger.info(f"📦 Auto-scaled batch size: using batch size {base_batch_size}")
    dynamic_eval_steps = calculate_dynamic_eval_steps(len(tokenized_train), base_batch_size)
    logger.info(f"📊 Eval every {dynamic_eval_steps} steps based on ~1/3 epoch heuristic.")

    hf_args = TrainingArguments(
        run_name=run_name,
        logging_dir=f"logs/{run_name}",
        output_dir=args.output_dir,
        per_device_train_batch_size=base_batch_size,
        per_device_eval_batch_size=base_batch_size,
        num_train_epochs=args.total_epochs,
        eval_strategy="steps",
        eval_steps=dynamic_eval_steps,
        save_strategy="steps",
        save_steps=dynamic_eval_steps,
        save_total_limit=999,
        logging_steps=200,
        report_to="tensorboard",
        prediction_loss_only=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        warmup_steps=400,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=args.bf16,
        include_for_metrics=["loss"],
        label_names=["labels"],
    )

    trainer_class = ClassificationTrainer if args.task == "classification" else RegressionTrainer

    def compute_metrics(eval_preds):
        from sklearn.metrics import f1_score, classification_report
        logits = eval_preds.predictions
        labels = eval_preds.label_ids

        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(labels, tuple):
            labels = labels[0]

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        preds = torch.argmax(logits, dim=1)
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        # Compute F1 per class
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), output_dict=True, zero_division=0)
        per_class_f1 = {
            f"f1_{label_encoder.classes_[int(k)]}": v["f1-score"]
            for k, v in report.items()
            if k.isdigit() and int(k) < len(label_encoder.classes_)
        }

        # Log per-class F1 to TensorBoard under class_eval/
        for k, v in per_class_f1.items():
            writer.add_scalar(f"class_eval/{k}", v, trainer.state.global_step)

        result = {"eval_loss": loss.item(), "eval_f1": f1}
        result.update(per_class_f1)
        return result

    global writer  # So it can be used in compute_metrics
    writer = SummaryWriter(log_dir=args.logging_dir)

    # --- ExtraEvalCallback definition ---
    class ExtraEvalCallback(TrainerCallback):
        def __init__(self, model, dataloader, writer, name="monitor_eval"):
            self.model = model
            self.dataloader = dataloader
            self.writer = writer
            self.name = name

        def on_evaluate(self, args, state, control, **kwargs):
            self.model.eval()
            all_preds, all_labels = [], []
            for batch in self.dataloader:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch["input_ids"].to(self.model.device),
                        attention_mask=batch["attention_mask"].to(self.model.device),
                    )
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    labels = batch["labels"].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            self.writer.add_scalar(f"{self.name}/accuracy", acc, state.global_step)
            self.writer.add_scalar(f"{self.name}/f1_macro", f1, state.global_step)

    trainer = trainer_class(
        model=model,
        args=hf_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(torch.optim.AdamW(optimizer_grouped_parameters), None),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=7),
            EpochNormalizedLogger(writer),
            MemoryUsageLogger(model, args.model_checkpoint, base_batch_size, input_size=(1024 if "deberta" in model_checkpoint.lower() else 512)),
            *([ExtraEvalCallback(model, monitor_eval_loader, writer)] if monitor_eval_loader else []),
            ManualEarlyStopCallback(stop_file=os.path.join(args.output_dir, "stop_training.txt")),
        ],
        compute_metrics=compute_metrics,
        args_cli=args,
        label_encoder=label_encoder,
    )

    logger.info(f"✅ Label class count (from encoder): {len(label_encoder.classes_)}")
    if hasattr(model, "head") and hasattr(model.head, "classification"):
        logger.info(f"✅ Model output classes: {model.head.classification[-1].out_features}")
    elif hasattr(model, "classifier"):
        logger.info(f"✅ Model output classes: {model.classifier.out_features}")
    else:
        model_name = getattr(model.config, "architectures", ["Unknown"])[0]
        logger.warning(f"⚠️ Could not determine model output classes for model type: {model_name}")
    logger.info(f"✅ Max label index in dataset: {max(train_dataset[label_column_internal])}")

    trainer.train()

    # Reload best checkpoint to ensure we save the best model's head
    best_path = trainer.state.best_model_checkpoint
    logger.info(f"🌟 Reloading best model weights from: {best_path}")
    if os.path.exists(os.path.join(best_path, "model.safetensors")):
        logger.info("💾 Loading full model weights from model.safetensors")
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(os.path.join(best_path, "model.safetensors")), strict=False)
    elif os.path.exists(os.path.join(best_path, "pytorch_model.bin")):
        logger.info("💾 Loading full model weights from pytorch_model.bin")
        model.load_state_dict(torch.load(os.path.join(best_path, "pytorch_model.bin"), weights_only=True), strict=False)
    else:
        logger.warning("⚠️ No model weight file found in checkpoint. Skipping model weight load.")

    # Save final model to versioned path
    logger.info(f"🌟 Best model loaded from: {trainer.state.best_model_checkpoint}")
    root = Path(args.output_dir)
    i = 1
    while Path(f"{root}-v{i}").exists():
        i += 1
    final_path = Path(f"{root}-v{i}")
    final_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Saving full model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(final_path / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump({"classes": [int(x) if isinstance(x, (np.integer,)) else x for x in label_encoder.classes_]}, f)

    logger.info(f"✅ Saved final model to {final_path}")

    # --- Final evaluation with best model ---
    if monitor_eval_loader:
        logger.info(f"📊 Final evaluation on monitoring set using best model:")
        model.eval()
        all_preds, all_labels = [], []
        for batch in monitor_eval_loader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                )
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
        report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, digits=2)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)

        print("\n🎯 Final Monitoring Set Metrics")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1 : {macro_f1:.4f}")
        print("\nPer-Class Report:\n" + report)
        print("Confusion Matrix:\n", conf_matrix)


# Allow running as a script
if __name__ == "__main__":
    main()
