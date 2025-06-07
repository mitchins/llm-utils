import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from torch.nn.functional import softmax
import numpy as np
from sklearn.metrics import f1_score
import logging
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

parser = argparse.ArgumentParser()
parser.add_argument("--no-batching", action="store_true", help="Disable batching to reduce VRAM usage")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--output-dir", type=str, default="ranker_model", help="Output directory for model and encoder")
parser.add_argument("--clean", action="store_true", help="Remove output_dir before training if it exists")
parser.add_argument("--total-epochs", type=int, default=15, help="Total number of training epochs (default: 15)")
parser.add_argument("--task", choices=["classification", "regression"], default="classification", help="Specify which head to train")
parser.add_argument("--model-checkpoint", type=str, default="microsoft/deberta-v3-base", help="HuggingFace model checkpoint to use")
parser.add_argument("--signal-threshold", type=float, default=None, help="Minimum score for at least one of vividness/emotion/action to retain sample")
parser.add_argument("--data-path", type=str, default="training_data.jsonl", help="Path to training data JSONL file")
parser.add_argument("--balance-class-ratio", type=float, default=None,
                    help="Balance tone classes such that no class exceeds the smallest by more than this ratio (e.g., 1.0 = equal)")
parser.add_argument("--focus-weak-classes", action="store_true", help="Dynamically increase reward for underperforming classes and reduce focus on overperformers")
parser.add_argument("--use-focal-loss", action="store_true", help="Apply focal loss instead of standard cross-entropy")
# New arguments:
# New evaluation-path argument
parser.add_argument("--evaluation-path", type=str, default=None, help="Optional evaluation dataset path for post-training scoring")
# New argument to penalize a specific class
parser.add_argument("--penalize-class", type=str, default=None,
    help="Class label (e.g. 'Neutral') to apply extra penalties during loss computation.")
# New argument for label field
parser.add_argument("--label-field", type=str, default="tone", help="Name of the label column in the input dataset (default: 'tone')")
# New argument for label list
parser.add_argument("--label-list", type=str, default=None,
                    help="Optional comma-separated list of labels to clamp to. If not provided, all seen labels are used.")
args_cli = parser.parse_args()
if args_cli.use_focal_loss and args_cli.focus_weak_classes:
    raise ValueError("❌ --use-focal-loss and --focus-weak-classes cannot be enabled at the same time. Please choose only one.")

logging.basicConfig(level=logging.DEBUG if args_cli.debug else logging.INFO)
logger = logging.getLogger(__name__)

data_path = Path(args_cli.data_path)
if data_path.suffix == ".jsonl":
    df = load_and_filter_dataframe(data_path, args_cli.signal_threshold, tone_field=args_cli.label_field)
elif data_path.suffix == ".csv":
    df = pd.read_csv(data_path)
elif data_path.suffix == ".tsv":
    df = pd.read_csv(data_path, sep="\t")
else:
    raise ValueError(f"Unsupported file type: {data_path.suffix}")

# --- Label list logic ---
if args_cli.label_list:
    label_list = [lbl.strip() for lbl in args_cli.label_list.split(",")]
    logger.info(f"🔖 Using label list from --label-list: {label_list}")
else:
    label_list = sorted(df[args_cli.label_field].unique())
    logger.info(f"🔖 No --label-list provided, using all seen labels: {label_list}")

# Balance classes using top-ranked entries if requested and task is classification
if args_cli.balance_class_ratio and args_cli.task == "classification":
    if {"vividness", "emotion", "action"}.issubset(df.columns):
        df["combined_score"] = df["vividness"] + df["emotion"] + df["action"]
    top_by_class = []
    min_count = df[df[args_cli.label_field].isin(label_list)][args_cli.label_field].value_counts().min()
    max_allowed = int(min_count * args_cli.balance_class_ratio)
    for tone in label_list:
        class_rows = df[df[args_cli.label_field] == tone]
        if "combined_score" in class_rows:
            class_rows = class_rows.sort_values(by="combined_score", ascending=False)
        top_class = class_rows.head(max_allowed)
        top_by_class.append(top_class)
    df = pd.concat(top_by_class).reset_index(drop=True)
    # Insert per-class count logging
    final_counts = df[args_cli.label_field].value_counts().to_dict()
    logger.info(f"🎯 Balanced dataset using ratio {args_cli.balance_class_ratio:.2f}:")
    logger.info(f"   Minimum class size = {min_count}")
    logger.info(f"   Class caps enforced at {max_allowed} per label")
    logger.info(f"   Final label distribution:")
    for tone in label_list:
        logger.info(f"     {tone:<12}: {final_counts.get(tone, 0)}")
    logger.info(f"   Total training samples: {len(df)}")

# Encode labels as classification label
output_dir = Path(args_cli.output_dir)
if args_cli.clean and output_dir.exists():
    logger.info(f"🧹 Removing existing model output: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
encoder_path = output_dir / "tone_label_encoder.json"

label_encoder = LabelEncoder()
df = df[df[args_cli.label_field].isin(label_list)].reset_index(drop=True)
label_encoder.fit(label_list)
df["tone_label"] = label_encoder.transform(df[args_cli.label_field])

# --- Penalize class logic ---
penalize_class_index = None
if args_cli.penalize_class:
    if args_cli.penalize_class not in label_encoder.classes_:
        raise ValueError(f"❌ Specified --penalize-class '{args_cli.penalize_class}' not in known classes: {list(label_encoder.classes_)}")
    penalize_class_index = list(label_encoder.classes_).index(args_cli.penalize_class)
    logger.info(f"⚖️ Will apply special loss handling to class: {args_cli.penalize_class} (index {penalize_class_index})")

with open(encoder_path, "w", encoding="utf-8") as f:
    json.dump({"classes": list(label_encoder.classes_)}, f)

min_label, max_label = df["tone_label"].min(), df["tone_label"].max()
assert min_label >= 0, f"Tone label below 0: {min_label}"
assert max_label < len(label_encoder.classes_), f"Tone label above expected range: {max_label}"

# Conditional split logic for train/eval
if args_cli.balance_class_ratio:
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle once
    eval_indices = list(range(0, len(df), 10))
    train_indices = [i for i in range(len(df)) if i not in eval_indices]
    train_df = df.iloc[train_indices].reset_index(drop=True)
    eval_df = df.iloc[eval_indices].reset_index(drop=True)
    logger.info(f"📊 Using 1-in-10 slicing for eval split (balanced set).")
else:
    dataset = Dataset.from_pandas(df)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_df = split["train"].to_pandas()
    eval_df = split["test"].to_pandas()
    logger.info(f"📊 Using HuggingFace random split for eval set (unbalanced set).")

# --- Save eval dataset to JSONL ---
eval_jsonl_path = output_dir / "last_eval_set.jsonl"
eval_df.to_json(eval_jsonl_path, orient="records", lines=True, force_ascii=False)
logger.info(f"📝 Saved eval dataset used during training to: {eval_jsonl_path}")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(eval_df)

# Tokenizer
model_checkpoint = args_cli.model_checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_fn(example):
    max_length = 1024 if "deberta" in model_checkpoint.lower() else 512
    return tokenizer(example["text"], truncation=True, max_length=max_length)

# --- Optional holdout evaluation set ---
monitor_eval_loader = None
if args_cli.evaluation_path:
    logger.info(f"🧪 Loading holdout monitoring set: {args_cli.evaluation_path}")
    eval_path = Path(args_cli.evaluation_path)
    if eval_path.suffix == ".jsonl":
        holdout_df = load_and_filter_dataframe(eval_path, args_cli.signal_threshold, tone_field=args_cli.label_field)
    elif eval_path.suffix == ".csv":
        holdout_df = pd.read_csv(eval_path)
    elif eval_path.suffix == ".tsv":
        holdout_df = pd.read_csv(eval_path, sep="\t")
    else:
        raise ValueError(f"Unsupported evaluation file type: {eval_path.suffix}")
    holdout_df = holdout_df[holdout_df[args_cli.label_field].isin(label_list)].reset_index(drop=True)
    holdout_df["tone_label"] = label_encoder.transform(holdout_df[args_cli.label_field])
    holdout_dataset = Dataset.from_pandas(holdout_df)
    tokenized_holdout = holdout_dataset.map(tokenize_fn, batched=True)
    tokenized_holdout = tokenized_holdout.rename_column("tone_label", "labels")
    tokenized_holdout.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    from torch.utils.data import DataLoader
    monitor_eval_loader = DataLoader(tokenized_holdout, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))

# Tokenize both splits
tokenized_train = train_dataset.map(tokenize_fn, batched=True)
tokenized_val = val_dataset.map(tokenize_fn, batched=True)

# Format dataset
def format_labels(example):
    example["labels"] = [example["vividness"], example["emotion"], example["action"], example["tightness"]]
    example["tone_label"] = int(example["tone_label"])
    example["tone_label"] = np.int64(example["tone_label"])
    return example

if args_cli.task == "classification":
    tokenized_train = tokenized_train.rename_column("tone_label", "labels")
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_train = tokenized_train.shuffle(seed=42)

    tokenized_val = tokenized_val.rename_column("tone_label", "labels")
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
else:
    tokenized_train = tokenized_train.map(format_labels)
    tokenized_train = tokenized_train.rename_column("labels", "regression_labels")
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "regression_labels", "tone_label"])
    tokenized_train = tokenized_train.shuffle(seed=42)
    tokenized_val = tokenized_val.map(format_labels)
    tokenized_val = tokenized_val.rename_column("labels", "regression_labels")
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "regression_labels", "tone_label"])

from transformers import AutoConfig, AutoModelForSequenceClassification
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=6)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)


base_lr = 5e-5
# Setup parameter groups
optimizer_grouped_parameters = [{"params": model.parameters(), "lr": base_lr}]

# Classification Trainer
class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Ensure DistilBERT doesn't receive unexpected arguments like 'labels'
        model_input_keys = ["input_ids", "attention_mask", "token_type_ids"]
        inputs_clean = {k: v for k, v in inputs.items() if k in model_input_keys}
        outputs = model(**inputs_clean)
        logits = outputs.logits
        labels = inputs["labels"].long()

        # --- Dynamic class weighting for weak/strong classes
        class_weights = None
        if args_cli.focus_weak_classes and hasattr(self, "trainer") and hasattr(self.trainer, "state") and self.trainer.state.log_history:
            # Pull most recent eval F1 scores per class
            history = self.trainer.state.log_history[::-1]
            for entry in history:
                if all(f"eval_f1_{tone}" in entry for tone in label_encoder.classes_):
                    f1_scores = np.array([entry[f"eval_f1_{tone}"] for tone in label_encoder.classes_])
                    max_f1 = np.max(f1_scores)
                    class_weights = 0.5 + 1.5 * ((max_f1 - f1_scores) / (max_f1 + 1e-6))
                    logger.info(f"🎯 Dynamic class weighting (scaled by F1 gap): {class_weights.round(3).tolist()}")
                    break

        # Compute individual sample loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=logits.device) if class_weights is not None else None,
            reduction='none'
        )

        # --- Focal loss integration ---
        if args_cli.use_focal_loss:
            gamma = 2.0
            logits_softmax = torch.nn.functional.softmax(logits, dim=1)
            true_prob = logits_softmax[torch.arange(logits.size(0)), labels]
            modulating_factor = (1 - true_prob) ** gamma
            ce_loss = torch.nn.functional.cross_entropy(
                logits, labels, reduction='none',
                weight=torch.tensor(class_weights, device=logits.device) if class_weights is not None else None
            )
            loss = modulating_factor * ce_loss
        else:
            loss = loss_fct(logits, labels)

        # Apply penalties for a specific class if configured and no other loss mods active
        if (not args_cli.use_focal_loss and not args_cli.focus_weak_classes
            and 'penalize_class_index' in globals() and penalize_class_index is not None):
            preds = torch.argmax(logits, dim=1)
            penalty_mask = (labels == penalize_class_index) | (preds == penalize_class_index)
            fn_mask = (labels == penalize_class_index) & (preds != penalize_class_index)

            weights = torch.ones_like(loss)
            fn_weights = torch.where(fn_mask, torch.tensor(3.0, device=weights.device), weights)
            penalty_weights = torch.where(penalty_mask, torch.tensor(2.0, device=weights.device), torch.tensor(1.0, device=weights.device))
            weights = torch.max(fn_weights, penalty_weights)
            weighted_loss = (loss * weights).mean()

            # Additional penalties for entropy and softmax margin if class matches
            mask = labels == penalize_class_index
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
    model_name = args_cli.model_checkpoint.split("/")[-1]
    dataset_name = Path(args_cli.data_path).stem
    signal_strength = f"s{int(args_cli.signal_threshold)}" if args_cli.signal_threshold is not None else "sNA"
    ratio_tag = f"r{args_cli.balance_class_ratio}"
    peft_tag = f"full"
    run_name = f"{model_name}-{dataset_name}-{ratio_tag}-{signal_strength}-{peft_tag}-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    #
    # Determine batch size from shared utility
    base_batch_size = determine_batch_size(args_cli.model_checkpoint, args_cli.no_batching)
    logger.info(f"📦 Auto-scaled batch size: using batch size {base_batch_size}")
    dynamic_eval_steps = calculate_dynamic_eval_steps(len(tokenized_train), base_batch_size)
    logger.info(f"📊 Eval every {dynamic_eval_steps} steps based on ~1/3 epoch heuristic.")

    args = TrainingArguments(
        run_name=run_name,
        logging_dir=f"logs/{run_name}",
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=base_batch_size,
        per_device_eval_batch_size=base_batch_size,
        num_train_epochs=args_cli.total_epochs,
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
        fp16=True,
        bf16=False,
        include_for_metrics=["loss"],
        label_names=["labels"],
    )

    trainer_class = ClassificationTrainer if args_cli.task == "classification" else RegressionTrainer

    def compute_metrics(eval_preds):
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
        from sklearn.metrics import classification_report
        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(), output_dict=True, zero_division=0)
        per_class_f1 = {f"f1_{label_encoder.classes_[int(k)]}": v["f1-score"] for k, v in report.items() if k.isdigit()}

        # Log per-class F1 to TensorBoard under class_eval/
        for k, v in per_class_f1.items():
            writer.add_scalar(f"class_eval/{k}", v, trainer.state.global_step)

        result = {"eval_loss": loss.item(), "eval_f1": f1}
        result.update(per_class_f1)
        return result

    global writer  # So it can be used in compute_metrics
    writer = SummaryWriter(log_dir=args.logging_dir)

    # --- ExtraEvalCallback definition ---
    from transformers import TrainerCallback
    from sklearn.metrics import accuracy_score, f1_score
    import torch
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
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            self.writer.add_scalar(f"{self.name}/accuracy", acc, state.global_step)
            self.writer.add_scalar(f"{self.name}/f1_macro", f1, state.global_step)

    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(torch.optim.AdamW(optimizer_grouped_parameters), None),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=7),
            EpochNormalizedLogger(writer),
            MemoryUsageLogger(model, args_cli.model_checkpoint, base_batch_size, input_size=(1024 if "deberta" in model_checkpoint.lower() else 512)),
            *([ExtraEvalCallback(model, monitor_eval_loader, writer)] if monitor_eval_loader else []),
            ManualEarlyStopCallback(stop_file="ranker_model/stop_training.txt"),
        ],
        compute_metrics=compute_metrics,
    )

    logger.info(f"✅ Label class count (from encoder): {len(label_encoder.classes_)}")
    if hasattr(model, "head") and hasattr(model.head, "classification"):
        logger.info(f"✅ Model output classes: {model.head.classification[-1].out_features}")
    elif hasattr(model, "classifier"):
        logger.info(f"✅ Model output classes: {model.classifier.out_features}")
    else:
        model_name = getattr(model.config, "architectures", ["Unknown"])[0]
        logger.warning(f"⚠️ Could not determine model output classes for model type: {model_name}")
    logger.info(f"✅ Max label index in dataset: {df['tone_label'].max()}")

    trainer.train()

    # Reload best checkpoint to ensure we save the best model's head
    best_path = trainer.state.best_model_checkpoint
    logger.info(f"🌟 Reloading best model weights from: {best_path}")
    adapter_path = os.path.join(best_path, "adapter_model.safetensors")
    head_path = os.path.join(best_path, "ranker_head.pt")

    if os.path.exists(adapter_path):
        logger.info("🪛 Detected PEFT adapter. Skipping full model weight load.")
        head_path = os.path.join(best_path, "ranker_head.pt")
        assert os.path.exists(head_path), f"❌ Missing expected classifier head weights: {head_path}"
        model.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        logger.info(f"🧠 Reloaded best classifier head weights from {head_path}")
        # --- Checksum PEFT adapter after reload if debug enabled ---
        if args_cli.debug:
            adapter_checksum = checksum_peft(model.model_backbone)
            logger.debug(f"🧾 Loaded PEFT adapter checksum during best model reload: {adapter_checksum}")
    elif os.path.exists(os.path.join(best_path, "model.safetensors")):
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
    root = Path("ranker_model")
    i = 1
    while Path(f"{root}-v{i}").exists():
        i += 1
    final_path = Path(f"{root}-v{i}")
    final_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Saving full model to {final_path}")
    model.save_pretrained(final_path)

    tokenizer.save_pretrained(final_path)
    with open(final_path / "tone_label_encoder.json", "w", encoding="utf-8") as f:
        json.dump({"classes": list(label_encoder.classes_)}, f)

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
