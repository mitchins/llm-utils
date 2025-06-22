import argparse
import json
import os
import sys
import torch
import numpy as np
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer  # Not BertTokenizer
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, Dataset
from typing import cast, Any, Tuple
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn.functional as F

# Add imports for dynamic model loading
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

def load_model(path: str) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    if "flair/" in path:
        from flair.data import Sentence
        from flair.models import SequenceTagger
        model = SequenceTagger.load(path)
        return model, None, "flair"
    arch = None
    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        arch = config.get("architectures", [""])[0]
    else:
        downloaded_config = hf_hub_download(repo_id=path, filename="config.json")
        with open(downloaded_config, "r") as f:
            config = json.load(f)
        arch = config.get("architectures", [""])[0]

    if "TokenClassification" in arch:
        model: torch.nn.Module = AutoModelForTokenClassification.from_pretrained(path)
        model_type = "token"
    elif "SequenceClassification" in arch:
        model: torch.nn.Module = AutoModelForSequenceClassification.from_pretrained(path)
        model_type = "sequence"
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, model_type

# Suppress non-critical logging from Hugging Face Transformers and Torch
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

def load_label_encoder(path):
    with open(path, "r") as f:
        data = json.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(data["classes"])
    return encoder

def predict_single(text, model, tokenizer, label_encoder, device, min_confidence=None, model_type="sequence"):
    if model_type == "flair":
        sentence = Sentence(text)
        model.predict(sentence)
        entities = []
        for entity in sentence.get_spans('ner'):
            entities.append({
                "label": entity.get_label("ner").value,
                "span": entity.text,
                "score": entity.score
            })
        return entities, None

    if isinstance(text, tuple):
        inputs = tokenizer(text[0], text[1], return_tensors="pt", truncation=True, padding=True).to(device)
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    top_prob = probs.max()
    prediction = torch.argmax(logits, dim=1).item()

    if min_confidence and top_prob < min_confidence:
        return "uncertain", probs

    label = label_encoder.inverse_transform([prediction])[0]
    return label, probs

def handle_prediction_output(text, pred, probs, label, model_type, detail=False):
    if model_type == "flair":
        if not pred:
            print("⚠️ No entities found.")
        else:
            for entity in pred:
                entity_label = entity["label"]
                span = entity["span"]
                score = round(entity["score"], 3)
                print(f"🔎 [{entity_label}] \"{span}\" (score={score})")
    else:
        if pred == "uncertain":
            print(f"🟡 Prediction uncertain — Softmax: {np.round(probs, 3)}")
        else:
            print(f"🔎 Predicted Label: {pred} — Softmax: {np.round(probs, 3)}")
        if detail and label is not None:
            display_text = text if len(text) < 100 else text[:97] + "..."
            print(f"[{label}] — Text: {display_text}")

def run_repl(model, tokenizer, label_encoder, device, min_confidence=None, model_type="sequence", nli_mode=False):
    print("📥 Interactive REPL mode. Type input to classify. Ctrl+C to exit.")
    while True:
        try:
            if nli_mode:
                premise = input("Premise: ")
                hypothesis = input("Hypothesis: ")
                text = (premise, hypothesis)
            else:
                text = input(">>> ")
            pred, probs = predict_single(text, model, tokenizer, label_encoder, device, min_confidence, model_type)
            handle_prediction_output(text, pred, probs, None, model_type)
        except KeyboardInterrupt:
            print("\n👋 Exiting REPL.")
            break

def evaluate_bulk(data_path, model, tokenizer, label_encoder, device, text_field="text", label_field="label", min_confidence=None, detail=False, model_type="sequence"):
    dataset_obj = load_dataset("json", data_files=data_path)
    dataset = dataset_obj["train"] if isinstance(dataset_obj, dict) and "train" in dataset_obj else dataset_obj
    if not isinstance(dataset, Dataset):
        raise TypeError("Loaded dataset is not a regular Dataset; check if streaming mode or a different format was used.")
    y_true = []
    y_pred = []
    uncertain_count = 0
    ambiguous_total = 0
    ambiguous_correct = 0
    false_abstain = 0
    false_confident = 0

    show_progress = not detail
    for item in tqdm(dataset, disable=not show_progress, leave=False, ncols=100, dynamic_ncols=False):
        item = cast(dict[str, Any], item)
        text = item[text_field]
        label = cast(dict, Any).get(label_field)
        pred, probs = predict_single(text, model, tokenizer, label_encoder, device, min_confidence, model_type)

        handle_prediction_output(text, pred, probs, label, model_type=model_type, detail=detail)

        if label is None:
            ambiguous_total += 1
            if pred == "uncertain":
                ambiguous_correct += 1
            else:
                false_confident += 1
            continue

        if pred == "uncertain":
            uncertain_count += 1
            false_abstain += 1
            continue

        y_true.append(label)
        y_pred.append(pred)
    print(classification_report(y_true, y_pred, labels=label_encoder.classes_))
    print(f"⚠️ Uncertain predictions skipped: {uncertain_count}")
    print(f"🟡 Ambiguous examples evaluated: {ambiguous_total}")
    print(f"✅ Correctly abstained (uncertain): {ambiguous_correct}")
    if ambiguous_total:
        rate = ambiguous_correct / ambiguous_total
        print(f"📊 Abstention Accuracy on Ambiguous: {ambiguous_correct}/{ambiguous_total} ({rate:.1%})")
    print(f"❌ False Abstentions: {false_abstain}")
    print(f"❌ False Confidences: {false_confident}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the trained model directory")
    parser.add_argument("--data-path", help="Path to a JSONL file for bulk evaluation")
    parser.add_argument("--text-field", default="text", help="Text field name in JSONL")
    parser.add_argument("--label-field", default="label", help="Label field name in JSONL")
    parser.add_argument("--min-confidence", type=float, help="Minimum softmax confidence to accept a prediction")
    parser.add_argument("--detail", action="store_true", help="Print detailed predictions in bulk mode")
    parser.add_argument("--labels", help="Comma-separated class labels (e.g., 'negative,positive')")
    parser.add_argument("--nli", action="store_true", help="Enable Natural Language Inference mode (premise + hypothesis input)")
    args = parser.parse_args()

    model, tokenizer, model_type = load_model(args.model_path)
    print(f"📦 Loaded model type: {model_type}")
    if model_type != "flair":
        if args.labels:
            labels = [l.strip() for l in args.labels.split(",")]
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(labels)
        else:
            label_encoder_path = os.path.join(args.model_path, "label_encoder.json")
            if os.path.exists(label_encoder_path):
                label_encoder = load_label_encoder(label_encoder_path)
            else:
                raise FileNotFoundError(
                    f"Missing label_encoder.json and no --labels provided. "
                    f"Please supply --labels 'label1,label2' or include label_encoder.json."
                )
    else:
        label_encoder = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type != "flair":
        model.to(device)
        model.eval()

    if args.data_path:
        evaluate_bulk(args.data_path, model, tokenizer, label_encoder, device, args.text_field, args.label_field, args.min_confidence, args.detail, model_type)
    else:
        run_repl(model, tokenizer, label_encoder, device, args.min_confidence, model_type, args.nli)

if __name__ == "__main__":
    main()