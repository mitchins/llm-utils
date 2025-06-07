import argparse
import json
import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn.functional as F

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

def predict_single(text, model, tokenizer, label_encoder, device, min_confidence=None):
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

def run_repl(model, tokenizer, label_encoder, device, min_confidence=None):
    print("📥 Interactive REPL mode. Type input to classify. Ctrl+C to exit.")
    while True:
        try:
            text = input(">>> ")
            pred, probs = predict_single(text, model, tokenizer, label_encoder, device, min_confidence)
            if pred == "uncertain":
                print(f"🟡 Prediction uncertain — Softmax: {np.round(probs, 3)}")
            else:
                print(f"🔎 Predicted Label: {pred} — Softmax: {np.round(probs, 3)}")
        except KeyboardInterrupt:
            print("\n👋 Exiting REPL.")
            break

def evaluate_bulk(data_path, model, tokenizer, label_encoder, device, text_field="text", label_field="label", min_confidence=None, detail=False):
    dataset = load_dataset("json", data_files=data_path)["train"]
    y_true = []
    y_pred = []
    uncertain_count = 0
    ambiguous_total = 0
    ambiguous_correct = 0
    false_abstain = 0
    false_confident = 0

    show_progress = not detail
    for item in tqdm(dataset, disable=not show_progress, leave=False, ncols=100, dynamic_ncols=False):
        text = item[text_field]
        label = item.get(label_field)
        pred, probs = predict_single(text, model, tokenizer, label_encoder, device, min_confidence)

        if detail:
            display_text = text if len(text) < 100 else text[:97] + "..."
            print(f"[{label}] => {pred} — Softmax: {np.round(probs, 3)} — Text: {display_text}")

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
    args = parser.parse_args()

    model = BertForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    label_encoder = load_label_encoder(os.path.join(args.model_path, "label_encoder.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.data_path:
        evaluate_bulk(args.data_path, model, tokenizer, label_encoder, device, args.text_field, args.label_field, args.min_confidence, args.detail)
    else:
        run_repl(model, tokenizer, label_encoder, device, args.min_confidence)

if __name__ == "__main__":
    main()