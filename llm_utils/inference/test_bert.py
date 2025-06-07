import argparse
import json
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn.functional as F

def load_label_encoder(path):
    with open(path, "r") as f:
        data = json.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(data["classes"])
    return encoder

def predict_single(text, model, tokenizer, label_encoder, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    prediction = torch.argmax(logits, dim=1).item()
    label = label_encoder.inverse_transform([prediction])[0]
    return label, probs

def run_repl(model, tokenizer, label_encoder, device):
    print("📥 Interactive REPL mode. Type input to classify. Ctrl+C to exit.")
    while True:
        try:
            text = input(">>> ")
            pred, probs = predict_single(text, model, tokenizer, label_encoder, device)
            print(f"🔎 Predicted Label: {pred} — Softmax: {np.round(probs, 3)}")
        except KeyboardInterrupt:
            print("\n👋 Exiting REPL.")
            break

def evaluate_bulk(data_path, model, tokenizer, label_encoder, device, text_field="text", label_field="label"):
    dataset = load_dataset("json", data_files=data_path)["train"]
    y_true = []
    y_pred = []
    for item in tqdm(dataset):
        text = item[text_field]
        label = item[label_field]
        pred = predict_single(text, model, tokenizer, label_encoder, device)
        y_true.append(label)
        y_pred.append(pred)
    print(classification_report(y_true, y_pred, labels=label_encoder.classes_))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the trained model directory")
    parser.add_argument("--data-path", help="Path to a JSONL file for bulk evaluation")
    parser.add_argument("--text-field", default="text", help="Text field name in JSONL")
    parser.add_argument("--label-field", default="label", help="Label field name in JSONL")
    args = parser.parse_args()

    model = BertForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    label_encoder = load_label_encoder(os.path.join(args.model_path, "label_encoder.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.data_path:
        evaluate_bulk(args.data_path, model, tokenizer, label_encoder, device, args.text_field, args.label_field)
    else:
        run_repl(model, tokenizer, label_encoder, device)

if __name__ == "__main__":
    main()