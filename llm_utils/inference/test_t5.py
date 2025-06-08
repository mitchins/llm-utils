

import argparse
import json
from pathlib import Path
import readline
import atexit

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
from sklearn.metrics import precision_recall_fscore_support

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def generate_single(text, model, tokenizer, max_new_tokens=64, num_beams=1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def evaluate_bulk(dataset_path, model, tokenizer, text_field="input", target_field="output", max_new_tokens=64, num_beams=1, show_detail=False):
    ds = load_dataset("json", data_files=dataset_path)["train"]

    predictions = []
    references = []
    skipped = 0

    for item in ds:
        input_text = item[text_field]
        ref = item.get(target_field, "")
        pred = generate_single(input_text, model, tokenizer, max_new_tokens, num_beams).strip()

        if show_detail:
            print(f"📝 Input: {input_text}")
            print(f"✅ Target: {ref}")
            print(f"🤖 Predicted: {pred}")
            print("")

        predictions.append(pred)
        references.append(ref)

    # Simple exact match accuracy
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    total = len(references)
    empty = sum(1 for p in predictions if not p.strip())

    print(f"\n🎯 Accuracy: {correct}/{total} ({100*correct/total:.2f}%)")
    print(f"⚠️ Empty predictions: {empty}/{total} ({100*empty/total:.2f}%)")

    p, r, f1, _ = precision_recall_fscore_support(
        references, predictions, average="micro", zero_division=0
    )
    print(f"📊 Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

def main():
    HISTORY_PATH = ".t5_repl_history"

    try:
        readline.read_history_file(HISTORY_PATH)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, HISTORY_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--text-field", type=str, default="input")
    parser.add_argument("--target-field", type=str, default="output")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--detail", action="store_true", help="Show detailed output per row")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.data_path:
        evaluate_bulk(
            args.data_path,
            model,
            tokenizer,
            args.text_field,
            args.target_field,
            args.max_new_tokens,
            args.num_beams,
            args.detail
        )
    else:
        # Interactive REPL
        print("🧠 T5 REPL mode. Type input text to generate, Ctrl+C to exit.")
        while True:
            try:
                text = input(">>> ").strip()
                if not text:
                    continue
                output = generate_single(text, model, tokenizer, args.max_new_tokens, args.num_beams)
                print(f"🤖 {output}")
            except KeyboardInterrupt:
                print("\n👋 Exiting.")
                break

if __name__ == "__main__":
    main()