import argparse
import json
from pathlib import Path
import readline
import atexit
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llm_utils.data.dataset_loading import load_dataset_auto
from sklearn.metrics import precision_recall_fscore_support

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Debug: concise special token info
    try:
        core_tokens = {k: tokenizer.special_tokens_map[k] for k in ["pad_token", "eos_token", "unk_token"] if k in tokenizer.special_tokens_map}
        print("Core special tokens:", core_tokens)
        user_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])
        # Separate mask and extra_id tokens
        extra_ids = [tok for tok in user_tokens if tok.startswith("<extra_id_")]
        other_tokens = [tok for tok in user_tokens if not tok.startswith("<extra_id_")]
        # Print non-extra_id tokens individually
        if other_tokens:
            print("User-added tokens:")
            for tok in other_tokens:
                print(f"  {tok}: {tokenizer.convert_tokens_to_ids(tok)}")
    except Exception as e:
        print(f"⚠️ Could not print special tokens info: {e}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

def generate_single(text, model, tokenizer, max_new_tokens=64, num_beams=1, max_input_length=4096):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print(f"🧾 Token count: {len(inputs['input_ids'][0])}")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def evaluate_bulk(dataset_path, model, tokenizer, text_field="input", target_field="output", max_new_tokens=64, num_beams=1, show_detail=False, max_input_length=4096):
    ds = load_dataset_auto(dataset_path)

    predictions = []
    references = []
    skipped = 0

    for item in ds:
        input_text = item[text_field]
        ref = item.get(target_field, "")
        pred = generate_single(input_text, model, tokenizer, max_new_tokens, num_beams, max_input_length).strip()

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

def multiline_input(prompt=">>> "):
    print(prompt + " (End with empty line)")
    lines = []
    while True:
        try:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

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
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--detail", action="store_true", help="Show detailed output per row")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print(f"📏 Model supports up to {getattr(model.config, 'max_position_embeddings', '???')} tokens")

    if args.data_path:
        evaluate_bulk(
            args.data_path,
            model,
            tokenizer,
            args.text_field,
            args.target_field,
            args.max_new_tokens,
            args.num_beams,
            args.detail,
            args.max_input_length
        )
    else:
        # Interactive REPL
        print("🧠 T5 REPL mode. Type input text to generate, Ctrl+C to exit.")
        while True:
            try:
                text = multiline_input("📜 Enter or paste sample")
                if not text:
                    continue
                output = generate_single(text, model, tokenizer, args.max_new_tokens, args.num_beams, args.max_input_length)
                print(f"🤖 {output}")
            except KeyboardInterrupt:
                print("\n👋 Exiting.")
                break

if __name__ == "__main__":
    main()