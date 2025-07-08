

import argparse
import json
from pathlib import Path

def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(line)["text"].strip() for line in f if line.strip()}

def main():
    parser = argparse.ArgumentParser(description="Find overlapping text entries between two JSONL datasets.")
    parser.add_argument("file1", type=Path, help="Path to first JSONL file")
    parser.add_argument("file2", type=Path, help="Path to second JSONL file")
    parser.add_argument("--output", type=Path, help="Optional output JSONL file to save intersected texts")
    args = parser.parse_args()

    texts1 = load_texts(args.file1)
    texts2 = load_texts(args.file2)
    intersection = texts1 & texts2

    print(f"🔍 Found {len(intersection)} overlapping text entries.")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out:
            for text in sorted(intersection):
                out.write(json.dumps({"text": text}) + "\n")
        print(f"✅ Saved intersected texts to {args.output}")

if __name__ == "__main__":
    main()