import argparse
import logging
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 20000
DEFAULT_WINDOW_SIZE = 1000
DEFAULT_SUB_BATCH_SIZE = 10
EOS_TOKEN_LENGTH = 1  # Used to adjust max_length for T5 mode

def process_and_filter_batch(batch, tokenizer, min_length=None, max_length=None, sub_batch_size=DEFAULT_SUB_BATCH_SIZE, mode="T5"):
    inputs = batch["input"]
    outputs = batch["output"]
    filtered = []

    for i in range(0, len(inputs), sub_batch_size):
        input_slice = inputs[i:i+sub_batch_size]
        output_slice = outputs[i:i+sub_batch_size]

        with ThreadPoolExecutor() as executor:
            tokenized_inputs = list(executor.map(lambda s: tokenizer(s, truncation=False, add_special_tokens=False), input_slice))
            tokenized_outputs = list(executor.map(lambda s: tokenizer(s, truncation=False, add_special_tokens=False), output_slice))

        for j, toks in enumerate(tokenized_inputs):
            length = len(toks["input_ids"])
            if (min_length and length < min_length) or (max_length and length > max_length):
                continue
            out_len = len(tokenized_outputs[j]["input_ids"])
            if (min_length and out_len < min_length) or (max_length and out_len > max_length):
                continue
            labels = tokenized_outputs[j]["input_ids"]
            if mode == "T5" and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                labels = labels + [tokenizer.eos_token_id]
            filtered.append({
                "input_ids": toks["input_ids"],
                "attention_mask": toks["attention_mask"],
                "labels": labels,
            })

    return filtered

def tokenize_and_shuffle(data_path, tokenizer_name, output_prefix, data_format="csv", batch_size=20000, min_length=None, max_length=None, mode="T5"):
    logger.info(f"🔍 Loading dataset from {data_path} ({data_format})")
    ds_iter = load_dataset(data_format, data_files=str(data_path), split="train", streaming=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    chunk_index = 1

    window_size = DEFAULT_WINDOW_SIZE
    batch = []
    for i, sample in enumerate(tqdm(ds_iter, desc="🧠 Tokenizing and filtering")):
        batch.append(sample)
        if len(batch) >= window_size:
            tqdm.write(f"⚙️  Processing greedy window of {len(batch)} samples (chunk {chunk_index:03d})...")
            dataset = Dataset.from_list(batch)
            filtered = process_and_filter_batch(dataset, tokenizer, min_length, max_length, mode=mode)
            random.shuffle(filtered)
            chunk = Dataset.from_list(filtered)
            output_path = f"{output_prefix}-{chunk_index:03d}"
            tqdm.write(f"💾 Writing chunk {chunk_index:03d} with {len(chunk)} examples to {output_path}")
            chunk.save_to_disk(output_path)
            logger.info(f"💾 Saved {len(chunk)} examples to {output_path}")
            batch = []
            chunk_index += 1

    if batch:
        tqdm.write(f"⚙️  Processing final window of {len(batch)} samples (chunk {chunk_index:03d})...")
        dataset = Dataset.from_list(batch)
        filtered = process_and_filter_batch(dataset, tokenizer, min_length, max_length, mode=mode)
        random.shuffle(filtered)
        chunk = Dataset.from_list(filtered)
        output_path = f"{output_prefix}-{chunk_index:03d}"
        tqdm.write(f"💾 Writing chunk {chunk_index:03d} with {len(chunk)} examples to {output_path}")
        chunk.save_to_disk(output_path)
        logger.info(f"💾 Saved {len(chunk)} examples to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--data-format", choices=["csv", "json"], default="csv")
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--merge", action="store_true", help="Merge all output chunks into one final dataset")
    parser.add_argument("--train-split", type=float, default=None, help="Proportion of the final dataset to use for training (e.g., 0.9 means 90% train, 10% eval)")
    parser.add_argument("--mode", choices=["T5", "GPT"], default="T5", help="Model type to adapt data for.")
    args = parser.parse_args()

    if args.mode == "T5" and args.max_length:
        logger.info(f"✂️  Adjusting max_length from {args.max_length} to {args.max_length - EOS_TOKEN_LENGTH} to account for EOS token in T5 mode.")
        args.max_length -= EOS_TOKEN_LENGTH

    tokenize_and_shuffle(
        data_path=args.data_path,
        tokenizer_name=args.tokenizer,
        output_prefix=args.output_prefix,
        data_format=args.data_format,
        batch_size=args.batch_size,
        min_length=args.min_length,
        max_length=args.max_length,
        mode=args.mode,
    )

    if args.merge:
        logger.info("🧬 Merging all chunks into one final dataset (manual round-robin)...")
        chunk_paths = sorted(Path(args.output_prefix).parent.glob(f"{Path(args.output_prefix).name}-[0-9][0-9][0-9]"))
        random.shuffle(chunk_paths)  # Optional shuffle for better randomness

        def round_robin_stream(chunks):
            iterators = []
            for chunk in chunks:
                try:
                    ds = Dataset.load_from_disk(str(chunk))
                    iterators.append(iter(ds))
                except Exception as e:
                    logger.warning(f"⚠️ Skipping chunk {chunk} due to error: {e}")

            while iterators:
                for it in iterators[:]:
                    try:
                        yield next(it)
                    except StopIteration:
                        iterators.remove(it)

        merged_examples = list(round_robin_stream(chunk_paths))
        if not merged_examples:
            logger.warning("⚠️ No examples found during merge — check your chunk paths and format.")
        else:
            merged_dataset = Dataset.from_list(merged_examples)
            merged_path = f"{args.output_prefix}-merged"
            merged_dataset.save_to_disk(merged_path)
            logger.info(f"✅ Merged dataset saved to {merged_path}")

            if args.train_split is not None:
                logger.info(f"✂️  Splitting merged dataset with train ratio: {args.train_split}")
                split = merged_dataset.train_test_split(test_size=1 - args.train_split, seed=42)
                split["train"].save_to_disk(f"{merged_path}-train")
                split["test"].save_to_disk(f"{merged_path}-eval")
                logger.info(f"✅ Train/eval datasets saved to {merged_path}-train and {merged_path}-eval")

        # Cleanup intermediate chunks
        for p in chunk_paths:
            import shutil
            shutil.rmtree(p)
            logger.info(f"🧹 Removed chunk {p}")

if __name__ == "__main__":
    main()