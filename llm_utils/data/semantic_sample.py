import csv
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import json

# ---- Configuration ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # BGE-mini not on HF, small is equivalent
PCA_DIM = 32
HASH_BITS = 64
RANDOM_SEED = 42

torch.set_default_device("mps" if torch.backends.mps.is_available() else "cpu")

def load_data(input_file, label_field):
    rows = []
    skipped = 0
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '').strip()
                    classification = obj.get(label_field, '').strip()
                    if text and classification:
                        rows.append((text, '', classification))
                except Exception:
                    skipped += 1
                    continue
    else:  # default to csv
        with open(input_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader((line for line in f if '\x00' not in line))
            for row in reader:
                try:
                    text = row.get('text', '').strip()
                    classification = row.get(label_field, '').strip()
                    if text and classification:
                        rows.append((text, '', classification))
                except Exception:
                    skipped += 1
                    continue
    if skipped > 0:
        print(f"Skipped {skipped} malformed or NUL-containing rows.")
    return rows

def embed_entries(rows, model):
    texts = [f"{title} by {author}" for title, author, _ in rows]
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    embeddings = model.encode(
        texts,
        batch_size=128,  # Double previous now we have FP16
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        convert_to_tensor=False,
        device=device,
        dtype=torch.float16
    )
    return np.array(embeddings)

def reduce_dimensions(embeddings, dim):
    pca = PCA(n_components=dim, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    return reduced

def fingerprint(vec):
    """Convert to 64-bit hash using sign binarization"""
    bits = (vec > 0).astype(int)
    as_int = 0
    for b in bits[:HASH_BITS]:
        as_int = (as_int << 1) | b
    return as_int

def bucket_items(vectors, rows):
    buckets = defaultdict(list)
    for idx, vec in enumerate(vectors):
        h = fingerprint(vec)
        buckets[h].append(rows[idx])
    return buckets

def sample_buckets(buckets, target_count):
    sampled = []
    bucket_ids = list(buckets.keys())
    random.shuffle(bucket_ids)

    for bucket_id in bucket_ids:
        if len(sampled) >= target_count:
            break
        items = buckets[bucket_id]
        sampled.append(random.choice(items))
    return sampled

def write_output(output_file, rows):
    if output_file.endswith('.jsonl'):
        with open(output_file, 'w', encoding='utf-8') as f:
            for title, author, classification in rows:
                json.dump({"text": f"{title} by {author}", "label": classification}, f)
                f.write("\n")
    else:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            for title, author, classification in rows:
                writer.writerow([f"{title} by {author}", classification])

def holdout_split(rows, holdout_fraction):
    # Split rows by class
    by_class = defaultdict(list)
    for row in rows:
        by_class[row[2]].append(row)
    main_rows = []
    holdout_rows = []
    for cls, items in by_class.items():
        random.shuffle(items)
        holdout_size = int(len(items) * holdout_fraction)
        holdout_rows.extend(items[:holdout_size])
        main_rows.extend(items[holdout_size:])
    return main_rows, holdout_rows

def main(args):
    print("Loading input...")
    data = load_data(args.input, args.label_field)
    print(f"Loaded {len(data)} rows.")

    from collections import defaultdict
    by_class = defaultdict(list)
    for row in data:
        by_class[row[2]].append(row)
    min_class_size = min(len(rows) for rows in by_class.values())
    data = []
    for rows in by_class.values():
        random.shuffle(rows)
        data.extend(rows[:min_class_size])
    print(f"Balanced to {min_class_size} samples per class, total: {len(data)}")

    print("Embedding...")
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_entries(data, model)
    from sklearn.preprocessing import StandardScaler

    # Normalize and clean embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    mask = np.all(np.isfinite(embeddings), axis=1)
    if not np.all(mask):
        print(f"Removed {np.sum(~mask)} rows due to non-finite values.")
        embeddings = embeddings[mask]
        data = [row for i, row in enumerate(data) if mask[i]]

    print(f"Reducing to {PCA_DIM} dimensions...")
    reduced = reduce_dimensions(embeddings, PCA_DIM)

    print("Bucketing...")
    buckets = bucket_items(reduced, data)

    print(f"Sampling to {len(data)} total...")
    selected = sample_buckets(buckets, len(data))

    if args.holdout and args.holdout > 0.0:
        print(f"Performing holdout split with fraction {args.holdout}...")
        main_set, holdout_set = holdout_split(selected, args.holdout)
        print(f"Main set size: {len(main_set)}, Holdout set size: {len(holdout_set)}")
        print(f"Writing main set to {args.output}")
        write_output(args.output, main_set)
        if args.holdout_output:
            print(f"Writing holdout set to {args.holdout_output}")
            write_output(args.holdout_output, holdout_set)
    else:
        print(f"Writing {len(selected)} examples to {args.output}")
        write_output(args.output, selected)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='book_categories_nonfiction_cleaned.csv')
    parser.add_argument('--output', type=str, default='training_nonfiction_sampled.csv')
    parser.add_argument('--count', type=int, default=143084)
    parser.add_argument('--label-field', type=str, default='classification')
    parser.add_argument('--holdout', type=float, default=0.0)
    parser.add_argument('--holdout-output', type=str, default=None)
    args = parser.parse_args()
    main(args)
