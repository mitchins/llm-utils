import numpy as np
from datasets import load_from_disk
import argparse
from transformers import BertTokenizerFast

def analyze_token_lengths(dataset_path):
    # Load the dataset from the specified path using load_from_disk
    dataset = load_from_disk(dataset_path)

    # Initialize the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Analyze the token lengths of each sample
    token_lengths = []
    for example in dataset:  # Iterate over all examples in the dataset
        # The dataset might have columns like 'input_ids' and 'labels'
        token_lengths.append(len(example['input_ids']))  # Get token length from 'input_ids'

    # Calculate statistics
    max_tokens = np.max(token_lengths)
    min_tokens = np.min(token_lengths)
    mean_tokens = np.mean(token_lengths)
    median_tokens = np.median(token_lengths)

    print(f"Max tokens: {max_tokens}")
    print(f"Min tokens: {min_tokens}")
    print(f"Mean tokens: {mean_tokens}")
    print(f"Median tokens: {median_tokens}")

    # Optionally, print out the length distribution (e.g., for more insights)
    token_histogram, bin_edges = np.histogram(token_lengths, bins=50)
    print(f"Token Length Histogram: {token_histogram}")

    return token_lengths

def main():
    parser = argparse.ArgumentParser(description="Analyze Token Lengths of Dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset (e.g., 'processed_data_train.arrow')")
    
    args = parser.parse_args()

    # Analyze the token lengths of the dataset provided
    token_lengths = analyze_token_lengths(args.dataset_path)

    # Optionally, you can return or process the token lengths further if needed
    return token_lengths

if __name__ == "__main__":
    main()
