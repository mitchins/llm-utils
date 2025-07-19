# Model Training

This directory contains scripts for fine-tuning common Language and Encoder models.

## Core Components

- **`train_t5.py`**: A script for fine-tuning T5-style models for sequence-to-sequence tasks. It includes support for logging, evaluation, and checkpointing.

- **`train_bert.py`**: A script for fine-tuning BERT-style models for classification and regression tasks.

- **`prepare_dataset.py`**: A utility for preparing datasets for training. It can be used to tokenize and format data from a variety of sources.

- **`callbacks.py`**: Contains custom callbacks for the Hugging Face Trainer, such as a callback for logging evaluation results.

- **`evaluation_utils.py`**: Provides utilities for evaluating model performance.

## Usage

For detailed usage examples, please refer to the main `README.md` file in the root of the repository.
