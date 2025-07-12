# tests/training/test_trainer_integration.py
import os
import csv
import argparse
import pytest
from llm_utils.training.train_t5 import parse_args, build_pipeline, state

class CaptureTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, processing_class, data_collator, callbacks, compute_metrics):
        # Keep for assertions
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = processing_class
    def train(self):
        pass  # never actually train