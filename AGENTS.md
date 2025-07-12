# Agent Guidelines

## Testing
To run the full test suite you **must** install the optional training and test
dependencies (this pulls in heavy packages such as `datasets`):

```bash
pip install -e .[training,test]
pip install evaluate
```

Once dependencies are installed, execute the tests with `pytest`.

## Code Overview
Agents should begin by familiarizing themselves with the core training pipeline implementation:
- **Location**: `llm_utils/training/train_t5.py`
- **Key phases**:
  1. **Tokenizer Load**: Loading and configuring `AutoTokenizer`.
  2. **Tokenization Phase**: `log_length_histogram` and `tokenize_dataset` pipeline.
  3. **Filtering**: Removing examples exceeding `--max-input-length`.
  4. **Model Load**: Instantiating `AutoModelForSeq2SeqLM`.
  5. **Collator Setup**: Configuring `DataCollatorForSeq2Seq` for padding inputs and labels.
  6. **Trainer Construction**: Building `HFSeq2SeqTrainer` subclass and setting callbacks.
  7. **Training Loop**: `trainer.train()` execution.

## Reviewing Tokenization
- Inspect the `preprocess_t5` function for input/target tokenization logic.
- Confirm `tokenize_dataset` drops original text columns and retains `"input_ids"`, `"attention_mask"`, and `"labels"`.
- Verify the `log_length_histogram` output matches expected distributions when running on sample data.

## Testing Specific Components
Agents should run and expand existing tests under `tests/training/`, including:
- **`test_collator_padding` / `test_collator_pad_behavior`**: Ensure the seq2seq collator pads both inputs and labels uniformly.
- **`test_log_length_histogram`**: Confirm that histogram bins scale to the data range and respect the `max_bins` parameter.
- **`test_tokenizer_override`**: Validate that overriding the tokenizer (e.g., with a fake `model_max_length`) correctly adjusts `--max-input-length`.

## Expanding and Debugging
- To add new functionality or edge-case handling, create corresponding unit tests in `tests/training/`.
- Use `pytest -q --disable-warnings --maxfail=1` to quickly iterate on failures.
- For debugging, leverage `rank_logger` and set `--debug` for verbose logs.
- When modifying padding or truncation behavior, check both real tokenizer outputs and dummy stubs to ensure consistency.


## Environment and Dependencies
- Python 3.10+, `pip install -e .[training,test] evaluate numpy pytest`
- Ensure `transformers`, `datasets`, and `evaluate` packages are installed at compatible versions.

Agents should reference this file when onboarding new reviewers or automating test runs.

## Versioning
- After making any significant changes to `train_t5.py`, update the project version to reflect the changes.
- Locate the version string in `setup.py` or `pyproject.toml` (e.g., `version="X.Y.Z"`).
- Increment the patch number (e.g., from `1.2.3` to `1.2.4`) for bug fixes or minor improvements.
- Commit the version bump as part of the same pull request.
