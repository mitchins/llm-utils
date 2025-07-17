import pytest
from llm_utils.training.evaluation_utils import calculate_dynamic_eval_steps


def test_zero_batch_size_raises():
    with pytest.raises(ValueError):
        calculate_dynamic_eval_steps(100, 0)


def test_dataset_smaller_than_batch():
    assert calculate_dynamic_eval_steps(5, 10) == 1


def test_preferred_fraction_zero_returns_one():
    assert calculate_dynamic_eval_steps(30, 10, preferred_fraction=0) == 1


def test_preferred_fraction_gt_one_caps_to_epoch():
    assert calculate_dynamic_eval_steps(30, 10, preferred_fraction=2) == 3
