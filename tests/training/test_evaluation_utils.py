import importlib

def test_dynamic_steps_basic():
    utils = importlib.import_module('llm_utils.training.utils')
    assert utils.calculate_dynamic_eval_steps(1000, 32) == 10
