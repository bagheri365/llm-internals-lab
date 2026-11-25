"""CPU-only checks for Experiment 004."""
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/004-qwen-token-window/run.py"
spec = importlib.util.spec_from_file_location("token_window", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_conditions():
    assert len(module.conditions()) == 24
    assert module.conditions()[0] == ("Who are you?", "none")


@pytest.mark.parametrize("mode,expected", [
    ("none", []), ("first1", [1]), ("first5", list(range(1, 6))),
    ("first10", list(range(1, 11))), ("after10", list(range(11, 16))),
    ("all", list(range(1, 16))),
])
def test_step_windows(mode, expected):
    assert [i for i in range(1, 16) if module.should_edit(mode, i)] == expected


def test_invalid_window():
    with pytest.raises(ValueError):
        module.should_edit("unknown", 1)
    with pytest.raises(ValueError):
        module.should_edit("all", 0)


def test_activation_copy():
    x = torch.zeros(1, 1, 5)
    y = module.edit_activation(x, 2, -98)
    assert x[0, 0, 2] == 0
    assert y[0, 0, 2] == -98


def test_save_results(tmp_path):
    path = tmp_path / "out" / "result.json"
    row = {"prompt": "Who are you?", "mode": "first1", "dose": -98,
           "prefill_calls": 1, "decode_calls": 4, "edited_steps": [1], "response": "example"}
    csv_path = module.save_results(path, {"results": [row]})
    assert path.exists() and csv_path.exists()
    assert '"edited_steps": [\n        1' in path.read_text()
    assert "first1,-98,1,4,,1,example" in csv_path.read_text()


@pytest.mark.parametrize("generated,decode,extra", [
    (53, 52, False), (53, 53, True), (1, 0, False), (1, 1, True),
])
def test_decode_call_validation(generated, decode, extra):
    assert module.validate_decode_calls({"prefill": 1, "decode": decode}, generated) is extra


@pytest.mark.parametrize("generated,decode,prefill", [
    (53, 51, 1), (53, 54, 1), (53, 53, 2), (0, 0, 1),
])
def test_decode_call_validation_rejects_unexpected(generated, decode, prefill):
    with pytest.raises(RuntimeError):
        module.validate_decode_calls({"prefill": prefill, "decode": decode}, generated)
