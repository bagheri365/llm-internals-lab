"""CPU-only tests for Experiment 008 helpers."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/008-qwen-prefix-dose-threshold/run.py"
spec = importlib.util.spec_from_file_location("prefix_dose_threshold", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_parse_doses_sorted_and_zero():
    assert m.parse_doses("-98,0,-60") == [0.0, -60.0, -98.0]


@pytest.mark.parametrize("raw", ["", "-10,-10,0", "0,1", "0,nan", "0,inf", "-20"])
def test_invalid_doses(raw):
    with pytest.raises(ValueError):
        m.parse_doses(raw)


def test_first_flip_is_grid_only():
    rows = [{"prefix": "first6", "dose": dose, "greedy_token_id": greedy,
             "target_token_ids": {"open": 2}} for dose, greedy in
            [(0, 1), (-80, 1), (-90, 2), (-98, 1), (-110, 2)]]
    assert m.first_flip(rows, "first6") == 90
    assert m.first_flip(rows, "baseline") is None


def test_save(tmp_path):
    path = tmp_path / "results" / "output.json"
    row = {"prefix": "baseline", "dose": 0.0, "greedy_token_text": " Alibaba",
           "targets": {"open": {"probability": 0.01},
                       "alibaba": {"probability": 0.99}},
           "open_minus_alibaba_logit": -4.5, "edited_steps": []}
    m.save(path, {"results": [row]})
    assert path.exists()
    assert path.with_suffix(".csv").exists()
    assert 'baseline' in path.with_suffix(".csv").read_text()
