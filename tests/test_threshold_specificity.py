"""CPU-only tests for Experiment 009 utilities."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/009-qwen-threshold-specificity/run.py"
spec = importlib.util.spec_from_file_location("threshold_specificity", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_default_doses_and_controls():
    doses = mod.parse_doses(mod.DEFAULT_DOSES)
    assert doses[0] == 0 and doses[-1] == -150
    assert -62 in doses and -70 in doses
    assert mod.parse_neurons(mod.DEFAULT_NEURONS) == [385, 386, 387]


@pytest.mark.parametrize("raw", ["-10,-10,0", "-10", "0,1", "0,nan", "0,inf", ""])
def test_bad_doses(raw):
    with pytest.raises(ValueError):
        mod.parse_doses(raw)


@pytest.mark.parametrize("raw", ["386,386", "-1,386", "abc", ""])
def test_bad_neurons(raw):
    with pytest.raises(ValueError):
        mod.parse_neurons(raw)


def test_thresholds_are_separate_from_margin():
    rows = [
        {"prefix": "first6", "neuron": 386, "dose": -60, "greedy_token_id": 1,
         "open_minus_alibaba_logit": -1},
        {"prefix": "first6", "neuron": 386, "dose": -70, "greedy_token_id": 2,
         "open_minus_alibaba_logit": 0.5},
        {"prefix": "first6", "neuron": 385, "dose": -70, "greedy_token_id": 1,
         "open_minus_alibaba_logit": 1.0},
    ]
    assert mod.first_greedy(rows, "first6", 386, 2) == 70
    assert mod.first_greedy(rows, "first6", 385, 2) is None
    assert mod.first_margin_crossing(rows, "first6", 385) == 70
    assert mod.first_margin_crossing(rows, "baseline", 386) is None


def test_save_json_and_csv(tmp_path):
    targets = {name: {"probability": 0.1, "logit": 0, "token_id": i}
               for i, name in enumerate(("alibaba", "open", "microsoft", "meta", "google", "azure"))}
    row = {"prefix": "first6", "neuron": 386, "dose": -70,
           "greedy_token_text": " Open", "greedy_token_id": 2,
           "open_minus_alibaba_logit": 0.5, "targets": targets, "edited_steps": [7]}
    path = tmp_path / "nested" / "out.json"
    mod.save(path, {"results": [row]})
    assert json.loads(path.read_text())["results"][0]["neuron"] == 386
    csv_data = path.with_suffix(".csv").read_text()
    assert "microsoft_probability" in csv_data and "first6,386,-70" in csv_data
