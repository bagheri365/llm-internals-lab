"""CPU-only unit tests for Experiment 006 helpers."""
import importlib.util
import math
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "experiments/006-qwen-logit-causality/run.py"
spec = importlib.util.spec_from_file_location("logit_causality", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_baseline_steps():
    assert m.intervention_steps("baseline", False) == set()
    assert m.replay_steps("baseline", False) == set()
    assert m.replay_steps("baseline", True) == {7}


def test_first6_steps():
    assert m.intervention_steps("first6", False) == set(range(1, 7))
    assert m.replay_steps("first6", False) == set(range(1, 7))
    assert m.replay_steps("first6", True) == set(range(1, 8))


def test_bad_prefix():
    import pytest
    with pytest.raises(ValueError):
        m.replay_steps("other", True)


def test_target_metrics():
    metrics = m.target_metrics([0.0, math.log(2), math.log(3)], {"alibaba": 1, "open": 2})
    assert math.isclose(metrics["alibaba"]["probability"], 2 / 6)
    assert math.isclose(metrics["open"]["probability"], 3 / 6)
    assert math.isclose(metrics["open"]["logit"] - metrics["alibaba"]["logit"], math.log(1.5))


def test_choose_token_id():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [17] if text == " Open" else [1, 2]
    assert m.choose_token_id(FakeTokenizer(), " Open") == 17
    import pytest
    with pytest.raises(ValueError):
        m.choose_token_id(FakeTokenizer(), " Alibaba")


def test_save(tmp_path):
    path = tmp_path / "results" / "experiment.json"
    row = {"prefix": "first6", "step7_edited": True, "greedy_token_text": " Open",
           "targets": {"open": {"probability": 0.7}, "alibaba": {"probability": 0.2}},
           "open_minus_alibaba_logit": 1.25, "edited_steps": list(range(1, 8))}
    m.save(path, {"results": [row]})
    assert path.exists()
    assert "first6" in path.with_suffix(".csv").read_text()
