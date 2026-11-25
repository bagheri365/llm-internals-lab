"""CPU-only checks for Experiment 005 scheduling and output format."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/005-qwen-critical-token/run.py"
spec = importlib.util.spec_from_file_location("critical_token", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_conditions_count_and_uniqueness():
    conditions = module.conditions()
    assert len(conditions) == 21
    assert len({mode for mode, _ in conditions}) == 21
    assert conditions[0] == ("none", ())
    assert conditions[-1] == ("all", None)


@pytest.mark.parametrize("k", range(2, 11))
def test_cumulative_windows(k):
    assert dict(module.conditions())[f"first{k}"] == tuple(range(1, k + 1))


@pytest.mark.parametrize("k", range(1, 11))
def test_single_step_windows(k):
    steps = dict(module.conditions())[f"step{k}"]
    assert steps == (k,)
    assert module.should_edit(steps, k)
    assert not module.should_edit(steps, k + 1)


def test_first_divergence():
    assert module.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert module.first_divergence([1, 2], [1, 2]) is None
    assert module.first_divergence([1, 2], [1, 2, 3]) == 2


def test_save(tmp_path):
    path = tmp_path / "results" / "test.json"
    csv_path = module.save(path, {"results": [{"mode": "none", "edited_steps": [],
        "first_divergence_token": None, "response": "Qwen"}]})
    assert path.exists()
    assert csv_path.exists()
    assert "Qwen" in csv_path.read_text()
