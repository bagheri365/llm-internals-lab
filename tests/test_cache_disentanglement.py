"""CPU-only tests for Experiment 007."""
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/007-qwen-cache-disentanglement/run.py"
spec = importlib.util.spec_from_file_location("cache_disentanglement", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_factorial_conditions():
    values = m.conditions()
    assert len(values) == 8
    assert len(set(values)) == 8
    assert values[0] == ("baseline", "clean", False)
    assert values[-1] == ("first6", "edited", True)


@pytest.mark.parametrize("history,edit7,expected", [
    ("clean", False, set()), ("clean", True, {7}),
    ("edited", False, set(range(1, 7))),
    ("edited", True, set(range(1, 8))),
])
def test_steps(history, edit7, expected):
    assert m.steps_for(history, edit7) == expected


def test_invalid_history():
    with pytest.raises(ValueError):
        m.steps_for("other", False)


def test_pairwise_effects():
    rows = [{"prefix": prefix, "history": history, "step7_edited": on,
             "open_minus_alibaba_logit": (10 if on else -2) + (1 if history == "edited" else 0)}
            for prefix, history, on in m.conditions()]
    effects = m.pairwise_effects(rows)
    assert len(effects) == 4
    assert all(r["step7_margin_shift"] == 12 for r in effects)


def test_save(tmp_path):
    rows = [{"prefix": "baseline", "history": "clean", "step7_edited": False,
             "greedy_token_text": " Alibaba", "targets": {"open": {"probability": 0.1},
             "alibaba": {"probability": 0.9}}, "open_minus_alibaba_logit": -2,
             "edited_steps": []}]
    target = tmp_path / "nested" / "result.json"
    m.save(target, {"results": rows})
    assert target.is_file()
    assert "baseline,clean,False" in target.with_suffix(".csv").read_text()
