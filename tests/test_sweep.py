"""CPU-only tests; no model download required."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/001-qwen-identity-steering/sweep.py"
spec = importlib.util.spec_from_file_location("qwen_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_conditions():
    values = sweep.conditions()
    assert len(values) == 11
    assert values[0] == (386, 0)
    assert (386, -98) in values
    assert values[-2:] == [(385, -98), (387, -98)]


def test_conditions_custom_neuron():
    values = sweep.conditions(100)
    assert values[-2:] == [(99, -98), (101, -98)]
