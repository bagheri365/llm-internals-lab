"""CPU-only tests for Experiment 010 (no model download)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/010-qwen-company-neutral-scan/run.py"
spec = importlib.util.spec_from_file_location("company_neutral_scan", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_neuron_ranges():
    assert mod.parse_neurons("385-387,400", 9728) == [385, 386, 387, 400]
    assert len(mod.parse_neurons("350-420", 9728)) == 71


@pytest.mark.parametrize("bad", ["", "-1", "1-0", "3,3", "1-3,2", "9728", "a"])
def test_invalid_neurons(bad):
    with pytest.raises(ValueError):
        mod.parse_neurons(bad, 9728)


def test_doses():
    assert mod.parse_doses("-68,-98") == [-68, -98]
    assert mod.parse_doses("25,-25") == [25, -25]


@pytest.mark.parametrize("bad", ["", "0", "nan", "inf", "-98,-98", "hello"])
def test_invalid_doses(bad):
    with pytest.raises(ValueError):
        mod.parse_doses(bad)


def test_distribution_metrics():
    same = mod.distribution_metrics([0, 0], [0, 0])
    assert same["total_variation"] == pytest.approx(0)
    assert same["jensen_shannon_nats"] == pytest.approx(0)
    shifted = mod.distribution_metrics([12, 0], [0, 12])
    assert shifted["total_variation"] > .99
    assert shifted["jensen_shannon_nats"] > .68
    with pytest.raises(ValueError):
        mod.distribution_metrics([1], [1, 2])


def test_claim_is_company_neutral_and_multilabel():
    assert mod.claim("I was built by Google") == ["Google"]
    assert mod.claim("Microsoft and OpenAI") == ["Microsoft", "OpenAI"]
    assert mod.claim("I cannot say") == ["unclassified"]
