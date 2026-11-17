"""CPU-only unit tests; do not download model weights."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/001-qwen-identity-steering/run.py"
spec = importlib.util.spec_from_file_location("qwen_identity_experiment", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_additive_intervention_targets_only_one_neuron():
    original = torch.zeros((2, 3, 512))
    changed = module.add_dose_to_neuron(original, 386, -98)
    assert torch.all(changed[..., 386] == -98)
    assert torch.count_nonzero(changed).item() == 6
    assert torch.count_nonzero(original).item() == 0


def test_invalid_neuron_raises():
    with pytest.raises(IndexError):
        module.add_dose_to_neuron(torch.zeros(2, 10), 386, -98)


def test_chat_template_batch_encoding_extracts_input_ids():
    ids = torch.tensor([[1, 2, 3]])
    assert module.input_ids_from_chat_template({"input_ids": ids}) is ids
    assert module.input_ids_from_chat_template(ids) is ids
