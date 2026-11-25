"""CPU-only checks for Experiment 014 helpers."""
import importlib.util
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'experiments/014-qwen-prompt-activation/run.py'
spec = importlib.util.spec_from_file_location('exp014', P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_candidate_selection_reproducible():
    a = mod.choose_candidates(mod.PREFIXES, 5, 14)
    assert a == mod.choose_candidates(mod.PREFIXES, 5, 14)
    assert a[0] == '' and len(a) == len(set(a)) == 5


def test_deltas():
    x = mod.summarize_deltas([3.0, 5.0], [-95.0, -93.0])
    assert x['deltas'] == [-98.0, -98.0]
    assert x['mean_abs_error_to_minus_98'] == 0


def test_invalid_inputs():
    import pytest
    with pytest.raises(ValueError):
        mod.choose_candidates(mod.PREFIXES, 0, 14)
    with pytest.raises(ValueError):
        mod.summarize_deltas([1.0], [1.0, 2.0])


def test_chat_ids_accepts_batch_encoding_and_tensor():
    import torch
    class FakeTokenizer:
        def __init__(self, wrap):
            self.wrap = wrap
        def apply_chat_template(self, *args, **kwargs):
            ids = torch.tensor([[1, 2, 3]])
            return {'input_ids': ids} if self.wrap else ids
    for wrap in (True, False):
        result = mod.chat_ids(FakeTokenizer(wrap), 'hello', 'cpu')
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3)
