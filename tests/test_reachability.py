import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/016-qwen-reachability/run.py'
spec = importlib.util.spec_from_file_location('exp016', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_candidates_deterministic():
    a = mod.candidates(20, 16)
    assert a == mod.candidates(20, 16)
    assert a[0] == '' and len(a) == 20 and len(set(a)) == 20


def test_metrics():
    m = mod.metrics([2, 3], [1, 2], -98)
    assert m['deltas'] == [-1, -1]
    assert m['last_target_error'] == 97
    with pytest.raises(ValueError):
        mod.metrics([1], [1, 2], -98)


def test_dose_hook_decode_only():
    import torch
    hook = mod.dose_hook(1, -98)
    prefill = torch.zeros((1, 3, 4))
    assert hook(None, (prefill,)) is None
    decode = torch.zeros((1, 1, 4))
    result = hook(None, (decode,))[0]
    assert result[0, 0, 1].item() == -98
    assert decode[0, 0, 1].item() == 0
    assert result[0, 0, 0].item() == 0


def test_append_jsonl(tmp_path):
    p = tmp_path / 'a.jsonl'
    mod.append_jsonl(p, {'type':'test','value':1})
    assert p.read_text().strip() == '{"type": "test", "value": 1}'
