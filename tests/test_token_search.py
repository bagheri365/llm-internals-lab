"""CPU-only tests; no model download."""
import importlib.util
from pathlib import Path
import random
import pytest

P = Path(__file__).resolve().parents[1] / 'experiments/015-qwen-token-search/run.py'
spec = importlib.util.spec_from_file_location('exp015', P)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_score_last_and_all():
    assert mod.score([2,3], [-3,-2], -5, 'last') == 0
    assert mod.score([2,3], [-3,-2], -5, 'all') == 0
    assert mod.score([2,3], [2,-2], -5, 'all') == 2.5


def test_proposal_deterministic_and_one_piece_changes():
    start = ('A','B','A')
    first = mod.propose(start, random.Random(3), ('A','B','C'))
    assert first == mod.propose(start, random.Random(3), ('A','B','C'))
    assert sum(a != b for a,b in zip(start,first)) == 1


def test_invalid_scores_and_proposals():
    with pytest.raises(ValueError):
        mod.score([1], [1,2], -98, 'last')
    with pytest.raises(ValueError):
        mod.score([1], [1], -98, 'middle')
    with pytest.raises(ValueError):
        mod.propose((), random.Random(0))
