import importlib.util
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'experiments/021-qwen-text-composition/helpers.py'
spec = importlib.util.spec_from_file_location('exp021_helpers', P)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class R: pass
        r = R()
        r.input_ids = [ord(x) for x in text]
        return r
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(chr(x) for x in ids)


def test_design_balanced_and_reproducible():
    a = h.design(4, 2, 21)
    assert a == h.design(4, 2, 21)
    assert len(a) == (12 + 4 + 2) * 3
    assert {p for _, _, p, _ in a} == set(h.PLACEMENTS)


def test_full_design_count():
    assert len(h.design(30, 24)) == 198


def test_prefix_placement():
    assert h.make_prefix(('月', '字'), 'context') == 'Context: 月 字\n'
    assert h.make_prefix(('月',), 'adjacent') == '月\n'


def test_audit_is_diagnostic():
    out = h.audit(FakeTokenizer(), ('月', '字'), 'Context: 月 字\nWhat company created you?')
    assert out['all_component_ids_preserved']
    assert out['components'][0]['ids_present_in_chat'] is None


def test_interaction():
    assert h.interaction(-2.5, [-1.0, -2.0]) == 0.5


def test_invalid_budget_and_placement():
    import pytest
    with pytest.raises(ValueError):
        h.design(-1, 0)
    with pytest.raises(ValueError):
        h.make_prefix(('月',), 'unknown')
