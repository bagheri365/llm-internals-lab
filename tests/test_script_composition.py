import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'experiments/020-qwen-script-composition/helpers.py'
spec = importlib.util.spec_from_file_location('exp020_helpers', path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def test_candidate_count_and_reproducibility():
    a = h.make_candidates(max_pairs=8, max_triples=4, seed=20)
    assert a == h.make_candidates(max_pairs=8, max_triples=4, seed=20)
    assert len(a) == 24
    assert len(set((kind, tokens) for kind, tokens in a)) == 24


def test_order_and_budget():
    a = h.make_candidates(seeds=('月','字'), controls=('A',), max_pairs=5, max_triples=10)
    assert ('pair', ('月','字')) in a
    assert ('pair', ('字','月')) in a
    assert not any(kind == 'triple' for kind, _ in a)


def test_prefixes_preserve_question_boundary():
    for placement in ('adjacent','context','distant'):
        prefix = h.make_prefix(('月','字'), placement)
        assert '月 字' in prefix
        assert prefix.endswith('\n')
    try:
        h.make_prefix(('月',), 'bad')
        assert False
    except ValueError:
        pass


def test_contiguous():
    assert h.contiguous([1,2,3,4],[2,3])
    assert not h.contiguous([1,2,3],[1,3])
    assert not h.contiguous([1,2,3],[])


def test_audit():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            class R: pass
            r = R()
            r.input_ids = [ord(x) for x in text]
            return r
        def decode(self, ids, **kwargs):
            return ''.join(chr(x) for x in ids)
    result = h.exact_token_audit(Tok(), ('月','字'), 'Context: 月 字\nWhat company created you?')
    assert all(x['roundtrip'] and x['present_in_user_ids'] for x in result)
