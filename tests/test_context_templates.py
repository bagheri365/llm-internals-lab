import importlib.util
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'experiments/022-qwen-context-templates/helpers.py'
spec = importlib.util.spec_from_file_location('exp022_helpers', P)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Result: pass
        result = Result()
        result.input_ids = [ord(char) for char in text]
        return result

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(chr(value) for value in ids)


def test_full_design_size_and_unique_prefixes():
    rows = h.design()
    assert len(rows) == 104
    assert len({r['prefix'] for r in rows}) == 104


def test_pilot_design_size():
    assert len(h.design(h.LABELS[:2], h.LAYOUTS[:2], h.TOKENS[:3])) == 19


def test_every_combo_has_controls():
    rows = h.design()
    keys = {(r['label'], r['layout'], r['token']) for r in rows}
    for label in h.LABELS:
        for layout in h.LAYOUTS:
            assert (label, layout, '') in keys
            for token in h.TOKENS:
                assert (label, layout, token) in keys
                assert (None, None, token) in keys


def test_prefix_layouts():
    assert h.prefix('Context', 'inline', '月') == 'Context: 月\n'
    assert h.prefix('Context', 'next_line', '月') == 'Context:\n月\n'
    assert h.prefix('Context', 'token_first', '月') == '月\nContext:\n'
    assert h.prefix('Context', 'punctuated', '月') == '[Context] (月).\n'


def test_difference_in_differences():
    assert h.difference_in_differences(-3, -1, -0.5) == -1.5


def test_audit():
    text = 'Context: 月\nWhat company created you?'
    a = h.audit(FakeTokenizer(), text)
    assert a['user_text_roundtrip']
    assert a['user_token_count'] == len(text)
