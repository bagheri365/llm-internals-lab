import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/017-qwen-local-dose/run.py'
spec = importlib.util.spec_from_file_location('exp017', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_doses_deduplicated_and_baseline():
    assert m.parse_doses('-5,0,-5,-98') == [0.0, -5.0, -98.0]


@pytest.mark.parametrize('value', ['1', '-inf', 'nan', '', '-5,abc'])
def test_bad_doses(value):
    with pytest.raises(ValueError):
        m.parse_doses(value)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return type('Tokens', (), {'input_ids': [1] if text == ' Open' else [1, 2]})()


def test_one_token_guard():
    assert m.one_token_id(FakeTokenizer(), ' Open') == 1
    with pytest.raises(ValueError):
        m.one_token_id(FakeTokenizer(), ' Alibaba')


def test_last_position_hook_only_changes_target():
    torch = pytest.importorskip('torch')
    x = torch.zeros((1, 3, 5))
    x[0, -1, 2] = 3
    seen = []
    result = m.last_position_hook(2, -2, seen)(None, (x,))
    assert seen == [3.0]
    assert x[0, -1, 2] == 3
    assert result[0][0, -1, 2] == 1
    assert torch.equal(result[0][0, :-1], x[0, :-1])
