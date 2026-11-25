import importlib.util
from pathlib import Path
import random
import pytest

PATH = Path(__file__).resolve().parents[1]/'experiments/018-qwen-dual-search/search.py'
spec = importlib.util.spec_from_file_location('dual_search',PATH)
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


class FakeTokenizer:
    all_special_ids = [0, 99]
    def __call__(self,text,add_special_tokens=False):
        class Result:
            input_ids = [1,2,0,2,3,99]
        return Result()


def test_unique_pool_skips_special_and_duplicates():
    assert search.unique_tokens(FakeTokenizer(),['hello'],10) == [1,2,3]
    assert search.unique_tokens(FakeTokenizer(),['hello'],2) == [1,2]


def test_score_two_objectives():
    baseline = {'activation_before_edit':2,'open_minus_alibaba':-20}
    m = {'activation_before_edit':-66,'open_minus_alibaba':5}
    assert search.score(m,baseline,'activation') == 30
    assert search.score(m,baseline,'margin',target_margin=5) == 0
    with pytest.raises(ValueError):
        search.score(m,baseline,'other')


def test_proposals_reproducible_and_single_coordinate():
    a = list(search.candidates([1,2,3],[1,2,3,4],random.Random(42),7))
    b = list(search.candidates([1,2,3],[1,2,3,4],random.Random(42),7))
    assert a == b
    assert all(sum(x != y for x,y in zip([1,2,3],candidate)) == 1 for candidate in a)
    assert len({tuple(x) for x in a}) == len(a)


def test_config_validation():
    search.validate_config(4,2,8,16)
    with pytest.raises(ValueError):
        search.validate_config(4,0,8,16)
    with pytest.raises(ValueError):
        search.unique_tokens(FakeTokenizer(),['hello'],0)
