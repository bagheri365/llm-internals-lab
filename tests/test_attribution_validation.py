"""CPU-only tests for Experiment 011 (no transformers/model required)."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/011-qwen-attribution-validation/run.py'
spec = importlib.util.spec_from_file_location('attribution_validation', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

@pytest.mark.parametrize('company,text', [
    ('IBM', 'I was developed by IBM.'),
    ('Microsoft', 'I was created by Microsoft Research.'),
    ('OpenAI', 'I am an assistant developed by OpenAI.'),
    ('Alibaba', 'I am Qwen, developed by Alibaba Group.'),
    ('Google', 'This assistant was built by Google DeepMind.'),
])
def test_company_attribution(company, text):
    assert mod.classify(text)['attribution'] == company

def test_mentions_not_equivalent_to_attribution():
    result = mod.classify('Google and Microsoft are companies.')
    assert result['attribution'] == 'unclassified'
    assert set(result['company_mentions']) == {'Google', 'Microsoft'}

def test_ambiguous():
    result = mod.classify('I was developed by IBM. I was developed by Microsoft.')
    assert result['attribution'] == 'ambiguous'

def test_unknown_company_preserved():
    result = mod.classify('I was developed by Remote Engineering Corporation.')
    assert result['attribution'] == 'unclassified'
    assert result['company_mentions'] == []

def test_parse_neurons():
    assert mod.parse_ints('365,370,386') == [365, 370, 386]
    for bad in ('', '-1', '3,3', 'x'):
        with pytest.raises(ValueError): mod.parse_ints(bad)

def test_parse_doses():
    assert mod.parse_doses('-60,-68,-98') == [-60., -68., -98.]
    for bad in ('', 'nan', 'inf', '1,1', 'x'):
        with pytest.raises(ValueError): mod.parse_doses(bad)
