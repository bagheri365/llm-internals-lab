import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('response',Path(__file__).resolve().parents[1]/'experiments/024-qwen-causal-response/run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_grid():
    assert len(m.FULL)==23 and len(m.PILOT)==8
    assert set(m.PILOT)<=set(m.FULL)

def test_nonuniform_quadratic():
    r=m.slopes([dict(after=x,margin=x*x) for x in (3,0,1)])
    assert r[0]['slope']==1 and r[1]['slope']==4
    assert r[1]['curvature_at_left']==2

def test_duplicate():
    with pytest.raises(ValueError): m.slopes([dict(after=1,margin=0)]*2)
