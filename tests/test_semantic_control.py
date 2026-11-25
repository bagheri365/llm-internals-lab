import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('exp025',Path(__file__).resolve().parents[1]/'experiments/025-qwen-semantic-control/run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_hook_position_and_no_mutation():
    x=torch.ones(1,3,400); seen=[]
    y,=m.hook_for(-85,seen)(None,(x,))
    assert seen==[[1.,-84.]] and x.sum()==1200
    assert torch.count_nonzero(y-x)==1 and y[0,-1,386]==-84

def test_disabled_hook():
    seen=[]; assert m.hook_for(-98,seen,lambda:False)(None,(torch.zeros(1,1,400),)) is None
    assert not seen

def test_design():
    assert len({c[0] for c in m.CASES})==16 and len(m.DOSES)==5
    assert m.CASES[1][2].startswith(m.MATH)
