import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('restoration',Path(__file__).resolve().parents[1]/'experiments/023-qwen-restoration/run.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_decomposition():
    r=m.decompose(-20,-16,-15,-21)
    assert r['total']==4 and r['target_in_modified']==-1 and r['residual_at_clean_activation']==5
    assert r['interaction']==0

@pytest.mark.parametrize('clamp,delta,expected',[(2,0,2),(None,-98,-95),(None,0,3)])
def test_hook(clamp,delta,expected):
    torch=pytest.importorskip('torch'); x=torch.zeros(1,3,5); x[0,-1,2]=3; seen=[]
    result=m.hook_for(2,seen,clamp,delta)(None,(x,)); y=x if result is None else result[0]
    assert seen==[(3,expected)] and y[0,-1,2]==expected and x[0,-1,2]==3
    assert torch.equal(y[0,:-1],x[0,:-1]); assert torch.equal(y[0,-1,[0,1,3,4]],x[0,-1,[0,1,3,4]])
