import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1]/'experiments/019-qwen-token-rarity/helpers.py'
spec = importlib.util.spec_from_file_location('exp019_helpers',PATH)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

class Tok:
    all_special_ids = [0]
    def decode(self, ids, **kwargs):
        return {0:'<special>',1:' hello',2:'world',3:'\ufffd',4:'\n',5:'42'}[ids[0]]
    def __call__(self,text,add_special_tokens=False):
        class Result: pass
        r=Result()
        r.input_ids = {' hello':[1],'world':[2],'\ufffd':[3],'\n':[4],'42':[5]}.get(text,[1,2])
        return r

def test_round_trip_and_rejections():
    t=Tok()
    assert h.usable(t,1)==' hello'
    assert h.usable(t,0) is None
    assert h.usable(t,3) is None
    assert h.usable(t,4) is None

def test_classes():
    assert h.token_class(' Hello')=='latin_letters'
    assert h.token_class('42')=='digits'
    assert h.token_class('你好')=='non_ascii'

def test_strata_reproducible_balanced():
    rows=[{'class':'latin_letters','token_id':i,'logprob':-i} for i in range(40)]
    a=h.strata(rows,3,19)
    assert a==h.strata(rows,3,19)
    assert len(a)==12
    assert [sum(x['likelihood_bin']==i for x in a) for i in range(4)]==[3]*4

def test_summary():
    s=h.summarize([{'class':'digits','likelihood_bin':0,'activation_delta':-2,'margin_change':1},
                   {'class':'digits','likelihood_bin':0,'activation_delta':3,'margin_change':5}])
    assert s[0]['mean_delta']==0.5
    assert s[0]['min_delta']==-2
    assert s[0]['mean_abs_delta']==2.5
    assert s[0]['mean_margin_change']==3
