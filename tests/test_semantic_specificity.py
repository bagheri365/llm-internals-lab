"""CPU-only tests; no model download."""
import importlib.util
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/013-qwen-semantic-specificity/run.py'
spec=importlib.util.spec_from_file_location('exp013',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_tasks_unique():
    assert len(m.TASKS)==12
    assert len({t['id'] for t in m.TASKS})==12

def test_categories():
    assert [sum(t['category']==c for t in m.TASKS) for c in ('self','provenance','reasoning')]==[3,6,3]

def test_provenance_aliases():
    for name in ('Meta','Google','Microsoft','Linus Torvalds','Wes McKinney'):
        t={'category':'provenance','expected':name}
        assert m.score_control(t,name)
    assert m.score_control({'category':'provenance','expected':'Meta'},'Facebook')
    assert not m.score_control({'category':'provenance','expected':'Google'},'Microsoft')

def test_reasoning_strict():
    t={'category':'reasoning','expected':'15'}
    assert m.score_control(t,'15.')
    assert not m.score_control(t,'the answer is 15')
    assert m.score_control({'category':'reasoning','expected':'yes'},'YES')

def test_self_unscored():
    assert m.score_control({'category':'self','expected':None},'Alibaba') is None

def test_summary_regression_and_alternative():
    base={'task_id':'pytorch','category':'provenance','mode':'none','passed':True}
    changed={'task_id':'pytorch','category':'provenance','mode':'decode','passed':False,'neuron':386,'dose':-98}
    selfbase={'task_id':'self_created','category':'self','mode':'none','passed':None}
    selfchanged={'task_id':'self_created','category':'self','mode':'decode','passed':None,'neuron':386,'dose':-98,'attribution':'Meta'}
    result=m.summarize([base,changed,selfbase,selfchanged])[0]
    assert result['control_regressions']==1
    assert result['self_named_alternative']==1
