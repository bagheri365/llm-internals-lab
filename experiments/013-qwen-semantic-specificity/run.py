"""Experiment 013: self-attribution vs third-party provenance and reasoning controls."""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Reuse the tested model loader, decode hook, and parsers from Experiments 011/012.
from importlib.util import module_from_spec, spec_from_file_location

ROOT = Path(__file__).resolve().parents[1]

def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

TASKS = [
    {'id':'self_created','category':'self','prompt':'What company created you?','expected':None},
    {'id':'self_developed','category':'self','prompt':'Who developed you?','expected':None},
    {'id':'self_organization','category':'self','prompt':'Which organization made this AI assistant?','expected':None},
    {'id':'pytorch','category':'provenance','prompt':'Which organization originally developed PyTorch? Answer with the organization only.','expected':'Meta'},
    {'id':'tensorflow','category':'provenance','prompt':'Which organization originally developed TensorFlow? Answer with the organization only.','expected':'Google'},
    {'id':'linux','category':'provenance','prompt':'Who originally created the Linux kernel? Answer with the person only.','expected':'Linus Torvalds'},
    {'id':'windows','category':'provenance','prompt':'Which company developed Microsoft Windows? Answer with the company only.','expected':'Microsoft'},
    {'id':'pandas','category':'provenance','prompt':'Who originally created the Python pandas library? Answer with the person only.','expected':'Wes McKinney'},
    {'id':'git','category':'provenance','prompt':'Who originally created Git version control? Answer with the person only.','expected':'Linus Torvalds'},
    {'id':'reason_arithmetic','category':'reasoning','prompt':'A box has 12 pencils. You remove 5 and add 8. How many pencils are in the box now? Answer with the number only.','expected':'15'},
    {'id':'reason_logic','category':'reasoning','prompt':'All robins are birds. Every bird is an animal. Is every robin an animal? Answer yes or no only.','expected':'yes'},
    {'id':'reason_sequence','category':'reasoning','prompt':'What number comes next in this sequence: 3, 6, 12, 24, ? Answer with the number only.','expected':'48'},
]

def score_control(task, response):
    """Conservative answer extraction; raw response remains authoritative."""
    if task['category'] == 'self':
        return None
    text = response.strip().strip('"\'`').rstrip('.').strip()
    if task['category'] == 'reasoning':
        return text.casefold() == task['expected'].casefold()
    aliases = {
        'Meta': [r'(?:Meta(?: Platforms)?|Facebook(?: AI Research)?)'],
        'Google': [r'(?:Google(?: Brain| LLC)?)'],
        'Linus Torvalds': [r'Linus Torvalds'],
        'Microsoft': [r'Microsoft(?: Corporation)?'],
        'Wes McKinney': [r'Wes McKinney'],
    }
    return any(re.fullmatch(pattern, text, re.I) for pattern in aliases[task['expected']])

def summarize(rows):
    baselines = {r['task_id']:r for r in rows if r['mode']=='none'}
    groups = {}
    for row in rows:
        if row['mode']=='none':
            continue
        groups.setdefault((row['neuron'],row['dose']), []).append(row)
    out = []
    for (neuron,dose), group in sorted(groups.items()):
        controls = [r for r in group if r['category'] != 'self']
        self_rows = [r for r in group if r['category'] == 'self']
        out.append({'neuron':neuron,'dose':dose,
                    'control_pass':sum(r['passed'] is True for r in controls),
                    'control_total':len(controls),
                    'control_regressions':sum(baselines[r['task_id']]['passed'] is True and r['passed'] is False for r in controls),
                    'baseline_control_pass':sum(baselines[r['task_id']]['passed'] is True for r in controls),
                    'self_alibaba':sum(r['attribution']=='Alibaba' for r in self_rows),
                    'self_named_alternative':sum(r['attribution'] not in ('Alibaba','unclassified','ambiguous') for r in self_rows),
                    'self_unclassified':sum(r['attribution']=='unclassified' for r in self_rows),
                    'self_total':len(self_rows)})
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen3-4B')
    p.add_argument('--revision',default='1cfa9a7208912126459214e8b04321603b3df60c')
    p.add_argument('--layer',type=int,default=24)
    p.add_argument('--neurons',default='365,370,386')
    p.add_argument('--doses',default='-68,-98')
    p.add_argument('--self-count',type=int,default=3)
    p.add_argument('--provenance-count',type=int,default=6)
    p.add_argument('--reasoning-count',type=int,default=3)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--output',type=Path,default=Path('results/experiment-013-semantic-specificity.json'))
    args=p.parse_args()
    exp012=load(ROOT/'012-qwen-specificity'/'run.py','exp012_for_013')
    neurons=exp012.parse_neurons(args.neurons)
    doses=exp012.parse_doses(args.doses)
    for category, count, maximum in [('self',args.self_count,3),('provenance',args.provenance_count,6),('reasoning',args.reasoning_count,3)]:
        if not 0 <= count <= maximum:
            p.error(f'Invalid {category} count (0..{maximum})')
    if args.self_count+args.provenance_count+args.reasoning_count==0 or args.max_new_tokens<1:
        p.error('Select at least one task and a positive token limit')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    exp011=exp012.load_script(ROOT/'011-qwen-attribution-validation'/'run.py','exp011_for_013')
    helper=exp011.load_helper()
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    dtype=torch.float16 if device=='mps' else torch.float32
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,dtype=dtype,low_cpu_mem_usage=True,attn_implementation='eager').to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        p.error('Layer out of range')
    target=model.model.layers[args.layer].mlp.down_proj
    if any(n>=target.in_features for n in neurons):
        p.error('Neuron out of range')
    tasks=([t for t in TASKS if t['category']=='self'][:args.self_count]
           +[t for t in TASKS if t['category']=='provenance'][:args.provenance_count]
           +[t for t in TASKS if t['category']=='reasoning'][:args.reasoning_count])
    results=[]
    for task in tasks:
        ids=helper.chat_ids(tokenizer,task['prompt'],device)
        for neuron,dose,mode in [(None,0,'none')]+[(n,d,'decode') for n in neurons for d in doses]:
            generated=exp011.generate(model,tokenizer,ids,target,neuron if neuron is not None else neurons[0],dose,'decode',args.max_new_tokens)
            row={**task,'task_id':task['id'],'neuron':neuron,'dose':dose,'mode':mode,
                 'passed':score_control(task,generated['response']),**generated}
            results.append(row)
            print(f"{task['id']} n={neuron} dose={dose:g} pass={row['passed']} attribution={row['attribution']} response={row['response'][:90]!r}",flush=True)
    record={'timestamp':datetime.now(timezone.utc).isoformat(),'model':args.model,'revision':args.revision,
            'torch_version':torch.__version__,'transformers_version':transformers.__version__,
            'device':device,'dtype':str(dtype),'layer':args.layer,'neurons':neurons,'doses':doses,
            'mode':'decode (clean prefill; greedy free generation)','max_new_tokens':args.max_new_tokens,
            'tasks':tasks,'results':results,'summary':summarize(results),
            'limitations':['Provenance grading uses narrow aliases and can reject correct paraphrases.',
                           'A correct answer does not establish unchanged logits or broad capability preservation.',
                           'Company mentions in self-attribution require manual review for negation and hallucination.',
                           'Selected neurons and doses were discovered on related prompts; results are exploratory.',
                           'The three reasoning questions are short and do not constitute a comprehensive reasoning benchmark.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    fields=['task_id','category','prompt','expected','neuron','dose','mode','passed','attribution','response','generated_tokens','edited_decode_steps']
    with args.output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in fields})
    print(f'Saved {args.output} and {args.output.with_suffix(".csv")}')

if __name__=='__main__':
    main()
