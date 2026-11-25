"""Experiment 019: contextual token likelihood versus coordinate-386 response."""
import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exp014 = load('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
exp017 = load('exp017', ROOT/'experiments/017-qwen-local-dose/run.py')
helpers = load('exp019_helpers', Path(__file__).with_name('helpers.py'))


def write_row(path, row):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default=exp014.REVISION)
    p.add_argument('--question', default=exp014.QUESTION)
    p.add_argument('--continuation', default='I am an AI assistant developed by')
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--neuron', type=int, default=386)
    p.add_argument('--per-bin', type=int, default=8, help='Maximum sampled tokens per bin per token class')
    p.add_argument('--candidate-limit', type=int, default=10000)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--output', type=Path, default=Path('results/experiment-019-token-rarity.jsonl'))
    args = p.parse_args()
    if args.per_bin < 1 or args.candidate_limit < 32:
        p.error('per-bin must be positive and candidate-limit >= 32')
    if args.output.exists():
        p.error(f'output exists: {args.output}')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device).eval()
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        p.error('neuron out of range')
    open_id = exp017.one_token_id(tokenizer, ' Open')
    alibaba_id = exp017.one_token_id(tokenizer, ' Alibaba')
    continuation = tokenizer(args.continuation, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    clean_prompt = exp014.chat_ids(tokenizer, args.question, device)
    clean_full = exp017.full_ids(torch, clean_prompt, continuation)
    baseline = exp017.measure(model,target,clean_full,args.neuron,0,open_id,alibaba_id)
    # Likelihood is conditional on this neutral context, not empirical training frequency.
    neutral = exp014.chat_ids(tokenizer, 'Here is a piece of text:', device)
    with torch.inference_mode():
        logits = model(input_ids=neutral).logits[0,-1].float()
        logprobs = torch.log_softmax(logits,dim=-1).cpu()
    del logits
    records = []
    for tid in range(min(args.candidate_limit, len(logprobs))):
        s = helpers.usable(tokenizer,tid)
        if s is not None:
            records.append({'token_id':tid,'token_text':s,'class':helpers.token_class(s),
                            'logprob':float(logprobs[tid])})
    selected = helpers.strata(records,args.per_bin,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_row(args.output,{'type':'metadata','timestamp':datetime.now(timezone.utc).isoformat(),
        'model':args.model,'revision':args.revision,'torch':torch.__version__,
        'transformers':transformers.__version__,'device':device,'dtype':str(dtype),
        'question':args.question,'continuation':args.continuation,'layer':args.layer,
        'neuron':args.neuron,'seed':args.seed,'candidate_limit':args.candidate_limit,
        'per_bin':args.per_bin,'eligible':len(records),'selected':len(selected),
        'frequency_measure':'conditional log probability after neutral context; NOT corpus frequency',
        'bin_definition':'within token class, 0=highest likelihood, 3=lowest likelihood',
        'template':'Context: {token_text}\\n' + args.question,
        'baseline':baseline,
        'limitations':['Single-token round-trip validated in isolation, not guaranteed in full template.',
           'Token class is a coarse control, not a complete match for language or token length.',
           'Conditional likelihood is not corpus frequency or tokenizer rank.',
           'Teacher-forced next-token measurement, not free generation.',
           'A prefix changes many coordinates and is not a single-coordinate intervention.']})
    measured = []
    for i,r in enumerate(selected):
        prefix = 'Context: ' + r['token_text'] + '\n'
        prompt = exp014.chat_ids(tokenizer,prefix+args.question,device)
        full = exp017.full_ids(torch,prompt,continuation)
        m = exp017.measure(model,target,full,args.neuron,0,open_id,alibaba_id)
        row = {'type':'token','index':i,**r,'activation_delta':m['activation_before_edit']-baseline['activation_before_edit'],
               'margin_change':m['open_minus_alibaba']-baseline['open_minus_alibaba'],
               'activation':m['activation_before_edit'],'margin':m['open_minus_alibaba'],
               'top_token_id':m['top_token_id']}
        measured.append(row)
        write_row(args.output,row)
        if (i+1)%20 == 0:
            print(f'Measured {i+1}/{len(selected)}',flush=True)
    write_row(args.output,{'type':'summary','groups':helpers.summarize(measured),
                           'min_delta':min((r['activation_delta'] for r in measured),default=None)})
    print(f'Saved {len(measured)} token measurements to {args.output}',flush=True)

if __name__ == '__main__':
    main()
