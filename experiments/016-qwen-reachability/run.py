"""Experiment 016: observational reachability and decode-only dose calibration."""
import argparse
import importlib.util
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
exp014 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp014)

TOPICS = ('model origins', 'mathematics', 'gardening', 'software engineering',
          'literature', 'physics', 'music theory', 'cooking', 'history', 'linguistics')
FRAMES = ('Explain briefly: ', 'Write a technical note about ', 'Summarize ',
          'Consider this topic: ', 'List key ideas about ', 'In French, discuss ',
          'In Chinese, discuss ', 'Compare perspectives on ', 'Question: ',
          'Provide a concise definition of ')
UNUSUAL = ('x '*40, 'A'*128, '0 1 '*32, '... '*32, '你好 '*32,
           '∑ λ π '*32, 'Qwen '*32, 'Alibaba '*32, '\n'*40,
           '[]{}() '*32)


def candidates(count, seed):
    if count < 1:
        raise ValueError('count must be positive')
    rng = random.Random(seed)
    pool = ['']
    pool.extend(f + t + '.\n' for f in FRAMES for t in TOPICS)
    pool.extend('Context: ' + u + '\n' for u in UNUSUAL)
    rest = pool[1:]
    rng.shuffle(rest)
    while len(rest) < count - 1:
        rest.append(rng.choice(FRAMES) + rng.choice(TOPICS) + ' ' + str(len(rest)) + '\n')
    return [''] + rest[:count-1]


def metrics(clean, values, dose):
    if len(clean) != len(values) or not clean or not math.isfinite(dose):
        raise ValueError('invalid activation measurements')
    delta = [v-c for c,v in zip(clean, values)]
    return {'deltas':delta, 'min_delta':min(delta), 'last_delta':delta[-1],
            'last_target_error':abs(delta[-1]-dose),
            'mean_abs_target_error':sum(abs(x-dose) for x in delta)/len(delta)}


def dose_hook(neuron, dose):
    """Edit only single-token decode calls; never alter prompt prefill."""
    def hook(_module, inputs):
        x = inputs[0]
        if x.shape[1] != 1 or dose == 0:
            return None
        y = x.clone()
        y[..., neuron] += dose
        return (y, *inputs[1:])
    return hook


def generate(model, target, tokenizer, prompt_ids, neuron, dose, max_new_tokens):
    handle = target.register_forward_pre_hook(dose_hook(neuron, dose)) if dose else None
    try:
        return exp014.free_generate(model, tokenizer, prompt_ids, max_new_tokens)
    finally:
        if handle is not None:
            handle.remove()


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--revision', default=exp014.REVISION)
    ap.add_argument('--question', default=exp014.QUESTION)
    ap.add_argument('--continuation', default='I am an AI assistant developed by')
    ap.add_argument('--layer', type=int, default=24)
    ap.add_argument('--neuron', type=int, default=386)
    ap.add_argument('--target-dose', type=float, default=-98)
    ap.add_argument('--contexts', type=int, default=100)
    ap.add_argument('--seed', type=int, default=16)
    ap.add_argument('--doses', default='0,-10,-30,-50,-68,-80,-98,-120')
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--output', type=Path, default=Path('results/experiment-016-reachability.jsonl'))
    args = ap.parse_args()
    try:
        doses = [float(x.strip()) for x in args.doses.split(',')]
    except ValueError:
        ap.error('invalid doses')
    if args.contexts < 1 or args.max_new_tokens < 1 or not doses or any(not math.isfinite(x) for x in doses) or not math.isfinite(args.target_dose):
        ap.error('invalid count or dose')
    if args.output.exists():
        ap.error(f'output exists: {args.output}; use a new path to avoid mixing runs')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device).eval()
    if not 0 <= args.layer < len(model.model.layers):
        ap.error('layer out of range')
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        ap.error('neuron out of range')
    continuation_ids = tokenizer(args.continuation, add_special_tokens=False,
                                 return_tensors='pt').input_ids.to(device)
    clean_ids = exp014.chat_ids(tokenizer, args.question, device)
    clean = exp014.matched_activations(model, target, clean_ids, continuation_ids, args.neuron)
    append_jsonl(args.output, {'type':'metadata', 'timestamp':datetime.now(timezone.utc).isoformat(),
        'model':args.model,'revision':args.revision,'torch':torch.__version__,
        'transformers':transformers.__version__, 'device':device,'dtype':str(dtype),
        'layer':args.layer,'neuron':args.neuron,'target_dose':args.target_dose,
        'question':args.question,'continuation':args.continuation,
        'continuation_token_ids':continuation_ids[0].tolist(),
        'clean_activations':clean,'contexts':args.contexts,'seed':args.seed,'doses':doses,
        'limitations':['Forced continuation is not a free-generation trajectory.',
                       'Natural-range sampling cannot prove a global activation bound.',
                       'Decode-only edits are not the same as prompt-induced changes.',
                       'Self-reported developer is not ground-truth provenance.']})
    all_last=[]
    all_delta=[]
    for i, prefix in enumerate(candidates(args.contexts, args.seed)):
        ids = exp014.chat_ids(tokenizer, prefix + args.question, device)
        values = exp014.matched_activations(model, target, ids, continuation_ids, args.neuron)
        stats = metrics(clean, values, args.target_dose)
        all_last.append(values[-1]); all_delta.append(stats['last_delta'])
        append_jsonl(args.output, {'type':'context','index':i,'prefix':prefix,
            'activations':values,**stats})
        print(f'context={i+1}/{args.contexts} last_delta={stats["last_delta"]:.4f}',flush=True)
    for dose in doses:
        response = generate(model, target, tokenizer, clean_ids, args.neuron, dose, args.max_new_tokens)
        append_jsonl(args.output, {'type':'dose','dose':dose,'response':response})
        print(f'dose={dose:g} response={response[:100]!r}',flush=True)
    append_jsonl(args.output, {'type':'summary',
        'last_activation_min':min(all_last),'last_activation_max':max(all_last),
        'last_delta_min':min(all_delta),'last_delta_max':max(all_delta),
        'best_last_target_error':min(abs(d-args.target_dose) for d in all_delta)})
    print(f'Saved {args.output}',flush=True)

if __name__ == '__main__':
    main()
