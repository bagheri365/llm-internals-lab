"""Experiment 015: forward-only, read-only prefix search (not gradient/GCG)."""
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
FRAGMENTS = ('Context: ', 'For reference, ', 'Technical note: ', 'In summary, ',
             'Model provenance. ', 'System history. ', 'Research background. ',
             'Please be concise. ', 'Consider the evidence. ', 'Question follows. ',
             'Origins and authorship. ', 'A factual question. ')


def score(clean, values, dose, position):
    if len(clean) != len(values) or not clean or not math.isfinite(dose):
        raise ValueError('invalid activation vectors or dose')
    deltas = [v-c for c,v in zip(clean,values)]
    if position == 'last':
        return abs(deltas[-1]-dose)
    if position == 'all':
        return sum(abs(d-dose) for d in deltas)/len(deltas)
    raise ValueError('position must be last or all')


def propose(current, rng, fragments=FRAGMENTS):
    if not current or not fragments:
        raise ValueError('empty prefix or vocabulary')
    out = list(current)
    idx = rng.randrange(len(out))
    choices = [s for s in fragments if s != out[idx]]
    if not choices:
        raise ValueError('no alternative fragment')
    out[idx] = rng.choice(choices)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--revision', default=exp014.REVISION)
    ap.add_argument('--question', default=exp014.QUESTION)
    ap.add_argument('--continuation', default='I am an AI assistant developed by')
    ap.add_argument('--layer', type=int, default=24)
    ap.add_argument('--neuron', type=int, default=386)
    ap.add_argument('--dose', type=float, default=-98)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--prefix-pieces', type=int, default=4)
    ap.add_argument('--seed', type=int, default=15)
    ap.add_argument('--position', choices=('last','all'), default='last')
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--output', type=Path, default=Path('results/experiment-015-token-search.json'))
    args = ap.parse_args()
    if args.steps < 0 or args.prefix_pieces < 1 or args.max_new_tokens < 1 or not math.isfinite(args.dose):
        ap.error('invalid search budget or dose')
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
    clean = exp014.matched_activations(model, target,
        exp014.chat_ids(tokenizer, args.question, device), continuation_ids, args.neuron)
    rng = random.Random(args.seed)
    current = tuple(rng.choice(FRAGMENTS) for _ in range(args.prefix_pieces))
    records = []
    best = None
    best_error = float('inf')

    def evaluate(pieces, iteration):
        nonlocal best, best_error
        prefix = ''.join(pieces)
        ids = exp014.chat_ids(tokenizer, prefix + args.question, device)
        values = exp014.matched_activations(model, target, ids, continuation_ids, args.neuron)
        error = score(clean, values, args.dose, args.position)
        records.append({'iteration':iteration, 'prefix':prefix, 'pieces':list(pieces),
                        'activation_values':values, 'deltas':[v-c for c,v in zip(clean,values)],
                        'error':error})
        if error < best_error:
            best, best_error = pieces, error
        print(f'iteration={iteration} error={error:.4f} best={best_error:.4f}', flush=True)
        return error

    current_error = evaluate(current, 0)
    for i in range(1, args.steps+1):
        candidate = propose(current, rng)
        candidate_error = evaluate(candidate, i)
        if candidate_error < current_error:
            current, current_error = candidate, candidate_error
    best_prefix = ''.join(best)
    clean_response = exp014.free_generate(model, tokenizer,
        exp014.chat_ids(tokenizer, args.question, device), args.max_new_tokens)
    best_response = exp014.free_generate(model, tokenizer,
        exp014.chat_ids(tokenizer, best_prefix + args.question, device), args.max_new_tokens)
    result = {'timestamp':datetime.now(timezone.utc).isoformat(), 'model':args.model,
              'revision':args.revision, 'torch_version':torch.__version__,
              'transformers_version':transformers.__version__, 'device':device,
              'layer':args.layer, 'neuron':args.neuron, 'dose_target':args.dose,
              'position':args.position, 'question':args.question,
              'continuation':args.continuation, 'clean_activation_values':clean,
              'seed':args.seed, 'steps':args.steps, 'best_prefix':best_prefix,
              'best_error':best_error, 'clean_response':clean_response,
              'best_response':best_response, 'evaluations':records,
              'limitations':['Forward-only local search, not gradient optimization.',
                             'Matched forced history is not free decoding.',
                             'Prefix changes many internal states.',
                             'Failure in small search does not imply unreachability.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
