"""Experiment 021: balanced, text-first composition and placement; no activation edits."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

h = load('exp021_helpers', Path(__file__).with_name('helpers.py'))
exp014 = load('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
exp017 = load('exp017', ROOT/'experiments/017-qwen-local-dose/run.py')

def emit(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    f.flush()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default=exp014.REVISION)
    p.add_argument('--question', default=exp014.QUESTION)
    p.add_argument('--continuation', default='I am an AI assistant developed by')
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--neuron', type=int, default=386)
    p.add_argument('--pairs', type=int, default=12)
    p.add_argument('--triples', type=int, default=8)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--output', type=Path, default=Path('results/experiment-021-text-composition.jsonl'))
    args = p.parse_args()
    if args.output.exists():
        p.error(f'Output already exists: {args.output}')
    if args.pairs < 0 or args.triples < 0:
        p.error('budgets must be nonnegative')

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    continuation_ids = tokenizer(args.continuation, add_special_tokens=False,
                                 return_tensors='pt').input_ids.to(device)

    def measure(user_text):
        chat = exp014.chat_ids(tokenizer, user_text, device)
        full = exp017.full_ids(torch, chat, continuation_ids)
        with torch.inference_mode():
            result = exp017.measure(model, target, full, args.neuron, 0, open_id, alibaba_id)
        return result, chat[0].tolist()

    baseline, baseline_chat_ids = measure(args.question)
    design = h.design(args.pairs, args.triples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open('x', encoding='utf-8') as f:
        emit(f, {'type':'metadata', 'timestamp':datetime.now(timezone.utc).isoformat(),
            'model':args.model, 'revision':args.revision, 'torch':torch.__version__,
            'transformers':transformers.__version__, 'device':device, 'dtype':str(dtype),
            'question':args.question, 'continuation':args.continuation,
            'layer':args.layer, 'neuron':args.neuron, 'seed':args.seed,
            'pairs':args.pairs, 'triples':args.triples, 'attempted':len(design),
            'baseline':baseline, 'baseline_chat_token_count':len(baseline_chat_ids),
            'reference_dose':-98, 'reference_margin':5.078125,
            'note':'No activation editing. Every valid text prefix measured regardless of isolated token-ID preservation. Teacher-forced continuation only.'})
        for i, (kind, parts, placement, prefix) in enumerate(design, 1):
            result, chat_ids = measure(prefix + args.question)
            check = h.audit(tokenizer, parts, prefix + args.question, chat_ids)
            row = {'type':'candidate', 'kind':kind, 'parts':list(parts),
                'placement':placement, 'prefix':prefix,
                'activation':result['activation_before_edit'],
                'activation_delta':result['activation_before_edit']-baseline['activation_before_edit'],
                'margin':result['open_minus_alibaba'],
                'margin_change':result['open_minus_alibaba']-baseline['open_minus_alibaba'],
                'top_token_id':result['top_token_id'], 'audit':check}
            emit(f, row)
            rows.append(row)
            if i % 20 == 0:
                print(f'Measured {i}/{len(design)}; best shift {min(r["activation_delta"] for r in rows):.4f}', flush=True)
        singles = {(r['parts'][0], r['placement']):r for r in rows if len(r['parts']) == 1}
        for r in rows:
            if len(r['parts']) < 2:
                continue
            component = [singles[(part, r['placement'])] for part in r['parts']]
            emit(f, {'type':'interaction', 'parts':r['parts'], 'placement':r['placement'],
                'activation_departure_from_additive':h.interaction(r['activation_delta'],
                    [x['activation_delta'] for x in component]),
                'margin_departure_from_additive':h.interaction(r['margin_change'],
                    [x['margin_change'] for x in component])})
        best = min(rows, key=lambda r:r['activation_delta'])
        emit(f, {'type':'summary', 'measured':len(rows), 'skipped':0,
                 'best_negative':best, 'negative_count':sum(r['activation_delta'] < 0 for r in rows),
                 'component_ids_not_preserved':sum(not r['audit']['all_component_ids_preserved'] for r in rows)})
    print(f'Saved {len(rows)} measurements to {args.output}', flush=True)

if __name__ == '__main__':
    main()
