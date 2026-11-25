"""Experiment 020: exploratory composition and placement of script-diverse prefixes."""
import argparse
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

h = load('exp020_helpers', Path(__file__).with_name('helpers.py'))
exp014 = load('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
exp017 = load('exp017', ROOT/'experiments/017-qwen-local-dose/run.py')


def emit(file, record):
    file.write(json.dumps(record, ensure_ascii=False) + '\n')
    file.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default=exp014.REVISION)
    p.add_argument('--question', default=exp014.QUESTION)
    p.add_argument('--continuation', default='I am an AI assistant developed by')
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--neuron', type=int, default=386)
    p.add_argument('--pairs', type=int, default=8)
    p.add_argument('--triples', type=int, default=4)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--strict-audit', action='store_true', help='Skip prefixes if isolated token IDs do not appear in user-message encoding')
    p.add_argument('--output', type=Path, default=Path('results/experiment-020-script-composition.jsonl'))
    args = p.parse_args()
    if args.output.exists():
        p.error(f'Output exists: {args.output}')
    if args.pairs < 0 or args.triples < 0:
        p.error('pair and triple budgets must be nonnegative')

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
    continuation = tokenizer(args.continuation, add_special_tokens=False, return_tensors='pt').input_ids.to(device)

    def measure(question):
        prompt = exp014.chat_ids(tokenizer, question, device)
        full = exp017.full_ids(torch, prompt, continuation)
        return exp017.measure(model, target, full, args.neuron, 0, open_id, alibaba_id)

    baseline = measure(args.question)
    candidates = h.make_candidates(max_pairs=args.pairs, max_triples=args.triples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = None
    with args.output.open('x', encoding='utf-8') as f:
        emit(f, {'type':'metadata', 'timestamp':datetime.now(timezone.utc).isoformat(),
            'model':args.model, 'revision':args.revision, 'torch':torch.__version__,
            'transformers':transformers.__version__, 'device':device, 'dtype':str(dtype),
            'question':args.question, 'continuation':args.continuation,
            'layer':args.layer, 'neuron':args.neuron, 'seed':args.seed,
            'pairs':args.pairs, 'triples':args.triples, 'strict_audit':args.strict_audit,
            'placements':['adjacent','context','distant'], 'baseline':baseline,
            'reference_dose':-98, 'reference_margin':5.078125,
            'note':'No activation edits. Fixed forced continuation; candidate selection is exploratory.'})
        n = 0
        for kind, tokens in candidates:
            for placement in ('adjacent','context','distant'):
                prefix = h.make_prefix(tokens, placement)
                user_text = prefix + args.question
                audit = h.exact_token_audit(tokenizer, tokens, user_text)
                audit_ok = all(x['roundtrip'] and x['present_in_user_ids'] for x in audit)
                if args.strict_audit and not audit_ok:
                    emit(f, {'type':'skipped', 'kind':kind, 'tokens':tokens, 'placement':placement,
                             'reason':'token_audit', 'audit':audit})
                    continue
                result = measure(user_text)
                row = {'type':'candidate', 'kind':kind, 'tokens':tokens, 'placement':placement,
                    'prefix':prefix, 'audit_ok':audit_ok, 'audit':audit,
                    'activation':result['activation_before_edit'],
                    'activation_delta':result['activation_before_edit']-baseline['activation_before_edit'],
                    'margin':result['open_minus_alibaba'],
                    'margin_change':result['open_minus_alibaba']-baseline['open_minus_alibaba'],
                    'top_token_id':result['top_token_id']}
                emit(f,row)
                n += 1
                if best is None or row['activation_delta'] < best['activation_delta']:
                    best = row
                if n % 20 == 0:
                    print(f'Measured {n} candidates; best shift {best["activation_delta"]:.4f}',flush=True)
        emit(f, {'type':'summary', 'measured':n, 'best_negative':best})
    print(f'Saved {n} measurements to {args.output}', flush=True)

if __name__ == '__main__':
    main()
