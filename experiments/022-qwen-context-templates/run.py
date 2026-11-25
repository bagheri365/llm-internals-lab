"""Experiment 022: balanced text template sensitivity; no activation editing."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = load('exp022_helpers', Path(__file__).with_name('helpers.py'))


def emit(file, row):
    file.write(json.dumps(row, ensure_ascii=False) + '\n')
    file.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default='1cfa9a7208912126459214e8b04321603b3df60c')
    p.add_argument('--question', default=h.QUESTION)
    p.add_argument('--continuation', default='I am an AI assistant developed by')
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--neuron', type=int, default=386)
    p.add_argument('--pilot', action='store_true', help='2 labels, 2 layouts, 3 tokens (19 measurements including baseline)')
    p.add_argument('--output', type=Path, default=Path('results/experiment-022-context-templates.jsonl'))
    args = p.parse_args()
    if args.output.exists():
        p.error(f'Output already exists: {args.output}')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    exp014 = load('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
    exp017 = load('exp017', ROOT/'experiments/017-qwen-local-dose/run.py')
    labels = h.LABELS[:2] if args.pilot else h.LABELS
    layouts = h.LAYOUTS[:2] if args.pilot else h.LAYOUTS
    tokens = h.TOKENS[:3] if args.pilot else h.TOKENS
    plan = h.design(labels, layouts, tokens)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation='eager').to(device).eval()
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        p.error('neuron out of range')
    open_id = exp017.one_token_id(tokenizer, ' Open')
    alibaba_id = exp017.one_token_id(tokenizer, ' Alibaba')
    continuation_ids = tokenizer(args.continuation, add_special_tokens=False,
                                 return_tensors='pt').input_ids.to(device)

    def measure(text):
        chat = exp014.chat_ids(tokenizer, text, device)
        full = exp017.full_ids(torch, chat, continuation_ids)
        with torch.inference_mode():
            result = exp017.measure(model, target, full, args.neuron, 0, open_id, alibaba_id)
        return result, chat[0].tolist()

    baseline, baseline_chat = measure(args.question)
    base_a = baseline['activation_before_edit']
    base_m = baseline['open_minus_alibaba']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        emit(file, {'type': 'metadata', 'timestamp': datetime.now(timezone.utc).isoformat(),
                    'model': args.model, 'revision': args.revision,
                    'torch': torch.__version__, 'transformers': transformers.__version__,
                    'device': device, 'dtype': str(dtype), 'question': args.question,
                    'continuation': args.continuation, 'layer': args.layer,
                    'neuron': args.neuron, 'labels': labels, 'layouts': layouts,
                    'tokens': tokens, 'planned': len(plan), 'baseline': baseline,
                    'baseline_chat_ids': baseline_chat,
                    'note': 'No activation edits. Forced next-token measurement; not free generation.'})
        results = {}
        for index, entry in enumerate(plan, 1):
            prompt = entry['prefix'] + args.question
            result, chat_ids = measure(prompt)
            row = {**entry, 'type': 'candidate', 'activation': result['activation_before_edit'],
                   'activation_delta': result['activation_before_edit'] - base_a,
                   'margin': result['open_minus_alibaba'],
                   'margin_change': result['open_minus_alibaba'] - base_m,
                   'top_token_id': result['top_token_id'],
                   'chat_token_count': len(chat_ids),
                   'audit': h.audit(tokenizer, prompt)}
            emit(file, row)
            results[(entry['label'], entry['layout'], entry['token'])] = row
            if index % 20 == 0:
                print(f'Measured {index}/{len(plan)}', flush=True)
        for label in labels:
            for layout in layouts:
                template = results[(label, layout, '')]
                for token in tokens:
                    combo = results[(label, layout, token)]
                    bare = results[(None, None, token)]
                    emit(file, {'type': 'interaction', 'label': label, 'layout': layout,
                                'token': token,
                                'activation_interaction': h.difference_in_differences(
                                    combo['activation_delta'], template['activation_delta'],
                                    bare['activation_delta']),
                                'margin_interaction': h.difference_in_differences(
                                    combo['margin_change'], template['margin_change'],
                                    bare['margin_change'])})
        best = min((r for r in results.values()), key=lambda r: r['activation_delta'])
        emit(file, {'type': 'summary', 'measured': len(plan), 'best_negative': best,
                    'negative_count': sum(r['activation_delta'] < 0 for r in results.values()),
                    'roundtrip_failures': sum(not r['audit']['user_text_roundtrip']
                                              for r in results.values())})
    print(f'Saved {len(plan)} candidates to {args.output}', flush=True)


if __name__ == '__main__':
    main()
