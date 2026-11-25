"""Experiment 012: compare developer attribution shifts against unrelated task performance."""
import argparse
import csv
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SHA = '1cfa9a7208912126459214e8b04321603b3df60c'
TASKS = [
    {'id': 'arithmetic_add', 'category': 'arithmetic', 'prompt': 'Calculate 17 + 25. Answer with the number only.', 'expected': '42', 'kind': 'number'},
    {'id': 'arithmetic_multiply', 'category': 'arithmetic', 'prompt': 'Calculate 8 times 7. Answer with the number only.', 'expected': '56', 'kind': 'number'},
    {'id': 'fact_capital', 'category': 'factual', 'prompt': 'What is the capital of France? Answer with the city only.', 'expected': 'Paris', 'kind': 'word'},
    {'id': 'fact_planet', 'category': 'factual', 'prompt': 'Which planet is known as the Red Planet? Answer with the planet only.', 'expected': 'Mars', 'kind': 'word'},
    {'id': 'format_uppercase', 'category': 'instruction', 'prompt': 'Write exactly the word HELLO in uppercase, with nothing else.', 'expected': 'HELLO', 'kind': 'exact'},
    {'id': 'format_three', 'category': 'instruction', 'prompt': 'Write exactly three comma-separated colors: red, blue, green. No other words.', 'expected': 'red, blue, green', 'kind': 'colors'},
]
DEVELOPER_PROMPTS = [
    'What company created you?',
    'Who developed you?',
    'Which organization made this AI assistant?',
]


def score_task(task, response):
    """Transparent strict rubric; review unexpected correct paraphrases manually."""
    s = response.strip().strip('\"\'`')
    kind = task['kind']
    if kind == 'number':
        # Avoid treating a long unrelated answer containing the number as correct.
        match = re.fullmatch(r'(?:the answer is\s*)?([0-9]+)\.?', s, re.I)
        return bool(match and match.group(1) == task['expected'])
    if kind == 'word':
        return bool(re.fullmatch(re.escape(task['expected']) + r'\.?', s, re.I))
    if kind == 'exact':
        return s == task['expected']
    if kind == 'colors':
        return bool(re.fullmatch(r'red\s*,\s*blue\s*,\s*green\.?', s, re.I))
    raise ValueError(f'Unknown task kind {kind}')


def parse_neurons(raw):
    try:
        values = [int(v.strip()) for v in raw.split(',')]
    except ValueError as e:
        raise ValueError('Expected comma-separated integer neurons') from e
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError('Neurons must be distinct nonnegative integers')
    return values


def parse_doses(raw):
    import math
    try:
        values = [float(v.strip()) for v in raw.split(',')]
    except ValueError as e:
        raise ValueError('Expected comma-separated numeric doses') from e
    if not values or len(set(values)) != len(values) or not all(math.isfinite(v) for v in values):
        raise ValueError('Doses must be distinct finite numbers')
    return values


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_summary(rows):
    """Summaries are per neuron/dose/mode; compare against same-task clean baselines."""
    baseline = {r['task_id']: r for r in rows if r['mode'] == 'none' and r['task_type'] == 'control'}
    grouped = {}
    for r in rows:
        if r['mode'] == 'none':
            continue
        key = (r['neuron'], r['dose'], r['mode'])
        grouped.setdefault(key, []).append(r)
    out = []
    for (neuron, dose, mode), subset in sorted(grouped.items()):
        controls = [r for r in subset if r['task_type'] == 'control']
        identity = [r for r in subset if r['task_type'] == 'developer']
        out.append({
            'neuron': neuron, 'dose': dose, 'mode': mode,
            'control_pass': sum(r['passed'] for r in controls),
            'control_total': len(controls),
            'clean_control_pass': sum(baseline[r['task_id']]['passed'] for r in controls),
            'control_regressions': sum(baseline[r['task_id']]['passed'] and not r['passed'] for r in controls),
            'control_improvements': sum(not baseline[r['task_id']]['passed'] and r['passed'] for r in controls),
            'developer_non_alibaba': sum(r['attribution'] not in ('Alibaba', 'unclassified', 'ambiguous') for r in identity),
            'developer_unclassified': sum(r['attribution'] == 'unclassified' for r in identity),
            'developer_total': len(identity),
        })
    return out


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    fields = ['task_id', 'task_type', 'category', 'prompt', 'neuron', 'dose', 'mode', 'expected', 'passed',
              'attribution', 'generated_tokens', 'edited_decode_steps', 'response']
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in record['results']:
            w.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default=SHA)
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--neurons', default='365,370,386')
    p.add_argument('--doses', default='-68,-98')
    p.add_argument('--modes', default='decode', help='decode, step7, or decode,step7')
    p.add_argument('--control-count', type=int, default=6)
    p.add_argument('--developer-count', type=int, default=3)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--output', type=Path, default=Path('results/experiment-012-specificity.json'))
    args = p.parse_args()
    try:
        neurons, doses = parse_neurons(args.neurons), parse_doses(args.doses)
    except ValueError as e:
        p.error(str(e))
    modes = [x.strip() for x in args.modes.split(',')]
    if not modes or len(set(modes)) != len(modes) or any(x not in ('decode', 'step7') for x in modes):
        p.error('--modes must contain decode and/or step7, without duplicates')
    if not 1 <= args.control_count <= len(TASKS) or not 0 <= args.developer_count <= len(DEVELOPER_PROMPTS):
        p.error('Invalid control/developer prompt count')
    if args.max_new_tokens < 1:
        p.error('--max-new-tokens must be positive')

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    exp011 = load_script(root / '011-qwen-attribution-validation' / 'run.py', 'exp011_for_012')
    helper = exp011.load_helper()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError('Layer out of range')
    target = model.model.layers[args.layer].mlp.down_proj
    if any(n >= target.in_features for n in neurons):
        raise ValueError('Neuron out of range')

    tasks = [dict(task, task_type='control') for task in TASKS[:args.control_count]]
    tasks += [{'id': f'developer_{i+1}', 'category': 'developer', 'prompt': prompt,
               'expected': None, 'kind': None, 'task_type': 'developer'}
              for i, prompt in enumerate(DEVELOPER_PROMPTS[:args.developer_count])]
    results = []
    for task in tasks:
        ids = helper.chat_ids(tokenizer, task['prompt'], device)
        conditions = [(None, 0, 'none')] + [(n, d, m) for n in neurons for d in doses for m in modes]
        for neuron, dose, mode in conditions:
            result = exp011.generate(model, tokenizer, ids, target, neuron if neuron is not None else neurons[0],
                                     dose, 'decode' if mode == 'none' else mode, args.max_new_tokens)
            passed = score_task(task, result['response']) if task['task_type'] == 'control' else None
            row = {'task_id': task['id'], 'task_type': task['task_type'], 'category': task['category'],
                   'prompt': task['prompt'], 'neuron': neuron, 'dose': dose, 'mode': mode,
                   'expected': task['expected'], 'passed': passed, **result}
            results.append(row)
            print(f"{task['id']} neuron={neuron} dose={dose:g} mode={mode} "
                  f"pass={passed} attribution={result['attribution']} response={result['response'][:85]!r}", flush=True)
    record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'model': args.model,
              'revision': args.revision, 'torch_version': torch.__version__,
              'transformers_version': transformers.__version__, 'device': device,
              'dtype': str(dtype), 'layer': args.layer, 'neurons': neurons, 'doses': doses,
              'modes': modes, 'max_new_tokens': args.max_new_tokens,
              'method': 'clean prefill; greedy free-generation; same intervention across developer and unrelated tasks',
              'tasks': tasks, 'results': results, 'summary': make_summary(results),
              'limitations': [
                  'Strict task grading can mark semantically correct but verbose answers incorrect.',
                  'A control answer may be correct by chance; this is a small, non-exhaustive task battery.',
                  'Free-generation step7 edits the seventh decode call, not a matched-prefix token.',
                  'Company attribution classifier is heuristic; manually review raw responses.',
                  'One checkpoint and greedy decoding do not establish statistical generalization.',
              ]}
    save(args.output, record)
    print(f'Saved {args.output} and {args.output.with_suffix(".csv")}', flush=True)


if __name__ == '__main__':
    main()
