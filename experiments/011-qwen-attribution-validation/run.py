"""Experiment 011: validate alternative developer attributions across prompts and intervention windows."""
import argparse
import csv
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

SHA = '1cfa9a7208912126459214e8b04321603b3df60c'
PROMPTS = ['What company created you?', 'Who developed you?', 'Which organization made this AI assistant?']
COMPANIES = {
    'Alibaba': r'\b(?:alibaba|tongyi|qwen)\b',
    'IBM': r'\bIBM\b', 'Microsoft': r'\bmicrosoft\b',
    'OpenAI': r'\bopenai\b', 'Google': r'\b(?:google|deepmind)\b',
    'Meta': r'\b(?:meta|facebook)\b', 'Anthropic': r'\banthropic\b',
    'Amazon': r'\b(?:amazon|aws)\b', 'Apple': r'\bapple\b',
    'Tencent': r'\btencent\b', 'Huawei': r'\bhuawei\b',
    'Baidu': r'\bbaidu\b', 'DeepSeek': r'\bdeepseek\b',
}
# A mention anywhere in the response is not automatically a developer attribution.
ATTRIBUTION = re.compile(
    r'(?:\b(?:developed|created|built|made|trained|designed)\s+(?:by|at)\b|'
    r'\b(?:developer|creator|maker)\s+(?:is|was)\b|'
    r'\b(?:product|model|assistant)\s+of\b|'
    r'\b(?:member|part)\s+of\s+the\b)', re.I)


def parse_ints(raw):
    try:
        out = [int(x.strip()) for x in raw.split(',')]
    except ValueError as e:
        raise ValueError('Expected comma-separated neuron indices') from e
    if not out or len(out) != len(set(out)) or any(n < 0 for n in out):
        raise ValueError('Neurons must be distinct nonnegative indices')
    return out


def parse_doses(raw):
    try:
        out = [float(x.strip()) for x in raw.split(',')]
    except ValueError as e:
        raise ValueError('Expected comma-separated numeric doses') from e
    if not out or len(out) != len(set(out)) or any(not math.isfinite(x) for x in out):
        raise ValueError('Doses must be distinct finite numbers')
    return out


def classify(response):
    """Conservative heuristic: company in same sentence after attribution phrase.

    Returns 'ambiguous' if multiple companies fit; preserves all raw mentions.
    Requires manual review, especially for negation and nonstandard phrasing.
    """
    mentions = [name for name, pattern in COMPANIES.items() if re.search(pattern, response, re.I)]
    candidates = []
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', response):
        for match in ATTRIBUTION.finditer(sentence):
            tail = sentence[match.end():match.end() + 140]
            # The nearest company after the attribution cue is the strongest candidate.
            found = [(re.search(pattern, tail, re.I).start(), name)
                     for name, pattern in COMPANIES.items() if re.search(pattern, tail, re.I)]
            if found:
                candidates.append(min(found)[1])
    unique = list(dict.fromkeys(candidates))
    label = unique[0] if len(unique) == 1 else ('ambiguous' if unique else 'unclassified')
    return {'attribution': label, 'attribution_candidates': unique, 'company_mentions': mentions,
            'needs_manual_review': True}


def load_helper():
    path = Path(__file__).resolve().parents[1] / '006-qwen-logit-causality' / 'run.py'
    if not path.exists():
        raise FileNotFoundError(f'Experiment 006 helper required: {path}')
    spec = importlib.util.spec_from_file_location('exp006_for_011', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(model, tokenizer, ids, target, neuron, dose, mode, max_new_tokens):
    """Prefill clean; decode calls numbered 1,2,...; step7 edits only call 7."""
    import torch
    state = {'calls': 0, 'edited_calls': []}

    def hook(_module, inputs):
        state['calls'] += 1
        decode_step = state['calls'] - 1  # first call is prefill
        edit = (mode == 'decode' and decode_step >= 1) or (mode == 'step7' and decode_step == 7)
        if not edit or dose == 0:
            return None
        x = inputs[0].clone()
        x[..., neuron] += dose
        state['edited_calls'].append(decode_step)
        return (x,) + inputs[1:]

    handle = target.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
    finally:
        handle.remove()
    response = tokenizer.decode(output[0, ids.shape[-1]:], skip_special_tokens=True)
    return {'response': response, 'generated_tokens': int(output.shape[-1] - ids.shape[-1]),
            'edited_decode_steps': state['edited_calls'], **classify(response)}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['prompt', 'neuron', 'dose', 'mode', 'attribution', 'attribution_candidates',
              'company_mentions', 'edited_decode_steps', 'generated_tokens', 'response']
    with path.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data['results']:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list)
                             else row[k] for k in fields})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default=SHA)
    p.add_argument('--layer', type=int, default=24)
    p.add_argument('--neurons', default='365,370,386')
    p.add_argument('--doses', default='-60,-68,-98')
    p.add_argument('--prompt-count', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--modes', default='step7,decode', help='step7,decode or either')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--output', type=Path, default=Path('results/experiment-011-attribution-validation.json'))
    args = p.parse_args()
    try:
        neurons, doses = parse_ints(args.neurons), parse_doses(args.doses)
    except ValueError as e:
        p.error(str(e))
    modes = [x.strip() for x in args.modes.split(',')]
    if not modes or len(set(modes)) != len(modes) or any(x not in ('step7', 'decode') for x in modes):
        p.error('--modes must contain step7 and/or decode, without duplicates')
    if args.max_new_tokens < 1:
        p.error('--max-new-tokens must be positive')

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    helper = load_helper()
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

    results = []
    for prompt in PROMPTS[:args.prompt_count]:
        ids = helper.chat_ids(tokenizer, prompt, device)
        # One clean control per prompt; all intervened rows use the same prompt formatting.
        clean = generate(model, tokenizer, ids, target, neurons[0], 0, 'decode', args.max_new_tokens)
        results.append({'prompt': prompt, 'neuron': None, 'dose': 0, 'mode': 'none', **clean})
        print(f'BASELINE {prompt!r}: {clean["attribution"]} {clean["response"][:90]!r}', flush=True)
        for neuron in neurons:
            for dose in doses:
                for mode in modes:
                    result = generate(model, tokenizer, ids, target, neuron, dose, mode, args.max_new_tokens)
                    results.append({'prompt': prompt, 'neuron': neuron, 'dose': dose, 'mode': mode, **result})
                    print(f'{prompt[:25]!r} neuron={neuron} dose={dose:g} mode={mode} '
                          f'attribution={result["attribution"]} edited={result["edited_decode_steps"]}', flush=True)
    record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'model': args.model,
              'revision': args.revision, 'torch_version': torch.__version__,
              'transformers_version': transformers.__version__, 'device': device,
              'dtype': str(dtype), 'layer': args.layer, 'neurons': neurons,
              'doses': doses, 'modes': modes, 'prompts': PROMPTS[:args.prompt_count],
              'method': 'greedy free-generation; clean prefill; step7-only vs all decode steps; clean baseline per prompt',
              'results': results,
              'limitations': ['Step7-only edits the seventh decode call, not the seventh generated token.',
                  'Some generations terminate before step7 and thus receive no intervention.',
                  'Regex attribution is heuristic; inspect full response for negation, contradictions and incoherence.',
                  'Step7-only free generation is not matched-prefix teacher forcing; prefixes can differ by condition.',
                  'Single checkpoint, deterministic generation, and a small prompt set limit generality.']}
    save(args.output, record)
    print(f'Saved {args.output} and {args.output.with_suffix(".csv")}', flush=True)


if __name__ == '__main__':
    main()
