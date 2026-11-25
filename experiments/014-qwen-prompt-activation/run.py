"""Experiment 014: search text prefixes for a natural -98 activation displacement.

No activation editing is performed. Hooks only observe down_proj input.
"""
import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
QUESTION = 'What company created you?'
# Pre-registered, answer-neutral contexts. No explicit instruction to claim a company.
PREFIXES = [
    '',
    'Please answer the following question accurately.\n',
    'This is a question about the origin of an AI assistant.\n',
    'Consider the distinction between the developer of a model and its users.\n',
    'The topic is the history of artificial intelligence research.\n',
    'We are discussing how AI systems are developed and maintained.\n',
    'For an encyclopedia entry, provide a factual answer.\n',
    'The following question concerns a language model.\n',
    'In a technical interview, explain the provenance of an AI system.\n',
    'Think about research laboratories, model families, and their origins.\n',
    'The subject is organizations that publish language models.\n',
    'Answer the final question in one clear sentence.\n',
]


def choose_candidates(prefixes, budget, seed):
    """Deterministic candidate selection; baseline always first."""
    if budget < 1:
        raise ValueError('budget must be positive')
    pool = list(dict.fromkeys(prefixes))
    if not pool or pool[0] != '':
        pool.insert(0, '')
    rest = pool[1:]
    random.Random(seed).shuffle(rest)
    return [''] + rest[:budget - 1]


def summarize_deltas(clean, candidate):
    if len(clean) != len(candidate) or not clean:
        raise ValueError('matched activation vectors must have equal nonzero length')
    deltas = [float(b-a) for a,b in zip(clean,candidate)]
    return {'deltas': deltas, 'mean_delta': sum(deltas)/len(deltas),
            'max_abs_error_to_minus_98': max(abs(d+98) for d in deltas),
            'mean_abs_error_to_minus_98': sum(abs(d+98) for d in deltas)/len(deltas)}


def chat_ids(tokenizer, prompt, device):
    encoded = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                                            tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors='pt')
    # Transformers versions may return either a Tensor or a BatchEncoding.
    ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    return ids.to(device)


def matched_activations(model, target, prompt_ids, continuation_ids, neuron):
    """Teacher-force identical continuation; measure down_proj input at its token positions.

    The first continuation token is predicted from the final prompt position; then each
    subsequent token is predicted from the previous continuation token. This yields
    one activation per next-token decision under the same forced output history.
    """
    import torch
    if continuation_ids.numel() < 1:
        raise ValueError('continuation must have at least one token')
    n = int(prompt_ids.shape[1])
    full = torch.cat([prompt_ids, continuation_ids[:, :-1]], dim=1)
    observed = []
    def observe(_module, inputs):
        x = inputs[0]
        observed.extend(x[0, n-1:n-1+continuation_ids.shape[1], neuron].detach().float().cpu().tolist())
    handle = target.register_forward_pre_hook(observe)
    try:
        with torch.inference_mode():
            model(input_ids=full, use_cache=False)
    finally:
        handle.remove()
    if len(observed) != continuation_ids.shape[1]:
        raise RuntimeError(f'Expected {continuation_ids.shape[1]} values, got {len(observed)}')
    return observed


def free_generate(model, tokenizer, ids, max_new_tokens):
    import torch
    with torch.inference_mode():
        output = model.generate(input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                                use_cache=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--revision', default=REVISION)
    ap.add_argument('--layer', type=int, default=24)
    ap.add_argument('--neuron', type=int, default=386)
    ap.add_argument('--dose', type=float, default=-98)
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--continuation', default='I am an AI assistant developed by')
    ap.add_argument('--budget', type=int, default=12)
    ap.add_argument('--seed', type=int, default=14)
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--output', type=Path, default=Path('results/experiment-014-prompt-activation.json'))
    args = ap.parse_args()
    if args.budget < 1 or args.top_k < 0 or args.max_new_tokens < 1 or not math.isfinite(args.dose):
        ap.error('budget and max-new-tokens must be positive; top-k nonnegative; dose finite')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
             dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device)
    model.eval()
    if not 0 <= args.layer < len(model.model.layers):
        ap.error('layer out of range')
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        ap.error('neuron out of range')
    continuation_ids = tokenizer(args.continuation, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    candidates = choose_candidates(PREFIXES, args.budget, args.seed)
    clean_ids = chat_ids(tokenizer, args.question, device)
    clean = matched_activations(model, target, clean_ids, continuation_ids, args.neuron)
    rows = []
    for prefix in candidates:
        ids = chat_ids(tokenizer, prefix + args.question, device)
        activations = matched_activations(model, target, ids, continuation_ids, args.neuron)
        metrics = summarize_deltas(clean, activations)
        metrics['mean_abs_error_to_target'] = sum(abs(d-args.dose) for d in metrics['deltas'])/len(metrics['deltas'])
        row = {'prefix':prefix, 'activation_values':activations, **metrics}
        rows.append(row)
        print(f"prefix={prefix[:48]!r} mean_delta={metrics['mean_delta']:.3f} error={metrics['mean_abs_error_to_target']:.3f}", flush=True)
    ranked = sorted(range(len(rows)), key=lambda i:(rows[i]['mean_abs_error_to_target'],i))
    # Run free generation only after activation ranking, without an editing hook.
    to_generate = list(dict.fromkeys([0]+ranked[:args.top_k]))
    for i in to_generate:
        rows[i]['free_response'] = free_generate(model, tokenizer,
            chat_ids(tokenizer, rows[i]['prefix'] + args.question, device), args.max_new_tokens)
        print(f"free prefix={rows[i]['prefix'][:40]!r}: {rows[i]['free_response'][:160]!r}", flush=True)
    record = {'timestamp':datetime.now(timezone.utc).isoformat(), 'model':args.model,
       'revision':args.revision,'torch_version':torch.__version__,
       'transformers_version':transformers.__version__, 'device':device,'dtype':str(dtype),
       'layer':args.layer,'neuron':args.neuron,'dose_target':args.dose,
       'question':args.question,'forced_continuation':args.continuation,
       'forced_token_ids':continuation_ids[0].tolist(),'clean_activation_values':clean,
       'budget':len(rows),'seed':args.seed,'ranked_indices':ranked,'results':rows,
       'limitations':['Prompt changes all internal states, not just one coordinate.',
          'A fixed continuation matches next-token decisions, not free-generation histories.',
          'A prompt prefix cannot guarantee a constant shift on every decode step.',
          'Candidate search is small and answer-neutral; no reachability claim from failure.',
          'A changed self-report is not a change in the model actual developer.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with args.output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['prefix','mean_delta','mean_abs_error_to_target','max_abs_error_to_minus_98','free_response'])
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in w.fieldnames})
    print(f'Saved {args.output} and {args.output.with_suffix(".csv")}',flush=True)

if __name__=='__main__':
    main()
