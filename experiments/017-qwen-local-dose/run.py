"""Experiment 017: matched-history local dose response and natural-prefix comparison."""
import argparse
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
exp014 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp014)
spec16 = importlib.util.spec_from_file_location('exp016', ROOT/'experiments/016-qwen-reachability/run.py')
exp016 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(exp016)


def parse_doses(value):
    values = [float(s.strip()) for s in value.split(',')]
    if not values or any(not math.isfinite(x) or x > 0 for x in values):
        raise ValueError('doses must be finite and nonpositive')
    return list(dict.fromkeys([0.0] + values))


def one_token_id(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f'{text!r} tokenizes to {len(ids)} tokens, expected one')
    return ids[0]


def last_position_hook(neuron, dose, observation):
    """Read unedited input and edit ONLY the final sequence position."""
    def hook(_module, inputs):
        x = inputs[0]
        observation.append(float(x[0, -1, neuron].detach().float().cpu()))
        if dose == 0:
            return None
        y = x.clone()
        y[0, -1, neuron] += dose
        return (y, *inputs[1:])
    return hook


def measure(model, target, input_ids, neuron, dose, open_id, alibaba_id):
    import torch
    observed = []
    handle = target.register_forward_pre_hook(last_position_hook(neuron, dose, observed))
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
            selected = torch.stack([logits[open_id], logits[alibaba_id]])
            probs = torch.softmax(logits, dim=-1)
            top = int(torch.argmax(logits))
            result = {'dose': dose, 'activation_before_edit': observed[0],
                      'activation_after_edit': observed[0] + dose,
                      'open_logit': float(selected[0]), 'alibaba_logit': float(selected[1]),
                      'open_minus_alibaba': float(selected[0] - selected[1]),
                      'p_open': float(probs[open_id]), 'p_alibaba': float(probs[alibaba_id]),
                      'top_token_id': top}
    finally:
        handle.remove()
    if len(observed) != 1:
        raise RuntimeError(f'Expected one pre-hook observation, got {len(observed)}')
    return result


def full_ids(torch, prompt_ids, continuation_ids):
    if continuation_ids.shape[1] < 1:
        raise ValueError('continuation cannot be empty')
    return torch.cat([prompt_ids, continuation_ids], dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--revision', default=exp014.REVISION)
    ap.add_argument('--layer', type=int, default=24)
    ap.add_argument('--neuron', type=int, default=386)
    ap.add_argument('--question', default=exp014.QUESTION)
    ap.add_argument('--continuation', default='I am an AI assistant developed by')
    ap.add_argument('--open-token', default=' Open')
    ap.add_argument('--alibaba-token', default=' Alibaba')
    ap.add_argument('--doses', default='0,-5,-15,-30,-50,-60,-68,-75,-85,-98')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--free-doses', default='0,-68,-98')
    ap.add_argument('--max-new-tokens', type=int, default=64)
    ap.add_argument('--output', type=Path, default=Path('results/experiment-017-local-dose.jsonl'))
    args = ap.parse_args()
    try:
        doses = parse_doses(args.doses)
        free_doses = parse_doses(args.free_doses)
    except ValueError as exc:
        ap.error(str(exc))
    if args.contexts < 1 or args.max_new_tokens < 1:
        ap.error('contexts and max-new-tokens must be positive')
    if args.output.exists():
        ap.error(f'output already exists: {args.output}; choose a fresh path')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    open_id = one_token_id(tokenizer, args.open_token)
    alibaba_id = one_token_id(tokenizer, args.alibaba_token)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device).eval()
    if not 0 <= args.layer < len(model.model.layers):
        ap.error('layer out of range')
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        ap.error('neuron out of range')
    continuation_ids = tokenizer(args.continuation, add_special_tokens=False,
                                 return_tensors='pt').input_ids.to(device)
    clean_prompt = exp014.chat_ids(tokenizer, args.question, device)
    clean_full = full_ids(torch, clean_prompt, continuation_ids)
    exp016.append_jsonl(args.output, {'type':'metadata','timestamp':datetime.now(timezone.utc).isoformat(),
        'model':args.model,'revision':args.revision,'torch':torch.__version__,
        'transformers':transformers.__version__,'device':device,'dtype':str(dtype),
        'layer':args.layer,'neuron':args.neuron,'question':args.question,
        'continuation':args.continuation,'open_token':args.open_token,'open_id':open_id,
        'alibaba_token':args.alibaba_token,'alibaba_id':alibaba_id,'doses':doses,
        'contexts':args.contexts,'seed':args.seed,
        'measurement':'Last position of prompt plus complete forced continuation; predicts next token.',
        'limitations':['Only last-position intervention, not a full decode-only trajectory.',
          'Matched continuation is teacher-forced and may be atypical under a new prefix.',
          'Prefixes alter the whole computation, not just coordinate 386.',
          'The two-token margin is not the same as the full-vocabulary top-token choice.']})
    clean = measure(model, target, clean_full, args.neuron, 0, open_id, alibaba_id)
    exp016.append_jsonl(args.output, {'type':'baseline', **clean,
                                     'top_token':tokenizer.decode([clean['top_token_id']])})
    for dose in doses:
        if dose == 0:
            result = clean.copy()
        else:
            result = measure(model, target, clean_full, args.neuron, dose, open_id, alibaba_id)
        result['top_token'] = tokenizer.decode([result['top_token_id']])
        result['margin_change_vs_clean'] = result['open_minus_alibaba']-clean['open_minus_alibaba']
        exp016.append_jsonl(args.output, {'type':'dose', **result})
        print(f'dose={dose:g} margin={result["open_minus_alibaba"]:.3f} top={result["top_token"]!r}',flush=True)
    for i,prefix in enumerate(exp016.candidates(args.contexts,args.seed)):
        ids = exp014.chat_ids(tokenizer,prefix+args.question,device)
        result = measure(model,target,full_ids(torch,ids,continuation_ids),args.neuron,0,open_id,alibaba_id)
        result['top_token'] = tokenizer.decode([result['top_token_id']])
        result['activation_delta'] = result['activation_before_edit']-clean['activation_before_edit']
        result['margin_change_vs_clean'] = result['open_minus_alibaba']-clean['open_minus_alibaba']
        exp016.append_jsonl(args.output,{'type':'context','index':i,'prefix':prefix,**result})
        print(f'context={i+1}/{args.contexts} delta={result["activation_delta"]:.3f} margin_change={result["margin_change_vs_clean"]:.3f}',flush=True)
    for dose in free_doses:
        response = exp016.generate(model,target,tokenizer,clean_prompt,args.neuron,dose,args.max_new_tokens)
        exp016.append_jsonl(args.output,{'type':'free_generation','dose':dose,'response':response})
        print(f'free dose={dose:g}: {response[:100]!r}',flush=True)
    print(f'Saved {args.output}',flush=True)

if __name__ == '__main__':
    main()
