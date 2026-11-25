"""Experiment 018: forward-only token-level search for activation vs logit-margin matching."""
import argparse
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp014 = load('exp014', ROOT/'experiments/014-qwen-prompt-activation/run.py')
exp016 = load('exp016', ROOT/'experiments/016-qwen-reachability/run.py')
exp017 = load('exp017', ROOT/'experiments/017-qwen-local-dose/run.py')
search = load('exp018_search', Path(__file__).with_name('search.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--revision', default=exp014.REVISION)
    parser.add_argument('--question', default=exp014.QUESTION)
    parser.add_argument('--continuation', default='I am an AI assistant developed by')
    parser.add_argument('--layer', type=int, default=24)
    parser.add_argument('--neuron', type=int, default=386)
    parser.add_argument('--slots', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--proposals', type=int, default=12)
    parser.add_argument('--pool-size', type=int, default=96)
    parser.add_argument('--seed', type=int, default=18)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('results/experiment-018-dual-search.jsonl'))
    args = parser.parse_args()
    try:
        search.validate_config(args.slots,args.rounds,args.proposals,args.pool_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_new_tokens < 1:
        parser.error('max-new-tokens must be positive')
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')

    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    open_id = exp017.one_token_id(tokenizer, ' Open')
    alibaba_id = exp017.one_token_id(tokenizer, ' Alibaba')
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device).eval()
    if not 0 <= args.layer < len(model.model.layers):
        parser.error('layer out of range')
    target = model.model.layers[args.layer].mlp.down_proj
    if not 0 <= args.neuron < target.in_features:
        parser.error('neuron out of range')
    continuation_ids = tokenizer(args.continuation, add_special_tokens=False,
                                 return_tensors='pt').input_ids.to(device)
    clean_prompt = exp014.chat_ids(tokenizer,args.question,device)
    clean_full = exp017.full_ids(torch,clean_prompt,continuation_ids)
    baseline = exp017.measure(model,target,clean_full,args.neuron,0,open_id,alibaba_id)
    hook_target = exp017.measure(model,target,clean_full,args.neuron,-98,open_id,alibaba_id)
    vocab = search.unique_tokens(tokenizer,search.SEED_TEXTS,args.pool_size)
    exp016.append_jsonl(args.output, {'type':'metadata','timestamp':datetime.now(timezone.utc).isoformat(),
        'model':args.model,'revision':args.revision,'torch':torch.__version__,
        'transformers':transformers.__version__,'device':device,'dtype':str(dtype),
        'question':args.question,'continuation':args.continuation,
        'layer':args.layer,'neuron':args.neuron,'slots':args.slots,'rounds':args.rounds,
        'proposals':args.proposals,'pool_size':len(vocab),'seed':args.seed,
        'baseline':baseline,'hook_minus_98':hook_target,
        'activation_delta_target':-98.0,'margin_target':hook_target['open_minus_alibaba'],
        'method':'forward-only single-token replacement, fixed forced continuation',
        'limitations':['Prefix is decoded to text and retokenized in the chat template; actual token IDs may differ.',
          'The measured position uses teacher forcing, not free decoding.',
          'Only a restricted token pool is searched; negative results do not prove unreachability.',
          'A changed margin does not establish causal mediation through coordinate 386.']})

    cache = {}
    def evaluate(tokens):
        # Search in token-ID space but measure the exact text that will be injected.
        prefix = tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if prefix not in cache:
            ids = exp014.chat_ids(tokenizer,prefix+args.question,device)
            full = exp017.full_ids(torch,ids,continuation_ids)
            measurement = exp017.measure(model,target,full,args.neuron,0,open_id,alibaba_id)
            measurement['activation_delta'] = measurement['activation_before_edit'] - baseline['activation_before_edit']
            measurement['margin_change'] = measurement['open_minus_alibaba'] - baseline['open_minus_alibaba']
            measurement['top_token'] = tokenizer.decode([measurement['top_token_id']])
            cache[prefix] = measurement
        return prefix,cache[prefix]

    winners = {}
    for objective in ('activation','margin'):
        rng = random.Random(args.seed + (0 if objective == 'activation' else 10000))
        current = [vocab[rng.randrange(len(vocab))] for _ in range(args.slots)]
        prefix, measurement = evaluate(current)
        best = search.score(measurement,baseline,objective,-98.0,hook_target['open_minus_alibaba'])
        exp016.append_jsonl(args.output,{'type':'search','objective':objective,'round':0,
            'prefix':prefix,'token_ids':current,'score':best,**measurement})
        for iteration in range(1,args.rounds+1):
            winner = (best,current,prefix,measurement)
            for proposal in search.candidates(current,vocab,rng,args.proposals):
                candidate_prefix,candidate_measurement = evaluate(proposal)
                candidate_score = search.score(candidate_measurement,baseline,objective,-98.0,
                                               hook_target['open_minus_alibaba'])
                if candidate_score < winner[0]:
                    winner = (candidate_score,proposal,candidate_prefix,candidate_measurement)
            best,current,prefix,measurement = winner
            exp016.append_jsonl(args.output,{'type':'search','objective':objective,'round':iteration,
                'prefix':prefix,'token_ids':current,'score':best,**measurement})
            print(f'{objective} round={iteration}/{args.rounds} score={best:.4f} '
                  f'delta={measurement["activation_delta"]:.4f} '
                  f'margin={measurement["open_minus_alibaba"]:.4f}',flush=True)
        winners[objective] = (prefix,measurement)

    for objective,(prefix,measurement) in winners.items():
        prompt_ids = exp014.chat_ids(tokenizer,prefix+args.question,device)
        # This function's zero-dose path installs no activation-changing edit.
        response = exp016.generate(model,target,tokenizer,prompt_ids,args.neuron,0,args.max_new_tokens)
        exp016.append_jsonl(args.output,{'type':'free_generation','objective':objective,
            'prefix':prefix,'response':response,'measurement':measurement})
        print(f'{objective} prefix={prefix!r} response={response[:120]!r}',flush=True)
    print(f'Saved {args.output}; unique prefixes evaluated: {len(cache)}',flush=True)


if __name__ == '__main__':
    main()
