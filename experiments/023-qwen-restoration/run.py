"""Matched-history scalar restoration and transplantation at layer 24."""
import argparse
import importlib.util
import json
from pathlib import Path


def load_previous():
    path = Path(__file__).resolve().parents[2] / 'experiments/017-qwen-local-dose/run.py'
    spec = importlib.util.spec_from_file_location('exp017', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hook_for(index, seen, clamp=None, delta=0):
    def hook(module, inputs):
        x = inputs[0]
        before = float(x[0, -1, index])
        if clamp is None and delta == 0:
            seen.append((before, before))
            return None
        y = x.clone()
        y[0, -1, index] = before + delta if clamp is None else clamp
        seen.append((before, float(y[0, -1, index])))
        return (y, *inputs[1:])
    return hook


def decompose(a, c, d, e):
    return dict(total=c-a, target_in_modified=c-d, residual_at_clean_activation=d-a,
                target_in_clean=e-a, interaction=(c-d)-(e-a),
                residual_at_modified_activation=c-e)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/experiment-023-restoration.jsonl'))
    p.add_argument('--full', action='store_true')
    p.add_argument('--repeat', type=int, default=2)
    args = p.parse_args()
    if args.repeat < 1 or args.output.exists():
        p.error('positive repeat and fresh output required')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    old = load_previous()
    model_name = 'Qwen/Qwen3-4B'
    revision = old.exp014.REVISION
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device == 'mps' else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager').to(device).eval()
    target = model.model.layers[24].mlp.down_proj
    oi, ai = [old.one_token_id(tok, s) for s in (' Open', ' Alibaba')]
    question = 'What company created you?'
    continuation = 'I am an AI assistant developed by'
    suffix = tok(continuation, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    def ids(prefix):
        return old.full_ids(torch, old.exp014.chat_ids(tok, prefix+question, device), suffix)
    def measure(seq, index=386, clamp=None, delta=0):
        seen = []
        h = target.register_forward_pre_hook(hook_for(index, seen, clamp, delta))
        try:
            with torch.inference_mode():
                logits = model(input_ids=seq, use_cache=False).logits[0,-1].float()
                probs = logits.softmax(-1)
                top = int(logits.argmax())
                result = dict(margin=float(logits[oi]-logits[ai]), p_open=float(probs[oi]), p_alibaba=float(probs[ai]), top_token=tok.decode([top]))
        finally:
            h.remove()
        if len(seen) != 1:
            raise RuntimeError('expected exactly one hook call')
        return dict(**result, coordinate=index, before=seen[0][0], after=seen[0][1], clamp=clamp, delta=delta)
    prefixes = ['Provide a concise definition of mathematics.\n']
    if args.full:
        prefixes += ['In Chinese, discuss physics.\n', 'In Chinese, discuss software engineering.\n', 'Topic:\n月\n', 'The what where Open\n']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        def write(row):
            f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush()
        write(dict(type='metadata', model=model_name, revision=revision, device=device, dtype=str(dtype), torch=torch.__version__, transformers=transformers.__version__, question=question, continuation=continuation, layer=24, coordinate=386, open_id=oi, alibaba_id=ai, repeat=args.repeat))
        clean = ids('')
        for rep in range(args.repeat):
            a = measure(clean); b = measure(clean, delta=-98)
            for prefix in prefixes:
                modified = ids(prefix); c = measure(modified)
                d = measure(modified, clamp=a['before']); e = measure(clean, clamp=c['before'])
                rows = dict(A=a,B=b,C=c,D=d,E=e)
                for name, row in rows.items():
                    write(dict(type='condition', repeat=rep, prefix=prefix, condition=name, input_ids=(clean if name in 'ABE' else modified)[0].tolist(), **row))
                for index in (385,387):
                    control = measure(modified,index=index,delta=d['after']-d['before'])
                    write(dict(type='control', repeat=rep,prefix=prefix, **control))
                effects = decompose(*(rows[k]['margin'] for k in 'ACDE'))
                write(dict(type='decomposition',repeat=rep,prefix=prefix, **effects))
                print(rep,repr(prefix),effects,flush=True)

if __name__ == '__main__':
    main()
