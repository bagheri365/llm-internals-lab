"""Map matched-position causal dose response using Experiment 023's hook."""
import argparse
import importlib.util
import json
from pathlib import Path

LOCAL = (-4,-2,-1,-.5,0,.5,1,2,4)
FULL = tuple(sorted(set(LOCAL + (-5,-10,-15,-20,-25,-30,-40,-50,-60,-68,-75,-85,-98,-110))))
PILOT = (-98,-68,-30,-4,-1,0,1,4)
PREFIXES = ('', 'Provide a concise definition of mathematics.\n')


def slopes(rows):
    """Secant slopes and nonuniform-grid second differences, using realized doses."""
    ordered = sorted(rows, key=lambda r:r['after'])
    out = []
    for a,b in zip(ordered, ordered[1:]):
        dx = b['after']-a['after']
        if dx <= 0:
            raise ValueError('duplicate realized activation')
        out.append(dict(left=a['after'], right=b['after'], slope=(b['margin']-a['margin'])/dx))
    for a,b in zip(out,out[1:]):
        b['curvature_at_left'] = 2*(b['slope']-a['slope'])/(b['right']-a['left'])
    return out


def previous():
    path = Path(__file__).resolve().parents[1]/'023-qwen-restoration/run.py'
    spec = importlib.util.spec_from_file_location('exp023',path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full',action='store_true')
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    p.add_argument('--output',type=Path,default=Path('results/experiment-024-causal-response.jsonl'))
    args=p.parse_args()
    if args.repeat<1 or args.output.exists(): p.error('positive repeat and fresh output required')
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    old=previous(); base=old.load_previous()
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    dtype=torch.float16 if args.precision=='fp16' else torch.float32
    name='Qwen/Qwen3-4B'; revision=base.exp014.REVISION
    tok=AutoTokenizer.from_pretrained(name,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(name,revision=revision,dtype=dtype,low_cpu_mem_usage=True,attn_implementation='eager').to(device).eval()
    target=model.model.layers[24].mlp.down_proj
    oi,ai=[base.one_token_id(tok,s) for s in (' Open',' Alibaba')]
    question='What company created you?'; continuation='I am an AI assistant developed by'
    suffix=tok(continuation,add_special_tokens=False,return_tensors='pt').input_ids.to(device)
    doses=FULL if args.full else PILOT
    def measure(seq,index,dose):
        seen=[]; h=target.register_forward_pre_hook(old.hook_for(index,seen,delta=dose))
        try:
            with torch.inference_mode():
                raw=model(input_ids=seq,use_cache=False).logits[0,-1]
                logits=raw.float(); probs=logits.softmax(-1); top=int(logits.argmax())
                result=dict(margin=float(logits[oi]-logits[ai]),open_logit=float(logits[oi]),alibaba_logit=float(logits[ai]),p_open=float(probs[oi]),p_alibaba=float(probs[ai]),top_id=top,top_token=tok.decode([top]),logits_dtype=str(raw.dtype))
        finally: h.remove()
        if len(seen)!=1: raise RuntimeError('expected exactly one hook call')
        return dict(**result,before=seen[0][0],after=seen[0][1],realized_delta=seen[0][1]-seen[0][0],dose=dose)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:
        def write(row):
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
        write(dict(type='metadata',model=name,revision=revision,device=device,dtype=str(dtype),torch=torch.__version__,transformers=transformers.__version__,question=question,continuation=continuation,layer=24,open_id=oi,alibaba_id=ai,doses=doses,repeat=args.repeat))
        for rep in range(args.repeat):
            for prefix in PREFIXES:
                seq=base.full_ids(torch,base.exp014.chat_ids(tok,prefix+question,device),suffix)
                for index in (385,386,387):
                    context=dict(repeat=rep,prefix=prefix,coordinate=index)
                    zero=measure(seq,index,0)
                    rows=[]
                    for dose in doses:
                        row=dict(zero) if dose==0 else measure(seq,index,dose)
                        row['margin_change']=row['margin']-zero['margin']; rows.append(row)
                        write(dict(type='measurement',**context,input_ids=seq[0].tolist(),**row))
                    write(dict(type='sensitivity',**context,intervals=slopes(rows)))
                    print(context,'baseline',zero['margin'],'minus98',next(r['margin'] for r in rows if r['dose']==-98),flush=True)
        write(dict(type='complete',measurements=args.repeat*len(PREFIXES)*3*len(doses)))

if __name__=='__main__': main()
