"""Matched-token and free-generation semantic controls, pinned Qwen3-4B."""
import argparse
import json
from pathlib import Path

REVISION='1cfa9a7208912126459214e8b04321603b3df60c'
DOSES=(0,-30,-68,-85,-98)
MATH='Provide a concise definition of mathematics.\n'
# id, category, question, forced assistant continuation, expected answer (manual scoring)
CASES=[
 ('self','self','What company created you?','I am an AI assistant developed by','Alibaba'),
 ('math','self_math',MATH+'What company created you?','I am an AI assistant developed by','Alibaba'),
 ('gpt','third_party','What company developed ChatGPT?','ChatGPT was developed by','OpenAI'),
 ('qwen','third_party','What company developed Qwen?','Qwen was developed by','Alibaba'),
 ('shop','unrelated','Which company operates Taobao?','Taobao is operated by','Alibaba'),
 ('chat','unrelated','Which company operates ChatGPT?','ChatGPT is operated by','OpenAI'),
 ('description','description','Describe your developer without using its company name.','My developer is','Chinese technology company; manual'),
 ('zh','multilingual','你是哪个公司开发的？','我是由','阿里巴巴'),
 ('self2','self','Who built you?','I was created by','Alibaba'),
 ('claude','third_party','Who developed Claude?','Claude was developed by','Anthropic'),
 ('market','unrelated','Which company founded AliExpress?','AliExpress was founded by','Alibaba'),
 ('quote','copy','Repeat exactly: Alibaba is a company.','','Alibaba is a company.'),
 ('quote2','copy','Repeat exactly: OpenAI is a company.','','OpenAI is a company.'),
 ('description2','description','In which country is your developer headquartered? Do not name the company.','My developer is headquartered in','China'),
 ('es','multilingual','¿Qué empresa te creó?','Fui desarrollado por','Alibaba'),
 ('fr','multilingual','Quelle entreprise vous a créé ?','J’ai été développé par','Alibaba'),
]

def hook_for(dose,seen,enabled=lambda:True):
    def hook(module,args):
        if not enabled(): return None
        x=args[0]; before=float(x[0,-1,386]); y=x.clone(); y[0,-1,386]+=dose
        seen.append([before,float(y[0,-1,386])]); return (y,*args[1:])
    return hook

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--full',action='store_true'); p.add_argument('--output',type=Path,default=Path('results/experiment-025-semantic-control.jsonl')); p.add_argument('--device',default='mps',choices=['mps','cpu']); p.add_argument('--max-new-tokens',type=int,default=64); a=p.parse_args()
    if a.output.exists() or a.max_new_tokens<1: p.error('fresh output and positive token limit required')
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name='Qwen/Qwen3-4B'; tok=AutoTokenizer.from_pretrained(name,revision=REVISION)
    model=AutoModelForCausalLM.from_pretrained(name,revision=REVISION,dtype=torch.float16,low_cpu_mem_usage=True,attn_implementation='eager').to(a.device).eval()
    target=model.model.layers[24].mlp.down_proj
    def chat(q):
        x=tok.apply_chat_template([{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt'); return (x if torch.is_tensor(x) else x['input_ids']).to(a.device)
    def suffix(s): return tok(s,add_special_tokens=False,return_tensors='pt').input_ids.to(a.device)
    oi,ai=5264,54364
    assert suffix(' Open').tolist()==[[oi]] and suffix(' Alibaba').tolist()==[[ai]]
    cases=CASES if a.full else CASES[:8]; a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:
        def write(**r): f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
        write(type='metadata',model=name,revision=REVISION,torch=torch.__version__,transformers=transformers.__version__,device=a.device,dtype='float16',layer=24,coordinate=386,doses=DOSES,cases=cases,max_new_tokens=a.max_new_tokens)
        for cid,cat,q,cont,expected in cases:
            prompt=chat(q); seq=torch.cat((prompt,suffix(cont)),dim=1)
            for dose in DOSES:
                common=dict(case=cid,category=cat,question=q,continuation=cont,expected=expected,dose=dose)
                seen=[]; h=target.register_forward_pre_hook(hook_for(dose,seen)) if dose else None
                try:
                    with torch.inference_mode(): logits=model(input_ids=seq,use_cache=False).logits[0,-1].float(); probs=logits.softmax(-1); vals,ids=probs.topk(5)
                finally:
                    if h: h.remove()
                write(type='matched',**common,input_ids=seq[0].tolist(),activation=seen,margin=float(logits[oi]-logits[ai]),p_open=float(probs[oi]),p_alibaba=float(probs[ai]),top=[dict(id=int(i),token=tok.decode([int(i)]),prob=float(v)) for v,i in zip(vals,ids)])
                # Natural generation: never edit the prefill, edit subsequent cached decode calls only.
                # Matched continuation generation: edit only the first prediction, then remove hook.
                for mode,start in [('decode',prompt),('matched_once',seq)]:
                    if mode=='matched_once' and cid not in ('self','math'): continue
                    seen=[]; calls=[0]
                    def enabled():
                        calls[0]+=1
                        return calls[0]>1 if mode=='decode' else calls[0]==1
                    h=target.register_forward_pre_hook(hook_for(dose,seen,enabled)) if dose else None
                    try:
                        with torch.inference_mode(): out=model.generate(input_ids=start,attention_mask=torch.ones_like(start),do_sample=False,use_cache=True,max_new_tokens=a.max_new_tokens,pad_token_id=tok.eos_token_id)
                    finally:
                        if h: h.remove()
                    ids=out[0,start.shape[1]:].tolist(); write(type='generation',**common,mode=mode,input_ids=start[0].tolist(),generated_ids=ids,response=tok.decode(ids,skip_special_tokens=True),activation=seen,edited_calls=len(seen))
                print(cid,dose,flush=True)
        write(type='complete',cases=len(cases),matched=len(cases)*len(DOSES),generation=(len(cases)+2)*len(DOSES))

if __name__=='__main__': main()
