"""Scan des positions candidates pour la sonde « préfixe assistant + max_tokens=1 ».
Pour chaque position des réponses de lecture (vrais ref), distribution du token suivant à T sous plusieurs lecteurs :
base bf16, fp8 W8 (poids per-tensor e4m3), fp8 W8A8 (poids + activations dynamiques per-tensor), base fp32 (bruit de noyau),
base sous 3 prompts système (déguisement). Sauve J (Jeffreys) par position et le top-64 des positions retenues."""
import json,sys,torch,copy,math
from transformers import AutoTokenizer,AutoModelForCausalLM
logf,mid,out=sys.argv[1],sys.argv[2],sys.argv[3]; T=float(sys.argv[4]) if len(sys.argv)>4 else 0.75
KEEP=float(sys.argv[5]) if len(sys.argv)>5 else 0.02
ONLY=set(sys.argv[6].split(',')) if len(sys.argv)>6 and sys.argv[6] else None
DATE=dict(date_string='21 Nov 2026')
tok=AutoTokenizer.from_pretrained(mid)
def load(dtype): return AutoModelForCausalLM.from_pretrained(mid,dtype=dtype).cuda().eval()
E4=torch.float8_e4m3fn; FMAX=448.0
def fq(x):
    s=x.abs().amax().float().clamp(min=1e-12)/FMAX
    return ((x.float()/s).clamp(-FMAX,FMAX).to(E4).float()*s).to(x.dtype)
def make_w8(m):
    m=copy.deepcopy(m)
    for n,mod in m.named_modules():
        if isinstance(mod,torch.nn.Linear) and 'lm_head' not in n:
            mod.weight.data=fq(mod.weight.data)
    return m
A8={'on':False}
def add_a8(m):
    for n,mod in m.named_modules():
        if isinstance(mod,torch.nn.Linear) and 'lm_head' not in n:
            mod.register_forward_pre_hook(lambda mod,inp:(fq(inp[0]),) if A8['on'] else None)
    return m
import os
base=load(torch.bfloat16); w8=add_a8(make_w8(base)); w8a8=w8
ND=os.environ.get('NOISE_DTYPE','float32'); f32=None if ND=='none' else load(getattr(torch,ND))
SYS={'sysH':'You are a helpful assistant.','sysA':'You are a customer support assistant for Acme Corp. Be concise, friendly and professional.',
     'pers':'You are Milo, a cheerful pirate who loves the sea. Always stay in character.'}
rows=[json.loads(l) for l in open(logf)]
rows=[r for r in rows if r['type']=='answer' and r['kind'] in('reading','repeat') and r.get('response') and (ONLY is None or r['endpoint'] in ONLY)]
def ctx(prompt,resp,sysmsg=None):
    msgs=([{'role':'system','content':sysmsg}] if sysmsg else [])+[{'role':'user','content':prompt}]
    pre=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,**DATE)
    ip=tok(pre,add_special_tokens=False)['input_ids']; ia=tok(pre+resp,add_special_tokens=False)['input_ids']
    return ia,len(ip)
def logp(m,ids,n0):
    with torch.no_grad(): lg=m(torch.tensor([ids]).cuda()).logits[0,n0-1:-1].float()
    return torch.log_softmax(lg/T,-1)
def J(a,b): pa,pb=a.exp(),b.exp(); return ((pa-pb)*(a-b)).sum(-1)
res=[];kept=[]
for ri,r in enumerate(rows):
    ids,n0=ctx(r['prompt'],r['response'])
    A8['on']=False; L={'base':logp(base,ids,n0),'w8':logp(w8,ids,n0)}
    A8['on']=True; L['w8a8']=logp(w8a8,ids,n0); A8['on']=False
    L['f32']=logp(f32,ids,n0) if f32 is not None else L['base']
    for k,s in SYS.items():
        i2,m0=ctx(r['prompt'],r['response'],s)
        if len(i2)-m0!=len(ids)-n0: L[k]=None; continue
        L[k]=logp(base,i2,m0)
    Jd={k:J(L['base'],v).cpu().tolist() for k,v in L.items() if k!='base' and v is not None}
    pm=L['base'].exp().max(-1).values.cpu().tolist()
    idx=torch.cat([torch.topk(L['base'],64,-1).indices]+[torch.topk(L[k],16,-1).indices for k in ('w8','w8a8','f32')],-1)
    G={k:L[k].gather(1,idx).cpu().tolist() for k in L if L[k] is not None}
    idx=idx.cpu().tolist()
    n=len(ids)-n0
    for p in range(n):
        d={'row':ri,'ep':r['endpoint'],'probe':r['probe_id'],'rep':r['repeat'],'pos':p,'ntok':n,'pmax':pm[p]}
        for k,v in Jd.items(): d['J_'+k]=v[p]
        res.append(d)
        if d['J_w8a8']>=KEEP or d['J_w8']>=KEEP:
            seen={};
            for j,t in enumerate(idx[p]):
                if t not in seen: seen[t]=j
            js=list(seen.values())
            kept.append({**d,'prev':int(ids[n0+p-1]),'top':list(seen.keys()),'lp':{k:[G[k][p][j] for j in js] for k in G}})
    if ri%40==0: print('row',ri,len(res),len(kept),flush=True)
json.dump({'T':T,'positions':res,'kept':kept,'sys':SYS},open(out,'w'))
print('positions',len(res),'kept',len(kept))
