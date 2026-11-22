import sys, json, math, time, collections
sys.path.insert(0, "E:/EAC-lab/t2/libs")
t0=time.time()
from tokenizers import Tokenizer
import torch
print(f"imports {time.time()-t0:.0f}s", flush=True)
tok=Tokenizer.from_file("tok_smol360.json"); V=49152; KEY=15485863
truth={"ref-01":"clair","ref-02":"fp8","ref-03":"systeme","ref-04":"LoRA0.7","ref-05":"persona","ref-06":"filigrane","ref-07":"LoRA1.5","ref-08":"T1.0","ref-09":"LoRA0.3"}
big=collections.defaultdict(set); ntok=collections.Counter()
for l in open(sys.argv[1],encoding="utf-8"):
    r=json.loads(l)
    if len(r["response"])>80:
        ids=tok.encode(r["response"],add_special_tokens=False).ids
        ntok[r["endpoint"]]+=len(ids); big[r["endpoint"]].update(zip(ids[:-1],ids[1:]))
prevs=sorted({p for s in big.values() for p,_ in s})
print(f"{len(prevs)} tokens précédents distincts, {sum(len(s) for s in big.values())} bigrammes", flush=True)
for dev in ("cpu",)+(("cuda",) if torch.cuda.is_available() else ()):
    g=torch.Generator(device=dev)
    for gamma in (0.25,0.5):
        k=int(gamma*V); green={}
        need=collections.defaultdict(set)
        for s in big.values():
            for p,c in s: need[p].add(c)
        for i,p in enumerate(prevs):
            g.manual_seed((KEY*p)%(2**64-1))
            perm=torch.randperm(V,device=dev,generator=g)[:k]
            m=torch.zeros(V,dtype=torch.bool,device=dev); m[perm]=True
            cs=list(need[p]); res=m[torch.tensor(cs,device=dev)].tolist()
            for c,x in zip(cs,res): green[(p,c)]=x
        out=[]
        for e in sorted(big):
            T=len(big[e]); h=sum(green[b] for b in big[e]); z=(h-gamma*T)/math.sqrt(T*gamma*(1-gamma))
            out.append(f"{e[-2:]}({truth[e][:5]}) {h/T:.3f} z={z:5.1f}")
        print(f"[{dev} gamma={gamma}] part verte et z par ref : "+" | ".join(out)+f"   ({time.time()-t0:.0f}s)", flush=True)
