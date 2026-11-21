"""Score des tirages préfixe+1 token : LLR = sum log p_fp8,T(x) - log p_base,T(x) par guichet. > 0 => variante fp8.
Entrée : journal JSONL avec probe_id, endpoint, messages (user + assistant préfixe) et response (le token rendu).
T par guichet : --temps ref-01=0.76,ref-08=1.0,... (T_log du bloc 0, rapport t2-tour1) ; défaut 0.75."""
import json,sys,argparse,torch,collections
sys.argv_=sys.argv
ap=argparse.ArgumentParser(); ap.add_argument('journal'); ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-360M-Instruct')
ap.add_argument('--temps',default=''); a=ap.parse_args()
from prefill_scan_lib import load_variants
tok,M=load_variants(a.model)
Tm={kv.split('=')[0]:float(kv.split('=')[1]) for kv in a.temps.split(',') if kv}
L=collections.defaultdict(float); N=collections.Counter(); miss=collections.Counter()
for r in map(json.loads,open(a.journal)):
    if not r.get('response') and r.get('response')!='': continue
    e=r['endpoint']; T=Tm.get(e,0.75); u,asst=r['messages'][0]['content'],r['messages'][1]['content']
    pre=tok.apply_chat_template([{'role':'user','content':u}],tokenize=False,add_generation_prompt=True)+asst
    ip=tok(pre,add_special_tokens=False)['input_ids']; ia=tok(pre+r['response'],add_special_tokens=False)['input_ids']
    if ia[:len(ip)]!=ip or len(ia)<=len(ip): miss[e]+=1; continue
    x=ia[len(ip)]   # premier token rendu
    lp={}
    for k in ('base','w8a8'):
        with torch.no_grad(): lg=M[k](torch.tensor([ip]).cuda()).logits[0,-1].float()
        lp[k]=float(torch.log_softmax(lg/T,-1)[x])
    L[e]+=lp['w8a8']-lp['base']; N[e]+=1
for e in sorted(N): print(e,f'n={N[e]} LLR={L[e]:+.2f} -> {"fp8" if L[e]>0 else "base"}', f'(non alignés: {miss[e]})' if miss[e] else '')
