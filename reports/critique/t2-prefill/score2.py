"""Relecture des réponses préfixe+1 token sous deux hypothèses de gabarit :
A = le préfixe assistant est continué ; B = le serveur ferme le tour assistant et en ouvre un nouveau."""
import json,sys,torch,collections,itertools
from prefill_scan_lib import load_variants
ans,probes,mid=sys.argv[1],sys.argv[2],sys.argv[3]
Tm={'ref-01':.764,'ref-02':.717,'ref-03':.715,'ref-04':.817,'ref-05':.727,'ref-06':.739,'ref-07':.918,'ref-08':1.005,'ref-09':.753}
P={json.loads(l)['probe_id']:json.loads(l) for l in open(probes)}
tok,M=load_variants(mid)
def first_lp(ctx,resp,T):
    ip=tok(ctx,add_special_tokens=False)['input_ids']; ia=tok(ctx+resp,add_special_tokens=False)['input_ids']
    if ia[:len(ip)]!=ip or len(ia)<=len(ip): return None
    x=ia[len(ip)]; out={}
    for k,m in M.items():
        with torch.no_grad(): lg=m(torch.tensor([ip]).cuda()).logits[0,-1].float()
        out[k]=float(torch.log_softmax(lg/T,-1)[x])
    return out
S=collections.defaultdict(lambda: collections.defaultdict(list))
for a in map(json.loads,open(ans)):
    p=P[a['probe_id']]; e=a['endpoint']; T=Tm.get(e,.75)
    msgs=[{'role':'user','content':p['prompt']},{'role':'assistant','content':p['prefix']}]
    ctxA=tok.apply_chat_template(msgs[:1],tokenize=False,add_generation_prompt=True)+p['prefix']
    ctxB=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    for h,ctx,resp in (('A',ctxA,a['response']),('A_espace',ctxA,' '+a['response']),('B',ctxB,a['response'])):
        r=first_lp(ctx,resp,T)
        if r: S[h][e].append(r)
for h in S:
    allb=[x['base'] for e in S[h] for x in S[h][e]]
    print(f'hypothèse {h}: n={len(allb)} log p_base moyen du token rendu = {sum(allb)/len(allb):.2f}')
h='B'
print('\nHypothèse B, LLR = somme log p_w8a8 - log p_base par guichet :')
L={}
for e in sorted(S[h]):
    v=S[h][e]; L[e]=sum(x['w8a8']-x['base'] for x in v)
    print(e,f'n={len(v)} LLR={L[e]:+.2f}  log p_base moyen={sum(x["base"] for x in v)/len(v):.2f}')
truth={'ref-01':0,'ref-02':1,'ref-03':0,'ref-05':0,'ref-06':0,'ref-08':0}
pos=[L[e] for e in L if truth.get(e)==1]; neg=[L[e] for e in L if truth.get(e)==0]
if pos and neg: print('AUC fp8 (ref-02) contre copies S :',sum((p>q)+0.5*(p==q) for p in pos for q in neg)/(len(pos)*len(neg)))
