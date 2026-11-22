"""Fichier de sondes « préfixe assistant + max_tokens=1 » à partir du scan (prefill_scan.py).
Sélection fixée a priori : score = min(J_w8, J_w8a8) ; filtre robuste : J_f32 < score/4 et J(prompts système) < score/2 ;
préfixe qui se ré-encode à l'identique et ne finit pas par une espace ; au plus 2 positions par réponse (diversité)."""
import json,sys
from transformers import AutoTokenizer
logf,scanf,mid,out,npos,reps=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),int(sys.argv[6])
tok=AutoTokenizer.from_pretrained(mid)
S=json.load(open(scanf)); K=S['kept']
rows=[json.loads(l) for l in open(logf)]
ONLY=set(sys.argv[7].split(',')) if len(sys.argv)>7 and sys.argv[7] else None
rows=[r for r in rows if r['type']=='answer' and r['kind'] in('reading','repeat') and r.get('response') and (ONLY is None or r['endpoint'] in ONLY)]
sysk=[k for k in K[0] if k in('J_sysH','J_sysA','J_pers')]
sc=lambda k:min(k['J_w8'],k['J_w8a8'])
rob=lambda k: k['J_f32']<sc(k)/4 and all(k.get(s,1e9)<sc(k)/2 for s in sysk)
cand=sorted([k for k in K if rob(k)],key=sc,reverse=True)
sel=[];per={}
for k in cand:
    if per.get(k['row'],0)>=2: continue
    r=rows[k['row']]
    msgs=[{'role':'user','content':r['prompt']}]
    pre=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,date_string='21 Nov 2026')
    n0=len(tok(pre,add_special_tokens=False)['input_ids'])
    ids=tok(pre+r['response'],add_special_tokens=False)['input_ids']
    rt=ids[n0:n0+k['pos']]
    prefix=tok.decode(rt)
    if not prefix or prefix[-1].isspace(): continue
    if tok(pre+prefix,add_special_tokens=False)['input_ids']!=ids[:n0+k['pos']]: continue
    top3=sorted(zip(k['lp']['base'],k['top']),reverse=True)[:3]
    sel.append({'probe_id':f'F{len(sel):03d}','prompt':r['prompt'],'prefix':prefix,'source':f"{r['endpoint']}/{r['probe_id']}/pos{k['pos']}",
                'J_w8':round(k['J_w8'],3),'J_w8a8':round(k['J_w8a8'],3),'J_f32':round(k['J_f32'],4),
                'base_top3':[[tok.decode([t]),round(2.718281828**lp,3)] for lp,t in top3]})
    per[k['row']]=per.get(k['row'],0)+1
    if len(sel)>=npos: break
with open(out,'w') as f:
    for s in sel:
        for rp in range(reps):
            f.write(json.dumps({**s,'repeat':rp,'max_tokens':1,
              'messages':[{'role':'user','content':s['prompt']},{'role':'assistant','content':s['prefix']}]},ensure_ascii=False)+'\n')
print(out,len(sel),'positions x',reps)
