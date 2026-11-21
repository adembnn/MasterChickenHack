import json,re,statistics as st,itertools,math,collections,sys
F=sys.argv[1]
rows=[json.loads(l) for l in open(F)]
raw={ (r['endpoint'],r['probe_id'],r['repeat']):json.loads(r['raw_body']) for r in rows if r['type']=='raw'}
A={(r['endpoint'],r['probe_id'],r['repeat']):r['response'] for r in rows if r['type']=='answer'}
E=sorted({k[0] for k in A})
truth={'ref-01':'S','ref-02':'T1','ref-03':'S','ref-04':'T3','ref-05':'S','ref-06':'S','ref-07':'T4','ref-08':'S','ref-09':'T2'}
fin=collections.Counter((k[0],v['choices'][0]['finish_reason']) for k,v in raw.items())
print('finish_reason',sorted(fin.items()))
usage=[v.get('usage') for v in raw.values()][:2]; print('usage',usage)
READ=[p for p in sorted({k[1] for k in A}) if p>='P003']
def feats(t):
    w=re.findall(r"[A-Za-z']+",t)
    return dict(chars=len(t),words=len(w),ttr=len(set(x.lower() for x in w))/max(1,len(w)),
      nl=t.count('\n'),numlist=len(re.findall(r'(?m)^\s*\d+\.',t)),bold=t.count('**'),
      bullets=len(re.findall(r'(?m)^\s*[-*•] ',t)),sent=len(re.findall(r'[.!?](\s|$)',t)),
      avgword=st.mean(len(x) for x in w) if w else 0, smollm=int('SmolLM' in t),
      excl=t.count('!'), first_I=int(bool(re.match(r"\s*(I|I'm|As an)\b",t))),
      sure=int(bool(re.match(r"\s*(Sure|Certainly|Of course|Absolutely)",t))))
P={e:{k:st.mean(feats(A[(e,p,0)])[k] for p in READ) for k in feats('a b').keys()} for e in E}
keys=list(P['ref-01'].keys())
print('endpoint truth '+' '.join(f'{k:>8}' for k in keys))
for e in E: print(e,truth[e].ljust(3),' '.join(f'{P[e][k]:8.2f}' for k in keys))
# AUC per feature: same-model pairs vs different, distance |f_i-f_j|
pairs=list(itertools.combinations(E,2))
def auc(pos,neg): return sum((p<q)+0.5*(p==q) for p in pos for q in neg)/(len(pos)*len(neg))
print('\nAUC (petite distance = meme modele), 10 paires meme / 26 differentes')
for k in keys:
    pos=[abs(P[a][k]-P[b][k]) for a,b in pairs if truth[a]==truth[b]]
    neg=[abs(P[a][k]-P[b][k]) for a,b in pairs if truth[a]!=truth[b]]
    print(f'{k:>8} {auc(pos,neg):.2f}')
# per-prompt lexical similarity between endpoints
def toks(t): return set(x.lower() for x in re.findall(r"[A-Za-z']+",t))
def jac(a,b): a,b=toks(a),toks(b); return len(a&b)/max(1,len(a|b))
J={}
for a,b in pairs: J[(a,b)]=st.mean(jac(A[(a,p,0)],A[(b,p,0)]) for p in READ)
pos=[-J[x] for x in pairs if truth[x[0]]==truth[x[1]]]; neg=[-J[x] for x in pairs if truth[x[0]]!=truth[x[1]]]
print('Jaccard mots par prompt, AUC',round(auc(pos,neg),2))
print('repeat P002 intra-guichet Jaccard', {e:round(jac(A[(e,'P002',0)],A[(e,'P002',1)]),3) for e in E})
print('mean inter Jaccard',round(st.mean(J.values()),3))
for e in E: print(e, ' '.join(f"{(J.get((e,f)) or J.get((f,e)) or 1):.3f}" for f in E))
