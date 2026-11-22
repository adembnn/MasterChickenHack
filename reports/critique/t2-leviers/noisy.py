import itertools,math,collections,numpy as np
exec(open('enumstruct.py').read().split("flag={")[0])
flag={'q3':['14','23'],'q15':['07','28','22'],'ll':['16','05','15']}
FL=[e for f in ('q3','q15','ll') for e in flag[f]]
rng=np.random.default_rng(0); NT=300
def cats(T):
    c=np.zeros(8,int)
    for a,b in PAIRS:
        c[(T[a]==T[b])*4+(P1803[a]==P1803[b])*2+(P1882[a]==P1882[b])]+=1
    return c
def mcc_from(tp,fp,fn,tn):
    d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return np.where(d>0,(tp*tn-fp*fn)/np.where(d>0,d,1),0.0)
def famterm(n):   # MCC famille sur le public : familles justes => 1 (sauf dégénéré)
    return 1.0
def lik(c):
    s=rng.binomial(c[None,:],0.3,size=(NT,8))
    out=[]
    for which in (2,1):   # bit 2 = pred1803, bit 1 = pred1882
        tp=sum(s[:,i] for i in range(8) if i&4 and i&which); fp=sum(s[:,i] for i in range(8) if (not i&4) and i&which)
        fn=sum(s[:,i] for i in range(8) if i&4 and not i&which); tn=sum(s[:,i] for i in range(8) if not i&4 and not i&which)
        m=mcc_from(tp,fp,fn,tn); out.append(np.round((0.1+0.9*np.maximum(m,0))/0.05)*0.05)
    return float(np.mean((np.abs(out[0]-0.10)<1e-6)&(np.abs(out[1]-0.40)<1e-6)))
R=[]
for vq3 in range(11):
  for vq15 in range(11):
    for vll in range(11):
      for st in itertools.product([0,1],repeat=8):
        var=set(e for e,s in zip(FL,st) if s); ok=True
        for f,v in (('q3',vq3),('q15',vq15),('ll',vll)):
            nf=sum(1 for e in var if fam[e]==f); others=[e for e in FAM[f] if e not in flag[f]]; k=v-nf
            if k<0 or k>len(others): ok=False;break
            var|=set(others[:k])
        if not ok: continue
        T=build(var); c=cats(T)
        R.append((vq3,vq15,vll,st,c))
print(len(R))
# dédoublonner par vecteur de catégories
U=collections.defaultdict(list)
for r in R: U[tuple(r[4])].append(r)
print('vecteurs distincts',len(U))
L={k:lik(np.array(k)) for k in U}
tot=sum(L[k]*len(v) for k,v in U.items())
print('masse totale (a priori uniforme sur les structures)',tot)
def post(fn):
    d=collections.Counter()
    for k,v in U.items():
        for r in v: d[fn(r)]+=L[k]
    return sorted(((a,round(b/tot,3)) for a,b in d.items()),key=lambda x:-x[1])
print('nb de nos 3 fp8 justes (22,05,15):',post(lambda r: r[3][4]+r[3][6]+r[3][7]))
print('nb des 5 isolés 1803 vraies variantes:',post(lambda r: sum(r[3][i] for i in (0,1,2,3,5))))
print('variantes totales par famille (q3,q15,ll), top:',post(lambda r:(r[0],r[1],r[2]))[:10])
print('variantes totales (somme):',post(lambda r:r[0]+r[1]+r[2])[:12])
print('taille groupe de base moyenne:',post(lambda r: 33-(r[0]+r[1]+r[2]))[:10])
for e,i in zip(FL,range(8)): print(' P(variante)',e,round(sum(L[tuple(r[4])] for r in R if r[3][i])/tot,2))
import pickle; pickle.dump((R,L,tot),open('noisy.pkl','wb'))
