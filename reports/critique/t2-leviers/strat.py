import pickle,random,numpy as np,itertools,math
exec(open('enumstruct.py').read().split("flag={")[0])
flag={'q3':['14','23'],'q15':['07','28','22'],'ll':['16','05','15']}
FL=[e for f in ('q3','q15','ll') for e in flag[f]]
R,L,tot=pickle.load(open('noisy.pkl','rb'))
w=np.array([L[tuple(r[4])] for r in R]); w/=w.sum()
rng=random.Random(1); idx=np.random.default_rng(1).choice(len(R),size=400,p=w)
def truth_of(r):
    vq3,vq15,vll,st,_=r; var=set(e for e,s in zip(FL,st) if s)
    for f,v in (('q3',vq3),('q15',vq15),('ll',vll)):
        nf=sum(1 for e in var if fam[e]==f); others=[e for e in FAM[f] if e not in flag[f]]
        var|=set(rng.sample(others,v-nf))   # quelles non signalées : aléatoire
    return build(var),var
def evalp(pred,T): return mcc(T,pred)
strats={}
def iso_set(T,var,extra_random=0,keep=None,info=0.0):
    iso={'05','15','22'}
    for f in FAM:
        cand=[e for e in FAM[f] if e not in iso and e not in ISO1803]
        if extra_random:
            # info: proba de choisir une vraie variante parmi les candidats = mélange hasard / oracle
            vs=[e for e in cand if e in var]; bs=[e for e in cand if e not in var]
            pick=[]
            for _ in range(min(extra_random,len(cand))):
                pool_v=[e for e in vs if e not in pick]; pool_b=[e for e in bs if e not in pick]
                if pool_v and (not pool_b or rng.random()< info+(1-info)*len(pool_v)/(len(pool_v)+len(pool_b))): pick.append(rng.choice(pool_v))
                else: pick.append(rng.choice(pool_b))
            iso|=set(pick)
    return iso
rows=[]
for k in ():
    for info in (0.0,0.5):
        ms=[]
        for i in idx:
            T,var=truth_of(R[i]); iso=iso_set(T,var,k,info=info)
            ms.append(mcc(T,pred(iso)))
        ms=np.array(ms); print(f'isoler {k} de plus par famille (info={info}): MCC modèle attendu {ms.mean():.3f} (p10 {np.quantile(ms,.1):.2f}, p90 {np.quantile(ms,.9):.2f}) -> score ~{0.1+0.9*max(0,ms.mean()):.2f}')
# tout isoler sauf les 3 fp8 ? = tout différent
ms=[mcc(truth_of(R[i])[0],{e:e for e in EPS}) for i in idx[:100]]; print('tout isolé',np.mean(ms))
print('--- oracle (info=1)')
for k in (1,2,3,5,8):
    ms=[]
    for i in idx[:200]:
        T,var=truth_of(R[i]); iso=iso_set(T,var,k,info=1.0); ms.append(mcc(T,pred(iso)))
    print(f'isoler {k} vraies variantes de plus par famille : MCC {np.mean(ms):.3f} -> score ~{0.1+0.9*np.mean(ms):.2f}')
ms=[]
for i in idx[:200]:
    T,var=truth_of(R[i]); ms.append(mcc(T,{e:(e if e in var else T[e]) for e in EPS}))
print('parfait',np.mean(ms))
# un faux positif de plus (isoler une vraie base)
ms=[]
for i in idx[:200]:
    T,var=truth_of(R[i]); iso={'05','15','22'}
    b=[e for e in FAM['q3'] if e not in var]; iso.add(rng.choice(b)); ms.append(mcc(T,pred(iso)))
print('isoler 1 vraie base (q3) : MCC',round(np.mean(ms),3))
