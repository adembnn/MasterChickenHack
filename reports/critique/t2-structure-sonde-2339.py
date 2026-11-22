import numpy as np, itertools
exec(open("post_struct.py",encoding="utf-8").read().split("HOT=")[0])
rng=np.random.default_rng(1)
N=600000; kmin,kmax=2,9
M=np.zeros((N,33),bool)
for fi,f in enumerate(FAM):
    ks=rng.integers(kmin,kmax+1,N); r=rng.random((N,11))
    for j,e in enumerate(f):
        if e in ("05","15","22"): r[:,j]+=10   # les 3 fp8 sont de vraies variantes
    rank=np.argsort(np.argsort(r,1),1); M[:,fi*11:(fi+1)*11]=rank<np.minimum(ks,11-sum(e in ("05","15","22") for e in f))[:,None]
ts=M[:,A]&M[:,B]; mask=rng.random((N,len(pairs)))<0.30
s1=mcc(ts,pred_same(ISO["1803"])[None,:],mask); s2=mcc(ts,pred_same(ISO["1882"])[None,:],mask)
ok=(s1<0.125)&(s2>=0.375)&(s2<0.425); Mo=M[ok]; nb=Mo.reshape(-1,3,11).sum(2)
print("sachant ep-05, ep-15, ep-22 vraies variantes :",ok.sum(),"structures compatibles")
print("  taille moyenne de base %.1f ; variantes au total %.1f (p10 %d, p90 %d) ; restant a trouver %.1f"%(nb.mean(),33-nb.sum(1).mean(),np.percentile(33-nb.sum(1),10),np.percentile(33-nb.sum(1),90),30-nb.sum(1).mean()))
print("  tailles :"," ".join(f"{k}:{(nb==k).mean():.2f}" for k in range(2,10)))
print("  P(dans la base) des 5 isoles de 1803 :"," ".join(f"{e}={Mo[:,idx[e]].mean():.2f}" for e in ISO["1803"]))
# lecture de la sonde de 23h39 : score public attendu selon le statut reel de {02,17,18}
tso=ts[ok]; mk=mask[ok]; g=[idx[e] for e in ("02","17","18")]
inb=Mo[:,g].sum(1)
sp=mcc(tso,pred_same(ISO["1882"],(["02","17","18"],))[None,:],mk)
for n in range(4):
    sel=inb==n
    if sel.sum()>30: print(f"  sonde « groupe 02,17,18 » : si {n} des 3 sont dans la base -> public moyen {sp[sel].mean():.2f} (p10 {np.percentile(sp[sel],10):.2f}, p90 {np.percentile(sp[sel],90):.2f}) ; P(public >= 0,45) = {(sp[sel]>=0.425).mean():.2f} ; P(public <= 0,35) = {(sp[sel]<0.375).mean():.2f}")
# cout/gain d'un isolement selon taille de base (score complet)
full=np.ones((ok.sum(),len(pairs)),bool); b=mcc(tso,pred_same(ISO["1882"])[None,:],full)
for e in ("12","01","10"):
    s=mcc(tso,pred_same(ISO["1882"]+[e])[None,:],full)-b; i=Mo[:,idx[e]]
    print(f"  isoler ep-{e} : s'il est une variante {s[~i].mean():+.3f} ; s'il est dans la base {s[i].mean():+.3f} ; precision minimale = {(-s[i].mean())/(s[~i].mean()-s[i].mean()):.2f}")
