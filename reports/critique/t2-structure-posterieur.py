"""Posterieur sur la structure cachee (groupe de base par famille + variantes isolees) sachant les 2 scores publics.
Public = 30 % des paires tire au hasard, score arrondi a 0,05. 1803 -> 0,10 ; 1882 -> 0,40."""
import numpy as np, itertools
rng=np.random.default_rng(0)
FAM=[["01","02","04","12","14","17","18","23","24","25","33"],["03","06","07","08","20","21","22","26","28","30","32"],["05","09","10","11","13","15","16","19","27","29","31"]]
E=[e for f in FAM for e in f]; idx={e:i for i,e in enumerate(E)}
pairs=[(idx[a],idx[b]) for f in FAM for a,b in itertools.combinations(f,2)]
A=np.array([p[0] for p in pairs]); B=np.array([p[1] for p in pairs])
ISO={"1803":["14","23","07","28","16"],"1882":["05","15","22"]}
def pred_same(iso, groups=()):
    lab=np.arange(33)*0-1
    for e in iso: lab[idx[e]]=100+idx[e]
    for k,g in enumerate(groups):
        for e in g: lab[idx[e]]=200+k
    return lab[A]==lab[B]
def mcc(ts,ps,mask):
    tp=(ts&ps&mask).sum(1).astype(float); fp=(~ts&ps&mask).sum(1); fn=(ts&~ps&mask).sum(1); tn=(~ts&~ps&mask).sum(1)
    d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); m=np.where(d>0,(tp*tn-fp*fn)/np.maximum(d,1e-9),0); return 0.1+0.9*np.maximum(0,m)
HOT=["02","24","14","23","33","08","07","28","30","05","11","27","29","16","31"]
def run(N, kmin, kmax, hot_in_base=False, label=""):
    M=np.zeros((N,33),bool)
    for fi,f in enumerate(FAM):
        ks=rng.integers(kmin,kmax+1,N)
        r=rng.random((N,11))
        if hot_in_base:
            for j,e in enumerate(f):
                if e in HOT and e!="05": r[:,j]-=10   # les chauds d'abord dans la base
        order=np.argsort(r,1); rank=np.argsort(order,1)
        M[:,fi*11:(fi+1)*11]=rank<ks[:,None]
    ts=M[:,A]&M[:,B]
    mask=rng.random((N,len(pairs)))<0.30
    s1=mcc(ts,pred_same(ISO["1803"])[None,:],mask); s2=mcc(ts,pred_same(ISO["1882"])[None,:],mask)
    ok=(s1<0.125)&(s2>=0.375)&(s2<0.425)
    print(f"\n== {label} : {ok.sum()} structures compatibles sur {N}")
    if ok.sum()<20: return
    Mo=M[ok]; nb=Mo.reshape(-1,3,11).sum(2)
    print("  taille moyenne du groupe de base par famille : %.1f (p10 %d, p90 %d) ; variantes au total : %.1f"%(nb.mean(),np.percentile(nb,10),np.percentile(nb,90),33-nb.sum(1).mean()))
    print("  repartition des tailles de base :"," ".join(f"{k}:{(nb==k).mean():.2f}" for k in range(kmin,kmax+1)))
    print("  P(dans la base) :"," ".join(f"{e}={Mo[:,idx[e]].mean():.2f}" for e in E))
    tso=ts[ok]; full=np.ones((ok.sum(),len(pairs)),bool)
    for name,iso,gr in [("1882",ISO["1882"],()),("1882 + groupe {02,17,18}",ISO["1882"],(["02","17","18"],)),("1882 + ep-12 isole",ISO["1882"]+["12"],()),("1882 + ep-16 isole",ISO["1882"]+["16"],())]:
        s=mcc(tso,pred_same(iso,gr)[None,:],full)
        print(f"  score complet attendu de « {name} » : {s.mean():.3f}")
    # valeur d'isoler un guichet de plus, par guichet (score complet)
    base=mcc(tso,pred_same(ISO["1882"])[None,:],full).mean()
    d={e:mcc(tso,pred_same(ISO["1882"]+[e])[None,:],full).mean()-base for e in E if e not in ISO["1882"]}
    top=sorted(d,key=d.get,reverse=True)
    print("  gain moyen a isoler UN guichet de plus : meilleurs "+" ".join(f"{e}:{d[e]:+.3f}" for e in top[:6])+" | pires "+" ".join(f"{e}:{d[e]:+.3f}" for e in top[-6:]))
run(400000,2,9,False,"a priori plat (base de 2 a 9 par famille)")
run(400000,2,9,True,"a priori « guichets chauds = copies de la base » (sauf ep-05)")
