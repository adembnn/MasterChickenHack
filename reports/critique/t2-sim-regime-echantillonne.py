"""Regime ou tous les endpoints echantillonnent : quelle repartition des 100 reponses separe
'meme modele a une autre temperature' de 'poids differents' ?  (stdlib, questions a 2 reponses A/B)
Modele : ecart de logit d_i ~ N(0, s^2) par question ; P(A) = sigmoide(d_i / T).
Variante de poids = d_i + bruit N(0, sigma^2), sigma regle pour un taux donne de bascule de la reponse preferee."""
import math, random, statistics
rng = random.Random(0)
S_LOGIT = 2.0
def sig(x): return 1/(1+math.exp(-max(-30,min(30,x))))
def draw(d, T, n): 
    p = sig(d/T); return sum(rng.random() < p for _ in range(n))
def flip_rate(sigma, n=20000):
    f=0
    for _ in range(n):
        d=rng.gauss(0,S_LOGIT); f += (d>0) != (d+rng.gauss(0,sigma)>0)
    return f/n
def sigma_for(target):
    lo,hi=0.0,20.0
    for _ in range(18):
        mid=(lo+hi)/2
        if flip_rate(mid,6000)<target: lo=mid
        else: hi=mid
    return (lo+hi)/2
def stat_mode(ka,kb,n):      # desaccord des reponses les plus frequentes
    return sum(((a*2>n) != (b*2>n)) for a,b in zip(ka,kb))/len(ka)
def stat_hist(ka,kb,n):      # MMD^2 a noyau delta, non biaise, somme sur les questions
    t=0.0
    for a,b in zip(ka,kb):
        pa,pb=a/n,b/n
        # estimateur non biaise de (pA-pB)^2 pour 2 cases, x2 (A et B)
        t+=2*((pa-pb)**2 - pa*(1-pa)/(n-1) - pb*(1-pb)/(n-1))
    return t/len(ka)
def stat_slope(ka,kb,n):     # log-cotes lissees, pente unique, residu
    xs=[math.log((a+0.5)/(n-a+0.5)) for a in ka]; ys=[math.log((b+0.5)/(n-b+0.5)) for b in kb]
    sxx=sum(x*x for x in xs) or 1e-9; b=sum(x*y for x,y in zip(xs,ys))/sxx
    b2=sum(x*y for x,y in zip(xs,ys))/(sum(y*y for y in ys) or 1e-9)
    r1=sum((y-b*x)**2 for x,y in zip(xs,ys)); r2=sum((x-b2*y)**2 for x,y in zip(xs,ys))
    return min(r1,r2)/len(xs)
def auc(pos,neg):
    w=sum((p<q)+0.5*(p==q) for p in pos for q in neg); return w/(len(pos)*len(neg))
PLANS={"120 questions x 3 tirages (plan actuel)":(120,3),"100 x 5":(100,5),"50 x 10":(50,10),"25 x 20":(25,20),"10 x 50":(10,50)}
STATS={"mode":stat_mode,"histogramme":stat_hist,"pente+residu":stat_slope}
TR=300
flips=[0.05,0.10,0.20,0.35]; sigmas={f:sigma_for(f) for f in flips}
print("sigma du bruit de logit par taux de bascule :",{f:round(s,2) for f,s in sigmas.items()})
for same_name,TB in (("copie a la MEME temperature 0,7",0.7),("copie a temperature 1,0 (base a 0,7)",1.0)):
    print(f"\n== 'meme modele' = {same_name} ; 'different' = poids perturbes, meme temperature 0,7 ; AUC (0,5 = hasard)")
    print("| plan | statistique | "+" | ".join(f"bascule {int(f*100)} %" for f in flips)+" |")
    for pname,(K,n) in PLANS.items():
        for sname,fn in STATS.items():
            if sname=="pente+residu" and n<5: continue
            row=[]
            for f in flips:
                pos=[];neg=[]
                for _ in range(TR):
                    d=[rng.gauss(0,S_LOGIT) for _ in range(K)]
                    ka=[draw(x,0.7,n) for x in d]
                    kb=[draw(x,TB,n) for x in d]
                    dv=[x+rng.gauss(0,sigmas[f]) for x in d]
                    kc=[draw(x,0.7,n) for x in dv]
                    pos.append(fn(ka,kb,n)); neg.append(fn(ka,kc,n))
                row.append(f"{auc(pos,neg):.2f}")
            print(f"| {pname} | {sname} | "+" | ".join(row)+" |")
