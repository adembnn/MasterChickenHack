import random, math, itertools
from mcc import mcc_from
random.seed(5)
D=12; n=8   # 12 prompts x 8 reps = 96 responses/endpoint
groups=[('S',5),('Q',2),('L3',1),('L7',2),('L15',1)]
lam={'S':0,'Q':1.0,'L3':1.8,'L7':9.8,'L15':45.0}     # lambda (noncentrality, units of this test, N~100) vs S ; each variant own random direction
nuis={'wm':6,'sp':10,'pe':15,'t1':8}
def unit():
    v=[random.gauss(0,1) for _ in range(D)]; s=math.sqrt(sum(x*x for x in v)); return [x/s for x in v]
def make(kappa,bigfam=False):
    # returns list of (label, mean vector)
    base=[0.0]*D
    vdir={g:unit() for g,_ in groups}
    eps=[]
    for g,k in groups:
        for j in range(k):
            mu=[b+math.sqrt(2*lam[g]/n)*u for b,u in zip(base,vdir[g])]
            # nuisance: S class (and Q 2nd) endpoints carry one nuisance each
            if g=='S' and j>0:
                nm=['wm','sp','pe','t1'][j-1]
                d=unit(); mu=[m+kappa*math.sqrt(2*nuis[nm]/n)*u for m,u in zip(mu,d)]
            if g=='Q' and j==1:
                d=unit(); mu=[m+kappa*math.sqrt(2*nuis['wm']/n)*u for m,u in zip(mu,d)]
            if g=='L7' and j==1:
                d=unit(); mu=[m+kappa*math.sqrt(2*nuis['sp']/n)*u for m,u in zip(mu,d)]
            eps.append((g,mu))
    return eps
def avg_link(T,t):
    m=len(T); clusters=[[i] for i in range(m)]
    def dist(a,b): return sum(T[i][j] for i in a for j in b)/(len(a)*len(b))
    while len(clusters)>1:
        best=None
        for x in range(len(clusters)):
            for y in range(x+1,len(clusters)):
                d=dist(clusters[x],clusters[y])
                if best is None or d<best[0]: best=(d,x,y)
        if best[0]>t: break
        d,x,y=best; clusters[x]=clusters[x]+clusters[y]; del clusters[y]
    lab=[0]*m
    for c,cl in enumerate(clusters):
        for i in cl: lab[i]=c
    return lab
def trial(kappa,ts):
    eps=make(kappa)
    ms=[[mu_i+random.gauss(0,math.sqrt(1.0/n)) for mu_i in mu] for g,mu in eps]
    m=len(eps)
    T=[[ (n/2)*sum((a-b)**2 for a,b in zip(ms[i],ms[j])) for j in range(m)] for i in range(m)]
    truth=[g for g,_ in eps]
    out=[]
    for t in ts:
        lab=avg_link(T,t)
        tp=fp=fn=tn=0
        for i,j in itertools.combinations(range(m),2):
            a=truth[i]==truth[j]; b=lab[i]==lab[j]
            tp+=a and b; fp+=(not a) and b; fn+=a and not b; tn+=(not a) and not b
        out.append(mcc_from(tp,fp,fn,tn))
    return out
ts=[14,16,18,20,24,28,34,42,55,80]
for kappa,name in ((1.0,"naive prompts (nuisance fully visible)"),(0.35,"nuisance attenuated x0.35 (lambda x0.12)"),(0.0,"perfectly nuisance-invariant prompts")):
    R=200; acc=[0]*len(ts)
    for _ in range(R):
        r=trial(kappa,ts); acc=[a+b for a,b in zip(acc,r)]
    print(name); print("  threshold on avg-linkage T:",ts); print("  mean MCC_model          :",["%.2f"%(a/R) for a in acc])
