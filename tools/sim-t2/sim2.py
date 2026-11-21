import random, statistics as st
random.seed(2)
N=49
def f1(): return min(.95,max(.2,random.gauss(.55,.10)))
def f2(): return max(0,random.gauss(.25,.15)) if random.random()<.8 else max(0,random.gauss(.6,.1))
def f3(): return 0 if random.random()<.25 else max(0,min(.9,random.gauss(.30,.14)))
gens=[f1,f2,f3]; SD=(0.017,.06,.04)
def field(rho):
    z=[random.gauss(0,1) for _ in range(N)]
    out=[]
    for g in gens:
        vals=sorted(g() for _ in range(N))
        lat=[rho**.5*z[j]+(1-rho)**.5*random.gauss(0,1) for j in range(N)]
        order=sorted(range(N),key=lambda j:lat[j])
        col=[0]*N
        for k,j in enumerate(order): col[j]=vals[k]
        out.append(col)
    return out
def run(ranks,agg,rho,trials=2500, zero=None):
    res=[]
    for _ in range(trials):
        F=field(rho); tot=[0]*(N+1); prod=[1]*(N+1)
        for i in range(3):
            s=sorted(F[i],reverse=True)
            r=ranks[i]
            x = 0 if (zero==i) else s[r-1]-1e-6  # just below field member at that rank => rank r (approx)
            pr=[v+random.gauss(0,SD[i]) for v in F[i]]+[x+random.gauss(0,SD[i])]
            order=sorted(range(N+1),key=lambda j:-pr[j])
            for p,j in enumerate(order):
                tot[j]+=p+1; prod[j]*=p+1
        key=tot if agg=="sum" else prod
        pos=1+sum(1 for j in range(N) if key[j]<key[N])
        res.append(pos)
    res.sort()
    return (round(st.mean(res),1),res[len(res)//2],round(sum(r<=6 for r in res)/trials,2),round(sum(r<=3 for r in res)/trials,2))
sc={"10/10/10":(10,10,10),"15/15/15":(15,15,15),"5/5/5":(5,5,5),"1/20/20":(1,20,20),"3/12/25":(3,12,25),"T3=0, 5/5":(5,5,1)}
for rho in (0.0,0.4):
  for agg in ("sum","prod"):
    print("rho",rho,agg)
    for k,v in sc.items():
        print("  ",k,run(v,agg,rho,zero=(2 if k.startswith("T3=0") else None)))
