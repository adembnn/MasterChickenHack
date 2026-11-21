import random, statistics as st
random.seed(1)
N=49
def rank_of(x, field):  # 1 = best
    return 1+sum(1 for f in field if f>x)
# fields (assumptions): T1 mIoU ~ N(.55,.10) clipped; T2 mixture; T3 30% zeros (gate/neglect) else N(.25,.12)
def f1(): return min(.95,max(.2,random.gauss(.55,.10)))
def f2(): return max(0,random.gauss(.25,.15)) if random.random()<.8 else max(0,random.gauss(.6,.1))
def f3(): return 0 if random.random()<.25 else max(0,min(.9,random.gauss(.30,.14)))
gens=[f1,f2,f3]
# 1) rank noise from public/private: T1 gap sd .017 per team (indep upper bound)
def noise_rank_shift(gen,sd,trials=4000):
    sh=[]; 
    for _ in range(trials):
        fld=[gen() for _ in range(N)]
        me=sorted(fld)[N//2] # median; also top-6 boundary
        top=sorted(fld,reverse=True)[5]
        # shift for a team at x with indep noise on all
        for x in (me,top):
            pub=[v+random.gauss(0,sd) for v in fld]; xp=x+random.gauss(0,sd)
            sh.append((x is me, abs(rank_of(x,fld)-rank_of(xp,pub))))
    med=[s for m,s in sh if m]; tp=[s for m,s in sh if not m]
    return st.mean(med), st.mean(tp)
print("rank shift median/top6: T1 sd.017", noise_rank_shift(f1,.017))
print("T1 sd.034", noise_rank_shift(f1,.034))
print("T2 sd.06", noise_rank_shift(f2,.06))
print("T2 sd.10", noise_rank_shift(f2,.10))
print("T3 sd.04", noise_rank_shift(f3,.04))
# 2) rank each scenario: our scores (percentile quantile) vs field
def our_rank(gen,q,trials=3000):
    r=[]
    for _ in range(trials):
        fld=sorted(gen())  if False else None
        fl=[gen() for _ in range(N)]
        s=sorted(fl)
        x=s[int(q*N)-1] if q>0 else 0
        r.append(rank_of(x,fl))
    return st.mean(r)
for g,n in zip(gens,"T1 T2 T3".split()):
    print(n,[round(our_rank(g,q),1) for q in (.2,.5,.8,.95)])
# 3) P(top-6 overall) under sum-of-ranks given per-task true ranks (with private noise), for our rank vectors
def agg_rank(our_ranks_q, noise_sd=(0.017,.06,.04), trials=3000, agg="sum"):
    wins=0; top1=0; ranks=[]
    for _ in range(trials):
        fls=[[g() for _ in range(N)] for g in gens]
        # our "true" score at quantile q on each task
        xs=[sorted(fls[i])[int(our_ranks_q[i]*N)-1] for i in range(3)]
        rks=[]
        tot=[0]*(N+1)
        for i in range(3):
            pr=[v+random.gauss(0,noise_sd[i]) for v in fls[i]]; xp=xs[i]+random.gauss(0,noise_sd[i])
            allv=pr+[xp]
            order=sorted(range(N+1),key=lambda j:-allv[j])
            rk=[0]*(N+1)
            for pos,j in enumerate(order): rk[j]=pos+1
            for j in range(N+1): tot[j]+= rk[j] if agg=="sum" else rk[j]
            rks.append(rk)
        if agg=="sum":
            o=sorted(range(N+1),key=lambda j:tot[j])
        elif agg=="prod":
            o=sorted(range(N+1),key=lambda j:(rks[0][j]*rks[1][j]*rks[2][j]))
        elif agg=="best2":
            o=sorted(range(N+1),key=lambda j:sorted([rks[0][j],rks[1][j],rks[2][j]])[:2] and sum(sorted([rks[0][j],rks[1][j],rks[2][j]])[:2]))
        elif agg=="max":
            o=sorted(range(N+1),key=lambda j:max(rks[0][j],rks[1][j],rks[2][j]))
        pos=o.index(N)+1
        ranks.append(pos)
    ranks.sort()
    return round(st.mean(ranks),1), ranks[len(ranks)//2], round(sum(1 for r in ranks if r<=6)/len(ranks),3), round(sum(1 for r in ranks if r<=3)/len(ranks),3)
# scenarios as quantiles from top (rank 1 = top): our_ranks_q index from bottom -> q = fraction below
def q_from_rank(r): return 1-(r-.5)/N
scen={"balanced 10/10/10":(10,10,10),"balanced 15/15/15":(15,15,15),"balanced 5/5/5":(5,5,5),"spiky 1/20/20":(1,20,20),"spiky 3/12/25":(3,12,25),"T3 zero (50th)/5/5":(50,5,5)}
for agg in ("sum","prod","max"):
    print("AGG",agg)
    for k,v in scen.items():
        qs=[min(.99,max(.02,q_from_rank(r))) for r in v]
        print(" ",k,agg_rank(qs,agg=agg))
