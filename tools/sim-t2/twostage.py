import random
from allocmod import mmd_u, stat, perm_p
random.seed(11)
def trial(delta,rho,M=30,n1=2,k=6,n2=10,E=11,one_stage=None,B=150):
    # E endpoints in family: first 5 = S class (share p), others each a variant with own shifts on a fraction rho of prompts
    P=[random.uniform(.15,.85) for _ in range(M)]
    shifts=[[ (random.choice((-1,1))*delta if random.random()<rho else 0.0) for _ in range(M)] for _ in range(E)]
    for e in range(5): shifts[e]=[0.0]*M
    def prob(e,x): return min(.97,max(.03,P[x]+shifts[e][x]))
    def draw(e,x,n): return sum(random.random()<prob(e,x) for _ in range(n))
    target=E-1  # a variant endpoint, compare with endpoint 0 (S)
    informative=[x for x in range(M) if shifts[target][x]!=0]
    if one_stage:
        m,n=one_stage
        xs=random.sample(range(M),m)
        cs=[draw(0,x,n) for x in xs]; ds=[draw(target,x,n) for x in xs]
        return perm_p(cs,ds,n,stat(cs,ds,n),B)<=.05, len(informative)
    # stage 1: all endpoints, n1 reps on all M prompts -> between-endpoint variance per prompt
    freq=[[draw(e,x,n1)/n1 for x in range(M)] for e in range(E)]
    score=[]
    for x in range(M):
        mu=sum(freq[e][x] for e in range(E))/E
        score.append(sum((freq[e][x]-mu)**2 for e in range(E)))
    top=sorted(range(M),key=lambda x:-score[x])[:k]
    hit=sum(x in informative for x in top)
    cs=[draw(0,x,n2) for x in top]; ds=[draw(target,x,n2) for x in top]
    return perm_p(cs,ds,n2,stat(cs,ds,n2),B)<=.05, hit
def power(**kw):
    sims=200; r=0; h=0
    for _ in range(sims):
        a,b=trial(**kw); r+=a; h+=b
    return r/sims, h/sims
for delta in (0.1,0.2,0.3):
    print("delta",delta,"rho=.3 (variant shifts 30% of 30 prompts; ~9 informative for target)")
    print("  1-stage random 25x4 :",power(delta=delta,rho=.3,one_stage=(25,4))[0])
    print("  1-stage random 10x10:",power(delta=delta,rho=.3,one_stage=(10,10))[0])
    p,h=power(delta=delta,rho=.3,M=30,n1=2,k=6,n2=10)
    print("  2-stage screen 30p x2 (60 resp) then top6 x10 (60 resp)=120 resp: power %.2f, mean informative among top6 %.2f (chance ~1.8)"%(p,h))
    p,h=power(delta=delta,rho=.3,M=20,n1=2,k=5,n2=12)
    print("  2-stage screen 20p x2 (40 resp) then top5 x12 (60 resp)=100 resp: power %.2f, informative %.2f (chance 1.5)"%(p,h))
