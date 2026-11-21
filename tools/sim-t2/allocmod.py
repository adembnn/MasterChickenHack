import random, math, time
random.seed(7)
def mmd_u(c,d,n):   # binary answers, c,d = # of '1' among n each; unbiased U-stat of MMD^2 exact kernel
    ca=[n-c,c]; da=[n-d,d]
    if n<2: return 0.0
    return sum(x*(x-1) for x in ca)/(n*(n-1))+sum(x*(x-1) for x in da)/(n*(n-1))-2*sum(x*y for x,y in zip(ca,da))/(n*n)
def stat(cs,ds,n): return sum(mmd_u(c,d,n) for c,d in zip(cs,ds))
def perm_p(cs,ds,n,obs,B=150):
    cnt=0
    for _ in range(B):
        tc=[];td=[]
        for c,d in zip(cs,ds):
            pool=[1]*(c+d)+[0]*(2*n-c-d); random.shuffle(pool)
            tc.append(sum(pool[:n])); td.append(sum(pool[n:]))
        if stat(tc,td,n)>=obs: cnt+=1
    return (cnt+1)/(B+1)
def run(m,n,rho,delta,sims=150,B=150,sd_nuis=0.0):
    """m prompts x n reps; prompt prob p~U(.15,.85); fraction rho of prompts shifted by +-delta; sd_nuis = random shift sd on all prompts (nuisance). returns rejection rate."""
    rej=0
    for _ in range(sims):
        cs=[];ds=[]
        for x in range(m):
            p=random.uniform(.15,.85); q=p
            if random.random()<rho: q=min(.97,max(.03,p+random.choice((-1,1))*delta))
            if sd_nuis>0: q=min(.97,max(.03,q+random.gauss(0,sd_nuis)))
            cs.append(sum(random.random()<p for _ in range(n))); ds.append(sum(random.random()<q for _ in range(n)))
        obs=stat(cs,ds,n)
        if perm_p(cs,ds,n,obs,B)<=0.05: rej+=1
    return rej/sims
