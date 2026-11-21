import random, math, itertools
random.seed(21)
D=100
def mcc_from(tp,fp,fn,tn):
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return max(0,(tp*tn-fp*fn)/d) if d else 0.0
types=[('S',5,0),('Q',2,40),('L3',1,70),('L7',2,150),('L15',1,600)]
nuis=[80,120,200,100]
def unit():
    v=[random.gauss(0,1) for _ in range(D)]; s=math.sqrt(sum(x*x for x in v)); return [x/s for x in v]
def jacobi(A,sweeps=30):
    n=len(A); V=[[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    A=[r[:] for r in A]
    for _ in range(sweeps):
        off=sum(A[i][j]**2 for i in range(n) for j in range(i+1,n))
        if off<1e-9: break
        for p in range(n):
            for q in range(p+1,n):
                if abs(A[p][q])<1e-12: continue
                th=(A[q][q]-A[p][p])/(2*A[p][q]); t=(1 if th>=0 else -1)/(abs(th)+math.sqrt(th*th+1)); c=1/math.sqrt(t*t+1); s=t*c
                for k in range(n):
                    akp,akq=A[k][p],A[k][q]; A[k][p]=c*akp-s*akq; A[k][q]=s*akp+c*akq
                for k in range(n):
                    apk,aqk=A[p][k],A[q][k]; A[p][k]=c*apk-s*aqk; A[q][k]=s*apk+c*aqk
                for k in range(n):
                    vkp,vkq=V[k][p],V[k][q]; V[k][p]=c*vkp-s*vkq; V[k][q]=s*vkp+c*vkq
    return [A[i][i] for i in range(n)],V
def avg_link(T,t):
    m=len(T); cl=[[i] for i in range(m)]
    def dist(a,b): return sum(T[i][j] for i in a for j in b)/(len(a)*len(b))
    while len(cl)>1:
        best=None
        for x in range(len(cl)):
            for y in range(x+1,len(cl)):
                d=dist(cl[x],cl[y])
                if best is None or d<best[0]: best=(d,x,y)
        if best[0]>t: break
        _,x,y=best; cl[x]+=cl[y]; del cl[y]
    lab=[0]*m
    for c,l in enumerate(cl):
        for i in l: lab[i]=c
    return lab
def mcc_lab(truth,lab):
    tp=fp=fn=tn=0
    for i,j in itertools.combinations(range(len(truth)),2):
        a=truth[i]==truth[j]; b=lab[i]==lab[j]
        tp+=a and b; fp+=(not a) and b; fn+=a and not b; tn+=(not a) and not b
    return mcc_from(tp,fp,fn,tn)

def trial(mode,share=True):
    vd={g:unit() for g,_,_ in types}
    nd=[unit() for _ in range(4)]
    pts=[];truth=[];srows=[]
    nj=0
    for g,k,d2 in types:
        for j in range(k):
            mu=[math.sqrt(d2)*u for u in vd[g]]
            idx=None
            if g=='S' and j>0: idx=j-1
            elif g=='Q' and j==1: idx=0
            elif g=='L7' and j==1: idx=1
            if idx is not None:
                d=nd[idx] if share else unit()
                mu=[m+math.sqrt(nuis[idx])*u for m,u in zip(mu,d)]
            p=[m+random.gauss(0,1) for m in mu]
            pts.append(p); truth.append(g)
            if g=='S': srows.append(p)
    n=len(pts)
    if mode!='raw':
        # estimate nuisance subspace from the S group (rows centred)
        mS=[sum(r[k] for r in srows)/len(srows) for k in range(D)]
        X=[[r[k]-mS[k] for k in range(D)] for r in srows]
        m=len(X)
        G=[[sum(a*b for a,b in zip(X[i],X[j])) for j in range(m)] for i in range(m)]
        ev,V=jacobi(G)
        order=sorted(range(m),key=lambda i:-ev[i])[:4]
        basis=[]
        for c in order:
            v=[sum(V[i][c]*X[i][k] for i in range(m)) for k in range(D)]
            nrm=math.sqrt(sum(x*x for x in v)); basis.append([x/nrm for x in v])
        if mode=='oracle_dirs':
            basis=nd
        new=[]
        for p in pts:
            q=p[:]
            for b in basis:
                dot=sum(x*y for x,y in zip(q,b)); q=[x-dot*y for x,y in zip(q,b)]
            new.append(q)
        pts=new
    Dm=[[sum((a-b)**2 for a,b in zip(pts[i],pts[j])) for j in range(n)] for i in range(n)]
    vals=sorted(set(Dm[i][j] for i in range(n) for j in range(i+1,n)))
    qs=[vals[int(q*(len(vals)-1))] for q in (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6)]
    res=[mcc_lab(truth,avg_link(Dm,t)) for t in qs]
    return max(res)
for share in (True,False):
    for mode in ('raw','project_S_nuisance','oracle_dirs'):
        R=40; random.seed(5)
        v=[trial(mode,share) for _ in range(R)]
        print("shared nuisance directions" if share else "nuisance directions NOT shared", mode, round(sum(v)/R,3))
