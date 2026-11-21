import random, math, itertools, sys
sys.path.insert(0,'..'+'/sim')
from mcc import mcc_from
random.seed(11)
D=100
types=[('S',5,0),('Q',2,40),('L3',1,70),('L7',2,150),('L15',1,600)]
nuis=[80,120,200,100]   # squared norms of nuisance displacement (wm, sp, persona, temp)
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
def trial(nu_scale):
    vd={g:unit() for g,_,_ in types}
    pts=[];truth=[];nj=0
    for g,k,d2 in types:
        for j in range(k):
            mu=[math.sqrt(d2)*u for u in vd[g]]
            if (g=='S' and j>0) or (g=='Q' and j==1) or (g=='L7' and j==1):
                s2=nuis[nj%4]*nu_scale; nj+=1; d=unit(); mu=[m+math.sqrt(s2)*u for m,u in zip(mu,d)]
            pts.append([m+random.gauss(0,1) for m in mu]); truth.append(g)
    n=len(pts)
    def dmat(P):
        return [[sum((a-b)**2 for a,b in zip(P[i],P[j])) for j in range(n)] for i in range(n)]
    raw=dmat(pts)
    mean=[sum(p[k] for p in pts)/n for k in range(D)]
    X=[[p[k]-mean[k] for k in range(D)] for p in pts]
    G=[[sum(a*b for a,b in zip(X[i],X[j])) for j in range(n)] for i in range(n)]
    ev,V=jacobi(G)
    order=sorted(range(n),key=lambda i:-ev[i])
    res={}
    for k in (6,10,14):
        # scores = V[:,c]*sqrt(ev)
        Y=[[V[i][c]*math.sqrt(max(ev[c],0)) for c in order[:k]] for i in range(n)]
        res[k]=[[sum((a-b)**2 for a,b in zip(Y[i],Y[j])) for j in range(n)] for i in range(n)]
    # profile similarity (second-order) on raw distances, leave-pair-out
    prof=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i<j:
                prof[i][j]=prof[j][i]=sum((raw[i][k]-raw[j][k])**2 for k in range(n) if k!=i and k!=j)/(n-2)
    return truth,{'raw':raw,'pca6':res[6],'pca10':res[10],'pca14':res[14],'profile':prof}
def sweep(Dm,truth):
    vals=sorted(set(Dm[i][j] for i in range(len(truth)) for j in range(i+1,len(truth))))
    best=0;
    qs=[vals[int(q*(len(vals)-1))] for q in (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6)]
    out=[]
    for t in qs: out.append(mcc_lab(truth,avg_link(Dm,t)))
    return max(out),out
for nu in (0.0,1.0):
    R=40; acc={}
    for _ in range(R):
        truth,Ds=trial(nu)
        for k,Dm in Ds.items():
            b,_=sweep(Dm,truth); acc.setdefault(k,[]).append(b)
    print("nuisance scale",nu,{k:round(sum(v)/len(v),3) for k,v in acc.items()}); sys.stdout.flush()
