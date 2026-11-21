import json,sys,random,statistics as st,itertools
R=json.load(open(sys.argv[1])); sel=sys.argv[2] if len(sys.argv)>2 else 'reading,repeat'
kind={'ref-01':'clair','ref-02':'fp8','ref-03':'sys','ref-04':'LoRA.7','ref-05':'persona','ref-06':'KGW','ref-07':'LoRA1.5','ref-08':'T1.0','ref-09':'LoRA.3'}
TS=[0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.2,1.4]
E=sorted({r['endpoint'] for r in R})
def root(g):
    for i in range(len(TS)-1):
        if (g[i]>0)!=(g[i+1]>0): return TS[i]+(TS[i+1]-TS[i])*g[i]/(g[i]-g[i+1])
    return float('nan')
def interp(vals,T):
    for i in range(len(TS)-1):
        if TS[i]<=T<=TS[i+1]:
            w=(T-TS[i])/(TS[i+1]-TS[i]); return vals[i]*(1-w)+vals[i+1]*w
    return float('nan')
def feats(rs):
    n=sum(r['n'] for r in rs)
    t1=root([sum(r[f'nll_{T}']-r[f'H_{T}'] for r in rs) for T in TS])       # log identity
    t2=root([sum(r[f'sq_{T}']-r[f'px_{T}'] for r in rs) for T in TS])       # collision identity
    # ECE at grid point nearest t1
    Tn=min(TS,key=lambda T:abs(T-t1)) if t1==t1 else 0.8
    bins=[[0,0,0] for _ in range(10)]
    for r in rs:
        for i,(c,s,h) in enumerate(r[f'ece_{Tn}']): bins[i][0]+=c;bins[i][1]+=s;bins[i][2]+=h
    N=sum(b[0] for b in bins); ece=sum(abs(b[1]-b[2]) for b in bins)/N
    hi=bins[9]; # tokens where model is >90% sure: miss rate excess
    miss9=(hi[1]-hi[2])/max(1,hi[0])
    return {'T_log':t1,'T_col':t2,'dT':t1-t2,'ece':ece,'miss9':miss9,'ntok':n/len(rs)}
use=set(sel.split(','))
G={e:[r for r in R if r['endpoint']==e and r['kind'] in use] for e in E}
F={e:feats(G[e]) for e in E}
rng=random.Random(1); SE={}
for e in E:
    bs=[feats([rng.choice(G[e]) for _ in G[e]]) for _ in range(300)]
    SE[e]={k:st.pstdev([b[k] for b in bs]) for k in F[e]}
keys=list(F[E[0]])
for e in E: print(e,kind.get(e,'').ljust(8),'  '.join(f'{k}={F[e][k]:.3f}±{SE[e][k]:.3f}' for k in keys))
json.dump({'F':F,'SE':SE},open(sys.argv[1]+'.feat','w'))
