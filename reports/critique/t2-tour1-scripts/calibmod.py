import json,sys,random,statistics as st,itertools
kind={'ref-01':'clair','ref-02':'fp8','ref-03':'sys','ref-04':'LoRA.7','ref-05':'persona','ref-06':'KGW','ref-07':'LoRA1.5','ref-08':'T1.0','ref-09':'LoRA.3'}
TS=[0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.2,1.4]

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
