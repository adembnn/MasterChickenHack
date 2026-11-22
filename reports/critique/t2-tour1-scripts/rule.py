import json,sys,random,statistics as st,itertools,math,importlib.util
sys.argv_saved=list(sys.argv)
from calibmod import *
truth={'ref-01':'S','ref-02':'T1','ref-03':'S','ref-04':'T3','ref-05':'S','ref-06':'S','ref-07':'T4','ref-08':'S','ref-09':'T2'}
def mcc(pred):
    tp=fp=fn=tn=0
    for a,b in itertools.combinations(sorted(truth),2):
        s=truth[a]==truth[b]; p=pred[a]==pred[b]
        tp+=s and p; fp+=p and not s; fn+=s and not p; tn+=(not s) and (not p)
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/d if d else 0.0
def partition(G,zthr=2.0,B=200,rng=random.Random(0)):
    F={};Z={}
    for e,rs in G.items():
        F[e]=feats(rs); bs=[feats([rng.choice(rs) for _ in rs]) for _ in range(B)]
        se=st.pstdev([b['dT'] for b in bs if b['dT']==b['dT']]) or 1e-9
        Z[e]=F[e]['dT']/se
    flagged=[e for e in G if abs(Z[e])>zthr]
    pred={e:('S' if e not in flagged else e) for e in G}
    # merge flagged with similar T_log (within 2 pooled SE ~0.025)
    return pred,Z,F
if __name__=='__main__':
    R=json.load(open(sys.argv_saved[1])); use={'reading','repeat'}
    E=sorted({r['endpoint'] for r in R})
    G={e:[r for r in R if r['endpoint']==e and r['kind'] in use] for e in E}
    pred,Z,F=partition(G)
    print('z(dT):',{e:round(Z[e],1) for e in E}); print('pred',pred,'MCC',round(mcc(pred),3))
    # subsampling robustness: k responses per endpoint
    rng=random.Random(5)
    for k in (8,12,16,20):
        ms=[];flag=dict.fromkeys(E,0)
        for t in range(40):
            Gs={e:rng.sample(G[e],min(k,len(G[e]))) for e in E}
            p,z,_=partition(Gs,B=60,rng=rng); ms.append(mcc(p))
            for e in E: flag[e]+=p[e]!='S'
        print(f'k={k}: MCC moyen {st.mean(ms):.3f} (min {min(ms):.2f}, max {max(ms):.2f}) ; taux de signalement',{e:round(flag[e]/40,2) for e in E})
