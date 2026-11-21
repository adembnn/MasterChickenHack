import json,sys,math,statistics as st,itertools,random
R=json.load(open(sys.argv[1]))
truth={'ref-01':'S','ref-02':'T1','ref-03':'S','ref-04':'T3','ref-05':'S','ref-06':'S','ref-07':'T4','ref-08':'S','ref-09':'T2'}
kind={'ref-01':'clair','ref-02':'fp8','ref-03':'sys','ref-04':'LoRA.7','ref-05':'persona','ref-06':'KGW','ref-07':'LoRA1.5','ref-08':'T1.0','ref-09':'LoRA.3'}
TS=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4]
E=sorted(truth)
use=lambda r: r['kind'] in ('reading','repeat')
def tstar(rs):
    # T where sum(nll_T - H_T)=0 by linear interpolation
    g=[sum(r[f'nll_{T}']-r[f'H_{T}'] for r in rs) for T in TS]
    for i in range(len(TS)-1):
        if g[i]>0>=g[i+1] or g[i]<=0<g[i+1]:
            return TS[i]+(TS[i+1]-TS[i])*g[i]/(g[i]-g[i+1])
    return float('nan')
def feats(rs):
    n=sum(r['n'] for r in rs)
    f={}
    f['ntok']=n/len(rs)
    f['nll1']=sum(r['nll_1.0'] for r in rs)/n
    f['nll08']=sum(r['nll_0.8'] for r in rs)/n
    f['gap08']=sum(r['nll_0.8']-r['H_0.8'] for r in rs)/n
    f['argmax']=sum(r['argmax'] for r in rs)/n
    f['T*']=tstar(rs)
    # ECE at T=1 of argmax hit vs maxp (only meaningful at T*)
    lps=[x for r in rs for x in r['tok_lp1']]
    f['p_lt-6']=sum(x<-6 for x in lps)/len(lps)   # very unlikely tokens
    f['p_lt-10']=sum(x<-10 for x in lps)/len(lps)
    return f
F={e:feats([r for r in R if r['endpoint']==e and use(r)]) for e in E}
keys=list(F['ref-01'])
print('ep      vérité  '+' '.join(f'{k:>8}' for k in keys))
for e in E: print(e,kind[e].ljust(8),' '.join(f'{F[e][k]:8.4f}' for k in keys))
# bootstrap SE over responses for gap08 and T*
rng=random.Random(0)
print('\nbootstrap (réponses) : moyenne ± SE')
for e in E:
    rs=[r for r in R if r['endpoint']==e and use(r)]
    bs={k:[] for k in ('T*','gap08','nll1','p_lt-6')}
    for _ in range(400):
        s=[rng.choice(rs) for _ in rs]; f=feats(s)
        for k in bs: bs[k].append(f[k])
    print(e,kind[e].ljust(8),' '.join(f'{k}={F[e][k]:.3f}±{st.pstdev(v):.3f}' for k,v in bs.items()))
pairs=list(itertools.combinations(E,2))
def auc(pos,neg): return sum((p<q)+0.5*(p==q) for p in pos for q in neg)/(len(pos)*len(neg))
print('\nAUC distance |f_i-f_j| (meme modele = petite) :')
for k in keys:
    pos=[abs(F[a][k]-F[b][k]) for a,b in pairs if truth[a]==truth[b]]
    neg=[abs(F[a][k]-F[b][k]) for a,b in pairs if truth[a]!=truth[b]]
    print(f'{k:>8} {auc(pos,neg):.2f}')
