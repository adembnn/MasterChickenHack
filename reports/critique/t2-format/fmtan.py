import json,sys,random,math,statistics as st
import numpy as np
D=json.load(open('fmt_scores.json')); S=D['scores']; NB=len(D['bank'])
print('Jfp8',round(D['Jfp8'],4),'s_eq',D['s_eq'],'J(s_eq)',round(D['Jl'][str(D['s_eq'])],4),'s_half',D['s_half'],'J(s_half)',round(D['Jl'][str(D['s_half'])],4))
mean_lp=lambda row:[s/n if n else 0.0 for s,n in row]
REF=[np.array(mean_lp(r)) for r in S['ref']]
mu=np.array([r.mean() for r in REF]); sd=np.array([r.std(ddof=1) for r in REF])
W={w:[np.array(mean_lp(r)) for r in S[w]] for w in S if w!='ref'}
print('Décalage moyen de log p_base par token (monde - réf) / écart type de la réf :')
for w in W: print(f'  {w:9} {np.mean([ (W[w][j].mean()-mu[j])/sd[j] for j in range(NB)]):+.3f}  (écart type entre prompts {np.std([ (W[w][j].mean()-mu[j])/sd[j] for j in range(NB)]):.2f})')
rng=np.random.default_rng(0)
def stats(w,pidx,draws):
    """x = réponses du monde w aux prompts pidx (draws tirages chacun) ; renvoie (Z, p_CvM)"""
    xs=[];us=[];zs=[]
    for j in pidx:
        pick=rng.choice(len(W[w][j]),size=draws,replace=True)
        for k in pick:
            x=W[w][j][k]; xs.append(x); zs.append((x-mu[j])/sd[j])
            us.append((1+np.sum(REF[j]<=x))/(len(REF[j])+1))
    Z=np.sum(zs)/math.sqrt(len(zs)); u=np.sort(np.array(us)); n=len(u); p=1/(12*n)+float(np.sum((u-(2*np.arange(1,n+1)-1)/(2*n))**2))  # statistique W2 de Cramér-von Mises
    return Z,p
half1=list(range(0,NB,2)); half2=list(range(1,NB,2))   # sélection / évaluation
# B_sel : 5 prompts choisis sur half1 par |décalage| moyen fp8 + loraEq (les deux tournent contre la base)
sc=[]
for j in half1:
    sc.append((abs((W['fp8'][j].mean()-mu[j])/sd[j])+abs((W['loraEq'][j].mean()-mu[j])/sd[j]),j))
BSEL=[j for _,j in sorted(sc,reverse=True)[:5]]
print('B_sel prompts :',[D['bank'][j][:44] for j in BSEL])
formats={'A 25x1 (25 prompts distincts)':lambda w:stats(w,rng.choice(half2,25,replace=False),1),
         'B 5x5 (5 prompts au hasard)':lambda w:stats(w,rng.choice(half2,5,replace=False),5),
         'B 5x5 (5 prompts choisis)':lambda w:stats(w,BSEL,5)}
N=1500
res={}
for fn,f in formats.items():
    res[fn]={w:[f(w) for _ in range(N)] for w in W}
def auc(pos,neg): 
    pos=np.array(pos);neg=np.array(neg); return float((pos[:,None]>neg[None,:]).mean()+0.5*(pos[:,None]==neg[None,:]).mean())
VAR=['fp8','loraEq','loraHalf']; COP=['base','sysH','sysA','T1.0']
for stat,idx,tr in (('Stouffer |Z| (décalage de moyenne)',0,lambda v:abs(v)),('RUT : Cramér-von Mises (−log p)',1,lambda v:v)):
    print('\n###',stat)
    print('| format | '+' | '.join(f'{w}' for w in VAR+COP[1:])+' |   (taux de rejet à seuil = quantile 99 % du monde « base »)')
    for fn in formats:
        base_s=sorted(tr(r[idx]) for r in res[fn]['base']); thr=base_s[int(0.99*N)]
        row=[]
        for w in VAR+COP[1:]: row.append(np.mean([tr(r[idx])>thr for r in res[fn][w]]))
        print(f'| {fn} | '+' | '.join(f'{x:.2f}' for x in row)+' |')
    print('AUC variantes (fp8, loraEq) contre copies (base, sysH, sysA, T1.0) :')
    for fn in formats:
        print(f'  {fn}: fp8 vs copies '+', '.join(f'{c}:{auc([tr(r[idx]) for r in res[fn]["fp8"]],[tr(r[idx]) for r in res[fn][c]]):.2f}' for c in COP)+' | loraEq vs copies '+', '.join(f'{c}:{auc([tr(r[idx]) for r in res[fn]["loraEq"]],[tr(r[idx]) for r in res[fn][c]]):.2f}' for c in COP))
