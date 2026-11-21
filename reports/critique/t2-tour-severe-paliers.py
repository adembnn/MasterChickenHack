"""T2 tour severe : MCC modele sous des structures cachees plausibles (b copies + 11-b variantes par famille, 3 familles).
Rejouer : python t2-tour-severe-paliers.py (aucune requete). Score = 0,1 + 0,9 * MCC modele si familles justes ; toutes les paires, pas le tirage public de 30 %."""
import itertools, math
def mcc(truth, guess):
    tp=fp=fn=tn=0
    for fam in truth:
        t=truth[fam]; g=guess[fam]
        for a,b in itertools.combinations(range(len(t)),2):
            T=t[a]==t[b]; G=g[a]==g[b]
            tp+=T and G; fp+=(not T) and G; fn+=T and not G; tn+=(not T) and not G
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return max(0,(tp*tn-fp*fn)/d) if d else 0.0
def fam(b,v): return ['S']*b+[f'V{i}' for i in range(v)]
def score(m): return 0.1+0.9*m
for b in (4,5,6,7,8):
    v=11-b; T={k:fam(b,v) for k in range(3)}
    res={}
    res['tout fusionne']=mcc(T,{k:['S']*11 for k in range(3)})
    res['tout isole']=mcc(T,{k:list(range(11)) for k in range(3)})
    # 1 variant correct per family isolated, rest merged
    for n in range(1,v+1):
        g=['S']*b+[f'V{i}' if i<n else 'S' for i in range(v)]
        res[f'{n} var/fam trouvees']=mcc(T,{k:g for k in range(3)})
    # 3 variants in total (1 per family)= same as n=1
    # perfect variants but 1 copy isolated per family
    g=['S']*(b-1)+['X']+[f'V{i}' for i in range(v)]
    res['variantes ok + 1 copie isolee/fam']=mcc(T,{k:g for k in range(3)})
    g=['S']*(b-2)+['X','Y']+[f'V{i}' for i in range(v)]
    res['variantes ok + 2 copies isolees/fam']=mcc(T,{k:g for k in range(3)})
    # variants all lumped into one 'variant' group
    g=['S']*b+['V']*v
    res['variantes en 1 seul groupe']=mcc(T,{k:g for k in range(3)})
    # LoRA 0.3 merged with base only (1 var missed)
    # 1 variant found in only one family (like 0.40?)
    g1=['S']*b+['V0']+['S']*(v-1)
    res['1 var trouvee dans 1 seule fam']=mcc(T,{0:g1,1:['S']*11,2:['S']*11})
    print(f'base={b} variantes={v}')
    for k,m in res.items(): print(f'   {k:40s} MCCmod={m:.3f} score={score(m):.3f}')

# --- scenarios 1882 / 1803 / regroupements ---
for b in (5,6,7,8):
    v=11-b; T={k:fam(b,v) for k in range(3)}
    S=['S']*11
    # 1882: 2 correct in fam0, 1 in fam1, 0 fam2
    g0=['S']*b+['V0','V1']+['S']*(v-2); g1=['S']*b+['V0']+['S']*(v-1)
    a=mcc(T,{0:g0,1:g1,2:S})
    # 1882 with one of the 3 being a copy
    g0b=['S']*(b-1)+['X']+['V0']+['S']*(v-1)
    c=mcc(T,{0:g0b,1:g1,2:S})
    # 1.00-team minus LoRA0.3 merged each fam
    g=['S']*b+[f'V{i}' for i in range(v-1)]+['S']
    d=mcc(T,{k:g for k in range(3)})
    # LoRA 0.3 and fp8 merged each fam
    g=['S']*b+[f'V{i}' for i in range(v-2)]+['S','S']
    e=mcc(T,{k:g for k in range(3)})
    print(b, f'1882 3 justes {score(a):.3f} | 1882 2 justes+1 copie {score(c):.3f} | tout sauf LoRA0.3 {score(d):.3f} | sauf LoRA0.3+fp8 {score(e):.3f}')

# 1803: 2 isolated in fam0, 2 in fam1, 1 in fam2. how many true variants max for score<0.125?
import itertools
for b in (6,7):
    v=11-b; T={k:fam(b,v) for k in range(3)}
    for k0,k1,k2 in itertools.product(range(3),range(3),range(2)):
        def g(n_iso,k):
            # k true variants isolated, n_iso-k copies isolated
            lab=['S']*(b-(n_iso-k))+[f'X{i}' for i in range(n_iso-k)]+[f'V{i}' if i<k else 'S' for i in range(v)]
            return lab
        s=score(mcc(T,{0:g(2,k0),1:g(2,k1),2:g(1,k2)}))
        if s<0.125: print(b,'vraies variantes parmi les 5 isoles:',k0+k1+k2,(k0,k1,k2),round(s,3))
    # ep-05 & ep-15 same fp8 model (fp8 + disguise) isolated separately vs together
    T2={0:['S']*b+['F','F']+[f'V{i}' for i in range(v-2)],1:fam(b,v),2:fam(b,v)}
    sep={0:['S']*b+['F1','F2']+['S']*(v-2),1:['S']*b+['V0']+['S']*(v-1),2:['S']*11}
    tog={0:['S']*b+['F','F']+['S']*(v-2),1:sep[1],2:sep[2]}
    print(b,'si 05 et 15 = meme fp8: separes',round(score(mcc(T2,sep)),3),'ensemble',round(score(mcc(T2,tog)),3))
    # grouping two distinct LoRA (17,18) in one label vs separately (both true variants)
    base1882={0:['S']*b+['V0','V1']+['S']*(v-2),1:['S']*b+['V0']+['S']*(v-1)}
    grp=dict(base1882); grp[2]=['S']*b+['L','L']+['S']*(v-2)
    sepl=dict(base1882); sepl[2]=['S']*b+['L1','L2']+['S']*(v-2)
    print(b,'17+18 vraies LoRA distinctes: groupees',round(score(mcc(T,grp)),3),'separees',round(score(mcc(T,sepl)),3))
