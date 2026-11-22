import math, itertools, collections
truth={"01":"S","03":"S","05":"S","06":"S","08":"S","02":"fp8","09":"L0.3","04":"L0.7","07":"L1.5"}
E=sorted(truth); pairs=list(itertools.combinations(range(9),2)); T=[truth[E[a]]==truth[E[b]] for a,b in pairs]
def partitions(n):
    def rec(i,lab,k):
        if i==n: yield tuple(lab); return
        for c in range(k+1):
            lab.append(c); yield from rec(i+1,lab,max(k,c+1)); lab.pop()
    yield from rec(0,[],0)
def mcc(lab):
    tp=fp=fn=tn=0
    for (a,b),t in zip(pairs,T):
        g=lab[a]==lab[b]; tp+=t and g; fp+=(not t) and g; fn+=t and not g; tn+=(not t) and not g
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (max(0.0,(tp*tn-fp*fn)/d) if d else 0.0),(tp,fp,fn,tn)
vals=collections.defaultdict(list)
for lab in partitions(9):
    m,c=mcc(lab); vals[round(m,4)].append((c,lab))
print("partitions:",sum(len(v) for v in vals.values()),"valeurs distinctes du MCC modele:",len(vals))
def desc(lab):
    g=collections.defaultdict(list)
    for e,l in zip(E,lab): g[l].append(e+"("+truth[e]+")")
    return " | ".join(",".join(x) for x in g.values() if len(x)>1) or "tout isole"
print("\nvaleurs >= 0,45 et nombre de partitions :")
for v in sorted(vals,reverse=True):
    if v>=0.45: print(f"  {v:.4f}  x{len(vals[v]):4d}  ex: {desc(vals[v][0][1])}  TP,FP,FN,TN={vals[v][0][0]}")
for target,name in ((2/3,"2/3"),(1/3,"1/3"),(13/18,"13/18"),(5/18,"5/18"),(1/9,"1/9")):
    near=sorted(vals,key=lambda v:abs(v-target))[:3]
    print(f"\nplus proches de {name}={target:.4f} :", [(v,len(vals[v])) for v in near])
    for v in near[:2]:
        seen=set()
        for c,lab in vals[v]:
            if c not in seen:
                seen.add(c); print(f"   {v:.4f} TP,FP,FN,TN={c} : {desc(lab)}")
            if len(seen)>=3: break
# effet d'un arrondi a 0,05 du score public 0,1*1 + 0,9*MCC
print("\nscore = 0,1 + 0,9 x MCC arrondi a 0,05, repartition sur toutes les partitions :")
cnt=collections.Counter()
for v,l in vals.items(): cnt[round(round((0.1+0.9*v)/0.05)*0.05,2)]+=len(l)
print(sorted(cnt.items()))
bar=[("parfait",1.0),("fp8 fusionne",0.7338),("1 copie ratee",0.7211),("copies en 2 paquets 3+2",0.5694),("fp8+LoRA0,3 fusionnes",0.5241),("2 copies ratees",0.4862),("fp8+L0,3+L0,7 fusionnes",0.3315),("3 copies ratees",0.2708)]
print("\nbareme -> score public si famille juste et arrondi a 0,05 :")
for n,v in bar: print(f"  {n:28s} MCC {v:.3f} -> brut {0.1+0.9*v:.3f} -> arrondi {round(round((0.1+0.9*v)/0.05)*0.05,2):.2f}")
