"""Quels etiquetages 'naturels' donnent 0,70 / 0,75 / 0,35 / 0,10 au score public T2 arrondi a 0,05 ?
Hypotheses : 33 endpoints en familles facon corrige (1 base en clair + copies deguisees + variantes en clair),
public = 30 % des paires tirees au hasard (enonce : "a part of the pairs"). Formule de score_practice.py."""
import math, itertools, random, collections
def mcc(tp,fp,fn,tn):
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return max(0.0,(tp*tn-fp*fn)/d) if d else 0.0
def build(struct):
    """struct = liste de familles ; famille = (deguisements des copies, variantes)."""
    eps=[]
    for f,(dis,var) in enumerate(struct):
        eps.append((f,"base","clair"))
        for d in dis: eps.append((f,"base",d))
        for v in var: eps.append((f,v,"clair"))
    return eps
def label(eps,motif):
    lab=[]
    for i,(f,w,d) in enumerate(eps):
        if motif=="tout fusionne par famille": m="B"
        elif motif=="tout isole": m=f"x{i}"
        elif motif=="parfait": m=w
        elif motif=="quantifie fusionne avec la base": m="base" if w in("fp8","int8") else w
        elif motif=="copie filigranee isolee": m=f"x{i}" if d=="filigrane" else w
        elif motif=="copie T1,0 isolee": m=f"x{i}" if d=="T1" else w
        elif motif=="filigrane et T1,0 isoles": m=f"x{i}" if d in("filigrane","T1") else w
        elif motif=="quantifie fusionne + filigrane isole": m=f"x{i}" if d=="filigrane" else ("base" if w in("fp8","int8") else w)
        elif motif=="quantifie + LoRA 0,3 fusionnes": m="base" if w in("fp8","int8","L0.3") else w
        elif motif=="seules LoRA fortes separees": m=w if w in("L0.7","L1.5") else "base"
        elif motif=="toutes les copies deguisees isolees": m=f"x{i}" if d!="clair" else w
        elif motif=="quantifie pris pour la base": m={"base":"Q","fp8":"base"}.get(w,w) if d=="clair" else w
        lab.append((f,m))
    return lab
def score(eps,lab,pairs):
    c=[0,0,0,0]; cf=[0,0,0,0]
    for a,b in pairs:
        tf=eps[a][0]==eps[b][0]; gf=lab[a][0]==lab[b][0]
        cf[0 if tf and gf else 1 if (not tf) and gf else 2 if tf and not gf else 3]+=1
        if tf:
            t=eps[a][1]==eps[b][1]; g=lab[a]==lab[b]
            c[0 if t and g else 1 if (not t) and g else 2 if t and not g else 3]+=1
    return 0.1*mcc(*cf)+0.9*mcc(*c), mcc(*c)
r05=lambda x: round(round(x/0.05)*0.05,2)
D4=["systeme","persona","filigrane","T1"]; V4=["fp8","L0.3","L0.7","L1.5"]
STRUCTS={
 "A: 3 familles x 9 (corrige) + 1 de 6":[(D4,V4)]*3+[(["systeme","filigrane"],["fp8","L0.7","L1.5"])],
 "B: 3 x 11 (4 copies, 6 variantes dont int8)":[(D4,V4+["int8","L1.0"])]*3,
 "C: 4 familles (9,8,8,8)":[(D4,V4)]+[(D4,["fp8","L0.3","L0.7"])]*3,
 "D: 3 x 11 (6 copies, 4 variantes)":[(D4+["systeme2","persona2"],V4)]*3,
 "E: 5 familles (7,7,7,6,6), 2-3 copies":[(["systeme","filigrane","T1"],["fp8","L0.3","L0.7"])]*3+[(["persona","T1"],["fp8","L0.7","L1.5"])]*2,
}
MOTIFS=["tout isole","tout fusionne par famille","parfait","quantifie fusionne avec la base","copie filigranee isolee","copie T1,0 isolee",
        "quantifie fusionne + filigrane isole","filigrane et T1,0 isoles","quantifie + LoRA 0,3 fusionnes","seules LoRA fortes separees",
        "quantifie pris pour la base","toutes les copies deguisees isolees"]
rng=random.Random(0)
print("| motif (familles justes) | "+" | ".join(k.split(":")[0] for k in STRUCTS)+" |   (score complet brut -> public arrondi le plus frequent, part)")
for mo in MOTIFS:
    row=[]
    for name,st in STRUCTS.items():
        eps=build(st); assert len(eps)==33,(name,len(eps))
        lab=label(eps,mo); allp=list(itertools.combinations(range(33),2))
        full,_=score(eps,lab,allp)
        cnt=collections.Counter()
        for _ in range(400):
            pub=rng.sample(allp,int(0.3*len(allp))); cnt[r05(score(eps,lab,pub)[0])]+=1
        top=cnt.most_common(2)
        row.append(f"{full:.2f} -> "+" / ".join(f"{v:.2f} ({n/4:.0f}%)" for v,n in top))
    print(f"| {mo} | "+" | ".join(row)+" |")
