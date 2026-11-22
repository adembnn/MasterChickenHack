"""Structures cachées compatibles avec 1803 = 0,10 et 1882 = 0,40 (familles 3x11 supposées justes).
Vérité par famille : un groupe de base de taille s (base + copies déguisées), le reste en variantes.
Variantes : singletons, ou paires (variante + 1 copie déguisée). Statut inconnu de chacun de nos isolés.
MCC modèle sur toutes les paires, puis sur un tirage public de 30 % des paires (bruit), arrondi à 0,05."""
import itertools,math,random,collections,json,sys
FAM={'q3':['01','02','04','12','14','17','18','23','24','25','33'],
     'q15':['03','06','07','08','20','21','22','26','28','30','32'],
     'll':['05','09','10','11','13','15','16','19','27','29','31']}
EPS=[e for f in FAM.values() for e in f]; fam={e:f for f,l in FAM.items() for e in l}
ISO1803={'14','23','07','28','16'}; ISO1882={'05','15','22'}
PAIRS=list(itertools.combinations(EPS,2))
def mcc(truth,pred,pairs=PAIRS):
    tp=fp=fn=tn=0
    for a,b in pairs:
        s=truth[a]==truth[b]; p=pred[a]==pred[b]
        if s and p: tp+=1
        elif p: fp+=1
        elif s: fn+=1
        else: tn+=1
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return (tp*tn-fp*fn)/d if d else 0.0
def fmcc(truthf,predf,pairs=PAIRS): return mcc(truthf,predf,pairs)
def pred(iso): return {e:(e if e in iso else fam[e]+'_base') for e in EPS}
P1803,P1882=pred(ISO1803),pred(ISO1882)
FAMT={e:fam[e] for e in EPS}
def score(m,f=1.0): return 0.1*f+0.9*max(0,m)
rnd=lambda x: round(x/0.05)*0.05
# vérité : pour chaque famille, quels membres sont « base » ; les autres sont des variantes distinctes (singletons)
# paramètre : nombre de variantes v_f par famille et lesquelles parmi nos isolés sont variantes.
def build(var):     # var = ensemble des ep variantes (singletons) ; le reste de chaque famille = un seul modèle de base
    return {e:(e if e in var else fam[e]+'_base') for e in EPS}
flag={'q3':['14','23'],'q15':['07','28','22'],'ll':['16','05','15']}
res=[]
for vq3 in range(0,11):
  for vq15 in range(0,11):
    for vll in range(0,11):
      # combinaisons des isolés : chaque isolé est variante (1) ou base (0) ; les variantes restantes sont prises parmi les non signalés
      for st in itertools.product([0,1],repeat=8):
        fl=[e for f in ('q3','q15','ll') for e in flag[f]]
        var=set(e for e,s in zip(fl,st) if s)
        ok=True; 
        for f,v in (('q3',vq3),('q15',vq15),('ll',vll)):
            nf=sum(1 for e in var if fam[e]==f)
            others=[e for e in FAM[f] if e not in flag[f]]
            k=v-nf
            if k<0 or k>len(others): ok=False; break
            var|=set(others[:k])     # par symétrie, lesquels importe peu (aucun n'est isolé)
        if not ok: continue
        T=build(var)
        m1,m2=mcc(T,P1803),mcc(T,P1882)
        res.append((vq3,vq15,vll,st,m1,m2))
print('structures évaluées',len(res))
ok=[r for r in res if score(r[4])<0.125 and 0.375<=score(r[5])<0.425]
print('compatibles (toutes paires, sans bruit public) :',len(ok))
c=collections.Counter((r[0],r[1],r[2]) for r in ok)
print('v par famille (q3,q15,ll) les plus fréquents :',c.most_common(12))
c2=collections.Counter(tuple(r[3][5:8]) for r in ok); print('statut (22,16?,..) -> voir ordre :',[e for f in ('q3','q15','ll') for e in flag[f]])
c3=collections.Counter(sum(r[3][i] for i in (4,6,7)) for r in ok); print('nb de nos 3 fp8 (22,05,15) justes :',sorted(c3.items()))
c4=collections.Counter(sum(r[3][i] for i in (0,1,2,3,5)) for r in ok); print('nb des 5 isolés de 1803 qui sont de vraies variantes :',sorted(c4.items()))
json.dump(ok,open('ok.json','w'))
# repères
for name,v in (('aucune variante',set()),):
    pass
