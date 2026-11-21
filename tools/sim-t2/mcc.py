import itertools, math, random
def mcc_from(tp,fp,fn,tn):
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return max(0,(tp*tn-fp*fn)/d) if d else 0.0
def score(truth,pred,eps):
    # truth/pred: dict ep -> (family, model)
    fam=[0,0,0,0]; mod=[0,0,0,0]
    for a,b in itertools.combinations(eps,2):
        tf=truth[a][0]==truth[b][0]; pf=pred[a][0]==pred[b][0]
        fam[0]+=tf and pf; fam[1]+=(not tf) and pf; fam[2]+=tf and not pf; fam[3]+=(not tf) and not pf
        if tf:
            tm=truth[a]==truth[b]; pm=pred[a]==pred[b]
            mod[0]+=tm and pm; mod[1]+=(not tm) and pm; mod[2]+=tm and not pm; mod[3]+=(not tm) and not pm
    mf=mcc_from(*fam); mm=mcc_from(*mod)
    return 0.1*mf+0.9*mm, mf, mm, mod
# layout: 3 families x 11 endpoints ; per family groups: S x5, T1 x2 (fp8, one watermarked), T2 x1 (lora .3), T3 x2, T4 x1
eps=[]; truth={}
groups=[('S',5),('Q',2),('L3',1),('L7',2),('L15',1)]
i=0
for f in 'ABC':
    for g,k in groups:
        for _ in range(k):
            i+=1; e='ep-%02d'%i; eps.append(e); truth[e]=(f,f+g)
print("n endpoints",len(eps))
def variant(fn):
    pred={e:fn(e,truth[e]) for e in eps}; return pred
def sc(name,fn):
    s=score(truth,variant(fn),eps); print("%-62s score %.3f  MCCfam %.3f MCCmodel %.3f  (TP,FP,FN,TN)=%s"%(name,s[0],s[1],s[2],s[3]))
sc("perfect",lambda e,t:t)
sc("fp8 (Q) and lora.3 (L3) merged into S",lambda e,t:(t[0],t[0]+'S') if t[1][1:] in ('Q','L3') else t)
sc("also lora.7 merged into S (only L15 separated)",lambda e,t:(t[0],t[0]+'S') if t[1][1:] in ('Q','L3','L7') else t)
sc("all in-family same (family only)",lambda e,t:(t[0],t[0]+'1'))
sc("all different (every ep alone, family correct)",lambda e,t:(t[0],e))
sc("S class split in two (temp+wm endpoints apart), rest ok",lambda e,t:(t[0],t[0]+'S2') if (t[1][1:]=='S' and int(e[3:])%5 in (0,1)) else t)
sc("1 S endpoint wrongly isolated per family",lambda e,t:(t[0],e) if (t[1][1:]=='S' and int(e[3:])%11==1) else t)
sc("1 S endpoint isolated in family A only",lambda e,t:(t[0],e) if e=='ep-01' else t)
sc("Q,L3 merged + S not split + L7 pairs ok  (same as 2nd)",lambda e,t:(t[0],t[0]+'S') if t[1][1:] in ('Q','L3') else t)
# random pair error model, via noisy clustering: flip endpoint assignments randomly
random.seed(3)
def noisy(pm_misassign):
    def fn(e,t):
        if random.random()<pm_misassign:
            return (t[0],t[0]+random.choice(['S','Q','L3','L7','L15','X'+e]))
        return t
    return fn
for p in (0.05,0.1,0.2,0.3):
    vals=[]
    for _ in range(300):
        pred=variant(noisy(p)); vals.append(score(truth,pred,eps)[0])
    vals.sort(); print("each endpoint mis-assigned to wrong/new model w.p. %.2f -> score mean %.3f  (p10 %.3f, p90 %.3f)"%(p,sum(vals)/300,vals[30],vals[270]))
# threshold optimum formula
def pstar(pi,recall,prec_ratio=1.0):
    # near-perfect: .5 ; here Q=P (predicted positives = P), TP=recall*P
    P=pi; TP=recall*P; Q=P*prec_ratio
    return P+(TP-P*Q)*(1-2*Q)/(2*Q*(1-Q))
for pi in (0.1,0.2,0.3):
    print("prevalence",pi,[ "recall %.1f -> p* %.2f"%(r,pstar(pi,r)) for r in (1.0,0.8,0.6,0.4)])
