"""Puissance de la sonde préfixe + max_tokens=1 : N tirages, LLR = sum log p_fp8(x) - log p_base(x)."""
import json,sys,math,random,statistics as st
D=json.load(open(sys.argv[1])); T=D['T']; P=D['positions']; K=D['kept']
q=lambda xs,a: sorted(xs)[int(a*(len(xs)-1))]
keys=[k for k in P[0] if k.startswith('J_')]
print(f'{len(P)} positions, T={T}. Quantiles de J (base vs lecteur) :')
for k in keys:
    xs=[p[k] for p in P]; print(f'  {k:8} med={q(xs,.5):.4f} p90={q(xs,.9):.4f} p99={q(xs,.99):.4f} p99.9={q(xs,.999):.4f} max={max(xs):.3f} somme={sum(xs):.1f}')
sysk=[k for k in keys if k.startswith('J_sys') or k=='J_pers']
def lse(v): m=max(v); return m+math.log(sum(math.exp(x-m) for x in v))
def retemper(lp,Tn): v=[x*T/Tn for x in lp]; z=lse(v); return [x-z for x in v]
def norm(lp): z=lse(lp); return [x-z for x in lp]
def kgw(c,delta,gamma=0.25):
    rng=random.Random(c['prev']*7919+13); lp=c['lp']['base']
    return norm([x+(delta/T if rng.random()<gamma else 0) for x in lp])   # biais delta sur les logits, puis T
def sample(lp,rng):
    u=rng.random(); a=0
    for i,x in enumerate(lp):
        a+=math.exp(x)
        if u<=a: return i
    return len(lp)-1
def select(score,filt,n):
    c=[k for k in K if filt(k)]; c.sort(key=score,reverse=True); return c[:n]
sc_min=lambda k:min(k['J_w8'],k['J_w8a8'])
robust=lambda k: k['J_f32']<sc_min(k)/4 and all(k.get(s,1e9)<sc_min(k)/2 for s in sysk)
print('\npositions gardées (J>=0,02) :',len(K),'; robustes (bruit fp32 < J/4, prompts système < J/2) :',sum(robust(k) for k in K))
def worlds(c):
    w={'base':norm(c['lp']['base']),'f32':norm(c['lp']['f32']),'w8':norm(c['lp']['w8']),'w8a8':norm(c['lp']['w8a8'])}
    for s in ('sysH','sysA','pers'):
        if s in c['lp']: w[s]=norm(c['lp'][s])
    w['kgw2']=kgw(c,2.0); w['T1.0']=retemper(c['lp']['base'],1.0); w['w8a8_T1.0']=retemper(c['lp']['w8a8'],1.0)
    return w
import bisect,itertools
THR=0.0
def run(sel,reps,reader='w8a8',trials=2000,rng=random.Random(0)):
    W=[worlds(c) for c in sel]; out={}
    for wname in W[0]:
        if any(wname not in w for w in W): continue
        pre=[]
        for c,w in zip(sel,W):
            if wname.endswith('T1.0'): rb,rf=retemper(c['lp']['base'],1.0),retemper(c['lp'][reader],1.0)
            else: rb,rf=norm(c['lp']['base']),norm(c['lp'][reader])
            cum=list(itertools.accumulate(math.exp(x) for x in w[wname]))
            pre.append((cum,[a-b for a,b in zip(rf,rb)]))
        Ls=[]
        for _ in range(trials):
            L=0.0
            for cum,llr in pre:
                for _ in range(reps): L+=llr[min(bisect.bisect_left(cum,rng.random()*cum[-1]),len(llr)-1)]
            Ls.append(L)
        out[wname]=(st.mean(Ls),st.pstdev(Ls),sum(l>THR for l in Ls)/trials)
    return out

NULLS=['base','f32','sysH','sysA','pers','kgw2','T1.0']
def summary(npos,reps,trials=2000):
    sel=select(sc_min,robust,npos)
    if len(sel)<npos: return None
    r8a8=run(sel,reps,'w8a8',trials); r8=run(sel,reps,'w8',trials)
    rec=min(r8a8['w8a8'][2],r8a8['w8'][2],r8a8['w8a8_T1.0'][2])      # rappel au pire des 3 vérités fp8 (lecteur W8A8)
    fa=max(r8a8[w][2] for w in NULLS if w in r8a8)                    # fausse alerte au pire des 7 vérités « copie »
    fam=st.mean(r8a8[w][2] for w in NULLS if w in r8a8)
    out={'n':npos,'reps':reps,'J_moy':round(st.mean(sc_min(c) for c in sel),3),'rappel_pire':rec,'FA_pire':fa,'FA_moy':round(fam,3)}
    for pi in (1/11,2/11):
        out[f'precision_pi{round(pi*11)}/11']=round(rec*pi/(rec*pi+fa*(1-pi)),2) if rec+fa>0 else None
    # W8 contre W8A8 : signe de sum log p_w8a8 - log p_w8
    W=[worlds(c) for c in sel]; import bisect,itertools
    acc=[]
    for truth in ('w8','w8a8'):
        pre=[(list(itertools.accumulate(math.exp(x) for x in w[truth])),[a-b for a,b in zip(norm(c['lp']['w8a8']),norm(c['lp']['w8']))]) for c,w in zip(sel,W)]
        ok=0; rng=random.Random(3)
        for _ in range(trials):
            L=sum(llr[min(bisect.bisect_left(cum,rng.random()*cum[-1]),len(llr)-1)] for cum,llr in pre for _ in range(reps))
            ok+= (L>0)==(truth=='w8a8')
        acc.append(ok/trials)
    out['W8_vs_W8A8_juste']=round(sum(acc)/2,2)
    return out
for THR in (0.0,3.0):
    for npos,reps in ((25,1),(12,2),(5,5)):
        print('seuil',THR,json.dumps(summary(npos,reps)))
