# Toy synthetique pur Python : valeur de l'information pour le score forget (G1 seulement).
import random, math, statistics as st
random.seed(7)
C=100; N=1200; K=4
perm=list(range(C)); random.shuffle(perm)
NB=lambda y:[perm[(perm.index(y)+j)%C] for j in (1,2,3,4,5)]
nbm={y:NB(y) for y in range(C)}
def lse(z):
    m=max(z); return m+math.log(sum(math.exp(v-m) for v in z))
def logsm(z,T=1.0):
    zz=[v/T for v in z]; l=lse(zz); return [v-l for v in zz]
def sm(z,T=1.0): return [math.exp(v) for v in logsm(z,T)]
def build(A=3.4,B=1.6,Tg=1.5,sm_=0.8,rho_o=0.35):
    imgs=[]
    for i in range(N):
        y=random.randrange(C); e=random.gauss(0,1)
        w=[0.0]*C
        w[y]=A+B*e
        for c in nbm[y]: w[c]=random.gauss(0.6,1.0)
        golds=[]
        for k in range(K):
            golds.append([Tg*(w[c]+random.gauss(0,sm_)) for c in range(C)])
        orig=[Tg*rho_o*w[c]+random.gauss(0,0.5) for c in range(C)]; orig[y]=13+random.gauss(0,1)
        test=[Tg*(w[c]+random.gauss(0,sm_)) for c in range(C)]
        imgs.append(dict(y=y,e=e,w=w,golds=golds,orig=orig,test=test))
    return imgs
def ce(z,y): return -logsm(z)[y]
def stats(L): return st.mean(L), st.pstdev(L)
def kl(p,q): return sum(a*(math.log(a)-math.log(b)) for a,b in zip(p,q) if a>1e-12)
def score(imgs,subz):
    gm=0;gs=0
    for k in range(K):
        m,s=stats([ce(im['golds'][k],im['y']) for im in imgs]); gm+=m/K; gs+=s/K
    pg=[[sum(x)/K for x in zip(*[sm(g,2.0) for g in im['golds']])] for im in imgs]
    def d(zs): 
        m,s=stats([ce(z,im['y']) for z,im in zip(zs,imgs)]); return abs(m-gm)+abs(s-gs),m,s
    do=d([im['orig'] for im in imgs])[0]; ds,m,s=d(subz)
    ms=1-min(1,ds/do)
    ko=st.mean(kl(p,sm(im['orig'],2.0)) for p,im in zip(pg,imgs))
    ks=st.mean(kl(p,sm(z,2.0)) for p,z in zip(pg,subz))
    pr=1-min(1,ks/ko)
    return ms,pr,ms*pr,(gm,gs),(m,s),(ko,ks)
imgs=build()
acc=st.mean(1.0 if max(range(C),key=lambda c:g[c])==im['y'] else 0.0 for im in imgs for g in im['golds'])
print("gold indiv acc",round(acc,3),"gold mean,std CE",[round(v,2) for v in stats([ce(g,im['y']) for im in imgs for g in im['golds']])])
o=score(imgs,[im['orig'] for im in imgs]); print("orig",[round(v,3) for v in o[:3]],"KL(pg||orig)",round(o[5][0],2))
print("ceiling indep gold_test", [round(v,3) for v in score(imgs,[im['test'] for im in imgs])[:3]], "KLs",round(score(imgs,[im['test'] for im in imgs])[5][1],2))
# oracle ensemble-logit: z = mean of gold logits (soup)
print("oracle mean-logit(soup of K golds)",[round(v,3) for v in score(imgs,[[sum(x)/K for x in zip(*im['golds'])] for im in imgs])[:3]])
print("oracle z=2log pg",[round(v,3) for v in score(imgs,[[2*math.log(a) for a in [sum(x)/K for x in zip(*[sm(g,2.0) for g in im['golds']])]] for im in imgs])[:3]])
# global temperature
for tau in (2,4,6,8,12):
    r=score(imgs,[[v/tau for v in im['orig']] for im in imgs]); print("globalT",tau,[round(v,3) for v in r[:3]],"m,s",[round(v,2) for v in r[4]])
# moment-matched construction
def mm(imgs,eps,r_h,r_w,qlo=0.12,dsc=0.04):
    out=[]
    hs=[(-(im['e'])*r_h+math.sqrt(max(0,1-r_h**2))*random.gauss(0,1)) for im in imgs]  # high = hard
    order=sorted(range(len(imgs)),key=lambda i:-hs[i]); nh=int(eps*len(imgs)); hard=set(order[:nh])
    rank={i:r for r,i in enumerate(order)}
    for i,im in enumerate(imgs):
        y=im['y']; wr=sorted([c for c in range(C) if c!=y],key=lambda c:-im['w'][c])[0]
        guess=wr if random.random()<r_w else random.choice(nbm[y]+[random.randrange(C)])
        if guess==y: guess=(y+1)%C
        q=[0.0]*C
        if i in hard:
            py=max(0.01,qlo*math.exp(random.gauss(0,0.6)))
            pw=min(0.95-py,0.45+0.2*random.random()+0.0) ; 
            q[y]=py; q[guess]=pw
        else:
            frac=rank[i]/len(imgs)   # plus facile -> plus confiant
            py=1-dsc*(1+3*(1-(1-frac)))-0.0
            py=max(0.5,min(0.995,1-dsc*(1+4*frac)))
            q[y]=py; q[guess]=0.4*(1-py)
        rest=1-sum(q); nb=nbm[y]
        wts=[ (3.0 if c in nb else 1.0) for c in range(C)]
        s=sum(wts[c] for c in range(C) if q[c]==0.0)
        for c in range(C):
            if q[c]==0.0: q[c]=rest*wts[c]/s
        out.append([math.log(max(v,1e-9)) for v in q])
    return out

def lognorm_quantiles(n,m,sd):
    # lognormale de moyenne m, ecart-type sd ; quantiles deterministes
    s2=math.log(1+(sd/m)**2); mu=math.log(m)-s2/2; sg=math.sqrt(s2)
    nd=st.NormalDist(); return [math.exp(mu+sg*nd.inv_cdf((k+0.5)/n)) for k in range(n)]
def qa(imgs,m,sd,r_h,r_w,sharp=1.0):
    n=len(imgs); L=lognorm_quantiles(n,m,sd)             # pertes croissantes
    hs=[(im['e']*(-r_h)+math.sqrt(max(0,1-r_h**2))*random.gauss(0,1)) for im in imgs]
    order=sorted(range(n),key=lambda i:hs[i])            # facile -> dur
    out=[None]*n
    for rank,i in enumerate(order):
        im=imgs[i]; y=im['y']; l=L[rank]; qy=math.exp(-l)
        wr=sorted([c for c in range(C) if c!=y],key=lambda c:-im['w'][c])[0]
        guess=wr if random.random()<r_w else random.choice(nbm[y]+[random.randrange(C)])
        if guess==y: guess=(y+1)%C
        q=[0.0]*C; q[y]=qy
        rest=1-qy
        # si qy<0.5, le guess doit depasser qy pour etre une erreur
        pg_=min(rest*0.8, max(rest*0.5, qy*1.4)) if qy<0.5 else rest*0.5
        q[guess]=pg_; rem=rest-pg_
        wts=[(3.0 if c in nbm[y] else 1.0) for c in range(C)]
        ss=sum(wts[c] for c in range(C) if q[c]==0.0)
        for c in range(C):
            if q[c]==0.0: q[c]=rem*wts[c]/ss
        out[i]=[math.log(max(v,1e-12)) for v in q]
    return out
gm=o[3][0]; gs=o[3][1]
print("m*,s*",round(gm,2),round(gs,2))
print("--- quantile-assignment ; ligne = (r_h,r_w) ; colonnes = erreur relative sur (m*,s*) : -30% -15% 0 +15% +30%  -> score (ms/pred)")
for r_h,r_w in ((0.0,0.1),(0.5,0.1),(0.5,0.5),(0.8,0.5),(0.8,0.9)):
    row=[]
    for f in (0.7,0.85,1.0,1.15,1.3):
        random.seed(5); r=score(imgs,qa(imgs,gm*f,gs*f,r_h,r_w)); row.append("%.2f(%.2f/%.2f)"%(r[2],r[0],r[1]))
    print((r_h,r_w)," ".join(row))
print("--- erreur sur m* seul / s* seul (r_h .5, r_w .5)")
for fm,fs in ((0.7,1.0),(1.3,1.0),(1.0,0.7),(1.0,1.3)):
    random.seed(5); r=score(imgs,qa(imgs,gm*fm,gs*fs,0.5,0.5)); print(fm,fs,"%.2f(%.2f/%.2f)"%(r[2],r[0],r[1]))
print("--- ablation de sharp: aucune; compare a global T4 =",[round(v,3) for v in score(imgs,[[v/4 for v in im['orig']] for im in imgs])[:3]])
