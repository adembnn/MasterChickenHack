from statistics import NormalDist
import random, math
Z=NormalDist()
# 1. calibration from MET table 8 (N=250, L=50, token space, Hamming), power -> delta0
tab={'Llama-3 8B':(1.00,.93,.79),'Llama-3.1 8B':(.62,1.0,.08),'Llama-3 70B':(.97,1.0,1.0),'Llama-3.1 70B':(.54,1.0,1.0),'Mistral 7B':(.72,.89,.30),'OLMo 7B':(.47,.99,.36),'Gemma-2 9B':(.43,.12,.06),'Phi-3':(.75,.63,.24)}
def delta(p): 
    p=min(max(p,0.06),0.995); return Z.inv_cdf(p)+Z.inv_cdf(0.95)
a=0.68  # exponent from Table 11 : 0.30 -> 0.96 for 5x N
print("exponent check", math.log((Z.inv_cdf(.96)+1.645)/(Z.inv_cdf(.30)+1.645))/math.log(5))
print("int8 / nf4 / wm: N needed for 80% power  (from N=250 power), and power at N=100,  exponent a=.68")
for name,(wm,nf,i8) in tab.items():
    row=[]
    for p in (i8,nf,wm):
        d0=delta(p); need=250*(Z.inv_cdf(.8)+1.645)/d0 if d0>0 else 1e9
        need=250*((Z.inv_cdf(.8)+1.645)/d0)**(1/a)
        p100=Z.cdf(d0*(100/250)**a-1.645)
        row.append("N80=%d p100=%.2f"%(need,p100))
    print(name,"| int8",row[0],"| nf4",row[1],"| wm",row[2])
# 2. required noncentrality lambda for 80% power chi2_df, alpha .05 (MC)
random.seed(1)
def chi2_noncentral_samples(df,lam,n=4000):
    mu=math.sqrt(lam)
    out=[]
    for _ in range(n):
        s=(random.gauss(0,1)+mu)**2+sum(random.gauss(0,1)**2 for _ in range(df-1))
        out.append(s)
    return out
for df in (1,3,5,10,20,50,100):
    null=sorted(chi2_noncentral_samples(df,0,6000)); crit=null[int(.95*len(null))]
    lo,hi=0,200
    for _ in range(14):
        mid=(lo+hi)/2
        s=chi2_noncentral_samples(df,mid,2500)
        pw=sum(x>crit for x in s)/len(s)
        if pw<.8: lo=mid
        else: hi=mid
    print("df",df,"crit %.1f"%crit,"lambda80 %.1f"%hi)
# 3. Mann-Whitney MDE for n per group
for n in (30,50,100,200,500):
    sd=math.sqrt((2*n+1)/(12*n*n)); auc=.5+(1.96+.8416)*sd  # two-sided .05 80%
    d=Z.inv_cdf(auc)*math.sqrt(2)
    print("n/group",n,"AUC_min %.3f"%auc,"d_min %.2f sd"%d,"KLseq_min(d^2*2) %.3f"%(2*d*d*0.25*2/2))
