import sys, json, math, collections
sys.path.insert(0, "E:/EAC-lab/t2/libs")
from tokenizers import Tokenizer
tok=Tokenizer.from_file("tok_smol360.json")
truth={"ref-01":"clair","ref-02":"fp8","ref-03":"systeme","ref-04":"LoRA0.7","ref-05":"persona","ref-06":"filigrane","ref-07":"LoRA1.5","ref-08":"T1.0","ref-09":"LoRA0.3"}
uni=collections.defaultdict(collections.Counter); bi=collections.defaultdict(collections.Counter); L=collections.defaultdict(list)
for l in open(sys.argv[1],encoding="utf-8"):
    r=json.loads(l)
    if r["id"] in ("P000","P001"): continue           # sondes meta exclues
    ids=tok.encode(r["response"],add_special_tokens=False).ids
    uni[r["endpoint"]].update(ids); bi[r["endpoint"]].update(zip(ids[:-1],ids[1:])); L[r["endpoint"]].append(len(ids))
E=sorted(uni)
def chi2(counter):
    """chi2 par degre de liberte : endpoint e contre la somme des 8 autres, cases frequentes seulement."""
    out={}
    for e in E:
        a=counter[e]; b=collections.Counter()
        for o in E:
            if o!=e: b.update(counter[o])
        na,nb=sum(a.values()),sum(b.values()); x=0.0; k=0
        for t in set(a)|set(b):
            ca,cb=a.get(t,0),b.get(t,0)
            ea=(ca+cb)*na/(na+nb); eb=(ca+cb)*nb/(na+nb)
            if ea>=5:
                x+=(ca-ea)**2/ea+(cb-eb)**2/eb; k+=1
        out[e]=(x/max(1,k),k)
    return out
cu=chi2(uni); cb=chi2(bi)
import statistics
print("| ref | vérité | tokens/réponse (médiane) | types distincts / 1000 tokens | chi2/ddl unigrammes | chi2/ddl bigrammes |")
for e in E:
    n=sum(uni[e].values())
    print(f"| {e} | {truth[e]} | {statistics.median(L[e]):.0f} | {1000*len(uni[e])/n:.0f} | {cu[e][0]:.2f} ({cu[e][1]}) | {cb[e][0]:.2f} ({cb[e][1]}) |")
