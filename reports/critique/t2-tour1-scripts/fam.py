import json,sys,glob,os
TS=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4]
def root(g,ts):
    for i in range(len(ts)-1):
        if (g[i]>0)!=(g[i+1]>0): return ts[i]+(ts[i+1]-ts[i])*g[i]/(g[i]-g[i+1])
    return float('nan')
pref=sys.argv[1]; files=sorted(glob.glob(pref+'_*.json'))
tab={}
for f in files:
    m=os.path.basename(f)[len(os.path.basename(pref))+1:-5]
    if m in('360','360_empty','SmolLM2-360M-Instruct') : continue
    R=json.load(open(f))
    for e in sorted({r['endpoint'] for r in R}):
        rs=[r for r in R if r['endpoint']==e and r['kind'] in('reading','repeat')]
        n=sum(r['n'] for r in rs)
        tl=root([sum(r[f'nll_{T}']-r[f'H_{T}'] for r in rs) for T in TS],TS)
        # min over T of NLL_T per token (best-temperature NLL)
        best=min(sum(r[f'nll_{T}'] for r in rs)/n for T in TS)
        tab.setdefault(e,{})[m]=(sum(r['nll_1.0'] for r in rs)/n,best,tl,sum(r['argmax'] for r in rs)/n)
ms=sorted({m for e in tab for m in tab[e]})
short=lambda m:m.replace('-Instruct','').replace('SmolLM','Sm').replace('Qwen2.5-','Q').replace('OLMo-2-0425-','OL').replace('OLMo-2-1124-','OL').replace('Mistral-','Mis').replace('Phi-3.5-mini','Phi').replace('-v0.3','')
print('NLL/token a T=1 (T* calibration entre parentheses) ; * = minimum')
print('ep     '+' '.join(f'{short(m):>13}' for m in ms))
for e in tab:
    v=tab[e]; mn=min(v[m][0] for m in v)
    print(e,' '.join((f"{v[m][0]:.2f}({v[m][2]:.2f})"+('*' if v[m][0]==mn else ' ')).rjust(13) if m in v else ' '*13 for m in ms))
