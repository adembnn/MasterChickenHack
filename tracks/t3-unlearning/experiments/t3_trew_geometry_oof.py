"""T3: TREW-inspired class-geometry tilt, OOF teacher cross-validation ONLY.

Sources: Ebrahimpour-Boroojeny et al. ICML 2026,
https://proceedings.mlr.press/v306/ebrahimpour-boroojeny26a.html
This is a *new adaptation*, NOT a reproduction of TREW (class forgetting)
and NOT proof of EAC gold-model improvement.

Inputs:
  teacher_probs.npy [K,N,C], K even >=4: raw independent OOF ViT
      teacher predictions from the SAME temperature T=2, each normalized.
  head_weight.npy [C,D]: ORIGINAL 100-class classifier head weights.
  pseudo_labels.npy [N]: orig.argmax on forget samples (NOT true labels).
Uses TWO disjoint OOF teacher groups, and disjoint image groups for
hyperparameter selection vs evaluation. No gradient, no EAC API calls.

 python t3_trew_geometry_oof.py --teachers raw_teachers.npy \
   --head head_weight.npy --labels pseudo_labels.npy --out trew_cv.json
 python t3_trew_geometry_oof.py --smoke

If only mean-of-8 is available, DO NOT fabricate per-teacher predictions.
"""
import argparse, json
import numpy as np
EPS=1e-12


def conditional_tilt(p, y, class_similarity, beta):
    p=np.asarray(p,dtype=np.float64)
    y=np.asarray(y,dtype=np.int64)
    n,c=p.shape
    if class_similarity.shape!=(c,c) or y.shape!=(n,):
        raise ValueError('shapes do not align')
    if np.any(y<0) or np.any(y>=c):
        raise ValueError('invalid pseudo-labels')
    out=p.copy()
    # Fixed pseudo-label mass isolates the effect of wrong-class geometry.
    selected=class_similarity[y].copy()
    selected[np.arange(n),y]=0.
    selected-=selected.sum(1,keepdims=True)/max(1,c-1)
    selected[np.arange(n),y]=0.
    logw=np.log(np.maximum(p,EPS))+beta*selected
    logw[np.arange(n),y]=-1e9
    logw-=logw.max(1,keepdims=True)
    wrong=np.exp(logw)
    wrong[np.arange(n),y]=0.
    wrong/=np.maximum(wrong.sum(1,keepdims=True),EPS)
    orig_y=p[np.arange(n),y]
    out=(1-orig_y[:,None])*wrong
    out[np.arange(n),y]=orig_y
    return out


def kl_per_image(q,p):
    q=np.maximum(np.asarray(q,np.float64), EPS)
    p=np.maximum(np.asarray(p,np.float64), EPS)
    return np.sum(q*(np.log(q)-np.log(p)),axis=-1)


def class_sim(head):
    w=np.asarray(head,np.float64)
    w=w-w.mean(0,keepdims=True)
    w=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),EPS)
    return w@w.T


def evaluate(teachers,head,labels,seed=22,betas=(-2.,-1.,-.5,0.,.5,1.,2.)):
    t=np.asarray(teachers,dtype=np.float64)
    if t.ndim!=3 or t.shape[0]<4 or t.shape[0]%2:
        raise ValueError('teachers must be [K even>=4,N,C]')
    k,n,c=t.shape
    if n<40 or c<3 or len(labels)!=n or np.asarray(head).shape[0]!=c:
        raise ValueError('shape mismatch or insufficient images')
    if not np.isfinite(t).all() or (t<0).any() or not np.allclose(t.sum(2),1,atol=1e-4):
        raise ValueError('teachers must contain valid normalized probabilities at T2')
    S=class_sim(head)
    rng=np.random.default_rng(seed)
    order=rng.permutation(n)
    fit=order[:n//2];hold=order[n//2:]
    rng.shuffle(t) if False else None  # no in-place teacher permutation
    group1=t[:k//2].mean(0)
    group2=t[k//2:].mean(0)
    def avgkl(beta,sub):
        a=conditional_tilt(group1[sub],labels[sub],S,beta)
        b=conditional_tilt(group2[sub],labels[sub],S,beta)
        return .5*(kl_per_image(group2[sub],a)+kl_per_image(group1[sub],b))
    scored={float(beta):float(avgkl(beta,fit).mean()) for beta in betas}
    chosen=min(scored,key=lambda x:scored[x])
    base=avgkl(0.,hold)
    test=avgkl(chosen,hold)
    delta=base-test
    means=rng.choice(delta,(2000,len(delta)),replace=True).mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    return {'chosen_beta_from_fit_only':float(chosen),
            'fit_kl_by_beta':{str(k):v for k,v in scored.items()},
            'holdout_n':len(hold),
            'holdout_KL_beta0':float(base.mean()),
            'holdout_KL_chosen':float(test.mean()),
            'holdout_gain_beta0_minus_chosen':float(delta.mean()),
            'holdout_paired_bootstrap_CI95':[float(lo),float(hi)],
            'recommend_next_training':bool(chosen!=0. and lo>0.),
            'warning':'OOF teachers are PROXIES, not gold; no EAC score or control gate measured'}


def smoke():
    rng=np.random.default_rng(14)
    n,c,k=180,12,8
    head=rng.normal(size=(c,16))
    y=rng.integers(0,c,size=n)
    S=class_sim(head)
    raw=rng.normal(size=(n,c))
    raw-=raw.max(1,keepdims=True)
    base=np.exp(raw)
    base/=base.sum(1,keepdims=True)
    gold=conditional_tilt(base,y,S,1.)
    teachers=np.stack([rng.dirichlet(200*gold[i]+1.,size=k).T
                       for i in range(n)],axis=0).transpose(2,0,1)
    assert teachers.shape==(k,n,c)
    result=evaluate(teachers,head,y)
    assert np.isfinite(result['holdout_gain_beta0_minus_chosen'])
    assert result['holdout_n']==n//2
    assert np.allclose(conditional_tilt(base,y,S,0),base,atol=1e-9)
    print('PASS CPU synthetic: normalized tilted distributions; matched labels at beta0; CV OK')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('teachers','head','labels'):p.add_argument('--'+arg)
    p.add_argument('--out',default='trew_cv.json')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:
        smoke();return
    if not all([a.teachers,a.head,a.labels]):
        p.error('--teachers, --head and --labels are required')
    teachers=np.load(a.teachers,allow_pickle=False)
    head=np.load(a.head,allow_pickle=False)
    labels=np.load(a.labels,allow_pickle=False).astype(np.int64)
    result=evaluate(teachers,head,labels)
    with open(a.out,'w',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    print('NO new target trained, NO H200 EAC score, NO submit')


if __name__=='__main__':
    main()
