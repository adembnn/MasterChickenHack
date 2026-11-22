"""T3: read-only class-subspace vs head-readout entanglement diagnostic.

Motivation: SCOPE (Cai et al., Aug 2026, arXiv 2608.02058)
warns that any FIXED feature-direction erasure can damage shared retained
capabilities. This is a simplified PRINCIPAL-ANGLE RISK AUDIT, not SCOPE
itself and not an EAC submission. T3 is INSTANCE unlearning, not whole-class
unlearning; forget-set pseudo-label prototypes can overlap the entire
class signal, so a high score is a warning, not a reason to project.

Inputs already obtainable from unmodified EAC timm ViT:
 forget_features.npy [N,768] = model.forward_head(model.forward_features(x),
                                                   pre_logits=True)
 head_weight.npy [100,768] = orig.head.weight
 optional head_bias.npy [100], retain_features.npy [M,768] = heldout proxy
 NO true retained labels required.

Usage:
 python t3_scope_overlap_diagnostic.py --smoke
 python t3_scope_overlap_diagnostic.py --forget-features forget_feats.npy \
   --head head_weight.npy --bias head_bias.npy --retain-features proxy.npy \
   --ranks 1,2,4,8 --out overlap.json

This script CANNOT construct a valid conditional-gate submission; EAC demands
unaltered ViT architecture. Do not apply output projection as a real submission.
No EAC API, no model weight modification, no H200 jobs performed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np


EPS=1e-12


def _softmax(x):
    x=x-np.max(x,axis=1,keepdims=True)
    a=np.exp(x)
    return a/np.maximum(a.sum(1,keepdims=True), EPS)


def _jsd(p,q):
    mid=.5*(p+q)
    return .5*((p*np.log(np.maximum(p,EPS)/np.maximum(mid,EPS))).sum(1)
               +(q*np.log(np.maximum(q,EPS)/np.maximum(mid,EPS))).sum(1))


def _basis(features, labels, min_count):
    n,d=features.shape
    classes=np.unique(labels)
    rows=[]
    sizes=[]
    for y in classes:
        f=features[labels==y]
        if len(f)>=min_count:
            rows.append(f.mean(0))
            sizes.append(len(f))
    if len(rows)<3:
        raise ValueError('Need >=3 pseudo-classes with enough examples')
    means=np.stack(rows)
    # Center class centroids uniformly to prevent large pseudo-class dominance;
    # singular vectors are CLASS DISCRIMINANTS, NOT instance-memorization axes.
    means=means-means.mean(axis=0,keepdims=True)
    _,s,vh=np.linalg.svd(means,full_matrices=False)
    available=int(np.sum(s>max(s[0]*1e-7,EPS)))
    if available<1:
        raise ValueError('No nonzero class discriminant singular vectors')
    return vh[:available].T, np.array(sizes), s[:available]


def _pred_metrics(z,w,b,u):
    plain=z@w.T+b[None,:]
    zp=z-(z@u)@u.T
    edited=zp@w.T+b[None,:]
    p=_softmax(plain/2.)
    q=_softmax(edited/2.)
    return {
        'top1_disagreement':float((plain.argmax(1)!=edited.argmax(1)).mean()),
        'mean_JS_T2':float(_jsd(p,q).mean()),
        'mean_KL_orig_to_projection_T2':float(
            (p*(np.log(np.maximum(p,EPS))-np.log(np.maximum(q,EPS)))).sum(1).mean()),
        'median_l2_feature_change':float(np.median(np.linalg.norm(z-zp,axis=1))),
    }


def analyze(features,head,bias=None,retain_features=None,ranks=(1,2,4,8),
            min_count=3,seed=22):
    f=np.asarray(features,dtype=np.float64)
    w=np.asarray(head,dtype=np.float64)
    if f.ndim!=2 or w.ndim!=2 or f.shape[1]!=w.shape[1]:
        raise ValueError('Forget features [N,D], head [C,D] required')
    n,d=f.shape
    c=w.shape[0]
    if n<20 or c<3 or min_count<1:
        raise ValueError('Insufficient examples/classes')
    b=np.zeros(c,dtype=np.float64) if bias is None else np.asarray(bias,dtype=np.float64)
    if b.shape!=(c,):
        raise ValueError('Invalid head bias shape')
    proxy=None if retain_features is None else np.asarray(retain_features,dtype=np.float64)
    if proxy is not None and (proxy.ndim!=2 or proxy.shape[1]!=d or len(proxy)<5):
        raise ValueError('Invalid proxy retain features [M,D]')
    if not (np.isfinite(f).all() and np.isfinite(w).all() and np.isfinite(b).all()
            and (proxy is None or np.isfinite(proxy).all())):
        raise ValueError('Nonfinite inputs')
    y=(f@w.T+b[None,:]).argmax(1)
    U,counts,s=_basis(f,y,min_count)
    # Uniform head readout subspace, independent of class prior.
    wc=w-w.mean(0,keepdims=True)
    _,hs,hvh=np.linalg.svd(wc,full_matrices=False)
    hr=int(np.sum(hs>max(hs[0]*1e-7,EPS)))
    H=hvh[:hr].T
    rng=np.random.default_rng(seed)
    report={}
    for rank in ranks:
        k=int(rank)
        if k<1 or k>U.shape[1]:
            continue
        u=U[:,:k]
        cos=np.linalg.svd(H.T@u,compute_uv=False)
        cos2=(cos**2)
        frac=float(np.linalg.norm(wc@u)**2/np.maximum(np.linalg.norm(wc)**2,EPS))
        # Random projector control: preserve same rank and dimension;
        # helps detect if a 'special' forgetting projection is worse than chance.
        random=[]
        for j in range(3):
            Q,_=np.linalg.qr(rng.normal(size=(d,k)))
            random.append(_pred_metrics(f,w,b,Q)['mean_JS_T2'])
        val={
            'principal_cos2_min':float(cos2.min()),
            'principal_cos2_mean':float(cos2.mean()),
            'head_readout_energy_fraction_on_forget_class_axes':frac,
            'forget_class_pseudo_labels':int(len(counts)),
            'forget_disturbance':_pred_metrics(f,w,b,u),
            'forget_random_direction_JS_T2_mean':float(np.mean(random)),
        }
        if proxy is not None:
            val['proxy_retain_disturbance']=_pred_metrics(proxy,w,b,u)
        report[str(k)]=val
    if not report:
        raise ValueError('No ranks fit singular dimension')
    return {
        'WARNING':'Read-only, pseudo-class not instance membership. A high '
                  'overlap is NO-GO for blind fixed abliteration, a low '
                  'overlap is NOT evidence of EAC gold matching.',
        'n_forget':int(n),'n_proxy':None if proxy is None else int(len(proxy)),
        'feature_dim':int(d),'centroid_rank':int(U.shape[1]),
        'ranks':report
    }


def smoke():
    rng=np.random.default_rng(22)
    d,c,n=24,6,180
    w=rng.normal(size=(c,d))
    # Build diverse class-structured features so head alignment is measurable.
    y=rng.integers(0,c,size=n)
    x=w[y]+.2*rng.normal(size=(n,d))
    proxy=rng.normal(size=(120,d))
    a=analyze(x,w,retain_features=proxy,ranks=(1,2,4))
    assert a['centroid_rank']>=4
    for v in a['ranks'].values():
        assert 0<=v['head_readout_energy_fraction_on_forget_class_axes']<=1+1e-8
        assert 0<=v['principal_cos2_mean']<=1+1e-8
        assert np.isfinite(v['forget_disturbance']['mean_JS_T2'])
        assert 'proxy_retain_disturbance' in v
    print('PASS synthetic: class subspace, principal angles, readout overlap, proxy JS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forget-features')
    p.add_argument('--head')
    p.add_argument('--bias')
    p.add_argument('--retain-features')
    p.add_argument('--ranks',default='1,2,4,8')
    p.add_argument('--min-count',type=int,default=3)
    p.add_argument('--out',default='scope_overlap.json')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:
        smoke()
        return
    if not a.forget_features or not a.head:
        p.error('--forget-features and --head required')
    data=analyze(np.load(a.forget_features,allow_pickle=False),
                 np.load(a.head,allow_pickle=False),
                 None if not a.bias else np.load(a.bias,allow_pickle=False),
                 None if not a.retain_features else np.load(a.retain_features,allow_pickle=False),
                 tuple(int(v) for v in a.ranks.split(',')),a.min_count)
    Path(a.out).write_text(json.dumps(data,indent=2),encoding='utf8')
    print(json.dumps(data['ranks'],indent=2))
    print('Read-only diagnostic; NO model edit, NO H200 score, NO submit.')


if __name__=='__main__':
    main()
