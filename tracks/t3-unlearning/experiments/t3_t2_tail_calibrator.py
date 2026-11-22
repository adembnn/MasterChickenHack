#!/usr/bin/env python3
"""EAC T3: constrained T=2 tail distribution from held-out redistill teachers.

Input: original target probabilities q1[N,C], original pseudo-label yhat[N],
multiple OUT-OF-FOLD teacher probabilities or logits [N,C]. All row orders must
match forget_images.npz. The *per-image* original-class probability of q1 is
held fixed, so CE(yhat) moments of the TARGET are invariant. This does NOT
imply actual trained-model metrics or a positive EAC score.

For each teacher, construct p_teacher(T=2) = softmax(logits/2). To imitate
average gold reference, average p(T=2) ARITHMETICALLY across independent
teachers (rather than averaging logits or T=1 p and converting afterwards).
For each image form the conditional 99-class tail r = p2_wrong / sum(p2_wrong).
Blend r with the current q1's T=2 wrong-class tail, then invert temperature
exactly: q1_wrong ∝ r**2, q1_yhat is preserved.

Output is NPZ target tensors, NOT EAC model weights. Requires the existing
team trainer to consume q1 targets and enforce CE height; do not submit NPZ.
No SSH, model weights, contest/API calls, external datasets or GPU needed.

Usage on a connected GPU/CPU with target/teacher source files already present:
  python t3_t2_tail_calibrator.py --base cache/tgt_RDt24r_70_l1016.pt \
    --yhat cache/s1_prep.pt --teachers teacher0.pt teacher1.pt teacher2.pt \
    --out ~/runs/t3_t2_tail --betas 0,.2,.4,.6
"""
import argparse
import json
from pathlib import Path
import numpy as np

EPS = 1e-10


def raw_array(path, key, candidates):
    path = Path(path).expanduser()
    if path.suffix == '.npy':
        a = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            k = key or next((x for x in candidates if x in f.files), None)
            if k is None:
                raise ValueError(f'{path}: keys {f.files}; set --base-key/--teacher-key/--yhat-key')
            a = f[k]
    elif path.suffix in ('.pt', '.pth'):
        import torch
        obj = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(obj, dict):
            k = key or next((x for x in candidates if x in obj), None)
            if k is None:
                raise ValueError(f'{path}: keys {list(obj.keys())}; use an explicit key')
            obj = obj[k]
        a = obj.detach().cpu().numpy() if hasattr(obj, 'detach') else np.asarray(obj)
    else:
        raise ValueError(f'unsupported input {path}')
    return np.asarray(a)


def probabilities(a, kind):
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)):
        raise ValueError('teacher array must be finite [N,C]')
    if kind == 'logits':
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
    else:
        if np.any(x < 0) or not np.allclose(x.sum(1), 1., rtol=.0003, atol=.0003):
            raise ValueError('teacher probabilities must be nonnegative and row-normalized')
    return x / np.maximum(x.sum(1, keepdims=True), EPS)


def at_temperature_two(p):
    # For logits z and p1=softmax(z), sqrt(p1) normalized = softmax(z/2).
    s = np.sqrt(np.maximum(p, 0))
    return s / np.maximum(s.sum(1, keepdims=True), EPS)


def conditional_tail(p2, y):
    r = p2.copy()
    r[np.arange(len(y)), y] = 0
    s = r.sum(1, keepdims=True)
    if np.any(s < EPS):
        raise ValueError('teacher has ~zero wrong-class mass at T2; increase precision')
    r /= s
    return r


def blend_target(base_q1, yhat, tail_base, tail_teacher, beta):
    qy = base_q1[np.arange(len(yhat)), yhat].copy()
    r = (1-beta) * tail_base + beta * tail_teacher
    r[np.arange(len(yhat)), yhat] = 0
    tail = np.square(r)
    tail /= np.maximum(tail.sum(1, keepdims=True), EPS)
    q1 = (1-qy[:, None]) * tail
    q1[np.arange(len(yhat)), yhat] = qy
    q2 = at_temperature_two(q1)
    if np.max(np.abs(q1.sum(1)-1)) > 1e-6:
        raise AssertionError('target not normalized')
    if not np.allclose(q1[np.arange(len(yhat)), yhat], qy, atol=1e-9):
        raise AssertionError('target CE modified')
    return q1, q2


def tail_kl(ref, candidate):
    return (ref * (np.log(np.maximum(ref, EPS)) - np.log(np.maximum(candidate, EPS)))).sum(1)


def screen_cv(base_q1, yhat, teacher_p2s, betas):
    # Leave one independently trained cross-fit ensemble out at a time.
    # Important: validates *conditional tail* only, NOT gold-class p2 or EAC.
    base = conditional_tail(at_temperature_two(base_q1), yhat)
    mask = base_q1[np.arange(len(yhat)), yhat] < .1
    if not mask.any():
        raise ValueError('target has no low-true-class-probability flip images')
    scores = {}
    for beta in betas:
        per_partition = []
        for held in range(len(teacher_p2s)):
            tr = np.mean([p for j,p in enumerate(teacher_p2s) if j != held], axis=0)
            tail_tr = conditional_tail(tr, yhat)
            blended = (1-beta)*base + beta*tail_tr
            held_tail = conditional_tail(teacher_p2s[held], yhat)
            importance = 1.0-teacher_p2s[held][np.arange(len(yhat)), yhat]
            w = importance[mask]
            k = tail_kl(held_tail, blended)[mask]
            per_partition.append(float(np.sum(k*w)/np.maximum(np.sum(w), EPS)))
        scores[str(beta)] = {'heldout_tail_kl': float(np.mean(per_partition)),
                             'by_teacher': per_partition,
                             'evaluated_flip_images': int(mask.sum())}
    return scores


def run(args):
    q = raw_array(args.base, args.base_key, ('q','q1','probs','targets'))
    if q.ndim != 2 or not np.all(np.isfinite(q)) or np.any(q < 0) or not np.allclose(q.sum(1),1,rtol=.0003,atol=.0003):
        raise ValueError('baseline q1 must be nonnegative normalized [N,C]')
    q = q.astype(np.float64)
    y = raw_array(args.yhat, args.yhat_key, ('yhat','labels','cls')).astype(np.int64)
    if y.ndim != 1 or len(y) != len(q) or (y < 0).any() or (y >= q.shape[1]).any():
        raise ValueError('bad yhat class indices')
    if len(args.teachers) < 3:
        raise ValueError('>=3 independent teacher partitions required for leave-one-out CV')
    pp = []
    for path in args.teachers:
        x = raw_array(path,args.teacher_key, ('probs','logits','p1','q'))
        if x.shape != q.shape:
            raise ValueError(f'teacher array {path} shape mismatch {x.shape} != {q.shape}')
        p1 = probabilities(x,args.teacher_type)
        pp.append(at_temperature_two(p1))
    betas = sorted(set(float(x) for x in args.betas.split(',')))
    if not all(0 <= x <= 1 for x in betas):
        raise ValueError('beta must be in [0,1]')
    stats = screen_cv(q,y,pp,betas)
    best = min(betas, key=lambda b:stats[str(b)]['heldout_tail_kl'])
    qbase = conditional_tail(at_temperature_two(q),y)
    qteacher = conditional_tail(np.mean(pp,axis=0), y)
    output = Path(args.out).expanduser();output.mkdir(parents=True,exist_ok=True)
    loss_base = -np.log(np.maximum(q[np.arange(len(y)),y], EPS))
    summary = {'n':len(y),'teachers':len(pp),'scores':stats,'best_beta_cv':best,
               'target_ce_mean': float(loss_base.mean()),'target_ce_std': float(loss_base.std()),
               'warning':'OOF teacher-tail proxy only; not EAC score/utility gate, no GPU model trained.'}
    for beta in betas:
        z1,z2 = blend_target(q,y,qbase,qteacher,beta)
        loss_new = -np.log(np.maximum(z1[np.arange(len(y)),y],EPS))
        assert np.max(np.abs(loss_new-loss_base)) < 1e-7
        name = f'target_tail_t2_beta{round(beta*100):02d}'
        np.savez_compressed(output/(name+'.npz'),q1=z1.astype(np.float32),q2=z2.astype(np.float32),
                            yhat=y, beta=np.array(beta,dtype=np.float32))
        summary[name] = {'mean_target_ce':float(loss_new.mean()),
                         'std_target_ce':float(loss_new.std()),
                         'tail_entropy':float(np.mean(-np.sum(qteacher * np.log(np.maximum(qteacher,EPS)),1))),
                         'yhat_argmax_change': float(np.mean(z1.argmax(1)!=q.argmax(1)))}
    (output/'cv.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('base','yhat','out'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--teachers',nargs='+',required=True)
    p.add_argument('--base-key',default=None)
    p.add_argument('--yhat-key',default=None)
    p.add_argument('--teacher-key',default=None)
    p.add_argument('--teacher-type',choices=('prob','logits'),default='prob')
    p.add_argument('--betas',default='0,.2,.4,.6')
    run(p.parse_args())


if __name__ == '__main__':
    main()
