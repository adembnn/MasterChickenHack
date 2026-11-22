"""T2 / B3IT-inspired: rank candidate prompts by sampler-invariant top-two flips.

This is a LOCAL probe selection prepass, **not** the B3IT paper algorithm.
B3IT uses controllable T=0 and repeated one-token endpoint queries, which
the EAC API DOES NOT allow; do not replicate its sample-complexity claims.

Inputs are precomputed on SAME continuation prefix under the *base and local
synthetic variants*, using the two token IDs with highest base logits:
  base_top2:     [N,2]          raw logits of those SAME two tokens
  variants_top2: [N,K,2]        logits on same token IDs for K variants
  prompts:       optional [N]  unique candidate user prompts

No EAC endpoints queried. The scores favor near-ties that flip token order
under several weak LoRA/fp8 seeds and cannot be explained by a free
temperature nuisance. top-p pruning and hidden prompts remain unmodeled.

Usage:
 python t2_border_probe_ranker.py                            # self-test
 python t2_border_probe_ranker.py --npz logits.npz --out ranked.csv
"""
import argparse
import csv
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -x))


def rank_border_probes(base_top2, variants_top2, temperatures=(.6, .7, .9, 1.0),
                       quantile=.25, duplicate_groups=None):
    b = np.asarray(base_top2, dtype=np.float64)
    v = np.asarray(variants_top2, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError('base_top2 must be [N,2]')
    if v.ndim != 3 or v.shape != (len(b), v.shape[1], 2) or v.shape[1] == 0:
        raise ValueError('variants_top2 must be aligned [N,K,2]')
    if not (np.isfinite(b).all() and np.isfinite(v).all()):
        raise ValueError('NaN/Inf in logits')
    if not (0. <= quantile <= 1. and all(t > 0 for t in temperatures)):
        raise ValueError('bad quantile / temperatures')
    if not np.all(b[:, 0] >= b[:, 1]):
        raise ValueError('base logits must be top-1 then top-2')

    bgap = b[:, 0] - b[:, 1]
    vgap = v[:, :, 0] - v[:, :, 1]
    T = np.array(temperatures, dtype=np.float64)
    # Arrays of p(first-of-two) under separate nuisance temperatures.
    pb = sigmoid(bgap[:, None] / T[None, :])       # [N,T]
    pv = sigmoid(vgap[:, :, None] / T[None,None,:]) # [N,K,T]
    # Independently allow each side a temperature; this is intentionally conservative.
    delta = np.abs(pb[:, None, :, None] - pv[:, :, None, :])
    min_delta = delta.min(axis=(-1,-2))             # [N,K]
    robust = np.quantile(min_delta, quantile, axis=1)
    flipped = (vgap < 0.).mean(axis=1)
    # Border quality at T=.7: goes to 1 at an equal-probability tie,
    # 0 when top1 overwhelms top2; no claim about all vocabulary tokens.
    near_tie = 4. * sigmoid(bgap / .7) * (1. - sigmoid(bgap / .7))
    score = robust * near_tie
    # Avoid duplicate prompt templates if supplied; greedily select best per group.
    candidates = sorted(range(len(b)), key=lambda i: -score[i])
    if duplicate_groups is not None:
        if len(duplicate_groups) != len(b):
            raise ValueError('duplicate_groups length mismatch')
        seen = set()
        candidates = [i for i in candidates
                      if not (str(duplicate_groups[i]) in seen
                      or seen.add(str(duplicate_groups[i])))]
    return [(int(i), float(score[i]), float(robust[i]), float(near_tie[i]),
             float(flipped[i])) for i in candidates]


def synthetic_test():
    base = np.array([[3.,2.9], [3.,2.9], [3.,2.9], [8.,2.]], dtype=np.float64)
    variants = np.repeat(base[:,None,:], 4, axis=1)
    # Same model (temperature/rescaling) should have NO rank flip.
    variants[0] *= 1.25
    # Four independent weak variants consistently flip this near-tie.
    variants[1,:,0] = 2.85
    variants[1,:,1] = 3.00
    # Only one of four is perturbed: high FP/low power selection should not dominate.
    variants[2,0,0] = 2.8
    variants[2,0,1] = 3.1
    # Easy-margin prompt can flip but score strongly downweighted by border quality.
    variants[3,:,0] = 1
    variants[3,:,1] = 4
    r = rank_border_probes(base, variants)
    assert r[0][0] == 1, r
    assert next(x for x in r if x[0] == 0)[1] < 1e-12
    assert next(x for x in r if x[0] == 2)[1] < 1e-12
    assert next(x for x in r if x[0] == 1)[2] > .01
    assert np.isfinite([v for row in r for v in row[1:]]).all()
    print('PASS synthetic: robust near-tie rank flips selected; T-only and one-off rejected')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--npz', help='locally computed twin-model logits, not endpoint data')
    a.add_argument('--out', default='t2_border_ranked.csv')
    a.add_argument('--quantile', type=float, default=.25)
    args = a.parse_args()
    if args.npz is None:
        synthetic_test()
        return
    z = np.load(args.npz, allow_pickle=False)
    groups = z['groups'] if 'groups' in z else None
    prompts = z['prompts'] if 'prompts' in z else None
    ranked = rank_border_probes(z['base_top2'], z['variants_top2'],
                               quantile=args.quantile, duplicate_groups=groups)
    with open(args.out,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index','score','nuisance_robust_delta','border_quality','variant_flip_fraction','prompt'])
        for i,s,d,h,flip in ranked:
            w.writerow([i,s,d,h,flip,str(prompts[i]) if prompts is not None else ''])
    print('Wrote',len(ranked),'LOCAL candidates to',args.out)
    print('STOP: do not spend EAC API quota without ref+twin blinded validation and Titouan OK')


if __name__ == '__main__':
    main()
