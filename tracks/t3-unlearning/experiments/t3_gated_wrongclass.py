"""T3: disagreement-gated blending of *conditional* wrong-class targets.

Requires two independently-derived, aligned N x C probability tensors:
- baseline wrong-class distribution (e.g. PG voted degraded views)
- alternative wrong-class distribution (e.g. OOF redistill).
The original/pseudo-true class probability is frozen exactly; ONLY the
conditional probability over remaining C-1 labels changes.

Example:
    python t3_gated_wrongclass.py --baseline base.npz --alternative oof.npz \
      --labels labels.npy --true-mass mass.npy --out blended.npz --alpha 0.35

NPZ inputs use 'wrong_probs' key by default and must have identical index order.
No unlabeled test submission, EAC API, GPU or secret required.

Motivation: per-sample distribution alignment (SALA, arXiv:2611.04501).
This is an experimental adaptation, NOT the SALA algorithm or guaranteed gold.
"""
import argparse
import numpy as np

EPS = 1e-12


def _renorm_wrong(p, y):
    p = np.asarray(p, dtype=np.float64).copy()
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("expected aligned [N,C] probabilities")
    if not np.isfinite(p).all() or np.min(p) < 0:
        raise ValueError("probabilities must be finite and nonnegative")
    p[np.arange(len(y)), y] = 0.
    den = p.sum(axis=1)
    if np.any(den < EPS):
        raise ValueError("a row contains no off-label probability")
    return p / den[:, None]


def jsd(p, q):
    m = (p + q) / 2.
    return .5 * np.sum(np.where(p > 0., p * np.log(np.maximum(p, EPS) / np.maximum(m, EPS)), 0.), axis=1) + \
           .5 * np.sum(np.where(q > 0., q * np.log(np.maximum(q, EPS) / np.maximum(m, EPS)), 0.), axis=1)


def gated_targets(base_wrong, alt_wrong, y, p_y, alpha=.35, tau=.15,
                  floor=.0, mode="gated"):
    """Output T=1 targets, per-example alpha, and JSD.

    'constant' is the necessary ablation. 'gated' trusts the alternative
    more when pseudo-teachers AGREE; no labels or gold needed.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p_y = np.asarray(p_y, dtype=np.float64).reshape(-1)
    n = len(y)
    base = np.asarray(base_wrong)
    if base.ndim != 2:
        raise ValueError("base should be [N,C]")
    c = base.shape[1]
    if c < 3 or n == 0 or len(p_y) != n or np.any(y < 0) or np.any(y >= c):
        raise ValueError("invalid shapes or labels")
    if not np.isfinite(p_y).all() or np.any(p_y <= 0) or np.any(p_y >= 1):
        raise ValueError("true-class mass must satisfy 0<p_y<1")
    if not (0 <= floor <= alpha <= 1 and tau > 0 and mode in ("gated", "constant")):
        raise ValueError("invalid alpha, tau, floor or mode")
    b = _renorm_wrong(base_wrong, y)
    a = _renorm_wrong(alt_wrong, y)
    divergence = jsd(b, a)
    g = np.full(n, alpha) if mode == "constant" else \
        floor + (alpha - floor) * np.exp(-divergence / tau)
    # Convex mixture avoids spiky single-class alternatives.
    wrong = (1. - g[:, None]) * b + g[:, None] * a
    wrong = _renorm_wrong(wrong, y)
    q1 = (1. - p_y[:, None]) * wrong
    q1[np.arange(n), y] = p_y
    # Standard logits T=1 -> p(T=2) transformation, normalized.
    q2 = np.sqrt(np.maximum(q1, EPS))
    q2 /= q2.sum(axis=1, keepdims=True)
    if not np.allclose(q1.sum(axis=1), 1., atol=1e-6):
        raise AssertionError("bad normalization")
    if not np.allclose(q1[np.arange(n), y], p_y, atol=1e-9):
        raise AssertionError("target label mass changed")
    return q1.astype(np.float32), q2.astype(np.float32), g.astype(np.float32), divergence.astype(np.float32)


def _read_wrong(path):
    z = np.load(path)
    if isinstance(z, np.ndarray):
        return z
    for key in ("wrong_probs", "w", "probs"):
        if key in z:
            return z[key]
    raise ValueError("NPZ needs wrong_probs/w/probs")


def smoke_test():
    rng = np.random.default_rng(20261121)
    n, c = 128, 100
    y = rng.integers(0, c, size=n)
    p_y = rng.uniform(.01, .95, size=n)
    p = rng.random((n, c))
    q = rng.random((n, c))
    a, a2, weights, d = gated_targets(p, q, y, p_y)
    b, b2, bw, _ = gated_targets(p, q, y, p_y, alpha=0)
    orig, _, _, _ = gated_targets(p, p, y, p_y, alpha=0)
    assert np.allclose(a.sum(1), 1.)
    assert np.allclose(a2.sum(1), 1.)
    assert np.allclose(a[np.arange(n), y], p_y)
    assert np.allclose(b, orig, atol=1e-7)
    assert (weights >= 0).all() and (weights <= .35).all()
    assert np.isfinite(d).all() and np.isfinite(a).all()
    print("PASS CPU synthetic, N=128 C=100, true class fixed, T1/T2 normalized, alpha0 baseline.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline")
    ap.add_argument("--alternative")
    ap.add_argument("--labels")
    ap.add_argument("--true-mass")
    ap.add_argument("--out")
    ap.add_argument("--alpha", type=float, default=.35)
    ap.add_argument("--tau", type=float, default=.15)
    ap.add_argument("--floor", type=float, default=0.)
    ap.add_argument("--mode", choices=("gated", "constant"), default="gated")
    args = ap.parse_args()
    if not args.baseline:
        smoke_test()
        return
    for arg in (args.alternative, args.labels, args.true_mass, args.out):
        if not arg:
            ap.error("all --alternative/--labels/--true-mass/--out required")
    data = gated_targets(_read_wrong(args.baseline), _read_wrong(args.alternative),
                         np.load(args.labels), np.load(args.true_mass),
                         args.alpha, args.tau, args.floor, args.mode)
    np.savez_compressed(args.out, q1=data[0], q2=data[1], weights=data[2], jsd=data[3])
    print("Written", args.out, "mean weight", float(data[2].mean()),
          "mean JSD", float(data[3].mean()), "No EAC benchmark yet")


if __name__ == "__main__":
    main()
