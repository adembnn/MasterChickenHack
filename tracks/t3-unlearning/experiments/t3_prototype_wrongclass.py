"""T3: unsupervised class-prototype alternative to redistilled wrong-class prior.

Inputs features [3500,D] (frozen ViT penultimate layer), logits [3500,100]
(original model on exact forget images), identically ordered.
Usage: python t3_prototype_wrongclass.py --features features.npy --logits logits.npy --out wrong.npz
No model changes, submissions or secret handling.
"""
import argparse
import numpy as np


def _softmax(logits):
    x = logits - np.max(logits, axis=-1, keepdims=True)
    p = np.exp(x)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-20)


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def make_wrong_probs(features, logits, *, beta=0.35, topk=5, tau=0.06,
                     non_target_temp=2.0, min_count=3):
    features = np.asarray(features, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    if features.ndim != 2 or logits.ndim != 2 or len(features) != len(logits):
        raise ValueError('features [N,D] and logits [N,C] require the same N')
    n, classes = logits.shape
    if not (0 <= beta <= 1 and topk >= 1 and tau > 0 and non_target_temp > 0):
        raise ValueError('invalid beta/topk/tau/non_target_temp')
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(logits)):
        raise ValueError('features/logits must be finite')
    labels = logits.argmax(axis=1)
    # Remove shared direction to make cross-class feature comparisons informative.
    z = _normalize(features - features.mean(axis=0, keepdims=True))
    counts = np.bincount(labels, minlength=classes)
    centroids = np.zeros((classes, features.shape[1]), dtype=np.float32)
    for c in range(classes):
        if counts[c] >= min_count:
            centroids[c] = z[labels == c].mean(axis=0)
    centroids = _normalize(centroids)
    valid = counts >= min_count
    if valid.sum() < 2:
        raise ValueError('need >= 2 classes with at least min_count examples')

    similarities = z @ centroids.T / tau
    similarities[:, ~valid] = -1e9
    similarities[np.arange(n), labels] = -1e9
    k = min(topk, classes - 1)
    top = np.argpartition(similarities, -k, axis=1)[:, -k:]
    local = np.take_along_axis(similarities, top, axis=1)
    active = local > -1e8
    local = np.where(active, local, -1e9)
    geo = np.zeros((n, classes), dtype=np.float32)
    np.put_along_axis(geo, top, _softmax(local) * active, axis=1)
    geo /= np.maximum(geo.sum(axis=1, keepdims=True), 1e-12)

    # Keep original non-target ordering as a conservative default.
    non_target = logits / non_target_temp
    non_target[np.arange(n), labels] = -1e9
    orig = _softmax(non_target)
    out = ((1 - beta) * orig + beta * geo).astype(np.float32)
    out[np.arange(n), labels] = 0.0
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out, labels, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--beta', type=float, default=0.35)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--tau', type=float, default=0.06)
    args = ap.parse_args()
    probs, labels, counts = make_wrong_probs(
        np.load(args.features), np.load(args.logits),
        beta=args.beta, topk=args.topk, tau=args.tau)
    np.savez_compressed(args.out, wrong_probs=probs, pseudo_labels=labels,
                        counts=counts, top_wrong=probs.argmax(1))
    h = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    print(f'written {args.out}, shape={probs.shape}, classes_seen={(counts>=3).sum()}, '
          f'avg max probability={probs.max(1).mean():.3f}, avg entropy={h.mean():.3f}')


if __name__ == '__main__':
    main()
