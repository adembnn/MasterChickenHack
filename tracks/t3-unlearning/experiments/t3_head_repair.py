"""T3: experimental retention-oriented repair of a candidate classifier head.

Using independent unlabeled proxy images, match the original model's logits
with a small ridge-regression correction to the candidate's last layer.
Inputs: candidate final-norm pre_logits features and original logits,
computed on the SAME independent proxy images in the same order.
Must NOT be a proxy used to train or calibrate the unlearning candidate.
An independent proxy holdout and forget views are required before submitting.

CPU smoke test: python t3_head_repair.py
CLI: python t3_head_repair.py --candidate PG_f70.pt --features candidate_features.npy --original-logits orig_logits.npy --alpha .25 --ridge .05 --out PG_f70_repair_a25.pt
"""
import argparse
import numpy as np
import torch


def repair_head_ridge(features, original_logits, weight, bias, alpha=0.25, ridge=0.05):
    X = torch.as_tensor(np.asarray(features), dtype=torch.float64, device='cpu')
    Y = torch.as_tensor(np.asarray(original_logits), dtype=torch.float64, device='cpu')
    W = weight.detach().to('cpu', torch.float64)
    b = bias.detach().to('cpu', torch.float64)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError('features N×D and original_logits N×C must be aligned')
    n, d = X.shape
    c = Y.shape[1]
    if W.shape != (c, d) or b.shape != (c,) or n <= 1:
        raise ValueError('head/cache shapes mismatch')
    if not (0 <= alpha <= 1 and ridge > 0):
        raise ValueError('alpha in [0,1], ridge>0')
    if not all(torch.isfinite(z).all() for z in (X, Y, W, b)):
        raise ValueError('arrays must be finite')
    A = torch.cat([X, torch.ones((n, 1), dtype=X.dtype)], dim=1)
    before = X @ W.T + b
    residual = Y - before
    gram = A.T @ A / n
    scale = gram.diagonal()[:d].mean().clamp_min(1e-8)
    R = torch.eye(d + 1, dtype=X.dtype) * (ridge * scale)
    R[-1, -1] *= 0.1
    delta = torch.linalg.solve(gram + R, A.T @ residual / n).T
    w_new, b_new = W + alpha * delta[:, :d], b + alpha * delta[:, d]
    after = X @ w_new.T + b_new
    report = {'proxy_mse_before': float(((before-Y)**2).mean()),
              'proxy_mse_after_in_sample': float(((after-Y)**2).mean()),
              'delta_l2': float(torch.linalg.vector_norm(delta)),
              'samples': n, 'dim': d, 'classes': c, 'alpha': alpha, 'ridge': ridge}
    return w_new.to(weight.dtype), b_new.to(bias.dtype), report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True)
    ap.add_argument('--original-logits', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--alpha', type=float, default=.25)
    ap.add_argument('--ridge', type=float, default=.05)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sd = torch.load(a.candidate, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict) or 'head.weight' not in sd or 'head.bias' not in sd:
        raise ValueError('expected original ViT state_dict containing head.weight and head.bias')
    nw, nb, report = repair_head_ridge(np.load(a.features), np.load(a.original_logits),
                                       sd['head.weight'], sd['head.bias'], a.alpha, a.ridge)
    sd['head.weight'] = nw.float().contiguous()
    sd['head.bias'] = nb.float().contiguous()
    for value in sd.values():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('all floating-point weights must be finite float32')
    torch.save(sd, a.out)
    print(report)
    print('IMPORTANT: in-sample calibration MSE is not a T3 score; independently verify retained utility, forget distribution and gate.')


def smoketest():
    torch.manual_seed(20261121)
    n, d, c = 200, 20, 5
    f = torch.randn(n, d, dtype=torch.float64)
    w = torch.randn(c, d, dtype=torch.float64)
    b = torch.zeros(c, dtype=torch.float64)
    z = f @ (w + .3 * torch.randn_like(w)).T
    wn, bn, rep = repair_head_ridge(f.numpy(), z.numpy(), w, b, alpha=.5, ridge=.01)
    assert wn.shape == w.shape and bn.shape == b.shape
    assert 0 < rep['proxy_mse_after_in_sample'] < rep['proxy_mse_before']
    print('T3 CPU smoke test PASS', rep)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        smoketest()
    else:
        main()
