"""T3 — make a single strict float32 ViT checkpoint by interpolating two PG states.

Why: PG_T85_f70_l814c and PG_T85_f80_l814c have close public scores
and similar training methods. Weight interpolation may smooth harmful drift,
but can also reduce forgetting and has NOT been tested on contest data.

Do not submit a mixture without checking exact+altered forget scores,
holdout retain-KL, utility gate and independent maker-checker.
Example:
  python t3_pg_weight_mix.py \
    --a PG_T85_f70_l814c.pt --b PG_T85_f80_l814c.pt \
    --b-weight .5 --out PG_mix70_80_b50.pt

Input files MUST be strict float32 original-architecture state_dicts.
"""
import argparse
import os
import torch


def interpolate_state_dict(a, b, w):
    if not (0 <= w <= 1):
        raise ValueError("b-weight must be in [0,1]")
    if set(a) != set(b):
        raise ValueError("state_dict keys differ")
    out = {}
    for key in a:
        x, y = a[key], b[key]
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise ValueError(f"{key}: expected tensors")
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"{key}: different shapes or dtypes")
        if not torch.isfinite(x.float()).all() or not torch.isfinite(y.float()).all():
            raise ValueError(f"{key}: NaN/Inf in checkpoint")
        if x.is_floating_point():
            if x.dtype != torch.float32:
                raise ValueError(f"{key}: floating tensor must be float32")
            # Implement without merging into a different architecture.
            out[key] = torch.lerp(x, y, w).contiguous()
        else:
            if not torch.equal(x, y):
                raise ValueError(f"{key}: non-floating buffers differ")
            out[key] = x.contiguous()
    return out


def load(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict) or 'head.weight' not in sd:
        raise ValueError("expected state_dict, not training checkpoint/adapter")
    return sd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a')
    p.add_argument('--b')
    p.add_argument('--b-weight', type=float, default=.5)
    p.add_argument('--out')
    args = p.parse_args()
    if not args.a:
        toy_a = {'head.weight': torch.ones(2, 3, dtype=torch.float32),
                 'head.bias': torch.zeros(2, dtype=torch.float32)}
        toy_b = {'head.weight': torch.full((2, 3), 3., dtype=torch.float32),
                 'head.bias': torch.full((2,), 2., dtype=torch.float32)}
        o = interpolate_state_dict(toy_a, toy_b, .5)
        assert torch.allclose(o['head.weight'], torch.full((2,3), 2.))
        assert torch.allclose(o['head.bias'], torch.ones(2))
        print('CPU synthetic smoke test PASS (weights + bias, key/shape validation)')
        return
    if not args.b or not args.out:
        p.error('--a --b --out required')
    if os.path.abspath(args.out) in (os.path.abspath(args.a),os.path.abspath(args.b)):
        p.error('refusing to overwrite either input')
    sd = interpolate_state_dict(load(args.a), load(args.b), args.b_weight)
    torch.save(sd,args.out)
    print('wrote',args.out,'b-weight',args.b_weight,'size MiB',round(os.path.getsize(args.out)/2**20,2))
    if os.path.getsize(args.out)>450_000_000:
        raise RuntimeError('output exceeds EAC 450MB cap')
    print('NOT a validated candidate: run strict load / fp16 forward / gate / forget + retain offline')


if __name__=='__main__':
    main()
