"""T1: paired counterfactual small-hole loss (drop-in).

Fully watermarked image and same image with an original clean hole pasted back
must share background, message, method, geometry and augmentations.

Example:
  pairloss = counterfactual_hole_loss(
       forward(marked, message), forward(clean_hole_paste, message), hole128)
  loss = bce_dice + 0.1 * pairloss

Only supervised train; no test labels, no new watermark on test.
"""
import torch
from torch.nn import functional as F


def counterfactual_hole_loss(marked_logits, repaired_logits, hole,
                             margin=1.0, outside_weight=0.1,
                             sign_weight=0.25):
    """Learn local absence of watermark and invariance on unchanged background."""
    if marked_logits.ndim != 3 or marked_logits.shape != repaired_logits.shape:
        raise ValueError('expected aligned [B,H,W] logits')
    if hole.shape != marked_logits.shape:
        raise ValueError('expected hole [B,H,W] aligned to logits')
    if margin <= 0 or outside_weight < 0 or sign_weight < 0:
        raise ValueError('invalid penalty parameters')
    a = marked_logits.float()
    b = repaired_logits.float()
    h = hole.float().clamp(0, 1)
    valid = h.sum().clamp_min(1.0)
    outside = 1.0 - h
    outside_n = outside.sum().clamp_min(1.0)
    difference = (F.softplus(margin - (a - b)) * h).sum() / valid
    invariance = (((a - b).square().clamp(max=25)) * outside).sum() / outside_n
    signs = ((F.softplus(-a) + F.softplus(b)) * h).sum() / valid
    return difference + outside_weight * invariance + sign_weight * signs


def hole_size_weights(hole, thresholds=(0.02, 0.08), weights=(3.0, 2.0, 1.0)):
    """Optional fixed stratification for <2%, 2%-8%, >=8% restored area."""
    if hole.ndim != 3:
        raise ValueError('expected [B,H,W] hole')
    area = hole.float().mean(dim=(-2, -1))
    return torch.where(area < thresholds[0], weights[0],
                       torch.where(area < thresholds[1], weights[1], weights[2]))


def _smoketest():
    hole = torch.zeros(3, 16, 16)
    hole[:, 4:5, 4:7] = 1
    full = torch.ones_like(hole, requires_grad=True)
    repaired = full.detach().clone()
    repaired[hole.bool()] = -1
    repaired.requires_grad_(True)
    good = counterfactual_hole_loss(full, repaired, hole)
    good.backward()
    bad = counterfactual_hole_loss(full.detach(), full.detach(), hole)
    assert 0 < good.item() < bad.item()
    assert full.grad.abs().sum() > 0 and repaired.grad.abs().sum() > 0
    assert hole_size_weights(hole).tolist() == [3.0] * 3
    print('CPU smoke test PASS; no competition mIoU measured')


if __name__ == '__main__':
    _smoketest()
