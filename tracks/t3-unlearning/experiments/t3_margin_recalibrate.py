"""Recalibrate the forgotten original-class margin of SOUPs1 on a frozen-base ViT."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from t3_ridge_repair import forget_metrics, kl, read_cache, reference, utility_metrics


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--train-batch", type=int, default=48)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--holdout", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--mode", choices=("increase", "match"), default="increase")
    p.add_argument("--retain-weight", type=float, default=0.08)
    p.add_argument("--exact-scale", type=float, default=1.04)
    p.add_argument("--jpeg-scale", type=float, default=1.05)
    a = p.parse_args()
    a.base = a.base.expanduser().resolve()
    a.out = a.out.expanduser().resolve()
    if min(a.steps, a.train_batch, a.eval_batch, a.holdout) < 1 or a.lr <= 0 or a.retain_weight < 0:
        p.error("steps, batch sizes, holdout and lr must be positive")
    return a


def pixels(images, rows):
    return images[rows] if torch.is_tensor(images) else torch.from_numpy(images[rows.numpy()])


@torch.inference_mode()
def baseline(common, model, images, rows, batch_size):
    return torch.cat([common.logits_u8(model, pixels(images, batch), bs=batch_size)
                      for batch in rows.split(batch_size)]).cuda()


def except_labels(logits, labels):
    mask = F.one_hot(labels, logits.shape[1]).bool()
    return logits.masked_fill(mask, -1e4)


def calibrate(logits, labels, scale):
    originals = logits.gather(1, labels[:, None]).squeeze(1)
    log_tail_mass = torch.logsumexp(except_labels(logits, labels), dim=1)
    loss = F.softplus(log_tail_mass - originals)
    wanted = torch.where(logits.argmax(1) != labels, loss * scale, loss).clamp_min(1e-6)
    log_expm1 = wanted + torch.log(-torch.expm1(-wanted))
    adjusted = logits.scatter(1, labels[:, None], (log_tail_mass - log_expm1)[:, None])
    return adjusted, wanted


def match_marginal(logits, labels, target):
    """Match original-class probability at T=2, subject to a bounded logit change."""
    original = logits.gather(1, labels[:, None]).squeeze(1)
    target_original = target.gather(1, labels[:, None]).squeeze(1) / 2
    target_tail = torch.logsumexp(except_labels(target / 2, labels), dim=1)
    parent_tail = torch.logsumexp(except_labels(logits / 2, labels), dim=1)
    wanted = 2 * (target_original - target_tail + parent_tail)
    change = torch.where(logits.argmax(1) != labels, (wanted - original).clamp(-0.6, 0.6), 0.0)
    adjusted = logits.scatter(1, labels[:, None], (original + change)[:, None])
    return adjusted, F.cross_entropy(adjusted, labels, reduction="none")


def kl_for_train(logits, target):
    return 4 * F.kl_div(
        (logits / 2).log_softmax(1), (target / 2).softmax(1), reduction="batchmean"
    )


def forget_score(z, parent, labels, targets):
    score = forget_metrics(parent, z, labels, targets)
    score["kl_tail_parent"] = kl(except_labels(parent, labels), except_labels(z, labels))
    return score


def improved_projection(row, base, mode):
    preserve = all(
        row["kl_target"] <= base["kl_target"] + 0.004
        and row["kl_tail_parent"] <= 0.004
        and row["agree_parent"] >= 0.98
        and abs(row["flip_fraction"] - base["flip_fraction"]) <= 0.015
        for row, base in ((row["exact"], base["exact"]), (row["jpeg"], base["jpeg"]))
    )
    retain_ok = row["mix"]["retain_proxy"] >= base["mix"]["retain_proxy"] - 0.004
    clean_gain = row["exact"]["ce_mean"] - base["exact"]["ce_mean"]
    jpeg_gain = row["jpeg"]["ce_mean"] - base["jpeg"]["ce_mean"]
    if mode == "match":
        delta_kl = sum(row[k]["kl_target"] - base[k]["kl_target"] for k in ("exact", "jpeg"))
        return preserve and retain_ok and delta_kl <= -0.002 and abs(clean_gain) <= 0.16 and abs(jpeg_gain) <= 0.16
    return preserve and retain_ok and 0.04 <= clean_gain <= 0.28 and 0.035 <= jpeg_gain <= 0.35


def checkpoint(model, output):
    state = {name: value.detach().cpu().float().contiguous()
             for name, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("Invalid values in state dict")
    temp = output.with_suffix(".part")
    try:
        torch.save(state, temp)
        if temp.stat().st_size > 450_000_000:
            raise RuntimeError("Checkpoint exceeds size limit")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    with output.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(output), "sha256": sha, "bytes": output.stat().st_size}


def main():
    args = options()
    sys.path.insert(0, str(args.base / "code"))
    os.environ.update(T3BASE=str(args.base), T3ME=str(args.base))
    import common

    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise SystemExit(f"Use fresh output directory: {args.out}")
    torch.cuda.reset_peak_memory_stats()
    clock = time.perf_counter
    start = clock()
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(482611)
    cache = args.base / "cache"
    parent_path = args.base / "inbox/SOUPs1.pt"
    model = common.tt.load_model(parent_path)
    raw = common.load_forget()
    jpeg = read_cache(cache, "held_jpeg50")
    mild = read_cache(cache, "held_mild")
    mixtures = read_cache(cache, "mixes")
    cifar = read_cache(cache, "ood_cifar100")
    if len(raw) != len(jpeg) or len(raw) != len(mild):
        raise ValueError("Cached views must align with forget images")
    if min(len(raw), len(mixtures)) <= args.holdout or len(cifar) < 1408:
        raise ValueError("Insufficient data for holdouts")
    order = torch.randperm(len(raw), generator=gen)
    mix_order = torch.randperm(len(mixtures), generator=gen)
    forget_fit, forget_test = order[args.holdout:], order[:args.holdout]
    mix_fit, mix_test = mix_order[args.holdout:], mix_order[:args.holdout]
    cifar_test = torch.arange(1024, 1408)
    exact_labels = reference(cache, "exact", torch.arange(len(raw))).argmax(1)
    target_path = cache / "tgt_RDt24r_70_l1016.pt"
    all_target = torch.load(target_path, map_location="cpu", weights_only=True)["z"].float().cuda()
    target = all_target[forget_test]
    holdouts = {
        "exact": (raw, forget_test),
        "jpeg": (jpeg, forget_test),
        "mix": (mixtures, mix_test),
        "mild": (mild, forget_test),
        "cifar": (cifar, cifar_test),
    }
    baseline_start = clock()
    parent_logits = {
        name: baseline(common, model, images, indices, args.eval_batch)
        for name, (images, indices) in holdouts.items()
    }
    teacher_a = baseline(common, model, raw, forget_fit, args.eval_batch)
    teacher_b = baseline(common, model, jpeg, forget_fit, args.eval_batch)
    # Cache ordinary tensors for backpropagation through the student loss.
    teacher_a, teacher_b = teacher_a.clone(), teacher_b.clone()
    torch.cuda.synchronize()
    baseline_sec = clock() - baseline_start
    baseline_images = sum(len(indices) for _, indices in holdouts.values()) + 2 * len(forget_fit)

    cache_orig_mixes = reference(cache, "mixes", torch.arange(len(mixtures)))
    cache_orig_cifar = reference(cache, "ood_cifar100", cifar_test)

    def metrics(logits):
        out = {name: forget_score(logits[name], parent_logits[name], exact_labels[forget_test], target)
               for name in ("exact", "jpeg", "mild") if name in logits}
        if "mix" in logits:
            out["mix"] = utility_metrics(cache_orig_mixes[mix_test], logits["mix"])
        if "cifar" in logits:
            out["cifar"] = utility_metrics(cache_orig_cifar, logits["cifar"])
        return out

    base = metrics(parent_logits)
    print(json.dumps({"baseline": base, "cache_images_per_second": baseline_images / baseline_sec}), flush=True)
    if args.mode == "match":
        oracle = {}
        for name in ("exact", "jpeg", "mild"):
            z, _ = match_marginal(parent_logits[name], exact_labels[forget_test], target)
            oracle[name] = forget_score(z, parent_logits[name], exact_labels[forget_test], target)
        print(json.dumps({"analytic_marginal_ceiling": oracle}, allow_nan=False), flush=True)
    model.requires_grad_(False)
    train_blocks = [model.head, model.blocks[-1].mlp.fc2, model.blocks[-1].attn.proj]
    for block in train_blocks:
        block.requires_grad_(True)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.lr, weight_decay=0)
    controls = {name: tensor.detach().clone() for name, tensor in trainable.items()}
    evaluations = sorted(set((max(1, args.steps // 4), max(1, args.steps // 2), args.steps)))
    observations = []
    best = None
    best_snapshot = None
    train_seconds = 0.0
    first_train = clock()
    for step in range(1, args.steps + 1):
        ia = torch.randint(len(forget_fit), (args.train_batch,), generator=gen)
        ib = torch.randint(len(forget_fit), (args.train_batch,), generator=gen)
        im = mix_fit[torch.randint(len(mix_fit), (args.train_batch,), generator=gen)]
        ya, yb = exact_labels[forget_fit[ia]], exact_labels[forget_fit[ib]]
        if args.mode == "match":
            za, ca = match_marginal(teacher_a[ia], ya, all_target[forget_fit[ia]])
            zb, cb = match_marginal(teacher_b[ib], yb, all_target[forget_fit[ib]])
        else:
            za, ca = calibrate(teacher_a[ia], ya, args.exact_scale)
            zb, cb = calibrate(teacher_b[ib], yb, args.jpeg_scale)
        combined = torch.cat([pixels(raw, forget_fit[ia]), pixels(jpeg, forget_fit[ib]), pixels(mixtures, im)])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            student = model(common.norm(common.to01(combined))).float()
        sa, sb, sr = student.split(args.train_batch)
        loss_target = 0.5 * (kl_for_train(sa, za) + kl_for_train(sb, zb))
        loss_margin = 0.5 * (
            F.smooth_l1_loss(F.cross_entropy(sa, ya, reduction="none"), ca)
            + F.smooth_l1_loss(F.cross_entropy(sb, yb, reduction="none"), cb)
        )
        loss_retain = kl_for_train(sr, cache_orig_mixes[im])
        loss_anchor = sum((param - controls[name]).square().mean() for name, param in trainable.items())
        loss = loss_target + 0.12 * loss_margin + args.retain_weight * loss_retain + 10 * loss_anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        optimizer.step()
        if step not in evaluations:
            continue
        torch.cuda.synchronize()
        train_seconds += clock() - first_train
        evaluation = {
            name: baseline(common, model, images, indices, args.eval_batch)
            for name, (images, indices) in holdouts.items() if name in {"exact", "jpeg", "mix"}
        }
        row = metrics(evaluation)
        row.update(step=step, loss=loss.item())
        row["paired_pass"] = improved_projection(row, base, args.mode)
        observations.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        rank = (
            abs((base["exact"]["ce_mean"] + 0.18) - row["exact"]["ce_mean"])
            + abs((base["jpeg"]["ce_mean"] + 0.20) - row["jpeg"]["ce_mean"])
            + max(0.0, base["mix"]["retain_proxy"] - row["mix"]["retain_proxy"]) * 10
        )
        if args.mode == "match":
            rank = sum(row[name]["kl_target"] - base[name]["kl_target"] for name in ("exact", "jpeg"))
        if row["paired_pass"] and (best is None or rank < best):
            best = rank
            best_snapshot = row.copy(), {name: p.detach().clone() for name, p in trainable.items()}
        first_train = clock()

    report = {
        "parent": str(parent_path), "mode": args.mode, "baseline": base, "observations": observations,
        "exact_scale": args.exact_scale, "jpeg_scale": args.jpeg_scale,
        "retain_weight": args.retain_weight,
        "baseline_images_per_second": baseline_images / baseline_sec,
        "training_images_per_second": args.steps * args.train_batch * 3 / train_seconds,
        "train_seconds": train_seconds, "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "notes": "CIFAR original-agreement and teacher logits are local proxies. CIFAR and mild are excluded from model selection.",
    }
    if best_snapshot:
        finalist, snapshot = best_snapshot
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(snapshot[name])
        held = {name: baseline(common, model, images, indices, args.eval_batch)
                for name, (images, indices) in holdouts.items() if name in {"mild", "cifar"}}
        finalist.update(metrics(held))
        pass_mild = (
            finalist["mild"]["ce_mean"] >= base["mild"]["ce_mean"] + 0.025
            and finalist["mild"]["kl_target"] <= base["mild"]["kl_target"] + 0.005
            and finalist["mild"]["kl_tail_parent"] <= 0.005
            and finalist["mild"]["agree_parent"] >= 0.98
        )
        if args.mode == "match":
            pass_mild = (
                abs(finalist["mild"]["ce_mean"] - base["mild"]["ce_mean"]) <= 0.16
                and finalist["mild"]["kl_target"] <= base["mild"]["kl_target"] - 0.001
                and finalist["mild"]["kl_tail_parent"] <= 0.005
                and finalist["mild"]["agree_parent"] >= 0.98
            )
        pass_cifar = (
            finalist["cifar"]["retain_proxy"] >= base["cifar"]["retain_proxy"] - 0.004
            and finalist["cifar"]["agree_orig"] >= base["cifar"]["agree_orig"] - 0.01
        )
        report.update(finalist=finalist, pass_mild=pass_mild, pass_cifar=pass_cifar)
        if pass_mild and pass_cifar:
            report["checkpoint"] = checkpoint(model, args.out / "margin_shortlist.pt")
    report.update(
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        seconds=clock() - start,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    (args.out / "metrics.json").write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
