"""Fit a retain-logit correction to a frozen T3 classifier head on H200."""

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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.home() / "runs/pg")
    parser.add_argument("--parent", type=Path, help="Default: BASE/inbox/SOUPs1.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", default="RDt24r_70_l1016")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--fit", type=int, default=2048)
    parser.add_argument("--holdout", type=int, default=512)
    parser.add_argument("--cifar-start", type=int, default=1024)
    parser.add_argument("--cifar-count", type=int, default=384)
    parser.add_argument("--cifar-labels", type=Path, help="Optional .npy aligned to cached CIFAR rows")
    parser.add_argument("--objective", choices=("ridge", "kl"), default="ridge")
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()
    if min(args.batch, args.fit, args.holdout, args.cifar_count) < 1 or args.cifar_start < 0:
        parser.error("positive batch and sample counts and a nonnegative CIFAR start are required")
    args.base = args.base.expanduser().resolve()
    args.parent = args.parent.expanduser().resolve() if args.parent else args.base / "inbox/SOUPs1.pt"
    args.out = args.out.expanduser().resolve()
    if args.cifar_labels:
        args.cifar_labels = args.cifar_labels.expanduser().resolve()
    return args


def read_cache(cache, stem):
    return np.load(cache / f"{stem}.npy", mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def capture(common, model, images, indices, batch_size):
    """Capture exactly the input and output of the server's linear head."""
    features = []
    outputs = []

    def hook(_module, inputs, _output):
        features.append(inputs[0].detach().float())

    handle = model.head.register_forward_hook(hook)
    try:
        for rows in indices.split(batch_size):
            images_batch = images[rows] if torch.is_tensor(images) else torch.from_numpy(images[rows.numpy()])
            with common.ac():
                outputs.append(model(common.norm(common.to01(images_batch))).float())
    finally:
        handle.remove()
    return torch.cat(features), torch.cat(outputs)


def reference(cache, stem, indices):
    array = read_cache(cache, f"{stem}_orig_logits")
    return torch.from_numpy(array[indices.numpy()]).to(device="cuda", dtype=torch.float32)


def repair_directions(mix_features, mix_gap, forget_features):
    def gram(features):
        return features.T @ features / len(features)

    retain_gram = gram(mix_features)
    forget_gram = gram(forget_features)
    rhs = mix_features.T @ mix_gap / len(mix_features)
    identity = torch.eye(retain_gram.shape[0], device=retain_gram.device)
    for penalty in (1.0, 4.0, 16.0):
        matrix = retain_gram + penalty * forget_gram
        ridge = 0.001 * matrix.diagonal().mean().clamp_min(1e-8)
        yield penalty, torch.linalg.solve(matrix + ridge * identity, rhs).T.contiguous()


def kl_repair_directions(mix_features, mix_orig, mix_parent, forget_features, forget_parent, labels):
    """Optimize the head against temperature-2 KL and the parent CE per forget image."""
    target_retain = (mix_orig / 2).softmax(dim=1)
    target_forget = (forget_parent / 2).softmax(dim=1)
    parent_ce = F.cross_entropy(forget_parent, labels, reduction="none")
    scale = mix_features.square().mean().sqrt().item()

    # A zero-initialized linear update starts at the known parent checkpoint.
    # Use ordinary tensors since inference tensors cannot be saved for backward.
    with torch.inference_mode(False), torch.enable_grad():
        mx, mp, rt, fx, fp, ft, y, pc = [
            t.detach().clone()
            for t in (mix_features, mix_parent, target_retain, forget_features,
                      forget_parent, target_forget, labels, parent_ce)
        ]
        for penalty in (1.0, 4.0, 16.0):
            delta = torch.zeros(
                (mix_orig.shape[1], mix_features.shape[1]), device=mx.device, requires_grad=True
            )
            optimizer = torch.optim.LBFGS([delta], max_iter=45, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                mix_logits = mp + mx @ delta.T
                forget_logits = fp + fx @ delta.T
                retain = F.kl_div((mix_logits / 2).log_softmax(1), rt, reduction="batchmean")
                preserve = F.kl_div((forget_logits / 2).log_softmax(1), ft, reduction="batchmean")
                ce = F.cross_entropy(forget_logits, y, reduction="none")
                ce_match = (ce - pc).square().mean()
                loss = retain + penalty * (preserve + 0.10 * ce_match) + 0.01 * scale * delta.square().sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            yield penalty, delta.detach()


def kl(reference_logits, candidate_logits):
    return F.kl_div(
        (candidate_logits / 2).log_softmax(1),
        (reference_logits / 2).softmax(1),
        reduction="batchmean",
    ).item()


def utility_metrics(original, current, labels=None):
    result = {
        "agree_orig": (current.argmax(1) == original.argmax(1)).float().mean().item(),
        "kl_orig": kl(original, current),
    }
    result["retain_proxy"] = result["agree_orig"] / (1 + result["kl_orig"])
    if labels is not None:
        result["accuracy"] = (current.argmax(1) == labels).float().mean().item()
        result["original_accuracy"] = (original.argmax(1) == labels).float().mean().item()
        result["retain_proxy"] = min(
            result["accuracy"] / max(result["original_accuracy"], 1e-8), 1.0
        ) / (1 + result["kl_orig"])
    return result


def forget_metrics(parent, current, true_proxy, target):
    ce = F.cross_entropy(current, true_proxy, reduction="none")
    return {
        "ce_mean": ce.mean().item(),
        "ce_std": ce.std().item(),
        "flip_fraction": (current.argmax(1) != true_proxy).float().mean().item(),
        "agree_parent": (current.argmax(1) == parent.argmax(1)).float().mean().item(),
        "kl_parent": kl(parent, current),
        "kl_target": kl(target, current),
    }


def preserves_forget(candidate, baseline):
    return (
        abs(candidate["ce_mean"] - baseline["ce_mean"]) <= 0.10
        and abs(candidate["ce_std"] - baseline["ce_std"]) <= 0.10
        and abs(candidate["flip_fraction"] - baseline["flip_fraction"]) <= 0.02
        and candidate["agree_parent"] >= 0.97
        and candidate["kl_parent"] <= 0.01
        and candidate["kl_target"] <= baseline["kl_target"] + 0.01
    )


def predict_head(common, features, weight, bias):
    with common.ac():
        return F.linear(features, weight, bias).float()


def export_checkpoint(state, weight, out):
    state["head.weight"] = weight.detach().cpu().contiguous()
    for name, tensor in state.items():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid parameter {name}")
    path = out / "ridge_shortlist.pt"
    temporary = out / "ridge_shortlist.pt.part"
    try:
        torch.save(state, temporary)
        if temporary.stat().st_size > 450_000_000:
            raise ValueError("Checkpoint exceeds size limit")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


@torch.inference_mode()
def main():
    args = arguments()
    os.environ.update(T3BASE=str(args.base), T3ME=str(args.base))
    sys.path.insert(0, str(args.base / "code"))
    import common

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "metrics.json").exists() or (args.out / "ridge_shortlist.pt").exists():
        raise SystemExit(f"Use a fresh output directory: {args.out}")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    cache = args.base / "cache"
    state = torch.load(args.parent, map_location="cpu", weights_only=True)
    model = common.tt.build_model()
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    weight, bias = model.head.weight.detach(), model.head.bias.detach()
    exact = common.load_forget()
    jpeg = read_cache(cache, "held_jpeg50")
    mild = read_cache(cache, "held_mild")
    mixes = read_cache(cache, "mixes")
    cifar = read_cache(cache, "ood_cifar100")
    gen = torch.Generator().manual_seed(args.seed)
    if min(len(exact), len(mixes)) < args.fit + args.holdout:
        raise ValueError("Insufficient rows for disjoint fit and holdout")
    if args.cifar_start + args.cifar_count > len(cifar):
        raise ValueError("CIFAR holdout exceeds cached rows")
    selected = args.fit + args.holdout
    forget_rows = torch.randperm(len(exact), generator=gen)[:selected]
    mix_rows = torch.randperm(len(mixes), generator=gen)[:selected]
    cifar_rows = torch.arange(args.cifar_start, args.cifar_start + args.cifar_count)
    fit = slice(0, args.fit)
    hold = slice(args.fit, selected)
    tranches = {"exact": (exact, forget_rows), "jpeg": (jpeg, forget_rows), "mix": (mixes, mix_rows)}
    capture_start = time.perf_counter()
    feats, parent_logits = {}, {}
    for name, (images, rows) in tranches.items():
        feats[name], parent_logits[name] = capture(common, model, images, rows, args.batch)
        reconstructed = predict_head(common, feats[name], weight, bias)
        mismatch = (parent_logits[name] - reconstructed).abs().max().item()
        if mismatch > 0.04:
            raise RuntimeError(f"{name}: cached input does not reproduce head, max logit gap {mismatch}")
        print(f"Cached {name}: {len(rows)} rows, max reconstruction gap {mismatch:.5f}", flush=True)
    torch.cuda.synchronize()
    sweep_capture_seconds = time.perf_counter() - capture_start
    mix_orig = reference(cache, "mixes", mix_rows)
    forget_labels = reference(cache, "exact", forget_rows[hold]).argmax(1)
    target_file = cache / f"tgt_{args.target}.pt"
    target = torch.load(target_file, map_location="cpu", weights_only=True)["z"][forget_rows[hold]].cuda()
    if target.shape != parent_logits["exact"][hold].shape:
        raise ValueError("Target logits do not align with forget samples")
    gap = mix_orig[fit] - parent_logits["mix"][fit]
    gap -= gap.mean(dim=1, keepdim=True)
    forget_fit = torch.cat([feats[name][fit] for name in ("exact", "jpeg")])
    base = {
        name: forget_metrics(parent_logits[name][hold], parent_logits[name][hold], forget_labels, target)
        for name in ("exact", "jpeg")
    }
    base["mix"] = utility_metrics(mix_orig[hold], parent_logits["mix"][hold])
    rows = []
    shortlist = None
    best_proxy = base["mix"]["retain_proxy"]
    sweep_start = time.perf_counter()
    if args.objective == "kl":
        labels_fit = reference(cache, "exact", forget_rows[fit]).argmax(1).repeat(2)
        parent_forget_fit = torch.cat([parent_logits[name][fit] for name in ("exact", "jpeg")])
        directions = kl_repair_directions(
            feats["mix"][fit], mix_orig[fit], parent_logits["mix"][fit],
            forget_fit, parent_forget_fit, labels_fit,
        )
    else:
        directions = repair_directions(feats["mix"][fit], gap, forget_fit)
    for penalty, direction in directions:
        for strength in (0.25, 0.5, 1.0):
            candidate_weight = weight + strength * direction
            scores = {"penalty": penalty, "strength": strength}
            for name in ("exact", "jpeg"):
                logits = predict_head(common, feats[name][hold], candidate_weight, bias)
                scores[name] = forget_metrics(parent_logits[name][hold], logits, forget_labels, target)
            logits = predict_head(common, feats["mix"][hold], candidate_weight, bias)
            scores["mix"] = utility_metrics(mix_orig[hold], logits)
            scores["passes_forget"] = all(preserves_forget(scores[name], base[name]) for name in ("exact", "jpeg"))
            rows.append(scores)
            if scores["passes_forget"] and scores["mix"]["retain_proxy"] > best_proxy:
                shortlist = candidate_weight, scores
                best_proxy = scores["mix"]["retain_proxy"]
    torch.cuda.synchronize()
    sweep_seconds = time.perf_counter() - sweep_start

    # Neither mild nor CIFAR contributes to choosing the finalist.
    final_capture_start = time.perf_counter()
    mild_features, mild_parent = capture(common, model, mild, forget_rows[hold], args.batch)
    cifar_features, cifar_parent = capture(common, model, cifar, cifar_rows, args.batch)
    torch.cuda.synchronize()
    final_capture_seconds = time.perf_counter() - final_capture_start
    labels = None
    if args.cifar_labels:
        source_labels = np.load(args.cifar_labels, allow_pickle=False)
        if source_labels.shape != (len(cifar),) or not np.issubdtype(source_labels.dtype, np.integer):
            raise ValueError("CIFAR labels must supply one integer per cached row in cache order")
        labels = torch.from_numpy(source_labels[cifar_rows.numpy()]).long().cuda()
    cifar_orig = reference(cache, "ood_cifar100", cifar_rows)
    base["mild"] = forget_metrics(mild_parent, mild_parent, forget_labels, target)
    base["cifar"] = utility_metrics(cifar_orig, cifar_parent, labels)
    report = {
        "parent": str(args.parent),
        "target_proxy": str(target_file),
        "baseline": base,
        "sweep": rows,
        "fit_rows": args.fit,
        "holdout_rows": args.holdout,
        "cifar_rows": [args.cifar_start, args.cifar_start + args.cifar_count],
        "seed": args.seed,
        "sweep_capture_seconds": sweep_capture_seconds,
        "sweep_capture_images_per_second": 3 * selected / sweep_capture_seconds,
        "final_capture_seconds": final_capture_seconds,
        "final_capture_images_per_second": (args.holdout + args.cifar_count) / final_capture_seconds,
        "head_sweep_seconds": sweep_seconds,
        "cifar_labelled": labels is not None,
        "objective": args.objective,
        "notes": "Mixes are synthetic, CIFAR labels are optional, and mild views share the forget holdout image IDs. Metrics are local proxies.",
    }
    if shortlist:
        candidate_weight, scores = shortlist
        scores = scores.copy()
        mild_candidate = predict_head(common, mild_features, candidate_weight, bias)
        cifar_candidate = predict_head(common, cifar_features, candidate_weight, bias)
        scores["mild"] = forget_metrics(mild_parent, mild_candidate, forget_labels, target)
        scores["cifar"] = utility_metrics(cifar_orig, cifar_candidate, labels)
        report["finalist"] = scores
        checks = {
            "mild_preserved": preserves_forget(scores["mild"], base["mild"]),
            "cifar_retain_improved": scores["cifar"]["retain_proxy"] > base["cifar"]["retain_proxy"],
            "cifar_agreement_preserved": scores["cifar"]["agree_orig"] >= base["cifar"]["agree_orig"] - 0.005,
            "cifar_labelled_accuracy_preserved": None if labels is None else (
                scores["cifar"]["accuracy"] >= base["cifar"]["accuracy"] - 0.005
            ),
        }
        report["finalist_checks"] = checks
        if all(check is not False for check in checks.values()):
            model.head.weight.copy_(candidate_weight)
            probe_rows = cifar_rows[:args.batch]
            _, native_logits = capture(common, model, cifar, probe_rows, args.batch)
            if (native_logits - cifar_candidate[:len(probe_rows)]).abs().max().item() > 0.04:
                raise RuntimeError("Native forward pass disagrees with cached head evaluation")
            report["checkpoint"] = export_checkpoint(state, candidate_weight, args.out)
    report["seconds"] = time.perf_counter() - start
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    rendered = json.dumps(report, indent=2, allow_nan=False)
    temporary = args.out / "metrics.json.part"
    temporary.write_text(rendered + "\n")
    temporary.replace(args.out / "metrics.json")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
