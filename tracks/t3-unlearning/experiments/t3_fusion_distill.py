"""Distill a CE-preserving posterior ensemble into SOUPs1 on forget views."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from t3_output_fusion import ce_preserving_mix, suppress_original
from t3_ridge_repair import (
    forget_metrics, preserves_forget, read_cache, reference, utility_metrics,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--members", nargs="+", type=Path, required=True)
    parser.add_argument("--strength", default=.65, type=float)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--batch", default=224, type=int)
    parser.add_argument("--blocks", default=2, type=int, choices=(2, 4))
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--seed", default=20261122, type=int)
    args = parser.parse_args()
    args.base = args.base.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    return args


def kld(logits, probabilities):
    return F.kl_div((logits / 2).log_softmax(1), probabilities, reduction="batchmean")


def choose_rows(length, rng, count=3072):
    return torch.randperm(length, generator=rng)[:count]


def moments_unchanged(metrics, baseline):
    return (
        preserves_forget(metrics, baseline)
        and abs(metrics["ce_mean"] - baseline["ce_mean"]) <= .08
        and abs(metrics["ce_std"] - baseline["ce_std"]) <= .08
    )


def main():
    args = arguments()
    os.environ.update(T3BASE=str(args.base), T3ME=str(args.base))
    sys.path.insert(0, str(args.base / "code"))
    import common

    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    state = torch.load(args.base / "inbox/SOUPs1.pt", map_location="cpu", weights_only=True)
    model = common.tt.build_model()
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    trainable_blocks = tuple(f"blocks.{i}." for i in range(12 - args.blocks, 12))
    for name, param in model.named_parameters():
        param.requires_grad_(
            name.startswith("head.") or (
                name.startswith(trainable_blocks)
                and (".attn." in name or ".mlp." in name)
            )
        )

    def tail(features):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return common.top(model, features, args.blocks).float()

    @torch.no_grad()
    def capture(images, rows):
        result = []
        for idx in rows.split(args.batch):
            pixels = images[idx] if torch.is_tensor(images) else torch.from_numpy(images[idx.numpy()])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result.append(common.trunk(model, common.norm(common.to01(pixels)), args.blocks).to(torch.bfloat16))
        return torch.cat(result)

    @torch.no_grad()
    def predict(features):
        return torch.cat([tail(part) for part in features.split(args.batch)])

    cache = args.base / "cache"
    exact = common.load_forget()
    mixes = read_cache(cache, "mixes")
    cifar = read_cache(cache, "ood_cifar100")
    forget_rows = choose_rows(len(exact), rng)
    retain_rows = choose_rows(len(mixes), rng)
    cifar_rows = torch.arange(1024, 1408)
    training_size = 2450
    fit = slice(0, training_size)
    test = slice(training_size, None)
    pixels = {
        "exact": (exact, forget_rows),
        "jpeg": (read_cache(cache, "held_jpeg50"), forget_rows),
        "mild": (read_cache(cache, "held_mild"), forget_rows[test]),
        "rot15": (read_cache(cache, "held_rot15"), forget_rows[test]),
        "mix": (mixes, retain_rows),
        "cifar": (cifar, cifar_rows),
    }
    features, parent = {}, {}
    feature_count = 0
    for name, (images, rows) in pixels.items():
        features[name] = capture(images, rows)
        parent[name] = predict(features[name])
        feature_count += len(rows)
        print(f"Cached {name}: {len(rows)} views", flush=True)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - start
    print(f"Captured {feature_count} views in {capture_seconds:.3f}s", flush=True)
    labels = reference(cache, "exact", forget_rows).argmax(-1)
    orig_mix = reference(cache, "mixes", retain_rows)
    orig_cifar = reference(cache, "ood_cifar100", cifar_rows)
    targets = torch.load(
        cache / "tgt_RDt24r_70_l1016.pt", map_location="cpu", weights_only=True
    )["z"][forget_rows[test]].cuda()
    p_teacher_mix = (orig_mix[fit] / 2).softmax(1)
    train_names = ("exact", "jpeg")
    conditional = {name: [] for name in train_names}
    for member in args.members:
        model.load_state_dict(torch.load(member.expanduser(), map_location="cpu", weights_only=True), strict=True)
        for name in train_names:
            images, rows = pixels[name]
            prediction = predict(capture(images, rows[fit]))
            conditional[name].append(suppress_original((prediction / 2).softmax(1), labels[fit, None]))
        print(f"Added {member.name} to the teacher", flush=True)
    model.load_state_dict(state, strict=True)
    p_teacher_forget = {
        name: (ce_preserving_mix(parent[name][fit], torch.stack(conditional[name]).mean(0),
                                  labels[fit, None], args.strength) / 2).softmax(1)
        for name in train_names
    }
    if args.conditional:
        p_teacher_forget = {
            name: suppress_original(probabilities, labels[fit, None])
            for name, probabilities in p_teacher_forget.items()
        }
    teacher_ce = {
        k: F.cross_entropy(parent[k][fit], labels[fit], reduction="none") for k in ("exact", "jpeg")
    }
    baseline = {
        k: forget_metrics(parent[k][test], parent[k][test], labels[test], targets)
        for k in ("exact", "jpeg")
    }
    baseline["mild"] = forget_metrics(parent["mild"], parent["mild"], labels[test], targets)
    baseline["rot15"] = forget_metrics(parent["rot15"], parent["rot15"], labels[test], targets)
    baseline["mix"] = utility_metrics(orig_mix[test], parent["mix"][test])
    baseline["cifar"] = utility_metrics(orig_cifar, parent["cifar"])
    selected = None
    best_gain = 0.0
    history = []
    mutable = [name for name, param in model.named_parameters() if param.requires_grad]

    @torch.no_grad()
    def evaluate(epoch, penalty):
        exact_new = predict(features["exact"][test])
        jpeg_new = predict(features["jpeg"][test])
        mild_new = predict(features["mild"])
        scores = {
            "penalty": penalty,
            "epoch": epoch,
            "exact": forget_metrics(parent["exact"][test], exact_new, labels[test], targets),
            "jpeg": forget_metrics(parent["jpeg"][test], jpeg_new, labels[test], targets),
            "mild": forget_metrics(parent["mild"], mild_new, labels[test], targets),
            "rot15": forget_metrics(parent["rot15"], predict(features["rot15"]), labels[test], targets),
            "mix": utility_metrics(orig_mix[test], predict(features["mix"][test])),
        }
        scores["preserves_forget"] = all(
            moments_unchanged(scores[k], baseline[k]) for k in ("exact", "jpeg")
        )
        return scores

    penalties = (0.3, 0.8) if args.conditional else (1.0, 2.0)
    for penalty in penalties:
        model.load_state_dict(state, strict=True)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0
        )
        for epoch in range(1, args.epochs + 1):
            mix_batches = torch.randperm(training_size, generator=rng).cuda().split(args.batch)
            forget_batches = torch.randperm(training_size, generator=rng).cuda().split(args.batch)
            for step, (mix_rows, hold_rows) in enumerate(zip(mix_batches, forget_batches, strict=True)):
                kind = "exact" if (step + epoch) % 2 else "jpeg"
                optimizer.zero_grad(set_to_none=True)
                logits_retain = tail(features["mix"][mix_rows])
                logits_forget = tail(features[kind][hold_rows])
                retain = kld(logits_retain, p_teacher_mix[mix_rows])
                probabilities = p_teacher_forget[kind][hold_rows]
                if args.conditional:
                    pseudo_label = labels[hold_rows, None]
                    false_logits = (logits_forget / 2).scatter(1, pseudo_label, -1e4)
                    log_probs = false_logits.log_softmax(1)
                    preserve = F.kl_div(log_probs, probabilities, reduction="batchmean")
                else:
                    preserve = kld(logits_forget, probabilities)
                ce = F.cross_entropy(logits_forget, labels[hold_rows], reduction="none")
                ce_weight = 1.25 if args.conditional else 0.25
                forget = preserve + ce_weight * (ce - teacher_ce[kind][hold_rows]).square().mean()
                (penalty * retain + forget).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch not in {1, 2, 3, 6, args.epochs}:
                continue
            scores = evaluate(epoch, penalty)
            history.append(scores)
            print(json.dumps(scores), flush=True)
            gain = sum(baseline[k]["kl_target"] - scores[k]["kl_target"] for k in train_names) / 2
            mix_ok = scores["mix"]["retain_proxy"] >= baseline["mix"]["retain_proxy"] - .006
            if scores["preserves_forget"] and mix_ok and gain > best_gain:
                selected = {
                    "weights": {name: model.state_dict()[name].detach().cpu().clone() for name in mutable},
                    "scores": scores,
                }
                best_gain = gain
    result = {
        "baseline": baseline, "history": history, "seconds": time.perf_counter() - start,
        "blocks": args.blocks, "lr": args.lr, "strength": args.strength,
        "conditional": args.conditional,
    }
    result["feature_images_per_second"] = feature_count / capture_seconds
    result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    if selected:
        model.load_state_dict({**state, **selected["weights"]}, strict=True)
        scores = copy.deepcopy(selected["scores"])
        scores["cifar"] = utility_metrics(orig_cifar, predict(features["cifar"]))
        result["finalist"] = scores
        out_of_family = ("mild", "rot15")
        transfer = all(
            scores[k]["kl_target"] < baseline[k]["kl_target"] and moments_unchanged(scores[k], baseline[k])
            for k in out_of_family
        )
        scores["transfer"] = transfer
        if transfer and scores["cifar"]["retain_proxy"] >= baseline["cifar"]["retain_proxy"] - .006:
            checkpoint = args.out / "fusion_distill.pt"
            fixed = {**state, **selected["weights"]}
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in fixed.values()):
                raise ValueError("Invalid checkpoint tensor")
            torch.save(fixed, checkpoint)
            result["checkpoint"] = str(checkpoint)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: val for key, val in result.items() if key != "history"}), flush=True)


if __name__ == "__main__":
    main()
