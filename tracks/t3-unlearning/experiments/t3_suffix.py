"""Unlearn T3 by distilling augmented targets into the last two ViT blocks."""

from pathlib import Path
import argparse
import copy
import json
import time

import torch
import torch.nn.functional as F

import task_template as tt
from t3_head import TEMPERATURE, crop, mean_kl


def prefix(model, image):
    x = model.patch_embed(image)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    return model.blocks[:-2](x)


def suffix(model, feature):
    x = model.blocks[-2:](feature)
    return model.forward_head(model.norm(x)).float()


@torch.no_grad()
def collect(model, images, order, partners):
    clean, mild, anchor, targets, anchors = [], [], [], [], []
    for start in range(0, len(order), 96):
        indices = slice(start, start + 96)
        x = tt.preprocess(images[order[indices]])
        y = tt.preprocess(images[partners[indices]])
        with torch.autocast("cuda", dtype=torch.float16):
            views = (x, crop(x, 10, 10, 204), x * 0.8 + y * 0.2)
            c, m, a = [prefix(model, view).to(torch.float16) for view in views]
            if start == 0:
                reference = model(x).float()
                torch.testing.assert_close(reference, suffix(model, c), atol=0.025, rtol=0.02)
            clean.append(c)
            mild.append(m)
            anchor.append(a)
            anchors.append(suffix(model, a).div(TEMPERATURE).softmax(dim=1))
            strong_views = (crop(x, 25, 25, 174), crop(x, 29, 14, 165), crop(x, 14, 29, 165).flip(-1))
            probabilities = [(model(view).float() / TEMPERATURE).softmax(dim=1) for view in strong_views]
            targets.append(torch.stack(probabilities, dim=0).mean(0))
    return tuple(torch.cat(group, dim=0) for group in (clean, mild, anchor, targets, anchors))


@torch.no_grad()
def predict(model, features):
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.cat([suffix(model, batch) for batch in features.split(96)])


def measure(model, clean, mild, anchor, target, anchor_probs):
    clean_logits, mild_logits, anchor_logits = (predict(model, feature) for feature in (clean, mild, anchor))
    return {
        "forget_exact_proxy_kl": mean_kl(target, clean_logits),
        "forget_mild_proxy_kl": mean_kl(target, mild_logits),
        "retain_proxy_kl": mean_kl(anchor_probs, anchor_logits),
        "retain_proxy_top1_agreement": float((anchor_logits.argmax(1) == anchor_probs.argmax(1)).float().mean()),
    }


def optimize(model, clean, mild, anchor, target, anchor_probs, count, generator, args):
    model.requires_grad_(False)
    trainable = list(model.blocks[-2:].parameters()) + list(model.norm.parameters()) + list(model.head.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    for epoch in range(args.epochs):
        running_loss = []
        for ids in torch.randperm(count, generator=generator).split(64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, m, a = suffix(model, torch.cat((clean[ids], mild[ids], anchor[ids]))).chunk(3)
            forget = (
                F.kl_div((c / TEMPERATURE).log_softmax(1), target[ids], reduction="batchmean")
                + F.kl_div((m / TEMPERATURE).log_softmax(1), target[ids], reduction="batchmean")
            ) * 0.5
            keep = F.kl_div((a / TEMPERATURE).log_softmax(1), anchor_probs[ids], reduction="batchmean")
            loss = (forget + args.anchor_weight * keep) * TEMPERATURE ** 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            running_loss.append(loss.detach())
        print(f"epoch {epoch+1}: loss={torch.stack(running_loss).mean().item():.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("candidate_suffix_t3.pt"))
    args = parser.parse_args()
    if args.lr <= 0 or args.epochs < 1 or args.anchor_weight < 0 or args.out.suffix != ".pt":
        parser.error("Use a positive learning rate and epoch count, nonnegative anchor weight and .pt output")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(20261121)
    generator = torch.Generator().manual_seed(20261121)
    original = tt.load_model()
    images = tt.load_forget_images()
    order = torch.randperm(len(images), generator=generator)
    count = int(len(images) * 0.8)
    partners = torch.cat([part.roll(83) for part in (order[:count], order[count:])])
    print(f"Caching {len(images)} inputs for last-two-block distillation.", flush=True)
    clean, mild, anchor, target, anchor_probs = collect(original, images, order, partners)
    held = (clean[count:], mild[count:], anchor[count:], target[count:], anchor_probs[count:])
    model = copy.deepcopy(original)
    baseline = measure(original, *held)
    print(json.dumps({"scale": 0.0, **baseline}), flush=True)
    optimize(model, clean, mild, anchor, target, anchor_probs, count, generator, args)

    untouched = dict(original.named_parameters())
    tuned = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def interpolate(scale):
        for name, value in model.named_parameters():
            if name in tuned:
                value.copy_(torch.lerp(untouched[name].detach(), tuned[name], scale))

    variants = [{"scale": 0.0, **baseline}]
    for scale in (0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
        interpolate(scale)
        result = {"scale": scale, **measure(model, *held)}
        variants.append(result)
        print(json.dumps(result), flush=True)
    retained = [r for r in variants if r["retain_proxy_top1_agreement"] >= 0.97]
    best = min(retained, key=lambda r: r["forget_exact_proxy_kl"] + r["forget_mild_proxy_kl"])
    if best["scale"] <= 0:
        raise SystemExit("No nonzero T3 variant met the local criterion.")
    interpolate(best["scale"])
    state = {name: tensor.detach().cpu().float() for name, tensor in model.state_dict().items()}
    assert all(torch.isfinite(tensor).all() for tensor in state.values())
    reference = original.state_dict()
    changed = [name for name in state if not torch.equal(state[name], reference[name].cpu())]
    assert changed and set(changed).issubset(tuned)
    original.load_state_dict(state, strict=True)
    target_path = args.out
    temp = target_path.with_suffix(".part")
    torch.save(state, temp)
    assert temp.stat().st_size < 450_000_000
    temp.replace(target_path)
    summary = {"fit": count, "held": len(images)-count, "variants": variants, "selected": best,
               "lr": args.lr, "epochs": args.epochs, "anchor_weight": args.anchor_weight,
               "bytes": target_path.stat().st_size, "changed_tensors": len(changed),
               "elapsed_seconds": round(time.monotonic()-started, 1)}
    target_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {target_path} ({summary['bytes']} bytes) in {summary['elapsed_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
