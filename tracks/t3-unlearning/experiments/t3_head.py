"""Fit classifier-head unlearning candidates using only provided T3 data."""

from pathlib import Path
import json
import math

import torch
import torch.nn.functional as F

import task_template as tt


TEMPERATURE = 2.0
BATCH = 128


def crop(x, left, top, size):
    return F.interpolate(
        x[:, :, top:top + size, left:left + size],
        size=(224, 224), mode="bilinear", align_corners=False, antialias=True,
    )


@torch.inference_mode()
def collect(model, data, order, partners):
    clean, mild, anchor, augmented = [], [], [], []
    for start in range(0, len(order), BATCH):
        x = tt.preprocess(data[order[start:start + BATCH]])
        partner = tt.preprocess(data[partners[start:start + BATCH]])

        def encode(view):
            return model.forward_head(model.forward_features(view), pre_logits=True)

        with torch.autocast("cuda", dtype=torch.float16):
            clean.append(encode(x).float())
            mild.append(encode(crop(x, 10, 10, 204)).float())
            anchor.append(encode(x.mul(0.8).add_(partner, alpha=0.2)).float())
            logits = [
                model.forward_head(model.forward_features(view)).float()
                for view in (crop(x, 25, 25, 174), crop(x, 29, 14, 165), crop(x, 14, 29, 165).flip(-1))
            ]
        augmented.append(torch.stack([(l / TEMPERATURE).softmax(dim=1) for l in logits]).mean(0))
    return tuple(torch.cat(group, dim=0) for group in (clean, mild, anchor, augmented))


def add_bias(x):
    return torch.cat((x, x.new_ones(len(x), 1)), dim=1)


def mean_kl(target, logits):
    log_prediction = F.log_softmax(logits / TEMPERATURE, dim=1)
    return float(F.kl_div(log_prediction, target, reduction="batchmean"))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("T3 benchmark requires CUDA")
    torch.set_num_threads(4)
    model = tt.load_model()
    images = tt.load_forget_images()
    gen = torch.Generator().manual_seed(20261121)
    order = torch.randperm(len(images), generator=gen)
    fit_count = int(0.8 * len(order))
    partners = torch.cat([ids.roll(83) for ids in (order[:fit_count], order[fit_count:])])
    print(f"Collecting frozen teacher features for {len(images)} inputs", flush=True)
    clean, mild, anchor, target = collect(model, images, order, partners)
    weight = torch.cat((model.head.weight.detach().T, model.head.bias.detach()[None]), dim=0)
    clean, mild, anchor = (add_bias(x) for x in (clean, mild, anchor))
    shifts = TEMPERATURE * target.clamp_min(1e-7).log()

    train = slice(0, fit_count)
    held = slice(fit_count, None)
    anchor_weight = 0.30
    train_x = torch.cat((clean[train], mild[train], math.sqrt(anchor_weight) * anchor[train]), dim=0)
    delta_targets = []
    for x in (clean, mild):
        delta = shifts[train] - x[train] @ weight
        delta_targets.append(delta - delta.mean(dim=1, keepdim=True))
    delta_targets.append(torch.zeros_like(delta_targets[0]))
    train_y = torch.cat(delta_targets, dim=0)
    ridge = 20
    delta = torch.linalg.solve(
        train_x.T @ train_x + ridge * torch.eye(train_x.shape[1], device=train_x.device),
        train_x.T @ train_y,
    )

    held_target = target[held]
    anchor_baseline = anchor[held] @ weight
    anchor_probs = (anchor_baseline / TEMPERATURE).softmax(dim=1)
    candidates = []
    for scale in (0.0, 0.15, 0.3, 0.45, 0.6, 0.85, 1.0):
        head = weight + scale * delta
        logits = anchor[held] @ head
        result = {
            "scale": scale,
            "forget_exact_proxy_kl": mean_kl(held_target, clean[held] @ head),
            "forget_mild_proxy_kl": mean_kl(held_target, mild[held] @ head),
            "retain_proxy_kl": mean_kl(anchor_probs, logits),
            "retain_proxy_top1_agreement": float((logits.argmax(1) == anchor_baseline.argmax(1)).float().mean()),
        }
        candidates.append(result)
        print(json.dumps(result), flush=True)
    feasible = [r for r in candidates if r["retain_proxy_top1_agreement"] >= 0.97]
    selected = min(feasible, key=lambda r: r["forget_exact_proxy_kl"] + r["forget_mild_proxy_kl"])
    if selected["scale"] <= 0 or not all(math.isfinite(v) for v in selected.values()):
        raise SystemExit("No nonzero candidate passed the local selection criterion")

    # Refit at the selected scale on all provided inputs. Keep the split metrics above.
    train_x = torch.cat((clean, mild, math.sqrt(anchor_weight) * anchor), dim=0)
    delta_targets = []
    for x in (clean, mild):
        diff = shifts - x @ weight
        delta_targets.append(diff - diff.mean(dim=1, keepdim=True))
    train_y = torch.cat((*delta_targets, torch.zeros_like(delta_targets[0])), dim=0)
    delta = torch.linalg.solve(
        train_x.T @ train_x + ridge * torch.eye(train_x.shape[1], device=train_x.device),
        train_x.T @ train_y,
    )
    state = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
    selected_head = (weight + selected["scale"] * delta).cpu()
    state["head.weight"] = selected_head[:-1].T.contiguous()
    state["head.bias"] = selected_head[-1].contiguous()
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state, strict=True)
    path = Path("candidate_head_t3.pt")
    temp = path.with_suffix(".part")
    torch.save(state, temp)
    assert temp.stat().st_size < 450_000_000
    temp.replace(path)
    Path("metrics_t3.json").write_text(json.dumps({
        "num_fit": fit_count, "num_held_out": len(images) - fit_count,
        "validation": "KL against surrogate augmentation targets; retention proxy uses image mixtures",
        "variants": candidates, "selected": selected,
        "file": str(path), "bytes": path.stat().st_size,
    }, indent=2) + "\n")
    print(f"Selected scale={selected['scale']}. Wrote {path} ({path.stat().st_size} bytes).", flush=True)


if __name__ == "__main__":
    main()
