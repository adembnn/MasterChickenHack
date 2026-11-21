"""Calibrate positive output scaling for the T3 suffix checkpoint."""

from pathlib import Path
import argparse
import json

import torch
import torch.nn.functional as F

import task_template as tt


@torch.inference_mode()
def predictions(model, exact_pixels, anchor_pixels):
    exact, anchor = [], []
    for x, y in zip(exact_pixels.split(24), anchor_pixels.split(24)):
        a, b = tt.preprocess(x), tt.preprocess(y)
        with torch.autocast(
            "cuda", dtype=torch.float16, enabled=torch.cuda.is_available(),
        ):
            exact.append(model(a).float())
            anchor.append(model(a.mul(0.8).add_(b, alpha=0.2)).float())
    return torch.cat(exact), torch.cat(anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("candidate_suffix_temp_t3.pt"))
    args = parser.parse_args()
    if args.out.suffix != ".pt":
        parser.error("Use a .pt output path.")
    torch.set_num_threads(6)
    data = tt.load_forget_images()
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(20261121))
    held = order[2800:]
    exact, other = data[held[:192]], data[held.roll(83)[:192]]
    model = tt.load_model()
    original, anchor_original = predictions(model, exact, other)
    pseudo = original.argmax(1)
    file = Path("candidate_suffix_v2_t3.pt")
    source = torch.load(file, map_location="cpu", weights_only=True)
    model.load_state_dict(source, strict=True)
    base, anchor_base = predictions(model, exact, other)
    target_anchor = (anchor_original / 2).softmax(1)
    # A moderate target derived from the team's S1 diagnostic, not competition labels.
    goal = 0.12
    variants = []
    for scale in (1.0, 0.85, 0.75, 0.65, 0.60, 0.55, 0.50, 0.47, 0.44, 0.40, 0.35):
        x, a = base * scale, anchor_base * scale
        ce = F.cross_entropy(x, pseudo, reduction="none")
        report = {
            "scale": scale,
            "exact_ce_mean": float(ce.mean()),
            "exact_ce_std": float(ce.std(unbiased=False)),
            "agreement_with_base_exact": float((x.argmax(1) == base.argmax(1)).float().mean()),
            "anchor_agreement_with_original": float((a.argmax(1) == anchor_original.argmax(1)).float().mean()),
            "anchor_kl": float(F.kl_div((a / 2).log_softmax(1), target_anchor, reduction="batchmean")),
        }
        print(json.dumps(report), flush=True)
        variants.append(report)
    selected = min(variants, key=lambda row: abs(row["exact_ce_mean"] - goal))
    scale = selected["scale"]
    assert 0 < scale < 1 and selected["agreement_with_base_exact"] == 1
    for key in ("head.weight", "head.bias"):
        source[key].mul_(scale)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in source.values())
    model.load_state_dict(source, strict=True)
    after_exact, _ = predictions(model, exact, other)
    assert torch.isfinite(after_exact).all()
    agreement = float((after_exact.argmax(1) == base.argmax(1)).float().mean())
    assert agreement >= 0.995
    output = args.out
    temp = output.with_suffix(".part")
    torch.save(source, temp)
    assert temp.stat().st_size < 450_000_000
    temp.replace(output)
    output.with_suffix(".json").write_text(json.dumps({
        "n": len(exact), "pseudo_label_ce_target": goal,
        "variants": variants, "selected": selected,
        "end_to_end_top1_agreement": agreement,
        "device": tt.DEVICE,
    }, indent=2) + "\n")
    print("Selected", json.dumps(selected), "file", output, flush=True)


if __name__ == "__main__":
    main()
