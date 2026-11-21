"""Measure confidence drift against original predictions on the T3 holdout."""

from pathlib import Path
import json

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import task_template as tt
from t3_head import crop


@torch.inference_mode()
def predict_views(model, images):
    logits = {"exact": [], "rotation_15": [], "blur_sigma2": [], "mild_crop": [], "flip": []}
    for batch in images.split(96):
        x = tt.preprocess(batch)
        views = (
            x,
            TF.rotate(x, 15, interpolation=TF.InterpolationMode.BILINEAR),
            TF.gaussian_blur(x, [11, 11], [2.0, 2.0]),
            crop(x, 10, 10, 204),
            x.flip(-1),
        )
        with torch.autocast("cuda", dtype=torch.float16):
            for scores, image in zip(logits.values(), views):
                scores.append(model(image).float())
    return {name: torch.cat(scores) for name, scores in logits.items()}


def main():
    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.set_num_threads(4)
    model = tt.load_model()
    images = tt.load_forget_images()
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(20261121))
    held = images[order[2800:]]
    original = predict_views(model, held)
    pseudo_labels = original["exact"].argmax(dim=1)
    variants = [
        ("orig", None),
        ("suffix_v1", "candidate_suffix_t3.pt"),
        ("suffix_v2", "candidate_suffix_v2_t3.pt"),
        ("head_refit", "candidate_head_t3.pt"),
    ]
    if Path("candidate_suffix_stress_t3.pt").is_file():
        variants.append(("suffix_stress", "candidate_suffix_stress_t3.pt"))
    report = {}
    for name, file in variants:
        scores = original
        if file:
            weights = torch.load(file, weights_only=True, map_location="cpu")
            model.load_state_dict(weights, strict=True)
            scores = predict_views(model, held)
        metrics = {}
        for view_name, logits in scores.items():
            assert torch.isfinite(logits).all()
            ce = F.cross_entropy(logits, pseudo_labels, reduction="none")
            metrics[view_name] = {
                "ce_mean": float(ce.mean()),
                "ce_std": float(ce.std(unbiased=False)),
                "argmax_diff_vs_orig_exact": float((logits.argmax(1) != pseudo_labels).float().mean()),
            }
        report[name] = metrics
        print(name, json.dumps(metrics), flush=True)
    Path("diagnostic_ce_t3.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
