"""T3 · calibration de la force : combien peut-on oublier sans casser le reste ?

Probleme : on ne peut pas mesurer la porte d'utilite (accuracy sur un jeu cache) et chaque soumission
coute 30 minutes. Solution : mettre 500 des 3500 images a oublier de cote et ne JAMAIS les utiliser
pour le desapprentissage. Si le modele les traite encore comme avant, c'est que les poids n'ont pas
derive globalement, donc que l'accuracy sur un jeu cache tient encore. Si elles changent aussi, c'est
du degat collateral.

On balaie ensuite la force (nombre de pas, blocs touches) et on choisit le point qui maximise l'oubli
sur les 3000 en gardant la derive quasi nulle sur les 500.

Usage : python t3_holdout.py --steps 1,2,4,8 --blocks 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from t3_ami import build_model, preprocess, teacher_targets, kl

TEMP = 2.0


@torch.no_grad()
def probs(model, images, batch=128):
    out = []
    for i in range(0, len(images), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(F.softmax(model(preprocess(images[i:i + batch].cuda(), "cuda")).float(), 1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--steps", default="1,2,4,8")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--holdout", type=int, default=500)
    a = ap.parse_args()

    d = Path(a.data)
    with np.load(d / "forget_images.npz", allow_pickle=False) as z:
        images = torch.from_numpy(z["images"].copy())
    n = len(images)
    hold = images[n - a.holdout:]
    train_imgs = images[: n - a.holdout]
    print(f"{len(train_imgs)} images pour le desapprentissage, {len(hold)} en observation", flush=True)

    base = build_model()
    base.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
    base = base.cuda().eval()
    p_hold0 = probs(base, hold)
    p_train0 = probs(base, train_imgs)

    targets = teacher_targets(base, train_imgs, "cuda", seed=0)

    results = {}
    for steps in [int(s) for s in a.steps.split(",") if s]:
        model = build_model()
        model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
        model = model.cuda()
        depth = len(model.blocks)
        for i, blk in enumerate(model.blocks):
            if i < depth - a.blocks:
                for p in blk.parameters():
                    p.requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

        done = 0
        n_done = 0
        perm = torch.randperm(len(train_imgs))
        while n_done < steps:
            idx = perm[(n_done * a.batch) % len(train_imgs):][: a.batch]
            if len(idx) == 0:
                perm = torch.randperm(len(train_imgs))
                idx = perm[: a.batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(preprocess(train_imgs[idx].cuda(), "cuda")).float()
            loss = kl(targets[idx].cuda(), F.softmax(logits / TEMP, 1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            n_done += 1
            done += float(loss)

        model.eval()
        p_hold = probs(model, hold)
        p_train = probs(model, train_imgs)
        r = {
            "pas": steps,
            "kl_holdout": round(float(kl(p_hold0, p_hold)), 4),
            "accord_holdout": round(float((p_hold0.argmax(1) == p_hold.argmax(1)).float().mean()), 4),
            "kl_cibles": round(float(kl(p_train0, p_train)), 4),
            "confiance_cibles": round(float(p_train.max(1).values.mean()), 4),
            "confiance_holdout": round(float(p_hold.max(1).values.mean()), 4),
        }
        results[steps] = r
        print(json.dumps(r), flush=True)
        torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()},
                   f"t3_cand_steps{steps}.pt")
        del model
        torch.cuda.empty_cache()

    Path("t3_holdout.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
