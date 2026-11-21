"""T3 · cinquieme essai : supprimer la classe memorisee, laisser le reste decider.

Ce que dit la formule officielle, maintenant qu'on l'a lue. Le gold est un modele reentraine sans ces
images : sur une image a oublier, il predit la classe des images retenues qui lui ressemblent, donc
typiquement la deuxieme ou troisieme classe du modele original, pas la premiere. Sa perte sur l'image
n'est donc pas nulle, elle est celle d'une erreur plausible.

Nos deux premiers essais distillaient vers le modele original vu sous augmentation : la classe
memorisee reste la premiere, la perte reste proche de zero, donc le score moyenne/ecart-type est mauvais
et le score de prediction aussi. D'ou le zero.

Ici on force la sortie a oublier sa premiere classe : on met sa masse a zero et on renormalise le
reste. Le reseau apprend une distribution pointee sur une classe plausible mais differente.

Usage : python t3_suppress.py --data . --steps 8 --keep 0.0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from t3_ami import build_model, preprocess, kl

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
    ap.add_argument("--out", default="submission_t3_suppress.pt")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep", type=float, default=0.0, help="masse laissee sur la classe memorisee")
    ap.add_argument("--holdout", type=int, default=500)
    a = ap.parse_args()

    d = Path(a.data)
    with np.load(d / "forget_images.npz", allow_pickle=False) as z:
        images = torch.from_numpy(z["images"].copy())
    n = len(images)
    hold = images[n - a.holdout:]
    train_imgs = images[: n - a.holdout]

    base = build_model()
    base.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
    base = base.cuda().eval()

    p0 = probs(base, train_imgs)
    top = p0.argmax(1)
    t = p0.clone()
    t[torch.arange(len(t)), top] = a.keep
    t = t / t.sum(1, keepdim=True)
    m = float(p0.max(1).values.mean())
    print(f"classe memorisee supprimee sur {len(t)} images, masse gardee {a.keep}, confiance avant {m:.4f}", flush=True)

    p_hold0 = probs(base, hold)

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

    perm = torch.randperm(len(train_imgs))
    for s in range(a.steps):
        idx = perm[(s * a.batch) % len(train_imgs):][: a.batch]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(preprocess(train_imgs[idx].cuda(), "cuda")).float()
        loss = kl(t[idx].cuda(), F.softmax(logits / TEMP, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        print(f"  pas {s+1} perte {float(loss):.5f}", flush=True)

    model.eval()
    p1 = probs(model, train_imgs)
    p_hold = probs(model, hold)
    metrics = {
        "pas": a.steps,
        "keep": a.keep,
        "confiance_apres": round(float(p1.max(1).values.mean()), 4),
        "entropie_apres": round(float(-(p1 * p1.clamp_min(1e-9).log()).sum(1).mean()), 4),
        "kl_vs_original_oublie": round(float(kl(p0, p1)), 4),
        "meme_classe_qu_avant": round(float((p1.argmax(1) == top).float().mean()), 4),
        "kl_holdout": round(float(kl(p_hold0, p_hold)), 4),
        "accord_holdout": round(float((p_hold0.argmax(1) == p_hold.argmax(1)).float().mean()), 4),
    }
    print(json.dumps(metrics), flush=True)
    Path("metrics_t3_suppress.json").write_text(json.dumps(metrics, indent=2))
    torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, a.out)
    print("ecrit", a.out)


if __name__ == "__main__":
    main()
