"""T3, quatrieme essai : rendre uniforme la sortie sur les images memorisees seulement.

Raisonnement. Le gold est un modele reentraine sans ces images : sur elles, il n'a aucune raison d'etre
confiant, sa distribution ressemble a ce qu'un modele donne sur une image qu'il n'a jamais vue, donc
etalee, souvent proche du prior de classes. Nos deux essais precedents ont pousse le modele vers une
distribution structuree mais fausse, et ont fait zero. Ici on vise l'inverse : etaler la sortie sur les
images a oublier, en gardant le reste du reseau intact.

Garde-fou : seuls les deux derniers blocs et la tete bougent, penalite L2 forte vers les poids d'origine,
peu d'epoques. On mesure ensuite deux choses : l'ecart au modele original sur les images a oublier
(doit etre gros) et sur des vues augmentees des memes images (proxy du reste du monde, doit rester petit).

Usage : python t3_uniform.py --data . --blocks 2 --epochs 4 --lr 1e-4 --smooth 0.9
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from t3_ami import build_model, preprocess, augmented_view, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="submission_t3_uniform.pt")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smooth", type=float, default=0.9, help="melange vers l'uniforme : 1 = uniforme pur")
    ap.add_argument("--l2sp", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    d = Path(a.data)
    model = build_model()
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
    model = model.cuda()
    w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

    depth = len(model.blocks)
    for i, blk in enumerate(model.blocks):
        if i < depth - a.blocks:
            for p in blk.parameters():
                p.requires_grad_(False)

    with np.load(d / "forget_images.npz", allow_pickle=False) as z:
        images = torch.from_numpy(z["images"].copy())

    n = len(images)
    target = torch.full((n, 100), (1 - a.smooth) / 100, device="cuda")
    target[:, 0] += a.smooth        # une classe porte presque tout, puis on etale : voir plus bas
    # en realite on veut la distribution uniforme melangee a la sortie d'origine, pas a une classe fixe
    with torch.no_grad():
        p0 = []
        for i in range(0, n, 128):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p0.append(F.softmax(model(preprocess(images[i:i + 128].cuda(), "cuda")).float(), 1).cpu())
        p0 = torch.cat(p0)
    target = ((1 - a.smooth) * p0 + a.smooth * torch.full_like(p0, 1.0 / 100)).cuda()
    m0 = float(p0.max(1).values.mean())
    print(f"confiance avant {m0:.4f}, cible melangee a {a.smooth} vers l'uniforme", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(preprocess(images[idx].cuda(), "cuda")).float()
            loss = kl(target[idx], F.softmax(logits, 1))
            reg = sum(((v - w0[k].to("cuda")) ** 2).sum() for k, v in model.state_dict().items() if v.requires_grad)
            loss = loss + a.l2sp * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        print(f"epoque {ep} perte {tot/n:.5f} ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        p1, pa = [], []
        for i in range(0, n, 128):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p1.append(F.softmax(model(preprocess(images[i:i + 128].cuda(), "cuda")).float(), 1).cpu())
        p1 = torch.cat(p1)

        gen = torch.Generator().manual_seed(0)
        sub = images[:800]
        views = torch.stack([augmented_view(preprocess(sub[i:i + 1].cuda(), "cuda")[0], gen)
                             for i in range(len(sub))], 0)
        for i in range(0, len(views), 64):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pa.append(F.softmax(model(views[i:i + 64]).float(), 1).cpu())
        pa = torch.cat(pa)

    orig = build_model()
    orig.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
    orig.cuda().eval()
    with torch.no_grad():
        po = []
        for i in range(0, len(views), 64):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                po.append(F.softmax(orig(views[i:i + 64]).float(), 1).cpu())
        po = torch.cat(po)

    metrics = {
        "confiance_avant": round(m0, 4),
        "confiance_apres": round(float(p1.max(1).values.mean()), 4),
        "entropie_apres": round(float(-(p1 * p1.clamp_min(1e-9).log()).sum(1).mean()), 4),
        "kl_oublie_vs_original": round(float(kl(p0, p1)), 4),
        "kl_vues_vs_original": round(float(kl(po, pa)), 4),
        "accord_top1_oublie": round(float((p0.argmax(1) == p1.argmax(1)).float().mean()), 4),
    }
    print(json.dumps(metrics), flush=True)
    Path("metrics_t3_uniform.json").write_text(json.dumps(metrics, indent=2))
    torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, a.out)
    print("ecrit", a.out)


if __name__ == "__main__":
    main()
