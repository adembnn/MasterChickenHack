"""Tache 3, desapprentissage : viser le comportement d'un modele qui n'a jamais vu l'image.

Le piege du score : le gold n'est pas un modele detruit, c'est la moyenne de modeles reentraines
sans les images a oublier. Il garde donc une distribution structuree sur ces images, avec parfois
la bonne classe, parfois une classe fausse precise. Detruire la confiance fait chuter
prediction_score, qui est une KL par image.

Idee retenue : l'image augmentee fortement (recadrage severe, flou, changement leger de couleur)
n'a pas ete memorisee, donc la prediction du modele original dessus est deja proche de ce qu'un
modele qui n'a jamais vu l'image predirait. On distille cette vue augmentee vers l'image d'origine.
C'est une cible gratuite, sans labels, et qui respecte la structure des classes.

Modes, du plus prudent au plus agressif :
  distill       distillation de la vue augmentee vers l'original (par defaut)
  randomlabel   labels aleatoires, casse la memoire, la reference SCRUB sans jeu de retention
  ascent        montee de gradient sur la perte, avec arret precoce
  blend         distill plus une petite montée de gradient

Usage :
  python t3_unlearn.py --mode distill --epochs 2 --out submission.pt
  python t3_unlearn.py --mode randomlabel --epochs 3 --out sub_rl.pt
Le script affiche la perte moyenne sur les images a oublier avant et apres, et l'accord avec le
modele original sur les vues augmentees. Si l'accord sur les vues augmente et que la perte sur les
images remonte vers celle des vues, on est dans la bonne direction.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import task_template as tt


def make_aug():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.30, 0.65), ratio=(0.75, 1.333), antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.RandomApply([T.GaussianBlur(9, sigma=(1.0, 3.0))], p=0.6),
    ])


@torch.no_grad()
def soft(model, pixels_u8, temp=2.0, bs=64):
    out = []
    for i in range(0, len(pixels_u8), bs):
        x = tt.preprocess(pixels_u8[i:i + bs])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(tt.DEVICE).startswith("cuda")):
            out.append(F.softmax(model(x).float() / temp, dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def mean_entropy(model, pixels_u8, bs=64):
    """Entropie moyenne des predictions, faute de labels. Repere d'evolution seulement : une entropie
    qui s'envole veut dire que le modele ne croit plus a rien, ce qui est justement le piege."""
    tot, n = 0.0, 0
    for i in range(0, len(pixels_u8), bs):
        x = tt.preprocess(pixels_u8[i:i + bs])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(tt.DEVICE).startswith("cuda")):
            p = F.softmax(model(x).float(), dim=1)
        tot += float(-(p * (p + 1e-8).log()).sum(1).mean()) * len(p)
        n += len(p)
    return tot / max(n, 1)


def unlearn(model, forget, mode="distill", epochs=2, lr=1e-5, bs=32, temp=2.0, ascent_weight=0.0):
    model = model.train()
    teacher = model.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # On ne touche qu'aux derniers blocs et a la tete : c'est la memoire recente des images.
    for name, p in model.named_parameters():
        p.requires_grad_(any(k in name for k in ("head", "blocks.9", "blocks.10", "blocks.11", "norm")))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    aug = make_aug()
    n = len(forget)
    steps = epochs * ((n + bs - 1) // bs)

    print(f"mode {mode}, {sum(p.numel() for p in params)} parametres entrainables, {steps} pas")
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x_img = forget[idx]
            x = tt.preprocess(x_img)
            with torch.no_grad():
                aug_imgs = torch.stack([aug(x_img[j]) for j in range(len(idx))])
                target_aug = F.softmax(teacher(tt.preprocess(aug_imgs)).float() / temp, dim=1)
                target_orig = F.softmax(teacher(x).float() / temp, dim=1)
            logits = model(x)

            def kl_to(target):
                return F.kl_div(F.log_softmax(logits / temp, dim=1), target, reduction="batchmean") * (temp ** 2)

            if mode == "distill":
                loss = kl_to(target_aug)
            elif mode == "randomlabel":
                labels = torch.randint(0, tt.NUM_CLASSES, (len(idx),), device=logits.device)
                loss = F.cross_entropy(logits, labels)
            elif mode == "ascent":
                loss = -kl_to(target_orig)
            else:
                loss = kl_to(target_aug) - ascent_weight * kl_to(target_orig)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == steps:
                print(f"  ep {ep} pas {step}/{steps} loss {float(loss):.4f}", flush=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="distill", choices=["distill", "randomlabel", "ascent", "blend"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ascent-weight", type=float, default=0.05)
    ap.add_argument("--out", default="submission.pt")
    a = ap.parse_args()

    print("chargement du modele et des images a oublier")
    model = tt.load_model(Path("model.pt"))
    forget = tt.load_forget_images(Path("forget_images.npz"))

    before = mean_entropy(model, forget)
    aug = make_aug()
    ref = tt.predict(model, forget[:200]).argmax(1)
    ref_aug = tt.predict(model, torch.stack([aug(forget[i]) for i in range(200)])).argmax(1)
    print(f"entropie moyenne sur les images a oublier (avant) : {before:.4f}")
    print(f"accord original / vue augmentee sur 200 images : {float((ref == ref_aug).float().mean()):.3f}")

    model = unlearn(model, forget, a.mode, a.epochs, a.lr, a.bs, ascent_weight=a.ascent_weight)

    after = mean_entropy(model, forget)
    new_ref = tt.predict(model, forget[:200]).argmax(1)
    new_aug = tt.predict(model, torch.stack([aug(forget[i]) for i in range(200)])).argmax(1)
    print(f"entropie moyenne (apres) : {after:.4f}, ecart {after - before:+.4f}")
    print(f"accord apres / vue augmentee : {float((new_ref == new_aug).float().mean()):.3f}")
    print(f"images dont la prediction a change : {float((ref != new_ref).float().mean()):.3f}")

    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(sd, a.out)
    print("ecrit", a.out, sum(v.numel() for v in sd.values()) // 1_000_000, "millions de parametres")


if __name__ == "__main__":
    main()
