# Brief recherche : TOP1 T1 + T3 (EuroAI Championship Paris 2026)

**Pour :** agent online (recherche X / Twitter + GitHub + arXiv).  
**Équipe :** MasterChickenHack.  
**Deadline code :** dimanche 22/11/2026 12:00 (Paris).  
**But :** trouver des **méthodes nouvelles ou sous-exploitées** pour passer **TOP1** sur T1 et T3.  
**Hors scope :** T2 (Model Equality). Ne pas y chercher.

Date du brief : 21 nov. 2026 ~22:10 Paris. Leaderboard public = **30 %** des données ; final = **70 %** cachés. Ne pas sur-optimiser le public.

---

## Snapshot leaderboard (public, ~22:00)

| Track | Nous (MasterChickenHack) | Tête connue | Écart à combler |
|---|---|---|---|
| **T1** watermark localization | **~0.74** mIoU | ~0.74–0.75 (ex. Eugenia / WhatThePhoques) | **+0.01 à +0.03** |
| **T3** unlearning | **~0.22** (forget×retain) | **~0.31–0.32** (ex. Fable 4.2) | **+0.10** |

T3 est le gros retard. T1 est déjà au coude-à-coude.

---

## Ce que tu dois livrer

Un rapport court (≤ 80 lignes) + liste actionnable :

1. **5–10 méthodes / repos / threads** réellement utiles (pas de filler).
2. Pour chaque item : lien, 2 lignes « pourquoi ça match notre contrainte », effort estimé (h GPU), risque.
3. **Priorité TOP3** pour T1 et TOP3 pour T3 (ce qu’on code ce soir / demain matin).
4. Requêtes X/GitHub qui ont marché (pour qu’on re-run).
5. Ce qui est **déjà saturé chez nous** (ne pas re-proposer).

Format préféré : markdown, titres `## T1` / `## T3`, bullets denses.

---

## T1 — Localisation de filigrane

### Problème (officiel)

- Images test **512×512**, **1020** images. Rendre masque binaire **128×128** (1 = zone filigranée).
- 3 méthodes post-hoc : **PixelSeal**, **TrustMark**, **StegaStamp**. Message + méthode **connus** au test.
- Filigrane posé dans un **rectangle** (aspect max 16:1 PixelSeal/StegaStamp, **2:1 TrustMark**), puis éditions : outpainting, retrait, ajout, greffe d’objet.
- Score : mIoU moyen (IoU clean + IoU watermark) / 2, puis moyenne sur images.
- Soumission : `.pt` < 20 Mo. Décodeurs fournis dans le kit.

Consigne : `reports/sources/Watermark_Localization.txt`  
Kit idées : WAM n’est **pas** le watermark des images ; c’est seulement une inspiration « carte de localisation ».

### Ce qui marche déjà chez nous (ne pas rediscourir)

- Pipeline Adem : **cartes de décodage / fenêtre-canevas + vote + assemblage** → public **~0.74**.
- Boîtes TrustMark calées (`boxr3`) : gros gain local sur rectangles.
- StegaStamp déjà bon sur « pleines à trou » (IoU propre ~0.73).
- Decode-only Yassir ~**0.46** : plafond. Seul, insuffisant pour TOP1.

### Ce qui ne marche pas / saturé

- Seuils adaptatifs par image / quantile / morphologie / rectangle englobant naïf.
- U-Net sur image seule ou image+carte de decode (510 labels trop peu) → sous le decode.
- Forcer **tout-1** sur les images « pleines » : **catastrophe** (−0.13) car toutes ont un **trou propre** (médiane ~6 % de l’image).
- Affiner encore le seuil global : oracle parfait ≈ **+0.02–0.03** seulement.

### Le vrai levier restant (oracle Adem)

Gain max ≈ **+0.10** si on fixe :

1. **Trous propres** dans les pleines **TrustMark** et **PixelSeal** (IoU propre 0.32 / 0.37 vs StegaStamp 0.73).
2. Rectangles **PixelSeal** (IoU watermark 0.24).

Cherche donc surtout : **détection de petite région restaurée / inpaintée / non-watermarked hole** dans une image presque entièrement watermarkée, et **box localization** fine pour PixelSeal.

### Requêtes de recherche (T1)

**GitHub / code**

```
PixelSeal localization OR sliding window watermark
TrustMark watermark localization github
StegaStamp decode window OR patch
"watermark localization" mIoU OR mask
Watermark Anything WAM localization
EditGuard OmniGuard TAG-WM localization
image manipulation localization inpainting detection
"restored region" OR "inpainted region" detection watermark
bit accuracy heatmap watermark
```

**arXiv / papers**

```
arXiv: Watermark Anything (2411.07231)
arXiv: PixelSeal / videoseal
arXiv: TrustMark (2311.18297)
arXiv: EditGuard, OmniGuard, TAG-WM
manipulation localization deepfake inpainting mask
```

**X / Twitter**

```
("watermark localization" OR "localize watermark" OR PixelSeal OR TrustMark OR StegaStamp OR "Watermark Anything") (mIoU OR mask OR github)
("inpainting detection" OR "manipulation localization") (github OR code)
from: reputable ML accounts + EuroAI / hackathon watermark
```

### Pistes à évaluer (pas encore prouvées chez nous)

- Residual / re-embed correlation at **pixel** scale for small holes (PixelSeal/TrustMark).
- SAM / object masks as prior only where edit = object graft (déjà partiellement exploré).
- Ensemble of **decode map + residual + box** with learned fusion **sans** gros U-Net (GBM / petit MLP sur features locales).
- Multi-scale **canvas** (pas crop) avec calib par méthode déjà faite ; nouvelles échelles seulement si RAM StegaStamp OK (≤36 fenêtres).
- Papers 2025–2026 sur **post-hoc watermark localization** (très peu ; signaler s’il y en a de frais).

### Contraintes engineering

- 6× H200 dispo (équipe), mais jobs déjà souvent saturés.
- Ne pas recommander des méthodes qui exigent un watermark **entraîné pour localiser** (WAM) **appliqué aux images** : on ne peut pas re-watermarker le test.
- Sortie stricte : tenseur bool/masks 128×128, downsampling officiel du kit.

---

## T3 — Unlearning ViT-B/16

### Problème (officiel)

- Checkpoint `model.pt` (ViT-B/16, 100 classes) + **3500 forget images sans labels**.
- Aucune autre donnée d’entraînement fournie. Objectif : imiter un modèle **réentraîné sans** ces 3500.
- Submit : `state_dict` float32, same keys (`strict=True`), < 450 Mo.
- Score : **forget × retain**. Forget = (loss-distribution score) × (prediction score), sur exactes **et** copies altérées cachées. Gold = moyenne de plusieurs retrains.
- **Utility gate : accuracy < 70 % → score 0.**
- Softmax scorée à **température 2**.

Consigne : `reports/sources/Unlearning.txt`.

### Ce qui marche déjà chez nous

- Recette Adem type **`PG_T85_f*_l814c`** : basculer une **fraction** d’images forget vers une cible structurée (loss haute contrôlée), ancre retain.
- Progression publique récente : ~0.15 → **0.2179** (`f35_l814c`). Tête ~0.32.
- Calibration gold estimée (Adem) : mean loss ≈ **2.9–3.0**, std ≈ **4.7–4.9** ; ~28–35 % d’images « basculées », loss cible ~8–14.
- Axe **hauteur de loss** largement exploité / saturé. Levier restant = **prediction score × retain (~0.2)**.

### Ce qui ne marche pas / saturé

- Distillation « soft teacher on cropped/blurred views » sur tout le net ou 2 derniers blocs → **score 0** (utility ou mauvaise forme de cible).
- Uniform soft / random labels / naïf gradient ascent (le brief officiel le dit).
- Monter seulement mean/std CE sans coller aux **mauvaises classes du gold**.
- Mettre toute la masse sur le **2e choix d’orig seul** → prediction score ~0 dans le simulateur Adem.
- Softening trop fort des non-basculées (÷2) → collapse prediction.

### Insight simulateur Adem (à chercher des équivalents publiés)

Forme de cible qui fait passer oubli 0 → **0.19** en sim (sans oracle) :

- Sur images basculées : **abaisser la vraie classe** sous le max des autres, **garder l’ordre relatif des 99 autres logits d’orig**.
- Sur les autres forget : `orig` légèrement adouci (÷ **1.5**, pas 2).
- Sélection : ~30 % des images à **plus haute entropie** d’orig (T=2).

Donc cherche des méthodes de **targeted unlearning / preference optimization / scrubbing** qui :

1. Matchent **distribution de loss** (mean+std) du gold.
2. Matchent **quelle wrong class**, pas seulement « moins confiant ».
3. Gardent retain (KL à orig bas) + gate 70 %.

### Requêtes de recherche (T3)

**GitHub / code**

```
machine unlearning ViT OR "vision transformer" forget
SCRUB unlearning github
NPO Negative Preference Optimization unlearning
SalUn unlearning
SCAR unlearning proxy
"exact unlearning" OR "approximate unlearning" image classification
NeurIPS 2023 unlearning challenge winning
ReGUn unlearning
Amnesiac unlearning reset head
"forget set" KL distillation retain
```

**arXiv**

```
2302.09880 SCRUB
2404.05868 NPO
2310.12508 SalUn
2404.12922 SCAR
2406.09073 NeurIPS 2023 Machine Unlearning Challenge
ReGUn / zero-shot unlearning / gated knowledge transfer
```

**X / Twitter**

```
("machine unlearning" OR SCRUB OR SalUn OR NPO OR "forget set") (ViT OR CIFAR OR ImageNet OR github)
("unlearning challenge" OR "NeurIPS unlearning") (code OR method)
("match gold" OR "retrain without" OR "membership inference") unlearning
```

### Pistes à évaluer (haut potentiel pour +0.10)

- **Pseudo-gold ensemble** : reset/noise last blocks + distill on proxy never-seen images ; average softmax as `p_gold` (NeurIPS’23 winners style).
- **Conditional KL / class-structure preserving** forget targets (keep ranking of non-true classes).
- Better **proxy retain** (OOD natural images, not Gaussian noise — SCAR warning).
- Weight interpolation `orig + α(unlearn−orig)` after strong forget.
- Methods that explicitly optimize **loss mean+std** + **per-sample wrong-class** vs a reference.
- Membership-inference–aware unlearning (make forget look like non-members under MIA).

### Contraintes engineering

- No extra **labeled train** data from organizers. Public datasets as **proxy** : OK if used carefully (team already uses COCO / SDXL gens / IP-Adapter).
- Eval server uses **fp16 autocast** ; submit float32 state_dict.
- Team submit cooldown ~10 min ; public is noisy. Prefer robust recipes.
- Utility gate 70 % : never propose recipes that trash global accuracy.

---

## Ce que l’équipe a déjà lu (évite les doublons)

- `reports/t1-watermark-etat-de-lart.md`
- `reports/t3-unlearning-etat-de-lart.md`
- Issues GitHub `adembnn/EAC` #3 (T1) et #5 (T3) — lire les derniers commentaires Adem.
- Guides H200 : `docs/acces-h200-nebius*.md` (6 nœuds).

Repos : `adembnn/EAC` branche `yassir/baselines`.

---

## Critères de sélection (agent)

Garde une méthode seulement si **oui** à ≥ 3 :

1. Compatible contraintes (pas de labels forget T3 ; pas de re-watermark T1).
2. Code public réutilisable ou pseudo-code clair ≤ 1 page.
3. Adresse le levier restant (T1 : trous / boxes PixelSeal ; T3 : prediction×retain).
4. Faisable en **< 6 h** wall-clock sur 1×H200 avant deadline.
5. Pas déjà saturé dans la section « ne marche pas ».

---

## Message collable pour l’agent online

```text
You are a research agent. Goal: find NEW methods (X/Twitter + GitHub + arXiv) so team MasterChickenHack finishes TOP1 on EuroAI Paris 2026 tracks T1 and T3 before 2026-11-22 12:00 Paris.

Read this brief fully:
https://github.com/adembnn/EAC/blob/yassir/baselines/docs/brief-recherche-top1-t1-t3.md

Current public scores: T1 ~0.74 (need +0.01..0.03), T3 ~0.22 (need +0.10 vs leaders ~0.32).
Ignore T2.

Search X + GitHub + arXiv with the queries in the brief. Prefer 2024–2026 code.
Do NOT suggest: naive gradient ascent, random labels, uniform soften-only, WAM re-watermarking test images, or more T1 global threshold tuning.

Deliver: markdown with ## T1 and ## T3, each with Top-3 actionable methods (link, why it fits, hours, risk), plus saturated ideas to skip.
```
