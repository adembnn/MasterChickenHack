# Recherche ciblée TOP1 T1/T3 — 21 novembre 2026, nuit (MasterChickenHack)

> **Hors T2.** Méthodes externes = hypothèses, pas scores. Deadline : 22/11 12:00 Paris.
> Sources d'équipe : [T1 #3](../issues/3), [T3 #5](../issues/5), [brief](../blob/yassir/baselines/docs/brief-recherche-top1-t1-t3.md).
> Référence récente : T1 dernier public **0.7384**, T3 public confirmé **0.3159** (soumission 1895), leaders T3 env. 0.35–0.40. Le public est 30 %, ne pas optimiser dessus seul.

## T1

### Top 3, action avant demain midi

1. **Décision immédiate : candidat 22, après relecture indépendante**. [Issue #3](https://github.com/adembnn/EAC/issues/3) : réserve **0.7514** vs candidat 5 **0.7230** (+0.0284). Candidat 20 a déjà le feu vert du vérificateur, +0.0263 (IC95 [+0.010, +0.045]), mais le 22 ne gagne que +0.0021 sur le 20. Si le 22 est validé, **soumettre 22 plutôt que 20** ; sinon 20 déjà vérifié. **Effort GPU 0 h ; risque faible-moyen**, sous condition validation.
2. **Contraste contre-factuel *même image, même message*** : pour chaque image synthétique watermarkée, faire une version intacte et une version dont une petite zone est remplacée par le pixel original propre. À BCE/Dice existants, ajouter une perte de paire : la probabilité watermark **doit baisser seulement dans le trou**, toutes autres conditions fixées. Échantillonner trous <2 %, 2–8 %, 8–20 % d'aire et vraie toile 512+s asymétrique. Source de l'idée segmentation localisée : [WAM, arXiv 2411.07231](https://arxiv.org/abs/2411.07231), [code](https://github.com/facebookresearch/watermark-anything). À tester sur **PixelSeal pleines** (le réseau actuel y manque les petits trous), puis TrustMark. **GPU 1–3 h ; risque moyen-haut** (déplacement de domaine synthétique → vrai). *Pas la même chose que le BCE sur masques fabriqués déjà testé*.
3. **Rendu de bord conditionné par preuve watermark, pas par RGB seul** : branche PointRend utilisant **les features spatiales du segmentateur entraîné**, une carte de logit + features fines et points dans les trous <8 % (ne pas réutiliser le petit MLP RGB/logits 5×5 déjà négatif). Source [PointRend](https://arxiv.org/abs/1912.08193), [code](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend). Évaluer *trous déjà touchés vs entièrement manqués* ; si les trous sont manqués, préférer #2. **GPU 1–2 h ; risque haut** (gain faible au bord seul).

### Repos/méthodes secondaires (uniquement ablations rapides)

- [TruFor RGB + Noiseprint++](https://github.com/grip-unina/TruFor), [paper](https://arxiv.org/abs/2212.10957) : extraction de carte d'anomalie pour compléter TrustMark. **GPU 0.5–1 h ; risque très haut** car un « trou propre » peut être l'original restauré sans anomalie de splice. Vérifier AUC sur pleines *avant* entraînement.
- [2026 local watermark robustness benchmark](https://arxiv.org/abs/2609.16832) : géométrie et inpainting détruisent parfois la récupérabilité ; **lecture 0 h GPU ; risque nul**. Sert à empêcher les essais aveugles de décodage sur rectangles PixelSeal <32 px.
- [PixelSeal officiel](https://github.com/facebookresearch/videoseal), [TrustMark officiel](https://github.com/adobe/trustmark) : inspiration pour architecture/features, **0 h lecture ; faible risque**, mais leurs détecteurs globaux ne localisent pas les trous sans réapprentissage.

**Déjà saturé / ne pas refaire** : seuil adaptatif, U-Net image seule, force tout-1, CAM natif PixelSeal, SAM sur pleines PixelSeal, filtre adapté global, re-tamponnage seul, petits rectangles PixelSeal illisibles, stride-2, seuils de boîtes naïfs. Source : dernières expériences [#3](https://github.com/adembnn/EAC/issues/3).

## T3

### Top 3, après la file d'envois actuelle

1. **Garder le plan d'Adem en priorité** : le meilleur prouvé est `PG_T85_f70_l814c` à **0.3159** (1895). `G10_f80` était annoncé pour tester la marge avant `f80` ; **ne pas créer une seconde file d'envois**. Les ablations `PGw15/w2`, `PGs70m/s48m`, `PGv2/v3` et `PG_RDt24` sont déjà en cours ; attendre leurs mesures. **GPU 0 h pour notre agent ; risque bas**.
2. **Nouvelle hypothèse à faible coût : prototype des *wrong classes***. Construire, sur les **3500 forget images non étiquetées**, des prototypes de classes dans l'espace des features ViT gelées (labels = argmax original). Pour chaque image, exclure sa classe, calculer softmax cosine sur 5 classes voisines ; mélanger 35 %/70 % de cette distribution avec le softmax des **99 autres** logits originaux. Code : [`t3_prototype_wrongclass.py`](../blob/research/top1-t1-t3-nov21/tracks/t3-unlearning/experiments/t3_prototype_wrongclass.py). Remplacer UNIQUEMENT la distribution conditionnelle hors classe des basculées dans le pipeline PG actuel, **fixer mêmes exemples, m/s exactes et ancre**, comparer `beta=0, .35, .70` dans le simulateur, copies et retenue. **GPU 0.5–2 h ; risque haut** (pseudo-labels + proximité feature ≠ gold). Inspiré par les prototypes de feature [SCAR](https://arxiv.org/abs/2404.12922) mais ne nécessite pas de retain labels.
3. **MU-Mis comme faible régularisation, *pas* en remplacement de PG** : l'ICLR 2026 minimise la sensibilité différentielle des logits sur les images forget sans données de retain. Tester uniquement aux points déjà basculés, gradient limité à la tête / 2 blocs finaux, garder l'ancre proxy, et recalibrer m/s. [Paper](https://arxiv.org/abs/2402.15109), [code](https://github.com/poppopbean0903/MU-Mis). **GPU 1–3 h ; risque élevé** (deuxième dérivée, peut réduire prediction score).
 
### Autres sources utiles

- [DKD non-target KL](https://arxiv.org/abs/2203.08679), [code](https://github.com/Myleswork/DKD) : contrôler explicitement la KL de la distribution hors classe ; **GPU 0.5–1 h ; risque moyen**. Déjà proposé chez Yassir (`t3_conditional_kl.py`), **ne pas doubler** sauf validation sur PG actuel.
- [LoTUS (CVPR 2025)](https://arxiv.org/abs/2503.18314), [code](https://github.com/cspartalis/LoTUS) : incertitude ciblée vs globale ; **GPU 1–2 h ; risque élevé** parce qu'une distribution uniforme casse le score KL gold.
- [SCAR (ECCV 2024)](https://arxiv.org/abs/2404.12922), [code](https://github.com/jbonato1/SCAR) : ancrage sur images naturelles OOD avec distillation `orig` ; **GPU 1–2 h ; risque moyen**, mesurer sur proxy non-utilisé par l'entraînement.
- [Unmerge (sept. 2026)](https://arxiv.org/abs/2609.38895) : task arithmetic ciblée, **GPU estimé 3–5 h ; risque très élevé** car évalué surtout en class-level et pas notre score de mauvais logits. Pas prioritaire avant dimanche midi.

**Déjà saturé / ne pas refaire** : distillation de vues originale seule, random/uniform labels, gradient ascent brut, température globale, augmentation seule de mean/std de CE, seconde classe d'origine seule, `PGho` à 60 %, hypothèse « 9 classes entièrement oubliées ». Sources : [#5](https://github.com/adembnn/EAC/issues/5), brief.

## Garde-fous et exploitation

- T1 : **réserve de 102 images**, comparer mêmes identifiants, réutiliser les logits hors-pli, IC95 apparié doit exclure 0, **gain ≥ +0.02 contre dernier fichier envoyé** avant une soumission selon règle d'Adem.
- T3 : **`state_dict` float32 strict**, évaluer sous fp16 autocast, **accuracy ≥ 70 %** obligatoire, CE moments sur exactes + copies, KL à **T=2**, score de contrôle caché. Aucun score simulator/proxy ne prouve un gain officiel.
- **H200** : la documentation donne nœud 6 `204.12.188.1` (VPC séparé), direct SSH Yassir. **Charge GPU réelle non vérifiable ici** : le dashboard ne répond pas à nos outils réseau ; dernier message [#5](https://github.com/adembnn/EAC/issues/5) annonce n6 saturé, n5 occupé. Vérifier la UI et `nvidia-smi` sur place, annoncer sur #2 avant de lancer.
- Requêtes réutilisables : `"remaining-data-free" MU-Mis ICLR 2026 github`; `"feature prototype" unlearning SCAR wrong class`; `site:github.com TruFor Noiseprint++ segmentation`; `site:arxiv.org watermark localization PixelSeal inpainting 2026`.
- **Aucun score GPU ni soumission EAC obtenu par ce rapport/script.** Le script est testé uniquement sur tenseurs synthétiques pour forme/normalisation ; branche de recherche, **ne pas merger** sans mesure réelle et maker-checker.
