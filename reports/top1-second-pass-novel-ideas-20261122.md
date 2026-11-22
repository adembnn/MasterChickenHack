# TOP1 — deuxième passe indépendante : RESF/B3IT, MU-Mis, MPC (22/11/2026, ~05 h Paris)

**Décision : T2 en priorité absolue, ensuite T3 diagnostic ciblé, T1 seulement si carte PixelSeal libre.** Résultats ci-dessous lus dans le dépôt privé et les issues, NON dans un leaderboard live. Toute méthode citée est une **hypothèse de transfert** tant qu'une mesure EAC hors-pli et un checker ne l'ont pas validée.

## État réel et règles du dépôt
- **T1 public 0.8230**, candidat 38 / soumission 2910, CV 0.8347 [#3](https://github.com/adembnn/EAC/issues/3) : meilleur équipe à 04:32, gain +0.0099. Désormais le checker exige IC95 apparié >0, cartes OOF vérifiées et feu vert écrit, **plus de seuil fixe +.02** (décision Adem à 04:17).
- **T2 public 0.6000** [#4](https://github.com/adembnn/EAC/issues/4) : H=0.30, ep01 isolé=0.60. D'après le bus, le plus gros manque du classement général (~+0.14 si passage à 1) est T2. **Titouan a dit NON à la dépense des 50 dernières réponses ep pour l'instant** (commentaire 7073768011). Ne pas lancer nos nouvelles sondes sur ep. **Envoi ep26 4:59 géré par relais d'Adem** (commentaire 7073787173); ne pas proposer d'envoi concurrent.
- **T3 public 0.4107**, `PG_RDq_70_l915c_s1` / soumission 2827 [#5](https://github.com/adembnn/EAC/issues/5) : moyenne de softmax(logits/2) de 8 partitions hors pli, cible oubli m/s=5.09/4.62. `..._l1016c` = 0.4028 => augmenter la loss est défavorable. Les 16/24 partitions, graines et ancres inversées sont déjà en cours chez Adem; ne pas refaire.
- **Seule session relais d'Adem soumet** T1/T2/T3. Demander allocation de GPU sur bus [#2](https://github.com/adembnn/EAC/issues/2); n8/n9/n10 ont déjà été déclarés occupés. Aucune IP/VRAM « libre » n'est certifiée ici.

## 1 — T2, vraie nouveauté relative au minage existant : *near-tie top-token selection* sans score public

**[B3IT — ICML 2026](https://proceedings.mlr.press/v306/chauvin26a.html)**, [code](https://github.com/timothee-chauvin/token-efficient-change-detection-llm-apis), sélectionne des *border inputs* où les deux premiers logits sont proches; sous T≈0, de légères modifications de poids changent facilement le token dominant. [RESF — EMNLP 2025](https://aclanthology.org/2025.emnlp-main.247/) choisit des prompts sensibles via un proxy d'entropie-gradients et contrôle les fausses alertes de température. **NE PAS importer leurs scores** (B3IT annonce ×30 en son contexte, RESF jusqu'à 98.8 % sur ses modèles) : ces études auditent un endpoint par rapport à un modèle connu, pas notre classement de 33 endpoints.

**Le blocage exact EAC** : température et seed envoyées à l'API sont ignorées, seulement 100 réponses par endpoint, système parfois déguisé. B3IT original demande T=0 et beaucoup de répétitions d'un même input ; **il n'est donc PAS directement exécutable** sur EAC.

**Adaptation testable sur H200, 0 requête ep** :
1. Sur les 3 jumeaux LOCAUX de Titouan/Adem, échantillonner une banque d'environ 1 000 *contextes de continuation communs*, issus de prompts déjà minés. Faire un passage logits de la base et de variantes simulées *faibles* (LoRA ×0.3, fp8, et graines indépendantes) sur EXACTEMENT ces contextes. Sauvegarder seulement `base_top2[P,2]` et `variants_top2[P,K,2]` pour les **deux tokens favoris de la base**.
2. **Nouveau code** [`t2_border_probe_ranker.py`](../tracks/t2-model-equality/experiments/t2_border_probe_ranker.py) : sélection par quartile conservateur du changement de probabilité des deux tokens, **après min sur un INTERVALLE CONTINU de températures** `T∈[.6,1.0]` pour base et variante séparément, pondéré par proximité de 50/50 à T=.7. Pure température (logits multipliés) doit rendre score≈0, inversion de rang sous plusieurs variantes doit rendre score>0. La version discrète initiale donnait un **faux positif de température**; corrigée par commit `5ef85aa`. Le principe numérique a été contrôlé sur tableaux synthétiques (meilleur = véritable inversion; température seule = 0), **aucune métrique EAC réelle**.
3. Comparer TOP25 vs les **25 sondes minées LoRA/fp8 actuelles** sous les mêmes 50 réponses sur les 9 ref, y compris leurs 4 copies déguisées, et sur jumeaux avec **graines LoRA et templates non vus**. Mesurer MCC pairwise, recall LoRA0.3 et faux splits; aucun tuning sur ep ou le leaderboard.
4. GATE : le top25 doit améliorer MCC hors-split / recall de LoRA faible **sans séparer ref-01/03/05/06/08** et sans effet de température (0.7↔1.0). Sinon **abandon**. Ensuite Titouan seul décide si relancer du budget ep.

**Réserve** : la mesure sur deux tokens seulement ignore le reste du vocabulaire, les prompts système et le top-p. C'est un filtre de *candidats de sondes*, pas un nouveau lecteur garantissant l'égalité. Coût H200 <~30min pour banque de logits si jumeaux prêts; classement CPU.

**Ep17↔ep33** : t2-critique a trouvé z=+5.8 après 42 lectures, mais recalibrage demandé : à #4 commentaire 7073696998, Adem a **déjà lancé** validation A/B sur 100 faux endpoints avec 17↔33↔18; ne surtout pas doubler. Le lien n'est pas retenu sans signal sur les deux moitiés + max-statistic. Le fichier CSV doit être une partition cohérente.

## 2 — T3, piste différente de notre soup et de la classe redistillée : *input sensitivity* / MU-Mis

**[MU-Mis (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2e520272aa448c0b8662d13e878dd717-Abstract-Conference.html), [code public](https://github.com/poppopbean0903/MU-Mis/blob/main/unlearn_alg/Alg_Mumis.py).** La méthode, pensée pour le *retain-free unlearning*, minimise (en simplifiant) `||∇_x z_c||² - ||∇_x z_c'||²` sur données à oublier. Le code réel tire une autre classe `c'` et utilise les étiquettes **vraies** lors de l'entraînement. **EAC ne fournit pas ces étiquettes** : reprendre la loss telle quelle sur `orig.argmax` serait une faute de protocole et pourrait dégrader les vrais scores.

**Nouveau diagnostic zéro training**, [`t3_input_sensitivity_probe.py`](../tracks/t3-unlearning/experiments/t3_input_sensitivity_probe.py) :
```bash
python t3_input_sensitivity_probe.py --smoke
python t3_input_sensitivity_probe.py \
  --orig /chemin/model.pt \
  --candidate /chemin/PG_RDq_70_l915c_s1.pt \
  --forget /chemin/forget_images.npz \
  --num 96 --batch 4 --out sensitivity_rdq.json
```
- Charge EXACT ViT timm `vit_base_patch16_224`, LayerScale=1e-4, preprocess serveur `(pixels/255−.5)/.5`, strict weights.
- Sur 96 oublis, classe proxy fixe `argmax(orig)` et deuxième classe : calcule les gradients de logit de chacune par rapport à l'entrée, sur **image exacte et translation 1 px**. Compare `orig` à la base gagnante `RDq`; publie gap d'énergie des gradients et rebound de probabilité proxy à T=2. **Mesures de diagnostic**, pas vraie CE puisque labels absents.
- Si la sensibilité du gagnant NE prédit PAS l'instabilité sur copies (corrélation faible/non stable entre 3 sous-échantillons), **STOP**; ne pas consacrer de GPU à une loss à dérivées secondes. Sinon discuter avec Adem de 1 très courte ablation d'une faible pénalité de sensibilité *en plus* des cibles RDq, avec vérification de forget m/s, copie et gate 70 %; jamais appliquer MU-Mis brut.
- Coût 2 modèles × 96 images avec gradients, typiquement <~15 min sur une H200 **si GPU réellement disponible**. Smoke torch CPU de la primitive passé; **test réel ViT non effectué depuis cet environnement**.

**Pourquoi potentiellement utile** : la note EAC comprend pour moitié des copies perturbées invisibles ; modifier uniquement le « mauvais classe » ne garantit pas une stabilité locale du signal. MU-Mis traite une autre propriété du modèle que les loss/ancres de PG. Mais aucune preuve de gain EAC à ce stade.

## 3 — T1, **MPC/LPC** : contraste de *features locales*, pas la perte contrefactuelle déjà négative

[**MPC** — TIFS 2025](https://github.com/multimediaFor/MPC) a du code vérifiable : phase 1 représentation contrastive multi-vues, phase 2 tête de localisation gelée. [**LPCLNet** — Information Sciences 2026](https://www.sciencedirect.com/science/article/pii/S0020025526001362) compare des patches **intérieur / fond / contour**, avec InfoNCE à trois groupes et fusion multiscale. Leurs scores CASIA/NIST ne concernent PAS le watermark du challenge.

**Test réduit (~30 min–1 h GPU sous réserve de caches)** : geler `PixelSeal f_x2`, extraire 128×128 embeddings des 5 folds OOF; apprendre une tête **contrastive triplet (propre / marqué / contour)** à faible capacité sur plis d'entraînement, sans enseigner les pixels des folds tenus à l'écart. Comparer contre *tête ordinaire même paramètres/steps* ; décider **uniquement** sur mIoU PixelSeal CV 170 images (candidat38 = **0.7448**), contrôle réserve et rectangles <64 px, puis somme des trois filigranes contre c38. **GATE** : CV +0.010 PixelSeal, réserve même signe, IC95 globale candidat38 exclut 0; checker indépendant. Stop sinon.
**Ne pas faire un full training MPC** (domaine forgery différent, coût haut). Ne pas refaire les échecs mesurés chez Adem : trous contre-factuels λ=.1 (négatif à 6000 pas), suréchantillonnage petits rectangles, simple boundary BCE, post-processing SAM.

## Éléments importants refusés
- B3IT avec T=0 sur EAC : API ignore T; toute économie annoncée non transférable.
- RESF « 98.8 % » : leur problème de détection de tampering avec vérité et accès local n'est pas la partition MCC EAC.
- MU-Mis brut sur pseudo labels : vrai label absent; risque élevé de rétention et de perte de référence.
- MPC complet from scratch : T1 déjà 0.823, gain marginal incertain; privilégier T2.
- Nouveaux essais d'ensembles RDq 16/24 ou de sous-échantillonnage consensus : Adem y travaille déjà.

## Sécurité / exécution / responsabilité
- **Tests CPU** : script T2 score sur matrice synthétique avec faux positif température corrigé; script T3 primitive `torch.autograd.grad` sur petit modèle; pas de tests sur données réelles, aucun score public nouveau.
- **Aucune connexion GPU, aucun appel ep, aucun envoi au concours depuis cette session**. SSH port 22 indisponible. Ce rapport et les scripts restent sur branche PR #34, pour exécution par les agents ayant H200 réellement accessibles.
- **Aucun secret** dans ce rapport ou les scripts. Les clés SSH fournies précédemment ne doivent jamais être commitées.
