# T3 Unlearning : état de l'art, 2e passe (lecture des articles, pas des résumés)

Date : 21 nov. 2026, 17 h. Décision débloquée : quelle méthode implémenter. Aucune soumission, aucun appel payant.
Lu en détail : ReGUn (texte complet), Local Teacher Distillation (extraits longs + annexes), TREW, SCAR, MU-Mis, benchmark ViT 2602.20114, rapport NeurIPS 2023, 5 solutions Kaggle (1re, 3e, 5e, 9e, 10e).
Non relu en entier : SCRUB, SalUn, NPO (mémoire), HAMU, WIN-U, DISCERN (résumé seulement).
Limites : Semantic Scholar et OpenAlex ont répondu « 429 » (trop de requêtes) ; un seul graphe de citations obtenu (rapport NeurIPS 2023 : 18 citations depuis mars 2026, aucune ne propose une méthode pour notre métrique). Aucun papier SprintML/CISPA ne traite cette métrique (leurs travaux récents : LLM, contrastif). Le jeu SprintML/unlearning n'a pas de fiche (« No dataset card yet »), donc l'origine des 100 classes reste inconnue.

## 1. Ce qu'on nous demande (5 lignes)
1. Un ViT-B/16 (100 classes) a vu 3500 images ; on rend un `state_dict` float32 qui se comporte comme si elles n'avaient jamais été vues. Pas de labels, pas d'autres images.
2. Le « gold » = moyenne de plusieurs modèles réentraînés sans ces images. Sur ces images il agit comme sur des images de test : parfois faux, avec une mauvaise classe voisine.
3. Forget = score de loss (moyenne ET écart-type de la loss, relatif à orig) x score de prédiction (KL par image à T=2 vers la moyenne du gold), sur les images exactes et sur des copies modifiées cachées.
4. Retain = (accuracy / accuracy orig) x 1/(1 + KL(orig||sub)) sur un jeu de contrôle caché. Porte : moins de 70 % sur la validation = 0.
5. Score final = forget x retain. Public = 30 % des données, classement final = 70 %.

## 2. Ce que la métrique impose (déductions, chiffres illustratifs)
- **Le score de loss est le goulot, le score de prédiction monte vite.** Prédiction = 1 - KL(gold||sub)/KL(gold||orig). Orig est très sûr de lui là où le gold hésite : le dénominateur est grand, donc lisser les sorties rapproche déjà beaucoup. Loss = 1 - d/d_orig avec d = |écart de moyenne| + |écart d'écart-type| ; orig a une loss presque nulle et presque constante, donc d_orig vaut environ (moyenne gold + écart-type gold).
- **Une cible tempérée seule plafonne le score de loss.** Si toutes les images deviennent « un peu incertaines », la loss est quasi constante : moyenne bonne, écart-type presque nul. Exemple (hypothèse : gold moyenne 0,7, écart-type 1,4) : d_orig ≈ 2,0, d_sub ≈ 1,3, score ≈ 0,35. Pour gagner l'écart-type il faut une **queue** : environ 10 à 25 % des images où le modèle se trompe franchement (loss 2 à 6), et les autres confiantes. Et pour le score de prédiction, il faut que ces images soient **les bonnes** (celles que le gold rate).
- **KL(gold||sub) est une KL « avant »** : elle punit fort un sub qui met une probabilité minuscule là où le gold en met. Être un peu trop flou coûte moins cher qu'être trop sûr.
- **Deux températures** : la loss est à T=1, la KL à T=2. Le gold du score de prédiction est une moyenne de modèles (plus lisse qu'un seul modèle) ; celui du score de loss est une moyenne de statistiques de modèles individuels. Un seul vecteur de logits doit satisfaire les deux : on entraîne les deux termes (section 3).
- **Copies modifiées** : orig les a moins mémorisées, donc d_orig y est plus petit, donc chaque écart y pèse plus lourd en relatif. Le gold se trompe un peu plus dessus que sur les images exactes.
- **Retain** : pour de petits décalages de logits, la KL varie comme 1/T². À T=2 elle vaut environ le quart de la KL à T=1 : un KL(orig||sub) de 0,05 coûte 5 % de retain. Le vrai risque est l'accuracy sur le contrôle (si ce jeu contient des images d'entraînement très sûres, baisser la confiance des images à oublier peut « fuir » vers elles : « localized collateral forgetting », Dolgova & Stich, NeurIPS 2026 : l'écart au gold croît avec la proximité du forget set).

## 3. Tableau de l'état de l'art (pertinent pour nous)
Légende « retain » : la méthode exige-t-elle des données retain/held-out dans le domaine ? Lecture : E = entier, X = extraits, R = résumé.
| Papier | Date | Lieu | Idée | Retain ? | Code | Lu |
|---|---|---|---|---|---|---|
| ReGUn, Reference-Guided Unlearning 2603.11210 | v2 29 sept. 2026 | arXiv (ETH) | Distiller l'oubli vers la **moyenne par lot** des sorties d'orig sur un held-out labellisé de mêmes classes ; + CE retain | oui | non trouvé | E |
| Local Teacher Distillation 2605.31317 | 29 mai 2026 | NeurIPS 2026 (poster) | Petit professeur entraîné sur les **voisins retain** du forget set ; ses top-3 probabilités = cibles douces | oui (voisins) | non trouvé | X |
| TREW 2506.20893 | v5 juil. 2026 | ICML 2026 | Cible = sortie d'orig sans la classe oubliée, inclinée vers les classes proches (classe, pas image) | oui (CE retain) | github.com/CrowdDynamicsLab/ICML_TREW | X |
| MU-Mis 2402.15109 | v4 mars 2026 | ICLR 2026 | Retain-free : réduit la sensibilité d'entrée de la classe cible ; ViT : Adam, batch 32 | **non** | non trouvé | X |
| SCAR 2404.12922 | 2024 | à vérifier | Retain-free : distillation JS vers orig sur 10 000 images OOD (ImageNet ou COCO) ; ViT-B/16 testé ; suppression de classes | **non** (OOD) | github.com/jbonato1/scar | X |
| Benchmark ViT 2602.20114 | v2 2026 | arXiv | ViT/Swin/DINOv2 : sur ViT-B/16 (ImageNet val, 3000 à oublier) Fine-tune 0,843, NegGrad+ 0,842, SalUn 0,773 | oui | RUM : github.com/kairanzhao/RUM | X |
| RUM (NeurIPS 2024) | 2024 | NeurIPS | Trier le forget set par mémorisation (faible/moyenne/forte), appliquer la méthode par groupe | oui | idem | R |
| Rapport concours NeurIPS 2023 2406.09073 + Kaggle | 2024 | arXiv | Les meilleurs = « effacer puis réparer » ; scores bruts 0,07 à 0,10 | oui (9e : forget-free) | github.com/google-deepmind/unlearning_evaluation (scoreur) | X |
| SalUn / SCRUB (ICLR 2024 / NeurIPS 2023) | 2023-24 | — | Masque de saillance + labels aléatoires / alternance maximiser-minimiser | oui | Unlearn-Saliency ; meghdadk/SCRUB | mémoire |
| HAMU 2606.02119 | 1 juin 2026 | arXiv | Pondération selon la difficulté, multi-objectifs | oui (a priori) | github.com/aoi3142/HAMU | R |
| WIN-U 2604.13438 ; DISCERN 2611.07892 | avr. ; 5 nov. 2026 | COLM (soumis) ; arXiv | Retain-free par Newton ; par embeddings synthétiques (suppression de **classes**) | non | non trouvé | R |
| Courbure et mémorisation 2510.24256 | oct. 2025 | arXiv | Édition de poids de ViT/LLM en retirant les directions à faible courbure (mémorisation) | peu | non vérifié | R |
Écartés pour notre métrique : « Do unlearning algorithms learn the right features? » (2611.06681, 4 nov. 2026, mesure les features, pas les sorties) ; « BatchNorm Illusion » (2609.08901, notre ViT n'a pas de BatchNorm) ; DAMP, Spotter (suppression de classes).

### Ce que les lectures apprennent (réponses aux questions du brief)
- **Cible par image qui ressemble à un modèle réentraîné.** ReGUn prend la même cible pour tout le lot (histogramme de classes) : bon pour la moyenne, aveugle à « quelle mauvaise classe ». LTD entraîne un petit professeur sur des voisins et lit ses probabilités sur chaque image : c'est le mécanisme par image qui existe ; il bat la cible « voisins directs » (UA 53 % vs 67 à 83 %, cible gold 47 %), avec k=1000 voisins (similarité cosinus > 0,75), top-3 renormalisé (> 90 % de la masse), β=2 à 3, AdamW lr 1e-4, 10 à 20 époques. Mais testé seulement CIFAR-100/ResNet-56 et SVHN/ViT-Tiny (moins bon : écart moyen 2,6 à 9,8). TREW montre que la similarité **par classe** donne de meilleures cibles que la similarité **par image** (CMIA 82,1 vs 65,6) : β=10, cosinus des vecteurs de classe de la dernière couche, température inverse 5, PCA 32.
- **Données proxy pour garder le modèle** : SCAR : 10 000 images ImageNet ou COCO suffisent pour une suppression de classe (AUS 0,93 à 0,94) ; en « self-forget » le plateau commence vers 6 000 images ; le bruit pur est mauvais ; ces résultats portent sur des classes, jamais sur des images isolées. MU-Mis (sans aucun retain) reste net en dessous de RUM sur les sous-ensembles aléatoires : l'oubli d'images isolées sans retain est la partie dure.
- **Hyperparamètres chiffrés** : voir section 4 (aucun papier ne donne une recette ViT-B 224 complète ; les plus proches sont notés).
- **Concours NeurIPS 2023** (métrique « ressembler au réentraîné », comme ici) : 1er (Fanchuan) : 1 époque de KL vers l'uniforme sur l'oubli, puis 8 époques (batch 256, cosinus : 0,084 puis 0,091 au classement public) alternant contraste forget/retain et CE retain ; 5e : élaguer 99 % des poids, réinitialiser, puis CE + MSE sur l'entropie vers celle d'orig (meilleur score d'oubli, pire utilité) ; 9e (forget-free) : réinitialiser 6 couches tirées au hasard (couche entière > poids isolés ; pas la tête), MSE de distillation vers orig (MSE > CE > L1), plusieurs cycles, lr 1e-3 cosinus ; 10e : SalUn + labels aléatoires + retain. Toutes utilisent du retain. Étonnant : « Random Label » sans réglage bat plusieurs gagnants.

## 4. Méthodes classées, de la meilleure à la plus simple
Notation : orig = modèle fourni ; ŷ = classe prédite par orig (pseudo-label, fiable car orig a vu ces images) ; z = logits ; GPU-min = minutes sur 1 A100 (4 disponibles, essais en parallèle) ; h-p = heures-personne.

### Étape zéro commune (1 h, tout le monde en a besoin)
1. Charger orig et les images, `predict()` du template (fp16, comme le serveur). Histogramme de confiance d'orig : si plus de 95 % des images ont p(ŷ) > 0,9, les pseudo-labels sont fiables.
2. Afficher 40 images : pixelisées = CIFAR-100 agrandi, nettes = sous-ensemble ImageNet, autre = custom. Comparer le nombre d'images par classe prédite (attendu ≈ 35).
3. Poser aux superviseurs : (a) jeux publics autorisés comme proxy ? (b) quel quota de soumissions ? (c) d'où viennent les 100 classes ?
4. Test de 10 min si les classes semblent ImageNet : prendre les mêmes classes dans ImageNet (val), mesurer accuracy d'orig. Si ça colle, on dispose d'images **non vues et labellisées** (loss et accuracy réelles de non-membres = estimation directe du gold), et peut-être du vrai retain. Les nœuds de calcul Jülich n'ont peut-être pas Internet : télécharger depuis le nœud de connexion (à vérifier).

### Méthodes prouvées (littérature) ou adaptées de près
**P1. Plan « ReGUn/LTD par image + ancre SCAR » (proxy OOD)** : rang 3 en espérance, rang 1 en sécurité.
- Étapes : (1) cible q par image = orig tempéré (voir N1 pour mieux) ; (2) pertes : `L = KL(q || softmax(z))` sur l'oubli augmenté (T=1) + `KL(q2 || softmax(z/2))` (T=2) + λ·`KL(softmax(z_orig/2) || softmax(z/2))` sur 10 000 images proxy (ImageNet/COCO si disponibles, sinon mélanges de patchs des images à oublier).
- Réglages de départ : AdamW lr 2e-5 (essayer 1e-5, 5e-5), batch 128 + 128, 6 à 10 époques, cosinus, échauffement 50 pas, wd 0 ; blocs 8 à 11 + norme finale + tête ; `ls1/ls2.gamma` gelés ; bf16.
- Coût : 5 à 10 GPU-min par essai ; 5 à 7 h-p. Risque moyen : plafond de loss (section 2).

**P2. Effacer puis réparer (gagnants NeurIPS 2023, 9e « forget-free »)** : rang 4.
- Étapes : cycles de [réinitialiser 2 à 4 blocs choisis au hasard (poids ~ N(0 ; 0,02), `gamma` remis à 1e-4) ; 1 à 2 époques de distillation MSE des logits d'orig sur proxy + oubli augmenté]. 3 à 4 cycles, lr 1e-4 à 1e-3 cosinus (à baisser pour un ViT-B).
- Valable seulement avec un proxy **dans le domaine** : avec du proxy OOD le modèle réparé perd de l'accuracy (risque de porte 70 %). 10 à 20 GPU-min ; 6 à 8 h-p.

**P3. Filet de sécurité « recalibrage doux »** : rang 5, le plus rapide.
- Étapes : tête + norme finale seules ; cible `q = softmax(z_orig / τ)` avec τ=2 à 3 sur l'oubli augmenté ; ancre KL sur mélanges ; 2 époques, lr 1e-5. Retain ≈ 0,95 et plus ; forget attendu 0,15 à 0,35 (plafond de la section 2). 1 GPU-min ; 1 à 2 h-p ; risque faible. À soumettre en premier.

### Nos idées (non prouvées dans la littérature)
**N1. Cibles « quantiles » : fabriquer la distribution de loss du gold par construction** : rang 1 en espérance.
1. Difficulté d(x) = moyenne des rangs de : (a) 1 - pureté des 10 plus proches voisins parmi les 3500 (cosinus, features d'un bloc intermédiaire, moins mémorisées que la sortie) ; (b) instabilité sous 8 vues fortes (recadrage 0,3 à 1, flou, bruit, couleur) = 1 - moyenne de p_orig(ŷ) ; (c) 1 - probabilité de ŷ hors-échantillon d'une régression logistique à 5 plis sur les features ; (d) entropie d'orig sur les classes ≠ ŷ.
2. Probabilité de la bonne classe cible p_y(x) = sigmoïde(a - b·z(x)) où z est d standardisé ; (a, b) fixés pour que la moyenne et l'écart-type de -log p_y valent (μ_g, σ_g). Cela fixe **les deux moments par construction**.
3. Mauvaise classe : w(x) = softmax d'orig (T=2 à 3) sur les classes ≠ ŷ, incliné par exp(β·s), s = cosinus des vecteurs de classe de la dernière couche (TREW : β jusqu'à 10), mélangé 50/50 avec la prédiction hors-échantillon de l'étape 1(c). Cible `q = p_y·e_ŷ + (1 - p_y)·w`.
4. Cible T=2 : `q2 = 0,85·q + 0,15·w`, plus plate comme une moyenne de plusieurs modèles.
5. Perte : P1 avec ces cibles ; option : perte de moments sur le lot de 256 vues, `(moyenne(L) - μ_g)² + (écart-type(L) - σ_g)²`, poids 1.
6. Copies modifiées : même cible que l'image propre, p_y réduit de 15 % x sévérité de la vue ; augmentations à l'entraînement : recadrage 0,5 à 1, flip, couleur 0,2, flou, bruit, JPEG ; garder en réserve des types **non utilisés** à l'entraînement pour le contrôle.
- Inconnues à régler : (μ_g, σ_g). Sans jeu public : hypothèse initiale accuracy non-membre A = 0,85 (loss moyenne ≈ 0,6, écart-type ≈ 1,2), puis 2 à 3 sondes sur le classement public (A = 0,75 / 0,85 / 0,92). Avec jeu public : mesure directe.
- Coût : 6 à 10 GPU-min par essai ; 8 à 10 h-p (difficulté + cibles + validation). Risque moyen-fort : l'ordre de difficulté est une conjecture ; le simulateur N3 la teste.

**N2. Pseudo-gold : des élèves qui n'ont jamais vu les images** : rang 2, le plus difficile.
- K=4 élèves (un par GPU) : orig, 3 à 4 derniers blocs réinitialisés, entraînés **uniquement** par distillation d'orig (KL T=2) sur le proxy. Ils n'ont aucun contact avec l'oubli ; la moyenne de leurs softmax sur les images à oublier et leurs vues = cible par image avec vraies différences de classes voisines. Puis P1 avec cette cible, ou moyenne des poids des élèves si l'accuracy reste proche d'orig.
- Valable seulement avec un proxy dans le domaine ; sinon l'élève est moins précis qu'orig (biais). 15 à 25 GPU-min par élève ; 12 à 16 h-p.

**N3. Simulateur avec vérité terrain (à lancer en parallèle, agent dédié)** : validation, pas un rendu.
- ViT-Tiny sur CIFAR-100 (patch 4, 32 px, 100 époques) : « orig » sur 45 000 images dont 3 500 à oublier, 3 « golds » sans elles. Calcule la **vraie** métrique, mesure si la difficulté d(x) prédit la loss du gold (corrélation de rang), et quel (a, b) donne les bons moments. Environ 10 à 20 GPU-min par modèle sur 4 A100 ; 2 à 3 h-p. Limite : autre architecture et autre résolution ; ne valide que la méthode, pas les valeurs absolues.

### Hyperparamètres rencontrés (aucun n'est ViT-B 224 complet)
| Source | Modèle | Valeurs |
|---|---|---|
| Benchmark ViT, NegGrad+ | ViT-Small, Swin-T | lr 2e-5, β 0,97, 5 à 10 époques ; Fine-tune lr 1e-4 ; SalUn lr 5e-5 |
| ReGUn | Swin-T 224 | 20 époques, lr 1e-5 à 1e-3, poids retain/oubli w de 0,1 à 0,9 (λ_r = w, λ_f = 1 - w) |
| LTD | ResNet-56 / ViT-Tiny | AdamW lr 1e-4 (ViT-Tiny), β 2 à 3, 10 à 20 époques |
| SCAR | divers, ViT-B/16 | lr 1e-4 à 1e-3, batch 1024, T de 1 à 5, 25 à 30 époques avec arrêt par seuil, λ de 1 à 5 |
| MU-Mis | ViT | Adam, batch 32, arrêt quand la sensibilité revient à son niveau initial |
| Concours 2023 | ResNet-18 | lr 1e-3 cosinus, batch 256, 5 à 8 époques |
| **Notre point de départ** | ViT-B | AdamW lr 1e-5 à 5e-5, 6 à 10 époques, batch 128+128, blocs 8 à 11, bf16 |

## 5. Meilleur, plus difficile, plus technique, plus rapide
- **Meilleur (espérance) : N1**, avec le proxy dans le domaine s'il existe (alors N2 et P1 deviennent possibles) ; sinon N1 + ancre OOD de P1.
- **Plus difficile : N2** (pseudo-gold) : 12 à 16 h-p, utile seulement avec un proxy dans le domaine.
- **Plus technique : N1** (difficulté, quantiles, deux températures, perte de moments).
- **Filet le plus rapide : P3**, 1 à 2 h-p, retain ≈ 0,95+, forget faible mais non nul.
- Plan sur 19 h : H0-1 étape zéro + P3 soumis ; H1-4 N1 minimal (cibles par quantiles, sans option) ; en parallèle N3 par un second agent ; H4-12 balayage (A, τ, β, lr, blocs) sur 4 A100, 3 à 4 sondes publiques au maximum ; H12-16 meilleure variante, moyenne de graines, interpolation `θ = orig + α(θ_unlearn - orig)` ; 3 dernières heures : vérifications finales.

## 6. Évaluation locale que l'on calcule nous-mêmes
Sans labels ni gold, aucun score local n'est une preuve. Quatre niveaux :
1. **Santé (obligatoire)** : en fp16 comme le serveur : `strict=True`, float32, pas de NaN, taille < 450 Mo ; sub = orig doit donner forget 0 ; sub bruité doit effondrer retain.
2. **Fidélité aux cibles** : sur les images à oublier (propres + vues non vues à l'entraînement) : moyenne et écart-type de loss avec ŷ, comparés à (μ_g, σ_g) visés ; KL(q2 || sub) à T=2 ; part de ŷ non prédite.
3. **Retain proxy** : sur proxy (idéal : images non vues du domaine ; sinon OOD) : accord argmax avec orig (> 97 %), KL(orig||sub) à T=2 (< 0,05), accuracy si labels.
4. **Vérité par simulateur N3**, puis le classement public comme capteur (30 %, quota inconnu).
Règle de prudence : ne pas soumettre si l'accord argmax proxy tombe sous 95 % (porte 70 %).

## 7. Pièges
- **Étiquettes inconnues** : on s'entraîne avec ŷ ; le serveur mesure avec les vrais labels ; les rares erreurs d'orig sur ses images d'entraînement coûtent un peu.
- **fp16 côté serveur** : logits trop grands = saturation ; évaluer avec `predict()` du template ; entraîner en bf16 ou fp32, jamais fp16 sans scaler.
- **LayerScale 1e-4** : les `gamma` sont minuscules ; les geler (lr ou wd trop forts les cassent). Si un bloc est réinitialisé, remettre `gamma` à 1e-4.
- **Nom des clés** : pas de préfixe `module.` ; LoRA fusionné avant sauvegarde.
- **Écart-type** : ne régler que la moyenne donne un plafond d'environ 0,35 (section 2).
- **Copies modifiées** : transformations inconnues ; ne pas coller aux pixels exacts ; d_orig y est plus petit donc plus de risque.
- **Bruit du classement public** : 30 % des données seulement ; ne pas affiner un 4e chiffre significatif ; choisir le réglage robuste.
- **Multiplicatif** : retain 0,9 x forget 0,3 = 0,27 ; retain 0,6 x forget 0,6 = 0,36. D'abord retain ≥ 0,9, puis monter le forget.
- **Fuite vers le contrôle** : si le contrôle contient des images d'entraînement très sûres, une baisse de confiance trop large les touche ; mesurer l'accord argmax et KL sur proxy.
- **Ne rien committer** : `model.pt`, `forget_images.npz`.

## 8. Questions (défaut si pas de réponse)
1. Jeux publics comme proxy ? a. demander aux superviseurs maintenant et attendre oui (défaut) / b. s'en passer, proxy OOD seulement.
2. Qui porte T3 ? a. une personne + un agent pour le simulateur N3 (défaut) / b. deux personnes.
3. Premier jet : a. P3 en 2 h puis N1 (défaut) / b. directement N1.

## Sources principales
- ReGUn : https://arxiv.org/html/2603.11210v2
- Local Teacher Distillation : https://arxiv.org/abs/2605.31317
- TREW : https://arxiv.org/abs/2506.20893 ; code https://github.com/CrowdDynamicsLab/ICML_TREW
- MU-Mis : https://arxiv.org/abs/2402.15109
- SCAR : https://arxiv.org/abs/2404.12922 ; code https://github.com/jbonato1/scar
- Benchmark ViT : https://arxiv.org/abs/2602.20114 ; RUM : https://github.com/kairanzhao/RUM
- Rapport concours NeurIPS 2023 : https://arxiv.org/abs/2406.09073 ; scoreur https://github.com/google-deepmind/unlearning_evaluation
- Solutions Kaggle : https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/writeups (1er fanchuan ; 5e Oleszko ; 9e « Forget set-free » ; 10e Tu)
- SCRUB https://arxiv.org/abs/2302.09880 ; SalUn https://arxiv.org/abs/2310.12508 ; NPO https://arxiv.org/abs/2404.05868 ; HAMU https://github.com/aoi3142/HAMU
- Jeu : https://huggingface.co/datasets/SprintML/unlearning (sans fiche)
