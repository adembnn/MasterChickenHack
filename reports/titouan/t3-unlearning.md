# T3 Unlearning : métrique exacte, proxy de gold, simulateur, recettes, plan de nuit

Date : sam. 21 nov. 2026 ~16h. Lu : Unlearning.txt, task_template.py, submission.py, CLAUDE.md, rapport t3 des coéquipiers (non répété : je vais plus loin). Web : seuls les abstracts arXiv 2406.09073 et la fiche HF SprintML/unlearning (« No dataset card yet », aucune info sur le jeu) ont été lus. Tout ce qui touche SCRUB/SalUn/NPO/Bad Teacher/Fisher/SSD/concours NeurIPS 2023 vient de MA MÉMOIRE : marqué (mém.). Rien n'a été téléchargé ni exécuté ; aucune soumission.

## 0. Résumé en 8 lignes
1. La métrique a DEUX moitiés presque indépendantes : (a) moments de la loss (moyenne, écart-type) calculés avec les VRAIES étiquettes, qui ne dépendent que de la DISTRIBUTION des p(vraie classe) sur l'ensemble, pas image par image ; (b) KL image par image vers la moyenne des gold à T=2. On peut donc fabriquer une cible qui règle (a) par construction (« moment-matching ») et (b) par le choix de la classe fausse.
2. Cible gold à estimer = 3 nombres : précision du gold sur les images d'oubli (a_g), moyenne et écart-type de sa loss (m*, s*). Hypothèse forte (§2) : le gold est entraîné de zéro (LayerScale 1e-4 = recette DeiT-III, pas un checkpoint ImageNet), donc a_g ≈ accuracy de validation d'orig, probablement 70 à 85 %, et m* ~ 0,8 à 1,5, s* ~ 1,5 à 2,5 (surconfiance des ViT sur petit jeu). À MESURER avec les sondes du §6, pas à croire.
3. Une température globale ne suffit pas : elle ne crée pas d'erreurs. Il faut que ~(1-a_g) des images d'oubli soient prédites FAUX (classe voisine plausible), choisies parmi les plus « dures ».
4. Meilleure recette (R-A, §5) : distillation douce vers cibles moment-matchées + ancrage KL sur images proxy, derniers blocs, lr 2e-6 à 2e-5, 3 à 8 époques, 3 à 10 min GPU par essai. Amélioration (R-B) : cible = moyenne de pseudo-gold (élèves jamais exposés à l'oubli) si on trouve des images externes.
5. Le facteur n°1 de réussite = les images d'ancrage (retain proxy). À décider en début de soirée : jeu externe sur le cluster ? (§4).
6. Simulateur (§3) : CIFAR-100 (3500/50000 = 7 %, comme 3500 images) avec ResNet-18 ou ViT-Tiny sur 1 A100 : orig + 4 gold en ~25 min, puis ~1 min par recette notée avec la VRAIE formule. Sert à classer les familles de recettes, pas à fixer les valeurs absolues.
7. Protéger la porte 70 % : pas de validation locale possible => règles de sécurité sur proxy (§7) et première soumission conservatrice (alpha = 0,5).
8. Total GPU estimé : 15 à 25 h-A100 (§8), 1 GPU suffit pour avancer, 4 pour le simulateur et R-B.

## 1. La métrique exacte (lue dans le template, le .txt et ce que fait le serveur)
Notation : L = CE par image avec la vraie étiquette (logits à T=1, softmax standard ; seul le KL utilise T=2 : « p = softmax(logits/2) »). gold = moyenne de K modèles réentraînés sans l'oubli (K inconnu, « several »).
- Deux groupes évalués séparément : G1 = les 3500 images d'oubli (30 % pub / 70 % final par hash), G2 = copies ALTÉRÉES cachées de ces images (transformations inconnues ; une copie suit son image source dans la répartition pub/final). Poids égaux.
- Par groupe : d = |mean(L_sub) - mean(L_gold)| + |std(L_sub) - std(L_gold)| ; mean_std = 1 - clip(d/d_orig, 0, 1). mean_gold et std_gold = moyennes, sur les K gold, de la moyenne et de l'écart-type PROPRES à chaque modèle (donc stats d'un modèle individuel, pas du mélange). d_orig = même distance pour orig.
- Par groupe : pred = 1 - clip(mean_i KL(p_gold_i || p_sub_i) / mean_i KL(p_gold_i || p_orig_i), 0, 1) ; p_gold_i = MOYENNE des K softmax T=2 (distribution plus lisse que celle d'un seul modèle).
- forget = (moyenne des 2 mean_std) x (moyenne des 2 pred). Les deux moyennes sont faites AVANT le produit (« each averaged over the two groups first »).
- retain (jeu de contrôle caché) = clip(acc_sub/acc_orig, 0, 1) x 1/(1 + mean KL(p_orig || p_sub)), T=2. Le clip supprime tout gain au-dessus de l'accuracy d'orig. Le sens du KL est orig||sub (différent du forget).
- score = forget x retain ; porte : accuracy < 70 % sur une validation cachée => 0 (forget ET retain mis à zéro).
- Serveur : mode eval, autocast float16, prétraitement x = (pix/255 - 0.5)/0.5, state_dict float32 strict.
Conséquences (déductions) :
- d_orig est grand (orig mémorise : loss ~0 et écart-type ~0 sur l'oubli), donc mean_std est linéaire et tolérant : une erreur de |Δmean|+|Δstd| de 10 % de d_orig coûte 0,1.
- pred est un ratio de KL contre orig, très grand sur images mémorisées : même une cible approximative donne vite 0,5. Plafond réaliste < 1 : un modèle réentraîné indépendant a lui-même un KL > 0 vers la MOYENNE des gold. Un élève entraîné sur une cible moyenne d'ensemble peut battre un seul réentraînement sur ce terme. Plafond à mesurer au simulateur (§3, « gold de test »).
- Tension logit-échelle : les moments (T=1) veulent la confiance d'UN modèle (stats individuelles), le KL (T=2) veut la douceur de la MOYENNE d'ensemble. Même logits pour les deux => compromis à régler (poids entre les deux termes de perte).
- Le retain dépend aussi de ce que contient le « contrôle » : si ce sont des images d'entraînement (mémorisées par orig), tout adoucissement global des sorties coûte du KL ; si ce sont des images de test, orig y est déjà « non-membre » et il faut simplement ne rien y changer. Dans les deux cas : l'effet doit rester LOCAL aux 3500 images (+ leurs altérations), donc ancrage obligatoire.
- Aucun accès aux vraies étiquettes : pseudo-étiquette = argmax d'orig (juste à ~99+ % sur ses images d'entraînement si orig atteint ~100 % de train ; à vérifier via la confiance moyenne : si max-prob moyenne ~0,99, pas de label smoothing ; si ~0,8-0,9, la recette utilisait smoothing/mixup).

## 2. Ce que l'on peut déduire du jeu et du gold (à confirmer en 20 min)
- `init_values=1e-4` n'est pas un défaut timm pour vit_base_patch16_224 (pas de LayerScale) ni DeiT-III (1e-6). Donc modèle entraîné de zéro avec une recette custom. Si oui, gold = ViT-B de zéro sur le jeu moins l'oubli : accuracy gold sur l'oubli ≈ val d'orig (un peu moins si l'oubli retire 7 % du train). Un ViT-B de zéro sur un petit jeu 100 classes surapprend : val-loss élevée (queue lourde de fausses prédictions confiantes) => s* grand.
- Porte à 70 % : suggère accuracy val d'orig entre 75 et 85 % ; marge faible pour un coût d'accuracy. Hypothèse, pas une donnée.
- Mesures à faire sur orig + oubli (10 min) : histogramme des classes prédites (équilibré ~35/classe, ou concentré ? si concentré, l'oubli est ciblé, le gold est pire sur ces classes) ; max-prob moyenne ; loss pseudo-labelisée ; norme des logits ; afficher 40 images (pixelisées 32x32 agrandi => CIFAR-100 probable ; photos nettes => sous-ensemble ImageNet-like).
- Estimateur de (a_g, m*, s*) : (1) valeur a priori = stats d'un « modèle sur images jamais vues » ; (2) si jeu public identifiable : orig sur son split test donne exactement ces nombres (meilleur levier) ; (3) sinon, 3 à 4 sondes leaderboard sur ε (§6). Ne PAS estimer m* depuis les vues dégradées d'orig seules : biais.

## 3. SIMULATEUR local de la métrique (à construire en premier, ~2 h de code, 1 A100)
But : classer les recettes avec la vraie formule avant de brûler des soumissions. Principe : même protocole que le serveur, à petite échelle.
Données/architectures : CIFAR-100 (téléchargement sur nœud de login, à vérifier qu'il a Internet ; les nœuds de calcul n'ont en général pas d'accès, non vérifié). Pas de torch sur les Mac : tout tourne sur JURECA, données chargées en tenseurs GPU (augmentation sur GPU, pas de DataLoader).
- Archi A (rapide) : ResNet-18 CIFAR, 40-60 époques, SGD 0,1 cosine, bf16, ~3 s/époque => 3 min. Archi B (fidèle) : timm `vit_tiny_patch16_224` adapté en 32x32 patch 4 (64 jetons), `init_values=1e-4`, AdamW, 200 époques, ~15 min. Calibrer pour que orig : train ~100 %, test 65-78 %, loss moyenne sur oubli 0,01-0,05 (même niveau de mémorisation que le vrai).
- Ensembles : forget F = 3500 images au hasard (7 % comme le vrai ; faire aussi une variante F non uniforme par classe). orig = entraîné sur 50k. gold_1..4 = 4 graines, 50k \ F. gold_test = 5e graine (« modèle parfait » indépendant) : son score donne le PLAFOND atteignable ; sub = orig doit donner ~0 ; ces deux contrôles valident le code de métrique.
- Copies altérées : famille A (recadrage+flip, flou gaussien, bruit, jitter couleur, petite rotation, JPEG simulé) pour l'évaluation ; on entraîne la recette avec une famille B différente (et on teste l'inverse) pour mesurer la généralisation aux altérations inconnues.
- Contrôle : 2 variantes, (i) 5000 images de test, (ii) 5000 images de train retenues. Validation/porte : 5000 autres images test, porte adaptée = accuracy_orig - 10 points.
- Proxy (ce que l'unlearneur a le droit de voir) en 3 régimes : P0 aucun, P1 hors distribution (CIFAR-10 ou SVHN redimensionnés), P2 oracle (10k images retain sans labels). Mesure de la perte due à l'OOD = différence P1/P2.
- Contrainte d'information stricte : l'unlearneur reçoit orig + images de F (sans labels) + proxy, jamais gold ni retain.
Code de la métrique (à écrire une fois, utilisé partout) :
```
def group(sub_logits, orig_logits, gold_logits_list, y):          # logits [N,100] ; gold_logits_list: K tenseurs
    ce = lambda z: F.cross_entropy(z, y, reduction="none")
    Ls, Lo = ce(sub_logits), ce(orig_logits)
    gm = mean(ce(g).mean() for g in gold_logits_list); gs = mean(ce(g).std() for g in gold_logits_list)
    d  = lambda L: (L.mean()-gm).abs() + (L.std()-gs).abs()
    ms = 1 - clip(d(Ls)/d(Lo), 0, 1)
    pg = mean(softmax(g/2) for g in gold_logits_list)
    kl = lambda q: (pg*(log(pg)-log_softmax(q/2))).sum(1).mean()
    return ms, 1 - clip(kl(sub_logits)/kl(orig_logits), 0, 1)
forget = mean(ms1,ms2) * mean(pr1,pr2) ; retain = clip(acc_s/acc_o,0,1) / (1 + KL_T2(orig||sub)) ; score = forget*retain
```
(std : torch.std non biaisé ou biaisé ? inconnu ; sans effet sur les classements.) Évaluer en fp16 autocast comme le serveur.
Expériences prioritaires du simulateur (chacune ~1 min la recette, GPU 1) :
1. Plafond : score de gold_test ; sensibilité du score à ±30 % d'erreur sur (m*, s*) => combien de sondes leaderboard il faut.
2. Classement des familles R0-R10 (§5) sous P0/P1/P2, avec la formule complète. Mesurer aussi score séparé forget/retain et la porte.
3. Profondeur de ré-init et de blocs entraînables ; lr ; époques ; lambda d'ancrage ; alpha d'interpolation ; nombre de vues K.
4. Effet de ε (fraction forcée à l'erreur) sur le score : forme de la courbe (unimodale ?) pour dimensionner les sondes réelles.
5. Robustesse aux altérations : train famille A / test famille B.
6. Pseudo-gold (R-B) vs cible tempérée simple : l'écart de pred. Si R-B ne gagne pas > 0,05 de forget au simulateur, ne PAS le faire en réel.
Limites : CIFAR 32 px, petit modèle, du vrai ViT-B on ne mesure que la direction. Les hyperparamètres absolus (lr) se règlent sur le vrai ViT-B ; le simulateur choisit la famille et les ordres de grandeur. Calendrier : voir §9.

## 4. Proxy de gold et images d'ancrage SANS données de rétention
Trois fonctions distinctes, à ne pas confondre :
(1) cible de forget par image (ce que prédirait le gold) ; (2) ancrage retain (images où sub = orig) ; (3) estimation de m*, s*.
Sources d'images d'ancrage, par fiabilité :
- a. Données externes sur le cluster : vérifier `ls /p/data1/datasets` , le dossier partagé `/p/scratch/training2666/`, les caches HF (non vérifié que ImageNet y soit). Le dépôt SprintML/hackathon télécharge seulement les jeux des 3 tâches. Si Internet au login node : télécharger CIFAR-100/Imagenette/Tiny-ImageNet dans le scratch (je n'ai pas le droit de le faire ; l'équipe oui). Question à poser aux superviseurs : jeux publics autorisés ? (le texte dit seulement qu'aucune donnée supplémentaire n'est fournie, et interdit la collaboration, pas les jeux publics.) Si les images du jeu viennent d'ImageNet/CIFAR, utiliser les images publiques de ces classes serait un quasi-retain : à signaler explicitement aux organisateurs avant, pour éviter une disqualification.
- b. Images de l'oubli corrompues (mosaïque 2x2 d'images d'oubli, mélange patch, flou fort, 30 % de patchs masqués) avec cible = sortie d'orig : pas de mémorisation exploitable, ancre bon marché, locale. Pas de bruit gaussien pur (mauvais, SCAR).
- c. Images synthétisées depuis orig (inversion type DeepInversion : optimiser pixels pour classe c avec TV + statistiques, 2 à 5k images, ~10 min sur 4 GPU) : qualité moyenne, utile pour couvrir toutes les classes.
- d. Augmentations des images d'oubli avec cible = orig tempérée : dernier recours, risque de réapprendre la mémorisation.
Constructions de cibles (par image i, pseudo-label y_i = argmax orig) :
- Vues : V = 8 à 16 vues dégradées (RandomResizedCrop échelle 0,25-0,7, flou, cutout, jitter). p̃_i = moyenne des softmax d'orig (T=1) sur les vues. Dureté h_i = 1 - p̃_i[y_i] (+ désaccord entre vues). La mémorisation baisse sous dégradation forte : p̃ approche une prédiction « non-membre ».
- Voisinage de classes : matrice de confusion douce C[y, c] = moyenne des p̃_i[c] sur les images de pseudo-classe y (35 images/classe). Donne QUELLE classe fausse le gold choisirait. Variante LOO : tête linéaire/kNN sur features d'orig en validation croisée 10 plis.
- Moment-matching : choisir ε (fraction d'images forcées FAUSSES) et les paramètres d'une loi pour q_i = p cible de la vraie classe :
  1. trier par h_i ; les ⌊ε·N⌋ plus dures : q_i ~ Beta/log-normale centrée 0,1-0,3 (loss 1,2 à 2,3+), classe fausse = argmax_{c≠y} p̃_i[c] ; les autres : q_i = 1 - δ_i avec δ_i croissant avec h_i.
  2. Résoudre numériquement (m*, s*) : recherche à 2 paramètres (ε, échelle de q) pour que mean/std de -log q_i = (m*, s*). Reste de la masse sur les autres classes ∝ p̃_i (T=2).
  3. Cohérence par image : MÊME cible pour toutes les vues/altérations d'une image.
- Pseudo-gold R-B : élèves « blancs » (réinit tête + 2-6 derniers blocs, gamma LayerScale remis à 1e-4, éventuel bruit σ sur le reste) qui distillent orig UNIQUEMENT sur images externes ; leur moyenne sur l'oubli = p_gold-proxy. Pourquoi ça marche : un élève qui n'a jamais vu l'image généralise comme un non-membre. Risque : mémorisation résiduelle dans les couches gardées (la mémorisation n'est pas localisée dans les dernières couches, Maini et al. 2023 (mém.)), biais OOD. Le simulateur (exp. 3 et 6) tranche profondeur de ré-init et utilité.
- Autre proxy sans externe : pas possible d'entraîner un ViT-B de zéro sur 3500 images pour imiter un gold (trop peu de données) ; un petit modèle entraîné par validation croisée sur l'oubli seul (moitié A pour entraîner, moitié B pour prédire) donne des cibles « non-membre » mais de qualité faible ; à tester seulement au simulateur.
Température de calibration : trois températures différentes : (i) T_cible (adoucit orig pour fabriquer t_i, 1,5 à 5), (ii) T=1 des moments (CE d'entraînement), (iii) T=2 du KL du score. Recommandation : entraîner avec CE(t_i, softmax(z)) + β·KL(t̄_i^{T=2} || softmax(z/2)) où t̄ est la cible d'ensemble ; β {0, 0,5, 1}. Balayage T_cible {1,5 ; 2 ; 3 ; 5} seulement dans le cas de cibles non moment-matchées.

## 5. Recettes classées (littérature + les nôtres)
Rappel littérature (mém., à ne pas croire sur parole) : SCRUB (Kurmanji 2023, arXiv 2302.09880) : min-max KL élève/prof sur forget + retain, besoin de retain. NegGrad+ : ascent sur forget + descent sur retain. SalUn (2310.12508) : masque de saillance + étiquettes aléatoires. Bad Teacher (Chundawat 2023) : prof compétent (orig) sur retain, incompétent (aléatoire) sur forget. Fisher/NTK/influence (Golatkar), SSD/LFSSD (Foster) : pondération par importance, besoin du jeu complet. Boundary unlearning, l1-sparse (Jia 2023), Amnesiac. NPO (2404.05868). Concours NeurIPS 2023 (2406.09073) : les meilleures soumissions = « bruiter/réinitialiser puis réparer par distillation », très proches de notre R-B. Toutes ces méthodes cherchent à s'éloigner d'orig ou à casser une cible ; AUCUNE ne vise le comportement d'un réentraîné image par image ; la nôtre oui. Aucune n'utilise l'absence totale de retain.
Classement (dans l'ordre où je les jouerais) :
- R-A (rang 1) : cibles moment-matchées + ancrage. Init = orig, orig figé comme professeur. Perte = CE(t_i, p_sub) sur vues de l'oubli (cibles §4) + β KL T=2 vers cible d'ensemble + λ KL(p_orig||p_sub, T=2) sur ancres. Entraînables : blocs 6-11 + norm + head (≈ 43 M) ; essayer tous les blocs avec décroissance lr par couche 0,75 ; geler patch_embed/pos_embed/cls_token et les gamma LayerScale (ou lr 0,1x). AdamW lr pic {2e-6, 5e-6, 1e-5, 2e-5} cosine, warmup 10 %, wd 0, bf16 autocast + poids fp32 maître, lot 64 oubli + 64 ancres, clip grad 1,0. Époques sur 3500 : {3, 5, 8}. λ {1, 5, 20}, β {0, 0,5, 1}, ε {0,10 ; 0,15 ; 0,20 ; 0,30}. Coût : ~(3500+3500·ratio)/128 ≈ 60 pas/époque, ~0,35 s/pas => 3 à 10 min par essai sur 1 A100 ; 40 essais ≈ 4-6 h-GPU.
- R-A' : R-A + interpolation de poids theta = orig + α(theta_unlearn - orig), α {0,5 ; 0,7 ; 0,85 ; 1} (gratuit, ajuste oubli contre retain après coup) + moyenne de 2-3 graines (imite la moyenne de gold, réduit la variance).
- R-B (rang 2, si ancres externes) : pseudo-gold K=4 élèves (1 par GPU). Élève : init orig, réinit tête + 4 derniers blocs, AdamW 1e-4 pic, wd 0,05, KD T=2 vers orig sur 50-100k images externes, 5-10 époques, ~400-500 img/s/GPU (A100, bf16) => 100k x 6 époques ≈ 25-30 min par élève, 4 en parallèle ; débit disque/JPEG à surveiller (16 workers/GPU). Ensuite cibles d'oubli = mélange de p̃ (§4) et moyenne des élèves, puis R-A. Garder la décision au simulateur (exp. 6).
- R-C (rang 3) : SCRUB / bad-teacher avec ancres du §4 ; NegGrad+ avec cible tempérée au lieu d'ascent brut ; NPO doux (beta 0,1-1). Utile comme comparaison ; plus sensible au sur-oubli.
- R-S (filet) : recalibrage de la tête seulement (logit scale + biais de classe), 1-2 h, forget faible, retain ~intact SI contrôle=test ; mauvais si contrôle=train (voir §1).
- À écarter : ascent brut, étiquettes aléatoires seules (le gold ne se trompe pas au hasard), SSD (importance sur tout le train).

## 6. Sondes leaderboard (le seul vrai juge)
Nombre de soumissions inconnu : à demander aux organisateurs dès maintenant. Hypothèse prudente : ~5-10 de utiles. Utilisation :
1. S0 (sécurité, ~19h) : R-A avec α = 0,5, ε moyen (0,2). But : score non nul + format validé.
2. S1/S2/S3 : même modèle, ε ∈ {0,10 ; 0,20 ; 0,30} (ou deux valeurs de m*). Ajuster une parabole sur le score public (30 % seulement, bruité : différence < 0,03 non significative).
3. Puis un seul réglage à la fois (α, λ).
4. Ne PAS tuner finement sur le public (70 % final cachés) : prendre le plateau, pas le pic.
Test de fraîcheur du public : le score est un produit ; un score public très bas peut venir du retain (KL) ou de la porte (0) : sonder avec α=0,3 d'abord pour séparer.

## 7. Garde-fous 70 % et KL retain (règles de décision concrètes)
Sans validation locale, deux proxys obligatoires sur images ancres/externes avec pseudo-labels d'orig : accord argmax orig/sub et KL_T2(orig||sub).
- Ne PAS soumettre si : accord argmax < 95 %, ou KL_T2 > 0,05 (retain facteur 0,95), ou logits non finis en fp16, ou la norme des logits dérive de > 20 % vs orig.
- Chute d'accuracy estimée ≈ (1 - accord) ; accord 95 % => ≲ 3-5 points. Si val_orig est ~75 %, la marge vers 70 % est mince : viser accord ≥ 97 %.
- Mesurer l'accord sur images « proches de la distribution » (altérations légères de l'oubli ne comptent pas : mémorisation). Un modèle qui n'oublie que les 3500 images garde l'accord ailleurs.
- Mesurer après chaque entraînement exactement comme le serveur (`predict()` du template, fp16 autocast, entrées [-1,1]).
- Early stop sur KL ancre et sur le moment-matching (loss moyenne/écart-type sur pseudo-étiquette comparés à (m*, s*) cibles).
- Toujours garder sous la main une soumission « sûre » déjà validée (α=0,5) pour ne pas finir sans résultat.
- Avant toute soumission, script `verify_submission.py` (à écrire cette nuit) : charge en `build_model()` strict=True ; clés/formes identiques à model.pt ; tous les tenseurs float32 ; aucun NaN/Inf ; taille < 450 Mo (~345 attendu) ; sorties de `predict()` identiques entre le fichier relu et le modèle en mémoire.

## 8. Pièges de format et d'implémentation
- state_dict : pas de préfixe `module.` (DDP, `AveragedModel` SWA => préfixe `module.` !) ni `_orig_mod.` (torch.compile). Sauver `{k: v.detach().cpu().float() for k,v in model.state_dict().items()}` ; ne pas sauver le modèle entier. Pas de LoRA non fusionnée (fusion dans nn.Linear : timm qkv = un Linear 768→2304 ; W += (alpha/r)·B@A).
- Entraîner en bf16 autocast mais poids float32 ; jamais half() avant de sauver (float16 rejeté).
- Prétraitement serveur = (x/255 - 0.5)/0.5 ; PAS de moyenne/écart-type ImageNet. Les augmentations doivent rester dans [-1,1] et se faire en uint8 -> float avant normalisation (resize bilinéaire avec antialias ; arrondi uint8 sur les vues alterées pour ressembler aux copies serveur).
- fp16 côté serveur : la loss d'entraînement est en bf16 ; vérifier l'absence de débordement fp16 (activations > 65504 dans les blocs tardifs) et l'écart logits fp16/fp32.
- LayerScale 1e-4 : gamma minuscules ; weight decay ou lr trop forts les déstabilisent ; geler. Réinit d'un bloc : gamma = 1e-4.
- Pas de dropout ni drop_path à l'entraînement (build_model par défaut = 0). Mode train/eval identiques sinon.
- Rangement : modèle d'orig figé en eval pour les cibles ; précalculer les cibles des vues hors ligne (tenseur [N,V,100]) pour accélérer.
- Ne rien committer (model.pt, forget_images.npz, soumissions). La clé API reste dans .env (submission.py demande de l'éditer : ne jamais pousser la copie éditée).
- Soumission : POST de ~345 Mo, attente jusqu'à 500 s côté serveur : lancer depuis un nœud avec bonne bande passante, pas depuis le Wi-Fi hôtel ; relancer en cas de timeout n'est pas gratuit si les soumissions sont comptées (demander).

## 9. Plan chronométré (sam. 16h30 -> dim. 12h, code gelé)
H0 = 16h30. Un GPU = un 'tmux' par personne, pas de double usage du nœud sans prévenir sur le bus (#2).
- 16h30-17h30 (H+1) : jalon J1 : environnement Jülich prêt, model.pt/forget_images.npz chargés, 40 images affichées, histogramme de classes et max-prob d'orig, `verify_submission.py` écrit, questions envoyées aux organisateurs (jeux publics ? nombre/limite de soumissions ?). Inspection des datasets disponibles sur le cluster.
- 17h30-19h30 (H+3) : jalon J2 : simulateur : entraînement orig + gold_1..4 + gold_test (ResNet-18, 5 GPU-jobs 1 nœud) ; code de la métrique validé (orig≈0 ; gold_test = plafond). En parallèle (autre personne) : pipeline réel ViT-B : pseudo-labels, vues, cibles p̃, mini-scoreur.
- 19h30-22h30 (H+6) : jalon J3 : sweeps du simulateur (exp. 1-6, ~3 h-GPU), choix de la famille et des plages ; réel : R-A implémenté, premier modèle sur ViT-B, `verify_submission.py` passé.
- 22h30-23h (H+6,5) : jalon J4 : S0 soumis (α = 0,5, ε = 0,2). Résultat > 0 = format et porte OK. Sinon diagnostic (porte ? format ?) avant tout.
- 23h-02h (H+9,5) : sondes ε (S1-S3) et mise à jour de (m*, s*) ; balayage lr/époques/λ sur ViT-B (~20 essais, 1 GPU), 2 GPU en parallèle.
- 02h-06h (H+13,5) : R-B (si ancres externes et si le simulateur l'a validé) sur 4 GPU ; rotation de repos de l'équipe (T1/T2 non négligeables : ne pas garder T3 seule sur toute la nuit ; T3 consomme peu de GPU, peu d'humains après J4).
- 06h-09h (H+16,5) : jalon J5 : modèle R-A'/R-B final : moyenne de 2-3 graines + α ; contrôles §7.
- 09h-10h30 : jalon J6 : soumission finale + une dernière de secours si autorisé. Choisir le plateau, pas le pic du public.
- 10h30-12h : tampon, push du code (sans données ni poids), notes pour le pitch (6 min : le message = « copier le comportement du gold, pas casser le modèle », avec la courbe du simulateur).
Budget GPU : simulateur 3-4 h ; ViT-B R-A 6-8 h ; R-B 4 GPU x 1 h = 4 h ; marge 4 h => 15 à 25 h-A100.
Critères de décision : (i) si S0 = 0 : porte ou format ; (ii) si S0 < 0,1 : cibles trop douces (forget quasi nul), augmenter ε/α ; (iii) si score forget élevé mais retain bas : baisser lr/α, augmenter λ ; (iv) rien ne bouge entre S1-S3 : le public est dominé par le retain ou par pred.

## 10. Risques et incertitudes
- Aucune vérité terrain locale : (a_g, m*, s*) inconnus ; l'estimateur repose sur des hypothèses (§2) non vérifiées. Les sondes limitent le risque, pas à zéro.
- Altérations cachées inconnues : sur-ajuster à certaines altérations ne prépare pas aux autres ; entraîner avec des familles variées.
- Public 30 % / final 70 % : différence de composition possible ; ne pas sur-optimiser.
- Le simulateur peut classer autrement qu'en réel (modèle et résolution différents) ; ne l'utiliser que pour des écarts nets (> 0,05).
- Légalité des images externes et usage éventuel de poids préentraînés (par exemple pour la similarité de classes) : à demander avant d'utiliser ; sinon rester sur §4 b-d.

## 11. Questions à trancher (défaut entre parenthèses)
1. Données externes autorisées ? (demander ; en attendant, préparer §4 b-c.)
2. Nombre de soumissions ? (hypothèse 5-10, à confirmer.)
3. Qui fait le simulateur ? (Adem : 2 h après la connexion ; Titouan : pipeline réel ViT-B.)

## Sources
- Énoncé et gabarit : reports/sources/Unlearning.txt, t3/task_template.py, t3/submission.py
- NeurIPS 2023 (abstract lu) : https://arxiv.org/abs/2406.09073
- SCRUB : https://arxiv.org/abs/2302.09880 ; Zero-shot : https://arxiv.org/abs/2201.05629 ; SalUn : https://arxiv.org/abs/2310.12508 ; NPO : https://arxiv.org/abs/2404.05868 ; SCAR : https://arxiv.org/abs/2404.12922 ; LFSSD : https://arxiv.org/pdf/2402.19308 ; ReGUn : https://arxiv.org/html/2603.11210v1
- Fiche HF (aucune carte) : https://huggingface.co/datasets/SprintML/unlearning
