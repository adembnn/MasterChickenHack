# T3 Unlearning : deux candidats, diagnostics mesurés (v1, 21 nov. 2026)

Rien n'a été soumis. Tout est mesuré sur JURECA (A100, fp16 comme le serveur), avec `orig` pour référence. Scripts : `tracks/t3-unlearning/scripts-t3v1/`. Fichiers .pt : cluster `adem/candidates/` (jamais dans git). Aucune image externe n'a servi à entraîner : seulement les 3500 images à oublier, leurs vues et leurs mélanges.

## En images simples
Orig a appris les 3500 fiches par cœur. Le modèle de référence (le « gold », jamais exposé à ces fiches) hésiterait, et se tromperait parfois vers une classe voisine.
- **S0** baisse seulement le « volume de confiance » (tête et norme finale). Il ne change aucune réponse. Sûr, mais il oublie presque rien.
- **S1** fabrique pour chaque fiche une réponse attendue : une fiche « difficile » reçoit une probabilité plus basse pour la bonne classe, avec une classe voisine pour l'erreur. Il oublie bien plus, mais dérive plus ailleurs.

## Ce que fait chaque candidat
- **S0** (`S0_tau1.5/2/3`) : seuls head.weight/bias et norm.weight/bias changent. Cible softmax(z_orig/τ) sur des vues d'images à oublier (recadrage, flip, couleur, flou, bruit légers). Ancre : sorties d'orig à T=2 sur des mélanges (mixup, cutmix, blocs permutés) faits uniquement d'images à oublier, λ=10.
- **S1** (méthode « quantiles ») : 4 derniers blocs + norme + tête, gammas LayerScale gelés, AdamW, 800 pas lr 5e-5 (tour 1) ou 3000 pas (tour 2, préfixe `S1_r2`). Difficulté par image = moyenne de 3 rangs : instabilité sous 8 vues fortes, impureté des 10 plus proches voisins (bloc 8), difficulté hors-échantillon d'une régression logistique à 5 plis. Probabilité de la bonne classe p = sigmoïde(a − b·difficulté), (a, b) ajustés pour que moyenne et écart-type de −log p valent (μ, σ). Mauvaise classe : softmax d'orig à T=3 hors classe prédite, penchée vers les classes aux vecteurs de tête proches (cosinus), mêlée à 50 % à la prédiction hors-échantillon. Perte : KL à T=1 + KL à T=2 + λ·ancre + perte de moments. `_l30` = ancre λ=30, sinon 10. `_bc` = difficulté sans la mesure d'instabilité.
- **(μ, σ) selon A** (précision supposée du gold sur ces images) : modèle de mélange que j'ai posé, NON mesuré : bien classées loss 0,12 (écart-type 0,25), mal classées loss 3,0 (écart-type 1,3). Donne A=0,75 : 0,84 / 1,42 ; A=0,85 : 0,55 / 1,17 ; A=0,92 : 0,35 / 0,90 (moyenne / écart-type).

## Mesures
Contrôles de fichier, tous les modèles : chargement strict OK, float32, aucun NaN/Inf, 327,7 Mo (limite 450), mêmes clés qu'orig, forward fp16 fini sur tous les jeux. `validate_t3.py --reference model.pt` : « VALIDE (format) » pour les 10 fichiers du tour 1 ; accuracy non vérifiée. Tour 2 : contrôles du diagnostic faits, `validate_t3.py` pas relancé.
Légende : « pseudo-label » = classe prédite par orig (confiance moyenne 0,9997, donc fiable). « Copies » = 4 types jamais vus à l'entraînement : JPEG q50, rotation 15°, couleur forte + teinte, flou fort σ 1,5–2,5 (couleur et flou sont de la même famille que l'entraînement, à intensité plus forte ; JPEG et rotation sont nouveaux). « KL cible propre » = KL(cible T=2 du candidat ‖ sub). « Pseudo-gold A=0,85 » = cible de S1_A85, une hypothèse.

### Tableau 2. Oubli, images exactes (vs pseudo-label = prediction d'orig)
| modele | CE moy | CE ecart-type | part argmax != pseudo-label | max-prob moy | KL(cible propre, T=2) | KL(pseudo-gold A=0,85, T=2) |
|---|---|---|---|---|---|---|
| orig | 0.0003 | 0.0024 | 0.000 | 0.9997 | nan | 3.109 |
| S0_tau1.5 | 0.0008 | 0.0059 | 0.000 | 0.9992 | 0.112 | 2.634 |
| S0_tau2 | 0.0024 | 0.0129 | 0.000 | 0.9977 | 0.531 | 2.017 |
| S0_tau3 | 0.0119 | 0.0405 | 0.000 | 0.9889 | 1.259 | 1.195 |
| S1_A75 | 0.3466 | 1.1386 | 0.080 | 0.9158 | 1.120 | 0.889 |
| S1_A75_l30 | 0.1487 | 0.9383 | 0.028 | 0.9765 | 1.884 | 1.502 |
| S1_A85 | 0.2213 | 0.8491 | 0.051 | 0.9377 | 0.922 | 0.922 |
| S1_A85_bc | 0.2304 | 0.8678 | 0.055 | 0.9360 | 0.931 | 0.943 |
| S1_A85_l30 | 0.0874 | 0.6376 | 0.019 | 0.9819 | 1.619 | 1.619 |
| S1_A92 | 0.1401 | 0.5890 | 0.034 | 0.9519 | 0.788 | 0.973 |
| S1_A92_l30 | 0.0433 | 0.3752 | 0.011 | 0.9866 | 1.408 | 1.712 |
| S1_r2_A85_l10_lr1e-4 | 0.3757 | 0.9605 | 0.093 | 0.8785 | 0.389 | 0.389 |
| S1_r2_A85_l30_lr1e-4 | 0.2491 | 0.8540 | 0.063 | 0.9364 | 0.865 | 0.865 |
| S1_r2_A85_l30_lr5e-5 | 0.2174 | 0.8476 | 0.051 | 0.9499 | 1.049 | 1.049 |
| S1_r2_A92_l30_lr1e-4 | 0.1482 | 0.5822 | 0.042 | 0.9506 | 0.775 | 0.964 |

### Tableau 3. Oubli, 4 types de copies non utilisees a l'entrainement (moyenne des 4 ; detail ensuite)
| modele | CE moy | CE ecart-type | part argmax != pseudo-label | max-prob moy | KL(cible propre, T=2) | KL(pseudo-gold A=0,85, T=2) |
|---|---|---|---|---|---|---|
| orig | 0.9339 | 1.5699 | 0.158 | 0.9387 | nan | 2.531 |
| S0_tau1.5 | 0.9469 | 1.5374 | 0.167 | 0.9299 | 0.636 | 2.175 |
| S0_tau2 | 0.9560 | 1.4775 | 0.181 | 0.9152 | 0.793 | 1.724 |
| S0_tau3 | 1.0074 | 1.4137 | 0.206 | 0.8789 | 1.120 | 1.163 |
| S1_A75 | 1.3445 | 1.8691 | 0.308 | 0.7888 | 1.037 | 0.941 |
| S1_A75_l30 | 1.1061 | 1.8284 | 0.229 | 0.8693 | 1.528 | 1.311 |
| S1_A85 | 1.1774 | 1.6676 | 0.270 | 0.8053 | 0.934 | 0.934 |
| S1_A85_bc | 1.1808 | 1.6681 | 0.271 | 0.8050 | 0.944 | 0.946 |
| S1_A85_l30 | 1.0361 | 1.6804 | 0.214 | 0.8793 | 1.386 | 1.386 |
| S1_A92 | 1.0698 | 1.5062 | 0.248 | 0.8175 | 0.874 | 0.950 |
| S1_A92_l30 | 0.9832 | 1.5601 | 0.201 | 0.8869 | 1.270 | 1.445 |
| S1_r2_A85_l10_lr1e-4 | 1.1782 | 1.5200 | 0.268 | 0.6981 | 0.600 | 0.600 |
| S1_r2_A85_l30_lr1e-4 | 1.1878 | 1.6131 | 0.275 | 0.7640 | 0.881 | 0.881 |
| S1_r2_A85_l30_lr5e-5 | 1.1704 | 1.6820 | 0.265 | 0.8014 | 0.992 | 0.992 |
| S1_r2_A92_l30_lr1e-4 | 1.0668 | 1.4742 | 0.244 | 0.7870 | 0.858 | 0.922 |

Detail par type (CE moy / part argmax != pseudo-label) :
| modele | JPEG q50 | rotation 15 deg | couleur forte+teinte | flou fort s=1.5-2.5 |
|---|---|---|---|---|
| orig | 0.001 / 0.00 | 0.016 / 0.01 | 3.104 / 0.48 | 0.614 / 0.15 |
| S0_tau1.5 | 0.002 / 0.00 | 0.020 / 0.01 | 3.093 / 0.49 | 0.672 / 0.17 |
| S0_tau2 | 0.006 / 0.00 | 0.028 / 0.01 | 3.042 / 0.52 | 0.748 / 0.20 |
| S0_tau3 | 0.022 / 0.00 | 0.054 / 0.02 | 3.007 / 0.56 | 0.947 / 0.25 |
| S1_A75 | 0.370 / 0.09 | 0.459 / 0.12 | 3.032 / 0.60 | 1.517 / 0.41 |
| S1_A75_l30 | 0.157 / 0.03 | 0.210 / 0.05 | 3.002 / 0.54 | 1.056 / 0.29 |
| S1_A85 | 0.239 / 0.06 | 0.309 / 0.08 | 2.874 / 0.58 | 1.288 / 0.36 |
| S1_A85_bc | 0.246 / 0.06 | 0.311 / 0.08 | 2.872 / 0.58 | 1.294 / 0.36 |
| S1_A85_l30 | 0.093 / 0.02 | 0.139 / 0.04 | 2.958 / 0.53 | 0.953 / 0.27 |
| S1_A92 | 0.154 / 0.04 | 0.217 / 0.06 | 2.771 / 0.56 | 1.137 / 0.33 |
| S1_A92_l30 | 0.050 / 0.01 | 0.090 / 0.03 | 2.922 / 0.52 | 0.871 / 0.25 |
| S1_r2_A85_l10_lr1e-4 | 0.358 / 0.09 | 0.555 / 0.12 | 2.424 / 0.53 | 1.376 / 0.33 |
| S1_r2_A85_l30_lr1e-4 | 0.238 / 0.06 | 0.411 / 0.11 | 2.686 / 0.55 | 1.417 / 0.38 |
| S1_r2_A85_l30_lr5e-5 | 0.215 / 0.05 | 0.340 / 0.09 | 2.804 / 0.55 | 1.323 / 0.37 |
| S1_r2_A92_l30_lr1e-4 | 0.147 / 0.04 | 0.275 / 0.07 | 2.633 / 0.53 | 1.212 / 0.33 |

### Tableau 4. Derive par rapport a orig (accord argmax / KL(orig||sub) a T=2 / facteur 1/(1+KL))
| modele | melanges d'images oubliees | OOD ImageNet-100 | OOD CIFAR-100 |
|---|---|---|---|
| orig | 1.0000 / 0.0000 / 1.0000 | 0.9990 / 0.0000 / 1.0000 | 0.9990 / 0.0000 / 1.0000 |
| S0_tau1.5 | 0.9717 / 0.0057 / 0.9943 | 0.9564 / 0.0062 / 0.9938 | 0.9170 / 0.0176 / 0.9827 |
| S0_tau2 | 0.9293 / 0.0328 / 0.9682 | 0.9098 / 0.0326 / 0.9685 | 0.8408 / 0.0728 / 0.9321 |
| S0_tau3 | 0.8637 / 0.1280 / 0.8865 | 0.8352 / 0.1166 / 0.8956 | 0.7284 / 0.2172 / 0.8216 |
| S1_A75 | 0.8203 / 0.1573 / 0.8641 | 0.7598 / 0.2268 / 0.8151 | 0.6882 / 0.4942 / 0.6692 |
| S1_A75_l30 | 0.9047 / 0.0526 / 0.9500 | 0.8644 / 0.0790 / 0.9268 | 0.7838 / 0.2639 / 0.7912 |
| S1_A85 | 0.8507 / 0.1270 / 0.8873 | 0.7934 / 0.1920 / 0.8389 | 0.7168 / 0.4936 / 0.6695 |
| S1_A85_bc | 0.8463 / 0.1294 / 0.8855 | 0.7874 / 0.2039 / 0.8306 | 0.7116 / 0.4797 / 0.6758 |
| S1_A85_l30 | 0.9167 / 0.0406 / 0.9609 | 0.8802 / 0.0641 / 0.9397 | 0.8046 / 0.2321 / 0.8116 |
| S1_A92 | 0.8727 / 0.1055 / 0.9045 | 0.8152 / 0.1707 / 0.8542 | 0.7424 / 0.4940 / 0.6694 |
| S1_A92_l30 | 0.9260 / 0.0323 / 0.9688 | 0.8882 / 0.0538 / 0.9489 | 0.8208 / 0.2078 / 0.8280 |
| S1_r2_A85_l10_lr1e-4 | 0.8677 / 0.0794 / 0.9265 | 0.7722 / 0.3287 / 0.7526 | 0.6380 / 1.1393 / 0.4674 |
| S1_r2_A85_l30_lr1e-4 | 0.8950 / 0.0481 / 0.9541 | 0.8072 / 0.2065 / 0.8288 | 0.6482 / 1.0092 / 0.4977 |
| S1_r2_A85_l30_lr5e-5 | 0.9127 / 0.0438 / 0.9581 | 0.8340 / 0.1540 / 0.8665 | 0.6950 / 0.7641 / 0.5669 |
| S1_r2_A92_l30_lr1e-4 | 0.9030 / 0.0424 / 0.9593 | 0.8264 / 0.1766 / 0.8499 | 0.6802 / 0.9345 / 0.5169 |

KL(orig||sub) sur les images oubliees exactes (T=2) : orig 0.0000, S0_tau1.5 0.0038, S0_tau2 0.0249, S0_tau3 0.1319, S1_A75 0.5268, S1_A75_l30 0.1904, S1_A85 0.4128, S1_A85_bc 0.4187, S1_A85_l30 0.1349, S1_A92 0.3339, S1_A92_l30 0.0952, S1_r2_A85_l10_lr1e-4 0.7849, S1_r2_A85_l30_lr1e-4 0.4532, S1_r2_A85_l30_lr5e-5 0.3669, S1_r2_A92_l30_lr1e-4 0.3443

### Tableau 5. Score de loss SIMULE sous un gold SUPPOSE (1 - d/d_orig ; exact / copies) - hypothese, pas mesure
| modele | A=0,75 | A=0,85 | A=0,92 |
|---|---|---|---|
| orig | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| S0_tau1.5 | 0.00 / 0.02 | 0.00 / 0.01 | 0.00 / 0.01 |
| S0_tau2 | 0.01 / 0.07 | 0.01 / 0.04 | 0.01 / 0.04 |
| S0_tau3 | 0.02 / 0.12 | 0.03 / 0.08 | 0.04 / 0.08 |
| S1_A75 | 0.66 / 0.35 | 0.86 / 0.32 | 0.80 / 0.18 |
| S1_A75_l30 | 0.48 / 0.28 | 0.63 / 0.25 | 0.80 / 0.23 |
| S1_A85 | 0.47 / 0.33 | 0.62 / 0.29 | 0.86 / 0.29 |
| S1_A85_bc | 0.48 / 0.33 | 0.64 / 0.30 | 0.88 / 0.30 |
| S1_A85_l30 | 0.32 / 0.23 | 0.42 / 0.19 | 0.58 / 0.19 |
| S1_A92 | 0.32 / 0.30 | 0.42 / 0.27 | 0.58 / 0.27 |
| S1_A92_l30 | 0.18 / 0.19 | 0.24 / 0.15 | 0.33 / 0.15 |
| S1_r2_A85_l10_lr1e-4 | 0.59 / 0.47 | 0.78 / 0.48 | 0.93 / 0.39 |
| S1_r2_A85_l30_lr1e-4 | 0.49 / 0.37 | 0.64 / 0.35 | 0.88 / 0.32 |
| S1_r2_A85_l30_lr5e-5 | 0.47 / 0.33 | 0.62 / 0.30 | 0.85 / 0.29 |
| S1_r2_A92_l30_lr1e-4 | 0.32 / 0.31 | 0.42 / 0.30 | 0.58 / 0.30 |
(gold suppose : A=0.75 -> loss moy 0.84, ecart-type 1.42, A=0.85 -> loss moy 0.55, ecart-type 1.17, A=0.92 -> loss moy 0.35, ecart-type 0.90)

## Mesuré, supposé, non mesuré
- **Mesuré** : tous les tableaux (jobs 15901619 et 15901778).
- **Supposé** : le gold. Le tableau 5 suppose son (μ, σ) par le modèle de mélange ; il ne dit rien du score de prédiction (KL image par image vers le gold), qu'aucune mesure locale ne donne.
- **Non mesuré** : le retain réel. ImageNet-100 et CIFAR-100 agrandi sont une règle hors domaine, pas un contrôle in-domain : le vrai contrôle peut être plus stable ou plus touché.
- **Constat** : S1 ne tient pas ses cibles. Pour A=0,85 on visait loss 0,55 / écart-type 1,17 ; on obtient 0,22–0,38 / 0,85–0,96 sur les images exactes. Le tour 2 (plus de pas, lr 1e-4, λ=10) monte à 0,38 mais la dérive hors domaine explose (KL CIFAR 1,14). Le compromis oubli / dérive est net.
- La mesure d'instabilité est quasi indépendante des deux autres (corrélation de rang −0,11 et −0,04, contre 0,45 entre kNN et régression) ; `_bc` donne des résultats presque identiques, donc elle n'apporte rien de visible.
- **Copies** : orig se trompe déjà sur 48 % des copies « couleur forte » (loss 3,10) et 15 % des copies floues. S1 y bouge peu (3,0 à 2,4) : les cibles ne les couvrent pas.

## Recommandation
1. **Premier fichier à soumettre, sûr : `S0_tau2.pt`.** Argmax d'orig gardé sur 100 % des images exactes, mêmes taux d'erreur qu'orig sur copies JPEG et rotation, facteur 1/(1+KL) de 0,97 sur les mélanges et sur ImageNet-100. Il oublie très peu (loss moyenne 0,0024) : son forget sera faible. Il valide le format, la porte à 70 % et le pipeline sans risque. τ=2 plutôt que 1,5 (effet quasi nul, loss 0,0008) ou 3 (accord argmax ImageNet-100 0,84).
2. **Deuxième candidat, à soumettre ensuite : `S1_A85_l30.pt`** (tour 1). Il oublie nettement (loss 0,087, 1,9 % de mauvaises classes, score de loss simulé 0,42 sur les exactes pour A=0,85) avec une dérive limitée (KL(orig‖sub) 0,04 sur mélanges, 0,06 sur ImageNet-100, 0,23 sur CIFAR-100). `S1_r2_A85_l10_lr1e-4` oublie le plus (score simulé 0,78) mais dérive trop hors domaine pour passer en premier.
3. Ne pas lire le tableau 5 comme un score : c'est une règle posée sur une hypothèse.

## Questions ouvertes (défaut entre parenthèses)
1. Sondes sur le classement public : a. S1_A85_l30, puis A75 ou A92 selon l'écart (défaut) ; b. aucune.
2. Tour 3 sans nouvelle donnée : moyenne de graines et interpolation orig + α·(S1 − orig), par exemple α=0,5, pour réduire la dérive (défaut : oui).
3. Demander aux superviseurs si des images générées du même style sont permises comme ancre. L'énoncé dit « original model and forget set alone » (défaut : non).
