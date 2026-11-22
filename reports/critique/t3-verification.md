# Track 3 : vérification indépendante (session `t3-verificateur`), passages 1 à 8, 22/11 1 h 10

Décision débloquée : sur quel facteur du score les cinq sessions doivent travailler, et quelles sondes envoyer en premier.
Scripts et sorties : `E:\EAC-lab\t3\t3-verificateur\` et `h200:~/eac/t3/t3-verificateur/` (`vlogits.py`, `vcheck.py`, `whatif.py`, `fit1.py` à `fit3.py`, `fitk.py`, `out/`).
Passage 1 (§1 à §7) : sorties reprises des extractions de `t3-sondes`. Passage 2 (§8) : sorties recalculées par mon propre code à partir des fichiers `.pt`.
Tout chiffre ci-dessous sort d'une commande.

## 1. Verdicts
| Thèse ou outil | Verdict | Preuve |
|---|---|---|
| « Le score de prédiction est proche de 0 parce que 90 % des images restent tranchées ; adoucir toutes les images le débloque » (`t3-analyse-orchestrateur.md` §2-3, E1) | **Réfutée** | `PG_T85` adoucit toutes les images et obtient 0,0241, la valeur que donnait la loi sans effet d'adoucissement (0,021). L'envoi 1180, de loin le plus adouci (probabilité de la classe prédite à température 2 : 0,297), a le même rapport score / (m + s) que 1197, presque intact (0,874). |
| « Le score suit m, pas m + s » (issue #5, 19 h 45) | **Réfutée** (20 h 07) | `TS_tau4` (m 0,401, s 0,229) = 0,0057 ; la loi « ≈ 0,075 × m » prédisait 0,030. Avant cela : score / m variait d'un facteur 4,7 sur 7 points, score × D / (m + s) d'un facteur 2,1. |
| Mon modèle (score de loss = (m + s) / D, D ≈ 8 à 14, prédiction et rétention moyennes) | **Tient sur 6 envois hors ajustement** | ajusté sur 1180, 1197, 1265 ; les 6 autres envois notés tombent dans sa fourchette (§8). Deux prédictions écrites avant résultat : `TS_tau4` 0,004-0,011 (réel 0,0057), `PG_T85_f20_l48c` 0,05-0,09 (réel 0,0735). |
| Formule codée dans `t3-simulateur/score.py` et `t3-sondes/sim.py` | **Conforme** | relue ligne à ligne contre `Unlearning.txt` et l'en-tête de `task_template.py` (§2). |
| Simulateur, monde A (400 images par classe, poids pré-entraînés) | **Non fidèle** | son gold a m + s = 0,88 sur les images à oublier ; avec un tel gold, 1265 (m + s = 1,34) et `NEGce_all_a2` (s = 1,49) dépasseraient le gold et leur score chuterait, or il monte. |
| Simulateur, monde B (130 images par classe, 8 classes entièrement retirées) | **Non validé** | il rejoue bien deux envois (1180 : 0,004 pour 0,0039 réel ; 1197 : 0,015 pour 0,0114), mais un gold sans classe retirée les reproduit aussi (§3), et B suppose l'hypothèse que `Z9_b0` = 0 met en doute. À faire avant de s'y fier : rejouer dans B les 9 envois réels, `Z9_b0` compris. |
| Hypothèse « 9 classes entièrement oubliées » (`t3-sondes`) | **Réfutée** (20 h 40) | `G10_PG_T70` = 0,0208 > 0 : la porte tient quand 10 classes absentes sont échangées, donc le 0 de `Z9_b0` ne vient pas de la porte mais du score de prédiction ou de loss. Mon modèle à gold ordinaire prédisait 0 pour `Z9_b0`, celui à 9 classes retirées 0,11 à 0,26. |
| Colonne « KL(cible, sub) / KL(cible, orig) » de `t3-methodes/dash.py` | **Circulaire** | la cible fabriquée sert d'étalon au candidat entraîné dessus : elle mesure l'ajustement à la cible, pas le score de prédiction. À ne pas utiliser pour classer. |
| Colonnes « score de loss supposé » H1 à H4 (`dash.py`), tableau 5 de `t3-candidats-v1.md` | **Surestimées** | elles posent m_gold + s_gold entre 1,7 et 4,1 ; les scores réels demandent environ 10 (§3) : ces colonnes surestiment le score de loss d'un facteur 2,5 à 6. |

## 2. La formule, relue mot à mot
Notations : L = entropie croisée (perte de classification, température 1, vraie étiquette) ; p = softmax des logits divisés par 2 ; KL(a‖b) = divergence
de Kullback-Leibler, somme sur les classes de a × log(a / b) ; « clip » = troncature entre 0 et 1.
- Score de loss d'un groupe : 1 − clip(d_sub / d_orig), d = |moyenne(L) − m_gold| + |écart-type(L) − s_gold|. m_gold et s_gold sont la moyenne, sur les modèles gold,
  de la moyenne et de l'écart-type par modèle. `orig` a m = 0,0003 et s = 0,002 : tant que m < m_gold et s < s_gold, ce score vaut (m + s) / D avec D = m_gold + s_gold.
  Il est linéaire, m et s y pèsent autant, et il redescend dès que m ou s dépasse la valeur du gold.
- Score de prédiction d'un groupe : 1 − clip(moyenne des KL(p_gold‖p_sub) / moyenne des KL(p_gold‖p_orig)). C'est un **rapport de moyennes** (le gabarit l'écrit
  « mean KL / mean KL »), pas une moyenne de rapports : les images où le gold est loin d'`orig` pèsent le plus. La KL a le gold en premier argument : elle punit
  un sub qui met une probabilité faible là où le gold en met ; elle punit peu un sub trop étalé, sauf sur la classe où le gold est sûr de lui.
- Oubli = (moyenne des deux groupes du score de loss) × (moyenne des deux groupes du score de prédiction). La moyenne sur « images exactes » et « copies modifiées »
  est prise **avant** le produit (gabarit : « each averaged over the two groups first »). Chaque troncature est faite par groupe.
- Rétention = clip(précision_sub / précision_orig) / (1 + moyenne des KL(p_orig‖p_sub)) sur le jeu de contrôle. Ici `orig` est en premier argument.
- La température 2 s'applique aux deux KL, pas à L. Porte : précision de validation < 70 % → 0. On ne connaît pas la précision de validation d'`orig`.
Où la thèse de l'orchestrateur tient : le produit de trois facteurs, le sens des KL, le coût en rétention d'un adoucissement global. Où elle est fausse : §1, ligne 1.

## 3. Calcul : quels golds reproduisent les scores réels
Mesures sur les fichiers (sorties fp16, étiquette = classe prédite par `orig`, sortie de `fit1.py`) ; m, s = moyenne et écart-type de L sur les images exactes :
| Envoi | m | s | classes changées | p(classe prédite) à T=2 | KL(orig‖sub) à T=2 : mélanges / flou fort / couleur forte | score réel | score / (m + s) |
|---|---|---|---|---|---|---|---|
| `orig` | 0,000 | 0,002 | 0 % | 0,985 | 0 | 0 | |
| 1180 (tête × 0,35) | 0,139 | 0,115 | 0 % | 0,297 | 1,132 / 1,325 / 1,191 | 0,0039 | 0,0152 |
| 1197 (`S1_A85_l30`) | 0,087 | 0,637 | 1,9 % | 0,874 | 0,041 / 0,366 / 0,154 | 0,0114 | 0,0157 |
| 1265 (`S1_r2_A85_l10_lr1e-4`) | 0,376 | 0,960 | 9,3 % | 0,540 | 0,079 / 1,507 / 0,588 | 0,0273 | 0,0204 |
Les valeurs m, s et classes changées publiées par les fabricantes pour ces trois fichiers sont retrouvées à l'identique.

Modèle : j'ai codé la formule et fabriqué des golds paramétriques sur les 3 500 images (hypothèse, pas mesure) : précision A, marge de logit κ des images bien classées,
classes voisines reprises du classement d'`orig` plus un bruit propre à chaque modèle, 3 modèles moyennés. La rétention est laissée dans la fourchette des trois
substituts mesurés. Deux lectures comparées (`fit3.py`) :
| Lecture | Golds compatibles à un facteur 1,5 | Meilleur écart | Gold correspondant | Facteurs prédits pour 1265 |
|---|---|---|---|---|
| H_0 : gold ordinaire, sans classe retirée | 1 sur 72 | × 1,45 | précision 0,60, m_gold 3,9, s_gold 6,6, erreurs très confiantes (κ = 13) | loss 0,13 ; prédiction 0,54 ; rétention 0,32 à 0,80 |
| H_Z : les 9 classes à 129-130 images jamais vues par le gold | 23 sur 108 | × 1,08 | précision 0,64, m_gold 4,9, s_gold 7,2 ; 97 % de précision hors de ces classes | loss 0,11 ; prédiction 0,43 ; rétention 0,32 à 0,80 |
Ce que les deux lectures ont en commun, et qui ne dépend donc pas du choix entre elles :
1. **D = m_gold + s_gold vaut environ 10.** Avec un gold « raisonnable » (précision 0,85 à 0,92, D entre 1,7 et 3), mon modèle prédit des scores 6 à 15 fois
   plus hauts que les scores réels, pour les quatre familles de méthodes (`fit2.py`). Deux autres mesures vont dans le même sens : l'ajustement de `t3-sondes`
   (m_gold ≥ 2, s_gold ≥ 3,4) et la régression logistique hors échantillon de `t3-poids` (66,5 % de précision, loss 1,45, écart-type 2,56).
2. **Le facteur qui nous bloque est le score de loss** : 0,02 pour 1180, 0,06 pour 1197, 0,11 à 0,13 pour 1265. Le score de prédiction vaut 0,33 à 0,54 pour ces
   trois fichiers et la rétention 0,3 à 0,9. Passer de 0,03 à 0,15 demande surtout de multiplier m + s par 4 à 5, avec des erreurs confiantes (loss ≈ 10 par
   image fausse) sur environ un tiers des images, sans casser la rétention ni la porte.
Réserve : trois modèles (le mien, celui de `t3-sondes`, le monde B) s'accordent, mais ils partagent la même lecture de la formule ; un accord entre eux n'est pas une preuve.

## 4. Expériences qui départagent
- **X1, sonde de marge de porte** (`t3-sondes` l'a annoncée de son côté pendant la rédaction : `G10_PG_T70`, 20 h 29, 10 classes absentes échangées par
  paires ; je la confirme comme l'envoi le plus utile du moment). Prendre notre meilleur fichier (`PG_T70`, 0,0323) et mettre à zéro les lignes de la tête de
  9 classes **absentes** du forget set (même nombre que `Z9_b0`). Les statistiques d'oubli ne bougent pas ; seule la précision de validation perd environ
  9 points. Score > 0 : la porte tient à −9 points, donc le zéro de `Z9_b0` vient du score de loss ou de prédiction et H_Z est fausse. Score = 0 : la précision
  de validation d'`orig` est sous 0,77, H_Z reste ouverte, et toute méthode qui abîme la précision générale est à proscrire. Deux envois de plus (5 puis 3
  classes, ou 15) encadrent la précision de validation d'`orig` à quelques points : c'est la marge dont dépend tout verdict « go / no-go ».
- **X2, `TS_tau4` : fait, 0,0057 (soumission 1425).** Prédiction enregistrée avant le résultat : 0,004 à 0,011 dans mon modèle (score de loss 0,05 à 0,07 × prédiction
  0,35 à 0,39 × rétention 0,36 à 0,40). La loi « score ≈ 0,075 × m » prédisait 0,030. Conséquence : la famille « température seule » (τ = 6, 8) est à abandonner,
  elle monte m sans s et divise la rétention par deux.
- **X2 bis, `PG_T85_f20_l48c` (annoncé 20 h 18 ; m 1,066, s 2,236).** Prédiction enregistrée avant le résultat : 0,05 à 0,09 si le produit prédiction × rétention
  de la famille `PG` se maintient (score de loss ≈ 0,3). Ici ma loi et la loi « ∝ m » prédisent la même chose : cet envoi ne les départage pas, il teste
  seulement si l'on peut tripler m + s sans perdre les deux autres facteurs ni franchir la porte.
- **X3, écart-type élevé à dérive faible : `NEGce_all_hi_a4`** (m 0,37, s 2,13, dérive mesurée faible). Si D ≈ 10, le score de loss passe de 0,13 (1265) à 0,24 :
  score attendu 0,03 à 0,045. S'il retombe sous 0,02, s_gold est sous 2 et la lecture « D ≈ 10 » est fausse.

## 5. Risque de porte : règle provisoire, tant que X1 n'est pas faite
Référence mesurée : 1265 a passé la porte (score 0,0273 > 0) avec, sur les images à oublier modifiées, une précision de 0,874 sous rotation 15°, 0,885 sous
recadrage, 0,671 sous flou fort (`orig` : 0,994 / 0,995 / 0,849).
- **Go** : tout candidat dont ces trois précisions sont au moins égales à celles de 1265 (c'est le cas de `NEGce_all_hi_a4`, d'après les chiffres de `t3-poids`, à recalculer par moi).
- **Sonde seulement** (un score nul ne coûte qu'un créneau) : `NEGce_all_top_a8`, `PG85_NEGhi4_all`, tout fichier qui change des classes entières.
- Ce sont des mesures sur des images dérivées du forget set, pas sur la validation : la règle est relative à un fichier déjà noté, elle ne mesure pas la marge.

## 6. Regard neuf : ce que font probablement les équipes à 0,13-0,15
Avec D ≈ 10, un score de 0,15 demande un score de loss de 0,5 à 0,7, donc m entre 2 et 3 : ces équipes font très probablement échouer, avec une confiance élevée,
30 à 40 % des images à oublier (montée de gradient, étiquettes aléatoires, NPO, SalUn : les références de l'énoncé), là où nous en changeons 9 à 13 %.
Pistes, avec le test court de chacune (la 1 est désormais suivie par `t3-sondes`, les 2 et 3 par personne) :
1. X1 ci-dessus (un envoi).
2. Cibles tirées des prédictions hors échantillon : la régression logistique à 5 plis sur les caractéristiques du bloc 8 dit quelles images un modèle qui ne les a
   pas vues classe mal (33,5 %) et vers quelle classe. Test : un candidat `PG` dont les images basculées sont exactement ces erreurs, loss ≈ 10, comparé à un
   candidat de même m et s aux bascules choisies par instabilité. Écart de score = valeur du choix des images et des classes fausses.
3. Sondes par groupe de classes : basculer toutes les images d'un groupe de classes voisines (les 5 bateaux, 29 à 33) et lire la variation du score. Hausse = le gold
   se trompe beaucoup sur ce groupe. Six envois couvrent les 45 classes par groupes.

## 7. Relecture du code des fabricantes (passage 1)
- `t3-methodes/train.py` : couches entraînées = k derniers blocs + norme finale + tête, coefficients LayerScale (`gamma`) gelés ; prétraitement identique au gabarit
  (assertion dans `cache.py`) ; entraînement en bfloat16, évaluation en fp16. Rien à corriger. Le réglage `gold_moments(A)` vise D ≤ 4 : trop bas d'après §3.
- `t3-poids/tv.py` : tous les paramètres sont entraînés, donc le candidat modifie aussi `gamma`, `pos_embed`, `cls_token` et `patch_embed`. Autorisé par
  l'énoncé ; à contrôler sur fichier (passage 2).
- `t3-sondes/sim.py`, `fit.py` : formule conforme ; la rétention y repose sur des images dérivées du forget set, où `orig` est plus sûr de lui que sur de vraies
  images jamais vues : elle est probablement sous-estimée pour les adoucissements globaux.
- `t3-simulateur/score.py` : conforme ; les copies modifiées y sont un mélange de 4 familles légères, hypothèse non vérifiable.
- Aucun substitut du jeu de contrôle n'est fait d'images jamais vues du domaine : toute rétention annoncée localement est une borne, pas une mesure.

## 8. Passage 2 : fichiers recalculés, modèle testé hors ajustement, verdicts
Contrôle de fichier (mon code, H200, évaluation, fp16, prétraitement du gabarit), 11 fichiers : 1180, 1197, 1265, `PG_T85`, `PG_T70`, `PG_T85_f20_l48c`,
`NEGce_all_a2`, `NEGce_all_hi_a4`, `NEGce_all_top_a8`, `PG85_NEGhi4_all`, `G10_PG_T70`. Tous : chargement strict OK, float32, aucun NaN, 343,6 Mo,
écart fp16 / fp32 ≤ 0,017 sur les logits, 0 classe changée sur 500 images. Les m, s et parts de classes changées publiés sont retrouvés à la 3e décimale.
À savoir : les `NEG*` modifient les coefficients LayerScale ; `NEGce_all_a2` modifie aussi `pos_embed`, `cls_token`, `patch_embed` (autorisé).
`G10_PG_T70` a exactement les statistiques d'oubli de `PG_T70`, comme annoncé.

Modèle ajusté sur 3 envois (gold ordinaire), puis prédiction des autres (`vcheck.py`, 10 meilleurs golds × 4 substituts de rétention) :
| Fichier | m / s (exactes) | Score de loss | Score de prédiction | Rétention (substituts) | Score prédit : min .. médiane .. max | Réel |
|---|---|---|---|---|---|---|
| `NEGce_all_a2` | 0,229 / 1,488 | 0,117 | 0,212 | 0,62-0,89 | 0,011 .. 0,0225 .. 0,033 | 0,0219 |
| `PG_T85` | 0,301 / 0,802 | 0,083 | 0,486 | 0,67-0,84 | 0,024 .. 0,0310 .. 0,047 | 0,0241 |
| `PG_T70` | 0,435 / 0,752 | 0,091 | 0,574 | 0,41-0,78 | 0,021 .. 0,0304 .. 0,052 | 0,0323 |
| `PG_T85_f20_l48c` | 1,066 / 2,236 | 0,246 | 0,404 | 0,56-0,80 | 0,054 .. 0,0680 .. 0,098 | 0,0735 |
| `TS_tau4` | 0,401 / 0,229 | 0,048 | 0,480 | 0,36-0,40 | 0,004 .. 0,0086 .. 0,015 | 0,0057 |
| `Z9_b0` | 3,210 / 4,761 | 0,617 | 0,000 | 0,25-0,44 | 0 .. 0 .. 0,032 | 0 |
| `NEGce_all_hi_a4` | 0,373 / 2,127 | 0,171 | 0,158 | 0,72-0,88 | 0,014 .. 0,026 .. 0,036 | non envoyé |
| `NEGce_all_top_a8` | 0,523 / 2,612 | 0,219 | 0,170 | 0,65-0,81 | 0,017 .. 0,031 .. 0,046 | non envoyé |
| `PG85_NEGhi4_all` | 0,670 / 1,856 | 0,182 | 0,544 | 0,55-0,66 | 0,049 .. 0,061 .. 0,083 | non envoyé |
| `G10_PG_T70` | 0,435 / 0,752 | 0,091 | 0,563 | 0,40-0,73 | 0,020 .. 0,029 .. 0,047 (hors effet des 10 classes échangées, que mes substituts ne voient pas) | en attente |
Gold des 10 meilleurs ajustements : précision 0,5 à 0,7 sur les images à oublier, erreurs très confiantes, m_gold 2,6 à 6,1, s_gold 5,2 à 8,1.
Limite : la rétention reste une fourchette de substituts faits d'images dérivées du forget set ; le modèle ne sait pas quelles images le vrai gold rate.

Verdicts sur les candidats :
- `NEGce_all_hi_a4`, `NEGce_all_top_a8` : porte **go** (précisions ≥ celles de `PG_T70`), mais score prédit sous notre meilleur (0,0735) : pas de créneau candidat.
- `PG85_NEGhi4_all` : porte **go** ; score prédit 0,049 à 0,083, à cheval sur notre meilleur : créneau seulement s'il n'y a rien de mieux en file.
- `PW_W9a/b` (9 classes entières, `t3-methodes`) : à n'envoyer que si `G10_PG_T70` fait 0.

Où est le gain (`whatif.py`, bascules synthétiques ajoutées à `PG_T85_f20_l48c`) : monter à 30 % de bascules avec une loss de 10 à 14 gagne même mal ciblé ;
au-delà, le choix des images et des classes fausses fait un facteur 2 à 3 (confirmé ensuite : classe redistillée + 9 %, images de redistill + 5 %).

Règle de porte mise à jour : `PG_T85_f20_l48c` et `PG_T70` ont passé la porte. **Go** pour tout fichier dont la précision sur mes copies (rotation 10°,
recadrage, flou fort, recadrage fort) reste ≥ 0,80 / 0,81 / 0,47 / 0,73 (`orig` : 0,996 / 1,000 / 0,677 / 0,874) ; en dessous : sonde seulement.

## 8 bis. Marge de porte, mesurée par `G10_PG_T70` (soumission 1493 : 0,0208)
`G10_PG_T70` = `PG_T70` (0,0323) dont les lignes de tête de 10 classes absentes du forget set sont échangées par paires : les images de ces classes
sont toutes mal classées, soit environ 10 points de précision de validation en moins. Le score reste positif : **un fichier peut perdre environ
10 points de précision de validation de plus que `PG_T70` sans franchir la porte**, donc la précision de validation d'`orig` est d'au moins 0,78 environ.
Le rapport 0,0208 / 0,0323 = 0,64 est la rétention perdue (rapport de précision ≈ 0,90, et un facteur 1 / (1 + KL) multiplié par environ 0,72).
La règle de porte devient : **go** pour tout fichier dont la perte de précision sur des images jamais entraînées (réserve de `t3-methodes`) ne dépasse pas
celle de `PG_T70` de plus de 8 points (2 points de sécurité) ; au-delà, sonde. La sonde `G20_PG_T70` (20 classes) dirait si la marge va jusqu'à 20 points.

## 9. Passages 3 à 7 (21 h 10 à 23 h 10) : la montée de 0,07 à 0,30, ce qui a été prédit et ce qui a été raté
Tous les fichiers ci-dessous sont contrôlés conformes par mon code (chargement strict, float32, sans NaN, 343,6 Mo) ; m, s recalculés en fp16.
| Fichier | m / s exactes | Bascules | Prédit avant résultat | Réel |
|---|---|---|---|---|
| `PG_T85_f20_l814c` | 1,82 / 3,87 | 19,9 % | 0,083 (0,038 à 0,174) | 0,1487 |
| `PD_f20_l814` (ancre sur vues très dégradées) | 1,38 / 3,17 | 19,3 % | 0,11 à 0,14 | 0,0916 |
| `f20_l1218c` | 2,31 / 4,97 | 19,9 % | 0,16 à 0,24 | 0,1707 |
| `f30_l814c` | 2,56 / 4,29 | 29,6 % | 0,17 à 0,21 | 0,1962 |
| `f30_l1218c` | 3,25 / 5,50 | 29,8 % | 0,20 à 0,225 (posté à la minute du résultat) | 0,1998 |
| `f35_l814c` | 2,89 / 4,39 | 34,4 % | 0,20 à 0,215 | 0,2179 |
| `f40_l814c` | 3,19 / 4,46 | 38,9 % | non prédit | 0,2400 |
| `f50_l814c` | 3,70 / 4,47 | 48,1 % | 0,16 à 0,23 | 0,2779 |
| `f60_l814c` | 4,22 / 4,43 | 57,2 % | 0,26 à 0,31 | **0,3009** |
| `PGho_f60_l814c` (images non basculées intactes + 2e choix d'`orig`) | 4,92 / 4,86 | 58,1 % | 0,28 à 0,37 | 0,2701 |
| `f70_l814c` | 4,68 / 4,30 | 66,1 % | 0,27 à 0,325 | **0,3159** |
| `f80_l814c` | non recalculé | 73 à 74 % | non prédit | 0,3179 (palier 70 → 80 % : + 0,002, sous le seuil d'arrêt) |
Ce que ces points établissent :
- **Cassure du score de loss à s_gold ≈ 4,6** : passer la loss des bascules de 8-14 à 12-18 rapporte + 0,022 à 20 % puis + 0,004 à 30 %.
- **À loss 8-14, le score vaut 0,075 × m à 2 % près de 30 à 50 %**, puis 0,071 × m à 60 %. La pente par point de part faiblit : 0,0049 → 0,0041 → 0,0025.
  Le gold se trompe sur 40 à 50 % des images à oublier ; les images choisies par instabilité sont en bonne partie les siennes.
- **Mes ratés** : la partie « score de prédiction » de mon modèle paramétrique est trop pessimiste pour des bascules bien choisies (`f20_l814c`, `f50_l814c`
  sous-estimés), et mes prédictions par ingrédient sont fragiles (`PD_f20_l814` et `PGho_f60` surestimés). Depuis le passage 3 je prédis par la loi empirique
  de la série, pas par le modèle paramétrique.
- **`PGho_f60` = 0,2701** malgré m + 17 % : le produit des autres facteurs a perdu 23 %. Causes non séparées : images non basculées laissées tranchées,
  classe fausse = 2e choix d'`orig`, s au-dessus de 4,6. `PGh_f60` seul les départage. Piste inverse proposée à `t3-methodes` : adoucir davantage
  les images non basculées (probabilité de la classe à T=2 vers 0,80 au lieu de 0,92).

Porte et robustesse :
- `G10_f60` = 0,189 > 0 (`f60` avec 10 classes absentes échangées ; fichier vérifié : seule la tête diffère). **`f60_l814c` a au moins 8 à 10 points de marge.**
  La précision de validation d'`orig` est ≥ 0,78 (≥ 0,82 si `f60` perd 5 % en validation).
- Rapport sonde / fichier identique à 20 % (0,64) et à 60 % (0,63) de bascules : la KL du contrôle réel ne monte presque pas avec la part. Mes mesures sur
  les variations générées de `t3-nonmembres` (`vnm.py` : précision IP-Adapter `orig` 0,624, `f40` 0,451, `f50` 0,400, `f60` 0,351, `f70` 0,313) sont
  trop pessimistes pour la rétention ; elles restent utiles pour comparer deux fichiers de même part.
- 10 % d'images de contrôle faussées coûtent environ 29 % du score par le seul terme KL : ne jamais toucher aux 55 classes absentes.
- Sondes suivantes : `G10_f70` = 0,2025 et `G10_f80` = 0,209, toutes deux positives : `f70` et `f80` ont aussi ≥ 8 à 10 points de marge.
- Règle : un palier à la fois, sonde `G10` du palier d'abord, arrêt dès qu'un palier gagne moins de 0,01 (de 60 à 70 % : + 0,015).
  Raison : un envoi ne se retire pas et l'énoncé ne dit pas si le jeu de validation de la porte est coupé entre partie publique et partie cachée.
- Classe fausse redistillée (`t3-redistill`) : `PG_RDt24_60_l915c` (profil de `f60`, m − 4 %) prédit 0,27 à 0,325, réel **0,3273** : à profil égal, la classe redistillée vaut + 9 % sur la classe votée.

## 10. Passages 8 à 10 (22/11, 1 h à 2 h 25) : série redistillée, position du gold, suite
| Fichier (`t3-redistill`) | Prédit avant résultat | Réel |
|---|---|---|
| `RDm3t24_80_l1016c` (sonde `G10` : 0,2421) | 0,367 à 0,384 | 0,3713 |
| `RDt24_70_l1218c` | 0,345 à 0,370, resserré à tort à 0,346-0,365 | 0,3682 |
| `RDt24n_70_l1016c` (images non basculées suivant redistill) | 0,360 à 0,372, effet nul | 0,3663 |
| `RD6m4t24_70_l1117c` (classe à 6 blocs) | 0,350 à 0,385 | 0,3654 : ni mieux ni moins bien |
| `RDt24r_70_l1016c` (images choisies par redistill) | 0,358 à 0,385 | **0,3888** (meilleur) : + 5 % à loss égale |
| `RDt24r_80_l1016c` | 0,380 à 0,402 | 0,3657 : **raté**, 70 % est un maximum pour ce classement |
`fitk.py`, cinq paires notées qui ne diffèrent que par la loss visée : m_gold = 5,0 à 5,2, s_gold = 4,6 à 4,8 (à 1,5 % près). Avec des erreurs de loss 9 ± 3, cela fait 54 à 60 % d'images mal classées par gold ; nos fichiers à 70 % en basculent 66 %. Le score de loss est sur son plateau.
Alerte : `PG_RDt24_85_l1117`, `_85_l1218`, `_90_l1218` ont 47 à 49 % de bascules et m = 1,6 : entraînés avec `lam` 10 et 2 000 pas (ancre forte, vérifié dans leur journal), autre recette, à ne pas envoyer.
Passages 11 et 12 (3 h 25 à 3 h 48) : `SOUPs1` (même recette, graine 1) 0,3942, meilleur, sonde `G10_SOUPs1` 0,2519 (prédit 0,24-0,26) ; parts 65 / 72 / 75 % : 0,3743 / 0,3849 / 0,3785 ; `RV2_70_l8515c` 0,3847 (prédit 0,362-0,395, piste des copies fermée). Prédit, non noté : `PGB6` 0,345-0,380 ; `RDp_70` 0,383-0,400 ; `rc_70_l1117c` 0,375-0,400 ; `TM_H4` 0,32-0,36 ; dix fichiers `TM_*` et bascules graduées : tous sous `SOUPs1` en local (commentaire 7073114300).
Tour de critique : `t3-tour-plateau.md`. Ratés et réussites de 4 h à 4 h 40 : `rc_70_l1117c` 0,3753 (prédit 0,375-0,400, note locale fausse de sens : le gold de substitution ne range pas à l'intérieur d'une famille) ; `TM_wT2_4_a114` 0,3611 (prédit 0,335-0,380) ; `TM_H4a_a090` 0,3953 (prédit 0,375-0,405). **Meilleur : `PG_RDq_70_l915c_s1` 0,4107** (m 5,09 / s 4,62, sur le gold ajusté ; masse fausse de 8 partitions), noté avant mon contrôle. `RDq_70_l1016c_s1` 0,4028 (prédit 0,395-0,415) : durcir la base perd, série d'échelle close. Bande couverte fermée sans envoi. `RDq_70_l915c_s0` 0,4042 (prédit 0,400-0,416 ; bruit de graine 0,0065). `RDqp` 0,4051 et `RDq15_70_l814c_s1` 0,4089 (prédits 0,398-0,416 et 0,398-0,418). Recette `RDq` : 0,4067 ± 0,003. Prédit, non lu : `CPv3_RDq_70_l612` 0,365-0,392. Manque : la réponse d'Adem sur l'envoi qui sert au classement final.
