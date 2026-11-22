# T1 · critique adverse de la chaîne « réel », tour 3 : pourquoi l'IoU du trou des pleines plafonne

Sam. 21/11, nuit. Décision débloquée : où mettre le prochain effort sur le trou des pleines TrustMark (architecture, perte, données, ou signal).
Lu : `t1-t1-reel.md` (§1 à 3 ter), critique tour 2, `reel6.py` (519 l.), `holes.py`, `sam/` (clés), kit `utils.py` et `watermarks.py`.
Aucun job, aucune connexion. Trois calculs CPU locaux sur des fichiers existants (`preds/f_s1/pred_train_trustmark.npz`, masques 128,
images d'entraînement, `sam/*.npz`) ; scripts dans le dossier temporaire de la session (`bord.py`, `texture.py`, `samsnap.py`).
Marques : [LU DANS LE CODE] ; [MESURÉ] = calcul fait ici, sortie collée ; [DÉDUIT] = raisonnement ; [À VÉRIFIER] = hypothèse + test.
Mots : **bande de bord** = cases à moins de b cases du vrai bord du trou ; **RSB** = rapport signal sur bruit, ici force du filigrane
contre détail de l'image ; **AdaBN** = recalculer les statistiques de normalisation par lot sur des lots frais, poids figés ;
**TTA** = moyenne des prédictions sur des versions retournées de l'image.

## 0. Le fait qui change le diagnostic : les erreurs sont des régions, pas des bords [MESURÉ]
`bord.py` sur `f_s1`, pleines TrustMark. Reproduit d'abord le chiffre connu (hors-pli 0,485, réserve 0,402), puis :
```
hors-pli 67 pleines, IoU du trou 0.485
  erreurs à <= 2 cases du bord : 0.33 | <= 4 : 0.52 | <= 6 : 0.65
  IoU du trou en ignorant une bande de 1 / 2 / 3 / 4 cases : 0.510 / 0.520 / 0.512 / 0.522
  trou prédit dilaté de 1 case : 0.468 ; érodé : 0.446
  aire prédite / vraie du trou (médiane) : 0.90 ; cases faux trou 21666 contre trou manqué 28496
  part des cases faux trou dans des composantes qui ne touchent pas le vrai trou : 0.15
  AUC par image, cases à ]0,2] / ]2,4] / >4 cases du bord : 0.737 / 0.872 / 0.956
  trou < 2 % : IoU 0.242, aire prédite/vraie 0.33 | 2-8 % : 0.482, 0.87 | 8-20 % : 0.646, 1.01 | > 20 % : 0.603, 0.82
réserve 18 pleines : 0.402 ; bande de 2 ignorée 0.430 ; mêmes proportions
```
Lecture : même si toutes les cases à 2 cases du bord étaient justes, le trou passerait de 0,485 à 0,52 seulement. Près de la moitié des erreurs
sont à plus de 4 cases (16 px) du bord. Les faux trous sont des débordements du vrai trou (85 % y touchent), les trous manqués sont des morceaux
entiers, et les petits trous sont avalés (un tiers de leur aire est prédite). Le réseau se trompe par régions entières de 10 cases ou plus.
`texture.py`, 85 pleines (taux d'erreur par quintile de détail fin de l'image, 0 = le plus lisse) :
```
  quintile 0 : 0.031 | 1 : 0.044 | 2 : 0.052 | 3 : 0.052 | 4 : 0.053 ; cases saturées : 2,5 % des cases, 2 % des erreurs
```
Les erreurs ne viennent pas des zones lisses ni saturées : elles sont plus fréquentes là où l'image a du détail, qui masque le filigrane.

## A. Les 3 causes les plus probables, chacune avec son test (< 10 min de GPU)

**C1 · Le signal ne suffit pas à l'échelle d'une région (RSB local), le réseau comble par le voisinage** [DÉDUIT de §0 et de `detect.py`].
Sur une zone de ~10 × 10 cases, le filigrane (1,3 niveau en moyenne, ±76 % selon l'endroit) n'est pas séparable du détail de l'image. Faute de
preuve, le réseau prolonge la décision de ses voisins, ce qui explique les débordements accrochés au vrai trou et les trous avalés. Le même
plafond sur fabriqué va dans ce sens : la limite vient du signal et ne dépend pas de l'écart entre fabriqué et réel.
Test (évaluation seule, `g_s1` 8000 pas, 3 min) : sur un jeu fabriqué FIXE de 256 pleines (voir B1), multiplier le résidu par α :
`x' = base + α·(x − base)`, α ∈ {0,7 ; 1 ; 1,5 ; 2}, et relever `trou_fab` ; en même temps, ranger les cases fautives par quintile de |x − base|
(force vraie, connue sur fabriqué). Si C1 est la cause : trou à α = 1,5 ≥ 0,55 et à α = 2 ≥ 0,65 (contre ~0,42 à α = 1), et ≥ 35 % des faux trous
dans le quintile le plus faible. Si le trou à α = 2 reste < 0,50, le signal n'est pas le goulot : c'est la règle de décision du réseau (C3).

**C2 · L'instrument : `trou_fab` n'est pas le même jeu d'une variante à l'autre, et il est petit** [LU DANS LE CODE, l. 418-421, 329, 437].
Le jeu fabriqué tenu à l'écart est refait à chaque entraînement par `make_batch` avec les options de LA variante (`--pshift`, `--holesam`,
`--objsam`, `--psmall`, `--pfull`, réserves `--suffix`, toile) et des taches tirées par le générateur CUDA, graine `seed*100+fold`.
`g_c1*` (toile réelle, `pshift 0.15`) et `g_s1` ne sont donc pas notés sur les mêmes trous. Il y a ~72 pleines (144 × `pfull` 0,5) : l'écart-type
du trou moyen vaut ≈ 0,035 [DÉDUIT, écart-type par image ~0,3]. Les écarts « 0,38 ; 0,40 ; 0,39 » et les « sans effet à pas égal » tiennent dans ce bruit,
et ils sont lus en milieu de cycle `OneCycleLR` (taux d'apprentissage encore haut, donc points instables).
Test (5 min) : jeu fixe sauvé une fois, puis réévaluer `g_s1_4k`, `g_s1` 8000, `g_c1*`, `g_s1e`. Si deux de ces modèles diffèrent de plus de
0,04 sur le jeu fixe alors qu'ils étaient « égaux », des pistes closes sont à rouvrir ; si 4k → 8k gagne ≥ 0,04, le plateau n'existe pas (C3).

**C3 · Apprentissage encore incomplet : la règle « dans le doute, filigrané » domine** [DÉDUIT]. 8 pleines par lot ; le trou monte encore
(f_s1a 4000 → f_s1 8000 : CV pleines 0,698 → 0,716) ; les trous manqués (28 496 cases) dépassent les faux trous (21 666), et les petits trous
sont prédits à 33 % de leur aire. Le meilleur seuil par image donne 0,58 : un biais de seuil explique au plus +0,10, le reste est du classement.
Test (2 min) : les deux points de contrôle `g_s1_4k` et `g_s1` sur le jeu fixe, plus `ema` contre `net` contre `ema` + AdaBN (50 lots).
Attendu si C3 : 4k → 8k ≥ +0,04 et aire prédite/vraie des trous < 2 % qui monte (0,33 → ≥ 0,5). Si < +0,015, le plateau est réel et C1 domine.

Les six pistes demandées, une par une :
- (i) Perte [DÉDUIT] : le Dice par image et par classe donne au trou d'une pleine 0,5 du poids de l'image ; par case de trou, son gradient
  est ~50 fois celui de la BCE. Ce n'est pas le goulot, et la mesure de Lovász (pire) le confirme. Écarté.
- (ii) Cible 128 et regroupement [LU] : `to128` reprend exactement la règle officielle `downsample_mask` (≥ 50 % des 16 pixels). La bande de bord
  vaut au plus +0,035 [MESURÉ, §0]. Écarté.
- (iii) Lot, normalisation par lot, EMA [LU l. 464] : l'EMA recopie les statistiques de normalisation du réseau en ligne. L'effet est faible en fin
  de cycle (taux ≈ 0) mais fausse les évaluations intermédiaires. Test dans C3. Cause secondaire.
- (iv) Réseau aveugle au message : le motif moyen du message ne donne aucun pic même à cadre connu (corrélation 0,015, rapport §3). Le message
  n'apporte rien sous forme de gabarit ; il ne peut aider que comme cible apprise (expérience X3). Non prioritaire.
- (v) Décodeur smp et champ de vision [DÉDUIT] : le champ de vision d'un ResNet-34 dépasse 400 px, et le bord n'est pas le goulot (§0).
  Écarté. Voir B5 pour la taille de sortie avec `--s1`.
- (vi) `make_batch` [LU] : bords adoucis de 1-2 px (moins d'une demi-case) : négligeable. Les 30 % de taches aléatoires (`pblob`) sont plus
  irrégulières que les vrais trous, ce qui explique fabriqué 0,41-0,44 contre réel 0,485, sans créer de plafond. `holesam` repose sur une
  hypothèse fausse : sur les 85 pleines PixelSeal, la meilleure union de régions SAM n'atteint que **0,28** d'IoU avec le vrai trou
  (0,16 pour les trous < 2 %) [MESURÉ]. Les vrais trous ne suivent pas les objets SAM. `holesam` et le recalage des trous sur SAM sont clos.

## B. Bugs dans le code ajouté depuis le tour 2
| # | Ligne | Défaut | Correctif |
|---|---|---|---|
| B1 | 418-421 | [LU] jeu fabriqué de contrôle refait par variante (options + graine CUDA) : `trou_fab`/`fab` non comparables | avant l. 418 : `FV = ROOT / f"fabval_{method}.npz"` ; si `FV.exists()` : `z = np.load(FV); SX, SY, SF = z["X"], z["Y"], z["F"]` et sauter la boucle ; sinon la faire avec les défauts et `np.savez(FV, X=SX, Y=SY, F=SF)` (la viser plus grande : `range(512 // a.bs)`) |
| B2 | 437 | [LU] `sfu = SY.mean((1, 2)) > 0.5` : une pleine dont le trou dépasse 50 % (taches jusqu'à 0,5, `holesam` < 0,5) compte comme partielle | renvoyer le vrai type : à l'appel `full = j < nfull`, ajouter `KD.append(full)` (liste vidée avec `Ms`), sauver `KD` dans `SF` sans le basculement de 2 %, puis `sfu = SK` |
| B3 | 393, 413-414 | [LU] `to128` remplit `Ms` en effet de bord : appelée deux fois avec `--echan`, jamais pour `--nal` → `--fullsup` s'arrête sur l'`assert` (rien de silencieux) | l. 393 : remplacer `to128(m)` par `(F.avg_pool2d(m.float()[None, None], 4)[0, 0] >= 0.5)` ; avec `--nal`, interdire : `assert not (a.fullsup and a.nal)` |
| B4 | 391-393 | [LU] `--objsam` + `--echan` : pixels pris sur `base[(j+1)%n]` et `wf[j]` déplacé, mais 4e canal tiré de `ew[j]`/`eb[j]` non déplacés | dans la branche : `jb = (j + 1) % (nfull + nobj)` puis l. 393 `torch.where(to128_sans_effet(m) > 0, moved(ew[j], dy // 4, dx // 4), eb[jb])` (`echan` est clos : priorité basse) |
| B5 | 219 + 455 | [À VÉRIFIER] avec `--s1`, `smp` < 0.5 suréchantillonne par 2 à chaque bloc : sortie 1024², le dernier bloc travaille à 1024 sans saut (calcul ×4 pour rien) et `--fullsup` échoue sur l'`assert` (1024 contre 512). `smp` ≥ 0.5 suit la taille des sauts : sortie 512 | vérifier : `python -c "import segmentation_models_pytorch as s; print(s.__version__)"` sur h200d. Correctif valable dans les deux cas, l. 455 : `lf = R.last_full if R.last_full.shape[-1] == 512 else F.avg_pool2d(R.last_full[:, None], R.last_full.shape[-1] // 512)[:, 0]` |
| B6 | 464 | [LU] EMA avec statistiques de normalisation du réseau en ligne | `for be, bn in zip(ema.buffers(), net.buffers()): be.lerp_(bn, 1 - d) if be.dtype.is_floating_point else be.copy_(bn)` |

Vérifiés justes [LU] : `embed_pad` à 6 arguments (toile carrée `sz` : `b_ = sz − t`, entier après `int`, bornes du miroir) ; `--canvas real`
(20 % près du cadre exact, décalage uniforme dans `[0, sz]`) ; `sam_object` (déballage 2048 octets = 128², même ordre de symétrie que `dih_np`) ;
`moved` (le vide est hors du masque déplacé, donc aucun pixel noir collé) ; `holesam` et `objsam` (bons index `bip[j]`) ; `POOLNAMES`
(identifiant pris après `ascontiguousarray` ; 0 nom commun entre train et test [MESURÉ]) ; `hole` = IoU officiel de la classe 0 (reproduit 0,485) ;
`--wclean` ; `DecNet` PixelSeal (même `preprocess` et même redimensionnement que le kit l. 59 ; carte la plus fine 64², soit 8 px par case :
limite de conception).

## C. Cinq expériences, classées par gain attendu sur le trou des pleines / coût
| # | Expérience | Changement précis dans `reel6.py` | Coût | Chiffre qui décide |
|---|---|---|---|---|
| X1 | **Jeu fixe + élasticité au signal** (tranche C1/C2/C3) | B1, plus sous-commande `fabeval` : charge `fabval`, évalue une liste de poids, et pour α ∈ {0,7 ; 1 ; 1,5 ; 2} remplace `X` par `base + α(X − base)` (sauver `base` dans `fabval`) ; erreurs par quintile de \|X − base\| | 10 min | voir C1-C3. α = 2 ≥ 0,65 → X2/X3 d'abord ; 4k → 8k ≥ +0,04 → X4 d'abord |
| X2 | **Hors évident : deux têtes, force attendue / force observée.** Le réseau prédit (a) la force que l'encodeur AURAIT écrite, partout, y compris dans le trou (cible exacte sur fabriqué : \|WFULL − BASE\| par case), et (b) la force observée. Un trou = observé ≪ attendu. `g_s1e` donnait cette force en entrée, recalculée sur l'image réelle avec un mauvais cadre, et elle ne se transférait pas ; apprise comme cible, elle ne demande plus de re-filigraner le réel | `classes=3` ; `make_batch` renvoie `E` = `avg_pool(|wf−base|.mean(0), 4)` (attendu) et `E·m` (observé) ; perte + `0.5·smooth_l1` sur les canaux 1-2, **fabriqué seul** (masque 0 pour les vraies images) ; la segmentation reste le canal 0 | 2000 pas ~15 min + affinage 5 plis | trou fabriqué fixe ≥ +0,05 contre `g_s1` à pas égal, puis trou hors-pli ≥ 0,515 (+0,03) |
| X3 | **Hors évident : apprendre le résidu avant la segmentation.** Cible dense : le résidu signé `x − base` à 512 (3 canaux), connu exactement sur fabriqué. Chaque pixel porte alors un gradient fort, au lieu d'un seul bit par case (C3, efficacité des données) | `classes=4` ; `R.last_full` gardé pour les canaux 1-3 ; perte `+ λ·mse(o[:,1:4], (x−base)/2)` sur les seuls exemples fabriqués, λ = 1 ; segmentation inchangée | 2000 pas ~15 min | idem X2 ; si X2 et X3 gagnent tous les deux, les combiner |
| X4 | **Pré-entraînement long + TTA 8** (évident, mais C3 le désigne) | reprendre `g_s1` 8000 avec `--init`, 20 000 pas, `--lr 1e-4`, `--evalevery 2000` sur le jeu fixe ; à coût nul avant cela : `predict.py --tta 8` sur `f_s1` | TTA 3 min ; long ~2 h H200 | TTA : trou hors-pli ≥ +0,01 → défaut ; long : continuer tant que le jeu fixe gagne ≥ 0,02 par 4000 pas |
| X5 | **Hors évident : décision par composantes.** Les erreurs étant des régions accrochées au vrai trou, on choisit le seuil par image (plafond mesuré 0,58 contre 0,485) avec un petit classifieur : composantes du trou prédit à 7 seuils (−3 à +3), descripteurs (aire, logit moyen, contraste avec un anneau de 3 cases, aire prédite totale, bits justes du décodeur) | script CPU à côté de `holes.py`, gradient boosting, validation emboîtée sur les 67 pleines hors-pli + fabriqué fixe | 0 GPU, 30 min | trou hors-pli ≥ +0,03 en CV emboîtée et pas de perte sur la réserve |

Clos par les mesures de ce tour, à ne pas lancer : rognages à plus haute résolution, pertes de bord, sorties plus fines (le bord vaut au plus
+0,035) ; recalage sur régions SAM (plafond 0,28) ; distillation d'un modèle qui voit l'image propre (son professeur rendrait la cible elle-même ;
la forme utile de cette idée est X2/X3, qui apprennent ce qu'il voit).
Ordre : X1 d'abord (il dit lequel de X2-X4 vaut le GPU), X4-TTA et X5 en même temps (sans entraînement), puis X2 et X3 en parallèle.
