# T1 · critique adverse de la chaîne v3 « réel » (`reel.py`), tour 1

Sam. 21/11, soir. Lu : `reel.py` (338 lignes), `job1.sbatch`, `mkcmds.py`, `cmds1.txt` (75 entraînements), kit `wmloc` (watermarks, data, utils),
BRIEF-T1, critique tour 2, note des faits mesurés. Aucun job lancé, aucune connexion.
Décision débloquée : lancer `job1` tel quel, ou le corriger d'abord (et quoi retirer du plan).
Marques : [LU DANS LE CODE] = vu ligne par ligne ; [DÉDUIT] = raisonnement sur le code ; [À VÉRIFIER] = hypothèse, avec le test qui tranche.
Mots : **TTA** = moyenne des prédictions sur l'image tournée/retournée ; **EMA** = copie du réseau dont les poids sont une moyenne glissante des poids
appris ; **BN** (BatchNorm) = couches qui normalisent avec des moyennes mesurées en cours d'apprentissage ; **OOF** (hors-pli) = prédiction d'un pli par
le modèle qui ne l'a pas vu ; **confusion** = un indice parasite qui accompagne toujours l'étiquette et que le réseau peut apprendre à la place du vrai signal.

## Verdict court
Le code est propre et la plupart des points sensibles sont justes. **Deux bugs changent ce que le job va mesurer** (B1, B2) et un troisième
abîme le canal passe-haut (B3, pas utilisé dans job1). Corriger B1 (1 ligne) et retirer ou corriger `al` avant de lancer. Ajouter `--save 1`.

## 1. Bugs confirmés
**B1 · l.244-246 · le type de fond dépend du type d'exemple** [LU DANS LE CODE]. `bi = np.sort(rng.choice(BSEL, nfull + nobj))`, puis
`full = j < nfull` : les « pleines » prennent toujours les plus petits index, les « objets » les plus grands. BINFO est rangé train hack (544),
test hack (1360), puis COCO (1200, à partir de 61 % des index). [DÉDUIT] Les objets tombent donc à ~80-90 % sur COCO et les pleines presque jamais :
le gain « fonds réels » ne profite pas aux objets, et la comparaison `syn` / `syncoco` est biaisée.
Correctif : après la l.244, `p = torch.from_numpy(rng.permutation(len(bi))).to(dev); base, wf = base[p], wf[p]`.

**B2 · l.104-115 + tag `al` · confusion totale avec l'autre filigrane** [LU DANS LE CODE]. Les « alignés » prennent une vraie image d'une autre
méthode et y posent notre filigrane exactement sur SON masque. Toute zone étiquetée 1 contient les deux filigranes, toute zone 0 n'en contient
aucun. [DÉDUIT] Pour TrustMark (résidu 1,3 niveau), les fonds StegaStamp (6,5 niveaux) offrent un raccourci 5 fois plus visible que le vrai signal ;
au test il n'y a jamais d'autre filigrane, le raccourci tombe. Le tag `al` (10 entraînements, 13 % du job) mesurera surtout ce piège et fera croire
que « les formes alignées n'aident pas ». Correctif : retirer `("al", ...)` du plan de `mkcmds.py` ; version saine plus tard = ajouter une fois sur
deux l'image source intacte `RT[ALI[j]]` avec une étiquette toute à 0 (l'autre filigrane devient non informatif).

**B3 · l.171-172 · le passe-haut est calculé en fp16** [LU DANS LE CODE]. `s.prep(X, fl)` est appelé DANS `torch.autocast(float16)` ; les deux
`F.conv2d` du flou tournent donc en fp16 sur des valeurs 0-255 : pas d'arrondi 0,06 à 0,125 niveau, plus l'arrondi des poids du flou
(~0,1 niveau), contre un résidu PixelSeal de 0,8. Le canal fait exprès pour la précision la perd. Inactif dans job1 (`--hp 0`).
Correctif : `x = s.prep(X, fl)` avant le `with`, puis `o = net(x)` dedans.

**B4 · l.138-139 + l.168 + l.173 · `fpn`/`upernet` avec `--stem1` plantent** [DÉDUIT]. L'entrée dépliée fait 256, FPN sort au quart (64),
`k = 64 // 128 = 0`, la sortie reste en 64x64 et la perte plante contre Y en 128. Inactif dans job1.
Correctif : `o = F.interpolate(o, size=128, mode="bilinear") if o.shape[-1] < 128 else o` avant le pooling.

## 2. Points vérifiés justes (pas besoin d'y revenir)
- Polarité [LU DANS LE CODE] : trou `holes.append(~M)` = zone propre, puis `m = ~s_` = filigrané (l.213, 246) ; objet `m = s_` ; rectangle `m[box]=True`
  avec RECT filigrané dans la boîte ; vraies `MT > 127` ; `nmix` prend image ET masque de b dans la tache ; `nal` étiquette = vrai masque. `paste` met
  `wmk` là où `m` est vrai. Tout est dans le bon sens.
- Symétries [LU DANS LE CODE] : `dih_inv` défait `dih_t` dans l'ordre inverse, `dih_np` suit la même convention sur (H, W, 3). Image et masque
  reçoivent le même `k` (l.255, 261). TTA : `dih_inv(fwd(dih_t(X,k)),k)` sur la sortie 128x128, juste. Le pooling 4x4 commute avec les symétries.
- `to128` (l.238) = `downsample_mask` officiel (moyenne 4x4 ≥ 0,5). Que `masks_128.npz` égale bien `downsample_mask(mask)` : [À VÉRIFIER], 1 ligne.
- Fuites [LU DANS LE CODE] : bases = autres méthodes, réserve exclue partout (`i % 5 != 0`, l.71) ; banque de formes sans réserve ni pli tenu
  (l.211) ; vraies et `nmix` tirées de `tr_i` seulement ; pas de choix d'époque (on garde l'EMA finale) ; seuil d'`agg` réglé sur l'OOF puis
  appliqué à la réserve (l.312-313). Aucune fuite vers le pli tenu ni la réserve.
- Drapeau : `AALL` et `pos` suivent le même ordre (`own_tr` = `idx`), test idem (`own_te`). Seuil 0,75 conforme aux faits mesurés.
- FP32 (l.146-150) : dans smp 0.6, `ResNetEncoder` hérite du ResNet de torchvision, qui enregistre `conv1` en premier ; remplacer `conv1`
  (`--stem1`) garde sa place dans `_modules`, donc c'est bien la première convolution dans les deux cas [DÉDUIT ; À VÉRIFIER :
  `print(type(net.encoder.conv1).__name__)` doit afficher `FP32`]. Nuance : en mode fp16, TF32 reste permis, donc cette « fp32 » calcule en
  TF32 (10 bits, comme fp16 : ~9 pas d'arrondi par niveau). C'est suffisant, contrairement au bf16 (~1 pas) ; seul le tag `fp32` teste le vrai fp32.
- `GradScaler`, OneCycle (exactement `steps` appels, pas de dépassement), `quantile` (262 144 valeurs, sous la limite de 16 M), `pixel_unshuffle`,
  types et formes des lots : pas de plantage trouvé [DÉDUIT]. Le smoke test (`c5.sh`) a dû passer `nmix`, `nal`, `flag`, `hp` : relire sa sortie.
- EMA : les buffers BN sont copiés du réseau vivant (l.287). Écart léger entre statistiques et poids moyennés, acceptable avec 0,995 [DÉDUIT].

## 3. Défauts de méthode
**3.1 Le plan PixelSeal ne répond à aucune question** [LU DANS mkcmds.py]. PixelSeal n'a que `flag` (fp16 + drapeau) et `fp32` (sans drapeau) :
deux variables à la fois, pas de référence `mixr`. Ajouter `("mixr", ["pixelseal"])` juste après `mixr`, à la place de `al`.

**3.2 Orientation du filigrane et TTA** [DÉDUIT, À VÉRIFIER par E1]. Les exemples fabriqués (75 % du lot) gardent le filigrane dans son vrai sens
(la symétrie est appliquée AVANT `embed`, l.85), mais les vraies images sont tournées après coup (`raug`) : 7 vues réelles sur 8 montrent un motif qui
n'existe jamais au test. La TTA retournement (`tta=2`) donne ensuite au réseau un motif inversé. Si le détecteur dépend du sens (probable pour
StegaStamp, dont le résidu a une structure apprise), la TTA moyenne une bonne et une mauvaise prédiction.

**3.3 Mémorisation des 109 vraies images** [DÉDUIT]. 6 par lot × 2500 pas = ~138 vues chacune (~17 par orientation). Le log ne mesure pas le score
sur des vraies images D'ENTRAÎNEMENT : impossible de voir l'écart appris / OOF. Ajouter dans `evaluate` le mIoU sur 27 images de `tr_i`.
Seuil d'alerte : écart > 0,10.

**3.4 Ce que le fabriqué ne reproduit pas** [À VÉRIFIER par E5]. (a) Objets : la forme vient d'une autre image et ne suit aucun contour visible ;
dans le réel, le masque suit un objet greffé, peut-être avec une couture visible. (b) Rectangles : en outpainting, l'extérieur est généré par un modèle
de diffusion (autre texture, autre bruit) ; ici l'extérieur est le fond d'origine, continu. Seules les 6 vraies par lot enseignent ces indices.
(c) StegaStamp réel plus faible que le kit (bits 0,964 contre 0,993) : nos exemples sont trop faciles, le réseau apprend un signal trop fort.

**3.5 Filigrane des autres méthodes dans les fonds** [DÉDUIT]. Hors `al`, il est indépendant de l'étiquette (présent dans les zones 0 et 1) : pas
de raccourci, juste de la capacité dépensée à l'ignorer. Le risque est concentré dans B2 et dans B1 (qui décide quel fond va avec quel cas).

**3.6 Lois de tailles et 50/25/25** : conformes au réel (moitié pleines, partielles moitié rectangles moitié objets ; rectangles 0,3-45 %
log-uniforme, médiane 3,7 % ; trous 0,4-50 %, médiane 4,5 %) [LU DANS LE CODE]. Les taches `blob` peuvent faire plusieurs morceaux, le réel
une seule tache : défaut mineur.

**3.7 Seuil d'`agg`** [DÉDUIT]. Réglé sur des modèles seuls (OOF), appliqué à la moyenne de 5 modèles (réserve, test), dont les logits sont plus
tassés. Sans effet si le seuil reste près de 0 ; sinon regarder `reserve` contre `reserve_au_seuil_cv`.

## 4. Risques du job (coût d'une erreur)
- **Mémoire GPU** [À VÉRIFIER] : `PER=3` met 3 entraînements par A100 40 Go. En fp16 ça passe probablement (~6-10 Go chacun) ; les 10 lignes `fp32`
  (activations doublées) peuvent tomber à 3 sur la même carte et manquer de mémoire. Lire le `nvidia-smi` du smoke ; sinon lancer `fp32` dans un autre job.
- **Durée** [À VÉRIFIER] : 75 × 2500 pas en 2 h 15, génération comprise. Si le smoke donne plus de ~0,1 s par pas seul, `syncoco` et `fp32` ne
  finiront pas, et `agg` (dernière ligne du sbatch) ne tournera pas : la relancer à la main sur le nœud de login, elle ne demande pas de GPU.
- **`--save 0` par défaut** : aucun poids gardé, donc aucun diagnostic possible (TTA, seuils) sans réentraîner. Ajouter `--save 1` (~100 Mo par pli).

## 5. Expériences suivantes (par gain attendu / coût)
| # | Expérience | Changement précis | Coût | Chiffre qui décide |
|---|---|---|---|---|
| E1 | **Le réseau dépend-il du sens du filigrane ?** | job1 avec `--save 1`, puis prédire l'OOF de `mixr` avec chaque `k` seul (`predict(..., tta=1)` sur `dih_t(X,k)` puis `dih_inv`) | 3 min GPU, aucun entraînement | mIoU(k≠0) < mIoU(k=0) − 0,03 → passer `--tta 1` et lancer E2 |
| E2 | **Vraies images sans symétrie** | `--raug 0 --tta 1` sur TM et SS, 5 plis (10 lignes) contre `mixr` | ~15 % d'un job | cv ≥ +0,02 et réserve pas en baisse → nouveau défaut |
| E3 | **Couture visible ou pas** (hors évident, sans GPU) | énergie du gradient sur une bande de 2 px au bord du vrai masque, divisée par celle à 8 px à l'intérieur, réel contre fabriqué (objets et rectangles), 170 images | 2 min CPU | rapport réel > 1,5 × fabriqué → les objets fabriqués doivent venir d'une AUTRE image filigranée (greffe vraie), et les rectangles d'un fond différent |
| E4 | **StegaStamp à la bonne force** | dans `gen` : `WFULL = base + α·(embed − base)`, α tiré dans [0,6 ; 1] ; régler α pour que les bits justes des pleines fabriquées tombent à 0,964 ± 0,01 | 10 min + 5 plis SS | cv SS ≥ +0,02 → garder ; sinon la faiblesse vient d'ailleurs (recompression ?) |
| E5 | **Donner le message au réseau** (hors évident) | canal d'entrée = gabarit flottant `r = embed_float(x, message) − x` calculé sur l'image elle-même (connu au test), sans arrondi uint8 ; en `gen`, garder les bits tirés pour reproduire ce canal sur le fabriqué ; symétries appliquées à l'image ET au canal | 2 h de code, 1 job | d'abord : AUC par pixel de `corr_locale(passe-haut(x), r)` seule sur les pleines PixelSeal ≥ 0,70 ; puis cv PS ≥ +0,05 → bras PixelSeal par défaut |

À ne pas faire : lire les chiffres de `al` comme une mesure des formes alignées (B2) ; comparer `syn` et `syncoco` avant B1 ; conclure sur un écart
< 0,03 sur la réserve (34 images).
