# T1 · critique adverse de la chaîne « réel », tour 2 (toile agrandie, `reel3.py`, `DecNet`)

Sam. 21/11, soir. Lu : rapport `t1-t1-reel.md`, critique tour 1, `reel3.py` (396 l.), diff `reel2.py`→`reel3.py`, `predict.py`, `post.py`, `pad.py`,
`padr.py`, `frame.py`, `gap.py`, `gap2.py`, `gap3.py`, `h5.sh`, `job2.sbatch`, `repack.py`, kit `watermarks.py`, `utils.py`, et le code installé de
`trustmark` 0.8.0 (`encode`, `get_the_image_for_processing`, `put_the_image_after_processing`). Aucun job, aucune connexion. Seul calcul local :
`default_rng().integers(0, 0, 0)` dans le venv (numpy 2.5.5) pour vérifier le mode `fold = -1`.
Décision débloquée : garder la recette `embed_pad` telle quelle pour la soumission, ou la recaler d'abord (et comment).
Marques : [LU DANS LE CODE] ; [DÉDUIT] = raisonnement sur code ou chiffres ; [À VÉRIFIER] = hypothèse, avec le test qui tranche.
Mots : **cadre** = rectangle sur lequel l'encodeur a calculé son motif ; **décalage d'échelle** = écart de taille entre le cadre et la zone que le
décodeur redimensionne ; **rééchantillonnage** = redimensionnement d'une image déjà faite, qui laisse une trace périodique dans les pixels ;
**p\*** = marge de toile qui maximise la marge du décodeur pour UNE image.

## 1. Verdict sur la découverte
Le constat est solide : le motif réel est **désaligné en échelle** par rapport au cadre 512 [DÉDUIT des chiffres §1]. Mais « toile plus grande en
miroir, marge fixe centrée » n'est qu'une des façons de produire ce désalignement, et les tests faits ne la séparent pas des autres.

**1.1 Le signe n'est pas testé** [LU DANS LE CODE]. `pad.py`, `padr.py`, `frame.py` n'essaient que des toiles PLUS GRANDES (p ≥ 0, S ≥ 512). Or un décalage
d'échelle de +4,7 % ou de −4,7 % dégrade le décodeur et le réseau de façon presque symétrique (au premier ordre, seule la valeur absolue du décalage
compte) [DÉDUIT]. `gap3.py` teste le zoom dans les deux sens, mais sa sortie n'est pas citée dans le rapport [À VÉRIFIER].

**1.2 Une seule marge ne colle pas aux quatre chiffres StegaStamp** [DÉDUIT du tableau §1]. Bits justes 0,959 et marge 5,87 désignent ~15-16 px ;
logit réseau au centre 9,12 désigne ~8 px, aux bords 4,76 désigne moins de 8 px (+8 px donne 4,39, plus bas que le réel). Le décodeur dit 16, le réseau
dit 7. TrustMark est plus cohérent (~10-12 px partout). La plage 10-16 de `job2`/`h5.sh` est donc un compromis, pas une mesure.

**1.3 Explications concurrentes qui collent aux mêmes chiffres**, avec le test qui départage (chacune < 5 min de GPU ou du CPU) :

| # | Concurrente | Ce qu'elle prédit de différent | Test | Chiffre attendu |
|---|---|---|---|---|
| H1 | **Marge tirée au hasard par image** (ex. U[0 ; 24]) au lieu de 12 fixe | mélange : bords moins abîmés en moyenne, bits plus bas (les grandes marges cassent le décodage) : colle mieux à 1.2 | pour 40 pleines par filigrane : `np.pad(réel, p, "reflect")` puis décodeur, p de −16 à +24 par pas de 2 (p < 0 = rogner le centre), garder p\* par image | marge fixe : p\* serré (écart-type < 3 px) autour de 12 ; hasard : p\* étalé ; dans les deux cas, bits justes réels au p\* ≥ 0,985 (sinon le cadre n'est pas la bonne variable) |
| H2 | **Cadre plus petit que l'image** (signe inverse) | même dégradation, p\* négatif | le même balayage signé | p\* < 0 → recette fausse dans le sens |
| H3 | **Rognage aléatoire** d'une grande image filigranée (cadre décentré) | les 4 bords n'ont pas la même faiblesse | balayage 2D par image du décalage (oy, ox) dans [0 ; 2p\*] à p\* fixé | centré : (oy, ox) ≈ (p\*, p\*) pour > 80 % des images ; sinon étalé |
| H4 | **Zoom de l'image finale** (composition à 512 puis Resize 536 + CenterCrop 512) | même géométrie pour les pleines ; marge des rectangles **proportionnelle** à leur côté (≈ 2,3 %) ; tout le contenu, propre comme filigrané, porte une trace de rééchantillonnage de rapport 67/64 | (a) p\* des 43 rectangles, régressé sur le petit côté ; (b) CPU : variance de la dérivée seconde le long des lignes, spectre moyen sur 85 pleines (méthode de Gallagher 2005) | H4 : pente ≈ 0,023 px/px et pics à k/67 cycle/px ; marge fixe : pente ≈ 0 et pas de pic |
| H5 | **Toile non carrée** (`embed_full` tire py et px séparément) | le réel peut être isotrope | balayage (py, px) séparés sur 40 pleines | py\* ≈ px\* à ±2 px → tirer une seule marge |

Écartées par calcul [DÉDUIT] : **`align_corners`** (StegaStamp 400→512 : écart d'échelle 400/512 contre 399/511 = 0,06 %, 80 fois trop petit pour
4,7 %) ; **le bord adouci de TrustMark** (`FEATHERING_RESIDUAL = 0.01` [LU dans trustmark.py l.69] = 5 px, déjà présent dans le kit, et StegaStamp,
qui n'en a pas, montre le même effet) ; **une force réduite seule** (baisserait centre et bords dans le même rapport ; rapport centre / bords réel 1,92
contre 0,93 pour le cadre 512 chez StegaStamp). Non exclu : **force réduite + marge ~7 px** pour StegaStamp, qui fitte aussi 1.2 ; H1 le départage (si p\* donne des bits
≥ 0,985, la force est normale).
Point à noter [LU dans trustmark.py l.217-227] : TrustMark accepte `concentrate_wm_region` < 1 (filigrane posé sur le centre seulement, bords propres).
C'est le mécanisme naturel de H2 pour TrustMark ; le défaut vaut 1.0.

**1.4 Pour 12 px contre 4 px** : aucune recette unique à marge fixe ne donne les deux [DÉDUIT]. H4 (proportionnelle) en donne 12 et 1-7 selon la taille ;
`padr.py` préfère « 4 px fixe » par corrélation sur 43 images (TM 0,87, SS 0,63), un écart qui tient dans le bruit pour StegaStamp. Le test H4(a) tranche.
Autre lecture des rectangles, plus grave pour les petites zones : **la zone filigranée déborde la boîte étiquetée de ~4 px** (pièce collée plus grande
que son masque). Test gratuit sur les `oof.npz` existants : pour les rectangles OOF, distance signée entre bord prédit et bord de la vraie boîte.
Biais > +1 case (4 px) vers l'extérieur → éroder la prédiction des rectangles d'une case ; biais ≈ 0 → hypothèse écartée.

**1.5 Artefacts que `embed_pad` peut introduire** [DÉDUIT] : (a) anisotropie (H5), le risque le plus probable ; (b) centrage strict, faux si H3 ;
(c) miroir : le motif de la bande visible près des bords est calculé à partir d'un contenu symétrique, effet faible et local (quelques px, que la
grille 128 voit à peine) ; (d) pour TrustMark, l'adoucissement de 5 px tombe dans la marge rognée : correct si la vraie toile était plus grande, faux si H2.

## 2. Bugs dans `reel3.py` et voisins
**B1 · `DecNet` + symétries : le tronc lit le motif à l'endroit seulement** [DÉDUIT ; vérif 10 s : bits justes TrustMark sur vraies pleines
retournées par `x[:, ::-1]`, attendu ≈ 0,5]. `h5.sh` l.6 lance `p_dec` avec les défauts `--raug 1 --tta 2` (l.390, 392). Les vraies images sont
retournées APRÈS filigrane (l.299-300) : 7 vues sur 8 montrent un motif que le décodeur pré-entraîné ne reconnaît pas, étiqueté « filigrané » ; le
réseau apprend alors à s'appuyer sur autre chose que le filigrane. La TTA (l.212, 215) moyenne une vue lisible et une vue illisible. Les exemples
fabriqués, eux, sont filigranés après symétrie (l.91) : corrects. Correctif : `h5.sh` l.6 ajouter `--raug 0 --tta 1`, et `predict.py --tta 1` pour `p_dec`.

**B2 · l.326-328 · la reprise par `ckpt.pt` rejoue les mêmes lots** [LU DANS LE CODE]. `rng` et la graine torch ne sont pas sauvés : après reprise au pas
1500, les pas 1501-2500 revoient les lots 1-1000. Correctif, dans le `if CK.exists():` après `step0 = ck["step"]` :
`rng = np.random.default_rng(a.seed * 100 + a.fold + 7 + 7919 * step0); torch.manual_seed(a.seed * 100 + a.fold + 7919 * step0)` (les fonctions
internes lisent `rng` au moment de l'appel, la réaffectation suffit).

**B3 · `post.py` l.7 et l.59 · inutilisable sur la H200** [LU DANS LE CODE]. Chemin Jülich écrit en dur, et `data_{method}` sans suffixe alors que la H200
n'a que `data_trustmark_p` (`h5.sh` l.4). Correctif l.7-8 : `ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"` ;
l.59 : `next(ROOT.glob(f"data_{method}*/AALL_train.npy"))`. Il ne lit pas non plus les tags « a+b » (pas de dossiers `fold*`) : y reprendre `ld` de `cmd_agg`.

**B4 · l.203 · `--flag` et `--hp` ignorés sans message avec `--arch dec`** [LU DANS LE CODE]. Une variante `dec` + drapeau mesurerait « dec » seul.
Correctif dans `Runner.__init__` : `assert a.arch != "dec" or not (a.flag or a.hp), "dec n'utilise ni drapeau ni passe-haut"`.

**B5 · l.85 · toile non carrée** [LU DANS LE CODE] : `py` et `px` tirés séparément (voir H5). Correctif si H5 confirme l'isotropie :
`p = int(rng.integers(a.padlo, a.padhi + 1)); return embed_pad(img, bits, p, p)`.

Vérifiés justes : `embed_pad` (bornes du miroir, rognage `[py:py+h, px:px+w]`) ; `embed_rect` (marge ≤ côté − 1, rapport d'aspect de la toile ≤ celui de
la boîte, donc jamais de rognage carré de TrustMark) ; correctif B1 du tour 1 en place (l.289) ; `DecNet` : entrée identique à `_prep` du kit
(245, bilinéaire anticrénelé, ×2−1), BatchNorm figée par `train()` (l.150-154) et buffers copiés vers l'EMA (l.341), conv1 en fp32, sortie 128 ;
`OneCycleLR` avec `max_lr` en liste (deux groupes, tronc à lr/10) ; `fold = -1` (`integers(0, 0, 0)` rend `[]`, testé ; banque de formes sans le
filigrane propre) ; `lovasz2` (version symétrique de Berman, juste ; une image sans pixel positif donne la plus grosse erreur, comme l'original) ;
`cmd_agg` « a+b » (même `splits`, donc même ordre OOF ; moyenne des logits avant seuil) ; `predict.py` (mêmes drapeaux, mêmes index de réserve).
Limite de conception, pas bug [DÉDUIT] : le tronc de `DecNet` travaille en 245 px, sa carte la plus fine fait **62 × 62** (245 → conv1 pas 2 → 123 →
maxpool → 62), soit 8,3 px par case. Un rectangle de 0,3 % (28 × 28 px) couvre 3 cases : `DecNet` ne peut pas corriger seul les petites zones.

## 3. Expériences suivantes (gain attendu / coût)
| # | Expérience | Changement précis | Coût | Chiffre qui décide |
|---|---|---|---|---|
| E1 | **Mesurer p\* par image** (débloque tout le reste) | script de 30 lignes : balayage signé H1 + 2D H3 + séparé H5 sur 40 pleines et 43 rectangles par filigrane ; sortir l'histogramme de p\* | 5 min GPU | voir tableau 1.3 ; puis régler `--padlo/--padhi` sur les quantiles 10-90 % de p\* (et `--rfraclo/hi` si H4) |
| E2 | **Recadrer à l'inférence** (hors évident) : au lieu d'imiter le défaut dans l'entraînement, l'annuler au test. Pour chaque image, `np.pad(x, p*, "reflect")` puis redimensionner à 512 : le motif retrouve la géométrie du cadre 512, et les modèles `mixr` anciens (cadre 512) redeviennent adaptés. Prédire, rogner le centre, redimensionner le masque | 20 lignes dans `predict.py`, aucun entraînement | 5 min GPU | réserve TM ≥ +0,03 et SS ≥ +0,02 contre `mixr` brut ; si oui, réentraîner `mixr` avec les vraies images recadrées de la même façon |
| E3 | **`DecNet` à l'endroit** (B1) | `p_dec` avec `--raug 0 --tta 1`, 5 plis, contre `p_dec` actuel | ~1 h H200 | cv TM ≥ +0,02 → défaut pour `dec` ; si `p_dec` ≤ `p_mixr`, garder `dec` seulement dans la moyenne « a+b » si celle-ci gagne ≥ 0,01 sur la réserve |
| E4 | **Vote du message, recadré** : la carte d'accord au message connu (`per_position_logits` TrustMark 8 × 8, `vote = Σ (2m−1)·logit`) a déjà été tentée sur le cadre 512 ; la recalculer sur l'image recadrée par p\* (E2), et en 16 × 16 en décalant l'image de 32 px ; l'ajouter en canal d'entrée du U-Net. Cible : trous des pleines (10 pleines sans trou prédit) | 1 h de code | 5 min GPU pour l'AUC | AUC par case 8 × 8 sur les 85 pleines TM ≥ 0,75 (contre la valeur au cadre 512) → canal ; < 0,65 → abandon |
| E5 | **Forensique du support** (hors évident, CPU) : les pixels propres sont des pixels d'origine. Si les fonds viennent de JPEG non redimensionnés, leurs blocs 8 × 8 gardent des coefficients DCT multiples du pas de quantification ; le filigrane casse ce réseau. Carte à 8 px, indépendante de la force du filigrane | 30 lignes : histogramme du coefficient DCT (0,1) par bloc aligné, dans les trous contre dans la zone filigranée, 85 pleines | 2 min CPU | pics périodiques dans les trous et pas dans le filigrane → canal « bloc compatible JPEG » (gain attendu surtout trous TM) ; aucun pic → abandon immédiat. Le même passage lit les pics k/67 de H4 |

Ordre : E1 et E5 tout de suite (aucune dépendance) ; E2 dès que E1 donne p\* ; E3 en fond sur la H200 ; E4 après E2.
À ne pas faire : régler la marge sur le seul logit du réseau « fabriqué seul » (instrument entraîné sur l'ancien cadre, il dit 7 px là où le décodeur
dit 16) ; conclure entre `p_*` et l'ancienne recette sur un écart < 0,03 de réserve (34 images) ; lancer `p_dec` sur le test avant E3.
