# Critique T1 · t1-residus, tour 1 (rectangles TrustMark calés par R3)
Relecteur adverse, sam. 21/11. Décision débloquée : garder ou non `boxr3` dans l'assemblage, et quel gain annoncer.
Script de vérification (numpy seul, `miou_image` officiel du kit) : `C:\Users\Adem\AppData\Local\Temp\claude\71957d5572689cac\23588a15-2ceb-4765-8cd3-b181626c834d\scratchpad\v.py` et `v2.py` (temporaires, non commités).

## Verdict court
Le chiffre 0,6223 → 0,7246 est exact et ne vient pas d'un réglage sur la réserve. Mais la réserve est une réserve « facile » :
ses 11 rectangles ont tous été calés par R3 (11/11), contre 20/30 hors réserve et 50/77 au test. Gain attendu au test ≈ +0,053 sur
TrustMark (≈ +0,018 sur la moyenne des trois), pas +0,10 / +0,03.

## 1. Fuite ou réglage caché sur la réserve
- [VÉRIFIÉ par calcul] Seuil z : sans effet sur la réserve. Ses 11 rectangles ont z ≥ 9,4 ; réserve = 0,7246 pour z > 6,5, 7 ou 8.
  Conséquence inverse : la réserve ne valide pas du tout le seuil z (aucune image de réserve dans la zone 6 à 9).
- [VÉRIFIÉ par calcul] Seuil R3 : réserve plate (0,7246 pour R3 < 0,7, 0,8, 0,9 ou sans seuil) ; hors réserve maximum à 0,7 (0,7066). Choix fait hors réserve, cohérent.
- [VÉRIFIÉ par calcul] Rétrécissement : réserve 0,7245 (3 px) / 0,7246 (3,5) / 0,7237 (4) ; hors réserve 0,7066 à 3 et 3,5. Écart 0,001 : pas de fuite utile.
- [LU dans le code] `refine.py` calcule le décalage médian sur `ok & ~res_` (hors réserve) : propre.
- [LU + VÉRIFIÉ] **Fuite sur le routeur, au train seulement** : les boîtes train ont été cherchées avec `--all` (`bs.py` l. 151 : `or (a.all and truth_partial)`),
  donc le tri pleine/partielle utilise le vrai masque. Preuve : 86 images avec boîte = 85 partielles vraies + 1 pleine (sans `--all` : 83).
  Trois partielles que le vrai routeur classerait « pleines » (gidx 63, 243, 474, bits justes 0,75 à 0,85) ont une boîte. Aucune n'est en réserve,
  et la seule posée (474, un objet, z 7,6) perd 0,60. Le chiffre de réserve n'est donc pas faussé, mais le train ne mesure pas le vrai routeur.
- [HYPOTHÈSE] Base `rangP-reel-mixr` « hors-pli » au train : non vérifiable ici. Autre base `rang-reel-mixr` : réserve 0,6119 (même ordre) ; le gain ne dépend pas du choix.

## 2. Conventions de `fuse.py` : score recalculé
- [VÉRIFIÉ par calcul] Recalcul indépendant avec `miou_image` officiel et `masks_128.npz` (identique à `downsample_mask` du PNG sur 40 images) :
  réserve n=39, avant **0,6223**, après **0,7246**, écart +0,1023 ; hors réserve n=131, 0,6533 → 0,7066. Mon masque reconstruit = fichier livré (identique).
- [LU dans le code] y/x : `box`/`frame` = (y0, x0, y1, x1), `roi_align` reçoit (x0, y0, x1, y1), `aligned=True` = bords de pixels, tranche `[y0:y1)` : cohérent.
  Règle des 50 % : `mean >= 0.5`, comme le kit. Index de réserve = rang dans `metadata.json` complet (pas dans la liste TrustMark) : correct.
- [LU dans le code] Défaut mineur : `round(f + 3.5)` arrondit au pair, donc le rétrécissement vaut 3 ou 4 px selon la parité du bord ; la doc dit « 4 px ».
  Effet mesuré nul (arrondi plancher 0,7235, intérieur 0,7237).

## 3. Risques au test que le train ne montre pas
- [VÉRIFIÉ par calcul] **Mélange calé / non calé** : gain moyen par rectangle calé R3 +0,34 (réserve +0,36, hors +0,33) ; boîte du décodeur seule +0,036
  (10 images hors réserve, 5 dégradées de 0,07 à 0,11 car la base était déjà à 0,9). Réserve : 0 non calé. Test : 50 calés, 27 non calés sur 340 images
  → projection (50×0,342 + 27×0,036)/340 = **+0,053**, exactement l'écart hors réserve. Le +0,10 de réserve est un tirage favorable.
- [VÉRIFIÉ par calcul] Les « deux paquets » de z se touchent : objets à 6,2 / 6,4 / 6,4 / 7,6, pleine mal triée à 6,41, rectangles à 6,9 / 7,1.
  Hors réserve, z > 6,0 coûte −0,017 (0,6901). Au test, 10 boîtes entre 5,5 et 8,5 dont 6 posées (7,1 ; 7,4 ; 7,9 ; 8,2 ; 8,3 ; 8,3).
- [VÉRIFIÉ par calcul] Pleines mal triées : au test, 2 boîtes posées couvrent plus de 60 % de l'image (z 7,1 et 7,4, bits justes 0,65 et 0,70, juste
  sous le seuil 0,75 du routeur) ; au train, 1 seule. Si ce sont des pleines, chacune perd ~0,5 : risque ≈ −0,003 sur TrustMark, faible.
- [HYPOTHÈSE] Rectangle posé sur un objet : 1 cas au train (474, −0,60) sur 42 objets ; au test ≈ 2 attendus, soit ≈ −0,004.

## 4. Le gain est-il significatif
- [VÉRIFIÉ par calcul] Écarts appariés en réserve (11 non nuls) : 0,009 ; 0,046 ; 0,156 ; 0,169 ; 0,435 à 0,689. Bootstrap IC 95 % de l'écart
  [+0,044 ; +0,169], test par inversion de signe p ≈ 0,001. Hors réserve +0,053 ± 0,014. Le sens est sûr ; l'amplitude à annoncer est celle du hors réserve.

## 5. Bugs dans `bs.py` / `refine.py`
- [LU dans le code] Pas de bug qui fausse les boîtes. Écarts doc/code : grille par défaut ×1,22 et pas = côté/2,5 (le rapport dit ×1,3 et moitié).
- [LU dans le code] `refine.py` ne s'écarte que de ±13 px par bord (pas 3 rayon 9, puis 2×±2) : les boîtes du décodeur fausses de 20 à 60 px ne peuvent pas
  être calées, d'où les 27 non calés du test. C'est la vraie limite, et `refine2.py` (±25 % du côté) la vise.
- [LU dans le code] R3 = vrai message / moyenne de K=2 messages au hasard : dénominateur bruité près du seuil 0,7. `rng` fixe : reproductible.

## Expériences suivantes (gain attendu sur coût)
1. Lancer `refine2.py` sur les 27 non calés du test et les 10 du train ; mesurer hors réserve. Si la moitié se cale : ≈ +0,013 sur TrustMark test, ~1 h GPU.
2. Règle de repli pour R3 ≥ 0,7 (numpy, gratuit) : boîte du décodeur, masque du réseau, ou intersection des deux ; choisir hors réserve.
3. Refaire les boîtes train sans `--all` (ou retirer celles d'images à bits justes ≥ 0,75) et republier les chiffres avec le vrai routeur : 5 min.
4. Zone z 6 à 8,5 : exiger R3 < 0,7 en plus du z (un objet ou une pleine ne se cale pas) ; mesurer hors réserve ; touche 6 images du test.
5. Annoncer +0,053 (hors réserve) et non +0,10 dans `CONTEXTE.md` et l'assemblage, pour ne pas sur-pondérer ce lot face aux autres.
