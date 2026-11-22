# T1 · session t1-assemblage : rapport final

Mis à jour le 22/11 à 10 h 50. Décision débloquée : quel fichier T1 reste en ligne à la fin du hackathon.
Mots employés : **mIoU** = score officiel, moyenne par image de l'IoU de la zone propre et de l'IoU de la zone filigranée (IoU = surface commune
divisée par surface totale couverte). **Réserve** = images de train d'index % 5 == 0 (30 PixelSeal, 39 TrustMark, 33 StegaStamp). **Validation
croisée 5 plis** = toute la procédure d'assemblage refaite sur 4 cinquièmes du train et notée sur le cinquième restant, cinq fois (`cv5.py`).
**Écart apparié** = différence de mIoU image par image entre deux candidats, avec un intervalle de confiance à 95 % (IC95) tiré au hasard.
**Pleines** = images presque entièrement filigranées, avec un petit trou propre ; **partielles** = images où seule une petite zone est filigranée.
**SAM** = Segment Anything (Meta, Apache 2.0), réseau qui découpe une image en objets. **Logit** = sortie brute d'un réseau (> 0 = filigrané).

## 1. Résultat
**Le candidat 38 reste le fichier en ligne : 0,8230 au public** (0,8347 en validation croisée, 0,8159 en réserve).
Meilleur candidat ensuite : le **49**, +0,0048 contre le 38, IC95 [−0,0012 ; +0,0110]. Il ne remplit pas la règle d'envoi d'Adem de 4 h 17
(IC95 qui exclut 0 et feu vert écrit du vérificateur) : fichier prêt, non envoyé, sauf décision contraire d'Adem.
`h200b:~/eac/t1/t1-assemblage/submission_candidat_49.pt` (copie `E:\EAC-lab\t1s\t1-assemblage\h200\`), sha256
`a568a82e9f44c86dce67cfc1e3465a53fddb5e7053434d55d27fe323ff544688`, `validate_t1.py` : VALIDE (format).

## 2. Envois au public (track 1)
| Fichier | Public | Gain public | Gain mesuré avant envoi | Ce qui changeait |
|---|---|---|---|---|
| v6 (orchestrateur) | 0,7059 | | | cartes de t1-reel et t1-pixelseal, seuil par population |
| Candidat 5 | 0,7384 | +0,033 | | vote par région SAM, boîte du décodeur TrustMark |
| Candidat 22 | 0,7695 | +0,031 | +0,028 | cartes TrustMark pleine résolution de t1-reel, cadres calés au pixel |
| Fichier 27 (agent de Yassir) | 0,7935 | +0,024 | +0,011 | sortie intermédiaire (erreur de nommage de ma part, corrigée) |
| Candidat 35 | 0,8033 | +0,010 | +0,025 | cartes `tm_f_x1`, `ss_f_c4+f_s1`, `f_s1+f_deca`, vote refait sur chacune |
| Candidat 36 | 0,8131 | +0,010 | +0,012 | cartes `tm_f_x1+f_c5+f_r50`, `ss_f_x1+f_c4`, `ps_f_x1+f_s1+f_deca` |
| **Candidat 38** | **0,8230** | +0,010 | +0,0085 | cartes `ps_f_x2+f_x1+f_dx`, `tm_f_x2+f_c5+f_r50b` |
En cumul du 5 au 38, la mesure a prédit le public à 0,01 près. Une fois la validation croisée 5 plis en place (à partir du 35), l'écart
mesuré a prévu le public à 0,002 près sur les deux derniers envois. Avant, la réserve seule s'était trompée (candidats 30 et 33 : +0,013
et +0,014 en réserve, −0,0004 et −0,0005 au public).

## 3. Candidats 39 à 55, tous contre le 38 (validation croisée 5 plis)
Le gain TrustMark de la validation croisée est optimiste de 0,06 à 0,075 selon le vérificateur : chaque écart est donné aussi sans TrustMark
et avec TrustMark compté sur la réserve seulement.
| Cand. | Ce qui change par rapport au 38 | Écart (IC95) | Sans TrustMark | TrustMark sur réserve | Réserve |
|---|---|---|---|---|---|
| 39 | PixelSeal `ps_pop_x2+x2b+x1+dx__st+x2` | +0,0027 [−0,0027 ; +0,0082] | idem | idem | −0,0016 |
| 40 | boîtes PixelSeal du décodeur, seuil abaissé | −0,0003 [−0,0030 ; +0,0026] | idem | idem | 0 |
| 41 | 40 + cartes PixelSeal `ps_pop…`, `ps_partielles_st+x2` | +0,0023 [−0,0036 ; +0,0081] | idem | idem | −0,0016 |
| 42 | PixelSeal `ps_pop_C` | +0,0029 [−0,0026 ; +0,0086] | idem | idem | −0,0007 |
| 44 | TrustMark `tm_x1hp+x2+c5+r50b` | +0,0006 [−0,0017 ; +0,0027] | 0 | +0,0009 [−0,0057 ; +0,0078] | +0,0009 |
| 46 | cumul : pleines PS `ps_pleines_7`, partielles PS du 42, TM du 44 | +0,0039 [−0,0020 ; +0,0099] | +0,0034 | +0,0043 | +0,0017 |
| 47 | pleines PixelSeal de `ps_pleines_7` seules | −0,0003 [−0,0017 ; +0,0012] | idem | idem | +0,0013 |
| 48 | TrustMark `tm_pleines_x2+x1hp` dans les sources | +0,0015 [−0,0011 ; +0,0042] | 0 | +0,0039 | +0,0039 |
| **49** | **46 + pleines TrustMark du 48** | **+0,0048 [−0,0012 ; +0,0110]** | +0,0034 [−0,0022 ; +0,0089] | **+0,0073 [−0,0008 ; +0,0165]** | +0,0047 |
| 51 | 49 avec pleines PixelSeal de `ps_pleines_9` | +0,0046 [−0,0015 ; +0,0107] | +0,0031 | +0,0070 | +0,0034 |
| 52 | pleines PixelSeal de `ps_pleines_9` seules | −0,0006 [−0,0019 ; +0,0008] | idem | idem | +0,0001 |
| 53 | TrustMark `tm_x2+c5+r50b+3hp` dans les sources | +0,0000 [−0,0030 ; +0,0030] | 0 | +0,0017 | +0,0017 |
| 54 | 49 avec pleines TrustMark du 53 | +0,0049 [−0,0012 ; +0,0110] | +0,0034 | +0,0073 | +0,0047 |
| 55 | 49 avec tout TrustMark du 53 | +0,0034 [−0,0029 ; +0,0098] | +0,0034 | +0,0050 | +0,0025 |
(43, 45, 50 : intermédiaires de mélange ; leurs chiffres sont dans les commentaires de l'issue #3.)

Lecture : les cartes déposées par t1-reel après le 38 gagnaient seules +0,01 à +0,02, mais 70 à 85 % de ce gain disparaissait après le vote par
région et les boîtes, parce qu'elles amélioraient surtout ce que l'assemblage corrigeait déjà (objets et rectangles des partielles). Le seul gain
qui a traversé l'assemblage vient des modèles TrustMark à canaux passe-haut sur les pleines (`tm_pleines_x2+x1hp` : pleines TrustMark
0,764 → 0,778, réserve TrustMark +0,0118).

## 4. Où le 38 perd encore (`pires.py 38`, masques de validation croisée)
Perte totale 1 − 0,8347 = 0,165 ; en points sur la moyenne des trois filigranes :
| Cause | Images | mIoU moyen | Points |
|---|---|---|---|
| PixelSeal, pleines, trou mal dessiné (IoU du trou < 0,5) | 30 | 0,618 | 0,0225 |
| TrustMark, pleines, trou trouvé mais imprécis | 58 | 0,843 | 0,0178 |
| PixelSeal, pleines, trou trouvé mais imprécis | 48 | 0,834 | 0,0157 |
| PixelSeal, partielles, petit rectangle (< 32 px) manqué | 14 | 0,481 | 0,0142 |
| TrustMark, pleines, trou mal dessiné | 17 | 0,638 | 0,0121 |
| StegaStamp, pleines, trou imprécis | 85 | 0,944 | 0,0094 |
| TrustMark, pleines, trou manqué | 9 | 0,491 | 0,0090 |
Les images pleines portent 57 % de la perte. StegaStamp montre la cible : 0,944 sur ses pleines contre 0,73 (PixelSeal) et 0,76 (TrustMark).
Les petits rectangles PixelSeal manqués sont des bandes très fines (102 × 9 px, 147 × 11 px) où le décodeur ne lit rien (z médian 0,8).

## 5. Ce qui a marché (mesuré)
- **Vote par région appris** (`rank.py`, idée d'Adem) : pour chaque image partielle, un petit réseau note ~67 masques SAM (27 indices : accord
  avec la carte, décodeur, forme) et rend le meilleur si son gain prédit dépasse un seuil. Objets des partielles, hors-pli : PixelSeal
  0,75 → 0,89, TrustMark 0,76 → 0,83, StegaStamp 0,79 → 0,86. Le plafond d'un seul masque SAM sur les objets est 0,93 à 0,97.
- **Boîtes du décodeur** pour les rectangles : on cherche le rognage où le décodeur relit le message (z = écart au hasard en écarts-types),
  puis on cale le cadre au pixel par re-tamponnage (l'encodeur ajoute 2 à 3 fois moins d'énergie sur une image déjà filigranée dans le même
  cadre ; idée de t1-residus). TrustMark : recouvrement 0,866 → 0,958, partielles en validation croisée 0,697 → 0,806.
- **Population sans vérité** : bits justes du décodeur sur l'image entière > 0,75 = pleine ; juste sur 97,6 à 98,8 % des images.
- **Seuil par population sur grille fine** (quantiles + 61 pas réguliers) avec plancher de la classe rare : corrige −0,018 sur les pleines
  StegaStamp que la grille de quantiles seule perdait.
- **Mélange par population prédite** (`mixpop.py`, `composepop.py`) : prendre les pleines d'un assemblage et les partielles d'un autre.
- **Protocole de mesure** (`cv5.py`, `cv5merge.py`, `ecart.py`) : validation croisée de toute la procédure + bootstrap apparié.

## 6. Ce qui a échoué (mesuré, fermé)
| Piste | Résultat |
|---|---|
| Découpes par couleur (SLIC, Felzenszwalb) votées par région | plafond pas meilleur qu'une grille de même finesse |
| SAM sur les pleines et les rectangles | plafond 0,67 et 0,57 |
| Rectangle ajusté à la carte | 0,781 → 0,781 (StegaStamp) |
| Vote v2 (union de masques), v3 (masques jugés un par un), v4 (six cartes, 97 indices) | 0,700 / 0,858 / 0,743 contre 0,712 / 0,910 / 0,747 |
| Boîtes PixelSeal à seuil z abaissé (candidat 40) | −0,0003 : la boîte remplace des objets bien rendus |
| Porte rectangle / objet sur les partielles PixelSeal (`gate.py`) | −0,0010 ; plafond avec la vérité +0,0020 |
| Seuil par image sur les pleines (`fullthr.py`) | plafond avec la vérité +0,005 ; régression prédite −0,002 à −0,003 |
| Dilatation, érosion, enveloppe convexe des trous (`morph.py`) | toutes négatives |
| Calage par re-tamponnage sur PixelSeal et sur les pleines TrustMark | 0,840 → 0,849 ; AUC 0,57 |
| Pixels partagés entre images, VAE du générateur, couture au bord, bibliothèque de formes | aucun signal |
| Plancher de 48 % sur les pleines (candidats 13, 15) | erreur de borne, retirée |
| Correction de décalage des boîtes apprise avec la réserve | fuite de 0,003 mesurée par le vérificateur, retirée |

## 7. Leçons
- **Mesurer contre le fichier en ligne, sur la même échelle.** Un gain en réserve seule ne prédit pas le public ; la validation croisée de
  toute la procédure, si.
- **Une meilleure carte ne fait pas un meilleur assemblage** quand son gain tombe là où l'assemblage corrige déjà. Il faut mesurer après
  assemblage, image par image, et regarder où va la perte restante (`pires.py`).
- **Le gain restant était dans les cartes, sur les trous des images pleines** : aucun post-traitement ne l'atteint, il faut une carte qui
  dessine mieux le trou.
- Nommer `submission_*` seulement les fichiers vérifiés : un intermédiaire mal nommé a été envoyé (le 27 ; il s'est révélé bon par chance).

## 8. Fichiers
Scripts : `E:\EAC-lab\t1s\t1-assemblage\` (copie de travail `h200b:~/eac/t1/t1-assemblage/`) : `asm.py`, `assemble.py`, `compose.py`,
`composepop.py`, `cv5.py`, `cv5merge.py`, `ecart.py`, `mixpop.py`, `rank.py`, `boxnet.py`, `boxref.py`, `r3ss.py`, `mk_cadre.py`, `pires.py`,
`gate.py`, `fullthr.py`, `morph.py`, `rank4.py`, `kgen.sh` (cycle complet pour une nouvelle carte). `LISEZMOI.md` dit quels fichiers sont des
candidats. Tableau de bord : `reports/t1-tableau.md`. Critique : `reports/critique/t1-assemblage-tour1.md`. Détail daté : issue #3.
