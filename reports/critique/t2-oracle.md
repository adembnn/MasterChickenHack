# T2 · oracle des scores publics (session t2-oracle d'Adem, 22/11 vers 5 h)

**Décision débloquée** : quels fichiers T2 envoyer aux prochains créneaux, et ce que le public ne pourra jamais dire.
0 requête à l'API, aucun `.env` lu, aucun envoi, aucun fichier existant modifié. Scripts `reports/critique/t2-oracle-*.py`, CSV et tirages dans `E:\EAC-lab\t2\reel\oracle\`.

## En bref
1. **12 scores, 12 fichiers.** J'en ai retrouvé 8 à l'octet près et reconstitué 4 d'après le fil #4 (tableau ci-dessous). Une erreur est à corriger dans `pubdup.py`, `pubmask.py` et `pubS.py` : 2491 y est codé sans ep-11, alors que le vrai fichier isole 11, 28 et 03.
2. **L'énumération est exacte** (pas un échantillonnage) : **55 749 900 états** (S, nature des guichets publics) redonnent les 12 scores.
3. **Le masque « guichets publics » l'emporte nettement** : 0 tirage compatible sur 120 M sous « paires au hasard » (vraisemblance < 2,5e-8), contre 1,9e-7 sous « guichets » (facteur > 7). Le contrôle par rejet confirme le calcul exact (4 tirages observés pour 5,8 attendus).
4. **Hypothèses, de la plus à la moins vraisemblable** : gabarit b4 3,9e-7 > 3 variantes par famille 2,7e-7 > 4 variantes 2,2e-7 ≈ variante en double 2,2e-7 > 5 variantes 1,1e-7 > 6 variantes 3,5e-8. Aucune n'est tranchée : les écarts ne vont pas au-delà de ×2, sauf pour 6 variantes.
5. **Le candidat précis du brief est exclu** ({12,17,18,24} / {03,22,26,28} / {05,09,11,15}) : aucun S ne redonne les 12 scores, il bute sur 2707.
6. **Aucun envoi n'a plus de 30 à 38 % de chances de monter.** Ordre conseillé : ep-24 seul (4 h 59, quasi sans risque), puis des tests à 3 guichets, plus riches en information.
7. **Note finale** : sans les lecteurs, 1967 ne vaut en espérance que **0,25 sur les paires cachées**. Le public ne nous dit rien sur les guichets qui pèsent le plus.

## 1. Fichiers des 12 envois
| envoi | contenu (guichets isolés ; G = {17,18} groupés) | score | source |
|---|---|---|---|
| 1803 | 14, 23, 07, 28, 16 | 0,10 | `t2_critique_B_temperature.csv` d0bf0510… (octets) |
| 1882 | 05, 15, 22 | 0,40 | **reconstitué** (« fp8 seuls », fil #4) |
| 1967 | 05, 15, 22 + G | 0,60 | `t2_fp8_loraQ3.csv` da146778… (octets) |
| 2052 | 05, 15, 22, 17, 18 séparés | 0,60 | **reconstitué** |
| 2176 | 1967 + 11 | 0,60 | `t2_plus_ep11.csv` 94c7f492… (branche t2/collecte-25) |
| 2383 | 1967 + 11, 19 | 0,60 | `t2_plus_ep19.csv` b39f027f… (branche) |
| 2491 | 1967 + **11**, 28, 03 | 0,60 | `t2_ep28_ep03.csv` aba81377… (branche) |
| 2595 | 1967 + 09, 11 | 0,60 | **reconstitué** (base 1967 supposée, = `t2_repli_1967_plus_ep09_ep11`) |
| 2707 | 1967 + 12, 24 | 0,60 | **reconstitué** (testé aussi avec 11 : 911 triplets au lieu de 893, mêmes conclusions) |
| 2795 | H : 04, 12, 17, 18, 03, 21, 22, 32, 05, 09, 11, 15 | 0,30 | `t2_H_base_ep01_ep26.csv` bbe0d6ca… (octets) |
| 2899 | 1967 + 01 | 0,60 | `t2_1967_plus_ep01.csv` df99bbe4… (octets) |

Le fichier 1803 compte pour deux scores (12 au total avec le 0,10 de départ). Formule officielle (`task_template.py`) : MCC ramené à 0 s'il est négatif ; 10 % pour la famille sur toutes les paires, 90 % pour le modèle sur les paires internes ; familles supposées justes.

## 2. Méthode d'énumération (exacte)
Chaque guichet est dans un de trois états : hors public, copie publique, variante publique (son propre modèle). Une famille compte 3^11 = 177 147 états. On calcule leurs comptes de paires pour chaque fichier, puis on les projette sur (|S_f|, paires vraies, paires fausses, VP et FP de 1967). On obtient 84 802 triplets compatibles avec 1967. On filtre ensuite famille par famille avec les fichiers qui ne touchent qu'elle, puis on combine les trois familles sur les fichiers qui les touchent toutes ensemble (1803, 2491, H). Le tout prend 40 s.
```
$ python t2-oracle-exact.py etats
triplets de projections compatibles avec 1967 et |S| : 84802
triplets retenus apres tous les fichiers : 1241 ; masse totale (poids « etats ») : 5.57499e+07
taille de S : 14:0.11 15:0.09 16:0.17 17:0.13 18:0.18 19:0.17 20:0.08 21:0.06 22:0.03
```
Ce que mon échantillonneur uniforme (`t2-oracle-mcmc.py`) ne fait pas : explorer toutes les régions. Les graines visitent des ensembles d'états presque disjoints. Ses fréquences par guichet restent proches de l'exact (écart ≤ 0,05).

## 3. Hypothèses de structure (vraisemblance des 12 scores sous le masque S, |S| de 14 à 22)
```
$ python t2-oracle-evidence.py        (+ t2-oracle-gabarit.py pour b)
k3      2.734e-07     (3 variantes par famille)
k4      2.237e-07     (a : 7 copies + 4 variantes, comme les ref)
k5      1.111e-07     (e)
k6      3.481e-08     (e)
U       1.080e-07     (chaque guichet variante avec proba 1/2)
c4      2.162e-07     (c : 4 variantes dont une paire sert le même modèle modifié)
b4 : P(12 scores | b4, S) = 3.893e-07 ; ball (gabarit, nombre libre) = 1.416e-07
rejet a4 paires : 0 tirage gardé sur 1,2e8 (4 lancers) ; rejet a4 S : 4 sur 3e7 (attendu 5,8)
```
- **(a) 4 variantes par famille** : compatible. Une famille sans aucune copie publique reste rare : 0,7 % des tirages exacts k4.
- **(b) gabarit** : c'est le meilleur, mais il ne gagne que ×1,7 sur (a). Dans 7 % des tirages exacts « états », toutes les classes concordent. Le public y voit la classe T=1 et la classe prompt dans 93 % des cas, mais P(variante et visible) n'est que de 0,22 à 0,27 par classe.
- **(c) variante en double** : aussi vraisemblable que (a). Le public ne peut pas la distinguer : séparer 17 | 18 ne change qu'une paire, et 2052 = 0,60.
- **(d) variantes déguisées** : le score ne voit que « même modèle ou non ». Un déguisement ne change donc rien au calcul, et (d) donne exactement le même résultat que (a). Seule une contrainte de classe, comme (b), la distingue.
- **(e) 5 ou 6 variantes** : 5 reste possible (÷2) ; 6 est peu probable (÷6,4).
- **Candidat du brief** : 0 S compatible (`t2-oracle-exact.py "fix:12,17,18,24,03,22,26,28,05,09,11,15"`). Au mieux, il reproduit 10 fichiers sur 11 et donne 0,78 pour 2707 au lieu de 0,60.

## 4. Guichets : P(public) | P(variante si public), sous le mélange k3, k4, k5 et b4 (poids = vraisemblance)
```
Qwen-3B   01:0.03|0.00 02:0.66|0.22 04:0.74|0.06 12:0.23|0.07 14:0.64|0.06 17:0.54|1.00 18:0.53|1.00 23:0.64|0.08 24:0.25|0.81 25:0.65|0.23 33:0.66|0.21
Qwen-1.5B 03:0.06|0.10 06:0.57|0.25 07:0.63|0.07 08:0.57|0.22 20:0.58|0.19 21:0.51|0.02 22:0.49|0.98 26:0.57|0.31 28:0.06|0.72 30:0.56|0.22 32:0.51|0.02
Llama-1B  05:0.62|0.66 09:0.05|0.23 10:0.49|0.34 11:0.05|0.24 13:0.50|0.34 15:0.62|0.67 16:0.48|0.13 19:0.08|0.12 27:0.50|0.22 29:0.49|0.27 31:0.49|0.27
```
- **Variantes quasi sûres, si publiques** : 17, 18 (au moins l'un des deux est public avec P = 1,00, c'est le saut 0,40 → 0,60 ; les deux à la fois avec P = 0,08 seulement), 22, 24.
- **Copies publiques sûres** : 04, 21, 32 (la chute de H), 07, 14, 23.
- **Ce que le public ne tranchera JAMAIS, en pratique** (P(public) ≤ 0,08) : **01, 03, 09, 11, 19, 28**. S'y ajoutent 12 et 24 à 0,23-0,25. Les paires 17-18 et 05-15 non plus : chacune ne change qu'une paire, ce que l'arrondi efface.
- Les guichets que les envois n'ont jamais touchés (02, 25, 33 / 06, 08, 20, 26, 30 / 10, 13, 27, 29, 31) sont **interchangeables** pour le public. Seul un envoi qui les sépare peut les départager.

## 5. Plan d'envois (`t2-oracle-arbre.py`, loi a posteriori du mélange)
Une baisse ne coûte rien au classement : seul le meilleur score compte. Je classe donc les envois par E[max(score, 0,60)], puis par l'information que porte le score exact.
```
classe {10,13,26}            E 0.638 hausse 0.29 inch 0.18 baisse 0.53 info 3.66 bits
classe prompt {02,08,29}     E 0.637 hausse 0.20 inch 0.11 baisse 0.69 info 3.76
classe filigrane {25,06,19}  E 0.637 hausse 0.19 inch 0.17 baisse 0.64 info 3.44
classe T=1 {33,30,31}        E 0.636 hausse 0.19 inch 0.11 baisse 0.70 info 3.76
iso 24                       E 0.635 hausse 0.20 inch 0.78 baisse 0.02 info 1.16
classe {10,13}               E 0.627 hausse 0.27 inch 0.35 baisse 0.39 info 2.96
iso 26                       E 0.629 hausse 0.18 inch 0.44 baisse 0.38 info 2.85
```
Fichiers fabriqués à partir de `t2_fp8_loraQ3.csv`, avec les mêmes octets (CRLF) hors des lignes changées. Chacun a été relu par un contrôle séparé : 33 lignes, familles inchangées, prédiction identique au modèle de calcul.
| n° | fichier | changement | sha256 |
|---|---|---|---|
| 1 | `t2_oracle_1_iso24.csv` (= `t2_oracle_iso24.csv`) | 24 isolé | 2322ec0b…ccd90 |
| 2 | `t2_oracle_2_classe_10_13_26.csv` | 10, 13, 26 | 362cde33…a5e10 |
| 3 | `t2_oracle_3_classe_prompt_02_08_29.csv` | 02, 08, 29 | ab870001…44d02 |
| 4 | `t2_oracle_4_classe_T1_33_30_31.csv` | 33, 30, 31 | 7d557946…98cc9 |
| 5 | `t2_oracle_5_classe_filigrane_25_06_19.csv` | 25, 06, 19 | 048a5495…16629 |
| 6 | `t2_oracle_6_classe_10_13.csv` | 10, 13 | 7e225907…a5cb3c |
| 7 | `t2_oracle_7_iso26.csv` | 26 | 261ee5be…a12f8b |
| 8 | `t2_oracle_8_base05.csv` | 05 remis en base (05 est-il une copie ?) | 4a3569ac…88a6c |
| 2h–5h | `t2_oracle_{2h..5h}_sur24_*.csv` | mêmes tests sur la base 1967 + 24 | 90774f0f… / 76597d53… / 818d865e… / 04459e72… |

**Arbre** (`arbre.json` ; probabilités de l'issue d'ep-24) :
- **ep-24 → 0,60 (78 %)** : 24 est hors public ; la base reste 1967. Envoyer ensuite le n° 2 {10,13,26} (hausse 0,33), puis le n° 3, le n° 4 et le n° 5.
- **ep-24 → 0,75 (9 %) ou 0,80 (10 %)** : 24 est une variante publique ; la nouvelle base est 1967 + 24. Envoyer ensuite `2h` {26,10} (hausse 0,23) ou `3h` ep-26, puis `4h` filigrane.
- **Après un test à 3 guichets** : une hausse garde le fichier comme nouvelle base et en isole les variantes un par un. Une baisse donne un score exact qui sépare la plupart des cas (3,7 bits). Le planificateur relancé sur ce score donne alors le suivant (exemple d'une baisse à 0,45 après {10,13,26} : envoyer ep-24, avec une hausse à 0,34).
- **ep-26 seul**, prévu à 4 h 59 dans l'ancien plan : hausse 0,18 et baisse 0,38, en dessous de ep-24.

## 6. Ce que le public ne peut pas donner, et le fichier final
- **81 % des paires internes (de 75 à 84 %, quantiles 10-90) touchent un guichet hors public**, mesuré sur les tirages exacts k4. Leur nature ne vient que de la loi a priori et des lecteurs.
- Le public dit tout de même quelque chose sur les guichets cachés. Avec 4 variantes par famille, 17 et 18 en prennent deux en Qwen-3B : il en reste 2 parmi les guichets non vus (01, 12, 24…). P(variante) a priori complétée : 01 0,38 · 12 0,32 · 24 0,50 · 03 0,46 · 28 0,49 · 09 0,36 · 11 0,36 · 19 0,30.
```
$ python t2-oracle-final.py   (60 000 vérités complètes tirées du mélange)
fichier                          E[caché]  E[tout]  E[public]
1967                              0.252    0.310    0.599
1967 + 24                         0.246    0.311    0.631
1967 + {03,28}                    0.258    0.314    0.598
1967 + {09,11,19}                 0.211    0.274    0.600
isoler P(var)>0,4 (+17|18 séparés) 0.260    0.319
glouton sur E[caché] : +28, +03 -> 0.258
```
- **Lecture** : sans lecteurs, aucun fichier ne dépasse 0,26 en espérance sur les paires cachées. C'est le plafond du public seul. Ce sont les lecteurs (LoRA sur 09/11, z de +5 à +9 ; « un seul de 28/03 ») qui feront la note finale. Pour les brancher, il suffit de remplir `PRIOR_LECTEUR` dans `t2-oracle-final.py` ; je ne l'ai pas fait.
- **Coût public des choix sur les guichets cachés : quasi nul.** Ajouter {09, 11, 03, 28, 19} à un envoi ne change le score public que dans ≤ 3 % des cas, contre 22 % si l'on ajoute 01 et 12. Si le fichier noté à la fin est le meilleur fichier public, chaque envoi devrait déjà porter nos meilleurs choix sur 09, 11, 03, 28 et 19. Ces choix viennent des lecteurs, pas d'ici.

## Limites
- Les 4 fichiers reconstitués (1882, 2052, 2595, 2707) sont tirés du fil #4, pas des octets. Pour 2707 avec ep-11, les conclusions ne changent pas.
- Je suppose les familles justes, l'arrondi au 0,05 le plus proche et S de 14 à 22 guichets, toutes tailles traitées à égalité. Si l'on tire d'abord la taille au hasard, les vraisemblances baissent de 15 % sans changer l'ordre.
- Le gabarit est calculé par repondération (taille effective 3 500 tirages), pas énuméré.
- Les probabilités dépendent de l'a priori (k3/k4/k5/b4 mélangés) ; les classements d'envois restent stables d'une hypothèse à l'autre (§5, `t2-oracle-envois.py`).
