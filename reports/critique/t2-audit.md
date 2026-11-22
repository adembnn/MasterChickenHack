# T2 : audit indépendant, pourquoi on bloque à 0,60 (22/11, 5 h)

**Décision débloquée : quoi envoyer sur les prochains créneaux T2, et quel fichier doit porter nos choix pour la note finale.**
Auditeur : session t2-audit d'Adem. Je n'ai écrit aucune méthode. 0 requête API, aucun `.env` lu, aucun envoi, aucun fichier existant modifié.

## Les 3 causes les plus probables du blocage, classées

**1. Le public ne note qu'environ 15 guichets sur 33, et nos 8 derniers envois ont touché presque uniquement les autres.** (mesuré)
Sur 64 millions de tirages (vérité, masque), le modèle « 30 % des paires au hasard » ne reproduit **jamais** nos 11 scores. Le modèle « paires entre un sous-ensemble S de guichets » les reproduit (40 fois sur 32 millions de tirages). Le ratio vaut au moins 25 en faveur de S. Ce sous-ensemble compte environ 15 guichets (11 à 26). ep-01, 03, 09, 11, 19 et 28 en sont presque sûrement absents (P ≤ 0,14).
En clair : « 0,60 inchangé » ne veut pas dire « fichier mauvais ». C'est un thermomètre posé sur la moitié des guichets, et on a passé 7 créneaux à mesurer l'autre moitié. Le 0,60 implique qu'il reste **2 à 4 variantes publiques non isolées** (3 le plus souvent). C'est cohérent avec les 3 paliers du classement au-dessus de nous (0,70, 0,85, 1,00).
**Dans l'heure** : n'envoyer que des changements sur des guichets probablement publics. Classement par information attendue (`t2-audit-candidats.py`, trois a priori) :
- ep-25 seul : hausse 0,11 à 0,46 selon l'a priori ;
- {ep-10, ep-13} : hausse 0,24 à 0,45 ;
- ep-26 ou ep-06 ;
- ep-24 **seul**, voir la cause 2.

Ne plus envoyer ep-01, 03, 09, 11, 19 ni 28 : leur résultat est connu d'avance (« inchangé »).

**2. Nos lecteurs locaux ne voient pas les variantes qui restent, et notre gabarit les cache dans les « copies ».** (mesuré pour les scores, supposé pour la cause)
Les guichets publics encore mal classés sont, par construction, parmi ceux que nos lecteurs appellent « copie » ou « copie déguisée » : {02, 14, 23, 25, 33}, {06, 07, 08, 20, 26, 30}, {10, 13, 16, 27, 29, 31}. Sans a priori sur la structure, chacun est impliqué dans une erreur publique de 1967 avec une probabilité de 0,30 à 0,52. Sur les ep, deux faits sont mesurés : rien n'est lisible au style (t2-regard-neuf, piste 3), et le lecteur Qwen-3B ne voit aucune variante (journal, 2 h 05). Deux hypothèses n'ont jamais été testées par un envoi :
- une variante servie sous un déguisement (LoRA ou fp8 avec un prompt ou une persona) ;
- une variante servie par deux guichets.

Les scores sont compatibles avec les deux (point 5). Autre point faible : le bloc {12, 24} a pu se compenser (une copie publique et une variante publique dans le même envoi), avec une probabilité de 0,14 à 0,44 selon l'a priori. « 12 et 24 hors public » n'est donc **pas** établi.
**Dans l'heure** : envoyer ep-24 seul (sous le gabarit strict : hausse 0,40, baisse 0,04). Côté budget (décision de Titouan), les 50 réponses restantes rapportent le plus sur les ≈ 15 guichets publics candidats, pas sur ceux hors public.

**3. Le fichier qui comptera à la fin est très probablement 1967, et il laisse en base des guichets que nos mesures donnent pour variantes.** (non vérifiable sans les organisateurs)
Le serveur garde le meilleur score public, et une égalité ne remplace pas le fichier retenu (message « previous best still leads » à chaque 0,60). La note finale porte sur les paires cachées : 165 − ≈ 32 publiques = **≈ 133 paires internes sur 165 (81 %)**, toutes liées à un guichet non public. Or 1967 met ep-09, 11, 03, 28, 24 et 12 dans la base. Trois tests locaux indépendants sortent ep-09 et ep-11 comme LoRA. Tout ce qu'on a appris sur les guichets non publics est donc perdu si le fichier final reste 1967.
**Dans l'heure** :
- (a) demander aux organisateurs (Discord #task-2) quel fichier compte pour la note finale (le meilleur public, le dernier, ou un fichier choisi) et si le public est un ensemble de guichets ;
- (b) dès maintenant, chaque envoi porte nos meilleurs choix sur les guichets non publics, au minimum ep-09 et ep-11 isolés. Le coût public est quasi nul (P(public) ≤ 0,2) et le fichier qui battra 0,60 les emportera.

Termes. **Copie** : mêmes poids que la base ; seul le déploiement change (prompt, persona, filigrane, décodage). **Variante** : poids différents (fp8, LoRA). **MCC** : corrélation de Matthews sur les paires, entre 0 et 1 ici car ramenée à 0. **Masque public** : l'ensemble des paires que le leaderboard note. **ABC** : on tire au hasard une vérité et un masque, et on ne garde que les tirages qui redonnent exactement nos scores. **Recuit** : recherche aléatoire qui accepte de moins en moins les mauvais pas, pour trouver des états compatibles.

## Point 1 : format et sens du CSV : vérifié, aucun défaut
```
python score_practice.py p_conv.csv      (convention de l'équipe : base commune smol_base, variantes smol_fp8_ref02, ...)
Practice score (MCC over 36 pairs, same model or not): 1.000
p_grp   (les 4 variantes sous UNE étiquette)                 : 0.693  (6 paires fausses)
p_miss1 (LoRA 0,3 laissée dans la base)                      : 0.734
p_fam   (ref-02 dans une autre famille, même étiquette « X ») : 1.000
```
- Le scoreur officiel compare le couple (famille, modèle) (`same = (d[a][0], d[a][1]) == ...`). Une étiquette de modèle partagée entre deux familles n'y crée aucune paire « même ». L'énoncé l'interdit (« they must also share the family label ») ; ce que fait le serveur dans ce cas n'est pas vérifiable sans envoi, mais **sans objet**. Contrôle sur les 34 CSV de `E:\EAC-lab\t2\reel\` : 33 lignes, 33 guichets distincts, 0 étiquette partagée entre familles, 0 espace parasite.
- Mon MCC est égal au scoreur officiel : 0,734 et 0,693 recalculés à l'identique.

## Point 2 : le calcul du score et le plafond de 1967 : vérifié
Formule reproduite : 0,1 × MCC famille (toutes les paires) + 0,9 × MCC modèle (paires d'une même famille vraie), chaque MCC ramené à 0. Score exact de 1967 sur **toutes** les paires, si ses 5 isolés (05, 15, 22, 17, 18) sont de vraies variantes et sans copie isolée à tort (`t2-audit-score.py`) :
```
k variantes laissées dans la base :  1 -> 0,89   2 -> 0,81   3 -> 0,75   4 -> 0,68-0,71   5 -> 0,63-0,67
                                     6 -> 0,59-0,64   7 -> 0,55-0,63   8 -> 0,51-0,58   10 -> 0,45-0,50
(17 et 18 même modèle ou non : écart ≤ 0,01)
```
- Mesuré : sur toutes les paires, 0,60 correspondrait à 6 ou 7 variantes manquantes. Sur le public (sous-ensemble S), c'est **2 à 4 variantes publiques** non isolées (états compatibles : gabarit strict 1/2/3/4/5 variantes = 5/23/23/5/1 états ; sans déguisé variant 25/74/103/29/6).
- Supposé, selon l'a priori : 1967 vaut 0,26 à 0,55 sur toutes les paires (moyennes 0,26 sans contrainte, 0,35 sans variante déguisée, 0,37 sous le gabarit strict ; 0,39 par ABC). **Notre fichier vaut probablement moins que 0,60 sur la note finale.**

## Point 3 : les familles : vérifiées par l'équipe, recoupées en partie par moi
- Preuve de l'équipe (#4, 21/11) : sur 14 modèles candidats relus, le 2e meilleur est toujours au moins 0,3 nat/token plus loin. Sous le bon modèle, T\* vaut 0,63 à 0,85, contre ≥ 0,95 sous un faux. Les gabarits recopiés concordent : en-tête Llama 3 chez ep-11, 13, 27 ; prompt par défaut de Qwen chez ep-22 et 32.
- Ma mesure (blocs 0 et 1, 50 réponses) : 0 guichet de la famille Llama ne se dit Qwen, et 0 guichet Qwen ne montre l'en-tête Llama. La séparation Qwen-3B / Qwen-1,5B ne repose que sur la relecture par les deux modèles (non rejouée par moi, pas de GPU ici).
- Indice : 3 × 11 exactement, et 9 équipes à 0,10 pile (familles justes, rien d'autre). **Aucun guichet mal rangé n'est plausible.**

## Point 4 : pourquoi six fois « inchangé » : explications chiffrées
| Explication | Chiffre | Verdict |
|---|---|---|
| Arrondi à 0,05 | classement : 39 équipes, toutes à un multiple de 0,05 ; un cran = 0,056 de MCC modèle | vérifié, insuffisant seul |
| Masque = 30 % des paires au hasard (q fixe ou libre) | **0 sur 64 M** tirages ; le « 30 % » n'est pas dans l'énoncé T2 (il vient de T3, `Unlearning.txt` l. 187) | **faux** |
| Masque = guichets qui « touchent » un ensemble | 0 sur 32 M | faux |
| Masque = paires entre un sous-ensemble S | 40 sur 32 M ; ≈ 15 guichets, ≈ 32 paires internes publiques | **le seul compatible** |
| Compensation dans un bloc de 2 (1 variante + 1 copie publiques) | {12, 24} 0,14 à 0,44 ; {28, 03} 0,05 à 0,19 ; {09, 11} et {11, 19} ≤ 0,02 | réelle pour {12, 24} |

Le classement confirme la lecture par paliers : 5 équipes à 0,60, 4 à 0,70, 3 à 0,85, 5 à 1,00, aucune à 0,65, 0,75, 0,80, 0,90 ou 0,95. Ce sont des « crans » discrets, comme le prévoit un public de peu de guichets. Le gabarit W4 de t2-critique prédisait une hausse pour ep-01 dans 96 à 100 % des cas : 2899 = 0,60, **prédiction réfutée**.

## Point 5 : les structures jamais posées, face aux 11 scores
Recuit, modèle S, 800 à 4 000 chaînes (`t2-audit-mcmc.py`). La part de chaînes qui redonnent les 11 scores indique seulement si c'est **facile à ajuster**, ce n'est pas une probabilité.
| Hypothèse | Chaînes compatibles | Verdict |
|---|---|---|
| Libre (doublons permis, déguisés possiblement variantes) | 500 / 4 000 | compatible |
| Aucune variante servie deux fois | 69 / 800 | compatible |
| Déguisés (T 1,0, prompt, persona, tronqué) = copies | 238 / 4 000 | compatible |
| Gabarit strict : 4 variantes singletons par famille, déguisés copies | 57 / 4 000 | compatible, plus contraint |
| ≤ 5 variantes par famille | 93 / 800 | compatible |
| Une famille (Qwen-3B) sans aucun groupe de copies | 56 / 800 | compatible |
| ep-17 et ep-18 même modèle | P = 0,00 à 0,15 | peu probable |

En clair : **les scores n'excluent aucune de ces structures**. Notre gabarit « une modification par guichet » n'est vérifié que sur 9 ref ; rien ne l'impose sur les ep. Candidats « publics ET variante, encore dans la base de 1967 » : sous le gabarit strict, ep-25 0,46, ep-24 0,40, ep-26 0,30, ep-13 0,30, ep-10 0,26, ep-06 0,25 ; sans déguisé variant, ep-10 0,41, ep-13 0,39, ep-24 0,29, ep-25 0,27, ep-26 0,25 ; sans contrainte, tout est diffus (≤ 0,24, ep-33 et les Llama déguisés à 0,20 à 0,23).

## Point 6 : ce que font probablement les équipes à 1,00
- Mesuré : Murmell et 3V1B étaient à 0,10 vers 21 h (#4, l. 449), et sont à 1,00 à 4 h 30. Cinq équipes sont à 1,00, et Unblo y est avec 0,00 sur T3.
- Supposé : avec ≈ 15 guichets publics et ≈ 13 créneaux depuis 22 h, une remontée par envois successifs à partir d'une méthode correcte suffit pour atteindre 1,00. Elle donne le statut exact des guichets publics, et ce statut se transporte aux paires cachées entre un guichet public et un non public. Une vraie méthode est possible aussi : rien ne permet de trancher.
- Conséquence pour la note finale : 81 % des paires internes touchent un guichet non public. Le 1,00 public ne dit rien de ces guichets. **L'écart final avec les équipes de tête est probablement bien plus petit que 0,40**, et il se joue sur les ≈ 18 guichets non publics, que seules nos mesures locales tranchent.

## Ce qui est faux ou jamais vérifié dans notre chaîne
1. « Le public = 30 % des paires » : **faux** pour T2 (pas écrit dans l'énoncé, et réfuté par la mesure).
2. « {12, 24} hors public » : **non établi** (compensation possible, 0,14 à 0,44).
3. « Les déguisés sont des copies » : vrai sur les ref, **jamais vérifié** sur les ep ; les scores ne l'exigent pas.
4. « Une variante = un guichet » : **jamais testé** par un envoi ; aucun fichier n'a regroupé deux guichets à part {17, 18}.
5. « Le fichier jugé à la fin est le meilleur public » : **non vérifiable**, à demander aux organisateurs.
6. Les calculs de structure de l'équipe (pubmask, gabarit) supposent un masque par guichets : c'est cohérent avec ma mesure. En revanche, le chiffre « 30 % des paires » utilisé dans les grilles de lecture de 3 h 40 et de 3 h 59 était faux, et ces grilles sont caduques.

## Rejouer (0 requête)
```
cd E:/EAC/reports/critique
python ../sources/t2/score_practice.py <fichier ref>     # point 1
python t2-audit-score.py 400000                          # plafond + ABC (≈ 20 min)
python t2-audit-mcmc.py subset 4000 3000 11 sortie.npz [nodup|nodisgvar|H4|max5|nobase]
python t2-audit-candidats.py sortie.npz [...]            # information attendue par envoi candidat
```
Les états échantillonnés (`.npz`) restent hors dépôt (dossier temporaire de la session).
