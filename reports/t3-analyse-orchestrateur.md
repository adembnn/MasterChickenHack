# Track 3 : analyse du problème par l'orchestrateur (21/11, 18 h 55)

Ce document pose des hypothèses à tester. Rien ici n'est une mesure, sauf les lignes marquées [MESURÉ].

## 1. Le score est un produit de trois facteurs
score = S_loss × S_pred × S_ret, avec :
- S_loss = 1 − clip(d_sub / d_orig), d = |moyenne(L) − moyenne(L_gold)| + |écart-type(L) − écart-type(L_gold)|, L = entropie croisée à température 1 avec la vraie étiquette ;
- S_pred = 1 − clip(KL(p_gold ‖ p_sub) / KL(p_gold ‖ p_orig)), softmax à température 2, p_gold = MOYENNE des distributions de plusieurs modèles réentraînés ;
- S_ret = clip(acc_sub / acc_orig) × 1 / (1 + KL(p_orig ‖ p_sub)) sur un jeu de contrôle jamais vu par `orig`.
S_loss et S_pred sont chacun moyennés sur les images exactes et sur des copies modifiées cachées. Un facteur proche de 0 annule tout.

## 2. Ce que chaque facteur exige
- **S_pred exige d'adoucir TOUTES les images du forget set**, pas seulement celles que l'on fait échouer. [MESURÉ] `orig` garde une probabilité maximale de 0,985
  à température 2 sur ces images. Le gold est une moyenne de plusieurs modèles sur des images qu'ils n'ont pas vues : sa distribution est étalée sur les classes voisines
  pour presque toutes les images. La KL « avant » KL(gold‖sub) punit fort un sub qui met une probabilité quasi nulle là où le gold en met. Tant que 90 % de nos images
  restent aussi pointues qu'`orig`, KL(gold‖sub) ≈ KL(gold‖orig) sur ces images et S_pred ≈ 0.
- **S_loss exige une queue d'images franchement ratées** (loss élevée) pour reproduire l'écart-type, et le bon niveau moyen.
- **S_ret exige de ne RIEN changer sur les images jamais vues.** Un adoucissement global (multiplier les logits par un facteur) adoucit aussi le jeu de contrôle,
  donc KL(orig‖sub) monte et S_ret chute.
Conclusion : la cible est un modèle qui traite les 3 500 images, et leurs copies modifiées, comme il traite déjà une image jamais vue du même domaine, sans changer
son comportement sur les images jamais vues. Il s'agit de supprimer l'écart membre / non-membre, de façon sélective.

## 3. Lecture des scores réels à travers cette grille (hypothèses)
- Envoi 1197 (`S1_A85_l30`, 0,0114) : 98 % des images sont restées pointues (confiance moyenne 0,98). Hypothèse : S_pred ≈ 0,03 ; S_loss et S_ret corrects.
- Envoi 1180 (tête × 0,35, 0,0039) : adoucissement global. Hypothèse : S_ret fortement dégradé (KL de rétention mesurée localement 1,26 → facteur 0,44),
  S_loss faible (pas de queue : toutes les images ont à peu près la même loss), S_pred peut-être correct.
- Aucun de nos envois n'a combiné adoucissement sélectif + queue d'erreurs + rétention intacte.

## 4. Quadrillage des familles de méthodes
| Famille | S_pred | S_loss | S_ret | Session |
|---|---|---|---|---|
| Adoucissement global (température, échelle de la tête) | monte | moyenne seulement | chute | sonde de contrôle |
| Distillation vers des cibles par image, peu d'images modifiées (S1) | reste bas | monte | intact | t3-methodes |
| Distillation de TOUTES les images vers une cible « non-membre » + queue d'erreurs + ancre sur non-membres | monte | monte | intact si l'ancre est du bon domaine | t3-methodes + t3-nonmembres |
| Cohérence : sub(x) = moyenne d'orig sur des vues fortes de x, et sub(vue) = orig(vue) | monte sur les exactes | partiel | bon | à tester |
| Négation de vecteur de tâche, effacement ciblé des poids qui mémorisent | inconnu | inconnu | inconnu | t3-poids |
| Effacer puis réparer | monte | monte | dépend de la réparation | t3-poids |

## 5. Les trois inconnues à lever, par ordre d'utilité
1. Niveau « non-membre » : confiance et loss d'`orig` sur des images jamais vues du même domaine (t3-nonmembres, t3-simulateur).
2. Poids relatif des trois facteurs dans nos envois : sondes qui ne bougent qu'un facteur à la fois (t3-sondes).
3. Peut-on adoucir sélectivement sans données du domaine ? (t3-methodes, t3-poids)

## 6. Expériences décisives proposées
- E1 : adoucissement sélectif pur. Entraîner pour que, sur les images exactes ET leurs vues, la distribution à température 2 soit celle d'`orig` adoucie
  (probabilité maximale visée 0,5 à 0,7, masse restante sur les classes voisines), sans changer aucune classe prédite, avec une ancre forte. Si le score monte
  nettement au-dessus de 0,03, S_pred était bien le facteur limitant.
- E2 : E1 + bascule de 10 %, 20 %, 30 % des images les plus instables vers leur deuxième classe. Lit la précision du gold.
- E3 : la même cible, obtenue par un adoucissement global. Compare à E1 : lit le coût en rétention.
