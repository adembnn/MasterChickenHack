# Idées T3 Unlearning (ViT-B/16) : méthodes qui sortent du lot

Date : sam. 21 nov. 2026 ~17h. Lu : CONTEXT.md, IDEATION.md, Unlearning.txt, task_template.py, t3-unlearning.md (agent précédent), t3-unlearning-etat-de-lart.md (dépôt), extraits de strategy.md et submissions-ops.md. Rien téléchargé, aucun endpoint appelé. Légende de preuve : [MESURÉ-toy] = mesuré par moi sur un simulateur SYNTHÉTIQUE en Python pur (toy/t3toy3.py, dossier research/toy) : il vaut pour la STRUCTURE de la formule, pas pour les valeurs absolues ; [PAPIER mém.] = de mémoire, non relu ; [HYPOTHÈSE] = non vérifié. Je n'ai pas répété t3-unlearning.md (métrique, R-A/R-B, simulateur CIFAR-100, plan de nuit) : je pars de lui et j'ajoute ce qui change les priorités.

## 0. Message central (ce que le toy change dans le plan)
1. **Le score forget est dominé par le terme « moyenne/écart-type », pas par le terme « prédiction »** [MESURÉ-toy]. Une cible qui ne fait qu'aplatir les sorties donne déjà pred = 0,85 à 0,90 ; connaître la bonne classe fausse et la bonne image difficile n'ajoute que +0,03 à +0,07. En revanche mean_std est une **tente en L1** : une erreur relative de 15 % sur (m*, s*) coûte −0,14 de mean_std, 30 % coûte −0,29. Le levier n°1 est donc d'estimer (m*, s*) à ±10 %, pas de raffiner le pseudo-gold image par image. Conséquence : R-B (pseudo-gold par élèves) passe de « rang 2 » à « pari optionnel » ; le temps GPU/humain va à (a) cibles moment-matchées par quantiles, (b) estimation/ajustement du cadran (m*, s*), (c) robustesse aux copies altérées, (d) fuite vers le retain.
2. **Une température globale ne suffit pas** [MESURÉ-toy] : meilleur τ global = 0,45 de forget (mean_std 0,52 : la moyenne colle, l'écart-type reste à 0,2 contre 1,6) ; affectation par quantiles de pertes cibles = 0,84 à 0,91 avec les bons moments. Réel attendu plus bas, mais l'ordre est robuste.
3. **Le gold est une moyenne d'ensemble : la cible « moyenne des softmax » a un écart-type trop bas** [MESURÉ-toy] : cible z = 2·log(p_gold) donne pred = 1,00 mais mean_std = 0,81 ; soupe des logits de 4 gold : pred 0,99 / mean_std 0,81. Les stats de loss sont celles d'UN modèle (écart-type plus grand). Donc on ne copie pas la moyenne : on copie la moyenne en probabilités (T=2) puis on re-durcit image par image (exposant de netteté, I3).
4. **Plafond** [MESURÉ-toy] : un gold indépendant unique, noté contre la moyenne de 4 autres, obtient forget 0,85 (mean_std 0,99, pred 0,85). Plafond réaliste du forget : 0,8 à 0,9 ; avec retain 0,9 à 0,97, plafond du score ≈ 0,75 à 0,85. Médiane du champ estimée à 0,35 (strategy.md) : le gain disponible est grand, et se joue sur le calibrage des moments.
5. **Risque structurel que personne n'a chiffré : la fuite d'aplatissement vers le retain.** Aucune méthode connaissant seulement F ne distingue F des membres du retain (aussi mémorisés, aussi « typiques »). Fine-tuner pour moins de confiance sur F aplatit aussi le retain si le contrôle est fait de membres : coût typique KL(orig||sub) 0,05 à 0,15 → retain 0,87 à 0,95 [HYPOTHÈSE, à mesurer au simulateur, I12]. Si le contrôle est du non-membre, pas de coût.

## 1. Toy synthétique : protocole et chiffres (reproductible, ~10 s, Python pur)
100 classes, 1200 images, K=4 gold ; logits gold = Tg·(w partagé + bruit par modèle), voisinage de 5 classes par classe ; gold individuel : accuracy 0,70, CE moyenne 1,51, écart-type 1,62 ; orig : logit vrai ≈ 13 (loss ≈ 0) ; KL(p_gold||p_orig) = 1,5. Formule exacte du template, groupe G1 seul, retain ignoré.
| Stratégie | mean_std | pred | forget |
|---|---|---|---|
| orig | 0 | 0 | 0 |
| température globale τ=4 (la meilleure) | 0,52 | 0,86 | 0,45 |
| cible = 2·log(moyenne gold T=2) (oracle d'ensemble) | 0,81 | 1,00 | 0,81 |
| soupe des logits des 4 gold (oracle) | 0,81 | 0,99 | 0,80 |
| un gold indépendant (plafond d'un réentraînement) | 0,99 | 0,85 | 0,85 |
| quantiles lognormaux (m*, s*) exacts, dureté r_h=0, classe fausse au hasard | 0,99 | 0,85 | 0,84 |
| idem r_h=0,5, r_w=0,5 | 0,99 | 0,89 | 0,88 |
| idem r_h=0,8, r_w=0,9 | 0,99 | 0,92 | 0,91 |
Sensibilité à l'erreur sur (m*, s*) (les deux mis à l'échelle f) : f=0,70 → mean_std 0,69 ; 0,85 → 0,84 ; 1,00 → 0,99 ; 1,15 → 0,86 ; 1,30 → 0,71. Erreur sur m* seul de ±30 % : forget 0,74 à 0,75 ; sur s* seul : 0,74 à 0,77. (r_h = corrélation de notre classement de dureté avec la vraie ; r_w = taux où notre classe fausse est la vraie classe fausse du gold.)
Limites : pred de ce toy dépend de l'échelle des logits ; dans une variante à logits plus grands la même information ajoutait +0,05 à +0,08 de pred au lieu de +0,03 à +0,07, sans changer le classement des idées (calibrage de cette variante mauvais : accuracy 0,39, à ne pas citer comme chiffre). Dans le réel, les erreurs de hardness/classe sont structurées (pas des corrélations propres). Les tailles absolues viennent d'un calibrage manuel.

## 2. Idées détaillées (les 12 premières par importance ; format 1 à 8 condensé)

### I1. Cibles par quantiles moment-matchés (QA-MM) [cœur]
1. Idée : (a) estimer par image une dureté h_i (signaux de I5/I6/I7) ; (b) tirer N pertes cibles ℓ_(1)≤…≤ℓ_(N) comme quantiles d'une lognormale de moyenne m et d'écart-type s (m,s = cadran I2) ; (c) affecter ℓ aux images par rang de h_i ; (d) q_i[y_i]=exp(−ℓ_i) ; si q_i[y]<0,5 une classe fausse (confusion I7) passe au-dessus ; le reste de la masse ∝ voisinage. Le moment T=1 est exact PAR CONSTRUCTION, plus de balayage ε.
2. Mécanisme : sépare les deux moitiés de la métrique : moments par la loi des ℓ, pred par l'ordre/la classe. Remplace « choisir ε » par « choisir (m,s) » : un problème à 2 scalaires mesurable.
3. Preuve : [MESURÉ-toy] 0,84 à 0,91 contre 0,45 (τ global). [HYPOTHÈSE] pour le transfert au vrai ViT-B.
4. Gain : forget réel de ~0,35 (tempéré brut) à 0,55-0,75 ; score final +0,15 à +0,25 si (m,s) à ±15 % (hypothèse : retain ≥ 0,9).
5. Faisabilité : 4 à 6 h réelles (1 personne) ; 0,5 h-GPU de précalcul des vues + 6 à 10 min par essai d'entraînement ; dépend du pipeline R-A (distillation CE vers q, KL T=2, ancres) déjà décrit. Casse : la forme de la loi (lognormale) ≠ forme réelle des pertes du gold ; atténuation : mixture 2 composantes (faciles ≈ 0,05-0,3 ; difficiles ≈ 1,5-4) ajustée aux mêmes (m,s) ; la tente en L1 est tolérante à la forme, pas aux moments.
6. Risque : le réseau ne parvient pas à ajuster les ℓ_i imposés (on le mesure : moments obtenus de loss d'entraînement avec vraies pseudo-étiquettes vs cible). Alerte : mean/std obtenus à > 10 % de la cible après 8 époques.
7. Test 30 min : sur ViT-B, 3 époques, cibles QA-MM avec (m,s)=(1,1 ; 1,9) ; critère : pertes moyenne/écart-type sur pseudo-labels dans ±10 % de la cible ET accord argmax sur ancres ≥ 97 % → on garde.
8. Conformité : légale (forget + modèle uniquement).

### I2. Cadran (m*, s*) et sondes en tente [le multiplicateur n°1] (zone grise, voir §8)
1. Idée : (m*, s*) sont inconnus ; leurs priors sont larges (m* 0,8 à 2,0 ; s* 1,5 à 3,0 [HYPOTHÈSE]). On produit 3 modèles qui ne diffèrent QUE par f (m,s)=f·(m0,s0), f ∈ {0,85 ; 1,0 ; 1,15}, et on lit les 3 scores publics. Comme mean_std est une tente à pentes égales (±k autour de f*), 3 points dont un de chaque côté donnent f* exactement : si f1<f2<f* (deux points du même côté) k=(S2−S1)/(f2−f1), puis f* = (f1+f3)/2 + (S1−S3)/(2k). Puis 3 sondes sur le rapport s/m, même formule. 6 soumissions maximum au total pour ce cadran, jamais d'ajustement fin ensuite.
2. Mécanisme : on ne devine pas une image, on règle UN scalaire d'hyperparamètre. Pente de la tente ≈ 0,01 de mean_std par 1 % d'erreur sur les moments [MESURÉ-toy] : 5 à 10 fois le bruit public (écart-type du public ≈ 0,02, car notre propre moyenne et celle du gold sont calculées sur le MÊME sous-ensemble, le bruit d'échantillonnage se compense en partie).
3. Preuve : [MESURÉ-toy] forme en tente (0,69 / 0,84 / 0,99 / 0,86 / 0,71). [HYPOTHÈSE] pour la symétrie des pentes dans le réel (le produit avec pred et retain l'arrondit légèrement).
4. Gain : évite la perte typique de 0,15 à 0,25 sur le forget si le prior est faussé de 15-30 % : +0,1 à +0,2 de score final. Meilleur rapport gain/effort de la liste.
5. Faisabilité : 1 h de code (famille de cibles paramétrée), 3 entraînements en parallèle = 1 nœud 10 min ; upload 345 Mo x 6 ; dépend du nombre et du délai de soumissions T3 (inconnus : à demander).
6. Risque : si le score public est plafonné par un autre facteur (porte, retain) la tente est invisible. Alerte : S0 < 0,1.
7. Test : avant de soumettre, vérifier au simulateur que 3 points suffisent (f* retrouvé à ±5 % sur 10 tirages de bruit), 30 min.
8. Conformité : à valider (§8). Version conservatrice sans sonde : prendre le centre du prior (médiane, I13).

### I3. Exposant de netteté par image (cibles à deux températures) 
1. Idée : q_i au niveau T=1 est fixé par I1 ; la cible T=2 qui compte pour pred vaut idéalement la moyenne d'ensemble (plus lisse). Représenter z_i = a_i·2·log(p̂_i) avec p̂_i = estimation de la moyenne gold (T=2) et a_i ≥ 1 choisi par image pour que la CE à T=1 vaille ℓ_i ; on évite ainsi de payer pred pour obtenir le moment (et inversement). Entraînement : CE(softmax(z_i), softmax(logits)) + β·KL(p̂_i^{T=2} || softmax(logits/2)).
2. Mécanisme : [MESURÉ-toy] l'oracle d'ensemble pur perd 0,19 de mean_std ; les quantiles perdent ~0,12 de pred ; un compromis par image récupère une partie des deux.
3. Preuve : [MESURÉ-toy] des deux extrêmes ; le compromis lui-même [HYPOTHÈSE].
4. Gain : +0,02 à +0,06 de forget.
5. Faisabilité : 1,5 h en plus de I1 ; GPU nul.
6. Risque : sur-ingénierie ; alerte : pas de gain > 0,02 au simulateur → on garde I1 seul.
7. Test : au simulateur CIFAR (ou toy) comparer I1 seul et I1+I3 ; garder si +0,02.
8. Conformité : légale.

### I4. Cohérence sur les copies altérées (G2 pèse 50 %) 
1. Idée : G2 = copies cachées « légèrement » altérées. Pour orig la mémorisation survit aux altérations légères (orig reste confiant), donc G2 est AUSSI à oublier ; d_orig(G2) est plus petit que d_orig(G1) (orig est moins confiant sur copies) : chaque erreur de moments y coûte plus cher. Entraîner avec une famille large d'altérations (recadrage 0,7-1, flip, rotation ±15°, flou 0-1,5 px, bruit, jitter, JPEG q 40-95, arrondi uint8) en gardant LA MÊME cible par image pour toutes ses vues ; cibles de moments des vues = ceux de la clean plus un décalage Δ (le gold est un peu moins sûr sur copies) réglé par une 7e sonde ou par le simulateur.
2. Mécanisme : sans invariance, le modèle apprend « cette image exacte → cible » (nouvelle mémorisation par pixel) et retombe sur orig sur la copie. Avec invariance il apprend « ce contenu → incertain ».
3. Preuve : [HYPOTHÈSE] ; mécanisme plausible, la mémorisation robuste aux augmentations est documentée pour les ViT [PAPIER mém.].
4. Gain : évite de perdre jusqu'à 0,5 du terme G2 : +0,05 à +0,2 de forget (grand si les altérations sont plus fortes que prévu).
5. Faisabilité : 2 h (pipeline de vues GPU, uint8-arrondies) ; coût : x4 vues par époque.
6. Risque : famille d'altérations ≠ cachée. Alerte : sur le simulateur, entraînement famille A / test famille B : écart de forget > 0,1.
7. Test 30 min : simulateur A/B (I12) ; critère : forget(B) ≥ 0,85·forget(A).
8. Conformité : légale.

### I5. Sévérité de dégradation calibrée = difficulté de gold (enseignant multi-vues) 
1. Idée : dureté et p̂ par image = moyenne d'orig sur V=8-16 vues dégradées (recadrage 0,25-0,7, flou, cutout). Nouveau : choisir la SÉVÉRITÉ pour que l'accord moyen d'orig avec son propre pseudo-label sur les vues égale a_g estimé (≈ 0,75-0,80), au lieu de tirer au hasard. Le niveau de dégradation devient un étalon de la difficulté d'un non-membre.
2. Mécanisme : sous dégradation la mémorisation de pixel disparaît mais le contenu reste ; ce que garde orig est proche d'un modèle qui généralise.
3. Preuve : [HYPOTHÈSE] ; biais : les erreurs viennent de la perte d'information (petits objets), pas du défaut de généralisation du gold.
4. Gain : +0,01 à +0,05 de pred.
5. Faisabilité : 1,5 h ; 0,3 h-GPU.
6. Risque : corrélation faible avec la vraie dureté. Alerte : Spearman < 0,2 entre dureté vues et vraie perte du gold au simulateur.
7. Test 30 min : simulateur, Spearman(h, perte du gold réelle) par type de signal (vues, tête CV, kNN) ; garder les deux meilleurs.
8. Conformité : légale.

### I6. Sondes CV par profondeur (« prédiction de non-membre à la profondeur qui oublie ») 
1. Idée : features de blocs 4, 6, 8, 10, 12 d'orig sur les 3500 images (pseudo-labels), tête linéaire/kNN en validation croisée 10 plis ; choisir la profondeur dont la précision CV égale a_g. Les features précoces portent peu de mémorisation, ce qui donne des prédictions « jamais vues » mais calibrées par profondeur. Les voisins kNN en features précoces donnent aussi la classe fausse plausible.
2. Mécanisme : [PAPIER mém.] Papyan et al., effondrement des classes sur le train (arXiv 2008.08186) : les features de la dernière couche sont trop propres sur les images d'entraînement ; les couches intermédiaires moins. La mémorisation n'est pas localisée sur la dernière couche mais répartie (Maini et al., arXiv 2307.09542).
3. Preuve : [HYPOTHÈSE].
4. Gain : +0,01 à +0,04 de pred (classe fausse et dureté).
5. Faisabilité : 2 h ; 0,3 h-GPU.
6. Risque : 35 images par classe seulement → sondes bruyantes. Alerte : précision CV < 0,5 à toute profondeur.
7. Test 30 min : même Spearman que I5.
8. Conformité : légale.

### I7. Classe fausse par géométrie de la tête (gratuit) 
1. Idée : confusion a priori C[y,c] ∝ exp(cos(w_y,w_c)/τ) avec w = lignes de head.weight (100x768), combiné en log avec la 2e classe d'orig sur les vues et les prototypes de classe (moyenne des features finales par pseudo-classe) : log t = a·log p̃_vues + b·log C + c·log voisinage kNN.
2. Mécanisme : sans donnée, structure sémantique de classes déjà dans les poids.
3. Preuve : [HYPOTHÈSE] (corrélation entre cosinus des lignes de tête et confusions réelles bien connue dans les réseaux entraînés [PAPIER mém.]).
4. Gain : +0,01 à +0,03 de pred.
5. Faisabilité : 45 min ; 0 GPU.
6. Risque : faible. Alerte : r_w mesuré au simulateur < 0,2.
7. Test 15 min : au simulateur, rang de la vraie classe fausse du gold dans le classement par C.
8. Conformité : légale.

### I8. Cadrans foldables (température globale, biais et échelle par classe repliés dans la tête) 
1. Idée : logits = W·LN(φ) + b ; remplacer W,b par W/τ, b/τ (ou par échelle diagonale et biais par classe) est une édition EXACTE des poids float32, sans changer l'architecture. Cadran gratuit après entraînement : réglage fin des moments sans réentraîner, correction du décalage de prior de classes (gold vu 93 % des données), et famille de sondes pour I2 en secondes.
2. Mécanisme : transforme (m*, s*) en un réglage postérieur à coût nul. Limite : τ global agit aussi sur le retain (KL) : à garder proche de 1 (≤ 1,3) et pour de petites corrections.
3. Preuve : algébrique [MESURÉ-toy : τ global seul = 0,45 ; ne remplace pas I1].
4. Gain : +0,01 à +0,05 ; sécurise I2.
5. Faisabilité : 30 min ; vérifier que `norm`/`fc_norm` n'est pas touché, test avant/après repli : logits identiques à 1e-4 près.
6. Risque : faible. 7. Test 10 min : repli puis relecture strict=True, écart de sorties.
8. Conformité : légale.

### I9. Ancres « mosaïques alignées sur la grille 16 px » + MSE de logits 
1. Idée : retain proxy sans données externes : images composées en échangeant des PATCHS 16x16 (grille ViT) entre images d'oubli de pseudo-classes différentes (CutMix aligné), ainsi que flou fort/cutout ; cible = logits d'orig ; perte = MSE de logits + KL T=2 (MSE préserve mieux le classement que le KL seul). Les jetons restent des jetons naturels, contrairement au bruit (SCAR le trouve mauvais [PAPIER mém. arXiv 2404.12922]).
2. Mécanisme : borne la dérive de la fonction autour des images d'oubli sans y réapprendre la mémorisation (mosaïque : plus de motif propre à une image).
3. Preuve : [HYPOTHÈSE] ; [PAPIER mém.] CutMix arXiv 1905.04899 pour le mécanisme de composition.
4. Gain : réduit la fuite retain de 30-50 % : +0,02 à +0,06 de retain.
5. Faisabilité : 1,5 h ; faible GPU (précalcul des cibles d'orig).
6. Risque : ancrer trop fort bloque le forget. Alerte : forget plafonne quand λ monte.
7. Test 30 min : sim ; courbe retain-KL contre forget pour λ ∈ {0 ; 1 ; 5 ; 20}.
8. Conformité : légale (images dérivées du forget set seul).

### I10. Cinq-six graines + soupe de poids + EMA 
1. Idée : K=3 à 5 élèves (graines et variantes de cibles différentes) partant d'orig avec petit lr restent dans la même vallée ; leur moyenne de poids ≈ moyenne d'ensemble [PAPIER mém.] (Model soups, arXiv 2203.05482 ; WiSE-FT, arXiv 2109.01903 pour l'interpolation orig-ajusté). Imite la « moyenne de gold » et réduit la variance des cibles bruitées. EMA des poids pendant chaque entraînement.
2. Mécanisme : [MESURÉ-toy] la moyenne de logits donne pred 0,99 (contre 0,85 pour un seul) : la moyenne d'estimateurs bruités réduit la KL résiduelle.
3. Preuve : [MESURÉ-toy] pour la moyenne ; [PAPIER mém.] pour la linéarité de mode proche d'une init commune.
4. Gain : +0,01 à +0,04.
5. Faisabilité : 1 h en plus ; 4 GPU un nœud 10 min.
6. Risque : le moment de la soupe ≠ moyenne des moments (l'écart-type baisse). Alerte : après soupe mean/std hors ±10 % → recaler par I8.
7. Test 30 min : 3 élèves, soupe vs meilleur élève sur proxy score_hat.
8. Conformité : légale.

### I11. Diagnostic « forget retenu en validation » (K-fold de la généralisation du forget) 
1. Idée : entraîner sur 90 % des images d'oubli et mesurer sur les 10 % restants (et leurs copies) l'écart avec la cible : donne la capacité du modèle à généraliser son oubli à des images/copies JAMAIS entraînées, soit le proxy le plus fidèle de G2. Modèle final : réentraîné sur 100 %.
2. Mécanisme : pas de labels, pas de gold : c'est la seule validation locale utilisable ; sélection de l'arrêt précoce et du nombre d'époques.
3. Preuve : [HYPOTHÈSE].
4. Gain : +0,03 à +0,08 par meilleur arrêt (surtout G2).
5. Faisabilité : 1 h ; coût x1 d'un essai.
6. Risque : faible. Alerte : écart(train, hors-fold) > 0,3 en loss → mémorisation de cibles par pixel : renforcer I4.
7. Test : dès le premier essai ViT-B.
8. Conformité : légale.

### I12. Simulateur local avec la formule exacte + ligue des estimateurs 
1. Idée : celle de t3-unlearning.md (CIFAR-100, ResNet-18/ViT-tiny, orig + 4 gold + gold_test) mais utilisé pour 4 décisions précises : (i) Spearman de chaque signal de dureté (I5, I6, I7) avec la perte réelle du gold ; (ii) coût de fuite retain (contrôle=membres vs non-membres) selon λ, lr, profondeur ; (iii) simulation d'altérations A/B (I4) ; (iv) vérification que 3 sondes suffisent pour I2 (bruit public simulé).
2. Mécanisme : on obtient pour la première fois une vérité terrain du gold ; le toy ci-dessus ne remplace pas cela.
3. Preuve : [HYPOTHÈSE sur le transfert] ; l'objet est classé à travers.
4. Gain : indirect mais décisif : évite 2 à 3 impasses ; +0,03 à +0,1 de score.
5. Faisabilité : 2 à 3 h de code (une personne) ; 3 à 4 h-GPU ; dépend d'un accès réseau pour CIFAR-100 sur le login (non vérifié).
6. Risque : le simulateur diverge du vrai ViT-B (32 px, petit). Alerte : orig du simulateur n'atteint pas train ≈ 100 % / test 65-78 %.
7. Test 30 min : métrique validée : orig ≈ 0 ; gold_test ≈ plafond.
8. Conformité : légale.

## 3. Idées plus courtes
- **I13. Règle de décision sous incertitude (détail)** : L1 pour mean_std → médiane du prior pour (m,s) ; KL pour pred → moyenne a posteriori pour la cible T=2. Ne pas élargir ε par prudence : choisir la médiane. [HYPOTHÈSE sur l'effet, gain +0,02 à +0,04, 20 min.]
- **I14. Identification du jeu public et orig comme oracle de gold sur non-membres (détail à fort levier)** : un vrai non-membre étiqueté de la même distribution (test d'un jeu public) évalué par orig donne a_g, m*, s* à quelques % (orig ≈ gold hors entraînement, le gold ayant vu 93 % des données). Test 10 min : 40 images affichées, 100 classes, tailles. Si identifié : demander aux organisateurs si le jeu public peut servir en MESURE seulement (statistiques de loss de orig), jamais en ancre ni en entraînement. [Gain +0,05 à +0,2 si exact ; P(le jeu est public et identifiable) ≈ 25 % ; conformité douteuse : question à poser avant tout.]
- **I15. Interpolation orig→élève et échelle de pas (WiSE-FT)** : θ = orig + α(θ_u − orig), α ∈ {0,5 ; 0,7 ; 0,85 ; 1,0 ; 1,2}. Calculer en local pour chaque α : moments de la loss (pseudo-labels) et accord sur ancres ; c'est un cadran de plus pour (m,s) ET une échelle de sécurité (porte). α > 1 (extrapolation) possible si l'ancre sous-réalise le forget. Gain +0,02 à +0,05 ; 20 min ; légale.
- **I16. Échelle de qualité de l'exécution (détail format)** : évaluation locale = `predict()` du template (fp16 autocast, entrées [-1,1]) ; entraînement en bf16 avec poids fp32 ; contrôle logits fp16 finis ; clés exactes `model.pt` (pas de `module.`, `_orig_mod.`), tout en float32, `weights_only=True` relu ; `verify_submission.py` avant chaque envoi. Gain : évite un 0 (P ≈ 5 à 10 % sans contrôle). 
- **I17. Ancrage dans l'espace des poids (L2-SP, derniers blocs, lr par couche)** : régularisation « loin d'orig coûte » [PAPIER mém.] arXiv 1802.01483 ; combine avec I9. Gain +0,01 à +0,03.
- **I18. Pseudo-gold par bruit puis réparation sur images auto-générées (forget seul)** : bruit sur blocs 6-11 + réinit tête, réparation par distillation d'orig sur mosaïques (I9) et images inversées (DeepInversion arXiv 1912.08795 [PAPIER mém.]). Toy : valeur d'information de pred faible (+0,03 à +0,07) donc rang bas ; seulement si le simulateur donne > 0,05 de forget. Coût 4 h réel, 4 h-GPU.
- **I19. Élèves distillés sur images externes (si autorisées)** : R-B du rapport précédent ; même constat sur la valeur d'information ; en plus conformité douteuse (énoncé : « the unlearning method has to work from the original model and the forget set alone »). À ne PAS faire sans OK explicite.
- **I20. Édition locale de poids (SalUn : masque de saillance sur gradient du forget ; tête + dernier bloc seulement)** : [PAPIER mém.] arXiv 2310.12508 ; réduit la fuite mais limite l'oubli de mémorisation répartie ; bon filet « retain élevé, forget modeste » ; gain +0,0 à +0,03 en complément.
- **I21. Prior de classes (décalage d'effectif)** : histogramme des pseudo-classes de F vs uniforme ; le gold a vu 93 % des données, biais de log(n_c/(n_c−f_c)) par classe sur les logits ; repli par I8. Gain ≤ +0,01.
- **I22. Empilement appris sur le simulateur** : apprendre p_gold = f(p̃_vues, sonde CV, C, dureté) avec la vérité du simulateur, appliquer au réel ; élégant mais transfert douteux ; gain +0,01 à +0,04 ; 3 h ; P=2/5.
- **I23. Newton/influence sur la tête** (suppression certifiée, Guo et al. arXiv 1911.03030 [PAPIER mém.]) : mise à jour fermée θ += H⁻¹Σ∇ℓ_i, Hessien estimé sur F ; ne traite pas la mémorisation du corps du ViT ; utile seulement comme générateur de cibles « LOO » (rang bas).

## 4. Classement (G gain, P probabilité, F faisabilité en temps réel, C coût GPU : 5 = bon ; O originalité)
Composite = (G × P) / E, E = ((6−F) + (6−C))/2 (effort pondéré temps réel et GPU ; E ∈ [1,5]). G : 1 ≤ +0,01 ; 2 : +0,02-0,04 ; 3 : +0,05-0,09 ; 4 : +0,10-0,19 ; 5 : ≥ +0,20 de score final. P : 1 = <20 %, 5 = >80 %.
| Rang | Idée | G | P | F | C | O | E | Composite |
|---|---|---|---|---|---|---|---|---|
| 1 | I2 cadran (m*, s*) en tente, 6 sondes max | 4 | 4 | 5 | 5 | 4 | 1,0 | 16,0 |
| 2 | I15 interpolation α + échelle de sécurité | 3 | 5 | 5 | 5 | 2 | 1,0 | 15,0 |
| 3 | I8 cadrans foldables dans la tête | 3 | 4 | 5 | 5 | 3 | 1,0 | 12,0 |
| 4 | I16 contrôle d'exécution/format | 2 | 5 | 5 | 5 | 1 | 1,0 | 10,0 |
| 5 | I14 identification du jeu public (mesure seule) | 5 | 2 | 5 | 5 | 3 | 1,0 | 10,0 |
| 6 | I4 invariance sur copies altérées | 4 | 4 | 4 | 4 | 2 | 1,75 | 9,1 |
| 7 | I9 mosaïques alignées + MSE logits | 3 | 4 | 4 | 5 | 3 | 1,5 | 8,0 |
| 8 | I13 règle médiane/moyenne a posteriori | 2 | 4 | 5 | 5 | 3 | 1,0 | 8,0 |
| 9 | I1 cibles par quantiles (QA-MM) | 5 | 3 | 3 | 5 | 4 | 2,0 | 7,5 |
| 10 | I11 diagnostic K-fold du forget | 3 | 4 | 4 | 4 | 3 | 1,75 | 6,9 |
| 11 | I3 exposant de netteté par image | 3 | 3 | 4 | 5 | 4 | 1,5 | 6,0 |
| 12 | I5 sévérité calibrée (vues) | 3 | 3 | 4 | 5 | 3 | 1,5 | 6,0 |
| 12b | I7 classe fausse par géométrie de tête | 2 | 3 | 5 | 5 | 4 | 1,0 | 6,0 |
| 13 | I12 simulateur exact + ligue | 4 | 4 | 2 | 3 | 2 | 3,5 | 4,6 |
| 14 | I10 soupe + EMA | 2 | 4 | 4 | 4 | 2 | 2,0 | 4,0 |
| 15 | I19 élèves externes (si autorisés) | 4 | 3 | 2 | 3 | 2 | 3,5 | 3,4 |
| 16 | I6 sondes CV par profondeur | 2 | 3 | 3 | 5 | 4 | 2,0 | 3,0 |
| 17 | I17 L2-SP / lr par couche | 2 | 3 | 4 | 4 | 2 | 2,0 | 3,0 |
| 19 | I21 prior de classes | 1 | 3 | 4 | 5 | 3 | 1,5 | 2,0 |
| 20 | I20 SalUn / édition locale | 2 | 2 | 3 | 4 | 2 | 2,5 | 1,6 |
| 21 | I18 bruit-réparation sur synthétique | 3 | 2 | 2 | 2 | 3 | 4,0 | 1,5 |
| 22 | I22 empilement appris | 2 | 2 | 2 | 4 | 5 | 3,0 | 1,3 |
| 23 | I23 Newton sur la tête | 1 | 2 | 3 | 5 | 4 | 2,0 | 1,0 |
Notes : (1) classement par composite décroissant. (2) Le composite classe haut les « multiplicateurs » bon marché (cadrans, α) et bas le cœur I1 parce qu'on divise par l'effort ; sans I1 il n'y a rien à régler : I1, I4, I9, I11 forment le noyau obligatoire, I2/I15/I8 les multiplicateurs. (3) Au moins 6 « détails astucieux » : I2, I3, I5, I7, I8, I13, I14, I15, I16, I21 (réutilisation de la métrique, a priori de structure, post-traitements).

## 5. Portefeuille en trois étages (heures réelles ; 3 personnes ; plusieurs nœuds de 4 A100)
### Étage 0 : base sûre (< 2 h, objectif : un score public non nul avant ~19h30)
- Pipeline : charger orig + 3500 images, afficher 40 images, histogramme des pseudo-classes, confiance moyenne, `verify_submission.py` (I16), `predict()` exact. 45 min.
- Modèle S0 : R-A minimal : cibles tempérées simples (T cible 3 à 4, mélange de 8 vues, pas encore QA-MM) + ancre KL sur mosaïques (I9), derniers blocs 6-11, lr 5e-6, 3 époques, puis interpolation α = 0,5 (I15). Soumission : valide, score > 0 attendu ~0,1 à 0,3. Critères : accord argmax sur ancres ≥ 97 %, KL T2 ≤ 0,05.
- Go/no-go 19h30 : S0 reçu et > 0 → étage 1. S0 = 0 → diagnostiquer porte ou format AVANT tout autre travail (α = 0,3, vérifier clés, dtype).
### Étage 1 : méthode principale (2 à 9 h, noyau obligatoire)
- QA-MM (I1) avec famille de cibles paramétrée par (m,s), invariance (I4), mosaïques/MSE (I9), diagnostic K-fold (I11), simulateur en parallèle (I12) tenu par une 2e personne.
- Prior central choisi par I13 : (m0, s0) = (1,1 ; 1,9) avec tolérance ±35 % (HYPOTHÈSE, à remplacer par I14 si le jeu est identifié).
- Go/no-go 22h30 : moments obtenus à ±10 % de la cible, accord ≥ 97 %, fuite KL ≤ 0,1 : sinon baisser lr/époques, renforcer ancres.
- Go/no-go 00h00 : S1 (f = 1,0) public > S0 + 0,1 ; sinon diagnostiquer pred (cibles trop lisses) ou retain (fuite).
### Étage 2 : paris à haut rendement (parallèles, sans bloquer l'étage 1)
- Pari A : sondes en tente I2 : 3 sondes f ∈ {0,85 ; 1,0 ; 1,15}, 3 sondes sur s/m, 1 sur le décalage altéré ; 00h00 à 04h00 ; fin du cadran à 04h00, plus d'ajustement au public ensuite. Conditionné à la confirmation (§8) et au nombre de soumissions autorisées.
- Pari B : I14 (jeu public) : 10 min au début ; si oui, question aux organisateurs ; s'ils refusent, abandon.
- Pari C : soupe de 3 graines + EMA (I10) à 04h00-07h00 ; seulement si le simulateur montre un gain.
- Reporté sauf signal : I18, I19, I22, I23.
- 07h00-09h00 : modèle final : cadrans f*, α*, soupe ; `verify_submission.py`, accord ≥ 97 %, KL ≤ 0,08 ; 09h30 : soumission finale ; 10h30 : soumission de secours = la plus conservatrice déjà validée ; 12h00 : code gelé.
Points d'arrêt : ne rien régler après 04h00 sur le public ; prendre le plateau, pas le pic. Repos : T3 consomme peu de GPU, mais le cadran exige quelqu'un éveillé à 02h00-04h00 pour lire les scores.

## 6. Expériences à lancer en premier (par ordre ; chacune ≤ 30 min)
1. (Humain, 10 min) Afficher 40 images + histogramme de pseudo-classes + confiance moyenne d'orig : jeu public ? flou/net ? soutient I14 et la forme de la loi de pertes.
2. (30 min, 1 GPU) Run S0 + mesures : moments de pertes pseudo-étiquetées avant/après, accord, KL. Décide si R-A est stable.
3. (30 min) Validation de la métrique au simulateur : orig ≈ 0 ; gold_test = plafond ; formule identique au template.
4. (30 min, simulateur) Spearman de chaque signal de dureté avec la perte réelle du gold (I5/I6/I7) ; élimine les signaux sans valeur.
5. (30 min, simulateur) Tente : 3 sondes simulées suffisent-elles à retrouver f* à ±5 % avec bruit public 0,02 ? (valide I2).
6. (30 min, simulateur) Fuite : KL(orig||sub) sur contrôle=membres vs non-membres après entraînement (valide le plafond de retain).
7. (30 min, simulateur) Altérations A/B (valide I4).
8. (30 min, ViT-B) Capacité d'ajustement : moments obtenus vs cible QA-MM après 3 époques (valide I1).
9. Une fois les nombres de soumissions et la sélection finale connus (question aux organisateurs) : fixer le plan de sondes.

## 7. Hypothèses non vérifiées dont dépend le plan
1. Le toy reflète la structure : pred plafonnée par l'aplatissement, mean_std = tente en L1 (algèbre sûre pour la forme ; valeurs numériques [MESURÉ-toy] seulement).
2. (m*, s*) du gold ∈ ranges de §2 : m* 0,8-2,0, s* 1,5-3,0, a_g 70-85 % ; gold entraîné de zéro avec la même recette (indices : LayerScale 1e-4, énoncé).
3. Le nombre de soumissions T3 permet 6 à 8 sondes et le classement final n'est pas basé sur la seule meilleure soumission publique (à demander).
4. Les copies altérées sont « légères » (énoncé) et appartiennent à une famille classique ; leur moments gold diffèrent peu de G1 (+10-20 %).
5. Nature du contrôle inconnue : non-membres de orig (coût de fuite nul) ou membres (coût 0,05-0,15).
6. d_orig est grand (orig mémorise) : sinon la tente est plus étroite.
7. L'accord argmax sur mosaïques/vues reflète la précision sur le jeu de validation caché (porte) : faux si le modèle apprend un biais sur la distribution propre ; marge : accord ≥ 97 %, α ≤ 0,85 pour le premier envoi.
8. Pas d'accès Internet depuis les nœuds de calcul ; CIFAR-100 via le login non vérifié.
9. Calibrage de dureté (I5/I6/I7) transfère du simulateur au réel : à tester, pas à supposer.
10. Je n'ai relu aucun papier cité de mémoire (SCRUB, SalUn, NPO, soupe, WiSE-FT, L2-SP, Papyan, Maini, Guo, DeepInversion, CutMix, SCAR) : seuls les abstracts du rapport précédent ont été lus. Rien de leurs chiffres n'est utilisé.

## 8. Conformité
- Toutes les idées utilisent seulement orig, forget_images.npz, des dérivés de ces images et un simulateur local sur un jeu public de benchmark (CIFAR-100, à confirmer permis ; il ne sert pas à l'entraînement du modèle soumis).
- **I2 (sondes) est la seule zone grise.** Ce que je recommande : un hyperparamètre scalaire de la méthode (échelle f des moments cibles), au plus 6 à 7 soumissions au total, sans construire de modèle de sonde pour deviner des images, des labels ou des poids du gold ; équivalent d'un réglage de température sur le classement public. Je n'ai pas de règle écrite qui interdise ou autorise l'usage du classement de cette manière. À valider auprès des organisateurs avant le pari A ; en cas de doute, rester sur I13 (médiane du prior) et I15 (α) seulement.
- **I14 / I19 (données externes)** : l'énoncé demande une méthode qui travaille « from the original model and the forget set alone ». Les images externes servent seulement si les organisateurs répondent oui ; sinon uniquement les dérivés du forget set (I9, I18).
- Ne rien extraire du serveur ; aucun appel à un endpoint depuis ce rapport.

## 9. Sources
- Énoncé, gabarit : EAC/reports/sources/Unlearning.txt, t3/task_template.py. Rapports : research/t3-unlearning.md, EAC/reports/t3-unlearning-etat-de-lart.md, strategy.md, submissions-ops.md.
- Toy : research/toy/t3toy.py (calibrage 1), t3toy3.py (quantiles et sensibilité) ; sorties citées en §0-§1.
- [PAPIER mém.] SCRUB https://arxiv.org/abs/2302.09880 ; SalUn https://arxiv.org/abs/2310.12508 ; NPO https://arxiv.org/abs/2404.05868 ; Zero-shot https://arxiv.org/abs/2201.05629 ; SCAR https://arxiv.org/abs/2404.12922 ; NeurIPS 2023 https://arxiv.org/abs/2406.09073 ; Model soups https://arxiv.org/abs/2203.05482 ; WiSE-FT https://arxiv.org/abs/2109.01903 ; L2-SP https://arxiv.org/abs/1802.01483 ; Maini et al. https://arxiv.org/abs/2307.09542 ; Papyan et al. https://arxiv.org/abs/2008.08186 ; Guo et al. https://arxiv.org/abs/1911.03030 ; DeepInversion https://arxiv.org/abs/1912.08795 ; CutMix https://arxiv.org/abs/1905.04899.
