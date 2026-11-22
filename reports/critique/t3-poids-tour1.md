# Critique T3, session t3-poids, tour 1 (21/11, ~19 h 20)

Décision débloquée : quel fichier « poids » soumettre après `NEGce_all_a2.pt`, et avec quel instrument trancher entre candidats.
Critique en lecture seule : aucun calcul GPU, aucun fichier de code modifié. Une seule mesure ajoutée, sur CPU (lecture des poids, voir §4).
Sources : `ssh h200 'cat ~/eac/t3/t3-poids/logs/run1.log run2.log'` (run3 à run6 vides à 18 h 20 UTC, `loc.py ablate` en cours),
code `tracks/t3-unlearning/t3-poids/*.py`, `~/eac/t3/t3-methodes/code/{common,dash,cache,mk_tgt}.py` lus sur la H200, issue #5.

Termes. **Vecteur de tâche** : différence de poids `plus − orig` après un sur-entraînement. **Membre** : image vue à l'entraînement d'orig.
**Saturé** : probabilité de la classe prédite proche de 1. **KL** : divergence de Kullback-Leibler entre deux distributions de classes, ici à température 2.
**Réserve c** : le tiers des 3 500 images non utilisé pour apprendre le vecteur (`split.pt`, découpe 40/40/20 par classe prédite, graine 20261121).

## Les trois critiques les plus graves
1. **Le candidat livré est dominé par un autre déjà mesuré.** `NEGce_hi@4` (vecteur limité aux blocs 6 à 11 + norme + tête, α=4, run2.log) bat `NEGce@2`
   (tout le modèle, α=2, run1.log) sur presque toutes les colonnes, à vecteur appris sur a+b identique :
   | mesuré (run1/run2.log) | NEGce@2 | NEGce_hi@4 |
   |---|---|---|
   | exactes a+b : loss moy / écart-type / classes changées | 0,211 / 1,465 / 3,2 % | 0,356 / 2,110 / 4,6 % |
   | KL(orig‖sub) sur a+b ; probabilité max à T=2 | 0,124 ; 0,934 | 0,184 ; 0,938 |
   | réserve c exactes : KL | 0,0310 | 0,0198 |
   | réserve c flou fort : précision (orig 0,867) / KL | 0,847 / 0,066 | 0,842 / 0,039 |
   | réserve c couleur forte : précision (orig 0,523) / KL | 0,453 / 0,117 | 0,487 / 0,067 |
   | mélanges : KL | 0,163 | 0,108 |
   Plus d'oubli et moitié moins de dérive. `NEGce_all_a2.pt` (tout le modèle) n'a été choisi que parce qu'il a été écrit avant run2b.
   Réserve (supposé) : un écart-type de 2,11 peut dépasser celui du gold ; au-delà, le score de loss redescend.
2. **Aucune de ces négations ne règle le facteur que l'on croit bloquant.** La probabilité max à T=2 reste 0,919 à 0,938 (orig 0,985) : plus de 90 % des images
   gardent la distribution d'orig, et le score de prédiction est calculé image par image. Augmenter α ne fait que grossir la queue (α=4 : 0,762, mais réserve
   c KL 0,373 et précision couleur forte 0,307 contre 0,523, run1.log). Pousser α est donc une impasse ; il faut une opération qui adoucit toutes les images (§c).
3. **Le vecteur CE + AdamW est probablement surtout du bruit d'optimiseur** (supposé, appuyé par une mesure). AdamW fait des pas d'environ `lr` par poids,
   quelle que soit la taille du gradient. Ici la loss vaut 0,00028 puis 0,00002 (run1.log) : les gradients sont minuscules et de signe instable d'un lot à l'autre.
   108 pas à lr 2e-5 donnent au plus 2,2e-3 par poids si le signe est constant, et environ √108 × 2e-5 = 2,1e-4 s'il est aléatoire.
   Mesuré (CPU, §4) : |delta| moyen 1,2e-4 à 1,5e-4 sur tous les gains LayerScale, du bloc 0 au bloc 11, quelle que soit leur taille. C'est l'ordre de grandeur
   d'une marche aléatoire, pas d'un pas cohérent. Une partie de la dérive sur c peut donc venir de ce bruit. Rien n'a été mesuré sur une deuxième graine.

## a) La réserve c sous-estime-t-elle la dérive sur le contrôle ?
Oui. Pour une petite variation δ des logits, la KL vaut environ ½ δᵀ(diag(p) − ppᵀ)δ / T². Quand p est saturé (0,985 à T=2), la matrice diag(p) − ppᵀ
a des valeurs propres de l'ordre de 1 − p_max ≈ 0,015. Pour une image non saturée (p_max 0,7), elles sont de l'ordre de 0,2. À δ égal, la KL est 10 à 20 fois plus
grande (calcul, pas une mesure). Ce qui est mesuré sur le même candidat NEGce@2 va dans ce sens : KL 0,031 sur c exactes, 0,066 sous flou fort (orig correct à 86,7 %),
0,117 sous couleur forte (orig correct à 52,3 %). Le contrôle caché est fait d'images jamais vues du même domaine. Supposé : orig y est entre ces deux cas,
donc KL de contrôle autour de 0,06 à 0,12 pour NEGce@2, soit 2 à 4 fois la KL de c exactes. Facteur de rétention estimé 0,85 à 0,92, contre 0,966 affiché.
Trois limites de l'instrument actuel :
- les **mélanges** du cache sont faits avec les 3 500 images (`cache.py`, `torch.randint(0, N, …)`). Pour un vecteur appris sur a+b, 2/3 de leur contenu a été traité :
  la KL mélanges mêle oubli et dérive. Pour un candidat appris sur les 3 500 (`NEGce_all_a2`), aucune colonne n'est un contrôle ;
- c ne couvre que les 45 classes du forget set. Le contrôle couvre les 100 classes : la dérive sur les 55 autres n'est mesurée nulle part ;
- la règle CIFAR annoncée dans `dash.py` n'existe pas sur la H200 : `ood_cifar100.npy` est absent du cache, la colonne « mélanges ; CIFAR » n'a qu'une valeur.
Instrument plus fidèle, sans données externes pour entraîner : regrouper toutes les variantes de c (exactes, 4 copies légères, 2 fortes, soit environ 4 900 images),
les ranger par probabilité max d'orig, et tracer la KL et la perte de précision en fonction de cette confiance. On lit ensuite la courbe au niveau de confiance
d'une image jamais vue : celui que mesure t3-nonmembres, ou à défaut l'hypothèse de 85 % de précision. Y ajouter des mélanges faits seulement d'images de c.

## b) La négation CE + AdamW est-elle une montée de gradient ?
Au premier ordre, oui. `plus − orig ≈ −Σ_t lr · P_t · m_t`, avec m_t la moyenne mobile du gradient de la CE et P_t le facteur d'échelle d'Adam ; la négation fait un pas
de montée préconditionné, de longueur α. Le gradient de la CE par image vaut (p − e_y) : il est pondéré par 1 − p_y, donc dominé par les quelques images les moins sûres.
Cela explique la queue lourde. Trois différences avec la montée naïve, aucune décisive pour le score :
- pas de rétroaction : en montée itérée, les images déjà abîmées ont des gradients qui grossissent, jusqu'à l'effondrement. Ici, le pas est fixe et ses gradients viennent d'un modèle plus saturé ;
- la longueur est bornée par α et non par un nombre de pas ;
- la classe fausse choisie est celle qui reçoit le plus de gradient, probablement la deuxième classe d'orig. Le gold choisirait peut-être la même (non vérifié).
Le problème que l'énoncé reproche à la montée naïve est identique ici : la plupart des images sont inchangées et quelques-unes sont très fausses, alors que le gold
modifie la distribution de chaque image. Le score de prédiction, calculé image par image, voit cette différence. La sélectivité ne vient pas de la méthode :
elle vient de la concentration sur la queue (la marge, qui pèse chaque image pareil, devient générique à la classe, cos 0,72 avec le vecteur augmenté, run1.log).

## c) Quelle opération sur les poids adoucirait toutes les images traitées ?
Aucune n'est mesurée. Trois candidates, par ordre de coût :
1. **Somme de vecteurs** : `orig + (PG_T85 − orig) + β · (NEG − orig)`. PG_T85 adoucit tout (probabilité max 0,805, issue #5), NEG ajoute la queue. Aucun entraînement : `dash.py --combo`.
2. **Projection fermée sur la tête** : W' = W(I − βP). P projette sur les k premières directions de l'écart entre les caractéristiques d'une image exacte (CLS après la norme)
   et la moyenne de ses vues fortes. C'est la part propre au membre, retirée de la tête. Le choix est sélectif si ces directions servent peu aux images jamais vues : à mesurer sur c.
3. **Vecteur moyen sur plusieurs graines** : si la critique 3 est juste, faire la moyenne de k vecteurs réduit le bruit d'un facteur √k et garde le signal. Ensuite, comparer
   à sélectivité égale avec `cet` à T=4 (run5 en attente). Prédiction : `cet` sera générique comme la marge, puisque ses poids par image sont presque uniformes.

## d) Bugs et biais dans le code
- Pas de fuite : `tv.py` n'utilise que `S['ab']` ; le découpage est par classe prédite avec une graine fixe ; les copies de c sont tirées indépendamment de l'entraînement.
- Mode train et mode eval : timm crée le modèle sans dropout ni drop-path (`task_template.build_model`, seulement `init_values=1e-4`), donc pas d'écart.
- bf16 à l'entraînement, fp16 au test : à |z| ≈ 12, le pas d'un nombre bf16 vaut 0,06, d'où environ 6 % d'erreur relative sur le poids 1 − p de chaque image. C'est du bruit
  qui s'ajoute à la critique 3. Le eps d'Adam (1e-8) peut aussi peser face à des gradients de 1e-8 : la direction dépend alors d'un réglage arbitraire. Rien n'est faux au test.
- `NEGce_all_a2` est appris avec 54 pas par époque contre 36 pour la version a+b : la norme du delta passe de 1,27 à 1,45 (run1/run2.log). Le même α ne donne donc pas
  le même candidat, et la sélectivité mesurée sur a+b ne se transpose qu'approximativement.
- `loc.py fisher` fait le carré du gradient moyen d'un lot de 8, pas la moyenne des carrés par image. Les termes croisés renforcent ce qui est commun aux images du lot :
  le rapport exact/augmenté penche vers le générique, ce qui affaiblit `ssd.py`. Le Fisher est aussi calculé sur la marge et non sur la log-vraisemblance (choix assumé, à noter).
- LayerScale : `ssd.py` et `unl.py --init` excluent les gains, `tv.py` les entraîne. Mesuré (§4) : la variation relative des gains d'attention est de 1 à 1,4 %, contre 0,2 % pour
  ceux du MLP. À α=4, cela fait 4 à 6 % sur le gain d'une branche entière : c'est une piste pour expliquer la dérive de NEGce@4.
- La « loi 0,019 × (m + s) » est ajustée sur une seule famille, et elle suppose que m et s restent sous ceux du gold. L'appliquer à NEG (s = 1,49, prédiction 0,033) revient à extrapoler.

## e) La porte des 70 % et la rétention
Mesuré sur c, rapport de précision sub/orig : NEGce@2, 0,977 sous flou fort et 0,866 sous couleur forte ; NEGce@4, 0,856 et 0,587 (run1.log).
Les images non saturées perdent le plus. Supposé : si orig a 85 % sur la validation, une perte de 13 % comme sous couleur forte donne 0,74, tout près de la porte.
NEGce@2 et NEGce_hi@4 sont probablement au-dessus. Ne pas dépasser α=3 sur le modèle entier (à α=3, KL de c = 0,121 et précision couleur forte 0,383, run2.log).
La porte reste invérifiable sans images jamais vues : le score réel de `NEGce_all_a2` (envoi annoncé à 19 h 29) servira de point d'étalonnage.

## 4. Mesure ajoutée (CPU, lecture seule)
`ssh h200` puis lecture de `model.pt` et `deltas/tvE_ce.pt` en Python sur CPU. Sortie : `blocks.0.ls1.gamma |gamma| moy 0.009304 |delta| moy 0.000126 rel 0.0116` ;
`blocks.11.ls1.gamma 0.007341 / 0.000124 / 0.0143` ; `blocks.11.ls2.gamma 0.08523 / 0.000154 / 0.0021`. Variation relative par type de poids : qkv.weight 0,0055,
fc1.weight 0,0045, head.weight 0,0064, norm1.weight 0,00017.

## Expériences proposées (5 au plus, par utilité)
1. **Somme PG_T85 + β·NEGce_hi** (β = 0,5 ; 1 ; 1,5), avec le PG appris sans c et le vecteur appris sur a+b, dans `dash.py --fset ab --rset c`.
   Mesure : probabilité max à T=2, m + s, KL de c sous couleur forte. Décision : si la probabilité max est ≤ 0,82, m + s ≥ 2 et la KL couleur forte ≤ celle de PG seul + 0,05,
   on soumet la somme (version apprise sur les 3 500) à la place de toute négation seule. Coût : environ 5 min de GPU.
2. **Remplacer le candidat livré par `NEGce_hi` appris sur les 3 500** (`mkdelta.py blocks tvE_ce_all 6 13`, α = 3 et 4, `--save`). Mesure : chk.py et le tableau.
   Décision : soumettre ce fichier plutôt que `NEGce_all_a2` si l'envoi de 19 h 29 ne dépasse pas 0,03. Coût : environ 3 min.
3. **Bruit ou signal** : `tv.py` sur a+b avec les graines 1 et 2, cosinus entre graines par bloc. Témoin : un vecteur de signes aléatoires de même amplitude par poids (≈1,3e-4).
   Puis négation de la moyenne des 3 graines à α=2. Décision : si le cosinus entre graines est < 0,3 ou si le témoin donne une KL sur c ≥ la moitié de celle de NEG,
   on n'utilise plus que des vecteurs moyennés. Si la moyenne garde l'oubli avec moins de dérive, on la livre. Coût : environ 8 min.
4. **Instrument de rétention par niveau de confiance** : variantes de c groupées par probabilité max d'orig, plus des mélanges faits seulement de c ; courbe de KL et de perte
   de précision pour orig, NEGce@2, NEGce_hi@4 et PG_T85. Ajouter la corrélation, image par image, entre la loss de NEG et 1 − p_y d'orig pour confirmer la queue.
   Décision : remplace la KL de c exactes comme critère de choix entre candidats ; si la courbe dépasse 0,15 au niveau de confiance d'une image jamais vue, on baisse α. Coût : environ 5 min.
5. **Projection de la tête sur l'écart membre** (forme fermée, §c.2, k = 8 à 64, β = 0,5 à 1). Décision : si la probabilité max sur a+b descend à ≤ 0,85 avec une KL couleur forte
   sur c ≤ 0,05, c'est l'opérateur d'adoucissement sélectif qui manque à la famille poids ; on le combine avec NEG. Coût : environ 4 min.
