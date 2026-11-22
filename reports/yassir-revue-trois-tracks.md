# MasterChickenHack · trois décisions pour le prochain tour

21 novembre, relevés publics du soir (Paris). Adem pilote les envois T1/T3. Titouan pilote la collecte et l’envoi T2. Ami a confirmé la coordination dans GitHub, la validation par un deuxième agent et un seul expéditeur annoncé par créneau. [Bus d’équipe](https://github.com/adembnn/EAC/issues/2)

| Tâche | Équipe | Tête publique | Décision |
|---|---:|---:|---|
| T1 | 2e · 0,738431 | Eugenia · 0,751079 | Comparer sur les mêmes images à la soumission 1540. Il manque 0,01265 au public pour rejoindre la tête. |
| T2 | Aucun score affiché | 3V1B · 1,00 | Donner à Titouan une estimation de stabilité sur ses journaux existants. Le général affiche encore « 2 of 3 ». |
| T3 | 5e · 0,199790 | Fable 4.2 · 0,315994 | Tester séparément le choix des images et le choix de la classe cible. Le retard public est de 0,11620. |

Sources du relevé et des recettes précédentes. [Classement](http://34.147.229.55/leaderboard_page) · [1540, candidat 5](https://github.com/adembnn/EAC/issues/3#issuecomment-7068396043) · [1624, recette à 0,1962](https://github.com/adembnn/EAC/issues/5#issuecomment-7068812904)

## T1 · ce que rapporte SAM sur les cartes réellement partagées

L’assemblage d’Adem sur images réelles combine cinq plis, formes SAM et boîtes du décodeur. Son candidat 5 a gagné **+0,0314 sur réserve et +0,0325 au public**. Le candidat 6 ajoutait +0,0043 sur réserve. Notre propre soumission à 0,6369 et notre StegaStamp à 0,7444 sur réserve ont été comparés à un ancien candidat. Le StegaStamp du candidat 5 atteint déjà **0,8429** sur sa réserve officielle. La comparaison nominative des images reste nécessaire pour juger une fusion. Les nouvelles cartes PixelSeal sont disponibles sur Nebius, dans le répertoire partagé de h200b. [Vérification du candidat 5](https://github.com/adembnn/EAC/issues/3#issuecomment-7068347851) · [Candidat 6](https://github.com/adembnn/EAC/issues/3#issuecomment-7068398512) · [Essai de Yassir](yassir-t1-etat-des-lieux.md)

**Test indépendant sur CPU.** Comparaison de deux cartes au seuil fixe de zéro, alignées par nom avec les masques officiels. Réserve = indice global dans les métadonnées divisible par 5. IC95 par bootstrap apparié des images (20 000 tirages, graine 20261121). « Hors réserve » décrit les autres images de développement.

| PixelSeal | n | Carte de base | Carte v3sam | Écart | IC95 de l’écart |
|---|---:|---:|---:|---:|---|
| Réserve entière | 30 | 0,63625 | 0,64111 | +0,00486 | [−0,00254 ; +0,01501] |
| Pleines en réserve | 19 | 0,65286 | 0,65217 | −0,00069 | [−0,00247 ; +0,00067] |
| Partielles en réserve | 11 | 0,60755 | 0,62199 | +0,01444 | [−0,00562 ; +0,04136] |
| Hors réserve | 140 | 0,66272 | 0,67990 | +0,01719 | [+0,00944 ; +0,02615] |

Même en choisissant après coup le meilleur des deux masques pour chaque image, le plafond de gain sur ces 30 images vaut **+0,00654 PixelSeal**. Cette paire seule ne justifie pas de créneau. À faire par l’assemblage : joindre par nom la sortie honnête du candidat 5 et les boîtes TrustMark déjà affinées. Sur les rectangles, recalibrer la marge hors réserve. Sur les partielles PixelSeal, évaluer si la carte SAM améliore encore les images que le candidat 5 rate. Pour un nouveau synthétique PixelSeal, l’article motive une ablation de l’atténuation JND à résolution native sur les zones peu texturées. Garder les mêmes photos, masques et graines dans les deux conditions. [Rectangles mesurés](https://github.com/adembnn/EAC/issues/3#issuecomment-7068813991) · [Pixel Seal](https://arxiv.org/abs/2512.16874)

Reproduction avec le kit officiel importable et les deux archives de `h200b:/tmp/eac-t1-partage/`. Empreintes SHA-256 abrégées, dans l’ordre : `12ad1413c5af`, `91d26647f1d3`.

```bash
python tracks/t1-watermark/experiments/compare_pixelseal_maps.py \
  train.zip pred_train_pixelseal.npz pred_train_pixelseal_v3sam.npz
```

## T2 · valider le premier CSV de Titouan avec les réponses déjà collectées

Titouan a identifié trois familles candidates de onze endpoints, ajouté les jumeaux Llama et monté le bloc de 25 réponses par endpoint. Adem a relu les 225 réponses de référence. Sa règle de longueur et son lecteur LoRA donnent **0,734 de MCC modèle sur les références**. Le résultat des références est sensible au seuil (0,524 au seuil 3). La distribution des prompts et le décodage changent aussi la statistique. [Collecte](https://github.com/adembnn/EAC/issues/4#issuecomment-7068905489) · [Lecteur et seuils](https://github.com/adembnn/EAC/issues/4#issuecomment-7069052993)

**Contrôle proposé à Titouan.** Figer le lecteur, la température de référence et la règle avant de regarder les endpoints de test. Rééchantillonner les *identifiants de prompts* en bloc pour tous les endpoints ensemble, recalculer familles et partitions, puis lister les paires dont la décision change souvent. Vérifier que l’adaptateur appris et sa mesure utilisent des prompts disjoints pour les évaluations de référence. Avec les répétitions d’un même prompt, un test MMD peut éclairer un désaccord de distribution ; température et prompt système changent aussi la distribution d’un même modèle selon les règles du concours. Rapporter la stabilité de la partition et le MCC sur références. À famille parfaite, le palier 0,85 exige un MCC modèle de 0,8333. La nouvelle tête publique, 3V1B à 1,00, exige les deux MCC à 1 sur les paires évaluées. Ce calcul n’est pas une prévision sur le privé. [Model Equality Testing](https://arxiv.org/abs/2410.20247) · [Protocole équipe](https://github.com/adembnn/EAC/issues/4)

## T3 · ce que mesure réellement le vote de Titouan

La recette 1624 d’Adem entraîne quatre blocs et la tête, avec 30 % d’images choisies par instabilité, une loss cible de 8–14 sur ces images et une ancre sur mélanges. Elle atteint **0,1962 public**. À fraction égale, passer la loss à 12–18 a donné **0,199790** sur la soumission 1688, soit +0,00364. Cette comparaison soutient la priorité donnée au choix des images et des classes. [Reçu 1688](https://github.com/adembnn/EAC/issues/5#issuecomment-7069125122) Notre envoi 1180 avait atteint **0,003862**. Le chiffre **0,142372** était une entropie croisée locale sur pseudo-étiquettes. Les essais de redistillation d’Adem montrent combien une hausse de l’oubli peut coûter sur les images non vues. [Recette 1624](https://github.com/adembnn/EAC/issues/5#issuecomment-7068825664) · [Notre reçu](yassir-t3-h200.md) · [Redistillation](https://github.com/adembnn/EAC/issues/5#issuecomment-7069022521)

**Test indépendant sur les 3 500 lignes de Titouan.** Sur les 1 050 images de tête par `consensus`, **605** n’ont aucune classe soutenue par deux voix. Les **605/605** reçoivent le second choix d’`orig`. Au total **943/1 050** cibles coïncident avec ce second choix. Seulement **107** en diffèrent. Sur les 605 sans accord, la sonde hors pli propose une autre classe que l’original et le vote dans **320 cas**. Le vote actuel fournit donc peu de contraste pour tester l’effet du *choix de classe*. [Données, code et version exacte](https://github.com/adembnn/EAC/tree/008506b35b89172bfa4d1c43ef86c8c14eb719fd/tracks/t3-unlearning/critique) · [Expériences E1/E5 en cours](https://github.com/adembnn/EAC/issues/5#issuecomment-7069030862)

**Comparaison proposée à Titouan et Adem.** Réutiliser les cibles et checkpoints déjà en cours. À sélection d’images, graine, hauteur de loss et ancre identiques, conserver les 445 votes majoritaires et changer seulement les 320 cibles ambiguës pour lesquelles la sonde hors pli propose une classe différente. Si vous avez déjà cette variante, réutiliser son journal. Avant un entraînement supplémentaire, comparer les cibles par ID avec E1, E5 et PGo. Vérifier aussi que le train et la réserve ont les mêmes proportions d’images modifiées. Pour décider, mesurer moments de loss sur les exactes et des copies tenues à l’écart, accord et KL de rétention sur un lot indépendant, et comparer au dernier candidat accepté. Les vraies étiquettes de contrôle et le modèle gold sont cachés. Le classement reste la mesure du score public.

Le travail sur la profondeur de prédiction justifie de stratifier les images selon la difficulté. SCRUB motive la sélection du checkpoint sous contrainte de rétention ; ici le jeu de rétention officiel reste caché, donc toute mesure de remplacement doit être nommée et comparée sur les mêmes images. [Prediction depth](https://arxiv.org/abs/2106.09647) · [SCRUB](https://arxiv.org/abs/2302.09880)

## Calcul et envois

Les cinq H200 ont répondu via les alias d’Ami. Instantané vers 21 h 52, en pourcentage d’utilisation GPU pour les nœuds 1 à 5 : **24, 100, 100, 100, 100**. Le nœud 1 avait déjà 8 951 MiB alloués. Un relevé ponctuel laisse ouverte la question du prochain créneau. Les deux tests ci-dessus ont tourné sur CPU. Annoncer toute réservation dans #2 ; la règle d’Adem en T1 demande +0,02 sur la réserve des trois filigranes, IC95 apparié excluant zéro, et feu vert indépendant. En T3, partager la comparaison au dernier candidat dans #5 et nommer l’expéditeur. Les slides indiquent 30 min pour le **CSV T2**, et 100 **réponses par endpoint** pour sa collecte. Le registre d’équipe indique 10 min par tâche pour T1/T3. [Décisions](https://github.com/adembnn/EAC/blob/a5692ea/CLAUDE.md) · [File T3](https://github.com/adembnn/EAC/issues/5)
