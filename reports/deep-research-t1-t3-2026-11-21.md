# MasterChickenHack — Deep research sprint T1/T3 (21 Nov 2026, avant le cutoff du 22 Nov 12:00 Paris)

**Statut:** revue bibliographique arXiv + GitHub, recommandations à valider. Aucun job H200, aucune soumission, aucun appel d'endpoint T2. Le dashboard `h200.ami.exchange` ne répond pas depuis cet environnement (DNS indisponible), donc pas de détection fiable des ressources libres. Les clés SSH documentées résident sur les machines de Yassir/NemoClaw, **pas** dans la présente session.

## Sources du challenge et état actuel

- Instructions et règle maker-checker: `CLAUDE.md`, `reports/sources/Watermark_Localization.txt`, `reports/sources/Unlearning.txt`.
- T1 issue [#3](https://github.com/adembnn/EAC/issues/3) : **candidat 22 vérifié et FEU VERT explicite** (commentaire 7070172759), prêt sur h200b: `~/eac/t1/t1-assemblage/submission_candidat_22.pt`; reserve **0.7514** vs dernier envoi reserve **0.7230** (+0.0284, IC95 [+0.012,+0.047]); dernier public annoncé **0.7384**. La réserve n'est PAS un score public.
- T3 [#5](https://github.com/adembnn/EAC/issues/5) : **PG_T85_f70_l814c public 0.3159**; essai proxy "gold_blend55" 0.0804 public, donc la calibration locale du gold ne prédit PAS nécessairement le score final. Sondes `G10_f80`, variantes `PG_RDt24`, `PGw15/w2`, `PGv2/v3` déjà prises par les agents : ne pas dupliquer.
- T2 [#4](https://github.com/adembnn/EAC/issues/4) : public **0.60** sur soumission 1967 (groupement LoRA qwen3b ep17+ep18). L'essai « préfixe assistant + 1 token » est **invalidé** sur ref : le serveur ouvre un nouveau tour assistant (commentaire 7070321437). **Zéro requête T2 ici.**

## Littérature nouvelle vérifiée — ce qui est transférable ou non

| Source | Observation publiée | Adaptation EAC | Limite |
|---|---|---|---|
| [ReliaGuard, 12 nov. 2026](https://arxiv.org/abs/2611.16368) | Reader : *background-contrast evidence refinement* + *native-guided scale reassembly*, features locales et gradients des logits, pour petits contours après éditions | T1 : une petite tête **résiduelle** à 512px sur la carte de l'assemblage, feature pyramid et bords, non un post-traitement de seuil | Leurs images sont **marquées par leur propre writer** : leurs IoU publiées ne sont pas transposables à PixelSeal/TM/SS ; pas de code public vérifié |
| [What Breaks Local Watermarks?, sept. 2026](https://arxiv.org/abs/2609.16832) | 55 transformations ; décodage du payload et localisation ne sont pas équivalents ; in/outpainting casse le signal local | T1 : stratifier sur *pleines réparées*, petits trous, rectangles minuscules ; ne pas s'appuyer sur marge de bits seule | Benchmark, pas recette améliorant directement notre mIoU |
| [VideoSeal/PixelSeal, code Meta](https://github.com/facebookresearch/videoseal) + [papier](https://arxiv.org/abs/2512.16874) | Encodeur/décodeur réel, pattern pixel seal + champs récepteurs | T1 : reprendre features natives *du modèle fine-tuné*, pas CAM du décodeur global déjà négative | Localisation n'est pas native ; risque de non-localité |
| [SALA, 3 nov. 2026](https://arxiv.org/abs/2611.04501) | Alignement des prédictions d'unlearning **échantillon par échantillon**, estimation de distributions membre/non-membre | T3 : ne pas utiliser un unique « wrong teacher » aveuglément ; tester mélange de pseudo-teachers selon **leur accord par image** | Leur SMSE se base sur un **entraînement structuré avec les données retain** ; hors d'atteinte du challenge. Notre mélange n'est PAS une implémentation ou garantie SALA |
| [Forgetting Has Neighbors, mai 2026](https://arxiv.org/abs/2605.31317) | Dégâts de retain concentrés près des forget en espace de features ; targets pseudo incorrectes dégradent les voisins | T3 : mesurer dérive KL par **distance au forget** sur proxy retain indépendant et ancrer prioritairement les plus proches | Leur professeur local utilise de vrais voisins retain ; ici seulement proxy OOD ou images externes autorisées |
| [Do unlearning algorithms learn the right features?, 4 nov. 2026](https://arxiv.org/abs/2611.06681) | Similarité des seules sorties ne garantit pas similarité des features, notamment après adaptation de tête | T3 : tester représentation / dérive sous copies ; éviter les corrections agressives de tête si elles re-mémorisent | Metric EAC ne demande pas post-finetuning ; priorité diagnostique |
| [ReGUn](https://arxiv.org/abs/2603.11210) + [code](https://github.com/jmirlach/ReGUn) | Distillation guidée par jeu de référence disjoint et conditionné par classe | T3 : utiliser les proxies `holdout` strictement disjoints et distinguer restauration sur proxys de non-membres vs oubli | Les références conditionnées exigées par l'article ne sont **pas fournies** |

## Expériences prioritaires, sans dupliquer les agents

### A. T1 — envoyer le **candidat 22** avant une nouvelle exp
**GPU supplémentaire: 0.** Le fichier est déjà validé par maker-checker. Une seule soumission sous la responsabilité de l'orchestrateur, pas besoin d'attendre mes travaux. C'est la décision la plus solide à partir des résultats existants.

### B. T1 — *ReliaGuard-inspired residual native boundary head* (nouvel axe)
- **Objectif** : améliorer l'IoU des trous de pleine <2% et 2–8%, dont l'étape de segmentation manque parfois toute la zone. Les post-traitements morpohologiques/IoU espérée/SAM sans information ont tous échoué (#3 et PR #35) ; il faut un **nouveau canal/features apprises**.
- Données : prédictions hors-pli `f_s1/f_s1a` TrustMark, `f_deca` PixelSeal, `candidat_22` (partage sur `/tmp/eac-t1-partage/`), et feature maps natives des localisateurs fine-tunés. Labels **uniquement train** hors pli ; ne pas utiliser masque du test.
- Architecture (ablation petite) : apprendre `delta_logit = 0.5*tanh(ConvNet([coarse_logit_up, RGB_highpass, |∇coarse|, features_native]))` à résolution 512; restreindre delta par masque de confiance si possible, puis moyenne officielle vers 128. Geler le backbone. BCE+Dice et plus de poids sur régions où la classe minoritaire est <8%. Comparer **logits seuls vs logits+RGB vs logits+native features** au même budget (important pour réfuter le shortcut RGB).
- **Barrière** : aucun gain démontré; préférer `candidat22` si le CV 4 plis et réserve commune ne montrent pas de gain substantiel. **Coût GPU estimatif 0.5–2 h**, dépend de sauvegarde des features. Stop si plis hors réserve ≤ 0 gain ou si rectangle/objets se dégradent.
- Variante déjà codée dans [PR #34](https://github.com/adembnn/EAC/pull/34) : `t1_pair_counterfactual.py` sépare sur paires marquée vs trou rétabli, mais aucune mIoU réelle. Mesurer l'interaction des deux **seulement si** un premier axe donne un gain.

### C. T3 — *per-image target uncertainty* au lieu d'une autre température globale (nouveau code)
- [Script `t3_gated_wrongclass.py`](../tracks/t3-unlearning/experiments/t3_gated_wrongclass.py) : prend deux distributions de pseudo-teachers **indépendants** (votes sous vues dégradées vs prédiction OOF redistill) + classe et masse de classe actuelle. Calcule la divergence JSD **sur les 99 classes fausses** et dose la contribution OOF par image `alpha_i = alpha * exp(-JSD_i / tau)`. **La masse de la vraie classe est conservée exactement à T=1**, donc les objectifs moyens de CE peuvent rester au même niveau ; `q2=normalize(sqrt(q1))` respecte la température 2.
- Tests à comparer : `alpha=0` (**base témoin**), `alpha=.35, mode=constant` (**effet de la mixture**), `alpha=.35, mode=gated, tau=.15` (**effet du gating**), `alpha=.70, gated` (**stress**). Une étiquette source doit rester OOF : aucun enseignant entraîné sur l'image de test de ce bras. Ne pas sélectionner sur les 102 réserves T1 ni sur le public T3.
- Intégration PG-f70 : même sélection d'images, même nombre de pas, même ancre, calibration CE m/s exactes et copies, mesure KL p_gold proxy∥p_sub **à T=2**, utility gate sur proxies indépendants. Vérifier la forme de cible produite par le training, pas seulement le fichier `q`. **Estim. GPU 1–2 h ; risque élevé** (ensemble des golds inconnu).
- Une version brute n'est PAS prête à soumettre. Critère : gain concordant sur au moins 2 simulations de golds hors entraînement avec rétention non dégradée, puis maker-checker.

### D. T3 — *neighbor-weighted retain anchoring* (diagnostic d'abord, GPU réduit)
- Source : *Forgetting Has Neighbors*. Des proxys retain non utilisés dans les ancres PG sont évalués par cosine ViT contre les 3500 forget ; mesurer KL(orig∥PG) aux quantiles top1/5/20% de proximité. Si la dérive est significativement plus haute près des forget, tester une **pondération non uniforme de la perte d'ancre** sur ces voisins dans le PG (même poids total, pas de nouvelles images) ; contrôler accuracy/moments oubli.
- Sans effet de proximité ou si proxies non représentatifs, **abandonner**. Aucune hypothèse d'amélioration garantie. **0.2 h GPU diag + 1 h apprentissage**, seulement après priorité B/C.

## File d'attente proposée quand une H200 est réellement libre
1. Vérifier board **ET** `nvidia-smi`, annoncer sur bus #2 ; ne pas préempter les jobs d'Adem/Titouan.
2. T1: déposer/envoi 22 validé **sans GPU** ; si déjà envoyé, conserver nouvelle baseline.
3. Si carte libre ≥ 30 GiB : T3 script C sur caches existants / contrôle CPU d'abord ; 1–2 ablations courtes, pas de soumission si valeurs inconnues.
4. Si autre GPU libre : T1 residual native head B. Mettre à jour #3/#5 avec chiffres exacts, CI apparié T1.
5. T2 **aucun appel** ; le seul changement T2 dans l'équipe est sous Titouan.

## Risques / sécurité de la coordination
- T1 test public 30 % ; garder score privé pour robustesse. T3 70% des contrôles cachés ; pas de sur-optimisation au public.
- Ne pas confondre `reserve 0.7514` et public. Ne pas annoncer un « top1 » avant preuve.
- La documentation SSH montre des **chemins de clés privées propres à l'ordinateur de Yassir/NemoClaw**. Ici ni clé privée ni accès sortant au dashboard ; **ne pas programmer ni prétendre l'exécution H200 automatique**. Les scripts sont prêts pour être lancés par une session connectée habilitée.
- Aucun secret/API key dans les fichiers ni commentaires.
