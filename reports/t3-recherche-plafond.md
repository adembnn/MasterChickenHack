# T3 : recherche sur le plafond (de 0,39 à plus de 0,45)

Décision débloquée : quelle technique tester dans les 6 prochaines heures. Aucun entraînement lancé, aucun appel serveur. Sources lues le 22/11 (texte intégral téléchargé et lu par `grep` quand c'est écrit « PDF », sinon résumé de la page). Étiquettes : **VÉRIFIÉ** = lu dans la source ; **SUPPOSÉ** = mon raisonnement, aucune source.

Termes : **gold** = modèle réentraîné sans les 3 500 images ; **cross-fitting** = entraîner sur les autres plis et lire la prédiction sur le pli exclu ; **mémorisation (Feldman)** = P(bonne classe | image dans l'entraînement) − P(bonne classe | image retirée) ; **C-score** = précision attendue sur l'image quand elle n'est pas dans l'entraînement ; **KL** = coût de prédire q quand la référence est p.

## 0. Verdict en 5 lignes
1. Aucune source lue ne donne une méthode qui prédit la sortie d'un modèle réentraîné mieux que votre cross-fitting. Votre pipeline est la forme « réinitialiser puis distiller » des gagnants 2023 (§1), avec en plus la validation croisée, que je n'ai trouvée nulle part. VÉRIFIÉ (absence dans mes recherches, pas preuve d'absence).
2. Une théorie de 2026 dit que forger la sortie d'un gold sans les données retenues a des bornes basses : le plafond n'est pas forcément 1 (§1). VÉRIFIÉ (résumé de la page).
3. Le prochain gain réaliste est du **bruit de l'estimateur** : une lecture cross-fitting est UN tirage de gold, or la référence est la MOYENNE de plusieurs golds. Vos 3 graines ont échoué sans doute parce qu'elles partagent les mêmes plis et les blocs 0-7 figés (SUPPOSÉ). Il faut varier les PARTITIONS.
4. Les proxies de mémorisation bon marché reposent sur la dynamique d'entraînement de l'image (vitesse d'apprentissage) ; on peut la fabriquer chez vous (§2).
5. Je ne vois pas de chemin sourcé vers +0,06. Attendez +0,01 à +0,03 par expérience, non cumulable d'office.

## 1. Question 1 : méthodes qui prédisent la sortie d'un réentraîné
### Défi NeurIPS 2023 (Kaggle, 1 121 équipes) : arXiv 2406.09073, PDF
- VÉRIFIÉ : presque toutes les solutions = phase « effacer » puis phase « réparer ». Effacer : réinitialiser des couches (Amnesiacs, Sun, Forget, Kookmin, Sebastian) ou ajouter du bruit gaussien (Seif) ; Fanchuan tire la sortie vers l'uniforme puis un contraste. Réparer : cross-entropie sur les données retenues.
- VÉRIFIÉ : **Amnesiacs** réinitialise la 1re conv et la tête, distille le modèle d'origine sur des exemples de validation non vus, puis finit avec entropie croisée + KL symétrique vers l'original. **Forget** alterne « réinitialiser des couches au hasard » et « distiller l'original sur des entrées bruitées ». C'est votre recette (réinit + distillation + ancre sur entrées dégradées).
- VÉRIFIÉ : tous avaient les données retenues ; l'évaluation du défi est une attaque d'inférence d'appartenance contre la distribution des réentraînés, pas la KL image par image de notre énoncé. Les transferts de leurs classements vers notre score sont donc **SUPPOSÉS**.
- Je n'ai trouvé aucune rédaction de la solution n°1 (Fanchuan) ; le classement 1-2-3 vient d'une source secondaire (SUPPOSÉ).
### Autres méthodes de l'énoncé
- **SCRUB** (arXiv 2302.09880, PDF) : VÉRIFIÉ, alternance de pas « max » (éloigner l'élève du professeur sur l'oubli) et « min » (le rapprocher sur les retenues), terminée par des pas « min » seuls ; sans équilibre, l'erreur d'oubli retombe ou la rétention chute. Elle exige les données retenues pour les pas « min » : non applicable tel quel.
- **Bad teacher** (2205.08096) : VÉRIFIÉ (résumé de page seulement) : un professeur compétent + un incompétent ; le détail de la KL n'est pas lu. Votre « cible fabriquée » est de la même famille.
- **SalUn** (2310.12508) : VÉRIFIÉ (résumé) : masque de saillance sur les poids + étiquettes aléatoires. Même famille que votre bascule vers une classe fausse ; vous testez déjà mieux (classe fausse lue sur un modèle).
- **Zhao, Triantafillou, « what makes unlearning hard » (2406.01257, PDF)** : VÉRIFIÉ, plus l'ensemble oublié est mémorisé et emmêlé avec les retenues, plus c'est dur ; leur méta-algorithme **RUM** découpe l'oubli en sous-ensembles homogènes en mémorisation et les traite en séquence (ordre bas → moyen → haut testé, les deux ordres donnent un ToW proche). Piste pour vous : traiter vos bascules par tranches de rang (SUPPOSÉ applicable).
- **« Why Forget-Only Unlearning Needs Memorization »** (arXiv 2610.10519) : VÉRIFIÉ (résumé) : des jeux de données différents peuvent donner le même modèle entraîné et exiger des sorties différentes après retrait ; bornes basses sur l'écart au réentraîné. Lecture : une partie de votre écart au gold est irréductible sans les données.
- **Hayes et al. 2403.01218** : VÉRIFIÉ (résumé) : évaluer par attaque par image (LiRA) montre que les évaluations de population surestiment l'oubli. Sans lien direct avec notre KL.
### Estimateurs de la prédiction « leave-one-out » applicables ici
- **Feldman & Zhang, 2008.03703 (PDF)** : VÉRIFIÉ, la mémorisation est estimée sur des sous-échantillons (modèles entraînés sur 70 % des données, des milliers de modèles) ; les fortes valeurs = exemples atypiques, ambigus ou mal étiquetés ; certains exemples mémorisés « aident » des images de test visuellement proches (paires). Inapplicable en coût ; leur esprit = votre cross-fitting.
- **Jiang et al., C-score, 2002.03206 (PDF)** : VÉRIFIÉ, les proxies par distance dans l'espace des pixels corrèlent très mal avec le C-score (Spearman 0,07 à 0,15 sur CIFAR) ; dans l'espace des représentations, la densité locale pondérée par classe fait mieux (je n'ai pas relevé le chiffre) ; la **vitesse d'apprentissage** (perte cumulée de l'image pendant l'entraînement) corrèle le mieux. Donc un kNN sur les caractéristiques de l'original est un mauvais substitut (SUPPOSÉ pour notre cas : l'original a déjà vu ces images).
- **Zhao & Triantafillou, 2410.16516 (PDF)** : VÉRIFIÉ, Spearman avec la mémorisation sur CIFAR : confiance max −0,80 / −0,91, entropie −0,76 / −0,87, « holdout retraining » (KL entre l'original et l'original affiné sur des données non vues) −0,71 / −0,89, courbure de la perte +0,67 / +0,62. Calculés avec un modèle entraîné sur les données ; un ViT surappris saturera la confiance sur les 3 500 images (SUPPOSÉ), ce qui tue les deux premiers.
- **Google 2310.05337** : VÉRIFIÉ (résumé) : plusieurs proxies ne capturent pas les tendances de la mémorisation avec la taille du modèle. Prudence sur tout proxy.
- Sondes linéaires sur blocs précoces, moyenne par dropout ou élagage : je n'ai trouvé aucune source qui les valide comme estimateur de mémorisation. SUPPOSÉ.

## 2. Question 2 : les copies altérées
- Les gagnants 2023 n'avaient pas de copies altérées : rien à copier. VÉRIFIÉ.
- **RURK, arXiv 2601.22359** : VÉRIFIÉ (résumé), les modèles « oubliés » reconnaissent encore des versions perturbées (norme ~0,03) des images oubliées que le réentraîné rate ; plus de 7 % des images de CIFAR-10 ; la correction pénalise la perte sur des copies perturbées de l'image. Limite écrite par les auteurs : le réentraîné n'est pas disponible, donc un seul côté de la borne est imposable. Votre entraînement sur copies fortes est déjà de cette famille.
- Étude de l'augmentation (2508.18502) et vecteurs d'oubli (2412.16780) : résultats trouvés mais non lus en détail. SUPPOSÉ pour tout le reste.
- Vos chiffres (issue 5) : gold plus bas de 0,3 à 0,6 sur les copies, et vos copies sont déjà à 4,5 contre 4,85 sur les exactes : le gap de loss sur les copies n'est probablement pas le levier restant.

## 3. Question 3 : rétention sans données retenues
- **Poppi et al., ICPR 2024** (dé-apprendre des ViT sans données retenues par décomposition de bas rang, rétention par parcimonie du bas rang) : VÉRIFIÉ (résumé de recherche seulement). C'est la seule source trouvée qui ressemble à votre problème ; non lu en détail.
- **GS-LoRA** (2501.09705) : LoRA sur les couches feed-forward avec régularisation de groupe, données retenues rares : VÉRIFIÉ (résumé). **EWC-LoRA** (2602.17559) : intégrer EWC naïvement à un adaptateur de bas rang est sous-optimal : VÉRIFIÉ (résumé).
- **L2-SP ou EWC pour l'oubli en classification** : aucune source trouvée. SUPPOSÉ, et votre « ancre plus forte = meilleure rétention locale mais pire score réel » suggère que votre indicateur local de rétention (KL sur mélanges) ne suit pas le contrôle caché.
- Les gagnants remplaçaient les données retenues par des entrées non vues ou bruitées (Amnesiacs, Forget) : votre ancre sur mélanges est le même choix. VÉRIFIÉ.
- Rien de sourcé ne dit quelle restriction (tête seule, bas rang, sous-espace) préserve le mieux la sortie sur les données d'entraînement.

## 4. Question 4 : mémorisées contre jamais vues, et simuler mieux
- VÉRIFIÉ (Feldman) : ce que le gold rate = surtout les images atypiques, ambiguës et les paires d'images proches ; l'image « utile à un voisin » perd sa précision quand elle est retirée. La confusion d'un voisin visuel est donc la classe fausse la plus probable (SUPPOSÉ pour notre jeu).
- VÉRIFIÉ (blog/résumés, faible) : un ViT-B entraîné de zéro sur ~2 000 images sature à ~62 % de validation (blog) ; l'optimisation (échauffement, décroissance cosinus, régularisation forte) compte plus que l'architecture (résumé ICLR 2026). Un ViT de zéro sur 2 800 images (un pli de 3 500) sera nettement plus faible que le gold (≈ 9 500 images) : SUPPOSÉ.
- Constat chez vous (issue 5) : les redistillations à 6 blocs ont 56 % d'erreur hors pli, le gold ajusté en suppose 54 à 60 %. Un simulateur à plus de blocs ne se rapproche donc pas davantage du gold sur ce chiffre : la profondeur de réinitialisation n'est pas le goulot, c'est la quantité d'information (SUPPOSÉ).
- Fuite à garder en tête (SUPPOSÉ) : votre élève garde les blocs 0-7 de l'original, qui ont vu l'image exclue ; le gold ne l'a jamais vue. Un élève entièrement réinitialisé supprime cette fuite mais est plus faible.

## 5. Expériences classées par crédibilité
Coûts H200 = estimations non mesurées (je n'ai pas vos journaux d'entraînement) : à remplacer par la durée d'un de vos `RDt24r`.

**1. Cross-fitting répété sur plusieurs partitions, moyenne des probabilités à T=2.** Refaire la source avec 3 partitions différentes en 5 plis (15 entraînements courts), moyenner les p(classe) hors pli par image, puis refaire le classement et la classe fausse sur la moyenne. Pourquoi : la référence est une moyenne de golds ; une seule lecture est un tirage bruité (théorie : le q qui minimise l'espérance de la KL est la moyenne). Coût : 3 × votre source actuelle ; arrêter à 2 partitions si une source dure plus de 10 min (SUPPOSÉ). Signal attendu : mesurer d'abord le Spearman entre les rangs de deux partitions ; s'il est inférieur à 0,85, le bruit est réel et le gain possible (+0,01 à +0,03, SUPPOSÉ) ; s'il dépasse 0,95, abandonner. Mesure faite avant tout envoi.

**2. Vitesse d'apprentissage comme second classement.** Une distillation sur les 3 500 images sans pli exclu, en journalisant la perte de chaque image tous les N pas ; la perte cumulée par image est le proxy le mieux corrélé au C-score dans Jiang et al. Combiner son rang à celui du cross-fitting (moyenne des rangs). Coût : 1 entraînement + journal, négligeable. Signal : Spearman proxy contre p̄(classe vraie) entre 0,3 et 0,7 = information indépendante ; ≥ 0,9 = redondant, abandonner ; ≤ 0,2 = bruit. Puis un envoi à 70 %.

**3. Distribution fausse moyennée + perte vraie calée par image.** Pour chaque image basculée : la masse sur la vraie classe fixée pour viser la perte (comme aujourd'hui), mais le reste réparti selon la moyenne sur partitions de p renormalisée hors classe vraie (au lieu d'une seule classe), à température 1,5 à 2 ; pour les non basculées, p̄ au lieu d'une cible adoucie. Coût : un entraînement de la recette actuelle, après l'expérience 1. Signal : m et s inchangés (vérifier à 0,05 près) donc seul S_pred bouge ; adopter si ≥ +0,01 public. Note : le tour « classes entières » proposait déjà de répartir la masse fausse ; je ne vois pas dans l'issue qu'il ait été mesuré, à vérifier avant de dupliquer.

**4. Simulateur du gold par réentraînement complet (le plus cher, le moins crédible).** Élève entièrement réinitialisé, entraîné par distillation du professeur à T=2 sur mélanges et recadrages forts des 4 autres plis (cutmix, mixup), 5 plis. Coût estimé : 5 entraînements de ~300 époques sur 2 800 images, soit quelques centaines de milliers d'images chacun ; de l'ordre de 5 à 10 min par pli si le débit d'un ViT-B est de l'ordre de 2 000 images/s (SUPPOSÉ, non mesuré). Signal : sans toucher aux cibles, la perte hors pli doit valoir m ≈ 5,0 à 5,2 et s ≈ 4,6 à 4,8 (le gold ajusté par le vérificateur) et la précision hors pli environ 40 à 46 % ; si l'élève est sous 30 % de précision il est trop faible pour servir de source (SUPPOSÉ). La même exigence de signal vaut pour votre source à 4 blocs : à calculer d'abord, sans coût, comme test de réalisme avant toute dépense.

Piste écartée : **kNN, sondes linéaires, moyenne par dropout/élagage** (aucune source ne les valide, §1) et **tout apport extérieur de poids pré-entraînés** (règle de l'énoncé : « modèle d'origine et ensemble oublié seuls ») : ne pas l'essayer sans l'accord des organisateurs.

## 6. Limites de ce rapport
- Résumés d'article là où c'est dit ; le détail de Bad Teacher, SalUn et Poppi n'a pas été lu.
- Aucune des 4 expériences n'a été exécutée. Les chiffres de coût et de gain sont des estimations, pas des mesures.
- Les deux gains passés (+0,026 classe, +0,024 ordre) montrent que la source des cibles est le bon levier ; les expériences 1 à 3 restent dans cette direction.
