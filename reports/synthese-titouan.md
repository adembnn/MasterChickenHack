# Synthèse de l'agent de Titouan : ce que 8 agents de recherche ajoutent au plan de bataille

Écrit samedi 21/11 ~16h30. Rapports complets dans `reports/titouan/` (certains dépassent 200 lignes : ce sont des rapports bruts). Outils dans `tools/validators/` et `tools/sim-t2/`. Chaque point est tel qu'écrit par un agent : peu de chiffres sont mesurés, beaucoup sont des hypothèses.

## Décision demandée à tous : comment est calculé le classement ?
- Le plan de bataille d'Adem (et `CLAUDE.md`) dit : combinaison des **rangs** sur les trois tâches.
- Titouan a dit (samedi 16h13, source à confirmer à l'accueil) : classement par **somme des scores** des trois tâches.
- Un agent a trouvé, pour l'édition précédente, « score combiné des trois tâches » plus un jury sur le pitch, sans plus.
- Si c'est la somme des scores : un point de score gagné sur n'importe quelle tâche vaut pareil. Gagner 0,15 sur T3 (0,35 -> 0,50) vaut autant que 0,15 sur T1.
- À demander aux organisateurs, **en une seule visite** : (1) règle d'agrégation, (2) le privé retient-il la meilleure soumission publique ou la dernière, (3) délais entre soumissions pour T1 et T3, (4) accès internet des nœuds de calcul.

## Temps : heures-GPU contre heures réelles
Les « 25-35 h » de T1 et « 15-25 h » de T3 sont des **heures-GPU cumulées**, pas du temps d'attente. Avec 4 A100 par nœud et des jobs en parallèle, la durée réelle est bien plus courte. Il reste ~19 h 45 min (samedi 16h15 -> dimanche 12h00).
À vérifier ce soir sur JURECA : `scontrol show reservation cispahack` (fin de la réservation, nombre de nœuds, comptes autorisés).

## T1 Localisation de watermark
- Les agents d'Adem et les miens convergent : **expérience 0 en 30 min** (décoder le rectangle vrai contre l'image entière ; motif calé sur le rectangle ?). Si oui : chercher le rectangle + trous.
- Détails des extracteurs (`reports/titouan/t1-decoders.md`) : PixelSeal et TrustMark décomposables par position ; StegaStamp en ONNX, sans gradient. Résidu PixelSeal faible dans les aplats (masquage JND).
- Conception : statistique calibrée avec les messages aléatoires du jeu, tomographie linéaire des fenêtres vers 128x128, rectangle de somme maximale (Kadane 2D) avec sculpture des trous.
- Piège de métrique : sur une image à classe vide, une cellule fausse coûte 0,5 -> règle de forçage tout-0 / tout-1.
- Segmenteur (`t1-segmentation.md`) : backbone d'extracteur (PixelSeal ConvNeXt, TrustMark ResNet-50) + carte d'accord au message connu ; données synthétiques par banques recombinées ; perte BCE + Lovász + soft-IoU ; 5 plis stratifiés.
- Désaccord de chiffres : plan d'Adem 5-10 h-GPU ; mon agent 25-35 h-GPU. À trancher après l'expérience 0.

## T2 Model Equality
- **Le budget compte des réponses (100 par endpoint, 256 tokens), pas des tokens** : 25 600 tokens de signal par endpoint. Deux manières d'en profiter, à départager sur `ref-*` et un jumeau local : (a) lots de mini-questions numérotées (plan d'Adem) ; (b) réponses longues notées token par token sous des modèles locaux (`t2-fingerprinting.md`).
- Puissance (`t2-statistics.md`, estimations) : à N=100, MMD détecte nf4 (~0,6), pas int8 (~0,16), pas fp8, pas LoRA 0,3. Les nuisances (system prompt, persona, watermark) pèsent 10 à 300 fois plus que fp8/int8/LoRA 0,3 : les p-valeurs naïves sont invalides.
- À N fixe, puissance ~ N/sqrt(m) : peu de prompts, beaucoup de répétitions. Choisir les prompts hors ligne (ref + jumeau local), pas en ligne sur ep.
- **MCC asymétrique : fusionner à tort coûte plus cher qu'isoler.** Seuil optimal sur la probabilité postérieure de « même » : 0,35-0,45, pas 0,05. Décodage : lien moyen, clustering de corrélation, MBR par bootstrap.
- Budget proposé par l'agent statistique : 16 (famille) + 48 (16 prompts x 3) + 24 (adaptatif) + 12 (réserve). Plan d'Adem : 60 + 16 + 24. À unifier avant la première requête sur `ep-*`.
- Endpoints synthétiques locaux (1 000 à 3 000 : watermark KGW/Unigram, system prompts, personas, quantifs, LoRA par `scaling` PEFT) pour calibrer sans dépenser de budget : 2 à 5 h de nœud.
- Rien sur `ep-*` avant un plan posté sur l'issue #4.

## T3 Unlearning
- La métrique se coupe en deux : moments de la loss (moyenne, écart-type) et KL par image à T=2 vers la moyenne des gold (plus lisse). **Une température globale ne suffit pas** : il faut des cibles « moment-matchées » (une fraction epsilon d'images forcée vers une classe voisine plausible).
- Hypothèse non vérifiée : LayerScale 1e-4 n'est pas un défaut timm/DeiT-III, donc le gold serait un ViT-B entraîné de zéro avec ~70-85 % de précision sur l'oubli. Confirmer par des sondes sur epsilon (coûteuses en soumissions).
- Recettes (`t3-unlearning.md`) : R-A distillation moment-matchée + ancrage KL (derniers blocs, lr 2e-6 à 2e-5) ; R-A' + interpolation + moyenne de graines ; R-B pseudo-gold par élèves ré-initialisés (si images externes) ; R-C SCRUB / bad teacher / NPO ; R-S recalibrage de tête.
- **Simulateur local** de la formule exacte (CIFAR-100, ViT-Tiny/ResNet-18, gold = 4 graines) pour choisir la recette sans brûler de soumissions.
- Garde-fous : accord argmax avec l'original >= 95-97 %, KL de rétention < 0,05, prétraitement [-1,1], pas de préfixes `module.` / `_orig_mod.`. Soumission de sécurité vers 23h.

## Soumissions et exploitation (`submissions-ops.md`)
- Une soumission T2 refusée consomme quand même les 30 min [lu]. Après 10h30 dimanche : 3 essais T2 au plus.
- Aucun dry run serveur : valider en local avec `tools/validators/validate_t{1,2,3}.py` (stdlib, testés sur 15 cas avec de faux `.pt`). `selfcheck_t3.py` exige torch et n'est pas testé.
- Le kit T1 contient `BASE_URL="http://SERVER_IP"` à remplacer par `http://34.147.229.55`. Taille max 20 Mo (PDF) ou 25 Mo (README) : viser 20.
- Timeouts clients : T1 120 s, T2/T3 600 s. Après un timeout, regarder le classement avant de renvoyer.
- Le modèle de job `--gres=gpu:1` : un nœud = 4 GPU ; vérifier la réservation (voir plus haut).
- Veille (`intelligence.md`) : pièges des éditions passées : le serveur masque les scores inférieurs au meilleur ; limites de débit ; mauvaise normalisation ; nœuds de calcul sans internet (à vérifier) ; quota d'inodes partagé ; noms de jobs Slurm qui révèlent la méthode.

## Stratégie (`strategy.md`)
- Équilibrer d'abord : un zéro sur T3 fait perdre 8-14 places au combiné (simulation sur un champ adverse inventé).
- Scores réalistes (non mesurés) : T1 ~0,65, T2 0,45-0,50, T3 ~0,35. P(top 6) estimée 15-30 %.
- Bruit public/privé : modéré T1 et T3 (2-3 places), fort T2 (5-7 places).
- Calendrier de nuit, critères d'arrêt par heure, plan de pitch : dans `strategy.md`.

## À faire maintenant (proposition)
1. Trancher la règle d'agrégation et poser les 3 autres questions aux organisateurs.
2. Vérifier `scontrol show reservation cispahack` et l'accès internet des nœuds de calcul ; télécharger sur le nœud de login COCO val, les modèles d'inpainting et OLMo.
3. Lancer les expériences de 30 min : T1 exp. 0 ; T2 jumeau local + bootstrap sur `ref-*` (dès que la clé arrive) ; T3 simulateur CIFAR-100.
4. Une soumission de sécurité par tâche avant minuit (T1 bases tout-0/tout-1/boîte, T3 modèle recalibré, T2 CSV après analyse).

Décisions bloquantes : répartition des tracks (Adem), règle d'agrégation, qui prend le simulateur T3.
