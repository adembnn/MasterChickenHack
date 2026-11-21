# Contexte commun des sous-sessions du track 3 « Unlearning » (à lire en entier avant d'agir)

## Le hackathon
European AI Championship Paris. Fin du code dimanche 22/11/2026 à 12:00. Équipe MasterChickenHack (Adem, Titouan, Yassir).
Dépôt : `E:\EAC` (GitHub `adembnn/EAC`). Lis `E:\EAC\CLAUDE.md` (règles d'équipe, boucle mesurer → critiquer → corriger).
Classement : sur chaque tâche le score est ramené entre 0 et 1 sur toutes les équipes (la meilleure a 1, la plus basse 0), puis moyenne des 3 tâches.
Seules les équipes notées sur les 3 tâches sont classées. Nous sommes 1ers sur le track 1 (0,635) et n'avons RIEN sur le track 3.
Leaderboard track 3 à 17 h 25 : meilleur 0,09 ; puis 0,07 ; 0,03 ; 0,02 ; 0,01 ; plusieurs à 0,00. Tout le monde est très bas : il y a beaucoup à prendre.

## La tâche
Consigne exacte : `E:\EAC\reports\sources\Unlearning.txt` (à lire mot à mot, la formule du score y est). Gabarit : `E:\EAC\reports\sources\t3\task_template.py`.
- Modèle fourni « orig » : timm `vit_base_patch16_224`, 100 classes, LayerScale 1e-4. 3 500 images à oublier, sans étiquettes. Aucune autre donnée.
- L'énoncé dit que la méthode doit marcher « from the original model and the forget set alone » : n'utilise PAS d'images externes pour entraîner un candidat.
  Elles ne servent que de règle de mesure ou dans un simulateur.
- Cible « gold » : moyenne de plusieurs modèles réentraînés sans ces images.
- Score = oubli × rétention.
  - Oubli = score de loss × score de prédiction, moyenne sur les images exactes et sur des copies modifiées cachées.
    Score de loss = 1 − clip(d_sub / d_orig, 0, 1), avec d = |écart de moyenne de la loss| + |écart d'écart-type| par rapport au gold (loss = entropie croisée avec la vraie étiquette).
    Score de prédiction = 1 − clip(KL(gold‖sub) / KL(gold‖orig), 0, 1), par image, softmax à température 2.
  - Rétention = clip(acc_sub / acc_orig, 0, 1) × 1 / (1 + KL(orig‖sub)) sur un jeu de contrôle caché.
  - Porte : moins de 70 % de précision sur un jeu de validation caché = score 0.
- Rendu : state dict float32, même architecture, chargement strict, moins de 450 Mo. Le serveur évalue en fp16.

## Ce qui est déjà mesuré (rapports à lire)
- `E:\EAC\reports\t3-etape-zero-mesures.md` : orig est sûr à 99,97 % sur les 3 500 images (99,4 % au-dessus de 0,99, même à température 2 il reste à 0,985) ;
  seules 45 classes sur 100 y apparaissent (46 à 130 images par classe) ; ce sont des images générées par IA, classes fines et voisines
  (3 lanternes, 5 bateaux, 3 chaises) ; ni ImageNet ni CIFAR ; robustesse : accord ≥ 0,975 sous recadrage, symétrie, JPEG, bruit ; 0,88 sous flou ; 0,52 sous couleur forte.
  Hors domaine, orig n'est pas saturé (max-prob 0,74 sur ImageNet-100, 0,75 sur CIFAR-100).
- `E:\EAC\reports\t3-unlearning-etat-de-lart.md` (analyse de la métrique §2, méthodes P1 à P3, N1 à N3), `E:\EAC\reports\t3-communaute-et-combinaisons.md`,
  `E:\EAC\reports\titouan\t3-unlearning.md`, `ideas-t3.md`, `redteam.md` (parties T3), jouets `E:\EAC\tools\sim-t3\`.
- Premier agent (scripts `E:\EAC\tracks\t3-unlearning\scripts-t3v1\`, tableaux `E:\EAC-lab\t3\diag\tables.md`, rapport en cours `E:\EAC\reports\t3-candidats-v1.md`) :
  10 candidats entraînés sur Jülich, tous valides au format (327,7 Mo, float32, forward fp16 fini).
  - S0 « sûr » (tête + norme finale, cible = orig adouci τ 1,5 à 3) : argmax d'orig conservé à 100 %, mais loss moyenne 0,0008 à 0,012 (orig 0,0003) : n'oublie presque rien.
  - S1 « cibles par quantiles » (difficulté par image → probabilité visée de la bonne classe pour fixer moyenne ET écart-type de la loss ; mauvaise classe = classe voisine) :
    loss moyenne 0,04 à 0,35, 1 à 8 % de mauvaises classes, mais dérive forte hors domaine (accord d'argmax avec orig 0,67 à 0,82 sur CIFAR-100, KL jusqu'à 0,49).
    Sous-entraîné à 800 pas ; un tour 2 (3 000 pas) tourne sur Jülich (job 15901778).
  - La mesure de difficulté « instabilité sous augmentations » est presque indépendante des deux autres (kNN et tête validée croisée, corrélées à 0,45).
  - Inconnue centrale : la moyenne et l'écart-type de la loss du gold. Hypothèse de travail non vérifiée : précision de non-membre 0,85 (loss moyenne 0,55, écart-type 1,17).

## Calcul
- H200 de Yassir, à utiliser en priorité : `ssh h200 '<commande>'` depuis Git Bash, sans code. Internet. Une carte de 144 Go PARTAGÉE (Yassir, Titouan, quatre sessions du track 1) :
  regarde `nvidia-smi` avant de lancer, 30 Go de GPU par session au plus sans l'annoncer, 4 cœurs par processus, `nohup ... > log 2>&1 &`.
  Machine préemptible : points de contrôle fréquents, rapatrie ce qui compte. Dossier : `~/eac/t3` (`model.pt`, `forget_images.npz`, `.venv` avec torch, timm, scikit-learn).
  Écris dans `~/eac/t3/<nom-de-ta-session>/`.
- Jülich (4 A100 par job, file d'attente, connexion fragile) : script bash local en LF puis `/e/EAC-lab/remote/j.sh mon_script.sh` (3 accès simultanés au plus,
  commandes regroupées, pas de sondage serré). Dossier `/p/scratch/training2666/destors1/MasterChickenHack/unlearning` (`.venv`, données, `adem/candidates/` du premier agent).
  Jobs : `--account=training2666 --partition=dc-gpu --reservation=cispahack --gres=gpu:4 --cpus-per-task=32 --time=01:00:00`, 4 configurations par job.
- Copies locales : `E:\EAC-lab\t3\<nom-de-ta-session>\`. Rien de lourd hors de `E:\EAC-lab`.

## Règles
- Aucune soumission au leaderboard, aucun appel à l'API du hackathon, ne lis pas le fichier `.env` de l'équipe. Tu prépares des fichiers candidats et tu donnes
  leur chemin et leurs diagnostics ; l'orchestrateur demande l'accord d'Adem et soumet.
- N'ouvre pas les dossiers des autres équipes. Ne fais aucun `git commit` ni `git push`.
- Adem n'est pas développeur : rapports en français simple, chaque terme technique expliqué une fois, chaque chiffre collé depuis une sortie de commande,
  et dis toujours ce qui est mesuré et ce qui est supposé.

## Boucle de travail
Mesure, puis fais critiquer tes résultats et ton code par un sous-agent séparé (spécialiste, qui n'a pas écrit la méthode), corrige, relance.
Remets en question les choix déjà faits, y compris ceux de ce brief. À chaque tour, teste au moins une idée hors des réponses évidentes.

## Livrables communs
- Rapport `E:\EAC\reports\t3-<session>.md` (150 lignes au plus) et un commentaire de 6 lignes sur l'issue GitHub #5
  (`gh issue comment 5 -R adembnn/EAC --body "..."`, signé `[session <nom> d'Adem · T3]`) à chaque résultat qui change une décision.
- Préviens l'orchestrateur dès qu'un candidat mérite d'être soumis : chemin du fichier, diagnostics, et ce que la soumission nous apprendrait.
