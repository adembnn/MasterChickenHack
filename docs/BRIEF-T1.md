# Contexte commun des sous-sessions du track 1 (à lire en entier avant d'agir)

## Le hackathon
European AI Championship Paris. Fin du code dimanche 22/11/2026 à 12:00. Équipe MasterChickenHack (Adem, Titouan, Yassir).
Dépôt : `E:\EAC` (GitHub `adembnn/EAC`). Lis `E:\EAC\CLAUDE.md` (règles d'équipe, boucle mesurer → critiquer → corriger).
Classement : sur chaque tâche le score est ramené entre 0 et 1 sur toutes les équipes, puis moyenne des 3 tâches. Public = 30 % du test.

## La tâche (track 1)
Image 512x512 + méthode de filigrane (PixelSeal 256 bits, TrustMark-Q 100 bits, StegaStamp 100 bits) + message caché connu.
Rendre un masque 128x128 (1 = filigrané). Score = par image, moyenne de l'IoU de la classe propre et de la classe filigranée.
510 images de train avec masques (170 par filigrane), 1020 de test (340 par filigrane). Consigne : `E:\EAC\reports\sources\Watermark_Localization.txt`.
Deux populations, moitié-moitié (planche à regarder : `E:\EAC-lab\t1\planche_train.png`, rouge = zone propre) :
- « pleines » (`full_watermark` vrai, drapeau présent seulement dans le train) : 50 à 99 % filigrané, la zone propre est une tache irrégulière ;
- « partielles » : 1 à 18 % filigrané, petit rectangle ou forme d'objet.

## Ce qui est déjà mesuré (mIoU officiel sur le vrai train)
| Méthode | PixelSeal | TrustMark | StegaStamp |
|---|---|---|---|
| Constante | 0,25 | 0,25 | 0,25 |
| Filtre adapté (exp2, version rapide, peut-être mal faite) | 0,28 | 0,31 | 0,36 |
| Meilleure carte sans entraînement (occlusion 64 px, seuil global) | 0,51 | 0,48 | ? |
| Boîte englobante vraie (oracle) | 0,66 | 0,66 | 0,66 |
| U-Net ResNet-34, 6000 synthétiques COCO, aucune image réelle (seg v1) | 0,29 (n'apprend pas) | 0,59 | 0,63 (puis redescend) |
Plafonds si la carte était parfaite à k px près : 8 px 0,95 ; 16 px 0,90 ; 32 px 0,82 ; 64 px 0,71.
Bits justes sur le réel : image entière 0,75 ; boîte vraie 0,81-0,85 ; hors boîte 0,50.
En cours : seg v2 (job 15901531) = vraies images mêlées aux lots + tronc = décodeur du filigrane + accord au message connu.
À lire : `E:\EAC\reports\critique\t1-tour1.md`, `E:\EAC\reports\t1-comment-font-les-labos.md`, `E:\EAC\reports\t1-watermark-etat-de-lart.md`,
`E:\EAC\reports\titouan\ideas-t1.md`, et les scripts `E:\EAC\tracks\t1-watermark\experiences\` (exp0, exp1, exp2, seg.py, seg2.py).
Leçon d'Adem : l'amélioration doit se mesurer sur les images du hackathon, pas sur des images fabriquées.

## Le cluster Jülich (connexion déjà ouverte, ne la ferme pas)
- Exécuter un script bash sur le nœud de login : l'écrire en local (fins de ligne LF), puis `/e/EAC-lab/remote/j.sh mon_script.sh` depuis Git Bash.
  Commencer par `jutil env activate -p training2666`. Copier un fichier : `wsl -e bash -lc 'scp -q /mnt/e/chemin jureca:/p/scratch/.../'`.
- Dossier T1 : `/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task` avec `.venv` (torch 2.5.1, timm,
  segmentation-models-pytorch, trustmark, onnxruntime-gpu), `kit/` (wmloc + modèles), `release/train`, `release/test`, `adem/` (scripts, cartes
  `maps_train_*.npz`, données synthétiques `seg_v1_<methode>/X.npy, Y.npy`). COCO val2017 : `../shared_data/coco/val2017`.
- Écris UNIQUEMENT dans `adem/<nom-de-ta-session>/` sur le cluster. Pour un paquet Python en plus : `uv pip install -p .venv/bin/python <pkg>`,
  seulement s'il est léger, et dis-le dans ton rapport. Poids de modèles : à télécharger depuis le nœud de login (les nœuds de calcul n'ont pas internet ;
  exporter `HF_HOME=/p/scratch/training2666/destors1/MasterChickenHack/.cache/huggingface`, `TORCH_HOME=.../.cache/torch`, et `HF_HUB_OFFLINE=1` dans les jobs).
- Nœud de login : 2 GPU RTX 8000 partagés, calculs de moins de 5 minutes seulement.
- Jobs : `sbatch` avec `--account=training2666 --partition=dc-gpu --reservation=cispahack --gres=gpu:4 --cpus-per-task=96 --time=01:00:00`.
  Un job prend un nœud entier (4 A100 40 Go) et la file est saturée (30 nœuds pour toutes les équipes). Donc : regroupe 4 configurations par job
  (`CUDA_VISIBLE_DEVICES=0..3 python ... &` puis `wait`), fixe `torch.set_num_threads(8)`, fais tout le calcul lourd sur GPU, au plus 1 job en cours
  et 1 en attente par session, modèle de job : `E:\EAC\tracks\t1-watermark\experiences\seg2.sbatch`.
- Une passe sur 6000 images 512 avec un U-Net ResNet-34 prend 30 s sur une A100 : itère vite.

## Règles
- Aucune soumission au leaderboard, aucun appel à l'API du hackathon, ne lis pas le fichier `.env` du dossier d'équipe.
- N'ouvre pas les dossiers des autres équipes sur le cluster.
- Ne fais aucun `git commit` ni `git push` (plusieurs sessions partagent le dépôt) : l'orchestrateur s'en charge. Ne modifie pas les fichiers des autres.
- Copies locales de tes scripts : `E:\EAC-lab\t1s\<nom-de-ta-session>\`. Rien de lourd hors de `E:\EAC-lab`.

## Protocole de mesure commun (obligatoire, pour que les sessions soient comparables)
- Réserve : les images de train dont l'index dans `metadata.json` vérifie `index % 5 == 0` (34 par filigrane). Rien d'elles ne sert à l'entraînement ni au réglage.
  Le chiffre qui compte est le mIoU officiel (`wmloc.utils.miou_image`) sur la réserve, par filigrane, et séparé pleines / partielles.
- Pour une estimation moins bruitée, fais aussi une validation croisée à 5 plis sur les 4 autres cinquièmes quand ta méthode s'entraîne sur le réel.
- Le bruit sur 34 images vaut environ ±0,03 : ne conclus pas sur un écart plus petit.
- Seuil et post-traitement se règlent hors réserve.

## Boucle de travail
Mesure, puis fais critiquer tes résultats et ton code par un sous-agent séparé (spécialiste, qui n'a pas écrit la méthode), corrige, relance.
À chaque tour, teste au moins une idée hors des réponses évidentes. Arrête quand deux tours de suite n'améliorent plus la réserve.

## Livrables de chaque session
1. Prédictions réutilisables par l'orchestrateur pour combiner les méthodes : `adem/<session>/pred_train_<methode>.npz` (prédictions hors-pli pour les 170 images,
   clés `names` et `logits` float16 N x 128 x 128, valeur > 0 = filigrané) et `pred_test_<methode>.npz` (340 images), pour chaque filigrane que ta méthode couvre.
2. Rapport `E:\EAC\reports\t1-<session>.md`, 150 lignes au plus, en français simple, chaque terme technique expliqué une fois : ce qui a été essayé,
   le tableau des scores (collé depuis la sortie des commandes), ce qui est mesuré et ce qui est supposé, ce qui a échoué et pourquoi.
3. Un commentaire de 6 lignes sur l'issue GitHub #3 (`gh issue comment 3 -R adembnn/EAC --body "..."`), signé `[session <nom> d'Adem · T1]`,
   à chaque résultat qui change une décision.

## Critique Opus du tour 2 (à lire : `E:\EAC\reports\critique\t1-tour2.md`), points qui s'imposent à toutes les sessions
- **bfloat16 sur l'entrée** (`seg.py`, `seg2.py` : `torch.autocast(dtype=bfloat16)` autour du forward) : après normalisation, un pas de 1/255 vaut 1 à 2 pas
  d'arrondi dans ce format. Cause probable de l'échec PixelSeal (l'encre la plus faible). Garde la première couche et l'entrée en float32, ou tout en fp32.
- **Sélection de la meilleure époque** : uniquement sur la réserve ou en validation croisée, jamais sur des images vues à l'entraînement (`seg2.py` le fait mal).
- **`exp2.py` ne prouve pas que le filtre adapté échoue** : le gabarit y est arrondi en uint8 (pour PixelSeal il devient presque nul) et le passe-haut σ=1,5
  ne garde que le détail le plus fin alors que le résidu est calculé à 256 puis agrandi x2.
- **Fonds propres réels gratuits** : une image PixelSeal du hackathon est propre pour TrustMark et StegaStamp, et inversement. Cela donne 340 vraies images
  du bon style par filigrane comme fonds pour fabriquer des exemples (hors réserve pour celles qui ont un rôle dans la mesure).
- Rectangles synthétiques : 1 à 18 % de l'image dans le réel (pas 1 à 35 %), et pour l'outpainting le rectangle filigrané est une image source ENTIÈRE
  (le motif est calé sur le rectangle).
- Fichier de soumission : masques en `uint8` ou `bool`, pas `int64` (134 Mo, rejeté).
- Suivre un score sans seuil (AUC par pixel) à chaque époque pour séparer « mauvais seuil » de « mauvais modèle ».

## Deuxième calcul : la H200 de Yassir (ouverte à 17 h 40) : à utiliser en priorité, sans file d'attente
- Connexion directe depuis Git Bash, sans code à taper : `ssh h200 '<commande>'` (alias déjà dans `~/.ssh/config`, compte `adem`, pas de sudo).
  Copier : `scp fichier h200:~/eac/t1/<session>/`. La machine a internet.
- Une seule carte NVIDIA H200 de 144 Go, 16 cœurs, 196 Go de mémoire, 1,2 To de disque. Elle est PARTAGÉE avec Yassir et Titouan et entre nos sessions :
  regarde `nvidia-smi` avant de lancer, ne prends pas plus de 30 Go de mémoire GPU par session sans l'annoncer, 4 cœurs par processus
  (`torch.set_num_threads(4)`, 4 workers au plus), et lance en arrière-plan avec `nohup ... > log 2>&1 &`.
- Elle est préemptible (peut s'éteindre sans prévenir) : sauvegarde les points de contrôle à chaque époque et rapatrie ce qui compte
  (`scp h200:... /e/EAC-lab/t1s/<session>/`). Rien d'unique ne doit y rester.
- Dossier T1 : `~/eac/t1` avec `.venv` (python 3.11, torch 2.5.1, timm, segmentation-models-pytorch, trustmark, onnxruntime-gpu), `kit/` (wmloc + modèles),
  `release/train`, `release/test`. COCO val2017 : `~/eac/shared_data/coco/val2017`. Écris dans `~/eac/t1/<nom-de-ta-session>/`.
  Mêmes chemins relatifs que sur Jülich : un script qui fait `HERE = Path(__file__).resolve().parent.parent` marche tel quel depuis `~/eac/t1/<session>/`.
- Jülich reste disponible (4 A100 par job, mais file d'attente et connexion fragile) : garde-le pour les gros balayages en parallèle.
- Les livrables (`pred_train_*.npz`, `pred_test_*.npz`) vont dans `~/eac/t1/<session>/` sur la H200 ET sont rapatriés dans `E:\EAC-lab\t1s\<session>\`.
