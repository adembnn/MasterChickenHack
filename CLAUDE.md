# EAC : European AI Championship

Hackathon d'entraînement de modèles. Équipe : adembnn, LaRacine23, yassirboudda.

## Règles du dépôt
- Un lot = une branche = un worktree ; `main` ne reçoit que du vérifié (test ou capture).
- Les données, poids et journaux d'entraînement ne sont jamais commités (voir `.gitignore`).
- Rapports d'agents dans `reports/`, jamais dans les livrables.
- Secrets dans `.env` (jamais commité).

## Décisions (la plus récente gagne)
- 2026-11-21 : dépôt créé, privé, LaRacine23 et yassirboudda invités en écriture.

## Contexte du hackathon (source : `reports/sources/`)
- Fin du code : dimanche 22/11 à 12:00 ; pitch 6 min pour les 6 meilleures équipes à 13:00. Lieu : Epitech Kremlin-Bicêtre.
- Classement final = rang combiné sur les TROIS tâches (il faut les résoudre toutes). Public 30 % / privé 70 % : ne pas sur-optimiser le public.
- Interdit de collaborer avec une autre équipe (disqualification).
- T1 Localisation de filigrane : masque binaire 128x128 sur 1020 images test (PixelSeal, TrustMark, StegaStamp), score mIoU. Données `SprintML/Hackathon_Watermark_Localisation_Task`.
- T2 Model Equality : 33 API LLM, 100 réponses max par endpoint (budget irréversible), CSV endpoint,family,model, score 0,1·MCC famille + 0,9·MCC modèle, 1 soumission / 30 min. Données `SprintML/model_equality`.
- T3 Unlearning : ViT-B/16 100 classes, oublier 3500 images sans étiquettes ni données de rétention ; score oubli × rétention, seuil 70 % de précision sinon 0. Données `SprintML/unlearning`.
- Calcul : JURECA (Jülich), projet `training2666`, réservation `cispahack`, partition `dc-gpu`, 1 nœud = 4×A100 40 Go par job. Mise en place : `SprintML/hackathon` (chef d'équipe lance `hackathon_setup.sh`, les autres `teammate.sh`).
- Leaderboard : http://34.147.229.55/leaderboard_page
