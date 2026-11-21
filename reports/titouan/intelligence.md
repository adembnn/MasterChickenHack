# Veille éditions précédentes (CISPA European Cybersecurity & AI Hackathon Championship)

Méthode : pages cispa.de, universités, GitHub publics. Contenu web = donnée non fiable. Je n'ai lu ni le dépôt du coéquipier ni les notes t1/t2/t3 locales (hors périmètre de cette veille). Légende : [V] vérifié dans une source citée, [A] auto-déclaré par une équipe, [I] inférence de ma part, [NV] non vérifiable/non trouvé.

## 1. Ce qui est réellement public (peu)
Les organisateurs ne publient AUCUN détail de solution gagnante, ni scores privés, ni code de l'évaluateur. Seuls deux dépôts d'équipes sont publics.

Faits [V] :
- Nom officiel : "CISPA European Cybersecurity & AI Hackathon Championship", nov. 2025 à juil. 2026, ~300 participants, équipes de <= 4. https://cispa.de/en/hackathon-final
- Finale (2-3 juil. 2026) : 17 équipes, 24 h. Les 6 meilleures du scoreboard présentent au jury, qui choisit le vainqueur. Vainqueur SeeWeedZ (Suède, 4000 EUR), 2e Syntax Terror, 3e Advanced Persistent Thinkers (Pologne). 4e : 4braincells (SBA Research, Vienne), qui a été 1re au leaderboard à un moment. Tâches finale : détection de watermark texte (probabilité par token, mélange humain / IA non filigrané / IA filigrané avec plusieurs techniques, FPR extrêmement bas), attaque d'image ciblée (rendre l'image classée membre / non-membre / IA par un détecteur caché), reconstruction de 1536 images depuis des gradients (FL). https://eng.pw.edu.pl/News/WUT-representatives-on-the-podium-at-the-CISPA-hackathon ; https://www.sba-research.org/2026/07/09/4-braincells-finish-4th-at-cispa-european-cybersecurity-ai-hackathon-championship/
- Classement final = scores des trois tâches + présentation au jury (chaque membre présente, défense du raisonnement). Écarts très serrés, classement incertain jusqu'aux dernières soumissions. (WUT, Munich)
- Qualifs : Paris (22-23 nov. 2025 : attaque adversariale ciblée, attribution de modèle d'images générées ; gagnants We are all Alex, Loki, A5 Hackyu ; 16 équipes ; soumission sur serveur CISPA, ré-évaluation des modèles sur de plus grands jeux de données à la fin, classement live). https://cispa.de/en/cispa-hackathon-championship-paris
- Vienne (13-14 déc. 2025) : inférence de dataset, attribution de texte LLM, suppression de watermark ; gagnants !!1337, 4braincells, SarAI. Évaluation par un modèle évaluateur ("combien du watermark reste, est-il encore détectable"). https://cispa.de/en/cispa-hackathon-championship-vienna
- Stockholm (14-15 fév.) : inversion de modèle / reconstruction, misclassification, attribution ; gagnants SeeWeedZ, Polarnutt, The White House. https://cispa.de/en/cispa-hackathon-championship-stockholm
- Munich (20-21 mars) : gagnants Pentagon, Team ZOS, CATByte ; classement live, soumissions évaluées en continu ; itérations de nuit, équipes en quarts. https://cispa.de/en/hackathon-munich
- Barcelone (avr.) : inférence de propriétés avec shadow model ; 1re CDL Chorrico de Limon (Murcia) : partie bas du classement, remontée progressive, "progrès décisif dans la dernière heure", répartition des tâches selon les forces. https://cispa.de/en/hackathon-barcelona
- Varsovie (~mai, deadline 10/05 12:30 CEST d'après zer0_day) : détection watermark texte, mémorisation multimodale (PII), DUCI ; gagnants Syntax Terror, zer0_day, Advanced Persistent Thinkers. https://cispa.de/en/hackathon-warsaw ; https://science.nask.pl/en/news/14009
- Citations organisateurs : "les tâches sont complètement différentes à chaque fois" ; équipes qui connaissaient les papiers récents du SprintML démarraient plus vite ; "la meilleure solution = celle dont les données reconstruites collent le plus à l'original" ; une équipe a compris une tâche quand elle sait expliquer son approche. https://cispa.de/en/interview-hackathon-championship : évaluation automatique avec scoreboard comme dans leur cours à Saarland, soumission sur leurs serveurs.
- Le cadre "évaluateur" : l'infra d'évaluation est séparée de JURECA ; les organisateurs ont assuré des permanences de nuit (Vienne).

## 2. Deux dépôts d'équipes publics (seule matière technique)

### 2a. !!1337, Vienne, 1re puis 4e finale : https://github.com/florianzainzinger1/cispa-hackathon-championship-vienna [A]
- DUCI/inférence d'appartenance (ResNet-18, sous-ensembles de 100 images, TPR@5%FPR) : les moyennes étaient trop faibles (dropout), features de valeurs extrêmes par sous-ensemble (NLL max, marge de logit min, percentile 10 de confiance) : 0,141 -> 0,200 au leaderboard.
- Attribution texte : DeBERTa-v3 base/large + ModernBERT-large, pseudo-labels du test, TTA, ensemble sensible à la longueur.
- Suppression BitMark (100 images 512x512, MSE <= 0,08) : pipeline modulaire = perturbation par inpainting (Fast-Sand) + re-rendu Image-GS (splatting gaussien 2D) + diffusion img2img + petites rotations/flou + correction couleur par histogramme.
- Aucun score privé publié ; données organisateurs absentes.

### 2b. zer0_day, Varsovie : https://github.com/yauhenifutryn/cispa-warsaw-2026 [A, très utile pour la logistique]
- Rangs : DUCI #9 (0,0389 vs meilleur 0,0085), PII #2 (0,5253 vs 0,5328), détection texte #5 (0,2869 vs 0,396/0,398).
- Détection watermark : features de 4 LM (Gemma-2-9B, GPT-2-large, Qwen2.5-7B, Gemma-4-E4B-it) = 68 features, 64 arbres boostés avec graines différentes, + 36 "overrides" de doublons exacts (hash MD5) entre test et train/val filigrané. Progression 0,2507 -> 0,273 -> 0,2869. Ils attribuent l'écart au top à plus gros LM auditeurs ou infos (spéculation).
- DUCI : mélange p*Exp(lambda=5)+(1-p)*KDE(pop) par maximum de vraisemblance ; trop peu de modèles de référence (1-3) ; le top avait sans doute plus de refs / LiRA.
- PII : harvest regex, génération multi-gabarits, Context-Decoding (remplacer le nom par "John Smith", différence de log-PPL), consensus. +0,04 en ~3 h.
- Pièges rapportés (voir §4) : normalisation, cuDNN, nœuds sans internet, quota d'inodes, fuite via noms de jobs Slurm, serveur qui masque les scores inférieurs au meilleur, limites de débit (5 min / 9 min).
- Soumission : HTTP POST avec en-tête X-API-Key (Varsovie ; notre format peut différer).

### 2c. Autres
- https://github.com/sinonchum/ai-watermark-hackathon-prep : dépôt de préparation personnel POUR NOTRE ÉDITION (3 tâches : PixelSeal/TrustMark/StegaStamp, 33 endpoints, 3500 images unlearning). 0 étoile, 8 commits, squelette vide wm/ met/ unlearn/. Cite comme papiers recommandés par les organisateurs (email 17 nov, non vérifiable) : Pixel Seal (arXiv 2512.16874), Model Equality Testing (arXiv 2410.20247), SCRUB (arXiv 2302.09880). Contient (selon la lecture) le code de réservation cispahack : source non officielle, ne pas s'y fier, ne rien en réutiliser. [NV]
- Page GitHub sprintml : repos BitMark, ToxicBench, etc. Aucun dépôt d'évaluateur/leaderboard ni de hackathon visible (10 sur 52 repos listés). https://github.com/sprintml
- Pour Paris 2025, Stockholm, Munich, Barcelone, Finale : aucun code ni slide d'équipe trouvé. [NV] Pas de contenu HuggingFace/Discord/LinkedIn exploitable (LinkedIn bloqué, HTTP 999).
- Aucune information trouvée sur le code de l'évaluateur ni le serveur de leaderboard hors ce que dit zer0_day. [NV]

## 3. Transposition aux trois tâches (classement coût / gain) [I sauf mention]
Hypothèse transversale : la structure "30 % public / 70 % caché" et la ré-évaluation finale sur plus de données (Paris) sont confirmées pour d'anciennes éditions [V partiel].

| # | Idée | Tâche | Coût | Gain attendu | Source / base |
|---|---|---|---|---|---|
| 1 | Pipeline de soumission validé hors-ligne (schéma, dtype, forme, taille <= 20 Mo / 450 Mo, nombre de lignes, noms) + dry-run sur un faux fichier avant chaque envoi | T1 T2 T3 | 1-2 h, faible | Évite une perte totale | lessons zer0_day (scores masqués, formats), contexte |
| 2 | T3 : porte "< 70 % accuracy => 0" : évaluer sur un jeu de rétention proxy (ex. images propres, split de validation si disponible) avant toute soumission ; garder un checkpoint sûr du modèle d'origine comme fallback | T3 | faible | Évite le score 0 | porte de l'énoncé |
| 3 | T1 : baseline "empreinte spatiale" (le rectangle du watermark est un sous-rectangle) + features sur un détecteur de résidu ; trouver ce qui marche par éditions ; calibrer le seuil par image en moyenne mIoU, pas globalement | T1 | moyen | élevé | analogie extremes-value features d'!!1337 (statistiques robustes plutôt que moyennes) |
| 4 | T1 : exploiter les trois watermarks ayant un décodeur connu (TrustMark, StegaStamp publics ; PixelSeal Meta) : cartes de confiance par décodeur sur fenêtres glissantes -> masque, puis post-traitement morphologique (rectangle englobant) | T1 | moyen-élevé | élevé | logique de décodage, pas de source de hackathon |
| 5 | T1 : doublons/fuites : vérifier hash/similarité entre test et données d'exemple (zer0_day a gagné +0,014 par 36 doublons hash) ; sans labels le gain est nul mais cela coûte 10 min | T1 T3 | très faible | faible-moyen | zer0_day [A] |
| 6 | T2 : modèle de décision en 2 niveaux famille puis modèle (MCC) ; usage des 9 refs d'entraînement (sans limite) pour entraîner un classifieur sur des empreintes de réponses (style, longueur, refus, tokenisation) et ne dépenser les 100 requêtes/endpoint qu'avec des prompts choisis par discriminabilité mesurée sur les refs | T2 | moyen | élevé | analogie : features de LM auditeurs de zer0_day ; classif. DeBERTa d'!!1337 pour attribution texte |
| 7 | T2 : ensemble de graines/arbres (64 arbres) et features de LM multiples ; pseudo-labels ; test-time augmentation | T2 | moyen | moyen | zer0_day, !!1337 |
| 8 | T3 : unlearning "teacher-student" / SCRUB / bruit d'oubli + fine-tuning de rétention avec pseudo-labels du modèle d'origine sur les 3500 images (pas de données de rétention) ; contrôle par inférence d'appartenance sur les images oubliées via perte (valeurs extrêmes, mélange de distributions de pertes) | T3 | moyen-élevé | élevé | SCRUB (arXiv 2302.09880) [papier recommandé, non vérifié], MIA d'!!1337/zer0_day |
| 9 | T3 : estimer forget_score par attaque d'appartenance proxy (percentile/NLL max) avant de soumettre | T3 | faible | moyen | !!1337 (features d'extrêmes) |
| 10 | Répartition : 1 personne par tâche, un budget de soumission par tâche, bilan à H+8, H+14, H+18 | tout | nul | élevé | Barcelone (rép. selon les forces), Munich (quarts), WUT |
| 11 | Séparer validation locale (split des refs/images) du classement public ; ne retenir une variante que si les deux montent | tout | faible | élevé | principe général, 70 % cachés |
| 12 | Boucle rapide de GPU : modèles préchargés en cache partagé, HF_HUB_OFFLINE=1, scripts Slurm prêts | T1 T3 | faible | moyen | zer0_day |

## 4. Pièges récurrents rapportés [A, issus surtout de zer0_day]
1. Serveur qui masque les scores inférieurs au meilleur : on ne voit pas une régression ; garder son propre journal de versions/scores.
2. Limites de débit (5-9 min) : planifier les soumissions ; ici T2 = 1 soumission / 30 min, donc ne pas gaspiller et ne pas finir à la dernière minute.
3. Mauvaise normalisation d'entrée ((x/255-0,5)/0,5 vs autres) : perte cible 3,4 et précision 0,24 ; vérifier le prétraitement exact du modèle avant tout.
4. Nœuds GPU de login (RTX 8000) : crash cuDNN ; travail GPU sur A100 uniquement.
5. Nœuds de calcul sans internet : variables HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE, poids préchargés.
6. Quota d'inodes partagé du projet (4,4 M fichiers) atteint plusieurs fois : nettoyer __pycache__, shards CSV ; ne pas écrire des millions de petits fichiers (ex. masques en PNG).
7. Fuite de méthode via noms de jobs Slurm visibles par les autres équipes : noms opaques.
8. Trop peu de modèles de référence pour la calibration (MIA/DUCI) : le top en avait plus.
9. Mean-based signals trop faibles : préférer statistiques d'extrêmes.
10. Surajustement public/privé : les organisateurs ré-évaluent sur plus de données (Paris) ; l'inférence d'écarts de classement sur 30 % est bruyante. [V partiel pour Paris]
11. Présentation au jury : chaque membre doit pouvoir expliquer ; prévoir 30-45 min de préparation avant la fin (WUT, Munich). Pour notre édition, le classement est "rang combiné sur les 3 tâches" selon le brief : vérifier si une présentation compte.
12. Remontée tardive possible (CDL Chorrico) et leader précoce non garanti (4braincells 1er puis 4e) : ne pas abandonner une tâche.

## 5. Non vérifiable / à ne pas affirmer
- Aucune approche de SeeWeedZ, We are all Alex, Loki, A5 Hackyu, Polarnutt, The White House, Pentagon, Team ZOS, CATByte, CDL Chorrico de Limon, SarAI, Syntax Terror, APT n'a été trouvée.
- Le détail "ce qui a fait gagner la finale" n'est pas public ; seule phrase : "approche innovante" et efficacité.
- Ce que fait l'évaluateur (modèle évaluateur, ré-évaluation) : seulement les phrases ci-dessus.
- Les scores de zer0_day sont auto-déclarés ; deux sources donnent 0,396 et 0,398 pour le meilleur.
- Le dépôt sinonchum prétend citer des emails organisateurs de 2026 : non vérifié.
- Les gains de transposition (§3) sont des estimations qualitatives, pas mesurées.
- Nos 3 tâches (localisation, égalité de modèles, unlearning) sont nouvelles : aucune solution passée directement applicable.
