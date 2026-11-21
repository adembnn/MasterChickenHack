# EAC : European AI Championship Paris (21-22 nov. 2026)

Ce fichier est le contexte global, lu par tous les agents (Claude, Codex, Cursor : `AGENTS.md` pointe ici).
Équipe : Adem (`adembnn`), Titouan (`LaRacine23`), Yassir (`yassirboudda`).

## Mission
**Les TROIS tracks sont obligatoires.** Seules les équipes qui ont un score sur les trois tâches sont classées.
Classement général (lu sur le leaderboard le 21/11) : sur chaque tâche, les scores sont ramenés entre 0 et 1 sur l'ensemble des équipes (la meilleure équipe a 1, la plus basse 0), puis on fait la moyenne des trois. Ce n'est donc pas un classement par rang : l'écart au meilleur compte, et finir dernier sur une tâche vaut 0 même avec un score honorable.
Un délai s'applique entre deux soumissions sur une même tâche. Seule la meilleure soumission compte.
Le code s'arrête dimanche 22/11 à 12:00. Les 6 meilleures équipes font un pitch de 6 min à 13:00.

| Track | Ce qu'on rend | Score | Dossier |
|---|---|---|---|
| T1 Localisation de filigrane | `.pt` avec un masque binaire 128×128 pour 1020 images | mIoU (moyenne IoU propre / filigrané) | `tracks/t1-watermark/` |
| T2 Model Equality | CSV `endpoint,family,model` pour 33 API LLM | 0,1·MCC famille + 0,9·MCC modèle | `tracks/t2-model-equality/` |
| T3 Unlearning | poids float32 d'un ViT-B/16 qui a oublié 3500 images | oubli × rétention, 0 si précision < 70 % | `tracks/t3-unlearning/` |

Leaderboard : http://34.147.229.55/leaderboard_page. Le public ne couvre que 30 % des données et la note finale vient des 70 % cachés : ne pas sur-optimiser le public.

## Où est quoi
- `CLAUDE.md` : ce contexte global. Ne le modifier que pour une décision d'équipe, dans `## Décisions`.
- `tracks/<track>/CONTEXTE.md` : l'état du track (méthode en cours, meilleur score, prochain lot, qui fait quoi). Le responsable du track le tient à jour.
- `tracks/<track>/` : le code du track. Chaque track est indépendant pour qu'on ne se marche pas dessus.
- `equipe/<prénom>.md` : le contexte personnel de chacun, que seul son propriétaire édite (machine, outils, compte Jülich, ce que fait son agent en ce moment).
- `reports/` : les rapports de recherche des agents (200 lignes max, chacun nomme la décision qu'il débloque). `reports/sources/` contient les consignes officielles, les slides et la doc GPU.
- `docs/decisions.md` : le journal daté des décisions.

## Communication entre agents : GitHub Issues
Les issues GitHub du dépôt servent de bus. Tous les agents, quel que soit l'humain derrière, suivent ce protocole :
1. **Au début de chaque session**, lire le bus et l'issue de son track :
   `gh issue view 2 -R adembnn/EAC --comments` puis `gh issue list -R adembnn/EAC --label <t1|t2|t3>`.
2. **Après chaque lot** (une soumission, un résultat, un blocage), poster un commentaire de 3 lignes sur l'issue du track : ce qui est fait (avec le score ou la sortie de commande), ce qui change, la suite.
   `gh issue comment <n> -R adembnn/EAC --body "…"`
3. **Question pour un humain ou un autre agent** : commentaire sur le bus (#2) avec `@pseudo` et le label `question`. Pour une décision bloquante, ouvrir une issue avec le label `decision`.
4. **Ressource partagée** (nœud GPU, budget de requêtes T2, créneau de soumission) : annoncer sur le bus avant de l'utiliser.
5. Signer chaque commentaire : `[agent de <prénom> · <track>]`.

## Règles du dépôt
- Un lot = une branche `<track>/<sujet>` ; on fusionne dans `main` seulement ce qui est vérifié (score local ou sortie de commande collée dans la PR ou l'issue).
- Ne jamais commiter les données, les poids, les soumissions ni les journaux (`.gitignore`). Ils vivent sur Jülich : `/p/scratch/training2666/<chef>/<équipe>/`.
- Les secrets (clés API du leaderboard, jetons) vont dans `.env`, jamais commités et jamais collés dans une issue.
- **Budget T2 irréversible** : 100 réponses par endpoint pour toute la compétition. Aucune requête sur `ep-*` sans un plan posté dans l'issue T2. Les `ref-*` servent à tout régler d'abord.
- Soumissions limitées : T2 une toutes les 30 min. On annonce chaque soumission sur l'issue du track.
- Ne jamais collaborer avec une autre équipe : c'est la disqualification.

## Calcul : JURECA (Jülich)
Projet `training2666`, réservation `cispahack`, partition `dc-gpu`, 1 nœud par job = 4×A100 40 Go.
Équipe Jülich **MasterChickenHack**, chef `destors1` (Titouan) ; `banni1` (Adem), `boudda1` (Yassir). Dossier partagé : `/p/scratch/training2666/destors1/MasterChickenHack/` (`Hackathon_Watermark_Localisation_Task/`, `model_equality/`, `unlearning/`).
Connexion pas à pas (SSH, MFA, `teammate.sh`) : `reports/notes/connexion-cluster.md`, écrit par Titouan. Ne jamais lancer `hackathon_setup.sh`, c'est le travail du chef. Les jetons d'API des tâches arrivent par mail au premier membre inscrit dans le formulaire d'équipe.
À chaque connexion : `jutil env activate -p training2666`.

## Décisions (la plus récente gagne)
Le journal complet est dans `docs/decisions.md`.
- 2026-11-21 : les essais locaux (données, environnements, poids) vivent dans `E:\EAC-lab` sur le PC d'Adem, hors dépôt, supprimable après le hack.
- 2026-11-21 : dépôt privé créé, LaRacine23 et yassirboudda invités en écriture. Communication des agents par GitHub Issues (#2 = bus).
