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

## Méthode de travail : la boucle mesurer, critiquer, corriger (demande d'Adem, 21/11)
Tous les agents de l'équipe l'appliquent, sur chaque track :
1. **Mesurer** : chaque idée devient une expérience courte avec un chiffre (score local officiel), postée sur l'issue du track avec la sortie de commande. Un raisonnement sans mesure ne décide rien.
2. **Se faire critiquer** : après chaque série de mesures, un agent critique séparé (modèle fort, spécialiste du sujet, qui n'a pas écrit la méthode) cherche ce qui est faux, fragile ou manquant, et propose au plus 5 expériences suivantes. Rapports dans `reports/critique/<track>-tour<N>.md`.
3. **Corriger et relancer** : on applique, on remesure, on recommence jusqu'à ce que le score local ne monte plus ou que le temps manque.
4. **Innover** : à chaque tour, au moins une piste hors des réponses évidentes est testée. Ne jamais rester immobile en attendant quelqu'un.
5. **Expliquer** : chaque résultat posté dit en mots simples ce qui a été testé, ce que le chiffre veut dire et ce qu'on en déduit.

## Règles du dépôt
- Un lot = une branche `<track>/<sujet>` ; on fusionne dans `main` seulement ce qui est vérifié (score local ou sortie de commande collée dans la PR ou l'issue).
- Ne jamais commiter les données, les poids, les soumissions ni les journaux (`.gitignore`). Ils vivent sur Jülich : `/p/scratch/training2666/<chef>/<équipe>/`.
- Les secrets (clés API du leaderboard, jetons) vont dans `.env`, jamais commités et jamais collés dans une issue.
- **Budget T2 irréversible** : 100 réponses par endpoint pour toute la compétition. Aucune requête sur `ep-*` sans un plan posté dans l'issue T2. Les `ref-*` servent à tout régler d'abord.
- Soumissions limitées : T2 une toutes les 30 min. On annonce chaque soumission sur l'issue du track.
- Ne jamais collaborer avec une autre équipe : c'est la disqualification.

## Calcul : JURECA (Jülich)
Projet `training2666`, réservation `cispahack`, partition `dc-gpu`, un nœud = 4×A100 40 Go.
**Règle stricte (Adem, 21/11 20:30) : un seul GPU à la fois pour le compte `banni1`** : un job au plus, avec `--gres=gpu:1`, lancé par l'orchestrateur seulement (voir Décisions).
Équipe Jülich **MasterChickenHack**, chef `destors1` (Titouan) ; `banni1` (Adem), `boudda1` (Yassir). Dossier partagé : `/p/scratch/training2666/destors1/MasterChickenHack/` (`Hackathon_Watermark_Localisation_Task/`, `model_equality/`, `unlearning/`).
Connexion pas à pas (SSH, MFA, `teammate.sh`) : `reports/notes/connexion-cluster.md`, écrit par Titouan. Ne jamais lancer `hackathon_setup.sh`, c'est le travail du chef. Les jetons d'API des tâches arrivent par mail au premier membre inscrit dans le formulaire d'équipe.
À chaque connexion : `jutil env activate -p training2666`.

## Décisions (la plus récente gagne)
Le journal complet est dans `docs/decisions.md`.
- 2026-11-21 21:20 (Adem) : **T2, les sessions d'Adem travaillent directement avec Titouan sur ses vraies réponses** (journaux bruts des `ref-*` et `ep-*`) pour produire le CSV à soumettre. Inchangé : seul Titouan appelle l'API et soumet.
- 2026-11-21 20:50 (Adem) : **toutes les H200 doivent tourner à plein, la 3e comprise** (remplace la réserve du nœud 3 à Titouan, décision de 20:06). Côté Adem : nœud 3 (`h200c`) = T3 (t3-poids 70 Go, t3-nonmembres 40 Go) à côté de Titouan ; nœud 5 (`h200e`) = T3 (t3-methodes, t3-simulateur) ; nœud 4 (`h200d`) = T1 ; nœud 1 surchargé, à délester.
- 2026-11-21 20:47 (Adem) : **les gros jobs vont sur les H200, pas sur Jülich**, et deux cartes de plus sont ouvertes. Côté Adem : nœud 4 (`h200d`, 10.48.0.219) = T1 à grande échelle (t1-reel 90 Go, t1-pixelseal 15 Go, partagé avec l'agent de Yassir) ; nœud 5 (`h200e`, 10.48.0.130) = T3 (t3-methodes 80 Go, t3-simulateur 50 Go). Les deux n'ont pas d'IP publique : rebond SSH par le nœud 1. Le nœud 3 reste à Titouan. Jülich : aucun job en cours ; au plus un job à un GPU, lancé par l'orchestrateur.
- 2026-11-21 20:30 (Adem) : **Jülich, règle stricte : un seul GPU à la fois pour tout le compte `banni1`.** Au plus un job en cours ou en file, et ce job demande `#SBATCH --gres=gpu:1` (jamais `gpu:2` ni `gpu:4`, jamais deux jobs, même en attente). Aucune session ne lance `sbatch` : elle demande à l'orchestrateur, qui colle la sortie de `squeue -u banni1` avant et après. Un job qui réserve plus d'un GPU est annulé puis relancé à un GPU depuis son dernier point de contrôle. Tout ce qui demande plus part sur les H200 (nœuds 1 et 2). Les `.sbatch` du dépôt qui portent encore `gpu:4` sont à corriger avant tout réemploi (durcit la décision de 20:15).
- 2026-11-21 20:15 (Adem) : **Jülich, un seul job à la fois pour tout le compte `banni1`** (un nœud = 4 A100). Aucune session ne lance `sbatch` elle-même : elle demande à l'orchestrateur, qui vérifie `squeue -u banni1` avant. À 20 h 15, 4 jobs tournaient et 3 attendaient : tous annulés sauf un.
- 2026-11-21 20:06 (Adem) : **la 3e H200 (`66.201.6.94`) est réservée à Titouan.** Les agents et sessions d'Adem n'utilisent que le nœud 1 (`h200`), le nœud 2 (`h200b`) et Jülich.
- 2026-11-21 19:46 (Adem) : **T1, on n'envoie plus par petits gains.** Un fichier T1 ne part que s'il est « solide » : gain d'au moins +0,02 sur la moyenne de réserve des trois filigranes par rapport au dernier envoi, écart apparié dont l'intervalle à 95 % exclut 0, et feu vert écrit du vérificateur. Les gains plus petits s'accumulent hors ligne. T3 : les sondes de mesure restent autorisées, sans précipitation (remplace en partie la décision de 19:05).
- 2026-11-21 19:05 (Adem) : **autorisation permanente de soumettre sur T1 et T3**, y compris des sondes destinées à mesurer (un envoi toutes les 10 min par tâche, délai partagé par l'équipe ; annoncer chaque envoi T3 sur l'issue #5). T2 reste réservé à Titouan. **Maker-checker obligatoire sur chaque track** : celui qui produit un résultat n'est pas celui qui le valide ; un vérificateur séparé recalcule les scores et relit le code avant qu'un résultat serve de base.
- 2026-11-21 19:00 : deuxième H200 `h200b` (66.201.4.154) ouverte par Yassir, réservée en priorité au track 1 ; la première (`h200`) sert aux tracks 2 et 3 ; Jülich est de nouveau utilisé pour les gros entraînements en parallèle (remplace la décision de 18:35).
- 2026-11-21 18:35 (Adem) : **tout le calcul se fait sur la H200 de Yassir** (`docs/acces-h200-nebius.md` sur la branche `yassir/baselines`, comptes `adem`, `titouan`, `yassir`, `hermes`). On n'utilise plus Jülich : sa connexion tombe et exige un code MFA à chaque fois. Une seule carte de 144 Go partagée : annoncer les gros jobs sur le bus, points de contrôle fréquents (VM préemptible).
- 2026-11-21 : soumissions T1 autorisées par Adem (1103 : 0,635 ; 1190 : 0,674). Soumissions T3 autorisées (1180 : 0,0039 ; 1197 : 0,0114). T2 : seul Titouan appelle l'API et soumet. Délai de 10 min entre deux envois sur une même tâche, partagé par l'équipe.
- 2026-11-21 : les essais locaux (données, environnements, poids) vivent dans `E:\EAC-lab` sur le PC d'Adem, hors dépôt, supprimable après le hack.
- 2026-11-21 : dépôt privé créé, LaRacine23 et yassirboudda invités en écriture. Communication des agents par GitHub Issues (#2 = bus).
