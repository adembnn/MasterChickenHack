# Note pour les agents d'Adem et du 3e coéquipier : connexion au cluster JURECA

Équipe : **MasterChickenHack** (European AI Championship Paris, 21-22/11/2026).
Chef d'équipe : `destors1` (Titouan). Coéquipiers : `banni1` (Adem) et `boudda1` (Yassir). Projet de calcul : **training2666**.

Chaque personne a son propre compte et sa propre clé SSH. Ne partagez ni clé privée, ni code MFA, ni mot de passe, ni token d'API.
Règle des organisateurs : chaque équipe joue seule. Fusionner solutions ou tokens avec une autre équipe = disqualification.

## 0. Ce que l'agent ne peut pas faire à la place de l'humain
- Entrer un mot de passe, un code MFA à 6 chiffres ou un code de secours.
- Signer l'accord d'usage sur JuDoor et créer le compte (acceptation de conditions).
L'agent prépare tout et dit à l'humain exactement quoi cliquer ou taper.

## 1. Compte JuDoor (l'humain, une fois)
1. Compte sur judoor.fz-juelich.de avec l'email de l'école. Rejoindre le projet **training2666** (le responsable doit accepter).
2. Activer le MFA (appli d'authentification), sauvegarder les codes de secours dans un gestionnaire de mots de passe.
3. Sur la page du compte, section Systems : cliquer « sign the usage agreement » pour **JURECA** (et JUDAC). Attendre la coche « Usage agreement confirmed ».

## 2. Clé SSH (agent + humain)
Sur la machine de l'humain, créer une clé dédiée (ne pas écraser une clé existante) :
```bash
ssh-keygen -t ed25519 -C "hack-jureca" -f ~/.ssh/id_ed25519_jureca -N ""
cat ~/.ssh/id_ed25519_jureca.pub
```
Sur JuDoor : compte > JURECA > « Manage SSH-keys » :
- coller la clé **publique** dans « Your public key and options string » ;
- cliquer le bouton **« Use the from clause suggested above »** (restreint la clé au réseau de l'humain) ;
- « Start upload of SSH-Keys ». La clé est active en 15 minutes environ.
Attention : la clé ne marche que depuis le réseau détecté (même opérateur). Autre réseau (wifi, VPN, partage de connexion d'un autre opérateur) = « Permission denied (publickey) » : refaire l'upload avec une clause `from=` pour ce réseau.

## 3. Config SSH locale (`~/.ssh/config`)
Remplacer `<identifiant>` par `banni1` ou `boudda1` :
```
Host jureca
    HostName jureca.fz-juelich.de
    User <identifiant>
    IdentityFile ~/.ssh/id_ed25519_jureca
    IdentitiesOnly yes
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```
Avec ControlMaster, le code MFA n'est demandé qu'à la première connexion (valable 8 h).

## 4. Se connecter et tester
```bash
ssh jureca
```
- Empreinte attendue du serveur à la première connexion : `SHA256:IDGkpGi9hxyNBixHDLEXYjLeGLrfNIiuTpcVXu0Lzcc` (ED25519). Sinon, s'arrêter.
- Puis « TOTP Verification code » : l'humain tape son code.
- Diagnostic non interactif (l'agent peut le lancer) : `ssh -o BatchMode=yes -v jureca true 2>&1 | grep -E "Offering|accepts|Permission"`. « Server accepts key » = la clé est active (il ne reste que le MFA).

## 5. Brancher sur le dossier partagé de l'équipe (une fois, après que `destors1` a fini son setup)
Envoyer le script fourni par Titouan (`teammate.sh`, déjà configuré avec `TEAM_LEAD="destors1"` et `TEAM_FOLDER="MasterChickenHack"`) :
```bash
scp teammate.sh jureca:~/
ssh jureca
source ~/teammate.sh        # sur le cluster, depuis son propre compte
```
Ne jamais lancer `hackathon_setup.sh` (réservé au chef).

## 6. Travailler (chaque connexion)
```bash
jutil env activate -p training2666
cd /p/scratch/training2666/destors1/MasterChickenHack
ls
```
Dossiers des tâches (chacun avec `.venv/`, `requirements.txt`, `output/`) :
- `unlearning/` : tâche 3 (ViT-B/16, 3 500 images à faire oublier)
- `model_equality/` : tâche 2 (33 endpoints, ep-01 à ep-33 ; 100 réponses max par endpoint pour toute l'équipe)
- `Hackathon_Watermark_Localisation_Task/` : tâche 1 (localisation de watermark, grille 128x128)

Activer l'environnement : `cd <tâche> && source .venv/bin/activate`.
Ne pas recréer `.venv`, ne pas supprimer de dossiers partagés, ne pas modifier les ACL. Coordonner tout changement de paquets (environnements partagés). Mettre sorties et logs dans le dossier `output/` de la tâche.

## 7. Lancer un job GPU
Créer `main.sh` dans le dossier de la tâche (Slurm 25.11 : `--gres=gpu:X` est obligatoire) :
```bash
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

cd /p/scratch/training2666/destors1/MasterChickenHack/<tâche> || exit 1
source .venv/bin/activate
srun python <script>.py
```
```bash
mkdir -p output
sbatch main.sh
squeue -u "$USER"
```
Logs : `output/<job-id>.out` et `.err`. Après l'activation du compte, la soumission de jobs peut prendre jusqu'à 30 minutes.

## 8. Soumissions aux tâches
- Les tokens d'API arrivent par mail au premier coéquipier listé dans le formulaire d'équipe. Les garder hors des dépôts Git et des messages (variable d'environnement).
- Tâche 2 : une soumission toutes les 30 minutes ; le score public ne couvre qu'une partie des paires, ne pas sur-optimiser dessus. Utiliser les endpoints de référence `ref-01` à `ref-09` (sans limite de requêtes, réponses connues dans `practice_answers.csv`) pour valider les méthodes.
- Classement tâche 2 : http://34.147.229.55/leaderboard_page
- Le score final dépend du classement sur l'ensemble des 3 tâches.
