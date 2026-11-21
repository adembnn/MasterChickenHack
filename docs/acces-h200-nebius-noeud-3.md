# Troisième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 21 novembre 2026 à 18h45 UTC, depuis le laptop de Yassir et depuis
Hermes `ami` sur NemoClaw. Les commandes ci-dessous ont été exécutées.

## Ce que c'est

- VM `computeinstance-e01xmbm7xkz7h69bn5` (plateforme `gpu-h200-sxm`)
- Projet `default-project-eu-west1`, même tenant que les deux autres nœuds
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque de démarrage 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 200 GiB de RAM
- **Préemptible** (prix spot). IP publique dynamique. Sauver les résultats souvent.

| | Nœud 1 | Nœud 2 | Nœud 3 (celui-ci) |
|---|---|---|---|
| Identifiant | `computeinstance-e01w9a2hev5w1t9mdb` | `computeinstance-e01s01aed56r0tdfyy` | `computeinstance-e01xmbm7xkz7h69bn5` |
| Nom console | `green-starfish-instance-6` | `fuchsia-mosquito-instance-6` | (création 21 nov.) |
| IP publique | `66.201.7.7` | `66.201.4.154` | `66.201.6.94` |
| IP privée | `10.48.0.47` | `10.48.0.101` | `10.48.0.159` |
| Empreinte hôte ED25519 | `SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps` | `SHA256:VnLzDfZRhxVp9zIGpPWMuvzsnrqxEhWFLRCDHgrFgDY` | `SHA256:dyCoQ3+e1LGqjlkY7XB3nA3qoBA3p7PQid+yjn9ojWA` |

Les trois VM sont dans `default-network` / `default-subnet-eqxzbx6y`. Depuis le nœud 1, le
nœud 3 répond en **~2 ms** sur `10.48.0.159`, port 22 ouvert. Copier les données par l'IP
**privée**, pas par le public.

Guides des autres nœuds :

- Nœud 1 : [`acces-h200-nebius.md`](acces-h200-nebius.md)
- Nœud 2 : [`acces-h200-nebius-noeud-2.md`](acces-h200-nebius-noeud-2.md)

## Comptes

Comptes Linux : `hermes`, `adem`, `titouan`, `yassir`, `ubuntu`. Seul `yassir` a `sudo`
(bootstrap). Les agents n'ont **pas** de `sudo`. Chaque agent utilise son compte et sa
**propre** clé privée.

Vérification de l'empreinte hôte à la première connexion :

```bash
ssh-keyscan -t ed25519 66.201.6.94 2>/dev/null | ssh-keygen -lf -
# attendu : SHA256:dyCoQ3+e1LGqjlkY7XB3nA3qoBA3p7PQid+yjn9ojWA
```

Si l'empreinte ne correspond pas, ne pas se connecter et prévenir Yassir.

| Agent | Compte SSH | Empreinte de la clé publique autorisée |
|---|---|---|
| Hermes `ami` | `hermes` | `SHA256:GYvBY4LKZrmHf/g+mzu3+fWaDG2cSieJPBZYPg4rgEI` |
| Adem (`adembnn`) | `adem` | `SHA256:sHpMJSmi3RA98ou5n2VkUMeDiCtLDpaSbMh4Gy7RT0w` |
| Titouan (`LaRacine23`) | `titouan` | `SHA256:djnqKKh8yCddhiZzRoKizk+sP2CWtedCeJa0kiDj2uw` |
| Yassir (laptop) | `yassir` | clé `/home/boudda/sshkey/vultr_ed25519` |

```bash
# Laptop Yassir
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 yassir@66.201.6.94 nvidia-smi

# Agent Adem
ssh -o IdentitiesOnly=yes -i /chemin/vers/cle-privee-adem adem@66.201.6.94 nvidia-smi

# Agent Titouan
ssh -o IdentitiesOnly=yes -i /chemin/vers/cle-privee-titouan titouan@66.201.6.94 nvidia-smi
```

Clés **publiques** installées :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrcQHTRXMvoEw7wPBkshhzFP9mQL5oyd7YFsPIVy2gP hermes-ami-nebius-h200
```

Groupe commun `eac-h200` : `yassir`, `adem`, `titouan`, `hermes`.

## Depuis Hermes `ami` sur NemoClaw

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_3_config eac-h200-3 nvidia-smi
```

Config vérifiée : utilisateur `hermes`, clé `/clients/ami/.ssh/eac_nebius_h200`, known_hosts
`/clients/ami/.ssh/eac_nebius_h200_3_known_hosts`. Le test a vu le GPU et `torch.cuda` à `True`.

## Environnement partagé déjà prêt

Synchronisé depuis le nœud 2 (`10.48.0.101`) par le réseau privé (~9,1 Go, ~30 s) :

- `/home/hermes/eac/kit/.venv/` : python 3.12, torch 2.5.1 cu124, trustmark, onnxruntime-gpu, …
- `/home/hermes/eac/kit/models/` : PixelSeal, TrustMark-Q, StegaStamp
- `/home/hermes/eac/kit/release/` : train / test officiels
- `/home/hermes/eac/kit/wmloc/` + scripts T1
- `/home/hermes/eac/t3-codex/model.pt` et `forget_images.npz` (copiés depuis le nœud 1)

Lecture pour les autres comptes (`chmod -R a+rX` sur le kit, `/home/hermes` en `751`).
**Personne n'écrit dans `/home/hermes/eac/kit`.** Écrire dans `~/runs/<essai>/`.

```bash
/home/hermes/eac/kit/eac-run -c "import torch, wmloc; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA H200
```

Exemple T1 sans rien installer :

```bash
cd /home/hermes/eac/kit
PYTHONPATH=/home/hermes/eac/kit ./.venv/bin/python decode_localize.py \
  --data release --calibrate --limit 20 --calib-out ~/calib_essai.json
```

## Lancer un essai sans se marcher dessus

Une H200 par nœud → **un entraînement lourd à la fois par nœud**.

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Si la mémoire utilisée dépasse ~2 GiB, quelqu'un travaille déjà : attendre ou changer de nœud.

Répartition suggérée (à confirmer sur le [bus](https://github.com/adembnn/EAC/issues/2)) :

- Nœud 1 (`66.201.7.7`) : souvent saturé — vérifier avant.
- Nœud 2 (`66.201.4.154`) : T1 / T3 selon charge.
- Nœud 3 (`66.201.6.94`) : **nouveau** — prioritaire pour les sweeps T1/T3 tant que les deux autres sont pleins.

## Copier des données entre nœuds

Peering Hermes installé dans les deux sens (clés `id_eac_peering` sur nœud 1 et nœud 3,
pubkeys croisées dans `authorized_keys` des trois nœuds).

```bash
# depuis le nœud 1, vers le nœud 3
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@10.48.0.159:~/mon_dossier/

# depuis le nœud 3, depuis le nœud 2
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" hermes@10.48.0.101:~/eac/kit/ ~/eac/kit/
```

## Règles compétition (rappel)

1. **T2 gelée** pour les agents hors Titouan. Pas de `/chat/completions` ni `/33-model-equality`.
2. **Une soumission toutes les ~30 minutes** par équipe ; le serveur garde le meilleur score.
3. **Clé API jamais dans git ni dans une ligne de commande visible.** `~/.eac_env` + `~/.curlcfg` en `chmod 600`.
4. Endpoint T3 : `http://34.147.229.55/submit/31-unlearning` — T1 : `/submit/…` watermark (voir kit).

## Vérifier en quatre commandes

```bash
# 1. accès et GPU
ssh -o IdentitiesOnly=yes -i /chemin/vers/ma-cle moncompte@66.201.6.94 \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'

# 2. environnement partagé
ls /home/hermes/eac/kit/release/train | head -3
ls /home/hermes/eac/kit/models

# 3. python + torch
/home/hermes/eac/kit/eac-run -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. décodage T1 court
cd /home/hermes/eac/kit && PYTHONPATH=$PWD ./.venv/bin/python decode_localize.py \
  --data release --calibrate --limit 20 --calib-out ~/verif.json
```

Si l'étape 1 échoue avec `Permission denied (publickey)`, la clé publique n'est pas dans
`~/.ssh/authorized_keys` du compte visé : le dire sur le bus.
