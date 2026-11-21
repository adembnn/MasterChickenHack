# Sixième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 21 novembre 2026 à 21h00 UTC. **IP publique** `204.12.188.1`.
Attention : ce nœud est sur un **autre VPC** (`10.96.0.0`, subnet `default-subnet-v2keqmro`)
que les nœuds 1–5 (`10.48.0.0`). Le peering privé entre VPC **ne marche pas** ;
copier les données via l'IP **publique**.

## Ce que c'est

- VM console : `sapphire-perch-instance-9`
- Identifiant : `computeinstance-u00v72qvb96w27vgtq`
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 200 GiB de RAM
- **Préemptible**. IP publique dynamique. Sauver les résultats souvent.

| | Nœud 1 | Nœud 2 | Nœud 3 | Nœud 4 | Nœud 5 | Nœud 6 (celui-ci) |
|---|---|---|---|---|---|---|
| Nom | `green-starfish-instance-6` | `fuchsia-mosquito-instance-6` | `sapphire-skunk-instance-5` | `peach-rooster-instance-5` | `amber-swan-instance-5` | `sapphire-perch-instance-9` |
| IP publique | `66.201.7.7` | `66.201.4.154` | `66.201.6.94` | **aucune** | **aucune** | `204.12.188.1` |
| IP privée | `10.48.0.47` | `10.48.0.101` | `10.48.0.159` | `10.48.0.219` | `10.48.0.130` | `10.96.0.42` |
| Empreinte hôte ED25519 | `SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps` | `SHA256:VnLzDfZRhxVp9zIGpPWMuvzsnrqxEhWFLRCDHgrFgDY` | `SHA256:dyCoQ3+e1LGqjlkY7XB3nA3qoBA3p7PQid+yjn9ojWA` | `SHA256:/36HFyO4qzbBV3Qgz/e+cdUhe6l8Ikx077pUFGviVU8` | `SHA256:E9fX8FmzNc6w6vjmrrB7GC32VfJUE9Y1KeSW11ofvFw` | `SHA256:lyhVAA6mTI6j6193vbXO8GxFOA+tNI7/Dz/CGAkfNb0` |

Réseau nœud 6 : `default-network` / `default-subnet-v2keqmro` (autre VPC que 1–5).

Autres guides : [`acces-h200-nebius.md`](acces-h200-nebius.md),
[`acces-h200-nebius-noeud-2.md`](acces-h200-nebius-noeud-2.md),
[`acces-h200-nebius-noeud-3.md`](acces-h200-nebius-noeud-3.md),
[`acces-h200-nebius-noeud-4.md`](acces-h200-nebius-noeud-4.md),
[`acces-h200-nebius-noeud-5.md`](acces-h200-nebius-noeud-5.md).

## Accès SSH

Depuis le laptop de Yassir (direct, pas de ProxyJump) :

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 yassir@204.12.188.1
```

Empreinte à vérifier à la première connexion :

```bash
ssh-keyscan -t ed25519 204.12.188.1 2>/dev/null | ssh-keygen -lf -
# attendu : SHA256:lyhVAA6mTI6j6193vbXO8GxFOA+tNI7/Dz/CGAkfNb0
```

## Comptes

Mêmes comptes : `hermes`, `adem`, `titouan`, `yassir`, `ubuntu`.
Les agents n'ont **pas** de `sudo`. Groupe `eac-h200` partagé.

| Agent | Compte | Empreinte clé publique |
|---|---|---|
| Hermes `ami` | `hermes` | `SHA256:GYvBY4LKZrmHf/g+mzu3+fWaDG2cSieJPBZYPg4rgEI` |
| Adem | `adem` | `SHA256:sHpMJSmi3RA98ou5n2VkUMeDiCtLDpaSbMh4Gy7RT0w` |
| Titouan | `titouan` | `SHA256:djnqKKh8yCddhiZzRoKizk+sP2CWtedCeJa0kiDj2uw` |

```bash
# Adem
ssh -o IdentitiesOnly=yes -i /chemin/cle-adem adem@204.12.188.1 nvidia-smi

# Titouan
ssh -o IdentitiesOnly=yes -i /chemin/cle-titouan titouan@204.12.188.1 nvidia-smi
```

Clés publiques installées :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrcQHTRXMvoEw7wPBkshhzFP9mQL5oyd7YFsPIVy2gP hermes-ami-nebius-h200
```

## Depuis Hermes `ami` sur NemoClaw

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_6_config eac-h200-6 nvidia-smi
```

Config directe (pas de ProxyJump) : utilisateur `hermes`, clé
`/clients/ami/.ssh/eac_nebius_h200`, known_hosts
`/clients/ami/.ssh/eac_nebius_h200_6_known_hosts`.

## Environnement partagé

Synchronisé depuis le nœud 1 (`66.201.7.7`) **par IP publique** (autre VPC) :

- `/home/hermes/eac/kit/` : venv, models, release, wmloc, scripts T1, `eac-run`
- `/home/hermes/eac/t3-codex/model.pt` + `forget_images.npz`

Lecture seule pour les autres (`chmod -R a+rX` sur le kit, `/home/hermes` en `751`).
Écrire dans `~/runs/<essai>/`, **pas** dans le kit partagé.

```bash
/home/hermes/eac/kit/eac-run -c "import torch, wmloc; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA H200
```

## Peering

Clé Hermes `id_eac_peering` sur ce nœud (`eac-peering-hermes-h200-6`). Pubkey croisée
avec les nœuds 1–5. Empreinte :

```text
SHA256:LAMPfPu7e/aCqwVJVUA4VM3rnqzdddxXibPu+RMhtiU
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIADd6FJ7Ok40IJ28xxZM4ffz8Pjsb4lLXRJopnKWz9S eac-peering-hermes-h200-6
```

Rsync **via public** (pas le privé `10.96.0.42`) :

```bash
# depuis nœud 1 → nœud 6
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@204.12.188.1:~/mon_dossier/
```

## Charge GPU

Une H200 → un job lourd à la fois. Vérifier avant de lancer :

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## Règles compétition (rappel)

1. **T2 gelée** hors agent Titouan.
2. Soumissions espacées (~10 min / équipe) ; garder le meilleur score.
3. Clé API hors git ; `~/.curlcfg` en `chmod 600`.

## Vérifier

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 yassir@204.12.188.1 \
  'nvidia-smi -L; ls /home/hermes/eac/kit/models | head; /home/hermes/eac/kit/eac-run -c "import torch;print(torch.cuda.is_available())"'
```
