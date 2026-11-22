# Septième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 21 novembre 2026 à 23h55 UTC. **IP publique** `204.12.190.177`.
Attention : ce nœud est sur le **même VPC que le nœud 6** (`10.96.0.0`, subnet
`default-subnet-v2keqmro`), **pas** le VPC des nœuds 1–5 (`10.48.0.0`). Le peering
privé vers 1–5 **ne marche pas** ; copier les données via l'IP **publique**.

## Ce que c'est

- VM console : `teal-baboon-instance-1`
- Identifiant : `computeinstance-u00hnaskmmsqb67qfr`
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 200 GiB de RAM
- **Préemptible**. IP publique dynamique. Sauver les résultats souvent.

| | Nœud 1 | Nœud 2 | Nœud 3 | Nœud 4 | Nœud 5 | Nœud 6 | Nœud 7 (celui-ci) |
|---|---|---|---|---|---|---|---|
| Nom | `green-starfish-instance-6` | `fuchsia-mosquito-instance-6` | `sapphire-skunk-instance-5` | `peach-rooster-instance-5` | `amber-swan-instance-5` | `sapphire-perch-instance-9` | `teal-baboon-instance-1` |
| IP publique | `66.201.7.7` | `66.201.4.154` | `66.201.6.94` | **aucune** | **aucune** | `204.12.188.1` | `204.12.190.177` |
| IP privée | `10.48.0.47` | `10.48.0.101` | `10.48.0.159` | `10.48.0.219` | `10.48.0.130` | `10.96.0.42` | `10.96.0.58` |
| Empreinte hôte ED25519 | `SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps` | `SHA256:VnLzDfZRhxVp9zIGpPWMuvzsnrqxEhWFLRCDHgrFgDY` | `SHA256:dyCoQ3+e1LGqjlkY7XB3nA3qoBA3p7PQid+yjn9ojWA` | `SHA256:/36HFyO4qzbBV3Qgz/e+cdUhe6l8Ikx077pUFGviVU8` | `SHA256:E9fX8FmzNc6w6vjmrrB7GC32VfJUE9Y1KeSW11ofvFw` | `SHA256:lyhVAA6mTI6j6193vbXO8GxFOA+tNI7/Dz/CGAkfNb0` | `SHA256:lm4VVJ8XyQBYqbQGd60V52v5LYOG2sWv1ns36yczxb8` |

Réseau nœud 7 : `default-network` / `default-subnet-v2keqmro` (VPC `10.96`, comme nœud 6).

Autres guides : [`acces-h200-nebius.md`](acces-h200-nebius.md),
[`acces-h200-nebius-noeud-2.md`](acces-h200-nebius-noeud-2.md),
[`acces-h200-nebius-noeud-3.md`](acces-h200-nebius-noeud-3.md),
[`acces-h200-nebius-noeud-4.md`](acces-h200-nebius-noeud-4.md),
[`acces-h200-nebius-noeud-5.md`](acces-h200-nebius-noeud-5.md),
[`acces-h200-nebius-noeud-6.md`](acces-h200-nebius-noeud-6.md),
[`acces-h200-nebius-noeud-8.md`](acces-h200-nebius-noeud-8.md).

## Accès SSH

Depuis le laptop de Yassir (direct, pas de ProxyJump) :

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 yassir@204.12.190.177
```

Empreinte à vérifier à la première connexion :

```bash
ssh-keyscan -t ed25519 204.12.190.177 2>/dev/null | ssh-keygen -lf -
# attendu : SHA256:lm4VVJ8XyQBYqbQGd60V52v5LYOG2sWv1ns36yczxb8
```

## Comptes

Mêmes comptes : `hermes`, `adem`, `titouan`, `yassir`, `ubuntu`.
Les agents (`hermes`, `adem`, `titouan`) n'ont **pas** de `sudo`. Groupe `eac-h200` partagé.
`yassir` a `NOPASSWD` sudo (câblage / sync kit uniquement).

| Agent | Compte | Empreinte clé publique |
|---|---|---|
| Hermes `ami` | `hermes` | `SHA256:GYvBY4LKZrmHf/g+mzu3+fWaDG2cSieJPBZYPg4rgEI` |
| Adem | `adem` | `SHA256:sHpMJSmi3RA98ou5n2VkUMeDiCtLDpaSbMh4Gy7RT0w` |
| Titouan | `titouan` | `SHA256:djnqKKh8yCddhiZzRoKizk+sP2CWtedCeJa0kiDj2uw` |

```bash
# Adem
ssh -o IdentitiesOnly=yes -i /chemin/cle-adem adem@204.12.190.177 nvidia-smi

# Titouan
ssh -o IdentitiesOnly=yes -i /chemin/cle-titouan titouan@204.12.190.177 nvidia-smi
```

Clés publiques installées :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrcQHTRXMvoEw7wPBkshhzFP9mQL5oyd7YFsPIVy2gP hermes-ami-nebius-h200
```

## Depuis Hermes `ami` sur NemoClaw

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_7_config eac-h200-7 nvidia-smi
```

Config directe (pas de ProxyJump) : utilisateur `hermes`, clé
`/clients/ami/.ssh/eac_nebius_h200`, known_hosts
`/clients/ami/.ssh/eac_nebius_h200_7_known_hosts`.

## Environnement partagé

Synchronisé depuis le nœud 6 (`204.12.188.1`) **par IP publique** (même VPC 10.96,
mais rsync public reste le chemin fiable depuis le laptop) :

- `/home/hermes/eac/kit/` : venv, models, release, wmloc, scripts T1, `eac-run`
- `/home/hermes/eac/t3-codex/model.pt` + `forget_images.npz` (si présent)

Lecture seule pour les autres (`chmod -R a+rX` sur le kit, `/home/hermes` en `751`).
Écrire dans `~/runs/<essai>/`, **pas** dans le kit partagé.

```bash
/home/hermes/eac/kit/eac-run -c "import torch, wmloc; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA H200
```

## Peering

Clé Hermes `id_eac_peering` sur ce nœud (`eac-peering-hermes-h200-7`). Pubkey croisée
avec les nœuds 1 et 6 (et les autres peers déjà dans `authorized_keys` de hermes).
Empreinte :

```text
SHA256:1BlT25oY+yiBM9v4UVmPVuCFftK2fVL9lCZH3SXba4M
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILZ8b7msSIGXyTy+jhd4LAJuwYr1kjJbpDvWlagik+Mb eac-peering-hermes-h200-7
```

Rsync **via public** (pas le privé `10.96.0.58` depuis le VPC 10.48) :

```bash
# depuis nœud 6 → nœud 7
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@204.12.190.177:~/mon_dossier/
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
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 yassir@204.12.190.177 \
  'nvidia-smi -L; ls /home/hermes/eac/kit/models | head; /home/hermes/eac/kit/eac-run -c "import torch;print(torch.cuda.is_available())"'
```
