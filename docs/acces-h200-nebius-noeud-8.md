# Huitième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 22 novembre 2026 à 00h26 UTC. **Pas d'IP publique.**
Accès uniquement en **ProxyJump** via le nœud 6 (`204.12.188.1`) ou le nœud 7
(`204.12.190.177`). Même VPC `10.96.0.0` / subnet `default-subnet-v2keqmro`
que les nœuds 6–7 ; **pas** le VPC des nœuds 1–5 (`10.48.0.0`).

## Ce que c'est

- VM console : `olive-caterpillar-instance-6`
- Identifiant : `computeinstance-u00j8xrjv9pg6t9s77`
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 200 GiB de RAM
- **Préemptible**. Sauver les résultats souvent.

| | Nœud 6 | Nœud 7 | Nœud 8 (celui-ci) |
|---|---|---|---|
| Nom | `sapphire-perch-instance-9` | `teal-baboon-instance-1` | `olive-caterpillar-instance-6` |
| IP publique | `204.12.188.1` | `204.12.190.177` | **aucune** |
| IP privée | `10.96.0.42` | `10.96.0.58` | `10.96.0.55` |
| Empreinte hôte ED25519 | `SHA256:lyhVAA6mTI6j6193vbXO8GxFOA+tNI7/Dz/CGAkfNb0` | `SHA256:lm4VVJ8XyQBYqbQGd60V52v5LYOG2sWv1ns36yczxb8` | `SHA256:Lz2ic9iRL0vtCL5islWoIf6ztw5jG5hnqNCi+v4ejaI` |

Réseau : `default-network` / `default-subnet-v2keqmro` (VPC `10.96`).

Autres guides : [`acces-h200-nebius.md`](acces-h200-nebius.md),
[`acces-h200-nebius-noeud-6.md`](acces-h200-nebius-noeud-6.md),
[`acces-h200-nebius-noeud-7.md`](acces-h200-nebius-noeud-7.md).

## Accès SSH

Depuis le laptop de Yassir, jump via nœud 6 :

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@204.12.188.1" \
  yassir@10.96.0.55
```

Via nœud 7 :

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@204.12.190.177" \
  yassir@10.96.0.55
```

Empreinte à vérifier (depuis le jump, sur le privé) :

```bash
ssh yassir@204.12.188.1 'ssh-keyscan -t ed25519 10.96.0.55 2>/dev/null | ssh-keygen -lf -'
# attendu : SHA256:Lz2ic9iRL0vtCL5islWoIf6ztw5jG5hnqNCi+v4ejaI
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
# Adem (via nœud 6)
ssh -o IdentitiesOnly=yes -i /chemin/cle-adem \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /chemin/cle-adem -W %h:%p adem@204.12.188.1" \
  adem@10.96.0.55 nvidia-smi

# Titouan
ssh -o IdentitiesOnly=yes -i /chemin/cle-titouan \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /chemin/cle-titouan -W %h:%p titouan@204.12.188.1" \
  titouan@10.96.0.55 nvidia-smi
```

Clés publiques installées :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrcQHTRXMvoEw7wPBkshhzFP9mQL5oyd7YFsPIVy2gP hermes-ami-nebius-h200
```

## Depuis Hermes `ami` sur NemoClaw

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_8_config eac-h200-8 nvidia-smi
```

Config avec ProxyJump via `eac-h200-6` (`204.12.188.1`) : utilisateur `hermes`,
clé `/clients/ami/.ssh/eac_nebius_h200`, known_hosts
`/clients/ami/.ssh/eac_nebius_h200_8_known_hosts`.

## Environnement partagé

Synchronisé depuis le nœud 6 (`10.96.0.42` → `10.96.0.55`, VPC privé) :

- `/home/hermes/eac/kit/` : venv, models, release, wmloc, scripts T1, `eac-run`
- `/home/hermes/eac/t3-codex/model.pt` + `forget_images.npz`

Lecture seule pour les autres (`chmod -R a+rX` sur le kit, `/home/hermes` en `751`).
Écrire dans `~/runs/<essai>/`, **pas** dans le kit partagé.

```bash
/home/hermes/eac/kit/eac-run -c "import torch, wmloc; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA H200
```

## Peering

Clé Hermes `id_eac_peering` sur ce nœud (`eac-peering-hermes-h200-8`). Pubkey croisée
avec les nœuds 6 et 7. Empreinte :

```text
SHA256:g/QDDXFFhEE1/R0X5mDMd1OMt+moAsxJm18j4okpUs4
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILQGkWFhoHIcyubMKikT7KuVm/42X8b/36aOxC+MjEQO eac-peering-hermes-h200-8
```

Rsync **privé VPC 10.96** :

```bash
# depuis nœud 6 → nœud 8
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@10.96.0.55:~/mon_dossier/
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
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@204.12.188.1" \
  yassir@10.96.0.55 \
  'nvidia-smi -L; ls /home/hermes/eac/kit/models | head; /home/hermes/eac/kit/eac-run -c "import torch;print(torch.cuda.is_available())"'
```
