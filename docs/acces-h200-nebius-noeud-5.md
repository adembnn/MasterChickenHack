# Cinquième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 21 novembre 2026 à 19h40 UTC. **Pas d'IP publique** : tout passe par le
réseau privé `10.48.0.130`, en general via le nœud 3 (`66.201.6.94`) en ProxyJump.

## Ce que c'est

- VM console : `amber-swan-instance-5`
- Identifiant : `computeinstance-e01kkc1cyhtgxdhe9g`
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 200 GiB de RAM
- **Préemptible**. Pas d'IPv4 publique assignée.

| | Nœud 1 | Nœud 2 | Nœud 3 | Nœud 4 | Nœud 5 (celui-ci) |
|---|---|---|---|---|---|
| Nom | `green-starfish-instance-6` | `fuchsia-mosquito-instance-6` | `computeinstance-e01xmbm7xkz7h69bn5` | `peach-rooster-instance-5` | `amber-swan-instance-5` |
| IP publique | `66.201.7.7` | `66.201.4.154` | `66.201.6.94` | **aucune** | **aucune** |
| IP privée | `10.48.0.47` | `10.48.0.101` | `10.48.0.159` | `10.48.0.219` | `10.48.0.130` |
| Empreinte hôte ED25519 | `SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps` | `SHA256:VnLzDfZRhxVp9zIGpPWMuvzsnrqxEhWFLRCDHgrFgDY` | `SHA256:dyCoQ3+e1LGqjlkY7XB3nA3qoBA3p7PQid+yjn9ojWA` | `SHA256:/36HFyO4qzbBV3Qgz/e+cdUhe6l8Ikx077pUFGviVU8` | `SHA256:E9fX8FmzNc6w6vjmrrB7GC32VfJUE9Y1KeSW11ofvFw` |

Réseau : `default-network` / `default-subnet-eqxzbx6y`. Ping nœud 3 → nœud 5 : **~0,9 ms**.

Autres guides : [`acces-h200-nebius.md`](acces-h200-nebius.md),
[`acces-h200-nebius-noeud-2.md`](acces-h200-nebius-noeud-2.md),
[`acces-h200-nebius-noeud-3.md`](acces-h200-nebius-noeud-3.md),
[`acces-h200-nebius-noeud-4.md`](acces-h200-nebius-noeud-4.md),
[`acces-h200-nebius-noeud-6.md`](acces-h200-nebius-noeud-6.md).

## Accès SSH (obligatoire via saut)

Depuis le laptop de Yassir (saut par le nœud 3) :

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@66.201.6.94" \
  yassir@10.48.0.130
```

Ou depuis un nœud déjà sur le VPC (nœud 1/2/3/4), en direct sur le privé :

```bash
ssh -o IdentitiesOnly=yes -i /chemin/vers/cle moncompte@10.48.0.130
```

Empreinte à vérifier à la première connexion :

```bash
ssh-keyscan -t ed25519 10.48.0.130 2>/dev/null | ssh-keygen -lf -
# attendu : SHA256:E9fX8FmzNc6w6vjmrrB7GC32VfJUE9Y1KeSW11ofvFw
```

## Comptes

Mêmes comptes que les autres nœuds : `hermes`, `adem`, `titouan`, `yassir`, `ubuntu`.
Les agents n'ont **pas** de `sudo`. Groupe `eac-h200` partagé.

| Agent | Compte | Empreinte clé publique |
|---|---|---|
| Hermes `ami` | `hermes` | `SHA256:GYvBY4LKZrmHf/g+mzu3+fWaDG2cSieJPBZYPg4rgEI` |
| Adem | `adem` | `SHA256:sHpMJSmi3RA98ou5n2VkUMeDiCtLDpaSbMh4Gy7RT0w` |
| Titouan | `titouan` | `SHA256:djnqKKh8yCddhiZzRoKizk+sP2CWtedCeJa0kiDj2uw` |

```bash
# Adem (depuis un nœud VPC, ou ProxyJump nœud 3)
ssh -o IdentitiesOnly=yes -i /chemin/cle-adem \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /chemin/cle-adem -W %h:%p adem@66.201.6.94" \
  adem@10.48.0.130 nvidia-smi
```

Clés publiques installées (identiques au reste du cluster) :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrcQHTRXMvoEw7wPBkshhzFP9mQL5oyd7YFsPIVy2gP hermes-ami-nebius-h200
```

## Depuis Hermes `ami` sur NemoClaw

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_5_config eac-h200-5 nvidia-smi
```

Cette config fait `ProxyJump` via `eac-h200-3-jump` (`66.201.6.94`). Test OK : GPU vu,
`torch.cuda` → `True`, `~/.curlcfg` présent.

## Environnement partagé

Synchronisé depuis le nœud 3 (`10.48.0.159`) par le privé (~9,1 Go, ~40 s) :

- `/home/hermes/eac/kit/` : venv, models, release, wmloc, scripts T1, `eac-run`
- `/home/hermes/eac/t3-codex/model.pt` + `forget_images.npz`

Lecture seule pour les autres (`chmod -R a+rX` sur le kit, `/home/hermes` en `751`).
Écrire dans `~/runs/<essai>/`, **pas** dans le kit partagé.

```bash
/home/hermes/eac/kit/eac-run -c "import torch, wmloc; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA H200
```

## Peering

Clé Hermes `id_eac_peering` sur ce nœud (`eac-peering-hermes-h200-5`). Pubkey croisée avec
les nœuds 1–4. Empreinte :

```text
SHA256:VR1zWX8A7EB6XK3extk49IytpDQKR8x+XPfQ4fHl+O4
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAnByR9ICpuV/6fQzbXGD0ldFNkI/klvwt8XY88CTbzr eac-peering-hermes-h200-5
```

Exemple depuis le nœud 3 :

```bash
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@10.48.0.130:~/mon_dossier/
```

## Charge GPU

Une H200 → un job lourd à la fois. Vérifier avant de lancer :

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Nœud 5 est utile quand 1–4 sont saturés (pas d'IP publique = moins de bruit externe).

## Règles compétition (rappel)

1. **T2 gelée** hors agent Titouan.
2. Soumissions espacées (~10 min / équipe) ; garder le meilleur score.
3. Clé API hors git ; `~/.curlcfg` en `chmod 600`.

## Vérifier

```bash
# depuis laptop via nœud 3
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@66.201.6.94" \
  yassir@10.48.0.130 'nvidia-smi -L; ls /home/hermes/eac/kit/models | head; /home/hermes/eac/kit/eac-run -c "import torch;print(torch.cuda.is_available())"'
```
