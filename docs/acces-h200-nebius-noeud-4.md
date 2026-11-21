# Quatrième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 21 novembre 2026 à 19h15 UTC. **Pas d'IP publique** : tout passe par le
réseau privé `10.48.0.219`, en general via le nœud 3 (`66.201.6.94`) en ProxyJump.

## Ce que c'est

- VM console : `peach-rooster-instance-5`
- Identifiant : `computeinstance-e01jfkvbjns4sd3796`
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 200 GiB de RAM
- **Préemptible**. Pas d'IPv4 publique assignée (volontaire ou quota).

| | Nœud 1 | Nœud 2 | Nœud 3 | Nœud 4 (celui-ci) |
|---|---|---|---|---|
| Nom | `green-starfish-instance-6` | `fuchsia-mosquito-instance-6` | `computeinstance-e01xmbm7xkz7h69bn5` | `peach-rooster-instance-5` |
| IP publique | `66.201.7.7` | `66.201.4.154` | `66.201.6.94` | **aucune** |
| IP privée | `10.48.0.47` | `10.48.0.101` | `10.48.0.159` | `10.48.0.219` |
| Empreinte hôte ED25519 | `SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps` | `SHA256:VnLzDfZRhxVp9zIGpPWMuvzsnrqxEhWFLRCDHgrFgDY` | `SHA256:dyCoQ3+e1LGqjlkY7XB3nA3qoBA3p7PQid+yjn9ojWA` | `SHA256:/36HFyO4qzbBV3Qgz/e+cdUhe6l8Ikx077pUFGviVU8` |

Réseau : `default-network` / `default-subnet-eqxzbx6y`. Ping nœud 1 → nœud 4 : **~0,8 ms**.

Autres guides : [`acces-h200-nebius.md`](acces-h200-nebius.md),
[`acces-h200-nebius-noeud-2.md`](acces-h200-nebius-noeud-2.md),
[`acces-h200-nebius-noeud-3.md`](acces-h200-nebius-noeud-3.md).

## Accès SSH (obligatoire via saut)

Depuis le laptop de Yassir (saut par le nœud 3) :

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@66.201.6.94" \
  yassir@10.48.0.219
```

Ou depuis un nœud déjà sur le VPC (nœud 1/2/3), en direct sur le privé :

```bash
ssh -o IdentitiesOnly=yes -i /chemin/vers/cle moncompte@10.48.0.219
```

Empreinte à vérifier à la première connexion :

```bash
ssh-keyscan -t ed25519 10.48.0.219 2>/dev/null | ssh-keygen -lf -
# attendu : SHA256:/36HFyO4qzbBV3Qgz/e+cdUhe6l8Ikx077pUFGviVU8
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
  adem@10.48.0.219 nvidia-smi
```

Clés publiques installées (identiques au reste du cluster) :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOrcQHTRXMvoEw7wPBkshhzFP9mQL5oyd7YFsPIVy2gP hermes-ami-nebius-h200
```

## Depuis Hermes `ami` sur NemoClaw

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_4_config eac-h200-4 nvidia-smi
```

Cette config fait `ProxyJump` via `eac-h200-3-jump` (`66.201.6.94`). Test OK : GPU vu,
`torch.cuda` → `True`.

## Environnement partagé

Synchronisé depuis le nœud 3 (`10.48.0.159`) par le privé (~9,1 Go, ~25 s) :

- `/home/hermes/eac/kit/` : venv, models, release, wmloc, scripts T1, `eac-run`
- `/home/hermes/eac/t3-codex/model.pt` + `forget_images.npz`

Lecture seule pour les autres (`chmod -R a+rX` sur le kit, `/home/hermes` en `751`).
Écrire dans `~/runs/<essai>/`, **pas** dans le kit partagé.

```bash
/home/hermes/eac/kit/eac-run -c "import torch, wmloc; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# attendu : True NVIDIA H200
```

## Peering

Clé Hermes `id_eac_peering` sur ce nœud (`eac-peering-hermes-h200-4`). Pubkey croisée avec
les nœuds 1–3. Exemple depuis le nœud 3 :

```bash
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@10.48.0.219:~/mon_dossier/
```

## Charge GPU

Une H200 → un job lourd à la fois. Vérifier avant de lancer :

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Nœud 4 est utile quand 1–3 sont saturés (pas d'IP publique = moins de bruit externe).

## Règles compétition (rappel)

1. **T2 gelée** hors agent Titouan.
2. Soumissions espacées (~30 min / équipe) ; garder le meilleur score.
3. Clé API hors git ; `~/.curlcfg` en `chmod 600`.

## Vérifier

```bash
# depuis laptop via nœud 3
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 \
  -o ProxyCommand="ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 -W %h:%p yassir@66.201.6.94" \
  yassir@10.48.0.219 'nvidia-smi -L; ls /home/hermes/eac/kit/models | head; /home/hermes/eac/kit/eac-run -c "import torch;print(torch.cuda.is_available())"'
```
