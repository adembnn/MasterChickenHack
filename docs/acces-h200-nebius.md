# Accès SSH à la H200 Nebius

> **Deux nœuds depuis le 21 novembre 18h.** Ce document décrit le premier nœud
> (`green-starfish-instance-6`, `66.201.7.7`). Le deuxième nœud, `fuchsia-mosquito-instance-6`
> (`66.201.4.154`, IP privée `10.48.0.101`), a son propre mode d'emploi avec l'environnement déjà
> installé : [`acces-h200-nebius-noeud-2.md`](acces-h200-nebius-noeud-2.md).

Accès vérifiés le 21 novembre 2026. La VM `green-starfish-instance-6`
(`computeinstance-e01w9a2hev5w1t9mdb`) est dans `default-project-eu-west1`.
Elle a une NVIDIA H200 de **143 771 MiB**. Son adresse IPv4 publique actuelle
est `66.201.7.7`.

La VM est préemptible et son IP publique est dynamique. Si la connexion
échoue après un arrêt, vérifier l'état et l'adresse dans la console Nebius.
Vérifier l'identité de la VM avant de mettre à jour son adresse dans SSH.

## Depuis le laptop de Yassir

```bash
ssh -o IdentitiesOnly=yes -i /home/boudda/sshkey/vultr_ed25519 yassir@66.201.7.7
```

À la première connexion, comparer l'empreinte hôte ED25519 affichée avec
celle relevée sur la VM vérifiée :

```text
SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps
```

Dans la VM, vérifier le GPU avec :

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Depuis Hermes `ami` sur NemoClaw

Lancer cette commande depuis le terminal de l'agent :

```bash
ssh -F /clients/ami/.ssh/eac_nebius_h200_config eac-h200 nvidia-smi
```

L'alias se connecte au compte Linux `hermes`, sans `sudo`. Il utilise la clé
dédiée `/clients/ami/.ssh/eac_nebius_h200` et vérifie l'empreinte hôte via
`/clients/ami/.ssh/eac_nebius_h200_known_hosts`. Ces fichiers résident sur
NemoClaw. Le terminal de l'agent a confirmé l'accès SSH et le pilote CUDA
a détecté un GPU.

## Depuis les agents d'Adem et de Titouan

Les deux comptes sont actifs, chacun avec la clé publique fournie dans le
bus d'équipe. Ils voient la H200 et disposent de leur propre dossier
personnel. Les comptes n'ont pas de droit `sudo`.

| Agent | Compte SSH | Empreinte de la clé publique autorisée |
|---|---|---|
| Adem (`adembnn`) | `adem` | `SHA256:sHpMJSmi3RA98ou5n2VkUMeDiCtLDpaSbMh4Gy7RT0w` |
| Titouan (`LaRacine23`) | `titouan` | `SHA256:djnqKKh8yCddhiZzRoKizk+sP2CWtedCeJa0kiDj2uw` |

Lancer la commande depuis la machine qui a généré la paire SSH. Choisir la
ligne de son compte et remplacer le chemin par celui de la **clé privée
locale** correspondante. Au premier accès, vérifier l'empreinte hôte indiquée
plus haut.

```bash
# Agent d'Adem
ssh -o IdentitiesOnly=yes -i /chemin/vers/cle-privee-adem adem@66.201.7.7 nvidia-smi

# Agent de Titouan
ssh -o IdentitiesOnly=yes -i /chemin/vers/cle-privee-titouan titouan@66.201.7.7 nvidia-smi
```

Pour retrouver la paire locale, comparer l'empreinte du fichier `.pub` avec
le tableau via `ssh-keygen -lf /chemin/vers/cle.pub`. Utiliser ensuite le
fichier privé associé à cette paire avec `-i`.

Clés **publiques** installées sur la VM :

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM2pZtwVWgs6yqvQ1GJWa2PQZRtFHQo0zpBX8Yc5QHeF adem-eac-h200
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIpT/lgijgbORUBu3lC06TtGPgbGIgseBoVeU1zXZG28 titouan-eac-h200
```

Si SSH retourne `Permission denied (publickey)`, vérifier le compte dans la
commande et l'empreinte de la clé utilisée. Une fois connecté, vérifier les
processus avec `nvidia-smi` et coordonner l'usage de la carte unique sur le
[bus d'équipe](https://github.com/adembnn/EAC/issues/2).

## Accès pour un nouvel agent

L'agent génère sa propre paire SSH sur sa machine. Il transmet à Yassir
sa clé publique `.pub` et le nom de compte souhaité pour qu'il lui ouvre un
compte sans `sudo` sur la VM. Chaque clé privée reste sur la machine où elle
est utilisée. Git contient les instructions et l'empreinte hôte.
