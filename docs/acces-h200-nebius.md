# Accès SSH à la H200 Nebius

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

## Accès pour un autre agent de l'équipe

L'agent génère sa propre paire SSH sur sa machine. Il transmet à Yassir
sa clé publique `.pub` et le nom de compte souhaité pour qu'il lui ouvre un
compte sans `sudo` sur la VM. Chaque clé privée reste sur la machine où elle
est utilisée. Git contient les instructions et l'empreinte hôte.
