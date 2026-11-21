# Deuxième nœud H200 Nebius : accès et mode d'emploi

Guide vérifié le 21 novembre 2026 à 18h00 UTC, depuis le VPS de l'agent de Yassir et depuis le
premier nœud. Tout ce qui suit a été exécuté, pas supposé : les commandes sont celles qui ont servi
à vérifier les accès.

## Ce que c'est

- VM `fuchsia-mosquito-instance-6`, identifiant `computeinstance-e01s01aed56r0tdfyy`
- Projet `default-project-eu-west1`, tenant `plum-coral-tenant-sxp`
- Ubuntu 24.04 LTS pour GPU NVIDIA, CUDA 13, disque de démarrage 1280 GiB SSD
- NVIDIA H200, **143 771 MiB**, 16 vCPU, 196 GiB de RAM
- **Préemptible** (prix spot). Elle peut être arrêtée par le fournisseur, et son IP publique est
  dynamique. Conséquence pratique : sauver les résultats au fur et à mesure, ne pas compter sur une
  session longue sans surveillance.

| | Premier nœud | Deuxième nœud |
|---|---|---|
| Nom | `green-starfish-instance-6` | `fuchsia-mosquito-instance-6` |
| IP publique | `66.201.7.7` | `66.201.4.154` |
| IP privée | (réseau `default-network`) | `10.48.0.101` |
| Empreinte hôte ED25519 | `SHA256:z+SaIzu+B14CjewmXT/38KIub3w+ktoRqipczwSRsps` | `SHA256:VnLzDfZRhxVp9zIGpPWMuvzsnrqxEhWFLRCDHgrFgDY` |

Les deux VM sont dans le même réseau privé `default-network`, sous-réseau `default-subnet-eqxzbx6y`.
Depuis le premier nœud, le second répond en **1,2 ms** en moyenne sur `10.48.0.101`, et le port 22 est
ouvert. C'est le chemin à utiliser pour copier des données : passer par l'IP publique serait plus
lent et inutilement exposé.

## Comptes

Comptes Linux présents : `hermes`, `adem`, `titouan`, `yassir`, `ubuntu`. Aucun n'a `sudo`.
Chaque agent se connecte avec le compte qui lui a été attribué et sa **propre** clé privée. On ne
partage jamais une clé privée, et on ne copie jamais la clé d'un autre agent.

Vérification de l'empreinte hôte à la première connexion, à comparer avec la valeur du tableau :

```bash
ssh-keyscan -t ed25519 66.201.4.154 2>/dev/null | ssh-keygen -lf -
```

Si l'empreinte ne correspond pas, on ne se connecte pas et on prévient Yassir.

## Environnement déjà prêt, partagé entre les agents

Le dossier `/home/hermes/eac/kit` du deuxième nœud contient une copie complète de l'environnement de
travail, synchronisée depuis le premier nœud par le réseau privé :

- `.venv/` : python 3.12, torch 2.5.1 cu124, trustmark 0.8.0, onnxruntime-gpu, timm, videoseal
- `models/` : PixelSeal, TrustMark variante Q, StegaStamp en ONNX avec sortie logits
- `release/train` et `release/test` : 510 plus 510 images, masques 128 et métadonnées
- `wmloc/` : les wrappers officiels du kit
- `decode_localize.py` : localisation T1 par décodage, mode canevas
- `train_segmenter.py`, `postproc_t1.py`, scripts T3

Tout est en lecture pour les autres comptes (`chmod -R a+rX`, et `/home/hermes` est traversable).
**Personne n'écrit dans ce dossier partagé.** Chacun écrit dans son propre home, par exemple
`~/runs/<essai>/`.

Exemple complet pour un agent, sans rien installer :

```bash
cd /home/hermes/eac/kit
PYTHONPATH=/home/hermes/eac/kit ./.venv/bin/python decode_localize.py \
  --data release --calibrate --limit 20 --calib-out ~/calib_essai.json
```

Test réel effectué sur ce nœud après la copie : le décodage tourne et rend un mIoU, donc
l'environnement est fonctionnel, pas seulement présent.

## Lancer un essai sans se marcher dessus

Une H200 par nœud, donc **un seul entraînement lourd à la fois par nœud**. Avant de lancer :

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Si la mémoire utilisée dépasse 2 GiB, quelqu'un travaille déjà : attendre ou aller sur l'autre nœud.
Un échec fréquent et coûteux est l'erreur ONNX du décodeur StegaStamp, qui réclame 1,43 Go par tampon
dès que le lot de fenêtres grossit. En pratique : fenêtre 256 avec pas 64, pas plus de 25 fenêtres par
image, ou découper les lots.

Répartition suggérée, à ajuster sur le bus :

- Nœud 1 (`66.201.7.7`) : T1 et les essais longs.
- Nœud 2 (`66.201.4.154`) : T3, et les balayages parallèles de T1.

## Copier des données entre les deux nœuds

Une clé de rapprochement existe sur le premier nœud, `~/.ssh/id_eac_peering`, et sa clé publique est
autorisée sur le second. Commande vérifiée :

```bash
# depuis le premier nœud
rsync -a -e "ssh -i ~/.ssh/id_eac_peering" ~/mon_dossier/ hermes@10.48.0.101:~/mon_dossier/
```

Les 9 Go de l'environnement sont passés en moins de 90 secondes par ce chemin. Pour un autre compte,
la même clé de rapprochement ne fonctionne pas : demander à Yassir d'ajouter la clé publique de
l'agent au compte visé sur le second nœud, ou passer par le réseau privé avec sa propre clé si elle
est déjà autorisée.

## Règles de la compétition qui s'appliquent ici

1. **T2 est gelée pour les agents autres que celui de Titouan.** Aucune requête vers
   `/chat/completions` ni `/33-model-equality`. Le budget est irréversible et le compteur monte même
   sur les endpoints de référence.
2. **Une soumission toutes les 30 minutes par équipe**, et le serveur ne garde que le meilleur score.
   Inutile de soumettre deux fichiers dans la même fenêtre. Le message d'erreur donne le nombre de
   secondes restantes, il suffit d'attendre.
3. **La clé API ne se met jamais dans un dépôt ni dans une ligne de commande visible.** Usage
   recommandé, fichier `~/.eac_env` en `chmod 600`, puis un fichier de configuration curl en `600`
   pour les envois :

   ```bash
   printf 'header = "X-API-Key: %s"\n' "$EAC_API_KEY" > ~/.curlcfg && chmod 600 ~/.curlcfg
   curl -sS -K ~/.curlcfg -F "file=@submission.pt" http://34.147.229.55/submit/31-unlearning
   ```

4. **HTTP en clair, IP nue.** La clé voyage lisible sur le réseau. Ne rien envoyer depuis un Wi-Fi
   public partagé.

## Vérifier que tout est bon en quatre commandes

```bash
# 1. accès et GPU
ssh -o IdentitiesOnly=yes -i /chemin/vers/ma-cle moncompte@66.201.4.154 \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'

# 2. environnement partagé
ls /home/hermes/eac/kit/release/train | head -3
ls /home/hermes/eac/kit/models

# 3. python et torch voient le GPU
/home/hermes/eac/kit/.venv/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. décodage T1 en deux minutes
cd /home/hermes/eac/kit && PYTHONPATH=$PWD ./.venv/bin/python decode_localize.py \
  --data release --calibrate --limit 20 --calib-out ~/verif.json
```

Si l'étape 1 échoue avec `Permission denied (publickey)`, la clé publique de l'agent n'est pas dans
`~/.ssh/authorized_keys` du compte visé sur le deuxième nœud : le dire sur le bus, ne pas tenter de
contourner.

## Ce qui reste à faire côté Yassir

- Installer les clés publiques des agents d'Adem et de Titouan sur le deuxième nœud pour leurs
  comptes respectifs. Les comptes existent, les clés n'ont pas été vérifiées depuis ce côté.
- Décider si le dossier partagé doit rester chez `hermes` ou migrer vers un emplacement commun. En
  l'état, il fonctionne et il est en lecture seule pour les autres, ce qui suffit pour la nuit.
