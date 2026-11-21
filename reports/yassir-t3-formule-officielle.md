# T3 · ce que dit la formule officielle, et pourquoi nos candidats font zéro

21 novembre, 20h45 UTC. Le jeu de données `SprintML/unlearning` contient `task_template.py` avec la
formule de score complète. On ne l'avait pas lue. Voilà ce qu'elle dit, mot pour mot en substance.

## La formule

- **Score moyenne et écart-type.** Pour chaque groupe d'images, `d = |moyenne(L) - gold_mean| + |écart
  type(L) - gold_std|`, puis `1 - clip(d / d_original)`. Les deux groupes, images à oublier et copies
  altérées cachées, comptent pour moitié.
- **Score de prédiction.** `1 - clip(moyenne KL(p_gold || p_notre) / moyenne KL(p_gold || p_original))`,
  avec `p_gold` la moyenne des sorties de plusieurs modèles réentraînés, à température 2.
- **Score d'oubli** = moyenne/écart-type multiplié par prédiction.
- **Score de rétention.** `acc(contrôle) / acc_original(contrôle)` multiplié par
  `1 / (1 + KL(p_original || p_notre))` sur un jeu de contrôle caché.
- **Score final** = oubli multiplié par rétention, et **zéro sous 70 % d'accuracy** sur une validation
  cachée.

## Ce que ça change dans notre lecture

Le gold n'est pas « un modèle qui ne sait rien » sur ces images, c'est **un modèle qui se trompe d'une
façon précise**, celle d'un modèle qui n'a jamais vu l'image et qui choisit la classe des images
retenues qui lui ressemblent. La consigne du serveur est même explicite : il faut prédire **quelle
mauvaise classe** le gold choisirait.

Nos deux premiers essais distillaient vers le modèle original vu sous augmentation. La classe
mémorisée reste la première, la perte reste proche de zéro, et le gold a une perte élevée sur ces
images : le score moyenne/écart-type est donc mauvais, et le score de prédiction aussi. Zéro, deux fois,
pour cette raison précise.

## Essai du jour, supprimer la classe mémorisée

On met à zéro la masse de la première classe et on renormalise le reste, puis on distille sur la tête
et les deux derniers blocs.

| Réglage | Confiance après | Entropie | KL vs original | Même classe qu'avant | Accord observation |
|---|---|---|---|---|---|
| 8 pas, masse gardée 0 | 0,925 | 0,222 | 0,364 | 88,9 % | 91,0 % |
| 12 pas, masse gardée 0 | 0,828 | 0,523 | 1,237 | 67,3 % | 69,2 % |
| 8 pas, masse gardée 0,05 | 0,9995 | 0,003 | 0,007 | 99,9 % | 100 % |

Trois leçons. Un réglage trop faible ne change pas la classe décidée : 0,05 de masse et la sortie est
identique à 99,9 %. Un réglage moyen déplace la distribution sans déplacer la décision : 88,9 % de même
classe alors que la KL monte à 0,36, et déjà 9 % de dérive sur les images jamais touchées. Un réglage
fort casse tout.

## Conclusion honnête

Sur T3, la difficulté n'est pas la force du désapprentissage, c'est **de deviner la mauvaise classe
qu'un modèle réentraîné choisirait**. Ça demande la distribution du jeu d'entraînement, ou le jeu de
contrôle caché. Sans l'un des deux, toute la famille de méthodes locales plafonne : soit on ne change
rien, soit on casse la rétention et la porte d'utilité.

On sait maintenant régler la force au pas près grâce au jeu d'observation. Ce qu'on ne sait pas, c'est
la direction. C'est là qu'il faut un input de l'organisation, pas plus de réglage.
