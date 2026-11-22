# MasterChickenHack

Équipe : Adem ([`adembnn`](https://github.com/adembnn)), Titouan ([`LaRacine23`](https://github.com/LaRacine23)), Yassir ([`yassirboudda`](https://github.com/yassirboudda)) — European AI Championship, Paris, 21–22 novembre 2026.

Classement public au 22/11 vers 11 h (score ramené entre 0 et 1 sur les trois tâches, voir [CLAUDE.md](CLAUDE.md#mission)) : **3ᵉ**, score global **0,96** ; **1ʳᵉ place sur T1**.

| Tâche | Score public final | Fichier retenu | Dossier |
|---|---|---|---|
| T1 — Localisation de filigrane | **0,8252** | candidat 49 (sub 4089) | [`tracks/t1-watermark/`](tracks/t1-watermark/) |
| T2 — Model Equality | **1,00** | — | [`tracks/t2-model-equality/`](tracks/t2-model-equality/) |
| T3 — Unlearning | **0,4227** | `PG_RDq_70_l915c_v3d_X3b_s1` (sub 3902) | [`tracks/t3-unlearning/`](tracks/t3-unlearning/) |

Le code qui tournait sur nos machines de calcul (cartes H200) est rapatrié dans `tracks/<tâche>/serveur/`.

Le score public ne couvre qu'une partie des données ; le classement final retient les 70 % restants, cachés jusqu'à la clôture. Les trois sections ci-dessous expliquent la méthode et disent honnêtement ce qui reste incertain sur cette partie cachée.

## T1 — Localiser des filigranes invisibles (0,8252)

On reçoit des images marquées par un des trois filigranes (TrustMark, PixelSeal, StegaStamp) et on doit rendre, pour chacune, un masque pixel par pixel disant où se trouve la marque. Score = recouvrement moyen (mIoU) entre notre masque et le vrai, moyenné sur les trois filigranes.

**Méthode** : un localisateur par filigrane, entraîné uniquement sur des images que nous fabriquons nous-mêmes (jamais sur les vraies images du jeu, pour ne pas contaminer la validation). PixelSeal — le maillon faible — gagne avec des filtres passe-haut appris en première couche, puis un vote par région et des boîtes tirées du décodeur du filigrane. Chaque candidat n'est envoyé que si l'écart apparié contre le meilleur précédent a un intervalle à 95 % qui exclut zéro et que la réserve (images jamais utilisées pour choisir) a le même signe — plusieurs essais prometteurs en local ont été écartés pour ne pas passer ce test.

**Fichier final (candidat 49, 0,8252 au public, contre 0,8230 pour le 38)** : il cumule trois petits gains mesurés séparément sur le 38 — une nouvelle carte PixelSeal pour les images partiellement marquées, une moyenne de 7 localisateurs PixelSeal pour les images entièrement marquées, et un TrustMark à filtres passe-haut pour les images entièrement marquées. En validation croisée sur 510 images : +0,0048, IC95 [−0,0012 ; +0,0110], réserve +0,0047. C'est la seule exception à notre règle d'envoi, prise en fin de compétition parce que l'écart attendu était positif sur toutes les lectures. Ce qui reste à gagner : 57 % de la perte du 38 vient du contour des zones laissées propres au milieu des images entièrement marquées ; un modèle dédié à ces contours a été essayé et n'a pas battu les généralistes après affinage.

Code et rapports : [`tracks/t1-watermark/CONTEXTE.md`](tracks/t1-watermark/CONTEXTE.md), [`reports/t1-t1-assemblage.md`](reports/t1-t1-assemblage.md), [`reports/t1-t1-pixelseal.md`](reports/t1-t1-pixelseal.md), [`reports/t1-tableau.md`](reports/t1-tableau.md).

## T2 — Reconnaître un modèle de langage sans voir ses poids (1,00)

33 guichets d'API anonymes ; il faut dire lesquels servent le même modèle (même checkpoint, même quantification, même LoRA — un déguisement comme un prompt système, une persona ou un filigrane de décodage ne compte pas). On rend un CSV `endpoint,family,model`.

**Méthode**, en quatre étapes :
1. **Famille** : NLL par token sous nos modèles candidats locaux — marge minimale de 0,27 nat/token entre familles, jamais trompée.
2. **Variantes fortes (fp8)** : un détecteur calibré sur des guichets de référence (budget illimité, vérité connue), seuils fixés avant de lire les vrais guichets, confirmé sur deux moitiés indépendantes des questions.
3. **Variantes faibles (LoRA) et structure** : les variantes faibles échappent aux détecteurs de probabilités. On a reconstruit le gabarit probable de l'organisateur (symétrie par famille : des guichets déguisés qui restent des copies, des variantes concentrées ailleurs) et testé cette hypothèse un guichet à la fois, avec une prédiction écrite avant chaque résultat — hypothèse qui s'est trompée une fois (−0,20) avant d'être corrigée avec la preuve en main.
4. **Garde-fous** : détection des déguisements (prompt système, persona, température, filigrane) pour ne jamais isoler un guichet à tort.

Le 1,00 en public fige aussi des choix sur des guichets jamais vus par le score public, décidés uniquement par nos détecteurs locaux (validés en validation croisée sur les guichets réels confirmés par le score) : ce sont eux qui restent incertains pour le classement final.

Code et rapports : [`tracks/t2-model-equality/CONTEXTE.md`](tracks/t2-model-equality/CONTEXTE.md), [`reports/t2-t2-methode.md`](reports/t2-t2-methode.md), [`reports/critique/`](reports/critique/) (audits et contre-hypothèses).

## T3 — Faire oublier des classes à un modèle (0,4227)

Un ViT-B/16 à 100 classes doit « oublier » 3 500 images sans jamais être réentraîné dessus, en gardant le reste. Score = oubli × rétention (0 si la précision retenue tombe sous 70 %).

**Méthode** : on construit une cible par distillation à partir de modèles enseignants entraînés sur des partitions des images, calée sur un modèle de référence (gold) pour la perte moyenne et son écart-type, avec une ancre qui protège les images à garder. Bruit de graine de la recette : ≈ ± 0,005.

**Pistes testées et abandonnées, honnêtement** : masse de classe plus ou moins pointue, moyenne sur plus de partitions, ancre sur des images inversées — toutes dans le bruit de la recette. Deux leviers d'une autre nature testés sur notre propre calcul (pas encore un gain net confirmé) : régression directe vers la moyenne des enseignants (plafonne structurellement, abandonné) et bruit sur les derniers blocs puis redistillation (dégrade la rétention plus qu'il n'aide, abandonné). 

**Ce qui a payé le dernier matin** (scores publics, chaque cran mesuré seul) :

| Fichier | Ce qui change | Public |
|---|---|---|
| `PG_RDq_70_l915c_s1` | base : masse fausse moyennée sur 8 partitions, calée sur le gold | 0,4107 |
| `…_v2c_s1` | la perte compte double sur les versions altérées (recadrage, flou, couleur…) des images à oublier | 0,4194 |
| `…_v3d_s1` | compte triple | 0,4224 |
| **`…_v3d_X3b_s1`** | + ancre vers le modèle d'origine sur 7 110 images qui ne sont pas à oublier | **0,4227** |

Fermé au public ce matin : cibles calculées sur les copies combinées au poids (0,4184), masse fausse de 20 partitions (0,4152), enseignants entraînés seulement sur des non-membres (0,3968). Recette exacte et empreinte du fichier final : [`reports/t3-t3-redistill.md`](reports/t3-t3-redistill.md).

Code et rapports : [`tracks/t3-unlearning/CONTEXTE.md`](tracks/t3-unlearning/CONTEXTE.md), [`reports/t3-t3-methodes.md`](reports/t3-t3-methodes.md), [`reports/t3-t3-redistill.md`](reports/t3-t3-redistill.md).

## Repo

- [`CLAUDE.md`](CLAUDE.md) : contexte global de l'équipe, protocole de travail.
- `tracks/<tâche>/CONTEXTE.md` : l'état courant de chaque tâche (méthode, meilleur score, prochain pas).
- `reports/` : journal des expériences, un rapport par idée testée, y compris les échecs.
- `docs/decisions.md` : journal daté des décisions d'équipe.
- Les poids de modèles et gros fichiers (`*.pt`, `checkpoints/`) ne sont pas versionnés ; ils vivent sur nos machines de calcul et sur le serveur du défi, référencés par chemin et empreinte (sha256) dans les rapports.
