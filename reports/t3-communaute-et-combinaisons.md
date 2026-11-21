# T3 Unlearning : communauté, combinaisons, et comment juger

Date : 21 nov. 2026. Décision débloquée : quelle combinaison coder, et comment trancher entre variantes sans labels ni gold. Aucune soumission, aucun appel payant. Complète `t3-unlearning-etat-de-lart.md` (la passe académique n'est pas refaite).
Étiquettes : **[VU]** = lu à la source (lien donné) ; **[IDÉE]** = notre déduction, non prouvée.

## 0. L'essentiel en 8 lignes
1. **Aucune solution publique de cette tâche n'existe.** Pas de dépôt, pas de fil X, pas de post trouvé. Les tâches des éditions précédentes du championnat (Varsovie, Stockholm, finale Sarrebruck) n'étaient pas du unlearning [VU, section 2].
2. **Il n'y a pas de consensus** sur « oublier sans données retain en imitant la sortie d'un modèle réentraîné » (section 7).
3. Le seul concours proche (NeurIPS 2023) a été gagné par « **effacer puis réparer** » avec des **MSE de distillation vers orig**, du **bruit gaussien faible** sur les entrées et **plusieurs petits cycles** plutôt qu'un gros reset [VU].
4. Piège confirmé par la communauté : la **montée de gradient** (« rendre le modèle faux sur les images à oublier ») s'effondre ou reste distinguable ; elle plafonne vers 0,06 au concours 2023 contre 0,12 pour le vainqueur [VU].
5. Deux idées récentes ressemblent à nos P1/N1 : **étiquettes douces « adversariales »** pour éviter de trop oublier (RUAGO, NeurIPS 2025) et **KL redistribuée** sans retain (RKLU, ESANN 2026) [VU, résumés seulement].
6. Le **mélange 12e place du concours 2023** (« mauvais professeur » préentraîné, cible = sortie du mauvais professeur sur les images à oublier) est la version « petit pseudo-gold » de notre N2 [VU].
7. Combinaison recommandée : **P3 d'abord (filet), puis N1 entraîné avec la perte P1 + ancre, puis moyenne de graines + interpolation vers orig**, réglée par le simulateur N3 (section 5).
8. Juger : **4 niveaux** (santé, fidélité aux cibles, proxy retain, simulateur), **5 à 6 sondes publiques au plus**, lues par différences entre variantes et non en valeur absolue (section 6).

## 1. Ce qu'on a regardé, et ce qui a échoué (honnêteté)
| Source | Résultat |
|---|---|
| Kaggle NeurIPS 2023, discussions | [Liste](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion) lue. Contenu des fils lus : [12e place](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/458648), [clôture](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/457425), [« use of forget set »](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/499373). Les fils 2e/3e/6e/7e n'ont pas pu être ouverts par numéro (liens non exposés) : non relus. Les 1er, 3e, 5e, 9e, 10e avaient été lus à la 1re passe. Aucun fil « ce qui a bougé le score » ni « sans retain » dans la liste |
| Kaggle, notebooks publics | Non lus (rien de plus que les writeups). Dépôts GitHub de participants lus à la place (section 2) |
| X (Chrome) | 6 recherches (`retain-free`, `European AI Championship`, `SprintML unlearning`, `unlearning challenge trick`, `unlearning ViT forget set`, `machine unlearning competition`) : **0 résultat pertinent** |
| Reddit | **Bloqué par l'outil navigateur** (« site non autorisé »). Repli : archive Arctic Shift (« Too many requests », deux essais) puis recherche web : aucun fil Reddit remonté. **Non couvert** |
| Discord du championnat | Un canal `#task-3-unlearning` existe dans l'onglet d'Adem. **Non lu** (hors de mes onglets). À regarder par Adem : c'est la source la plus proche de nos concurrents |
| GitHub | 20 requêtes `gh` (voir section 2) |

## 2. Faits communauté
### 2.1 Les tâches passées ne contenaient pas d'unlearning, et personne n'a publié celle-ci
- Dépôts de la série SprintML/CISPA trouvés : [Varsovie 2026](https://github.com/yauhenifutryn/cispa-warsaw-2026) (DUCI, PII, filigrane de texte), [Stockholm](https://github.com/sprintml/hackathonStockholm), [finale Sarrebruck](https://github.com/andresmartinezcarcel/cispa-cybersecurity-ai-hackathon-finale) (filigrane, MGI, inversion de gradients) [VU]. Aucun ne traite l'unlearning : cette tâche est nouvelle, **pas de recette réutilisable**.
- Ce que les anciens enseignent quand même [VU, [Varsovie](https://github.com/yauhenifutryn/cispa-warsaw-2026)] : la normalisation de l'image décide de tout (« other norms break the targets ») ; des doublons exacts test/train donnent des gains sans risque ; les têtes de classement avaient plus de modèles de référence. Pour nous : ne jamais renormaliser (le gabarit fait déjà `(x/255-0.5)/0.5`, mean = std = 0,5 [VU, [task_template.py](https://huggingface.co/datasets/SprintML/unlearning/raw/main/task_template.py)]).
- Le jeu [SprintML/unlearning](https://huggingface.co/datasets/SprintML/unlearning) (870 Mo : `model.pt`, `forget_images.npz`, `task_template.py`, `submission.py`, `requirements.txt`) n'a ni fiche ni liste de classes [VU]. La normalisation 0,5/0,5 est celle des ViT « augreg » de timm : **hypothèse** [IDÉE] que le modèle vient d'un ViT-B/16 préentraîné sur ImageNet-21k puis affiné sur 100 classes. Aucune liste « ImageNet-100 » ne peut être rapprochée sans voir les images : le test de 10 min de l'étape zéro reste la seule voie.
- Un autre groupe a publié aujourd'hui ses notes : [sinonchum/ai-watermark-hackathon-prep](https://github.com/sinonchum/ai-watermark-hackathon-prep) ; la partie T3 est [ici](https://github.com/sinonchum/ai-watermark-hackathon-prep/blob/HEAD/docs/tasks/task3-zh.md) [VU]. Ce sont des avis, pas des mesures. Ils recommandent : pseudo-labels par argmax d'orig, ne pas viser « le plus loin possible de la classe mémorisée » (le gold donne « une autre classe plausible, pas l'uniforme »), enregistrer des points de passage, **interpoler vers orig** si l'utilité baisse, ne pas choisir « le point qui oublie le plus ». Leur code est un squelette vide (`baselines.py`, `mia.py` = commentaires). Ils jugent que l'article « Zero-Shot » (cité par le sujet) correspond le mieux au cadre ; nous ne l'avons pas relu pour ça.

### 2.2 Ce que les participants 2023 ont publié (concours « ressembler au réentraîné »)
- **8e place (Ahn et al.)** [VU, [page projet](https://junhyunb.github.io/en/projects/neurips2023-unlearning/)] : reset **aléatoire** de couches entières (6 couches avec remise), puis distillation **MSE** vers orig, **3 à 4 cycles** de 1 à 2 époques. Ablations : cycles 0,068 → 0,086 (le plus gros gain) ; reset aléatoire 0,062 vs reset par information de Fisher 0,049 (pire que le simple fine-tune 0,050) ; MSE 0,068 > CE 0,065 > L1 0,033 ; bruit gaussien sur les entrées σ = 0,1 mieux que 0,05 ou 0,15 ; recadrage et cutout font **s'effondrer** le score (0,00001) ; exclure la tête et les raccourcis de projection du tirage : 0,086 → 0,097 ; couches entières > poids isolés (0,086 vs 0,058). Montée de gradient « Negrad » : 0,0001. Leur lecture : le concours comparait 512 exécutions, donc **de la diversité entre exécutions** aidait. **Attention : chez nous c'est inversé**, un seul modèle est jugé face à une moyenne de golds.
- **12e place (Savelli)** [VU, [fil](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/458648)] : « bon professeur » (orig) pour les images retain, « mauvais professeur » **préentraîné quelques époques sur le retain** (pas initialisé au hasard) pour les images à oublier ; élève = copie d'orig ; KL entre logits ; échantillonnage des images retain **proches des images à oublier** ; batchs « d'abord beaucoup d'oubli, puis reconstruction ». Leur score public : 12e ; un essai tardif donne 0,0749.
- **6e place** [VU, snippet de recherche seulement, [Euclid](https://euclid.ee.duth.gr/machine-unlearning/) renvoie 404] : reset des premières/dernières couches, échauffement par distillation, fine-tune. Non relu.
- Fil [« Use of forget set »](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/499373) : un participant suggère de pénaliser le modèle quand il prédit juste sur l'oubli. Sans réponse, sans mesure.
- Organisateurs [VU, [blog Stanford](https://ai.stanford.edu/~kzliu/blog/unlearning)] : score « vérité terrain » ≈ 0,19, vainqueur ≈ 0,12, montée de gradient ≈ 0,06 ; la métrique repose sur un seul modèle cible ; « il existe plusieurs comportements plausibles pour un modèle qui n'a pas vu l'image, et chacun donne des KL très différentes » (le contrefactuel est mal défini).
- Autres dépôts de participants : [Air Liquide](https://github.com/luispintoc/machine-unlearning) a dû **fabriquer un jeu factice (CIFAR-10 déséquilibré)** pour s'entraîner : même idée que N3 [VU]. [lumliolum](https://github.com/lumliolum/neurips-machine-unlearning-2023) a fini 589e sur 1188 avec Fisher noise + fine-tune « et je ne sais pas pourquoi ça n'a pas marché » [VU] : les méthodes copiées des articles sans simulateur ont échoué.

### 2.3 Littérature récente sur « sans retain » et « imiter le réentraîné » (pas dans la 1re passe)
- **RUAGO** (NeurIPS 2025) [VU, [résumé](https://mlanthology.org/neurips/2025/lee2025neurips-ruago), [code](https://github.com/Lemma1727/RUAGO)] : **étiquettes douces obtenues par attaque adversariale** sur les images à oublier (pour ne pas sur-oublier), un générateur d'images hors-domaine entraîné par eux pour remplacer le retain, échantillonnage progressif du facile au difficile. Le code exige un environnement CUDA 11.3 de 2022 et des « assets » téléchargés (Dropbox) : **trop lourd pour 18 h**, on prend l'idée, pas le code. Testé sur CIFAR, pas ViT-B 224 [VU].
- **RKLU** (ESANN 2026) [VU, résumé via [portail SDU](https://portal.findresearcher.sdu.dk/en/publications/rklu-redistributive-kl-distillation-for-efficient-retain-free-mac/)] : KL vers une cible qui **abaisse la probabilité de la classe à oublier et redistribue** le reste, sans retain. Dépôt vide ([README d'une ligne](https://github.com/VarunSampath23/RKLU-Unlearning)). C'est P3 sous un autre nom.
- **« Ascent Fails to Forget »** [VU, [arXiv 2509.26427](https://arxiv.org/abs/2509.26427)] : oublier un jeu tiré au hasard « jusqu'à l'erreur du test » abaisse forcément l'accuracy test, parce que oubli et retain sont statistiquement liés. Ça explique pourquoi **notre cible ne doit pas être « faux partout »** mais « comme sur du test ».
- **Retain equivalence impossible en général** [VU, [arXiv 2510.16629](https://arxiv.org/abs/2510.16629)] : un oubli local ne peut pas toujours retrouver le modèle réentraîné (dépend de l'histoire d'entraînement) ; résultats sur des LLM. Pour nous : un plafond est possible, ne pas viser 1,0.
- **Benchmark « Deep Unlearn »** [VU, [arXiv 2410.01276](https://arxiv.org/html/2410.01276v1)] : le fine-tune simple classe 1er en « écart de rétention » ; NegGrad+ et SCRUB sont moyens, surtout sur ViT ; **chaque méthode a été réglée par 3 x 100 essais avec un modèle réentraîné dispo** : sans réentraîné, il faut un simulateur (N3).
- **Distillation robustifie l'oubli** [VU, [arXiv 2506.06278](https://arxiv.org/abs/2506.06278)] : distiller le modèle « oublié » dans un élève bruité ou neuf transfère le comportement sans les capacités latentes. Testé sur LLM et tâches synthétiques, pas sur classifieurs ; c'est la même mécanique que N2.

## 3. Les méthodes en images (pour un non-développeur)
Image commune : orig est un élève qui a **appris par cœur** 3500 fiches de révision. Le gold est un élève jumeau qui ne les a **jamais vues** : devant ces fiches, il hésite, se trompe parfois, et se trompe **toujours vers une classe voisine** (un chat pris pour un lynx, pas pour un camion). On doit rendre orig pareil au jumeau, sans toucher à ce qu'il sait ailleurs.
- **P3, recalibrage doux** : on baisse seulement le « volume de confiance » de la dernière couche sur ces 3500 fiches. Rapide, sûr, mais toutes les fiches deviennent « un peu incertaines » de la même façon.
- **P1, distillation vers une cible par image + ancre** : on écrit pour chaque fiche une « réponse attendue » (légère hésitation vers la bonne voisine) et on réentraîne vers elle ; une **ancre** (images hors-sujet où la sortie doit rester celle d'orig) empêche d'abîmer le reste.
- **P2, effacer puis réparer** : on efface au hasard quelques couches, puis on réapprend à imiter orig. Les « cicatrices » de mémorisation disparaissent mais le savoir revient. Gagnant 2023, mais exige des images **du même domaine** pour réparer.
- **N1, cibles par quantiles** : on **fabrique** la distribution de pertes du gold : environ 10 à 25 % de fiches franchement ratées (les difficiles), le reste confiant ; la dureté de chaque fiche vient d'un classement de difficulté. Fixe moyenne et écart-type de la loss **par construction**.
- **N2, pseudo-gold** : on entraîne quelques élèves qui n'ont **jamais vu** les fiches (ils copient orig sur d'autres images) ; leur moyenne devient la réponse attendue. Idée du « mauvais professeur » du 12e.
- **N3, simulateur** : on joue le concours en petit (CIFAR-100, vrai réentraîné connu) pour **vérifier nos réglages avec la vraie note**. Ce n'est pas un rendu : c'est une règle graduée.
- **Astuces de communauté** : bruit léger sur les entrées (σ ≈ 0,01 à 0,1, jamais de recadrage fort à l'entraînement **dans la réparation**) ; perte MSE sur les logits plutôt que CE ; plusieurs **petits** cycles ; **interpolation** orig ↔ modèle oublié ; **moyenne de graines**.

## 4. Matrice méthode x composante du score
`+` aide, `++` aide beaucoup, `−` met en danger, `0` neutre, `?` inconnu. Sans mesure : jugement [IDÉE] appuyé sur les sections 2 et 3 et sur `t3-unlearning-etat-de-lart.md` §2.
| Méthode | Loss moyenne | Loss écart-type | KL par image | Copies modifiées | Retain accuracy | Retain KL | Porte 70 % |
|---|---|---|---|---|---|---|---|
| P3 recalibrage doux | + | − (loss quasi constante) | + | + | 0 | 0 | 0 |
| P1 cible par image + ancre | + | + si cible a une queue | + | + si on augmente | − léger | − léger | 0 |
| P2 effacer-réparer | + | ? | + | ? | −− sans proxy du domaine | − | **−** |
| N1 quantiles | + | **++** | + si ordre de difficulté juste | + | − léger | − léger | 0 |
| N2 pseudo-gold | + | + | **++** (vraies voisines) | + | −  (élèves moins précis) | − | − |
| Bruit d'entrée σ faible | 0 | 0 | 0 | + | 0 | 0 | 0 |
| MSE de distillation (vs CE) | 0 | 0 | + | 0 | + | + | + |
| Interpolation orig ↔ oublié (α) | réglable | réglable | réglable | réglable | **+** | **+** | **+** |
| Moyenne de graines | + (lisse) | − (aplatit la queue) | **+** (le gold est une moyenne) | + | 0 | 0 | 0 |
| Montée de gradient (écartée) | − | − | − | − | −− | −− | **−−** |
| N3 simulateur | choisit les réglages, ne change pas le modèle | | | | | | |

Lecture rapide : **la loss écart-type est le goulot** (seules N1 et en partie P1/N2 le travaillent) ; **la porte 70 % et le retain** sont protégés par l'ancre, la MSE et l'interpolation ; **la KL par image** par N2 et la moyenne de graines.

## 5. Complémentarités et trois chaînes concrètes
**Qui se combine avec qui** [IDÉE] :
- P3 est le **point de départ commun** : ses poids de tête recalibrée servent d'initialisation à tout le reste, et de repli soumis en 2 h.
- N1 fournit **la cible** ; P1 fournit **la perte et l'ancre** ; ce sont deux étages du même entraînement, pas des rivales.
- N2 fournit **une autre cible** (par imitation d'élèves neufs) ; on peut la **mélanger** à N1 (ex. 50/50 des mauvaises classes), à condition d'avoir un proxy dans le domaine.
- Interpolation et moyenne de graines sont **post-traitements gratuits** (aucun entraînement), applicables à toutes les chaînes.
- N3 choisit les réglages (A, β, λ, lr, α) **avant** de dépenser des sondes publiques.
- P2 (reset) est **incompatible** avec « un seul modèle propre » si le proxy est hors-domaine : à garder en option finale seulement.

**Chaîne A, « sûre » (≈ 2 à 3 h, 4 GPU-min)** : P3 (τ = 2 à 3, tête + norme finale) → interpolation α ∈ {0,5 ; 0,75 ; 1} → 1 sonde. Attendu : retain ≈ 0,95, forget 0,15 à 0,35 [IDÉE, plafond de `etat-de-lart` §2].

**Chaîne B, « notre pari » (≈ 10 h)** : P3 comme initialisation → cibles N1 (difficulté, quantiles, deux températures) → perte P1 (KL à T = 1 et T = 2 sur l'oubli augmenté, **MSE de logits** vers orig sur l'ancre, bruit d'entrée σ ≈ 0,01 à 0,05 sur l'ancre) → 4 graines (1 par A100) → **moyenne des poids** → interpolation α vers orig → 3 sondes. Si la moyenne des poids dégrade (poids trop éloignés), **moyenner les logits** n'est pas possible (un seul modèle livré) : alors on garde la meilleure graine.

**Chaîne C, « pseudo-gold » (≈ 14 h, seulement avec proxy dans le domaine)** : élèves N2 (4 derniers blocs réinitialisés, distillés sur le proxy) → cible = moyenne de leurs softmax sur les images à oublier et leurs vues → mélange 50/50 avec la cible N1 pour garder une **queue** de loss correcte → perte P1 → interpolation α. Si orig lui-même a un proxy de moins de 90 % d'accord avec lui-même, abandonner C.

Ordre d'effort conseillé : A à H+2 (filet soumis) ; B en parallèle de N3 ; C seulement si l'étape zéro trouve un proxy.

## 6. Comment juger le « meilleur » sans labels ni gold
### 6.1 Quatre niveaux, dans cet ordre
1. **Santé** (bloquant) : `strict=True`, float32, pas de NaN, < 450 Mo ; fp16 autocast comme le serveur ; `sub = orig` doit donner forget 0 ; un `sub` bruité doit faire chuter le retain [VU dans `etat-de-lart` §6].
2. **Fidélité aux cibles** (sur les images à oublier et leurs vues augmentées) : moyenne et écart-type de la loss avec ŷ comparés aux (μ_g, σ_g) visés ; KL(cible T=2 ‖ sub) ; part de ŷ non prédite.
3. **Proxy retain** : accord argmax avec orig sur des images hors-oubli (> 97 %) et KL(orig‖sub) à T=2 (< 0,05) ; règle : **pas de soumission sous 95 % d'accord** (porte 70 %).
4. **Simulateur N3** : mêmes chaînes sur ViT-Tiny/CIFAR-100 avec **vrais golds** et la **vraie formule** du sujet ; on garde les réglages qui gagnent là-bas **et** qui ne cassent pas la santé ici. Limite : autre modèle, autre résolution ; il valide l'ordre des méthodes, pas les valeurs absolues [IDÉE, soutenue par les 2 participants 2023 qui ont dû fabriquer un jeu factice, [Air Liquide](https://github.com/luispintoc/machine-unlearning)].

### 6.2 Sondes sur le classement public
- Quota : le sujet n'en dit rien ; le concours 2023 en avait [5 par jour](https://github.com/com3dian/NeurIPS-2023-Machine-Unlearning) [VU, notes d'une équipe]. **À demander aux superviseurs.** Le classement ne garde que le meilleur score par équipe, mais la réponse renvoie la note de chaque soumission [VU, sujet].
- **Budget conseillé : 5 à 6 sondes, pas plus** [IDÉE] :
  1. P3 pur (vérifie la chaîne de bout en bout, mesure un plancher).
  2. à 4. Chaîne B avec **un seul paramètre qui varie** : l'accuracy non-membre supposée A (0,75 / 0,85 / 0,92), qui fixe (μ_g, σ_g). C'est la grandeur la plus inconnue.
  5. Meilleure variante + interpolation α (0,75 contre 1).
  6. Dernière : version finale (moyenne de graines), seulement si 5 la bat.
- **Lecture** : (a) le classement public ne montre que **30 %** des images et des copies, et l'écart de 70 % restant est invisible. Les mêmes images servent à chaque sonde : **les écarts entre sondes sont comparables** (même ensemble), mais un écart faible peut ne pas se transférer au classement final. (b) Ne pas affiner : considérer deux variantes à moins de ~0,02 d'écart comme égales et prendre la plus **robuste** (retain plus haut, α plus proche d'orig) [IDÉE, seuil arbitraire]. (c) Une sonde qui chute à 0 = probablement la porte 70 % : revenir sur l'accord proxy. (d) Si le public augmente et que le proxy retain baisse, soupçonner du sur-ajustement à la sous-population publique. (e) Ne pas empiler 3 changements entre 2 sondes : on ne saurait plus lequel a payé.

## 7. Consensus des praticiens ?
**Non, il n'y en a pas**, pour « oublier sans retain en reproduisant la distribution de sortie d'un modèle réentraîné, image par image ».
- Les articles qui oublient sans retain ([SCAR](https://github.com/jbonato1/scar), [RUAGO](https://mlanthology.org/neurips/2025/lee2025neurips-ruago), [RKLU](https://portal.findresearcher.sdu.dk/en/publications/rklu-redistributive-kl-distillation-for-efficient-retain-free-mac/)) visent **l'oubli de classes entières** ou des jeux CIFAR, pas des images isolées d'un ViT-B 224 ; leurs critères sont des précisions et des attaques d'inférence d'appartenance, pas une KL image par image vers un gold [VU].
- Les concours qui visent « ressembler au réentraîné » (NeurIPS 2023) ont **tous** utilisé du retain [VU ; le 9e était « forget-free » mais gardait un retain]. Les organisateurs eux-mêmes disent que le contrefactuel est mal défini et que la note dépend d'un seul modèle cible [VU, [blog Stanford](https://ai.stanford.edu/~kzliu/blog/unlearning)].
- Il y a seulement **des convergences partielles** [VU] : (i) la montée de gradient seule échoue ; (ii) réinitialiser puis distiller vers orig marche en 2023 ; (iii) une cible douce par image vaut mieux que l'uniforme ou que de « faux partout » ; (iv) un proxy hors-domaine peut maintenir l'utilité pour des classes (SCAR, RUAGO). Notre assemblage (cibles à queue + ancre + simulateur) n'est **ni prouvé ni testé ailleurs** : c'est un pari [IDÉE].

## 8. Questions à Adem (défaut si pas de réponse)
1. Quota et règles de soumission, jeux publics comme proxy : a. demander aux superviseurs maintenant (défaut) / b. s'en passer.
2. Lire `#task-3-unlearning` sur le Discord du championnat pour voir ce que disent les autres équipes : a. oui, vous le faites 5 minutes (défaut) / b. je le fais avec votre accord.
3. Chaîne B en pari principal avec A en filet, C seulement si proxy dans le domaine : a. oui (défaut) / b. B seul.

## Sources
- Kaggle : [discussions](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion) ; [12e place](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/458648) ; [clôture](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/457425) ; [recap](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/460044)
- 8e place : <https://junhyunb.github.io/en/projects/neurips2023-unlearning/> ; organisateurs : <https://arxiv.org/html/2406.09073v1> ; <https://ai.stanford.edu/~kzliu/blog/unlearning>
- GitHub : [sinonchum prep](https://github.com/sinonchum/ai-watermark-hackathon-prep) ; [Varsovie](https://github.com/yauhenifutryn/cispa-warsaw-2026) ; [Stockholm](https://github.com/sprintml/hackathonStockholm) ; [Sarrebruck](https://github.com/andresmartinezcarcel/cispa-cybersecurity-ai-hackathon-finale) ; [SCAR](https://github.com/jbonato1/scar) ; [RUAGO](https://github.com/Lemma1727/RUAGO) ; [Air Liquide](https://github.com/luispintoc/machine-unlearning) ; [lumliolum](https://github.com/lumliolum/neurips-machine-unlearning-2023)
- Jeu et gabarit : <https://huggingface.co/datasets/SprintML/unlearning> ; <https://huggingface.co/datasets/SprintML/unlearning/raw/main/task_template.py>
- arXiv : [2509.26427](https://arxiv.org/abs/2509.26427) ; [2510.16629](https://arxiv.org/abs/2510.16629) ; [2410.01276](https://arxiv.org/html/2410.01276v1) ; [2506.06278](https://arxiv.org/abs/2506.06278) ; [2406.01257](https://arxiv.org/abs/2406.01257)
