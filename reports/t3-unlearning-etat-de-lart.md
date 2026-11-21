# T3 Unlearning : état de l'art et choix de méthode

Date : 21 nov. 2026. Lu à la source : `Unlearning.txt`, `task_template.py`, `submission.py`, résumés arXiv de SCAR, Zero-Shot, ReGUn, rapport du concours NeurIPS 2023 (descriptions des méthodes gagnantes). SCRUB, SalUn, NPO : seul le résumé arXiv a été lu, le reste vient de ma mémoire (marqué « mémoire »). Aucune soumission, aucun appel payant.

## 0. Verdict en 4 lignes
- Méthode recommandée : **distillation douce depuis le modèle original (orig) sur les images à oublier, avec des cibles « non-membre » (tempérées, par image), plus un ancrage KL vers orig sur des images proxy**, petit pas d'apprentissage, derniers blocs seulement. C'est le plan A (section 3).
- Plan B qui améliore A : fabriquer un **pseudo-gold** (petits modèles qui n'ont jamais vu les images à oublier) et copier leur moyenne. Plus long, plus fidèle.
- Difficulté : technique 6/10, temps 5/10 (6 à 10 h pour un bon plan A), calcul 2/10 (1 A100 suffit, 2 à 15 min par essai).
- Risque n°1 : **ni labels, ni scoreur local**. Il faut un mini-scoreur maison et le leaderboard public (30 %) comme seul juge.

## 1. Ce qu'on nous demande, en 5 lignes
1. On reçoit un ViT-B/16 (timm, 100 classes) entraîné sur tout, et 3500 images « à oublier » sans labels. Aucune autre image.
2. On rend un `state_dict` float32 de même architecture (strict=True, moins de 450 Mo), LoRA fusionné si utilisé.
3. Forget = (score de loss) x (score de prédiction), sur les images exactes et sur des copies modifiées cachées. Gold = moyenne de plusieurs modèles réentraînés sans ces images.
4. Retain = (accuracy / accuracy orig) x 1/(1 + KL(orig||sub)) sur un jeu de contrôle caché, KL à température 2.
5. Porte : moins de 70 % sur un jeu de validation caché = score 0. Score final = forget x retain.

## 2. L'idée clé : copier le gold, pas casser le modèle
Image simple : orig a appris par coeur les 3500 images, comme un élève qui a vu le corrigé. Le gold est un élève qui ne l'a jamais vu : il se trompe parfois, hésite souvent, et quand il se trompe il choisit une classe **voisine et plausible**, pas une classe au hasard. On doit fabriquer cet élève-là, pas un élève qui rate tout.

Conséquences (déductions, à vérifier) :
- Sur les images vues, orig a une loss quasi nulle et très concentrée (moyenne basse, **écart-type bas**). Le gold a sur ces images une loss de type « image de test » : moyenne plus haute, **écart-type bien plus grand** (la plupart faciles, quelques-unes très ratées).
- Piège : une température unique pour toutes les images corrige la moyenne mais pas l'écart-type. Le score de loss compare moyenne ET écart-type. Il faut de l'hétérogénéité **par image** : faciles = restent confiantes, difficiles = incertaines ou fausses.
- Le score de prédiction est relatif à orig : `1 - KL(gold||sub)/KL(gold||orig)`. Rapprocher la sortie d'une version plus douce du gold fait déjà monter ce score, même imparfaite.
- Le gold est une **moyenne de plusieurs modèles** : sa distribution est plus lisse qu'un seul modèle. Notre cible doit aussi être une moyenne (plusieurs vues, plusieurs pseudo-modèles).
- Labels vrais inconnus, mais orig a presque sûrement raison sur ses images d'entraînement : **la classe prédite par orig sert de pseudo-label** (à mesurer : environ 35 images par classe attendues).

### Estimer ce que prédirait un modèle qui n'a jamais vu l'image, sans données retain
1. **Cible tempérée** : `q = softmax(logits_orig / tau_f)`, `tau_f` entre 2 et 5, réglable par image. Base la plus simple.
2. **Vues augmentées/dégradées** : la mémorisation baisse quand on recadre fort, floute, bruite, masque des patchs. Moyenne des sorties d'orig sur 8 à 16 vues = cible plus proche d'un non-membre. L'accord entre vues mesure aussi la difficulté de l'image.
3. **Tête validée croisée (leave-one-out)** : features d'orig sur les 3500 images (pseudo-labels), tête linéaire / LDA / kNN entraînée sur 90 %, prédiction des 10 % restants, on tourne. Donne des prédictions « jamais vues » qui reflètent la proximité entre classes (donc *quelle* mauvaise classe). Limite : les features d'orig portent encore la mémorisation, donc à mélanger avec 1 et 2.
4. **Pseudo-gold par modèles qui ne voient jamais ces images** (plan B). La mesure la plus honnête, au prix de plus de travail.
5. **Calibrage sur du vrai non-membre, si on en trouve** : si les 100 classes viennent d'un jeu public, la loss d'orig sur des images de test publiques donne la moyenne et l'écart-type attendus du gold. Meilleur levier du sujet.
6. **Le leaderboard comme capteur** : 1 ou 2 réglages globaux (`tau_f`, poids du mélange), jamais 50. Limite de soumissions : inconnue, à demander.

### Les 100 classes correspondent-elles à un jeu public ?
- Fiche HF `SprintML/unlearning` : « No dataset card yet ». Contient `model.pt` (343 Mo), `forget_images.npz` (526 Mo = 3500 x 150 528 octets, non compressé) et les 3 scripts. Le doc dit « custom ». Rien ne prouve un jeu public. Je n'ai pas téléchargé les 870 Mo.
- Test de 10 min : afficher 40 images. Pixélisées (32x32 agrandi) : CIFAR-100 probable. Photos nettes variées : sous-ensemble 100 classes d'ImageNet. Comparer la liste des classes à l'histogramme des classes prédites par orig.
- Si ça colle : on a de **vraies images non-membres labellisées** (test CIFAR-100 : 10 000 images). Orig mesuré dessus donne l'accuracy gold attendue, moyenne/écart-type de loss gold, et un vrai proxy retain. Question aux superviseurs : jeux publics autorisés ? (le texte dit seulement qu'aucune donnée d'entraînement supplémentaire n'est fournie).
- Déduction : le gold est entraîné de zéro avec la même recette ; une porte à 70 % suggère une accuracy orig de 75 à 85 %. Pas une donnée.

## 3. Approches classées

### Plan A (rang 1) : distillation douce « non-membre » + ancrage, derniers blocs
- Idée : init = orig ; sur les images à oublier (et leurs augmentations) la sortie doit coller à la cible `q` (cibles 1+2+3) ; sur des images proxy elle doit rester égale à celle d'orig.
- Proxy retain, du plus au moins fiable :
  1. Vrai jeu public non-membre si les classes correspondent.
  2. 10 000 images ImageNet/COCO. SCAR : sous-ensemble ImageNet de 10 000 images, COCO équivalent ; **bruit gaussien pur : mauvais**, il faut des motifs spatiaux. https://arxiv.org/html/2404.12922v1
  3. Images fabriquées à partir d'orig (DeepInversion / zero-shot : https://arxiv.org/abs/2201.05629). Plus long, qualité moyenne sur ViT.
  4. Mélanges forts des images à oublier avec cible = orig. Risque de réapprendre la mémorisation : petite dose ou rien.
- Pertes : `L = KL(q || p_sub)` sur l'oubli (T=2 comme le scoreur, plus un terme à T=1) `+ lambda * KL(p_orig || p_sub)` sur le proxy (T=2). Option : terme NPO doux (mémoire : `-(2/beta) log sigmoid(-beta log(p_sub/p_orig))`, https://arxiv.org/abs/2404.05868), beta 0,1 à 1.
- À balayer : `tau_f` {1,5 ; 2 ; 3 ; 5}, mélange tempérée/vues/tête CV {0 ; 0,5 ; 1}, `lambda` {1 ; 5 ; 10}, lr {1e-6, 3e-6, 1e-5} AdamW, 3 à 15 époques, blocs entraînables {2, 4, 6 derniers, tous}.
- Effort : 5 à 8 h (chargement, boucle, mini-scoreur, balayage). GPU : une époque (3500 + 10 000 proxy) = 20 s à 1 min ; essai complet 2 à 10 min ; 50 essais en moins de 4 h GPU sur 1 A100, ou répartis sur 4.

### Plan B (rang 2) : pseudo-gold par modèles « blancs » (perturber puis réparer)
- Idée du concours NeurIPS 2023 (https://arxiv.org/abs/2406.09073) : les meilleures méthodes **réinitialisent ou bruitent une partie des poids puis réparent**. Amnesiacs : réinitialise la tête et la première couche, puis distille orig sur des données autres que l'oubli. Sun : bruit sur des couches tirées au hasard. Forget : réinitialise des blocs au hasard et redistille.
- Adapté : K = 4 élèves (un par GPU) partent d'orig bruité, avec tête et 2 à 4 derniers blocs réinitialisés, puis **distillent orig sur le proxy uniquement**. Ils se comportent donc comme des modèles qui n'ont pas vu ces images. Leur **moyenne de softmax** sur les images à oublier (et augmentations) = pseudo p_gold, avec de vraies différences par image.
- Puis modèle final = plan A avec `q` = pseudo p_gold (mélangé à la cible tempérée). On peut aussi soumettre la moyenne des poids des élèves si son accuracy est proche d'orig.
- Limites : qualité = qualité du proxy (hors distribution : élève moins précis qu'orig, donc biais) et part de mémorisation qui survit au bruit. À mesurer.
- Effort : +6 à 8 h après A. GPU : un élève = 50 000 images x 10 époques = 10 à 15 min sur 1 A100 ; K=4 en parallèle sur 1 noeud.

### Rang 3 : SCRUB / SCAR avec pseudo-labels
- SCRUB (https://arxiv.org/abs/2302.09880 ; mémoire pour les détails) : alterne une étape qui **éloigne** l'élève d'orig sur l'oubli (KL maximisé) et une étape qui le rapproche sur le retain, plus CE. SCAR (https://arxiv.org/abs/2404.12922) remplace le retain par un proxy hors distribution, distille avec une divergence JS ; lr 5e-4 à 1e-4, batch 1024, T 1 à 5, max 30 époques. Son terme d'oubli (classe voisine en Mahalanobis) est conçu pour supprimer une classe, pas des images isolées.
- Pourquoi rang 3 : « s'éloigner d'orig » ne dit pas où aller. Le but est la position du gold. Utile en terme bonus ou baseline rapide.
- Effort : 3 à 4 h pour un premier essai. GPU faible.

### Rang 4 : recalibrage seul (filet de sécurité, très rapide)
- Ne toucher que la tête / un facteur de température appris + biais par classe, avec les cibles de la section 2. Les features ne changent pas : retain presque intact. Forget probablement modeste (écart-type mal reproduit). Effort 1 à 2 h. Bonne première soumission.

### À écarter
- Montée de gradient naïve, étiquettes aléatoires seules (SalUn random labeling, https://arxiv.org/abs/2310.12508 ; mémoire : masque de saillance + étiquettes aléatoires, conçu avec retain). Le gold ne choisit pas au hasard et ne s'effondre pas ; le texte du défi le dit.
- Méthodes dont l'importance des paramètres se calcule sur tout le train (SSD ; LFSSD https://arxiv.org/pdf/2402.19308 n'exige que le forget non étiqueté mais a encore besoin de l'importance sur le train). Zero-shot « gated knowledge transfer » : jugé instable (résumé web, à vérifier).
- ReGUn (https://arxiv.org/html/2603.11210v1) : même intuition que nous (distiller les sorties sur l'oubli vers une distribution de référence non-membre, histogramme de classes apparié) mais exige un jeu tenu de côté labellisé et du retain, et n'est pas testé sur ViT. À reprendre **si** on trouve un proxy labellisé : sa cible par lot (moyenne d'orig sur des non-membres de mêmes classes) est un bon `q`.

## 4. Protéger la porte des 70 % et le KL retain
- Mesurer en continu sur un proxy : accord argmax et KL(orig||sub) T=2. Arrêt précoce si KL dépasse 0,05 à 0,1 (KL 0,1 coûte environ 9 % de retain).
- Petit lr (1e-6 à 1e-5), peu d'époques, warmup court, pas de weight decay fort. Garder orig figé pour l'ancrage.
- Limiter les poids touchés : derniers blocs + norme finale + tête ; geler patch_embed, pos_embed, cls_token, premiers blocs.
- LoRA possible (rang 8 à 16 sur qkv, proj, fc1, fc2 des derniers blocs) avec fusion obligatoire `W += B @ A * alpha/r` dans les `nn.Linear` de timm (qkv = un seul Linear 768 -> 2304). Vérifier `strict=True` et sorties identiques avant/après fusion (1e-5).
- Interpolation de poids : `theta = orig + alpha * (theta_unlearn - orig)`, alpha {0,3 ; 0,5 ; 0,7 ; 1}. Coût nul, bon réglage oubli/retain après coup.
- Moyenne de poids de plusieurs graines : réduit la variance, rappelle la moyenne de plusieurs modèles du gold.
- Porte : ne pas soumettre si l'accuracy proxy chute de plus de 3 à 5 points vs orig ; sans labels, garder un accord argmax avec orig supérieur à 95 % sur proxy.

## 5. Pièges
- **Pas de labels** : l'entraînement utilise des pseudo-labels d'orig ; le serveur utilise les vrais.
- **fp16 autocast côté serveur** : évaluer localement exactement comme `predict()` du template. Logits trop grands = saturation ; garder la norme des logits proche d'orig. Entraîner en bf16 ou fp32 (fp16 sans scaler : non).
- **strict=True** : ne rien ajouter ni renommer (pas de préfixe `module.` de DDP). `torch.save({k: v.cpu().float()})`. Pas de NaN. Taille environ 345 Mo (limite 450).
- **LayerScale (init 1e-4)** : les `ls1.gamma` / `ls2.gamma` sont minuscules ; lr trop haut ou weight decay peut les dériver et casser le modèle. Les geler ou lr 10x plus petit. Si on réinitialise un bloc, remettre les gamma à 1e-4.
- **Copies modifiées cachées** : transformations inconnues (flou, recadrage, bruit, couleur : hypothèse). Augmentations larges à l'entraînement, ne pas sur-ajuster aux pixels exacts, rester dans la normalisation [-1, 1] du serveur.
- **Public 30 % / final 70 %** : peu d'images publiques, le balayage fin sur le public est du bruit. Soumettre le réglage le plus robuste, pas le plus haut.
- **Multiplicatif** : retain 0,9 x forget 0,3 = 0,27 ; retain 0,6 x forget 0,6 = 0,36. Viser d'abord KL(orig||sub) < 0,05, puis monter le forget.
- **Proxy hors distribution** : ImageNet sans les 100 classes peut baisser l'accuracy sur les vraies classes. Ajouter les augmentations des images à oublier avec cible **tempérée** (pas orig brut) comme retain faible.
- Ne pas committer `model.pt` ni `forget_images.npz`.

## 6. Mini-scoreur local (à faire en premier, 1 à 2 h)
- Pseudo-labels = argmax d'orig. Sur l'oubli et ses augmentations : moyenne/écart-type de loss, KL(cible||sub), accord avec orig. Sur proxy : accord argmax (ou accuracy si labels) et KL(orig||sub) T=2, en fp16 autocast.
- Contrôles : sub = orig doit donner forget 0 ; sub = bruit fort sur orig doit effondrer le retain.
- Ce n'est pas le vrai scoreur : un bon score local n'est pas une validation (les cibles sont inventées par nous).

## 7. Difficulté (sur 10)
| Axe | Note | Pourquoi |
|---|---|---|
| Technique | 6 | Idée simple (distiller vers une cible douce), mais pas de labels, pas de scoreur, cible gold à deviner, LayerScale, fusion LoRA. |
| Temps | 5 | Plan A utile en 6 à 8 h, plan B 14 à 16 h au total. Tenable en 20 h avec 1 à 2 personnes. |
| Calcul | 2 | 1 A100 suffit, 2 à 15 min par essai ; les 4 GPU servent au balayage et aux K élèves du plan B. |
Tâches 1 et 2 non lues : pas de comparaison chiffrée. T3 est parmi les moins coûteuses en GPU ; son vrai coût est l'incertitude (pas de vérité terrain, leaderboard bruité).

## 8. Plan de 20 h proposé
1. 0 à 1 h : 40 images, classes prédites d'orig, test jeu public, questions aux superviseurs (jeux publics permis ? limite de soumissions ?), dépôt `SprintML/hackathon` pour l'environnement Jülich.
2. 1 à 3 h : mini-scoreur + soumission de sécurité (rang 4).
3. 3 à 9 h : plan A, balayage, 2 à 3 soumissions au maximum.
4. 9 à 16 h : plan B si A plafonne ; moyenne de graines, interpolation alpha.
5. 3 dernières heures : choix robuste, vérification strict=True / float32 / taille / fp16, soumission finale.

## 9. Questions (défaut si pas de réponse entre parenthèses)
1. Jeux publics comme proxy retain ? a. demander aux superviseurs maintenant, oui si accord (défaut) / b. non, uniquement images à oublier + synthétiques.
2. Qui porte T3 ? a. une personne, renfort sur le plan B (défaut) / b. équipe entière 3 h pour le mini-scoreur.
3. Premier jet : a. rang 4 en 2 h pour sécuriser un score non nul (défaut) / b. directement plan A.

## Sources
- SCAR : https://arxiv.org/abs/2404.12922 (HTML : https://arxiv.org/html/2404.12922v1)
- SCRUB : https://arxiv.org/abs/2302.09880
- Zero-shot : https://arxiv.org/abs/2201.05629
- SalUn : https://arxiv.org/abs/2310.12508
- NPO : https://arxiv.org/abs/2404.05868
- Concours NeurIPS 2023 (méthodes en annexe A.4) : https://arxiv.org/abs/2406.09073
- ReGUn : https://arxiv.org/html/2603.11210v1
- LFSSD : https://arxiv.org/pdf/2402.19308
- Jeu de données HF : https://huggingface.co/datasets/SprintML/unlearning
