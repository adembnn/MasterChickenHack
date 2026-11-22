# T3 · réparer la rétention d'un bon PG par la tête seule (hypothèse expérimentale)

**Statut : CODE CPU VÉRIFIÉ, aucune exécution sur les vraies images, aucun score leaderboard.**
À évaluer **après** les essais d'Adem, seulement s'il reste une H200 libre et un cache proxy indépendant.
[Code](../tracks/t3-unlearning/experiments/t3_head_repair.py) sur branche `research/top1-t1-t3-nov21`.

## Idée et pourquoi elle diffère des essais saturés
Le modèle PG fait monter la loss des exemples à oublier mais peut perdre des points de **rétention KL** sur un contrôle caché. Sa tête `head.weight, head.bias` est une application linéaire de features pré-logits. Sans labels, on peut ajuster **uniquement cette tête** sur un jeu de *proxy-retain indépendant* pour rapprocher les logits à T=2 des sorties du modèle original.

Pour une matrice de features du candidat `X` sur les proxies, un modèle original `Zorig`, et la tête actuelle `W,b` : `R=Zorig-(XWᵀ+b)`. On centre R par image, car le softmax et la KL sont invariants au décalage constant. On résout par ridge `Δ = (X̃ᵀX̃/N + λI)^(-1) X̃ᵀR/N`, puis `(W,b) += α Δᵀ`, avec un biais moins pénalisé. La correction ne touche ni au tronc, ni aux entrées, ni aux données à oublier.

**Ce n'est PAS la réparation du modèle par distillation** qu'a déjà testée `t3-poids` avec échec (500 pas sur vues des forget). C'est une régression fermée sur des images naturelles *indépendantes*, sans mettre à jour les blocs. Risque important : l'effet sur proxies hors distribution peut ne pas se transférer au retain caché et peut restaurer la mémorisation du forget. **Ne pas envoyer sur seule baisse de MSE proxy.**

## Protocole minimal H200 (viser 30-60 minutes)
1. Partir du meilleur **PG_T85_f70_l814c** public `0.3159`, pas du `gold_blend55` public `0.0804`.
2. Cacher sur **500-1000 images naturelles inédites** (COCO non utilisées dans les ancres de PG) : `candidate.forward_head(candidate.forward_features(images), pre_logits=True)` shape [N,768], et `orig(images)` logits [N,100], mêmes prétraitements officiels (fp16 autocast) et même ordre.
3. `python t3_head_repair.py --candidate PG_f70.pt --features candidate_prelogits.npy --original-logits orig_logits.npy --alpha 0.25 --ridge 0.05 --out PG_f70_repair_a25.pt`. Ablations `alpha={0,0.25,0.5}` ; `ridge={0.05,0.2}`. Garder une deuxième partie de proxy pour validation *hors ajustement*.
4. Mesurer sur les 3500 exactes **et vues légères/fortes** : CE m/s, changement d'argmax, KL proxy vs original T=2. Sur proxies indépendants : top1/orig, KL(orig||sub) à T=2. Comparer au PG-f70 identique, **retenue avant/après**, et marge estimée au gate 70 %.
5. Ne considérer un envoi que si le simulateur de score complet prédit une hausse robuste avec réserve différente, si le score de loss oubli ne baisse pas nettement, si le niveau de confiance de rétention augmente, et après maker-checker. Ne pas consommer un créneau public sur cet objectif sans validation.

## Test déjà exécuté
Sur 200 échantillons synthétiques, 20 features et 5 classes : la MSE des logits proxy passe de **2.1682 à 0.8671** avec α=0.5, λ=0.01, en conservant des poids finis. Ce n'est **pas** un résultat sur EAC. Le contrôle CPU ne prédit pas l'effet sur oublis, copies et gate.

## Références
- [ReGUn arXiv 2603.11210](https://arxiv.org/abs/2603.11210) : calibrer sur données disjointes plutôt que simplement abîmer les images forget ; utilise des références class-conditionnées, ici le proxy non labellisé est moins fort.
- [DISCERN arXiv 2611.07892](https://arxiv.org/abs/2611.07892) : replay sans retain sur géométrie ViT, mais pour *class unlearning*, différent de nos exemples individuels.
- [Critique et expériences T3](https://github.com/adembnn/EAC/issues/5).
