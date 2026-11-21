# T2 : détecteur appris « même modèle ? » (sous-session, 21-22/11)

Décision débloquée : quelle règle appliquer aux 33 ep pour la partie modèle (0,9 du score), et avec combien de réponses par guichet.
Branche `t2/detecteur-appris`, code dans `tracks/t2-model-equality/detecteur/`. **0 requête vers l'API du hackathon.**

## Méthode
1. **Jumeaux** (`gen.py`, h200c) : pour une base locale (SmolLM2-360M, Llama-3.2-1B unsloth, Qwen2.5-1.5B, Qwen2.5-3B), des « mondes » de 60 à 100 guichets.
   - Variantes de poids : base, fp8 par tenseur (plus 4 autres implémentations fp8 : activations, par canal, e5m2, toutes matrices), int8 par canal, nf4, et LoRA × {0,3 ; 0,7 ; 1,5}.
   - LoRA : entraînées sur des corpus variés (`lora_train.py`).
   - Déguisements seuls ou combinés : system prompt, persona, filigrane KGW (liste verte hachée sur le token précédent), température. La température par défaut varie selon le monde (0,7 / 0,75 / 0,8), car les vrais ref sont plus proches de 0,75 que de 0,8.
   - Mêmes 100 prompts que `probes_v4.py`, 256 tokens max.
2. **Lecteur** (`feats.py`) : chaque réponse est relue sous la base et sous des lecteurs modifiés (fp8, int8, nf4, 3 LoRA d'entraînement). On en tire :
   - la log-vraisemblance, l'entropie et la varentropie de la base à 6 températures (calibration par tranche d'entropie) ;
   - pour chaque lecteur q, un λ = (Σd − E_p[d]) / (E_q[d] − E_p[d]) : 0 si le texte vient de la base, 1 s'il vient de q.
   Tout est stocké en sommes par réponse, ce qui permet de recalculer pour 25, 50 ou 100 réponses.
3. **Classifieur de paires** (`pairs.py`, `train_eval.py`) : environ 80 caractéristiques par guichet (λ, z de calibration, T effective, rangs, style de texte, longueur), symétrisées pour la paire (|écart|, moyenne, min, max). S'y ajoutent des caractéristiques croisées (divergence de vocabulaire, corrélation des longueurs et des z par prompt, Jaccard). Modèle LightGBM. **Partition par lien complet**, avec un seuil choisi uniquement sur les jumeaux de validation.
4. **Validation stricte** : mondes à graines neuves, LoRA jamais vues (platypus, no_robots, medical, alpaca+dolly), system prompts, personas et paramètres de filigrane jamais vus. Score = MCC modèle moyen sur 200 lots tirés avec la structure du corrigé (5 copies + fp8 + une LoRA aux 3 forces), et sur des lots de 11 type ep.

## Résultats par tour (MCC modèle)
| tour | ce qui change | jumeaux neufs (25 rép.) | vrais ref (25 rép.) |
|---|---|---|---|
| 1 | LightGBM, seuil 0,5, monde SmolLM2 de 96 guichets | 0,601 | **0,721** (seule erreur : ref-03 séparé) |
| 2 | seuil choisi sur les jumeaux (0,1) | 0,668 | 1,000 |
| 3 | + monde de 60 copies déguisées | 0,740 | 1,000 |
| 4 | + Llama-3.2-1B (2 mondes) | 0,641 (Smol) / 0,583 (Llama) | 1,000 |

Réserves sur le 1,000 des ref :
- les marges sont minces : ref-09 (LoRA 0,3) et ref-02 (fp8) ne restent à part que grâce au lien complet (P(ref-02, ref-05) = 0,63 au tour 3) ;
- la matrice des ref a été vue dès le tour 1 ;
- seul le tour 1 (0,721) est un premier regard réellement aveugle.

## Budget → score (jumeaux SmolLM2 à graines neuves, tour 3)
| réponses par guichet | 25 | 50 | 100 |
|---|---|---|---|
| MCC modèle (tirages type corrigé) | 0,740 | 0,744 | 0,856 |
| vrais ref | 1,000 | à mesurer (blocs 1-3 pas encore collectés) | à mesurer |

## Essais négatifs
- Retirer les caractéristiques de style de texte : aucun effet (0,601 / 0,721).
- Caractéristiques relatives à la médiane du lot, entraînées sur des lots tirés : 0,477 (contre 0,723), abandonné.

## Limites connues
- Erreurs dominantes sur les jumeaux : personas et system prompts jamais vus séparés de la base ; LoRA 0,3 avec filigrane confondue avec la base.
- Le vrai fp8 des ref ne donne que λ_fp8 ≈ 0,6 sous notre lecteur (1,0 pour nos jumeaux) : l'implémentation des organisateurs diffère.
- GPU h200c partagée et saturée : nombreux manques de mémoire, LoRA Qwen-3B retardées.
