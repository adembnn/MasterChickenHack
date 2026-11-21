# T2 · tour 2 : repérer une LoRA faible sans connaître l'adaptateur (agent recherche)

**Décision débloquée** : quelle statistique, hors des chantiers en cours, peut isoler une LoRA faible, et peut-on isoler ep-02, ep-16 et ep-23 ?
Verdict en deux lignes. Une statistique marche très bien **sur les jumeaux** (précision 0,89, rappel 0,80 sur val), mais **ne détecte aucune des 3 vraies LoRA des ref** (0/3), et sur les ep réels elle ne signale que des **personas / prompts système variables**. Elle ne justifie donc pas d'isoler un guichet de plus. 0 requête ep. Code : `tracks/t2-model-equality/nouvelles-pistes/np2_*.py`.

## Méthode commune
Lecteur unique = la base locale (jamais les poids d'un guichet). Pour chaque guichet : décodeur ajusté (T au pas de 0,05, top-p 1 / 0,95 / 0,9), puis par token, au décodeur ajusté, le résidu de score `m = log p(y) − E[log p]` (z ~ N(0,1) sous « copie de la base »), le PIT `u`, et le rang observé contre ses probabilités attendues. 25 réponses par guichet. Jumeaux : 3 familles × (val 36 / train 56 guichets), vérité connue, copies déguisées (système, persona, filigrane KGW, T de 0,7 à 1,0 et au-delà) et LoRA 0,3 / 0,7 / 1,5 (`~/eac-t2/detecteur/worlds`, lus sans modification).

## Les trois idées testées
| Idée | Résultat sur jumeaux (AUC LoRA contre copies) | Verdict |
|---|---|---|
| **A. Corrélation entre guichets** des résidus par prompt (même LoRA ⇒ même direction), recentrée sur la médiane des pairs | val : 0,67 (30 paires positives), meilleur trait z 0,67 | Trop bruité à 25 prompts. Abandonné. |
| **B. Signature de rang** (le rang 0 ne dépend pas de T) : khi² sur toute la réponse | 0,56 (val) | Inutile seul. En fenêtre de position : voir C. |
| **C. Résidu aux premiers tokens** : \|z de m\| sur les 5 premiers tokens, seuil 4 | **0,93 train / 0,95 val** (guichets sans prompt système) | Excellent sur jumeaux. Voir les réserves. |

Pourquoi C et pas le « max sur une banque de z » : ce max est noyé par les filigranes et les personas (max \|z\| des copies persona 24-40, filigrane 10-18, contre 7 / 26 / 13 pour les LoRA 0,3 / 0,7 / 1,5, AUC 0,48-0,72). Les LoRA des jumeaux changent surtout les **premiers tokens** (cohérent avec `first_div`, médiane 0 dès s = 0,3). Fenêtres essayées : 1, 2, 3, 5 tokens (AUC train 0,70 / 0,81 / 0,81 / 0,95 avec rms rang+score ; |zm| seul à 5 : 0,93). Le choix de 5 est fait après lecture : à refaire sur d'autres jumeaux.

**Chiffres de C (guichets sans prompt système, copies chaudes, filigranées et T = 1,0 comprises) :**
| Seuil 4 | précision | rappel | par force (0,3 / 0,7 / 1,5) |
|---|---|---|---|
| train (22 LoRA, 36 copies) | 1,00 | 0,59 | 2/7, 4/7, 7/8 |
| val (10 LoRA, 27 copies) | **0,89** | 0,80 | 2/3, 2/3, 4/4 |
- Avec un prompt système ou une persona **inconnus** du lecteur, l'AUC tombe à 0,60-0,83 : le prompt déplace les premiers tokens.
- **Prompt système connu** (oracle, 31 guichets val) : les fausses alertes passent de **11/21 à 0/21** (médiane de \|z\| 4,7 → 0,8) et la T ajustée retombe à la vraie (ex. persona T 0,85 → 0,75).

## Validation sur les VRAIES LoRA (ref, SmolLM2-360M, `ref_bloc0`, 25 réponses)
| ref | type | zm@1 / 2 / 3 / 5 / 8 tokens | T ajustée |
|---|---|---|---|
| 04 | LoRA 0,7 | −0,4 / −0,1 / +0,7 / +0,2 / +0,8 | 0,80 |
| 07 | LoRA 1,5 | −0,3 / −0,1 / +0,1 / +0,3 / +0,5 | 0,90 |
| 09 | LoRA 0,3 | +1,2 / +1,3 / +1,7 / +1,0 / +0,9 | 0,75 |
| 01 base, 03 système, 06 filigrane, 08 T 1,0, 02 fp8 | copies | tous \|z\| ≤ 2,7 | 0,70-1,00 |
**Aucune des 3 LoRA des organisateurs n'est vue (\|z\| ≤ 1,7 à toutes les fenêtres).** Leur signature n'est pas dans le préfixe. Les LoRA des jumeaux (entraînées sur alpaca / gsm / stories) sont donc plus « visibles » que celles des ref : les 0,89 de précision ne se transfèrent pas. Sur les ref, la LoRA se voit seulement comme une T ajustée plus haute (0,80 / 0,90 / 0,75 contre 0,70-0,75) et un PIT peu uniforme (AD 0,75 / 1,79 / 1,07, tour 1) : c'est exactement l'ambiguïté « température ou LoRA » déjà connue.

## Application aux 33 ep (score C, \|zm\| sur 5 tokens, seuil 4)
- **Dépassent 4 en lecture simple : ep-02 (−16,5), ep-16 (−26,7), ep-23 (−25,0), ep-29 (−6,9), ep-20 (−5,7).** Ce sont les 5 guichets à prompt système ou persona relevés par la sonde extract.
- **Lus avec leur prompt extrait** (en rôle système ou collé dans le tour utilisateur, même résultat) : **ep-20 (3,8 / 3,3) et ep-29 (2,8 / 3,8) retombent sous 4. ep-02 (−9,3 / −9,5) et ep-16 (−19,1 / −32,5) restent hors norme**, ep-23 n'a pas de prompt extractible.
- **Pourquoi ep-16 et ep-23 ne se réparent pas** : leurs réponses commencent par des voix différentes d'une requête à l'autre (ep-16 : « A most…fascinating task » en majordome, puis « Little sis… As your coach » ; ep-23 : « **Warning** »). Hypothèse la plus simple (non vérifiée par une requête) : le prompt ajouté par le fournisseur est **tiré au hasard à chaque requête** dans une liste ; le prompt unique de la sonde extract n'en décrit qu'un. ep-16, ep-23 et très probablement ep-02 sont des **copies déguisées**, pas des LoRA.
- **Aucun guichet sans prompt système ne dépasse 4** : maximum ep-01 −3,7 et ep-32 −3,6 (ep-32 lit « You are Qwen » : prompt par défaut). **ep-17 (−0,5) et ep-18 (+0,6) ne ressortent pas.** Avec les ref, ce résultat nul est cohérent : ce score ne voit pas ces LoRA.

## Ce qu'il faut en faire
1. **Ne pas isoler ep-02, ep-16, ep-23** sur la base de signaux de poids : leur anomalie est expliquée par un prompt fournisseur variable. À garder dans la base de leur famille (ep-02 : voir aussi son NF4 3,03 du tour 1, probablement de même origine).
2. Le score C est un **détecteur de persona / prompt système** fiable (jumeaux et ep concordent) : utile comme garde-fou contre les fausses LoRA, pas pour en trouver.
3. **Pas de précision ≥ 0,75 démontrée en conditions réelles** sur une LoRA faible sans adaptateur. La seule validation réelle (ref) est négative. Les pistes restantes, par ordre : lecture avec 100 réponses longues (plus de tokens, z × 2,7), distillation d'une LoRA sur un guichet **connu** (ref-07 pour SmolLM2) pour tester d'abord le détecteur appris sur les ref, et T ajustée fine comme second signal.
4. Limites : jumeaux sans top-p (les ep réels ont T 1,0 + top-p 0,95, ep-07/14/27) ; Qwen-3B sans LoRA dans les mondes extraits ; train_s2 non exploité (arrêté pour libérer le GPU) ; 25 réponses seulement par guichet.

Reproduire : `np2_extract.py` (extraction, `--oracle`, `--system_map`, `--sys_in_user`), `np2_feat.py` (banque de z), `np2_stats.py` / `np2_pairs.py` (statistiques globales et inter-guichets). Sorties : `h200c:~/nouvelles-pistes/o/*.pkl`.
