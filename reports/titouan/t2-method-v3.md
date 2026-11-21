# T2 Model Equality : méthode v3, unifiée et actionnable

Ce document remplace `t2-method-v1.md`/`t2-method-v2.md` comme référence d'exécution T2. Il ne refait pas les démonstrations : il **sélectionne et enchaîne** ce qui a été mesuré dans les 4 PR ouvertes à l'écriture de ce document (21/11, ~17h40) et dans les rapports déjà fusionnés dans `main` (`t2-statistics.md`, `t2-fingerprinting.md`, `submissions-ops.md`). Chaque chiffre cité est mesuré (jumeau OLMo-2-1B local ou simulation calibrée sur ces mesures), jamais une estimation théorique seule — les estimations théoriques des rapports v1/v2 (ex. tables de puissance MET extrapolées) sont citées seulement pour mémoire et marquées comme telles.

**Sources** : PR adembnn/EAC#10 (plan de déploiement), #11 (validateur/dry-run), #12 (simulation MCC), #13 (robustesse filigrane). Non encore prêt à cette heure : clustering optimal (algorithme recommandé au-delà du lien moyen déjà dans `t2-statistics.md` §6.3), mining de prompts, calibration fine (LoRA/int8), nuisances cumulées, famille — voir §3.

## 0. Résumé exécutif (les 5 décisions)

1. **Statistique principale : comparaison gloutonne exacte**, avec un seuil de décision **tau = 0,43** sur le taux de divergence à N=100 réponses/endpoint (#12). MCC_modèle attendu **0,93-0,97** si le bruit réel ressemble à celui du jumeau.
2. **Garde-fou watermark, avant toute décision** : si la première divergence entre deux endpoints de la même famille est quasi immédiate (médiane ≤ 4 tokens sur plusieurs prompts), **ne pas conclure "modèles différents"** — c'est la signature d'un filigrane, pas d'un écart de poids (#13). Bascule vers la NLL croisée si l'endpoint expose des logprobs ; sinon vers un score tolérant (position de première divergence, n-grammes) plutôt que l'égalité stricte.
3. **Décodeur** : clustering hiérarchique à lien moyen coupé à tau, famille d'abord puis modèle à l'intérieur de chaque famille (déjà dans `t2-statistics.md` §6.3 ; pas de nouvelle mesure en v3 — le clustering optimal mesuré est à venir, §3).
4. **Allocation du budget** : vagues F(16)+A(48) partout, réserve B(24)+résidu(12) ciblée sur les paires ambiguës, algorithme séquentiel détaillé (#10).
5. **Avant tout appel `ep-*` réel** : CSV validé par `validate_t2.py --strict`, dry-run `dryrun_submit_t2.py` passé, checklist de `README_T2.md` cochée (#11).

## 1. Pipeline, étape par étape

### Étape 0 — Pré-vol sur `ref-*` (gratuit, illimité, avant tout `ep-*`)
- Mesurer le taux de divergence gloutonne entre copies du même modèle ref (bruit de serving) : **chiffre de référence mesuré sur le jumeau, pas sur les vrais `ref-*`** = 0,30-0,34 (§2, source #12 §1). À reproduire sur `ref-*` réels dès que possible : c'est l'intrant qui recalibre tau si le bruit réel diffère.
- Mesurer, pour chaque paire `ref-*` de la même famille présumée, la position de première divergence (médiane sur ≥10 prompts courts). Si elle est ≤ 4 tokens pour une paire qu'on attend identique : suspecter un filigrane déjà actif sur `ref-*`, appliquer la règle §2 avant de continuer.
- Zoo de candidats locaux (OLMo-2 par taille), identification de la famille par vraisemblance (`t2-fingerprinting.md` §3.5), extraction du prompt système si possible.
- Valider la chaîne de soumission : `validate_t2.py` + `test_validate_t2.py` (22 tests, OK) + `dryrun_submit_t2.py` sur un CSV factice (#11). Geler ce code dans `main` avant le premier appel `ep-*`.

### Étape 1 — Choix de la statistique par paire, AVEC la règle de bascule watermark
Pour chaque paire d'endpoints (e,f) de la même famille présumée, sur les prompts de la Ronde F (voir §3) :

1. Calculer le **taux de divergence gloutonne** (comparaison exacte des textes, prompt par prompt) ET la **position de première divergence** (médiane sur tous les prompts testés).
2. **Règle de détection du filigrane (boîte noire, sans clé)** (#13 §1) :
   - Si la médiane de première divergence est **≤ 4 tokens** sur plusieurs prompts **et** que le taux d'identité stricte est quasi nul (< 5 %) : c'est la signature d'un biais de décodage (filigrane), **pas** une preuve de poids différents. Un écart de poids réel (int8/fp8/NF4) diverge plus tard : médiane mesurée **10-16 tokens** (#13 §1, tableau "poids vraiment différents").
   - Dans ce cas, **ne pas décider sur le taux de divergence seul.** Basculer :
     - **Si l'endpoint renvoie des logprobs** (ou c'est un jumeau local) → calculer la **NLL croisée** : `gap = NLL_f(texte_e) − NLL_e(texte_e)` dans les deux sens, moyenne + t de Student apparié sur les prompts disponibles. `gap` mesuré **exactement 0** pour des poids identiques quel que soit le filigrane (δ testé jusqu'à 8) ; `gap` croît avec l'écart de poids et résiste au filigrane (t = 2,6 à 10 avec δ=8 des deux côtés, contre t=0 pour le jumeau — #13 §3). C'est la statistique à utiliser dès qu'elle est disponible : elle est insensible au filigrane par construction.
     - **Si l'endpoint est en boîte noire pure** (texte seul, cas de l'API du hackathon à ce jour, cf. `README_T2.md` : pas de logprobs exposés) → ne pas utiliser l'égalité stricte. Utiliser un score tolérant au niveau du token (similarité de n-grammes, ou la position de première divergence elle-même comme feature continue plutôt que comme seuil binaire) sur un grand nombre de prompts courts et sûrs. **Non encore mesuré en boîte noire pure** à l'écriture de ce document (prochaine expérience notée par #13) : à vérifier sur `ref-*` avant de s'appuyer dessus pour une décision `ep-*`.
3. **Si pas de signature de filigrane** (première divergence tardive, médiane > 8-10 tokens, ou taux d'identité significatif) : utiliser directement le **taux de divergence gloutonne** avec le seuil tau (§2). C'est le cas nominal mesuré par #12, le plus informatif pour quantification/LoRA quand il n'y a pas de filigrane actif.

En résumé, la règle de bascule est : **détecter d'abord (indice de décodage immédiat = filigrane probable), puis choisir l'arme** — comparaison exacte + seuil tau si pas de filigrane détecté, NLL croisée si logprobs disponibles et filigrane suspecté, score tolérant en dernier recours boîte noire.

### Étape 2 — Seuil de décision (si comparaison gloutonne retenue)
- **tau = 0,43** : isoler un endpoint de sa base de famille présumée si > 43 % de ses réponses divergent (#12 §0, §9). Valeur = `max(bruit mesuré sur ref) + 2·sd(p̂, N=100)`, à **recalculer sur le bruit réel mesuré à l'étape 0**, pas à garder figée si le bruit `ref` réel diffère de 0,34 (jumeau). La courbe MCC(tau) est plate entre 0,40 et 0,45 (perte < 0,02) : pas besoin de viser l'optimum exact, juste rester dans cette fenêtre.
- Si le bruit réel mesuré sur `ref-*` dépasse 0,42 (jumeau +0,08) : int8 devient indiscernable, MCC plafonne à 0,64-0,85 selon la structure (#12 §7) — l'annoncer sur l'issue #4 dès détecté, ne pas attendre la fin.

### Étape 3 — Allocation du budget (100 réponses/endpoint, irréversible)
Reprise directe de #10 §1, §3 (aucune nouvelle mesure en v3) :

| Vague | Réponses/endpoint | Rôle | Déclenchée si |
|---|---|---|---|
| F (famille) | 16 | 16 prompts ouverts distincts. Sépare les familles, révèle le prompt système, premier signal de filigrane (position de 1ère divergence). | Toujours, en premier |
| A (cœur) | 48 | 16 prompts choisis hors ligne x3 répétitions. Matrice de distances complète, 1ère soumission valide. | Toujours, après F |
| B (adaptative) | jusqu'à 24 | 4 prompts les plus discriminants de A, rejoués contre le **prototype** du cluster suspecté (réduit la variance de ~40 %). | Seulement sur les paires avec P(même\|T) ∈ [0,2 ; 0,6] après F+A |
| Réserve | 12 | Vérification finale, priorisée par taille de cluster (coût d'erreur). | Seulement si B ne tranche pas, et jamais dépensée d'un coup sur toutes les paires restantes |

Algorithme séquentiel complet (verbatim #10 §3) : calculer P(même|T) par paire après chaque vague ; P≥0,6 → "même" gelé ; P≤0,2 → "différent" gelé ; sinon ambigu → vague suivante ; si l'ajout d'une vague change le MCC estimé de moins de 0,03, **arrêter toute dépense supplémentaire** ; par défaut final si toujours ambigu après 100/100 : **isoler** (un singleton ne coûte aucun FP ; une fusion fausse dans un gros cluster coûte cher).

### Étape 4 — Décodage en partition
- Clustering hiérarchique à **lien moyen** sur T (ou 1 − P(même)), coupé à tau. Famille d'abord (seuil franchement bimodal sur la vraisemblance de la Ronde F), modèle ensuite à l'intérieur de chaque famille (`t2-statistics.md` §6.3, point 1). **Algorithme recommandé "au-delà" du lien moyen (corrélation/MBR) : pas encore mesuré en v3** — voir §3, "clustering optimal".
- Règles de prudence inchangées : isoler en cas de doute réel ; vérifier que la distance intra-cluster < 0,5 × distance au cluster le plus proche ; ne jamais couper sur le score public (30 % des paires, contrôle de format seulement).

### Étape 5 — Validation avant chaque soumission
Checklist #11 (`README_T2.md`), à cocher dans l'ordre avant tout `submission.py` :
1. CSV produit par le code fusionné dans `main` (pas à la main).
2. `validate_t2.py submission.csv --strict` → `RESULTAT: VALIDE`.
3. `--previous <dernier CSV soumis>` : nombre de paires changées > 0 (sinon créneau gaspillé).
4. Score local `ref-01..ref-09` via `score_practice.py`, noté sur l'issue #4.
5. `dryrun_submit_t2.py submission.csv` → `DRY-RUN TERMINE`.
6. ≥ 30 min depuis la dernière soumission T2.
7. Annonce sur l'issue #4 : « soumission T2 à HH:MM, méthode X, score local Y ».
8. Titouan seul lance `submission.py` (clé dans `.env`).
9. Score public reporté dans `tracks/t2-model-equality/CONTEXTE.md` et l'issue #4.
10. Ne jamais sur-optimiser sur le public (30 % des paires).

Règles R1-R11 du format CSV (3 colonnes, pas de virgule même quotée, même `model` ⇒ même `family`, pas de `ref-*`, UTF-8 sans BOM, etc.) : détail complet dans `tools/validators/README_T2.md`.

## 2. Tableau récapitulatif des chiffres mesurés (pas d'estimation théorique)

| Mesure | Valeur | Condition | Source (PR / fichier) |
|---|---|---|---|
| Bruit de serving entre copies du même modèle (taux de divergence gloutonne) | 0,300-0,340 (moyenne 0,323, max 0,340) | jumeau OLMo-2-1B, N=300, variations lot/dtype | #12, `results_exp1_a100.json` |
| Taux de divergence : int8 | 0,497 (sd≈0,050 à N=100) | idem | #12 §1 |
| Taux de divergence : fp8_e4m3 / fp8_e5m2 | 0,777 / 0,933 | idem | #12 §1 |
| Taux de divergence : nf4 | 0,983 | idem | #12 §1 |
| Taux de divergence : LoRA 0,3 / 0,7 / 1,0 / 1,5 | 0,820 / 0,983 / 1,000 / 1,000 | idem | #12 §1 |
| Seuil optimal tau | 0,42-0,43 (identique sur 6 structures simulées) | simulation 2000 tirages, N=100 | #12 §0, §4 |
| MCC_modèle attendu (règle a priori tau=0,43) | 0,93-0,97 (p10 0,85-0,92) | idem | #12 §5 |
| MCC naïf A (seuil = moyenne du bruit 0,32) | 0,50-0,54 | idem | #12 §5 |
| MCC naïf B (seuil 0,50) | 0,70-0,92 | idem | #12 §5 |
| Puissance de détection int8 à tau=0,43 | 0,891 (N=100) | idem | #12 §3 |
| Coût du budget N=100 vs N=300 | −0,027 à −0,035 MCC (−0,064 si beaucoup d'int8) | idem | #12 §6 |
| Comparaison exacte, identité texte, filigrane même poids (δ=2, k1 vs k2) | 0 % identique, 1ère divergence médiane = 3 tokens | OLMo-2-0425-1B-Instruct, 60 prompts, 64 tokens, job JURECA t2wm #15901652 | #13 §1 |
| Comparaison exacte, poids différents sans filigrane (base vs NF4) | 1,7 % identique, 1ère divergence médiane = 10 tokens | idem | #13 §1 |
| NLL croisée, mêmes poids (quel que soit δ ≤ 8) | gap = 0,000, t = 0 | idem | #13 §3 |
| NLL croisée, base vs int8 (sans / avec filigrane δ=8) | gap 0,001 (t=1,4) / 0,003 (t=2,6) | idem | #13 §3 |
| NLL croisée, base vs fp8_e4m3 | gap 0,006 (t=4,7) | idem | #13 §3 |
| NLL croisée, base vs fp8_e5m2 | gap 0,024 (t=10,8) | idem | #13 §3 |
| NLL croisée, base vs NF4 (sans / avec filigrane δ=8) | gap 0,059 (t=11,0) / 0,114 (t=10,0) | idem | #13 §3 |
| Détecteur sans clé « hors-argmax » (glouton nu / δ=1 / δ=2 / δ=4 / δ=8) | 0,4 % / 6-8 % / 16-17 % / 35-37 % / 61 % | idem | #13 §2 |
| Détecteur sans clé sur poids différents (int8 / fp8_e4m3 / fp8_e5m2 / NF4), texte sans filigrane | 1,5 % / 3-4 % / 7 % / 10-11 % | idem | #13 §2 |
| z-score KGW, vraie clé (δ=4 / δ=8) | z moyen 7,0 (88 % > 4) / 11,2 (100 % > 4) | idem | #13 §2 |
| 22 tests unitaires validateur CSV | `Ran 22 tests ... OK` | `python3 -I test_validate_t2.py` | #11 |
| Dry-run requête de soumission | `DRY-RUN TERMINE`, réseau vérifié désactivé | `python3 -I dryrun_submit_t2.py valid.csv` | #11 |

## 3. Ce qui n'est pas encore prêt (à l'heure de rédaction, 21/11 ~17h40)

Sessions en cours, sans PR au moment de la rédaction — vérifié par `gh pr list -R adembnn/EAC --state open` (seules #10-#13 sont ouvertes) et par les fichiers non committés présents localement (`tracks/t2-model-equality/clustering.py`, `tracks/t2-model-equality/twin/` non trackés) :

1. **Clustering optimal** : la v3 utilise le lien moyen déjà documenté dans `t2-statistics.md` §6.3 (point 1), qui est une recommandation théorique, pas une mesure de cette v3. Un `clustering.py` existe déjà dans le répertoire de travail (non commité, pas de PR) : à relire et intégrer dans une v4 dès qu'il est fusionné, avec ses chiffres mesurés.
2. **Mining de prompts** : la Ronde A (§1, étape 3) suppose une liste de 16 prompts choisis hors ligne par critère de Fisher (méthode décrite dans `t2-statistics.md` §4, jamais exécutée à l'échelle réelle à ce jour). Pas de liste gelée issue d'un mining mesuré à intégrer en v3.
3. **Calibration fine LoRA/int8** : les chiffres de puissance int8 (#12) viennent du jumeau 1B ; aucune calibration spécifique par force LoRA ou par taille de modèle réel n'est encore mesurée au-delà de ce qui figure dans le tableau §2.
4. **Nuisances cumulées** (prompt système + température + filigrane simultanés) : #13 ne teste que le filigrane seul sur un jumeau sans autre déguisement. La prochaine expérience notée par #13 elle-même ("température > 0 avec graine" + filigrane à fenêtre plus large) n'a pas encore de résultat.
5. **Famille** : aucune PR dédiée ; la méthode de famille reste celle de `t2-fingerprinting.md` (vraisemblance locale, non re-mesurée en v3).

**Conséquence pratique** : la v3 est actionnable telle quelle pour démarrer l'étape 0 et la Ronde F/A (§1), mais la liste de prompts de la Ronde A et l'algorithme de clustering fin devront être complétés par une v4 dès que les PR correspondantes arrivent — sans bloquer le calendrier de #10 §5, qui reste valide tel quel.

## 4. Historique
- v1/v2 (fusionnés dans `main`, non branchés) : estimations théoriques et simulations de conception initiale (`t2-statistics.md`, `t2-fingerprinting.md`). Conservés comme référence de méthode, remplacés comme plan d'exécution par cette v3.
- v3 (ce document) : première version où chaque chiffre cité est une mesure sur le jumeau local ou une simulation calibrée sur cette mesure, avec traçabilité PR par PR.
