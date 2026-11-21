# T2 · Méthode v4 : jumeau local, puis vrais ref (score_practice.py)

**Décision débloquée** : quoi brancher sur les 33 `ep-*`, et avec quelle confiance.
Code : `tracks/t2-model-equality/twin/method_v4.py` (+ `run_real.sh`, `run_ep.sh`) ; collecte : branche `t2/collecte-25`.
Juge : `reports/sources/t2/score_practice.py`, inchangé. Les mondes jumeaux sont 100 % locaux ; les seules vraies
requêtes (bloc 0 sur les 9 ref) ont été envoyées par la session principale, pas par cette session.

## 0. Résultat en clair
| où | régime | MCC modèle |
|---|---|---|
| jumeau OLMo-1B, structure du corrigé | glouton | **1,000** (stable de 25 à 97 prompts) |
| jumeau Qwen2.5-1.5B / SmolLM2-1.7B, même procédure en aveugle | glouton | 0,721 / 0,721 (system prompt non extrait) |
| jumeau, variantes elles aussi déguisées (monde A) | glouton | 0,734 (LoRA 0,7 + filigrane passe pour S) |
| jumeau OLMo, tout échantillonne à T=0,7 | échantillonné | 0,02-0,23 |
| jumeau SmolLM2-360M, tout échantillonne à T=0,8, + rapport de vraisemblance | échantillonné | 0,734 / 0,446 / 0,721 (22 / 47 / 89 prompts) |
| **vrais ref, bloc 0 (224 réponses, 20 prompts lus)** | **échantillonné** | **0,273** (plancher local) ; 0,000 (référence pool) |

Ce qui est solide : (1) les vrais ref **échantillonnent** (question répétée différente sur 9/9) ; (2) **famille = SmolLM2-360M-Instruct**,
identifiée par la NLL/token la plus basse sur chacun des 9 ref (0,85-1,19 contre ≥ 1,23 pour les autres candidats) ; (3) la
**garde jumeau** refuse 6/6 jumeaux faux et accepte 3/3 jumeaux justes (en glouton). Ce qui ne l'est pas : séparer fp8, LoRA 0,3
et filigrane des copies en régime échantillonné avec ~20-25 réponses par guichet.

## 1. Le banc jumeau
9 faux endpoints, mêmes identifiants et labels que `practice_answers.csv` : ref-01 S clair, ref-02 fp8_e4m3, ref-03 system prompt,
ref-04 LoRA 0,7, ref-05 persona, ref-06 filigrane KGW δ=6, ref-07 LoRA 1,5, ref-08 T=1,0, ref-09 LoRA 0,3 (LoRA r=8 q/v, recette
PR #15). Mondes : `practice`, `combinedA/B` (variantes déguisées), `allT07`, `allT08`, `allT07combined`, option `--model`.
**Règle d'honnêteté** : la méthode n'atteint un endpoint que par `Endpoint.query()` ; toute log-probabilité vient d'une copie locale
distincte ; la LoRA de calibration est entraînée sur un autre corpus (graine 456) que celle des endpoints (123).

## 2. La méthode (état final)
Par guichet : 2 envois de la même question (échantillonne ? bruit serveur), 1 sonde d'extraction du prompt caché, k questions de lecture.
- **Glouton** : relecture des sorties par la copie locale sous chaque contexte candidat (aucun, 1-3 premiers paragraphes extraits) et
  chaque décodage candidat (logits bruts, `generation_config` du modèle, ex. `repetition_penalty` 1,1 de Qwen) ; part de tokens
  hors argmax ; « même modèle » = test de Welch par prompt (t < 3) contre une référence mesurée sur place (plancher de la copie locale
  ou ancre du pool, la plus haute) ; filigrane = beaucoup d'écarts mais marges plafonnées (relatif à la LoRA locale).
- **Échantillonné** : test de calibration (T* qui annule NLL − entropie, puis ECE) et **rapport de vraisemblance token par token**
  sous des variantes locales de la même base (fp8, int8, LoRA 0,3/0,7/1,5).
- **Garde / sélecteur de jumeau** : écarts durs de l'ancre vs plancher (glouton) ; NLL/token la plus basse entre candidats (échantillonné).
- **Ep** : `assign` (famille = jumeau de NLL minimale) → `decide` par famille → `merge_ep` (CSV `Fi` / `FiS`, `FiTk`) → `validate_t2.py`.
- La couche A demandée (choix forcé façon AgentProv) est **mesurée inversée** (fp8 0,93 d'accord avec le clair, filigrane 0,24) : diagnostic seulement.

## 3. Itérations (chaque chiffre = sortie de `score_practice.py`)
| it. | changement | monde | MCC |
|---|---|---|---|
| 0 | couches B/C lisant les poids cachés | — | invalide (oracle), jeté |
| 1 | accès par `query()` seulement, couche A pour regrouper | OLMo practice | 0,000 |
| 2 | contexte = paragraphes extraits, plafond filigrane, calibration T=1 | OLMo practice | 1,000 |
| 3 | décisions auto-calibrantes (Welch, plafond relatif), `--model` | OLMo / Qwen / SmolLM2 | 1,000 / 0,231 / 0,721 |
| 4 | décodage `generation_config` (pénalité de répétition) | Qwen | 0,721 |
| 5 | mondes combinés | A | 0,734 (1,000 à 22 prompts) |
| 6 | tout échantillonne | OLMo T=0,7 | 0,02-0,23 |
| 7 | rapport de vraisemblance, grille de T fine | SmolLM2-360M T=0,8 | 0,72 (instable 0,45-0,73) |
| 8 | **vrais ref**, jumeau choisi par NLL (SmolLM2-360M) | réel | **0,273** |
| 9 | vrais ref, référence = pool (règle fixée avant, appliquée une fois) | réel | 0,000 |

Sonde d'extraction, 7 formulations × 3 familles : verbatim **seulement sur OLMo** (formulations 0 et 1) ; Qwen paraphrase à la
1re personne, SmolLM2 jamais. Neutraliser le system prompt par extraction est donc propre à OLMo.

## 4. Pourquoi le jumeau est trop optimiste (à lire avant tout chiffre jumeau)
- **Variantes alignées** : dans le jumeau, la fp8 locale est la même fausse quantification que la fp8 cachée, et la LoRA locale suit la
  même recette ; le rapport de vraisemblance les voit (fp8 t=4,4, LoRA 0,3 t=7,9). Sur les vrais ref, fp8 et LoRA des organisateurs
  diffèrent : la colonne « LoRA 1,5 locale » se déclenche sur 7 ref / 9 sans rapport avec la vérité.
- **Plancher** : le jumeau sert et relit avec le même code ; le vrai serveur (vLLM probable, autres noyaux, peut-être top-p) ne ressemble
  pas exactement à nos propres tirages, d'où des faux « différent » (ref-01, le clair lui-même, déclaré variante).
- **Puissance** : 20 prompts × ~120 tokens ne suffisent pas en échantillonné. Le jumeau montre une forte instabilité selon k.

## 5. Sensibilité (jumeau OLMo practice, glouton)
| paramètre | valeurs → MCC |
|---|---|
| k (prompts lus), décision auto-calibrante | 5 → 0,33 ; 8 → 0,52 ; 10 → 0,73 ; 12-18 → 1,00 ; 20 → 0,73 ; 22-89 → 1,00 |
| plafond filigrane (ancienne version absolue) | 5 → 0,72 ; 6-12 → 1,00 |
| multiplicateur du plancher (ancienne version) | 1-2 → 1,00 ; 3 → 0,73 ; 4 → 0,52 |

## 6. Jumeau faux (garde)
Matrice 3×3 (OLMo / Qwen / SmolLM2, chaque monde relu par les deux autres) : 6/6 jumeaux faux refusés (Welch t = 9 à 30, « aucun jumeau
ne colle », pas de soumission), 3/3 jumeaux justes acceptés (t = 0). Sans garde : 0,021 (réponse fausse et sûre d'elle).
En régime échantillonné l'ECE ne départage pas les jumeaux ; la NLL/token le fait (vrais ref : 360M nettement en tête).

## 7. Règle pré-enregistrée pour les 33 ep (issue #4, avant toute donnée ep)
`run_ep.sh` : famille = jumeau de NLL minimale parmi 12 candidats (SmolLM2 135M/360M/1.7B, SmolLM3-3B, Qwen2.5 0,5/1,5/3/7B,
OLMo-2 1B/7B, Phi-3.5-mini, Mistral-7B) ; puis `decide` par famille avec le plancher local (pas de référence pool, 0,273 contre 0,000
sur les vrais ref) ; puis CSV validé. Attendu : terme famille (10 %) correct si les vraies familles sont parmi les candidats ; terme
modèle (90 %) faible et variable (0 à 0,3).

## 8. Suite proposée (par ordre de gain attendu)
1. Envoyer les blocs 1-3 sur les **ref** (illimités) et mesurer la courbe MCC(k) réelle ; c'est la seule calibration qui compte.
2. Construire les variantes locales comme le ferait le serveur (fp8 vLLM par tenseur, LoRA sur données d'instruction génériques)
   pour que le rapport de vraisemblance s'aligne sur les vraies variantes.
3. Tester la comparaison de répartitions de réponses d'Adem (50 questions × 10 tirages) sur les ref illimités.

## 9. Reproduire
```bash
python method_v4.py collect --world practice --model allenai/OLMo-2-0425-1B-Instruct --skip_a --out raw.json   # jumeau, GPU
python method_v4.py reread --raw raw.json --out reread.json                                                     # GPU, local
python method_v4.py decide --raw raw.json --reread reread.json --submission sub.csv --k_bc 89                 # appelle score_practice.py
./run_real.sh '<journaux ref>' <tag>      # vrais ref : 5 jumeaux candidats, decide, score_practice.py
./run_ep.sh '<journaux ep>' <tag>         # vrais ep : règle pré-enregistrée, CSV + validate_t2.py
```

## 10. Levier 5 : fp8 dans la famille qwen3b, cas ep-12 (`twin/lever5.py`, 0 requête)
Jumeau Qwen2.5-3B : 8 nulls (base) contre 8 fp8 (4 e4m3 poids par tenseur + 4 W8A8 par tenseur), tirage par la
`generation_config` du modèle, sur les 24 prompts des vrais ep.
- fp8 d'un 3B **détectable à 24 réponses** : AUC 1,00, puissance 100 % à z ≥ 3, 0 faux positif ; « tous les tokens » ≥ « top
  20 % de KL » > pondération par la KL.
- Une fp8 simulée allume **les deux** lecteurs (z de 4,9 à 8,4 chacun) ; nulls ≤ 1,9 / 2,1, somme ≤ 4,0. Règle proposée :
  z_e4m3 + z_W8A8 ≥ 5 (jumeau optimiste : lecteur aligné ; vrai fp8 ref-02 = 3,4 / 2,8).
- Vrais ep qwen3b : ep-12 = +3,1 / +0,4 (somme 3,5, profil non fp8, dans la zone des nulls) ; tous les autres ≤ 1,3 / 1,5.
  Avec la règle : ep-22 (11,1), ep-15 (6,9) passent, ep-05 au seuil (5,0), ep-12 non.
