# T2 Model Equality : validation locale et checklist avant soumission

Tout ce qui est ici tourne en Python standard, **sans aucun appel réseau**. La soumission réelle
(`reports/sources/t2/submission.py`) est réservée à Titouan (issue #4) ; les autres agents s'arrêtent au dry-run.

## Fichiers

| Fichier | Rôle |
|---|---|
| `validate_t2.py` | Validateur du CSV. CLI + API Python (`parse_submission`, `changed_pairs`). |
| `test_validate_t2.py` | 22 tests unitaires (stdlib `unittest`), CSV valides et invalides fabriqués à la main. |
| `dryrun_submit_t2.py` | Reconstruit la requête HTTP exacte de `submission.py` sans l'envoyer (socket désactivée). |

```bash
python3 tools/validators/validate_t2.py submission.csv
```
```bash
python3 tools/validators/validate_t2.py submission.csv --previous derniere_soumise.csv --strict
```
```bash
python3 tools/validators/dryrun_submit_t2.py submission.csv --show-query-format
```
```bash
python3 tools/validators/test_validate_t2.py
```

Codes de sortie : `0` = valide, `1` = invalide (ou avertissement en `--strict`), `2` = erreur interne du dry-run.

## Règles du serveur (source : `reports/sources/t2/task_template.py`, `submission.py`, `Model_Equality_Testing.txt`)

Ce sont les règles dont la violation entraîne un **rejet** ou un **score faux**. Le validateur les traite en **ERREUR**.

| # | Règle | Source | Conséquence si violée |
|---|---|---|---|
| R1 | Extension `.csv` | `ALLOWED_SUFFIXES = (".csv",)` | `submission.py` refuse avant d'envoyer |
| R2 | Exactement 3 colonnes `endpoint,family,model` | docstring `task_template.py` | rejet serveur |
| R3 | En-tête `endpoint,family,model` **facultative** (détectée insensible à la casse) | idem | — |
| R4 | Ordre des lignes indifférent | idem | — |
| R5 | Une ligne par endpoint `ep-01` … `ep-33`, chacun **une seule fois** | `NUM_ENDPOINTS = 33` | endpoint manquant ou en double → 528 paires scorées fausses |
| R6 | Pas d'endpoint `ref-*` (jamais scorés) | `Model_Equality_Testing.txt` | rejet serveur |
| R7 | **Pas de virgule** dans une étiquette, même entre guillemets | « Labels are any text you like (no commas) » | décalage de colonnes ; on ignore si le serveur gère le quoting, donc on refuse systématiquement |
| R8 | Même étiquette `model` ⇒ même étiquette `family` | « they must also share the family label » | rejet serveur |
| R9 | Étiquettes non vides | prudence (une étiquette vide ferait un cluster fantôme) | ERREUR |
| R10 | UTF-8 **sans BOM** | prudence (`ï»¿ep-01` ≠ `ep-01`) | endpoint `ep-01` inconnu → rejet |
| R11 | Fichier UTF-8 décodable | prudence | ERREUR |

### Avertissements (pas forcément rejetés, mais un signal qu'on s'est trompé de fichier)

| Avertissement | Pourquoi |
|---|---|
| « tout différent » (33 modèles distincts) | c'est le baseline `solve()` de `task_template.py` : **score 0** |
| « tout identique » (1 famille, 1 modèle) | verdict dégénéré |

`--strict` transforme ces avertissements en erreurs (utile dans un script de soumission).

### `--previous`

Compare le verdict « même modèle » de chaque paire avec un fichier déjà soumis et affiche le nombre de paires
qui changent. Sert à vérifier qu'une nouvelle soumission est bien **différente** de la précédente (délai de 30 min
entre deux soumissions, seule la meilleure compte).

## Format exact de la requête de soumission (reconstruit par `dryrun_submit_t2.py`)

```
POST http://34.147.229.55/submit/33-model-equality
X-API-Key: <clé de l'équipe>
Content-Type: multipart/form-data; boundary=...
timeout: (600, 600)   # le serveur attend jusqu'à 500 s pour le score

--boundary
Content-Disposition: form-data; name="file"; filename="submission.csv"

<contenu du CSV>
--boundary--
```

Réponses :
- `200` + `{"status": "success", "score": …, "previous_best": …, "improved": …, "message": …}` → score public
- `200` + `{"status": <autre>, "message": …}` → échec de scoring
- `!= 200` + `{"detail": …}` → rejet (format, clé invalide, délai de 30 min non respecté…)

Le dry-run **remplace `socket.socket` par une fonction qui lève une exception** avant même de lire le CSV, donc il
est physiquement incapable de joindre `34.147.229.55`, même en cas de bug. Il le vérifie lui-même en fin
d'exécution. La clé n'est **jamais lue dans `.env`** : variable d'environnement `EAC_T2_API_KEY` ou placeholder.

## Format des requêtes de collecte (pour mémoire, `task_template.py`)

```
GET  http://34.147.229.55/33-model-equality/v1/models          -> budget {limit, used} par endpoint
POST http://34.147.229.55/33-model-equality/v1/chat/completions
     {"model": "ep-XX", "max_tokens": 256, "messages": [{"role": "user", "content": "<prompt>"}]}
```
- 4 requêtes en parallèle max par équipe ; `429` + `Retry-After` = occupé, `429` sans = **budget épuisé**
- Les messages `system` sont **rejetés** ; `temperature` et `seed` sont ignorés
- Requête invalide = pas décomptée du budget
- **Budget : 100 réponses par endpoint pour toute la compétition, irréversible.** Aucune requête `ep-*` sans plan posté sur l'issue #4.

## Checklist avant soumission (à cocher dans l'ordre)

1. [ ] Le fichier a été produit par le code de la branche fusionnée dans `main` (pas un fichier à la main)
2. [ ] `python3 tools/validators/validate_t2.py submission.csv --strict` → `RESULTAT: VALIDE`
3. [ ] `--previous <dernier CSV soumis>` : le nombre de paires changées est > 0 (sinon on gaspille le créneau)
4. [ ] Score local sur `ref-01..ref-09` via `reports/sources/t2/score_practice.py` noté dans l'issue #4
5. [ ] `python3 tools/validators/dryrun_submit_t2.py submission.csv` → `DRY-RUN TERMINE`
6. [ ] Dernière soumission T2 il y a **≥ 30 min** (délai serveur)
7. [ ] Annonce sur l'issue #4 : « soumission T2 à HH:MM, méthode X, score local Y » signée `[agent de <prénom> · T2]`
8. [ ] **Titouan uniquement** lance `reports/sources/t2/submission.py` (clé dans `.env`, jamais collée dans une issue)
9. [ ] Score public reporté dans `tracks/t2-model-equality/CONTEXTE.md` (tableau État) et dans l'issue #4
10. [ ] Ne pas sur-optimiser le public : il ne couvre que 30 % des paires, la note finale vient des 70 % cachés

## Ce que le validateur ne peut pas vérifier

- Que les clusters sont **justes** : seul `score_practice.py` sur les `ref-*` donne un chiffre.
- Que la clé API est valide ou que le délai de 30 min est écoulé (le serveur répond `!= 200` avec un `detail`).
- Le comportement exact du parseur serveur sur les cas limites (guillemets, espaces) : on reste du côté
  strict (pas de virgule du tout, étiquettes `strip()`ées) pour ne pas avoir à le découvrir avec un vrai créneau.
