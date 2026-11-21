# T2 · méthode v4 (session t2-methode d'Adem) : mode d'emploi pour Titouan

Rien ici n'appelle l'API ni ne lit la clé. Tu gardes l'accès : ce dossier te donne les sondes, le plan de dépense et l'analyse.

## Ordre des opérations (toutes les requêtes sont les tiennes)
Les fichiers de sondes sont des `.jsonl` (`id`, `prompt`) lisibles tels quels par `query_reference.py` et `task_template.py` (`PROMPTS_FILE`).

0. **Le régime de décodage d'abord** (rapport `reports/t2-t2-methode.md` §3.7 et §6) : la ronde 0 sur les 9 `ref-*` dit si le décodage par défaut est déterministe. Si seul ref-08 change d'une réponse à l'autre, cette méthode s'applique (score local 0,84 sur la structure du corrigé). Si tous changent, elle ne tient pas (MCC local 0,02 à 0,27) : ne pas dépenser les `ep-*` avec ce plan.
1. **Ronde 0** : `probes/ronde0.jsonl`, `REPEATS = 2`, tous les endpoints (8 réponses chacun).
2. Sépare les endpoints : `python analyse/plan_budget.py --answers answers.jsonl` écrit `endpoints_deterministes.txt` et `endpoints_echantillonnes.txt` (un endpoint échantillonne si deux réponses au même message diffèrent).
3. **Endpoints déterministes** : `probes/deterministe.jsonl`, `REPEATS = 1` (92 réponses). Total 100.
4. **Endpoints échantillonnés** : `probes/echantillonne_x2.jsonl` avec `REPEATS = 2` (32), puis `probes/echantillonne_x4.jsonl` avec `REPEATS = 4` (64, dont 4 déjà payées en ronde 0 que le script saute). Total 100. Si tous les endpoints échantillonnent, c'est ce plan pour tous, et le pipeline bascule seul sur le décodeur par distributions (rapport §3.7).
5. Analyse (stdlib seule) :
   ```
   python t2_pipeline.py --answers answers.jsonl --probes probes/probes_gelees.jsonl --out soumission --prefix ep-
   ```
   Sorties : `soumission.csv` (principale), `soumission_repli.csv` (familles seules, tout isolé), `soumission_diag.txt` (type détecté, prompt système extrait, groupe central, rattachements). Options : `--variante centre|ancre|prudent`, `--families fichier.csv` pour imposer tes familles.
6. Valide avec ton outil : `python tools/validators/validate_t2.py soumission.csv --strict`.

Faire d'abord les étapes 1 à 5 sur les 9 `ref-*` (`--prefix ref-`) puis `score_practice.py` : c'est la seule mesure réelle de la méthode.

## Ce que contient le dossier
| Fichier | Rôle |
|---|---|
| `probes/probes_gelees.jsonl`, `probes/plan.json` | les 96 sondes gelées et le plan de 100 réponses |
| `probes/make_probes.py`, `probes/geler.py` | génération déterministe des sondes (graine 2026) |
| `analyse/parse.py`, `analyse/features.py` | analyse des réponses, statistiques par endpoint et par paire |
| `analyse/rules.py` | décodeur à règles (types P, S, T ; rattachement) |
| `t2_pipeline.py` | réponses → CSV `endpoint,family,model` |
| `analyse/wm_default_keys.py` | détecteur de filigrane KGW aux clés par défaut (demande torch + le tokenizer candidat) |
| `bench/world.py`, `bench/train_lora.py` | monde local d'endpoints (H200), pour refaire les mesures |
| `analyse/sampled.py` | tests à deux échantillons du régime échantillonné (collision par item, textes libres, longueur) |
| `analyse/plan_budget.py` | après la ronde 0 : liste des endpoints déterministes et échantillonnés |
| `analyse/eval_rules.py`, `analyse/eval_S.py`, `analyse/learn2.py`, `analyse/e2e.py`, `analyse/simulate_ep.py`, `analyse/score_local.py` | banc de mesure (régime glouton, régime échantillonné, classifieurs appris, bout en bout) |
| `analyse/explore*.py`, `analyse/dump_*.py`, `analyse/learn.py`, `probes/make_probes_v2.py` | explorations dont les chiffres sont dans le rapport (statistiques seules, neutraliseurs, familles) |
