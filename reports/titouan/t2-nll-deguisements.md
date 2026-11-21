# T2 · Classifieur de paires fondé sur la NLL croisée (signal principal, pas les sondes à choix forcé)

**Décision débloquée :** la NLL croisée (PR #13) peut servir de signal *principal* pour décider « même modèle ? » entre deux guichets, en passant *uniquement* par `.query()` (texte seul) et un lecteur local unique — mais sur l'échantillon réellement collecté (21 prompts, 1 tirage chacun, modèle 360M), le signal n'a *pas assez de puissance statistique* : MCC = 0,000, même échec que les sondes à choix forcé. Pour que la méthode serve, il faut des tirages répétés par prompt (comme le mining d'histogrammes en cours sur l'issue #4), pas plus de prompts distincts.

Code : `tracks/t2-model-equality/twin/nll_pairs.py` (classifieur + simulation locale) et `nll_pairs_real.py` (même classifieur appliqué aux vraies réponses `ref-*` déjà collectées). Aucune requête envoyée vers l'API du hackathon par cette sous-session.

## Le signal : NLL croisée avec un seul lecteur, pas deux

PR #13 montre que deux copies des *mêmes poids* qui se relisent *l'une l'autre* donnent un écart (gap) exactement nul, quel que soit le déguisement de décodage — parce que le déguisement change le token choisi, pas la probabilité qu'un modèle assigne à un texte déjà écrit. Mais une vraie API ne renvoie que du texte (`Model_Equality_Testing.txt` : *"returns generated text only"*) : on n'a jamais les poids (ni les logprobs) d'aucun guichet, donc on ne peut pas faire *littéralement* le test de PR #13 (il lit chaque texte avec les *deux* jeux de poids).

Solution retenue : un **lecteur local unique** (`ref_model`, jamais partagé avec un guichet) relit le texte de *chaque* guichet. Deux complications empêchent de comparer la NLL brute :
1. **Contexte caché** (system prompt / persona) : le lecteur ne le voit pas → neutralisation par une sonde d'extraction, en gardant le candidat (`aucun contexte` / `contexte extrait`) qui minimise le **t apparié** (pas la NLL brute — une extraction imparfaite laisse un écart *moyen* consistant qui gonflerait le t à tort si on choisissait par NLL la plus basse).
2. **Décodage inconnu** (température, filigrane) : relève mécaniquement la NLL même pour les *mêmes poids*. Solution : un **atlas de nuisances** — le lecteur génère *lui-même* sur les mêmes prompts à une grille de températures et à une grille de filigranes KGW (γ = 0,25, **clé arbitraire du lecteur**, différente de toute clé cachée : la signature statistique NLL/taux-hors-argmax d'un filigrane ne dépend que de (γ, δ), jamais de la clé, donc on peut la reproduire sans la connaître). Chaque guichet est comparé au point le plus proche de cet atlas via un **t de Student apparié par prompt** (convention PR #13) : un vrai écart de poids donne un t grand et stable même quand l'écart moyen est petit (fp8, LoRA faible) ; un déguisement de décodage mal modélisé par la grille reste un t modéré car le résidu fluctue sans direction propre aux poids.

## 1. Validation sur jumeau simulé (OLMo-2-0425-1B, hypothèse initiale de la tâche)

60 prompts, décodage glouton pour les 5 variantes « mêmes poids » (plain/system/persona/watermark δ=8), échantillonné pour `ref-08`, LoRA *entraînée* (r=8, q_proj/v_proj, comme `exp1_divergence.py`) pour `ref-04/07/09`, fp8 réel pour `ref-02` :

| Guichet (vérité) | t apparié vs atlas | Décision |
|---|---|---|
| ref-01 (plain) | 0,00 | poids = lecteur |
| ref-06 (watermark δ=8) | 0,09 | poids = lecteur |
| ref-03 (system prompt) | 0,47 | poids = lecteur |
| ref-08 (température 1,0) | 1,57 | poids = lecteur |
| ref-09 (LoRA 0,3) | 0,92 | **faux** (poids = lecteur) |
| ref-02 (fp8) | 1,29 | **faux** (poids = lecteur) |
| ref-07 (LoRA 1,5) | 4,30 | poids ≠ lecteur |
| ref-04 (LoRA 0,7) | 11,07 | poids ≠ lecteur |

Les quatre déguisements de décodage (plain, system prompt, watermark δ=8, température) sont correctement reconnus comme « mêmes poids » malgré un résidu de distance non nul — l'atlas absorbe le déguisement. LoRA 0,7 et 1,5 sont nettement détectés comme poids différents (t = 11 et 4,3). LoRA 0,3 et fp8 restent au bord de la détection (t ≈ 1), exactement comme PR #13 l'avait déjà documenté pour int8 (t = 1,4) : un écart de poids faible donne un signal faible, pas un défaut de méthode. Un bug de sélection de contexte (choisir par NLL la plus basse plutôt que par t le plus bas) faisait à tort échouer `ref-05` (persona, t = 7,55) dans cette même série ; corrigé dans `nll_pairs.py` (sélection par t minimal) mais pas réexécuté sur GPU avant la réorientation ci-dessous — à revalider.

## 2. Réorientation : le vrai modèle est SmolLM2-360M, tous les guichets échantillonnent

Pendant ce travail, une autre sous-session a mesuré sur le vrai serveur (sondes *identity* et *repeat*) que le modèle de base des `ref-*` est **SmolLM2-360M-Instruct**, pas OLMo, et que **tous** les guichets échantillonnent (température de base ≈ 0,8), pas seulement `ref-08`. L'hypothèse de décodage glouton de `nll_pairs.py` était donc fausse pour les `ref-*` réels. `nll_pairs_real.py` réapplique le même classifieur (atlas centré sur T = 0,8, lecteur SmolLM2-360M local, CPU) directement sur les réponses déjà collectées (`collecte_v4_ref_bloc0_*.jsonl`, 21 prompts communs × 9 guichets, 1 tirage chacun) — zéro requête supplémentaire.

**Résultat : les 9 t appariés sont tous < 1 (0,13 à 0,89). Aucun guichet n'est distingué du lecteur. MCC = 0,000** (`score_practice.py` : 26 paires sur 36 appelées « même » à tort, 0 appelée « différent » à tort — la méthode dégénère en « tout pareil », exactement l'échec initial des sondes à choix forcé).

**Pourquoi :** avec un seul tirage par prompt et un modèle 360M (plus bruyant par token qu'un 1B), le bruit prompt-à-prompt domine l'écart de poids réel sur seulement 21 paires — le t apparié n'a tout simplement pas la puissance statistique nécessaire à cette taille d'échantillon. Ce n'est pas que le signal n'existe pas (la validation simulée au-dessus le montre, avec 60 prompts et un décodage glouton moins bruyant) : c'est qu'**il faut plus d'observations indépendantes par prompt** pour le faire émerger ici.

## Ce qu'on en déduit pour les 33 endpoints

1. **Le principe tient** (validé en simulation) : un lecteur local unique + atlas de nuisances + t apparié sépare correctement les déguisements de décodage (persona, system prompt, filigrane, température) des vrais écarts de poids (LoRA, fp8), sans jamais connaître l'identité du modèle ni le déguisement à l'avance.
2. **Il faut des tirages répétés par prompt, pas seulement plus de prompts distincts**, pour que le t apparié ait assez de puissance sur un petit modèle et un budget de requêtes limité — exactement l'infrastructure déjà construite pour la piste « histogrammes de réponses » (50 questions × 10 tirages, AUC 0,92 en simulation, issue #4). Si cette piste tourne déjà avec des tirages répétés sur les mêmes prompts, la NLL croisée par prompt-moyenné sur ces mêmes tirages serait un second signal indépendant à moindre coût (même lecteur local, même jeu de réponses) — à tester en complément, pas à la place.
3. Pour des questions **ouvertes** (pas de forme fermée), la NLL croisée reste le seul signal du genre applicable (un histogramme de réponses suppose un petit ensemble de réponses possibles) : elle garde un intérêt propre dès que le budget permet des tirages répétés sur quelques prompts ouverts.

## Reproduire

```
python tracks/t2-model-equality/twin/nll_pairs.py run --n 60 --max_new 64 --lora_steps 100 --out results.json   # GPU, jumeau simulé
python tracks/t2-model-equality/twin/nll_pairs_real.py --jsonl <collecte_v4_ref_bloc0_*.jsonl> --out results_real.json   # CPU, vraies réponses
python tracks/t2-model-equality/twin/nll_pairs.py decide --raw results_real.json --submission sub.csv
python reports/sources/t2/score_practice.py sub.csv reports/sources/t2/practice_answers.csv
```
