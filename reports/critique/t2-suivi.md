# T2 Model Equality : journal de la relecture critique

Tenu par la session « t2-critique » d'Adem. Décision que ce journal débloque : **peut-on dépenser le budget `ep-*` avec la méthode v3 de Titouan, ou faut-il d'abord la corriger ?** Aucune requête API, aucune clé lue, aucune branche de Titouan modifiée. Dernière mise à jour : samedi 21/11, 18 h 30.

## En une phrase
La méthode v3 compare les textes mot pour mot. Le corrigé d'entraînement montre que les endpoints « même modèle » sont justement ceux dont le texte est maquillé (prompt système, persona, filigrane, température). Telle quelle, la v3 les séparerait tous et marquerait autour de 0 sur les `ref-*`. À vérifier par une vraie mesure sur les `ref-*` avant de toucher aux `ep-*`.

## Ce que j'ai vérifié par le calcul
| Vérification | Résultat | Où |
|---|---|---|
| Structure du corrigé (`practice_answers.csv`) | 5 « même modèle » = ref-01 en clair + 4 copies déguisées ; 4 variantes seules (fp8, LoRA 0,3 / 0,7 / 1,5) | issue #4 |
| Score de la v3 sur le corrigé si les copies déguisées divergent à plus de 43 % | MCC 0,000 (tout isolé) ; 0,486 si 2 copies sur 4 sont retrouvées ; 0,72 si 3 sur 4 | issue #4 |
| `mcc_simulation.py` rejoué avec le comptage officiel (paires d'une même famille seulement) | 0,963 → 0,958 (S1), 0,916 → 0,902 (S5) : écart faible | PR #12 |
| Même script avec des copies déguisées | 80 % déguisées : 0,17-0,32 ; 100 % : 0,00 ; meilleur seuil possible : 0,45-0,56 | PR #12 |
| Coût d'une erreur sur le corrigé | isoler 1 copie = 0,72 ; fusionner fp8 = 0,73 ; isoler 2 copies = 0,49 ; fusionner fp8 + LoRA 0,3 = 0,52 | PR #14 |

Script de recalcul : `recheck.py` (scratch de la session, importe `mcc_simulation.py` sans le modifier). À recopier dans `tools/` si l'équipe veut le garder.

## Risques ouverts
| # | Risque | État | Mesure qui tranche |
|---|---|---|---|
| R1 | Les copies déguisées sont isolées par la comparaison exacte → MCC proche de 0 | **Ouvert, majeur.** Signalé issue #4 + PR #12, #14 | `score_practice.py` de la v3 sur les 9 ref |
| R2 | Le décodage par défaut n'est peut-être pas glouton (`query_reference.py` : « answers are sampled ») | Ouvert | Même prompt 3 fois sur ref-01 |
| R3 | Limite 100 sur les `ref-*` annoncée levée (Discord 16 h 37), jamais vérifiée | Ouvert | `GET /models` (gratuit) |
| R4 | Ronde A = 16 prompts × 3 répétitions : 32 réponses sur 48 perdues si l'endpoint est déterministe | Ouvert, PR #10 | Ronde 0 de 4 réponses par endpoint |
| R5 | Tests de déguisement du mining faits avec le même prompt système et la même graine des deux côtés : la fausse alerte entre mêmes poids n'est pas mesurée | Ouvert, PR #17 | 3 colonnes « mêmes poids, déguisement différent » sur le jumeau |
| R6 | Règle filigrane (« divergence ≤ 4 tokens ») incapable de distinguer filigrane et LoRA forte | Ouvert, PR #14 | AUC des statistiques sur les 36 paires ref |
| R7 | NLL croisée inutilisable (pas de logprobs) | Ouvert, PR #13 | Relecture à sens unique par les modèles locaux ; requête ref avec `logprobs: true` |
| R8 | Règle de famille qui exclut les copies déguisées | Ouvert, PR #16 | Relecture OLMo / Qwen / SmolLM de textes OLMo déguisés |
| R9 | « Isoler par défaut » coûte cher quand le doute porte sur une copie | Ouvert, PR #14 | Calcul fait (ci-dessus) |
| R10 | Jumeau OLMo-2-1B : rien ne prouve que les `ep-*` ni les `ref-*` (« un petit modèle séparé ») sont ce modèle | Ouvert, pas encore posté | Question d'identité + relecture locale sur les ref |
| R11 | Validateur : soumission « chaque famille fusionnée » (MCC 0) non détectée | Ouvert, PR #11 | 3 contrôles + 3 tests |

## Relectures postées (21/11)
- Issue #4 : constat principal + 3 mesures gratuites à faire avant la Ronde F de 21 h 30.
- PR #10 (plan de budget) : Ronde 0 de déterminisme, Ronde F trop chère pour 10 % de la note, étape 0 sur ref pas faite.
- PR #11 (validateur) : 3 trous.
- PR #12 (simulation) : comptage officiel + copies déguisées.
- PR #13 (filigrane) : NLL à sens unique, test `logprobs`.
- PR #14 (v3) : deux règles contredites par ses mesures, classement du régime d'abord.
- PR #15 (LoRA) : la courbe estime la force, pas l'égalité ; longueur confondue avec la persona.
- PR #16 (famille) : règle qui exclut les copies déguisées.
- PR #17 (mining) : fausse alerte non mesurée ; quasi-égalités fragiles à T 1,0 (≥ 40 % de divergence pour le même modèle).

## Réponses de Titouan
Aucune pour l'instant.

## Idées pas encore reprises
1. Ronde 0 : test de déterminisme par endpoint (4 réponses) avant de figer la forme des autres vagues.
2. Classer le régime de chaque endpoint (déterministe, échantillonné, prompt système, persona) puis choisir la statistique par couple de régimes.
3. Relecture à sens unique par les modèles locaux (remplace la NLL croisée).
4. Banc d'essai sur les 9 ref : AUC « même / différent » de chaque statistique sur les 36 paires.
5. À poster au prochain passage : lots de mini-questions numérotées dans une seule réponse (le budget compte des réponses, pas des tokens) ; faux historique de conversation long pour noyer le prompt système ; seuil par « plus grand trou » dans chaque famille plutôt qu'un seuil fixe ; soumission plancher précoce.
