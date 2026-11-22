# T2 : ce que révèle AgentProv (et le dépôt `agent_prov`) sur le dispositif des organisateurs
Sam. 21/11/2026. Aucun appel au serveur du hackathon, aucune API payante, aucun commit.
Étiquettes : [LU] = lu à la source (URL donnée) ; [IDÉE] = notre déduction, jamais mesurée ; [NON VÉRIFIÉ] = affirmation que je n'ai pas pu confirmer.

**Décision débloquée :** à quoi ressemble notre monde simulé (quantifications, LoRA, prompts système, filigranes) et quelles sondes copier d'AgentProv.

## Résumé en 5 lignes
1. Le texte intégral et le code sont publics : arXiv HTML/PDF et dépôt `github.com/sprintml/agent_prov` (créé le 27/09/2026, 2 étoiles). Je les ai lus en entier.
2. **AgentProv ne contient NI filigrane, NI persona, NI LoRA « de force x », NI fp8, NI variante de décodage.** Ces variantes (celles de `ref-*`) sont le dispositif propre du hackathon, écrit par les mêmes auteurs (Xun Wang, Bihe Zhao) mais absent du papier. Ce qu'on peut copier : les prompts système, la quantification 8/4 bits, la méthode de sonde.
3. Résultat négatif central : la quantification 8 bits est acceptée « même modèle » par les trois tests (AgentProv, MET, RUT). Or l'énoncé la classe « modèle différent » : `ref-02` (fp8) est probablement la paire la plus dure, et le papier ne dit pas comment la résoudre.
4. Le tableau de budget du papier contredit le plan « 20 gabarits x 5 réponses » : à 100 réponses, **10 gabarits x 10 réponses (91,9 %) bat 20 x 5 (65,6 %)**.
5. Les sondes d'AgentProv utilisent des appels d'outils (`tools`). L'API du hackathon n'expose peut-être que du texte : à tester gratuitement sur `ref-*` (pas de limite de requêtes là-bas).

Vocabulaire (une fois). **MMD** : distance entre deux nuages d'échantillons. **Noyau delta** : mesure de ressemblance qui vaut 1 si deux réponses tombent dans la même case, 0 sinon. **Test par permutation** : on mélange au hasard les étiquettes des deux endpoints pour fabriquer ce que le hasard seul donnerait, puis on regarde si la vraie distance dépasse. **Appel d'outil** : le modèle choisit une fonction (ex. `send_email`) au lieu d'écrire du texte. **MET / RUT** : les deux tests de texte existants (comparaison caractère par caractère / rang de chaque mot dans la distribution d'un modèle de référence).

## Sources et ce que j'ai essayé
- arXiv abs https://arxiv.org/abs/2609.00052 : OK. HTML https://arxiv.org/html/2609.00052v1 et PDF https://arxiv.org/pdf/2609.00052 : téléchargés et convertis en texte (curl + pdftotext), lus en entier, annexes comprises. Version v1 du 30/08/2026 ; la version finale EMNLP peut différer [NON VÉRIFIÉ].
- ACL Anthology https://aclanthology.org/2026.emnlp-main.638.pdf : la page se charge mais l'extracteur de l'outil web n'a pas pu décompresser le PDF ; je me suis rabattu sur arXiv.
- Dépôt https://github.com/sprintml/agent_prov : arborescence lue par l'API GitHub, fichiers clés téléchargés (`README.md`, `config/*.yaml`, `run_probe.py`, `src/tests/mmd.py`, `src/adapters/hf_adapter.py`, `src/harness/probe_runner.py`, `config/probes/*.yaml`).
- Hugging Face : `SprintML/model_equality` (6 fichiers : `practice_answers.csv`, `query_reference.py`, `requirements.txt`, `score_practice.py`, `submission.py`, `task_template.py`) ; `SprintML/hackathon` (installation Jülich seulement). Rien d'autre sur les modèles des endpoints.
- Recherche web pour d'autres papiers de Xun Wang / Bihe Zhao sur l'audit d'API : rien trouvé hors AgentProv. Papiers voisins repérés seulement par leurs titres, non lus : Zhang et al. « Real money, fake models » (arXiv 2603.01919, co-auteur Backes), Cai et al. « Are you getting what you pay for? » (arXiv 2504.04715).

## 1. Les « variantes de déploiement » d'AgentProv
**Modèles de base [LU, tableau 11 / annexe A.3].** 36 modèles ouverts (Qwen2.5 0,5 à 7B, Qwen3, Llama-3.1/3.2, Mistral, DeepSeek, R1-Distill, Hermes-3, Hammer-2.1, Phi, Gemma-3-1B, GLM, SmolLM3...) et 9 modèles OpenAI. **OLMo n'apparaît nulle part** (recherche du mot : 0 occurrence dans le papier et `models.yaml`). La base des endpoints du hackathon ne vient donc pas du papier.

**Quantification [LU, annexe A.8, tableau 14].**
- Test sur Qwen2.5-7B et Llama-3.1-8B, « servis via vLLM », pleine précision contre 8 bits et 4 bits.
- 8 bits : AgentProv, MET et RUT acceptent tous (« même modèle »), pour les deux modèles.
- 4 bits : MET et RUT rejettent les deux ; AgentProv accepte Qwen2.5-7B et rejette Llama-3.1-8B. Le papier conclut « mixte et dépendant du modèle ».
- Méthode exacte de quantification (bitsandbytes ? AWQ ? GPTQ ?) : **non donnée**. Le code `hf_adapter.py` lit des options `load_in_8bit` / `load_in_4bit` (l'interface bitsandbytes de transformers), mais aucune entrée de `models.yaml` ne les active : les modèles quantifiés de l'annexe ne sont pas dans le dépôt [LU]. Pas de GGUF, AWQ ni fp8 mentionnés.

**LoRA [LU, section 4.7, tableau 8].** Seulement comme attaque : Qwen2.5-1.5B distillé par LoRA sur les sorties d'appels d'outils de Qwen2.5-7B, appris sur 48 gabarits, testé sur 12 autres. Il passe le test sur les gabarits d'entraînement (p = 1,00) mais est rejeté sur les gabarits inconnus (p = 0,013). Rang, modules, données, notion de « force » : **non donnés**. Rien de commun avec les LoRA à force 0,3 / 0,7 / 1,5 du hackathon.

**Filigrane de décodage.** Aucun dans AgentProv. Seule mention : AGENTWM (Wang et al. 2026a), un filigrane d'actions d'agent, cité comme travail voisin et non utilisé [LU, section 2]. Pour le texte, les données de la finale de juillet 2026 contenaient KGW, Unigram, Gumbel-Max, TextSeal (voir `reports/labos-partenaires-attendus.md`, §0) : c'est le menu le plus probable pour Paris [IDÉE, non confirmé pour T2].

**Prompts système [LU, section 4.3 et `config/hidden_prompts.yaml`].** Textes exacts :
- vide : une seule espace `" "` (1 token) ;
- court : `"You are a helpful assistant."` (~5 tokens) ;
- long : `"You are an AI assistant deployed by Acme Corp for customer support. Always respond in a friendly, professional tone. Refuse requests unrelated to Acme products."` (~50 tokens).
- Annexe A.8 : deux autres (préambule de sécurité « helpful, harmless, honest » et « protocole d'agent de production » demandant un appel bien formé) dont le texte n'est pas donné.
- Prompt système propre aux sondes : `"You are a helpful assistant. Use the provided tools when needed to answer the user's question."` [LU, `config/prompts.yaml`].

**Personas.** Pas de persona au sens du hackathon. Plus proche : Hermes-3 (réglage « persona de chat » sur Llama/Mistral), qui reste plus proche de sa base que de ses « frères » Hermes [LU, tableau 2].

**Décodage [LU, annexe A.9 et code].** AgentProv : température 0,7 (code : défaut `temperature=0.7, top_p=0.9, max_new_tokens=256`). MET : température 1,0, top-p 1,0, 25 prompts Wikipedia (5 prompts x 5 langues EN/DE/FR/RU/ES), 50 réponses par prompt, 50 tokens max. RUT : température 0,5, 100 prompts WildChat, 1 réponse cible et 100 réponses de référence, 30 tokens max. Top-k et faisceau : jamais mentionnés. **Aucun test d'un désaccord de température entre endpoints** [LU par absence].

## 2. Serveur
- Quantifications servies via **vLLM** [LU, A.8]. Références locales : transformers (`hf_adapter.py`) sur GPU ; endpoints tiers : OpenRouter (`openrouter_adapter.py`) [LU].
- Le serveur du hackathon n'est pas décrit. `ref-02 = « quantized (fp8) »` [LU, `practice_answers.csv`] colle avec l'option `--quantization fp8` de vLLM, mais rien ne le prouve [IDÉE]. Le corps des réponses contient `choices[0].message.content` et un bloc `budget` [LU, `query_reference.py`] ; la présence d'un champ `usage` est inconnue [NON VÉRIFIÉ].

## 3. La méthode AgentProv
- **Gabarit de sonde [LU, 3.2, A.1, A.2].** Un message utilisateur neutre + 3 à 6 outils fonctionnellement équivalents à noms différents (`create_event`, `schedule_meeting`, `add_event`, `book_calendar_event` ; ou `mcp__gmail__send_email`, `mcp__outlook__send_email`, `mcp__smtp__send_mail`). Descriptions identiques, ordre des outils mélangé à chaque échantillon (graine fixe), destinataires neutres (`@example.com`), consigne sans nom d'outil (« envoie une note », pas « email »). Le modèle ne voit jamais les cases ; on lit le nom de l'outil choisi. Cases spéciales : `_no_call`, `_malformed`, `_other`.
- **Pool [LU, dépôt].** `s_redundant.yaml` : 15 catégories (météo, bourse, e-mail, agenda, code, fichiers, traduction, paiement...) x 3 prompts + 15 « neutres » = 60 gabarits ; `s_init.yaml` : 6 gabarits MCP (e-mail, agenda, stockage, messagerie, recherche, mémoire). Config principale : K = 20 gabarits fusionnés à tour de rôle. Le papier parle d'un pool de K = 60 [écart non réconcilié, NON VÉRIFIÉ].
- **Pourquoi le test résiste au prompt système [LU, 3.1, 4.3].** Il compare des choix discrets (quel outil), pas des phrases : un prompt système change les mots mais rarement l'outil choisi. Noyau delta : MMD = distance L2 au carré entre histogrammes de choix, case par case (formules 4 et 5).
- **Test [LU, algorithme 1].** Permutation à l'intérieur de chaque gabarit, B = 1 000 mélanges, rejet si T dépasse le quantile 95 %. Le test est conservateur (0 faux rejet sur presque toutes les cases, tableau 13). Diagnostic par gabarit : quels gabarits font rejeter.
- **Requêtes [LU, 4.5, tableau 5].** 1 000 par endpoint (20 x 50). Taux de rejet sur 630 paires de checkpoints **différents** :

| Gabarits K \ réponses N | 5 | 10 | 25 | 50 |
|---|---|---|---|---|
| 5 | 36,7 % | 66,7 % | 92,7 % | 97,5 % |
| 10 | 60,5 % | 91,9 % | 98,9 % | 99,8 % |
| 20 | 65,6 % | 94,1 % | 99,8 % | 100 % |

  Avec 100 réponses : K=10, N=10 donne 91,9 % ; K=20, N=5 donne 65,6 %. « Plus d'échantillons par gabarit est le levier le plus efficace. » Attention : ce sont des modèles différents (familles différentes), un signal bien plus fort qu'un fp8 ou une LoRA faible.
- **Cohésion des familles [LU, tableau 2].** Distance entre familles 1,8 à 3,0 fois la distance dans la famille (7 familles sur 8). Qwen3-Thinking est 14 fois plus loin de ses frères : un réglage de raisonnement change fortement le comportement. Utile pour la question « famille » (10 % du score).
- **Résultats négatifs ou limites [LU].** (a) 8 bits invisible pour tous. (b) 4 bits mixte. (c) Sous un « protocole d'agent » en prompt système : 2 faux rejets sur 6 pour AgentProv ; un long prompt de persona fait basculer Gemma-3-1B (le format d'appel change). (d) Il faut une référence fiable du modèle. (e) Un rejet n'est pas une preuve de fraude. (f) 36 paires « même modèle » seulement pour la calibration.

## 4. MET et RUT : pourquoi les fausses alertes [LU, 4.3, tableau 3]
- MET compare les réponses caractère par caractère (noyau de Hamming) ; RUT utilise le rang exact de chaque token. « Tout changement qui décale le flux de tokens, même d'un caractère, déplace les deux statistiques. »
- Expérience : 5 petits modèles, poids identiques, seul le prompt système change. MET rejette à tort 3/5 avec **une seule espace** et 4/5 avec le prompt long ; RUT passe de 1/5 à 5/5 ; AgentProv 0/5, 0/5, 1/5. Total : 7 % (AgentProv) contre 67 % (MET), 53 % (RUT), 20 % (LLMmap adapté, comparaison d'embeddings E5).
- Sur 9 endpoints OpenRouter : 5 injectent un prompt caché (détecté par le comptage de tokens, annexe A.5) ; MET les rejette tous, RUT 4 sur 5, AgentProv accepte 4 des 5 (le cinquième, +25 tokens, est rejeté par les trois).
- Pour nous : les endpoints « système prompt » et « persona » seront coupés des autres par tout test de texte brut. Les sondes devront lire un choix, pas une phrase.

## 5. Code public (noms de fichiers) [LU]
`run_probe.py` (commandes `list`, `smoke`, `run`, `test`), `config/{models,tool_sets,prompts,hidden_prompts}.yaml`, `config/probes/{s_redundant,s_init}.yaml`, `src/tests/mmd.py` (MMD delta par bloc + permutation), `src/tests/equality_report.py` (rapport à 3 niveaux : global, par probe, par gabarit), `src/harness/probe_runner.py`, `src/adapters/{hf,openai,openrouter}_adapter.py`, `src/tool_formatter.py`, `src/output_parser.py`. Pas de script de déploiement, pas de code de filigrane, de LoRA ni de quantification. Le dépôt ne contient rien de la tâche du hackathon.

## 6. Indices sur les endpoints du hackathon
- [LU, `practice_answers.csv`] Les 9 `ref-*` : famille P ; modèle S pour 5 endpoints (base, prompt système, persona, filigrane, température 1,0) ; T1 = fp8 ; T2 = LoRA force 0,3 ; T3 = LoRA 0,7 ; T4 = LoRA 1,5. Étiquettes T2/T3/T4 distinctes : même adaptateur à trois forces, ou trois adaptateurs ? Inconnu.
- [LU, énoncé] « a separate small model » pour les `ref`, OLMo seulement cité en exemple. Aucun autre indice trouvé (ni dans le papier, ni dans le dépôt, ni sur Hugging Face).
- [IDÉE] « Force » d'une LoRA = facteur multiplicateur appliqué à la correction de poids (1,0 = adaptateur normal) ; 0,3 serait un modèle très proche de la base, comme le fp8.
- [IDÉE] `ref-08` en « température 1,0 » suggère que les autres endpoints tournent à une température par défaut plus basse (le serveur ignore la nôtre, énoncé).

## Trois choses à changer dans notre monde simulé
1. **Fabriquer d'abord le cas dur : fp8 et 8 bits.** Servir le même modèle de base en fp8 (vLLM `--quantization fp8`) et en 8 bits / 4 bits (bitsandbytes, comme `load_in_8bit`/`load_in_4bit` d'AgentProv) et mesurer la plus petite différence détectable par nos sondes à 100 réponses. Le papier prédit que 8 bits est invisible ; si on le vérifie, ces paires sont à traiter comme un pari (une erreur sur 33 endpoints coûte peu, un mauvais regroupement coûte beaucoup).
2. **LoRA à forces 0,3 / 0,7 / 1,5 comme dans `ref`.** Entraîner un adaptateur, le fusionner avec un coefficient (0,3 ; 0,7 ; 1,0 ; 1,5) et tester deux variantes : même adaptateur à trois forces, et adaptateurs de données différentes. Les sondes qui échouent sur 0,3 sont celles qui réagissent seulement aux grosses dérives. Rappel du papier : une LoRA ne bouge pas toutes les sondes, seulement celles proches de ses données d'entraînement (tableau 8) : couvrir plusieurs domaines.
3. **Reprendre les prompts système littéraux et exiger 0 faux rejet.** Ajouter au monde les trois textes d'AgentProv (`" "`, « helpful assistant », Acme Corp), plus une persona et le filigrane (KGW, Unigram, Gumbel-Max), tous sur le même modèle : la règle de réussite est que nos sondes ne séparent JAMAIS ces variantes entre elles (tolérance du papier pour AgentProv : 1 faux rejet sur 15). Ajouter une variante à température 1,0 pour `ref-08`.

## Trois sondes à ajouter (coût en réponses par endpoint `ep-*`)
1. **Choix forcé sans outils, 10 gabarits x 10 réponses = 100 réponses (tout le budget).** Version texte d'AgentProv : « Réponds seulement par A, B ou C », options équivalentes, ordre mélangé à chaque réponse, histogramme des choix, test par permutation. À 100 réponses, K = 10 x N = 10 est le meilleur point du tableau 5 (91,9 % sur des modèles très différents). Variante moins chère : 5 gabarits x 10 = 50 réponses (66,7 %). Ne pas partir sur 20 x 5. [IDÉE sur [LU]]
2. **Comptage de tokens du prompt, 4 à 12 réponses (max_tokens = 1).** Si la réponse contient `usage.prompt_tokens` (à vérifier sur `ref-*`, gratuit), comparer un même texte envoyé aux endpoints : un compteur différent révèle un autre tokenizer (donc une autre famille) et un décalage constant révèle un prompt système caché (annexe A.5 : message d'un caractère, 20 tokens, température 0, 3 appels chacun). [IDÉE ; `usage` non confirmé]
3. **Appels d'outils natifs, 0 réponse pour tester, puis 100.** Envoyer sur `ref-*` une requête avec `tools=[...]` : si le serveur l'accepte, copier tel quel `s_redundant.yaml` / `s_init.yaml` (K = 10, N = 10) ; une requête invalide n'est pas comptée dans le budget [LU, énoncé] mais le plus sûr est de ne tester que sur `ref-*`. S'il refuse, retomber sur la sonde 1. [IDÉE]

Limites de cette note : le texte de la version finale EMNLP n'a pas été comparé à la v1 ; la méthode exacte de quantification n'est pas publique ; rien n'indique que les endpoints du hackathon exposent autre chose que du texte.
