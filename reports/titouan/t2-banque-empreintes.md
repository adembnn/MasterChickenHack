# T2 : banque d'empreintes textuelles par famille de modèle

Aucune requête sur l'API du hackathon (ni `ref-*` ni `ep-*`) : tout est local, sur la VM h200
partagée (`~/eac-t2`), un modèle téléchargé puis chargé à la fois. Script : `tracks/t2-model-equality/twin/fingerprint_bank.py`,
sortie brute `tracks/t2-model-equality/twin/results_fingerprint_bank.json`.

Ce rapport est **complémentaire** à `reports/titouan/t2-fingerprinting.md` (qui suppose qu'on peut charger un
candidat et calculer sa vraisemblance sur le texte d'un endpoint) et à `reports/titouan/t2-family-detection.md`
(NLL croisée, nécessite aussi les poids du candidat). Ici, le but est différent : des règles de lecture **sur
le texte seul**, utilisables même sans jamais charger de modèle candidat — un premier filtre à l'œil ou par
script simple sur une poignée de réponses d'un endpoint inconnu, avant d'investir dans la méthode lourde.

## 0. Modèles couverts et blocage HF

| Famille | repo_id | Statut |
|---|---|---|
| Qwen2.5 | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` | ✅ fait |
| SmolLM2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | ✅ fait |
| OLMo-2 | `allenai/OLMo-2-0425-1B-Instruct` | ✅ fait |
| Pythia | `EleutherAI/pythia-1.4b` (base, pas instruct) | ✅ fait |
| StableLM 2 | `stabilityai/stablelm-2-1_6b-chat` | ✅ fait |
| Phi-3.5-mini | `microsoft/Phi-3.5-mini-instruct` | ✅ fait |
| Ministral | `mistralai/Ministral-8B-Instruct-2410` | ✅ fait |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ fait |
| Llama-3.2 | `meta-llama/Llama-3.2-1B-Instruct` | ❌ **gated**, 401 (pas de `HF_TOKEN` sur la VM) |
| Gemma-2 | `google/gemma-2-2b-it` | ❌ **gated**, 401 (idem) |

**Priorité Llama/Gemma non traitée : blocage d'accès, pas un choix.** Meta est partenaire du hackathon
(keynote watermarking) et le papier des superviseurs (AgentProv) utilise Qwen2.5/Qwen3 dans ses propres
expériences (`reports/titouan/t2-identite-modele.md`) — les deux indices restent donc en tension, et on ne
peut trancher empiriquement sur Llama tant que personne n'a fait `huggingface-cli login` avec un compte ayant
accepté les licences Meta/Google sur `ssh h200`. Pas de contournement tenté (pas de création de compte ni de
token sans l'accord explicite de Titouan). Pour ces deux familles, la section 4 documente seulement ce qui est
public (cartes de modèle, templates de chat connus), explicitement marqué `[non mesuré ici]`.

## 1. Protocole

20 prompts, 5 catégories, génération gloutonne (`do_sample=False`), `max_new_tokens=160`, template de chat
natif de chaque tokenizer :
- **self_id** (2 prompts) : « Quel modèle es-tu, qui t'a créé ? »
- **cutoff** (2 prompts) : date de coupure des connaissances.
- **refusal** (3 prompts, pièges d'alignement bénins mais classiques : email de phishing, crochetage de serrure
  sur sa propre porte, faux certificat médical) : taux et formulation du refus.
- **open_ended** (5 prompts : explication, arithmétique, code, conseils, capitale) : style par défaut
  (ouverture, clôture, listes à puces/numérotées, gras, titres markdown, longueur).
- **tokenization** (3 prompts) : épeler un mot très rare lettre par lettre, écrire un grand nombre en chiffres,
  répéter verbatim une chaîne avec ponctuation/casse mixte (`xk7Qz_42-foo.bar()`).

## 2. Tableau comparatif : famille → signatures textuelles

| Famille | Auto-identification (verbatim) | Coupure citée | Refus (taux sur 3 pièges) | Style par défaut | Écho verbatim exact de `xk7Qz_42-foo.bar()` |
|---|---|---|---|---|---|
| **Qwen2.5-1.5B** | « I am Qwen, an AI language model developed by Alibaba Cloud. » | « I don't have a specific knowledge cutoff date » (puis évoque 2021/2025 selon le prompt) | **0,67** — refuse 2/3 pièges, avec justification mesurée | pas d'ouverture figée, listes à puces parfois, 386 car. en moyenne | Non (reformule) |
| **Qwen2.5-7B** | « I am Qwen, created by Alibaba Cloud. » (plus court que 1.5B) | « December 2022 » (date précise, contraste avec la 1.5B) | **0,33** — refuse 1/3 | ouvre souvent par un mot d'accroche (« Sure »/« Of course »), 362 car. | Non |
| **SmolLM2-1.7B** | « I am a SmolLM, a small language model trained by Hugging Face » | « last update in October 2021... cutoff September 2021 » (contradiction interne 2 dates dans la même phrase) | 0,0 — **écrit l'email de phishing sans hésiter** | aucune ouverture figée, 317 car. (le plus court) | Non |
| **OLMo-2-1B** | « I am a model trained by researchers and engineers at Ai2 » | « up to date until September 2021 » | **1,0** — refuse les 3 pièges, formule « I cannot fulfill your request. I'm just an AI... » | pas d'ouverture, pas de puces, 354 car. | **Oui** (seul avec Ministral) |
| **Pythia-1.4B** (base, pas instruct) | *aucune* — pas de notion de self, continue le texte en boucle (« What is the most important thing you learned in your life? » répété 10×) | mentionne une année sans réponse cohérente | 0,0 (ne « refuse » pas, il continue juste le texte du prompt) | dégénère en boucles de répétition exacte sous glouton, 519 car. | Non |
| **StableLM2-1.6B-chat** | « I am a custom-built AI model created by a programming language and specialized hardware » (réponse évasive/incohérente, ne cite aucune entreprise) | explique ce qu'est un cutoff sans en donner un | 0,0 — écrit l'email de phishing | ouverture parfois, 507 car. | Non (répond par une phrase méta, ne recopie pas) |
| **Phi-3.5-mini** | « I am Phi, a Microsoft-developed language model. » | « I don't have a knowledge cutoff date » puis dérive méta | 0,0 mais **moralise sans refuser net** (« I must clarify that it is unethical and illegal... » puis répond en partie) | gras fréquent, pas de puces, 342 car. | Non (ajoute une note méta après l'écho) |
| **Ministral-8B** | « I am a model created by Mistral AI, a leading AI company based in Paris, France. » | « October 2023 » (date précise) | 0,0 | ouverture fréquente, puces/titres markdown (seul des deux Mistral à en mettre), 346 car. | **Oui** |
| **Mistral-7B-v0.3** | « I am a model of the chat-based model category, specifically **a descendant of the T5 model**, created by the Mistral AI team. » — **hallucination d'architecture caractéristique** (T5 est faux, Mistral est un transformeur decoder-only) | « 2021 » (vague, pas de mois) | 0,0 | pas de puces/titres, 374 car. | Non (reformule) |
| Llama-3.2 | *non mesuré ici (gated)* — connu publiquement : refuse souvent par « I can't help with that », auto-identification « I'm Llama, an AI assistant created by Meta » [non mesuré, carte de modèle] | — | — | — | — |
| Gemma-2 | *non mesuré ici (gated)* — connu publiquement : template `<start_of_turn>user/model<end_of_turn>`, auto-identification « I am Gemma, an open-weights AI model trained by Google DeepMind » [non mesuré, carte de modèle] | — | — | — | — |

Sorties brutes complètes (les 20 réponses par modèle) dans `results_fingerprint_bank.json`.

## 3. Paires confusables — où l'auto-identification seule ne suffit pas

Consigne appliquée : quand deux familles se ressemblent, creuser plutôt que passer à la suivante.

### 3.1 Mistral-7B-v0.3 vs Ministral-8B-Instruct-2410 — la paire la plus dure
Les deux répondent « créé par Mistral AI » à l'auto-identification : **l'auto-ID seule ne distingue pas le
modèle précis dans une famille, seulement la famille** (ce qui est cohérent avec `t2-family-detection.md` :
l'auto-identification est un signal de *famille*, pas de *modèle*). Discriminants trouvés en creusant :
1. **Écho verbatim** : Ministral recopie `xk7Qz_42-foo.bar()` caractère pour caractère ; Mistral-7B-v0.3 le
   reformule toujours (3/3 tentatives). Signal robuste dans nos données mais sur un seul type de prompt — à
   confirmer sur plus d'échantillons avant de s'y fier seul.
2. **Date de coupure** : Ministral donne un mois précis (« October 2023 ») ; Mistral-7B-v0.3 reste vague
   (« 2021 », sans mois) — cohérent avec les générations réelles (Ministral est sorti en 2024, entraîné sur des
   données plus récentes).
3. **Mise en forme** : Ministral utilise puces et titres markdown sur les prompts ouverts (`bullet_rate` 0,4,
   `header_rate` 0,2) ; Mistral-7B-v0.3 n'en met jamais (0,0 partout) — Ministral a probablement un post-entraînement
   plus récent orienté formatage structuré.
4. **Hallucination d'identité** : seul Mistral-7B-v0.3 invente une ascendance « T5 » — une erreur factuelle
   stable et bizarre, donc une signature utile si elle se reproduit (à vérifier avec une seconde graine/un
   deuxième run avant de la considérer fiable, une hallucination gloutonne peut dépendre du prompt exact).

### 3.2 Les familles « auto-ID générique » (StableLM, Pythia-base) vs tout le reste
StableLM2-chat ne cite aucune entreprise ni aucun nom de modèle à l'auto-identification — contrairement à
Qwen/OLMo/Phi/Mistral/Ministral/SmolLM2 qui se nomment tous explicitement. Un endpoint qui refuse de se
nommer réduit le pool aux familles sans self-ID franche ; ici StableLM2 est la seule rencontrée. Pythia (base)
se distingue trivialement par la dégénérescence en boucle sous glouton (signature de modèle *non aligné
instruct*, pas d'une famille en particulier) — à ne pas confondre avec un vrai refus.

### 3.3 Qwen2.5-1.5B vs Qwen2.5-7B — même famille, tailles différentes
Le taux de refus chute nettement avec la taille (0,67 → 0,33) et la date de cutoff passe de « je n'en ai pas »
(1.5B, évasif) à « December 2022 » (7B, précis) : à l'intérieur d'une même famille, la taille module la
confiance/précision des réponses méta, un signal secondaire utile pour distinguer *quelle taille* de Qwen on a
en face, une fois la famille acquise par self-ID.

### 3.4 Limite générale de cette section
Toutes ces règles viennent d'**une seule génération gloutonne par prompt** : aucune n'a été testée sous
perturbation (system prompt, température, LoRA, quantification) qui est précisément ce que l'énoncé T2 fait
varier. Un system prompt imposant un style, ou un filigrane, peut casser l'écho verbatim ou le taux de refus
sans changer la famille (cf. `t2-watermark-robustness.md`, section sur les biais de décodage). À utiliser comme
indice de triage rapide, jamais comme preuve unique.

## 4. Lien avec le reste de la méthode T2

- Pour la **famille** (10 % du score) : auto-identification + style par défaut donnent un filtre rapide ;
  confirmer ensuite par prompts ouverts/NLL croisée (`t2-family-detection.md`) si deux candidats restent proches.
- Pour le **modèle précis** (90 % du score) à l'intérieur d'une famille identifiée : l'auto-ID ne suffit jamais
  (section 3.1) ; il faut la comparaison gloutonne exacte + seuil (`t2-mcc-simulation.md`, `t2-method-v3.md`)
  ou la vraisemblance token par token si on a les poids candidats.
- Si un jour une vraie réponse `ref-*`/`ep-*` est disponible : comparer son auto-identification et sa date de
  cutoff au tableau de la section 2 **avant** de lancer un calcul lourd — un filtre gratuit en une lecture.

## 5. Expérience de divergence (fp8/int8/nf4/LoRA) sur d'autres familles que OLMo

`exp1_divergence.py` (branche `t2/family-detection`/`t2/plan-deploiement`, résultats dans
`reports/titouan/t2-mcc-simulation.md`) mesurait les taux de divergence gloutonne pour OLMo-2-1B uniquement
(int8 ≈ 50 %, fp8 ≈ 78-93 %, nf4 ≈ 98 %, LoRA 0,3 ≈ 82 %, seuil optimal retenu τ ≈ 0,42-0,43). Question posée
par Titouan : ces chiffres sont-ils spécifiques à OLMo ou se généralisent-ils ?

Script généralisé : `tracks/t2-model-equality/twin/exp1_divergence_multi.py` — même protocole (plancher de
bruit à plusieurs tailles de lot et dtypes, fake-quant fp8 e4m3/e5m2, fake-quant int8, nf4 réel via
bitsandbytes, LoRA entraînée (r=8, q_proj/v_proj ou qkv_proj fusionné selon l'architecture) puis mise à
l'échelle 0,3/0,7/1,0/1,5), paramétré par `--model`, lancé sur Qwen2.5-1.5B, SmolLM2-1.7B, Mistral-7B,
Ministral-8B et Phi-3.5-mini (réglages allégés : n=100 prompts, 56 tokens, pour tenir sur une VM partagée).

Résultats bruts : `results_divergence_multi.json` (et un `*_outputs.json` par modèle avec les textes complets).
Job terminé (`TOUT FINI`) sur les 5 familles. Toutes les valeurs sont des **taux de divergence**
(1 − fraction de textes identiques à la base, même convention que `t2-mcc-simulation.md`/OLMo) pour comparer
directement aux chiffres OLMo rappelés en tête de section (bruit 0,30-0,34 ; int8 0,50 ; fp8 0,78-0,93 ;
nf4 0,98 ; LoRA 0,3 0,82 ; LoRA ≥0,7 ≥0,98).

| Modèle | Plancher de bruit (min-max sur bs∈{1,4,16}+dtype fp16) | fp8 e4m3 | fp8 e5m2 | int8 | nf4 | LoRA 0,3 | LoRA 0,7 | LoRA 1,0 | LoRA 1,5 |
|---|---|---|---|---|---|---|---|---|---|
| OLMo-2-1B (référence, `t2-mcc-simulation.md`) | 0,30-0,34 | — | — | 0,50 | 0,98 | 0,82 | ≥0,98 | ≥0,98 | ≥0,98 |
| Qwen2.5-1.5B-Instruct | 0,11-0,42 | 0,77 | 0,80 | 0,61 | 1,00 | 0,84 | 0,97 | 1,00 | 1,00 |
| SmolLM2-1.7B-Instruct | 0,13-0,30 | 0,59 | 0,76 | 0,47 | 0,95 | 0,85 | 1,00 | 1,00 | 1,00 |
| Mistral-7B-Instruct-v0.3 | 0,25-0,29 | 0,67 | 0,73 | 0,48 | 0,98 | 0,77 | 1,00 | 1,00 | 1,00 |
| **Ministral-8B-Instruct-2410** | **0,07-0,16** | **0,20** | 0,53 | **0,15** | 0,68 | 0,57 | 1,00 | 1,00 | 1,00 |
| **Phi-3.5-mini-instruct** | **0,14-0,58** | 0,77 | 0,85 | 0,56 | 1,00 | 0,64 | 1,00 | 1,00 | 1,00 |

### Lecture

**1. nf4 et LoRA ≥ 0,7 se généralisent très bien** : partout ≥ 0,68 (nf4) et ≥ 1,00 (LoRA ≥ 0,7), comme sur
OLMo. Ces deux variantes restent détectables quelle que soit la famille — pas de recalibrage nécessaire.

**2. Le seuil OLMo τ ≈ 0,42-0,43 et « seul int8 est fragile » NE SE GÉNÉRALISENT PAS : résultat majeur.**
- **Ministral-8B casse l'hypothèse « int8 proche du seuil mais détectable »** : int8 (0,15) et **fp8 e4m3 (0,20)**
  tombent *en dessous ou au niveau du plancher de bruit (0,07-0,16)* — avec ce prompt-set et ce τ, Ministral-8B
  en int8 ou en fp8-e4m3 serait classé **« même modèle »** alors qu'il est quantifié : faux négatif quasi garanti,
  pas seulement une perte de puissance comme sur OLMo (où int8 restait à 0,50, loin au-dessus du plancher 0,30-0,34).
  nf4 (0,68) est aussi anormalement bas pour Ministral comparé aux autres familles (0,95-1,00) — cette famille
  semble globalement plus tolérante/robuste à la quantification poids-seule sur ce protocole, à confirmer avec
  plus de prompts avant de généraliser plus loin.
- **Phi-3.5-mini casse l'hypothèse « le plancher de bruit est un petit intervalle stable »** : le plancher varie
  de **0,14 à 0,58** selon la seule taille de lot (bs=1 vs bs=16) — *au-dessus* du τ=0,42-0,43 calibré sur OLMo.
  Avec ce seuil, **deux copies du même Phi-3.5-mini servies à des tailles de lot différentes seraient classées
  « modèles différents »** (faux positif), et int8 (0,56) ne se distingue plus franchement du pire cas de bruit
  (0,58). Hypothèse à vérifier : Phi-3.5 (architecture avec `qkv_proj`/`gate_up_proj` fusionnés, RoPE à plage
  étendue) est peut-être plus sensible au padding/à l'ordre de batching sous HF `generate` que les architectures
  Llama-like ; si confirmé sur le vrai serving des organisateurs, le plancher de bruit devra être mesuré par
  famille (étape 0 du plan, déjà prévue dans `t2-method-v3.md`) et pas supposé universel.
- Qwen2.5-1.5B, SmolLM2 et Mistral-7B, en revanche, reproduisent assez bien le patron OLMo : plancher ≤ 0,30-0,42,
  int8 et fp8 clairement au-dessus (0,47-0,80), nf4 et LoRA forte proches de 1,0. Sur ces trois familles, le
  seuil τ ≈ 0,42-0,43 sépare encore raisonnablement int8/fp8 du bruit (marge plus courte pour Qwen : bruit max
  0,42 est déjà au seuil).

**3. Conséquence pour la méthode T2** : ne jamais figer un seuil unique avant d'avoir mesuré, sur les `ref-*`
réels, (a) le plancher de bruit à au moins deux tailles de lot et (b) si possible une comparaison fp8/int8
contre la base — l'écart entre OLMo et Ministral/Phi-3.5 ici (facteur 2-3 sur la marge signal/bruit) montre que
la calibration dépend de la famille réelle derrière les endpoints, pas seulement de la variante organisateur.
Si la vraie famille s'avère être une « famille Ministral-like » (fp8/int8 presque invisibles, bruit faible),
`t2-mcc-simulation.md` sous-estimerait fortement le risque de faux négatif sur les variantes quantifiées légères.

**Limites** : n=100 prompts (contre 200 pour OLMo), `max_new_tokens`=56 (contre 64), LoRA entraînée 80 pas
(contre 120), une seule graine de LoRA par modèle — réglages allégés pour tenir sur une VM partagée en
~30 min/modèle (LoRA incluse). Les écarts qualitatifs ci-dessus (facteur 2-3, pas quelques points) sont
probablement robustes à ces réglages plus légers, mais les valeurs exactes devraient être resserrées avec
n=200+ avant de les figer dans un seuil de soumission.
