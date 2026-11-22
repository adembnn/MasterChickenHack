# T2 : l'hypothèse "OLMo" est-elle fondée ? Et comment trancher à moindre coût

Rôle de ce rapport : relecture critique d'une hypothèse précise que Titouan a remise en cause, puis méthode
d'identification. Aucune requête API n'a été faite pendant ce travail (ni `ep-*`, ni `ref-*`, pas même un `GET
/models` gratuit). Marquage : [source] = lu directement dans le texte cité, [web] = lu via recherche web sur les
papiers de référence, [avis] = raisonnement propre.

## 0. Verdict en une phrase

**L'hypothèse "le modèle derrière les 33 endpoints est OLMo" n'est PAS fondée par l'énoncé.** Elle vient d'une
lecture trop rapide d'une phrase d'exemple pédagogique. Pire : la preuve indirecte la plus pertinente qu'on peut
trouver — le propre papier de recherche des organisateurs de cette tâche (AgentProv, mêmes auteurs que les
superviseurs) — **n'utilise jamais OLMo** dans ses expériences, et mise plutôt sur Qwen2.5/Qwen3, Llama-3.x,
Mistral et Gemma-3. `reports/titouan/t2-fingerprinting.md` §3.1 avait déjà écrit cette hypothèse comme un fait
("Hypothèse : les familles sont OLMo") sans la marquer comme non vérifiée : c'est corrigé ici.

## 1. Relecture de l'énoncé officiel — ce qui est écrit, pas ce qu'on a supposé

Texte exact de `reports/sources/Model_Equality_Testing.txt` (lignes 42-45) [source] :

> "For example, two deployments of the same OLMo base model with the same quantization and adapter settings are
> identical, even if one uses a decoding watermark and a different system prompt. A quantized variant of that
> checkpoint, or a version with a different LoRA adapter, is a different model."

C'est une **phrase d'illustration d'une règle générale** ("same base model + différences de watermark/system
prompt/décodage = même modèle ; différences de quantif/LoRA = modèle différent"). OLMo y joue exactement le rôle
que jouerait "deux Toyota" dans "deux Toyota avec la même motorisation sont la même voiture, même si l'une a un
autocollant différent" — un exemple concret pour fixer une règle abstraite, pas une affirmation sur l'identité
réelle du parc de véhicules du problème. Le mot "OLMo" apparaît **une seule fois** dans tout le document source
(grep sur `Model_Equality_Testing.txt`, `slides.txt`, et tous les fichiers de `reports/sources/` : aucune autre
occurrence, et aucune des trois ne cite Llama/Qwen/Mistral/Gemma/Pythia/StableLM non plus).

Sur les endpoints de référence, le texte dit seulement (ligne 117) [source] : **"They serve a separate small
model"** — aucun nom, aucune famille. "Small" est le seul qualificatif donné.

**Conclusion de la relecture : l'énoncé ne confirme ni n'exclut aucune famille.** Toute affirmation du type "le
modèle est OLMo" dans nos rapports précédents est une hypothèse de travail, pas un fait de l'énoncé — à reformuler
partout où c'est écrit comme acquis (au moins `t2-fingerprinting.md` §3.1, à corriger dans une v4 de la méthode).

## 2. Ce que disent les 3 papiers de référence sur leurs propres expériences

Les papiers ne disent jamais "le hackathon utilise le modèle X" — ce n'est pas leur rôle. Mais le choix des
modèles que les auteurs utilisent dans **leurs propres papiers** est le meilleur indice indirect dont on dispose :
c'est le zoo de modèles que ce laboratoire a sous la main, a l'habitude d'auditer, et a probablement réutilisé
pour fabriquer un jeu de 33+9 endpoints "avec les mêmes genres de déguisements et pièges" (texte de l'énoncé pour
les `ref-*`).

### 2.1 MET — Gao, Liang, Guestrin, ICLR 2025, arXiv 2410.20247 [web, lecture partielle du HTML v2, ~40k/178k caractères vérifiés mot à mot, le reste résumé]

- Expériences principales (quantification, watermark, fine-tuning) : **Mistral 7B Instruct, Llama-3 8B/70B
  Instruct, Llama-3.1 8B/70B Instruct** [citation vérifiée mot à mot]. C'est le cœur du papier.
- Audit des APIs commerciales réelles (§5, celui qui ressemble le plus à notre tâche) : **4 modèles Llama**
  (Llama-3/3.1, 8B/70B/405B) chez 9 fournisseurs [citation vérifiée pour l'intro, détail de taille via résumé non
  vérifié mot à mot].
- Expérience "distance entre 13 modèles" (celle qui ressemble le plus à "retrouver les familles de 33
  endpoints") : le texte vérifié mot à mot cite "huit modèles open-weight ([Phi-3 : Abdin et al.], [OLMo :
  Groeneveld et al.], [Gemma : Team et al.])" et un tableau de puissance où **OLMo 7B est une des alternatives
  testées contre Llama-3.1 8B comme référence**. La liste complète des 13 (résumé, non vérifiée mot à mot) :
  Mistral 7B, Llama-3 8B, Llama-3.1 8B, Llama-3 70B, Llama-3.1 70B, Phi-3 mini, Gemma-2 9B, **OLMo 7B**, et 5
  modèles OpenAI fermés.
- **Donc OLMo apparaît bien dans MET, mais comme UNE option parmi huit modèles open-weight, pas comme le modèle
  central de l'étude.** Le modèle central de MET est Llama (3/3.1, toutes tailles).

### 2.2 RUT — Zhu et al., ICLR 2026, arXiv 2506.06975 [web, lecture partielle du HTML v5, ~40k/94k caractères vérifiés mot à mot]

- Expériences de détection de quantification (le cœur du papier, le plus proche de notre tâche "quantif + LoRA +
  watermark") : **Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9B-it**, comparés chacun à leurs
  variantes 4-bit et 8-bit [citation vérifiée mot à mot, §5.2].
- Jailbreak/system-prompt caché, fine-tuning, substitution complète, APIs réelles : même trio **Llama-3.x /
  Mistral / Gemma-2** décliné en plusieurs tailles [résumé, non vérifié mot à mot pour ces sections].
- **OLMo n'apparaît dans aucune partie lue de RUT** (ni vérifiée mot à mot, ni dans le résumé de la partie non
  lue). C'est une absence, pas une confirmation absolue — 54k caractères sur 94k n'ont été vus qu'en résumé — mais
  c'est un deuxième papier de référence qui, autant qu'on peut le vérifier, ne touche jamais à OLMo.

### 2.3 AgentProv — Wang, Zhao, Backes, Boenisch, Dziedzic, EMNLP 2026, arXiv 2609.00052 [web, lecture mot à mot de l'essentiel du HTML]

**C'est le papier le plus important des trois pour notre question, et il a été sous-exploité jusqu'ici.** Ses
auteurs ne sont pas des tiers : **Franziska Boenisch et Adam Dziedzic sont les deux superviseurs de la tâche T2**
(`slides.txt` ligne 3-5), et **Xun Wang et Bihe Zhao sont les deux premiers "Developed by" de la tâche T2**
(`slides.txt` ligne 5). AgentProv est donc, au sens propre, **le papier de recherche de l'équipe qui a construit
nos 33 endpoints** — publié dans la même fenêtre temporelle (numéro arXiv très proche de la date du hackathon).
Si un laboratoire a un zoo de modèles qu'il réutilise d'une expérience à l'autre, c'est celui-là.

Modèles ouverts utilisés dans AgentProv, cités mot à mot dans le texte lu :
- Pool d'expériences : "13 familles (0,5B-14B paramètres) : 36 checkpoints de base... 5 checkpoints locaux testés
  sous prompt système injecté... 9 déploiements d'API tiers" [citation vérifiée mot à mot, chiffres de taille lus
  avec prudence car dupliqués dans l'extraction, §4.1].
- Les 5 checkpoints locaux du test "prompt système caché" (Table 3, vérifiée mot à mot) : **Llama-3.2-1B-Instruct,
  Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Gemma-3-1B-it, Hammer2.1-1.5B**.
- Familles citées en Table 2 (cohésion de famille, vérifiée mot à mot) : **R1-Distill, DeepSeek, Qwen3,
  Hammer-2.1, Qwen2.5, Llama, Mistral, Hermes, Qwen3-Think** (un modèle "pense à voix haute" distinct de Qwen3 — à
  retenir, §3.4 plus bas).
- Endpoints OpenRouter testés (Table 4, vérifiée mot à mot, chiffres à confirmer car l'extraction double les
  chiffres) : Llama-3.2-3B, Gemma-3n-E4B, Llama-3.2-1B, Qwen3-14B, Llama-3-8B, Llama-3.1-8B, Qwen3-8B, Gemma-3-4B,
  Mistral-Nemo-Instruct-2407.
- **"OLMo" n'apparaît NULLE PART dans tout le texte lu mot à mot** (résumé, §1 à §4.7, limitations, éthique,
  bibliographie jusqu'à "Yang et al.") — ni comme modèle expérimental, ni même cité en related work. Absence
  presque certaine (seule la queue de la bibliographie et l'annexe A n'ont été vues qu'en résumé).

**Ce que ça change pour nous** : le papier de recherche publié par les gens qui ont littéralement construit la
tâche T2 du hackathon utilise un zoo de **petits modèles instruct (0,5B-14B)**, très majoritairement **Qwen2.5 /
Qwen3** et **Llama-3.x**, avec Mistral, Gemma-3, DeepSeek-R1-Distill et Hermes-3 en complément — et jamais OLMo.
C'est un indice bien plus fort que la phrase d'exemple du PDF, parce que c'est un **comportement mesuré** (ce que
ce labo audite réellement), pas une phrase pédagogique isolée.

## 3. Hypothèses sur l'identité du modèle, classées par probabilité

| Rang | Famille | Pourquoi | Preuve |
|---|---|---|---|
| 1 | **Qwen2.5 / Qwen3** (petite taille, 0,5B-7B) | Famille la plus utilisée par AgentProv (mêmes auteurs que la tâche), y compris en local sur petits checkpoints — cohérent avec "a separate small model" pour les `ref-*` | [web] AgentProv Table 2, 3, 4 |
| 2 | **Llama-3.x** (1B-8B) | Présent dans LES TROIS papiers, cœur de MET et RUT, présent aussi dans AgentProv ; famille la plus auditée dans toute la littérature citée par l'énoncé | [web] MET §4.1/§5, RUT §5.2, AgentProv Table 2/3/4 |
| 3 | **Mistral-7B-Instruct** | Présent dans MET et RUT comme modèle central, et dans AgentProv (Hermes base + Mistral-Nemo) | [web] MET §4.1, RUT §5.2, AgentProv |
| 4 | **Gemma-2 ou Gemma-3** (petite taille) | Présent dans RUT (Gemma-2-9B, cœur de l'étude quantif) et AgentProv (Gemma-3-1B/4B/3n) | [web] RUT §5.2, AgentProv Table 3/4 |
| 5 | **OLMo** (1B/7B, famille citée dans l'énoncé) | Hypothèse de départ de l'équipe ; apparaît dans MET (1 des 8 open-weight), mais **absente de RUT** (dans la part lue) et **absente d'AgentProv** (le papier des organisateurs eux-mêmes) ; sa seule base dans l'énoncé est une phrase d'exemple pédagogique, pas une affirmation | [source] énoncé l.42-45 ; [web] MET, absence RUT/AgentProv |
| 6 | **DeepSeek-R1-Distill / Hermes-3 / Hammer2.1** | Présents dans AgentProv comme familles de fine-tune/distillation testées, mais plus exotiques comme "modèle de base" d'une tâche d'audit généraliste | [web] AgentProv Table 2 |
| 7 | **Pythia / StableLM** | Classiques de la recherche en interprétabilité/audit (poids et données d'entraînement ouverts), mais **aucune occurrence** dans l'énoncé ni dans les 3 papiers de référence lus | [avis] — aucune preuve trouvée, cité pour mémoire |
| 8 | **Modèle inconnu / custom / propriétaire** | SprintML peut avoir construit son propre jeu sans reprendre un zoo de papier ; toujours possible, à ne jamais exclure | [avis] |

**Point important : les rangs 1 à 4 sont tous "plus probables qu'OLMo" sur la preuve disponible, mais aucun n'est
confirmé.** Le bon niveau de confiance est "on a une liste de 4-5 familles plausibles, pas une certitude sur
aucune", et notre méthode doit être conçue pour **distinguer entre plusieurs candidats en parallèle**, pas pour
confirmer un seul candidat a priori (ce que faisait `t2-fingerprinting.md` §3.1 en se focalisant sur le zoo OLMo).

## 4. Autre indice trouvé dans les sources du dépôt : la structure des `ref-*`

`reports/sources/t2/practice_answers.csv` [source] donne déjà la vérité terrain des 9 référence : 1 modèle de
base `S` (ref-01, 03, 05, 06, 08 = system prompt / persona / watermark / température, toutes la même famille/
modèle que la base), 1 quantifié fp8 (`T1`, ref-02), et 3 forces de LoRA (`T2` 0,3 / `T3` 0,7 / `T4` 1,5 — ref-09/
04/07). C'est cohérent avec "a separate small model" : **un seul petit modèle de base**, pas un zoo de plusieurs
familles pour les `ref-*`. Ça ne dit toujours pas le nom de la famille, mais ça confirme que l'identification de
famille sur les `ref-*` est un problème à un seul candidat (pas un problème de clustering), ce qui simplifie
beaucoup l'étape 0 de la méthode v3 existante.

## 5. Méthode d'identification à très faible coût (texte seul, sans dépenser le budget `ep-*`)

Objectif : avant de dépenser quoi que ce soit, maximiser la confiance sur la famille via des sondes qui (a) sont
peu chères en réponses, (b) résistent aux déguisements prévus par l'énoncé (watermark, prompt système, décodage),
et (c) ne dépendent pas d'un seul candidat pressenti.

### 5.1 Sondes robustes aux déguisements, par ordre de coût/signal

1. **Auto-identification directe** ("Qui es-tu ? Qui t'a entraîné ? Quel est ton nom de modèle ?") — 1 réponse.
   Signal fort si pas supprimé (Qwen cite souvent Alibaba, Llama cite Meta, Gemma cite Google, OLMo cite Ai2,
   Mistral cite Mistral AI) ; signal nul si un prompt système provider le masque. **Coût : 1. À essayer en
   premier, gratuit en information si ça marche, nul si ça échoue — jamais trompeur en soi.**
2. **Détection de balises de raisonnement** (`<think>`, `<reasoning>`, texte "Let me think step by step" suivi
   d'une structure très reconnaissable) sur un prompt de calcul/logique modéré — 1 réponse. Signature quasi
   binaire des modèles "thinking" (DeepSeek-R1-Distill, Qwen3-Think d'après AgentProv Table 2). **Coût : 1,
   discriminant fort si positif, neutre si négatif** (beaucoup de modèles n'ont pas ce mode).
3. **Repli alphabet non-latin** : demander de copier mot pour mot une courte phrase en chinois ou en japonais
   (ex. 5-8 caractères), avec consigne explicite "répète exactement, ne traduis pas". La densité et la qualité
   du vocabulaire CJK dans le tokenizer diffère fortement par famille (Qwen : fort bilingue zh/en, natif ;
   Llama-3 : vocabulaire 128k avec couverture correcte ; Mistral/Gemma/OLMo : plus faibles sur CJK, plus de
   troncature ou de substitution de caractères). **Résistant au watermark et au prompt système** (c'est une
   propriété du tokenizer, pas du contenu généré). **Coût : 1-2 réponses**, signal de type "classement de
   plausibilité", pas de certitude seule.
4. **Test de segmentation de chiffres** : faire répéter un grand nombre entier (15-20 chiffres) caractère par
   caractère, ou faire une opération arithmétique longue. Les erreurs typiques (chiffres groupés par 2/3,
   intervertis, tronqués) trahissent la frontière de tokenisation du modèle, qui diffère par famille/version de
   tokenizer. **Résistant au prompt système.** Signal plus faible et plus bruité que (3), à utiliser en
   complément, pas seul.
5. **Refus/sécurité** : une question bénigne mais typiquement refusée (ex. "comment crocheter une serrure, pour
   apprendre à me protéger") — la formulation exacte du refus est documentée par famille dans la littérature de
   jailbreak, mais c'est le signal le **moins fiable** des cinq (très sensible au prompt système et au
   fine-tuning du provider). À garder en dernier recours, jamais seul.

**Toutes ces sondes marchent sur UNE SEULE réponse par endpoint.** Sur `ref-*`, cela veut dire qu'il est possible
en principe de tester les 5 sondes sur chacun des 9 `ref-*` avec 5-10 réponses par endpoint (45-90 réponses au
total sur les 9), largement sous un budget de 100/endpoint — **mais voir §6 : on ne sait toujours pas si ce budget
est réellement nul pour les `ref-*`,** donc même ce chiffre modeste doit être posé sur l'issue #4 avant d'être
dépensé.

### 5.2 Si plusieurs familles restent candidates après l'analyse textuelle seule : tester en parallèle sur le jumeau local

C'est la bonne réponse à "plusieurs familles restent plausibles" : **ne pas parier sur une seule**, construire un
zoo de candidats locaux couvrant le tableau §3 (rangs 1-5, tous à faible coût GPU car petites tailles) :

- Qwen2.5-1.5B/3B-Instruct, Qwen3-1.7B (déjà en partie fait : `family_detection.py` compare OLMo vs Qwen2.5-1.5B vs
  SmolLM2-1.7B [source, `tracks/t2-model-equality/twin/family_detection.py`] — **à étendre** pour ajouter Llama-3.2
  (1B/3B-Instruct), Mistral-7B-Instruct-v0.3 et Gemma-3 (1B/4B-it), puisque ce sont maintenant les candidats les
  mieux soutenus par la preuve du §2-3, pas seulement OLMo/Qwen/SmolLM2).
- Appliquer les 5 sondes du §5.1 à CHAQUE candidat local → construire une table de signatures attendues par
  famille (auto-ID oui/non par défaut, présence de balises thinking, qualité CJK, pattern d'erreur sur les
  chiffres).
- **Dès qu'on a l'autorisation de dépenser un budget réel sur `ref-*` (voir §6)**, comparer les réponses réelles
  des 9 `ref-*` à cette table de signatures : 5-10 réponses suffisent si la table est déjà construite et fine, au
  lieu d'improviser des prompts un par un sur le vrai budget.
- Cette approche a déjà commencé (`family_detection.py` teste "notre méthode sépare-t-elle trivialement deux
  familles différentes" avec NLL croisée et collisions de prompts fermés [source]) ; elle manque seulement
  l'élargissement du zoo de candidats et l'ajout des sondes de tokenizer/thinking du §5.1, qui sont des signatures
  d'identité directe, pas seulement des mesures de distance.

## 6. Ce qu'on NE PEUT PAS savoir sans dépenser du budget réel — honnêteté sur la limite

Même avec tout ce travail :
- **Aucune des sondes du §5 n'est une preuve mathématique**, seulement une preuve de plausibilité qui augmente la
  confiance. Un provider peut parfaitement masquer l'auto-identification (prompt système), et le test CJK/chiffres
  ne distingue pas deux modèles très proches en taille et en données (ex. deux variantes Qwen).
- **On ne connaît toujours pas l'identité réelle tant qu'on n'a pas lu au moins quelques réponses réelles de
  `ref-01` à `ref-09`.** Ce rapport réduit l'incertitude a priori (de "OLMo supposé acquis" à "4-5 candidats
  classés par preuve indirecte"), il ne la supprime pas. La seule façon de trancher est de lire du texte réel
  produit par les `ref-*` (ou, en dernier recours, par les `ep-*`) et de le comparer à la table de signatures du
  §5.2 — ou, mieux encore, de laisser le classifieur de vraisemblance déjà prévu en §3.4/3.5 de
  `t2-fingerprinting.md` élargir son zoo de candidats locaux au-delà d'OLMo.
- **Le budget des `ref-*` est une question non résolue, pas une supposition de notre part** : le docstring de
  `query_reference.py` dit "no query limit" [source], mais Titouan a mesuré un vrai compteur (`limit=100` affiché
  par `GET /models`, passé à `used=1/100` après une seule requête sur `ref-01`, commentaire du 21/11 sur l'issue
  #4). Un message Discord de l'organisatrice Maitri Shah ("it is removed") suggère que la limite a depuis été
  retirée côté serveur, **mais ce n'est pas encore vérifié dans le fil de l'issue #4** (la dernière demande de
  vérification par un `GET /models`, gratuit d'après l'énoncé, n'a pas de réponse postée à l'heure de ce rapport).
  **Ce rapport n'a fait aucune requête, ni `GET /models` ni génération, pour ne pas préempter cette vérification.**
- **Si la réponse finale devait être "on ne peut pas confirmer la famille sans dépenser du budget réel"**, ce
  serait la vérité honnête à ce stade, et ce rapport le dit explicitement : la relecture critique (§1-§4) élimine
  une fausse certitude (OLMo acquis), la méthode (§5) réduit le coût d'obtenir une vraie réponse, mais ne la
  remplace pas.

## 7. Recommandations concrètes, par ordre de priorité

1. **Corriger `t2-fingerprinting.md` §3.1** : remplacer "Hypothèse : les familles sont OLMo (l'énoncé cite 'OLMo
   base model' dans l'exemple)" par un renvoi vers ce rapport, et ne plus présenter OLMo comme hypothèse par
   défaut dans la méthode v3/v4.
2. **Poster sur l'issue #4** : (a) ce constat (OLMo = exemple pédagogique, pas une affirmation), (b) la table de
   candidats du §3, (c) une demande explicite de vérifier gratuitement le budget `ref-*` par `GET /models` avant
   toute dépense, (d) une proposition de budget précis (ex. 5-10 réponses x 9 ref = 45-90 réponses) pour les
   sondes du §5.1, **à valider avant exécution** — Titouan reste seul décisionnaire de la dépense réelle d'après la
   règle d'équipe (`CLAUDE.md` : "Budget T2 irréversible... Aucune requête sur `ep-*` sans un plan posté dans
   l'issue T2").
3. **Étendre `family_detection.py`** (ou un script frère) pour inclure Llama-3.2, Mistral-7B-Instruct-v0.3,
   Gemma-3 dans le zoo local, et ajouter les sondes du §5.1 (auto-ID, balise thinking, CJK, segmentation de
   chiffres) comme nouvelles mesures sur chaque candidat — tout en local, 0 requête API, avant toute décision sur
   `ref-*`.

## Sources

- `reports/sources/Model_Equality_Testing.txt` (texte officiel de la tâche)
- `reports/sources/slides.txt` (identité des superviseurs/développeurs de la tâche T2)
- `reports/sources/t2/practice_answers.csv`, `reports/sources/t2/query_reference.py`
- `reports/titouan/t2-fingerprinting.md` (hypothèse OLMo à corriger)
- `tracks/t2-model-equality/twin/family_detection.py` (zoo local existant, à étendre)
- Issue adembnn/EAC#4 (fil complet relu, budget `ref-*` et annonce Discord de Maitri Shah)
- MET : Gao, Liang, Guestrin, "Model Equality Testing: Which Model Is This API Serving?", ICLR 2025,
  https://arxiv.org/abs/2410.20247 — lu via https://arxiv.org/html/2410.20247v2
- RUT : Zhu et al., "Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test", ICLR 2026,
  https://arxiv.org/abs/2506.06975 — lu via https://arxiv.org/html/2506.06975v5
- AgentProv : Wang, Zhao, Backes, Boenisch, Dziedzic, "AgentProv: Auditing Agentic LLM API Providers via Tool-use
  Policy Probes", EMNLP 2026, https://arxiv.org/abs/2609.00052 — lu via https://arxiv.org/html/2609.00052v1
