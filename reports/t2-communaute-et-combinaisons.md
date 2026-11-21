# Tâche 2 "Model Equality" : ce que dit la communauté, et comment combiner (21 nov. 2026)

Décision débloquée : quelle combinaison de méthodes coder pour les 33 endpoints, et comment juger entre elles sur `ref-*`.
Marques : [VU] = lu ou mesuré aujourd'hui (lien donné) ; [IDÉE] = mon raisonnement, non mesuré. Aucun appel à l'API du hackathon, aucune API payante. M1 à M5 = méthodes de `t2-model-equality-etat-de-lart.md`.

## 0. Ce que j'ai pu lire (et ce qui m'a été fermé)
- Reddit : fermé. L'outil Chrome refuse le site ("safety restrictions"), le moteur de recherche web exclut reddit.com, un accès direct répond 403 [VU]. Contournement : l'archive Arctic Shift (partielle, capricieuse) a rendu les titres et commentaires de 3 fils utiles, listés en §1.
- X : les recherches ("model equality", "openrouter quantized", "LLM fingerprint", "SprintML", "European AI Championship", "model substitution") ont ouvert une page de résultats vide dans Chrome [VU]. Je ne sais pas si c'est faute de compte connecté ou faute de messages. Rien à rapporter.
- Hacker News : 1 seul fil, la soumission du papier MET (1 point, 0 commentaire) : [news.ycombinator.com/item?id=50239764](https://news.ycombinator.com/item?id=50239764). Aucune discussion de praticiens sur "détecter la quantification" [VU].
- GitHub (`gh`) : 10 outils d'audit trouvés, §2. Éditions passées du championnat : aucun dépôt qui traite Model Equality, §6.

## 1. Y a-t-il un consensus pour distinguer 4 bits / 8 bits / LoRA à partir du texte seul ? Non.
Image simple : tout le monde sent que "le fournisseur X est plus bête", personne n'a une règle partagée pour le prouver avec du texte.
- Les praticiens Reddit jugent à l'anecdote : un problème de logique qui échoue, des caractères non latins qui disparaissent, des fautes sur du texte médical ou italien. Fils : [qualité de Llama 3.1 hébergé, variable selon fournisseur](https://reddit.com/r/LocalLLaMA/comments/1ebzd5a/warning_the_quality_of_hosted_llama_31_may_vary/), [Groq Llama-3-70B "pas malin, sans doute trop quantifié"](https://reddit.com/r/LocalLLaMA/comments/1casosh/groq_hosted_llama370b_is_not_smart_probably/) [VU].
- Le second fil se termine par "c'était la température" : à T=0 la bonne réponse revient, à T=0,4 non, sur le même fournisseur. Un commentaire rappelle qu'un test à 1 question et 5 essais ne prouve rien [VU]. Leçon directe pour nous : la température est un déguisement qui imite une quantification.
- Quand ils veulent des chiffres, ils comparent en LOCAL les probabilités de chaque mot (divergence KL = écart entre deux distributions) de la version quantifiée contre la pleine précision : [classement de quantifications Gemma 4 par KL](https://reddit.com/r/LocalLLaMA/comments/1seua77/), [KL du cache KV q8/q4](https://reddit.com/r/LocalLLaMA/comments/1suh3sz/), [test KL de 2023](https://reddit.com/r/LocalLLaMA/comments/1816h1x/). Cela exige les probabilités (logprobs) : pas du texte seul [VU]. Leur folklore : jusqu'à 5 bits tout va bien, la dégradation commence en 4 bits ; le 8 bits est "très bas" [VU, commentaire du dernier fil].
- Les guides de fournisseurs listent des tests (répétition à T=0, JSON strict, aiguille dans une botte de foin, arithmétique en plusieurs étapes, test différentiel entre deux fournisseurs, latence) et disent eux-mêmes qu'aucun n'est concluant : [Lyceum](https://lyceum.technology/magazine/is-your-inference-provider-quantizing-the-model-how/) [VU].
- La recherche dit pareil : les tests texte seul sont insuffisants contre une quantification subtile ([Cai et al., 2504.04715](https://arxiv.org/abs/2504.04715), dépôt [sunblaze-ucb/llm-api-audit](https://github.com/sunblaze-ucb/llm-api-audit)). Seule la comparaison avec le vrai modèle local, avec ses probabilités, détecte vite : [DiFR](https://arxiv.org/html/2511.20621v1) voit le 4 bits en environ 300 tokens (AUC > 0,999), le fp8 de cache en 1000 à 5000 tokens, avec poids locaux et graine partagée [VU].
- Même diagnostic côté "marché" : à même étiquette fp8, des fournisseurs OpenRouter servent des capacités très différentes ([LessWrong](https://www.lesswrong.com/posts/KsyoSAyBRXtwzSugg/not-pinning-your-openrouter-provider-might-invalidate-your)) ; OpenRouter admet que l'étiquette de quantification est déclarative ([Exacto](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/)) [VU].
- Seul cas chiffré "texte seul" trouvé : un auteur applique MET à deux points d'accès FP4 de GLM et rejette l'égalité des distributions, mais pas pour un autre modèle ; effet secondaire : sorties 6 à 8 % plus longues ([Substack](https://collectgarbage.substack.com/p/quantized-endpoints-charge-less-per)) [VU, petit échantillon].
Conclusion : aucun consensus. Le meilleur accord pratique est "MET avec beaucoup d'échantillons, ou comparer à un modèle local". Nos 100 réponses sont loin des 250 de MET.

## 2. Outils trouvés sur GitHub (tous testent un fournisseur contre une référence connue, jamais 33 inconnus entre eux)
| Outil | Ce qu'il lit | Utile pour nous |
|---|---|---|
| [LLMmap](https://github.com/pasquini-dario/LLMmap) (arXiv [2407.15847](https://arxiv.org/abs/2407.15847)) | 8 questions fixes, classifieur pré-entraîné sur 52 modèles ; annonce plus de 95 % sur 42 versions, robuste aux prompts système et à l'échantillonnage [VU] | Famille seulement, si nos modèles sont dans ses 52 [IDÉE]. Ne sépare pas des variantes d'un même checkpoint [IDÉE] |
| [modelprint](https://github.com/unclecode/modelprint) | 15 sondes d'infrastructure : nombre de tokens de textes fixes (empreinte du tokenizer), erreurs, en-têtes HTTP, plafond de contexte, "géométrie des logprobs" [VU] | Empreinte de famille quasi gratuite si `usage` est renvoyé. Le README dit : le comportement plie sous un prompt système, pas le tokenizer [VU] |
| [B3IT](https://github.com/timothee-chauvin/token-efficient-change-detection-llm-apis) + [billet](https://tchauvin.com/change-detection-llm-apis) | Prompts d'un token où T=0 hésite ; 3 requêtes par entrée au départ, ~100 tirages pour la référence [VU] | Idée de sondes "au bord". Le billet ne chiffre pas la sensibilité à LoRA ni à la quantification [VU] |
| [One Token Is Enough](https://github.com/ToseaAI/llm-fingerprint-detector) | Réponse d'un token, divergence de Jensen-Shannon, bibliothèque TypeScript [VU] | Briques de M1 |
| [llm-verify](https://github.com/mintesnot-teshome/llm-verify), [api-relay-audit](https://github.com/toby-bridges/api-relay-audit) | 32 prompts d'identité, injections, flux SSE ; pensés contre les revendeurs frauduleux [VU] | Peu : un prompt "qui es-tu" ne marche pas ici (Cai : moins de 5 %) |
| [locuslab/llm-idiosyncrasies](https://github.com/locuslab/llm-idiosyncrasies) | Classifieur de style sur embeddings, sert de base à Cai [VU] | M5 (famille). À 4 bits Cai trouve ~50 % [VU, prior report] |
| [K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) | Rejoue des prompts d'appel d'outil chez chaque fournisseur, compare `finish_reason` et charge utile [VU] | Cousin d'AgentProv. Seulement si `tools` passe sur les endpoints [IDÉE] |
| [MET](https://github.com/i-gao/model-equality-testing) | MMD (distance entre deux paquets de textes) + permutation | M2 |

## 3. Chaque méthode et chaque astuce, en image
Légende des colonnes "voit" : 4b = 4 bits ; 8b = 8 bits / fp8 ; L- = LoRA faible (0,3) ; L+ = LoRA forte (0,7 à 1,5). "Voit" = signal attendu à 100 réponses. Tout cet étage est [IDÉE] bâti sur des chiffres [VU] des articles.
| # | Méthode | Image | Ce qu'elle lit | Voit |
|---|---|---|---|---|
| M1 | Questions fermées par lots, chi-deux + Cauchy | 20 mini-questions dans un seul message, on compare les profils de réponses | Quel choix fait le modèle quand il hésite (la réponse la plus fréquente par question) | 4b oui ; L+ oui ; 8b non ; L- à peine |
| M2 | MET, textes longs | Deux paquets de textes : sont-ils tirés de la même urne ? | Positions de tokens identiques sur 50 tokens | 4b faible à 100 rép. ; 8b non ; très sensible aux déguisements |
| M3 | Probabilité sous un modèle local | Le vrai élève relit la copie et note la surprise | Surprise moyenne par token | 4b oui ; le reste mal ; il faut les poids exacts |
| M4 | Simulateur local | Répéter l'examen à la maison avec des variantes fabriquées | Sert à régler seuils et prompts, pas à lire les endpoints | Tout, mais indirectement |
| M5 | Embeddings | Reconnaître l'écriture d'un auteur | Style moyen du texte | Famille seulement |
| C1 | Nombre de tokens (usage) | Peser le même texte sur deux balances de marques différentes | `prompt_tokens`/`completion_tokens` d'un texte fixe | Famille ; aucune variante (même tokenizer) |
| C2 | "Exact-match" de sorties longues | Deux photocopies identiques : même original | Sorties octet pour octet identiques à T=0 ; préfixe commun | 8b et 4b peut-être, par le point de divergence |
| C3 | Sondes d'un token "au bord" (B3IT) | Des pièces presque en équilibre : un souffle les fait tomber | Distribution du 1er token | L+ ; 4b ; 8b faible |
| C4 | Longueur et style de sortie | Compter les mots du bavard | Longueur moyenne, "wait/but/alternatively" | L+ oui ; L- et quantification : bruit |
| C5 | Chronométrage | Écouter le moteur de la voiture | Temps par token (fp8 plus rapide ? LoRA non fusionné plus lent ?) | 8b et L peut-être, si la charge est stable |
| C6 | En-têtes, forme des erreurs, ID de réponse | Lire les plaques d'immatriculation | Pile de serveur (même déploiement ou non) | Regroupe des endpoints partageant un serveur ; aucune sémantique |
| C7 | Anecdotes Reddit (énigmes, JSON strict, aiguille, arithmétique) | Faire passer un oral | Réussite sur tâches difficiles | 4b grossier ; confondu par la température |
| C8 | Logprobs / KL local | Mesurer l'écart de pente d'une route | Probabilités des mots | Tout, si l'API les renvoie (à vérifier sur ref) |

### Qui se fait piéger par quel déguisement
Symboles : OK = robuste ; ~ = dégradé ; X = trompé. [IDÉE], sauf les points où un chiffre [VU] existe (cités plus bas).
| Signal | Prompt système | Filigrane | Température | Glouton / faisceau |
|---|---|---|---|---|
| M1 mode des réponses | ~ (la consigne finale pèse, mais un persona la déplace) | ~ (peut forcer un choix sur les questions à égalité) | OK (le mode survit) | OK |
| M1 fréquences/chi-deux | ~ | ~ | X | X (un endpoint déterministe n'a pas de fréquences) |
| M2 MET Hamming | X (67 % de faux positifs [VU, AgentProv]) | X (puissance 0,43 à 1,0 [VU, MET]) | ~ | X |
| M3 / API-TAIL | X (signalé 79 à 99 % [VU]) | ~ | ~ si température alignée | ~ |
| M5 embeddings | ~ (persona = X) | OK | OK | OK |
| C1 tokens | OK (on retire la ligne de base d'un caractère, comme modelprint [VU]) | OK | OK | OK |
| C2 exact-match | X (prompt système change tout) | X | X | OK entre endpoints déterministes |
| C3 sondes un token | X (déplace les égalités) | X | X à T>0 ; OK à T=0 | OK |
| C4 longueur | ~ | ~ | ~ | ~ (beam : sorties plus longues et répétitives [IDÉE]) |
| C5 chrono | OK | OK | OK | ~ (beam coûte plus) |
| C6 en-têtes | OK | OK | OK | OK |
| C8 logprobs | ~ | X (change la distribution servie) | X (distribution réelle changée) | OK |
Chiffres [VU] : prompt système = pire faux positif des tests texte (67 % MET, 53 % RUT, 79-99 % API-TAIL) ; filigrane aussi visible que la quantification pour MET ; "Forty Questions" perd 6 à 9 points avec prompt système (prior report §2).
Lecture : aucun signal sémantique ne résiste aux quatre déguisements. Seuls C1, C5, C6 les ignorent, mais ils ne lisent pas la différence qui compte (quantification, LoRA). D'où la nécessité de combiner.

## 4. Pas de consensus, mais deux faits à exploiter
1. Asymétrie de l'exact-match [IDÉE, à tester sur ref] : deux endpoints qui rendent la même sortie de 256 tokens à l'octet près (deux endpoints déterministes) sont presque sûrement le même modèle. Mais une différence ne prouve rien (filigrane, prompt système, GPU). L'expérience de Thinking Machines : même modèle, T=0, 1000 essais = 80 sorties distinctes, avec divergence au token 103 [VU : [billet](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)]. Donc "même modèle" donne un long préfixe commun, pas toujours une copie. Mesure à prendre : longueur du préfixe commun, comparée entre paires. Un modèle quantifié devrait diverger plus tôt [IDÉE].
2. Le fp8 et le LoRA faible (0,3) sont les paires à enjeu. Sur `ref`, les fusionner avec le modèle de base coûte (mesuré par `score_practice` logique, calcul fait ci-dessous) : fp8 fusionné avec S = 0,734 ; fp8 et LoRA 0,3 fusionnés = 0,524 ; fp8, 0,3 et 0,7 fusionnés = 0,331. À l'inverse, séparer un seul déguisement (filigrane, prompt système ou T=1,0) hors de S = 0,721. Une coupure fausse et une fusion fausse coûtent donc pareil : on ne fusionne ni on ne coupe "par défaut", on coupe si la probabilité de "différent" dépasse 50 % [VU, calcul].

## 5. Trois pipelines combinés (budget : 100 réponses par endpoint, 1 réponse = 1 requête)
Principe commun : étage 1 gratuit (aucune requête) → étage 2 famille → étage 3 variante dans la famille → étage 4 réserve sur les paires ambiguës → clustering final.
Étage 1 (gratuit, 0 requête) : champs `usage` et en-têtes lus sur les réponses déjà reçues (C1, C6) ; chronométrage (C5) pris sur chaque requête.

### Pipeline A : sûr et rapide (4 à 5 h)
| Bloc | Req. | Contenu |
|---|---|---|
| Lots fermés (M1) | 60 | 6 lots x 10 ordres, mêmes ordres pour tous |
| Ouvert | 20 | Même prompt pour tous, 200 tokens : embeddings (M5) + longueur (C4) |
| Déterminisme | 8 | 4 prompts x 2 : sépare "déterministe" de "échantillonné" |
| Réserve | 12 | Paires ambiguës |
Famille : embeddings + tokens (C1) + 1 à 2 sauts de distance. Variante : accord sur le mode des réponses (M1), seuil réglé sur ref. Cible [IDÉE] : séparer 4b et L+ ; fusionner 8b et L-.

### Pipeline B : équilibré (8 à 10 h), mon choix par défaut
| Bloc | Req. | Contenu |
|---|---|---|
| Lots fermés (M1) | 44 | 6 lots x 7 ordres + chi-deux/Cauchy |
| Exact-match (C2) | 12 | 6 prompts ouverts identiques pour tous, 256 tokens, 2 passes : taux d'octets identiques et préfixe commun |
| Sondes d'un token (C3) | 12 | 12 prompts "au bord" trouvés sur ref (gratuit) puis figés |
| Ouvert + identité | 12 | Embeddings (M5), longueur (C4) |
| Réserve | 20 | Réservée aux paires dont la probabilité est entre 0,3 et 0,7 |
Fusion par paire : un modèle logistique (régression qui transforme des indices en probabilité) ajusté sur ref avec 3 à 4 variables seulement (accord de mode M1, préfixe commun C2, distance d'embeddings, écart de longueur) ; les autres signaux agissent comme règles dures. Raison : 10 paires "même" sur ref ne portent pas plus de 3 à 4 coefficients [IDÉE].

### Pipeline C : ambitieux (14 h et plus)
A ou B, plus M3/M4 : poids locaux de la famille identifiée (OLMo, l'énoncé le cite), simulateur 4b/8b/fp8/LoRA avec filigrane, prompt système, T=1, glouton, faisceau. Servira à choisir les prompts les plus sensibles et à mesurer la vraie puissance à 100 réponses. Seulement si la famille est connue avant 15 h [IDÉE]. Si `logprobs` ou `tools` passent sur ref (test gratuit), activer C8/K2VV ici : c'est le seul chemin vers le 8 bits.

### Fusion finale (tous pipelines)
1. Probabilité "même modèle" par paire (logistique). 2. Clustering par corrélation (on cherche la partition qui contredit le moins les probabilités ; 33 nœuds, résolution exacte rapide). 3. Stabilité : rééchantillonner les questions 200 fois, garder la co-appartenance, relancer la réserve sur les paires instables. 4. Chaque famille a au moins 2 endpoints ou on le justifie.

## 6. Éditions passées : ce qu'on peut en retenir
- Rien sur Model Equality dans les dépôts publics trouvés [VU]. Éditions repérées : Stockholm (févr. 2026, tâche d'image/"model fingerprinting génératif", [ChristofferNo/CISPA-HACKATHON-2026](https://github.com/ChristofferNo/CISPA-HACKATHON-2026)), Varsovie (mai 2026, [DrFlach](https://github.com/DrFlach/CISPA_WITH_MyTeam_Outcasts), 8e), Vienne (déc. 2025, vainqueurs [florianzainzinger1](https://github.com/florianzainzinger1/cispa-hackathon-championship-vienna)), grande finale (juil. 2026, [4e](https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026)).
- Leçon de la finale : une règle "ship gate" fixée à l'avance (une amélioration n'est gardée que si elle tient sur plusieurs découpes de validation, pour ne pas sur-ajuster le tableau public de 30 %). À copier [VU].
- L'équipe `sinonchum/euro-ai-championship-paris` a publié ce matin une note [(task2-zh.md)](https://github.com/sinonchum/euro-ai-championship-paris/blob/HEAD/docs/tasks/task2-zh.md) : elle avertit que MET brut (Hamming) traite filigrane et quantification comme "modèle changé", donc sur-sépare. Aucune méthode concurrente chiffrée [VU].

## 7. Comment juger "le meilleur" sans se fier au score public
Texte de l'énoncé : "DO NOT RELY ON THE PUBLIC SCORE" (trois fois) ; le tableau public ne couvre qu'une partie des paires [VU, `Model_Equality_Testing.txt` l.95-197, slide 388].
Protocole :
1. Collecte large sur les 9 ref (si le budget ref est bien illimité : lire le champ `budget` avant) : tous les signaux des pipelines, y compris des plus gros que le budget réel.
2. Bootstrap au budget réel : pour chaque pipeline, tirer 200 fois exactement ses requêtes (60+20+8+12 pour A, etc.) parmi la collecte. Mêmes tirages pour tous les pipelines (comparaison appariée). Pour chaque tirage : partition, puis `python score_practice.py` ; garder médiane et 10e percentile du MCC.
3. Leave-one-disguise-out : fixer poids et seuil sans ref-03 (prompt système), tester dessus ; idem sans ref-06 (filigrane), ref-08 (T=1,0), ref-05 (persona). Rapporter le MCC et les paires fausses des 4 passes.
4. Critères avant de toucher un `ep` : les 5 S ensemble dans 90 % des tirages ; T3 et T4 séparés de S dans 80 % ; pas de fusion T3-T4 (coût mesuré : 0,935). fp8 et L- : noter sans exiger.
5. Règle de choix : la médiane compte, mais on préfère le pipeline dont le 10e percentile est le plus haut (évite le coup de chance).
6. Geler (questions, poids, seuil) avant d'ouvrir les `ep`. Le score public ne sert qu'à contrôler le format et à détecter un effondrement.
Ce que les 36 paires de ref disent [VU, calcul] : 10 paires "même" (les 5 S) et 26 "différent" (20 S/T + 6 T/T). Un seul S mal séparé fait passer le MCC à 0,72 ; un fp8 fusionné à 0,73. Donc un seul endpoint se joue à 0,27 de MCC.
Ce qu'elles ne disent pas : (a) 9 endpoints seulement, les 36 paires ne sont pas indépendantes (un endpoint faux fausse 8 paires) ; (b) une seule famille : le seuil famille/non-famille n'est pas testable, ni la part de M5 et C1 ; (c) petit modèle : les questions ne se transfèrent pas, seuls les types de questions et les seuils relatifs le font ; (d) les 5 S ne couvrent pas le faisceau, absent des déguisements ref ; (e) on ignore la répartition fp8 / LoRA / quantification parmi les 33 endpoints, donc le gain réel dépend d'un mélange inconnu.
Parade partielle : simuler sur ref plusieurs mélanges possibles (par exemple 30 % de paires "même" fines contre 10 %) et regarder si le classement des pipelines change.

## 8. Recommandation
- Coder B, en livrant A d'abord (A est un sous-ensemble de B : une soumission valide très tôt).
- Tester gratuitement sur ref, avant tout : budget, `usage`, en-têtes, `logprobs`, `tools`, déterminisme de chaque ref, chrono par requête. Un seul de ces tests ouvert (`logprobs` ou `tools`) change le plan : passer au pipeline C.
- Ne rien fusionner ni couper "par défaut" pour fp8 et LoRA 0,3 : décider paire par paire à 50 %.
- Questions pour vous (défaut en gras si pas de réponse) : 1. Le temps restant permet-il B (10 h) ? a. **oui, B**, b. non, A seul. 2. M3/M4 : a. **seulement si logprobs sur ref**, b. jamais.
