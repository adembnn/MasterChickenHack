# Tâche 2 "Model Equality" : état de l'art, 2e passe (21 nov. 2026, 16h30)

Décision débloquée : quelle méthode on implémente, avec quel découpage des 100 réponses par endpoint, et comment on la valide sur les `ref-*`.
Aucun appel à l'API du hackathon, aucune API payante. Recherche faite par arXiv (API par date), Semantic Scholar (citations de MET, RUT, AgentProv), web, ACL Anthology.
Marques : [lu] = chiffre pris dans le texte de l'article (via un lecteur de page qui résume jusqu'à 100 000 caractères, pas relu à la main) ; [résumé] = abstract seulement ; [avis] = mon raisonnement, non mesuré ; [à tester] = se vérifie gratuitement sur les ref.
Non trouvé : rien sur OpenReview ICLR 2027 (aucune soumission pertinente ressortie), pas de dépôt public des auteurs de la tâche. Papier "prefix tree" (ICASSP 2026) non lu.

## 1. Ce qu'on nous demande (5 lignes)
1. 33 endpoints texte seul (`ep-01..33`), 100 réponses max chacun pour toute la compétition, 256 tokens en sortie, 1024 en entrée, pas de message système, température et graine ignorées.
2. Même modèle = même checkpoint + même quantification + même LoRA. Filigrane, prompt système du fournisseur et stratégie de décodage sont des déguisements : ils changent le texte, pas l'identité.
3. Note = 0,1 x MCC(même famille ?) + 0,9 x MCC(même modèle ? dans la famille). Chaque paire fausse coûte, dans un sens comme dans l'autre.
4. Terrain d'essai : 9 endpoints `ref-*` d'un petit modèle, une seule famille P. Vérité : 5 "S" identiques (plain, prompt système, persona, filigrane, T=1.0), puis fp8 (T1), LoRA 0,3 (T2), LoRA 0,7 (T3), LoRA 1,5 (T4).
5. Fichiers de départ lus : la réponse API contient `choices[0].message.content` et `budget{used,limit}` ; `/models` donne le budget par endpoint ; la seule contrainte vérifiée est le plafond de 100 réponses par `ep-*`.
   À vérifier gratuitement : les `ref-*` sont dits "sans limite" dans l'énoncé, mais `query_reference.py` recopie la phrase "chaque réponse coûte une requête du budget". Lire le champ `budget` d'un ref avant de compter dessus.

Image simple : 33 cuisiniers, on goûte 100 bouchées de chacun. Certains ont la même recette avec un tablier différent (filigrane), un chef qui souffle des consignes (prompt système), une cuisson plus ou moins vive (décodage). D'autres ont une recette à peine retouchée (quantification, LoRA).

## 2. État de l'art récent (tout ce qui suit sert nos 100 réponses ; "réf." = il faut un modèle de référence fiable)
Mots à connaître : quantification = poids du modèle arrondis sur moins de bits (8 bits, 4 bits, fp8), le modèle devient un peu "flou". LoRA = petit module ajouté au modèle pour le spécialiser ; "force 1,5" = module amplifié. Logprob = probabilité que le modèle donne à chaque mot (pas fournie ici). MMD = distance entre deux paquets de textes ; test de permutation = on mélange les étiquettes pour voir si la distance observée est exceptionnelle.

| Papier (id) | Date, lieu | Idée | Échantillons | Code |
|---|---|---|---|---|
| MET, Gao et al. (2410.20247) | ICLR 2025 | MMD avec noyau "positions de tokens identiques" sur des complétions longues, p-valeur par permutation ou simulation. [lu] | 25 prompts x 10 = 250 réponses de 50 tokens. Puissance : nf4 0,12 à 1,0 selon le modèle ; int8 0,06 à 1,0 (0,08 sur Llama-3.1 8B, 0,30 Mistral, 0,36 OLMo) ; filigrane KGW 0,43 à 1,0 | github.com/i-gao/model-equality-testing |
| RUT, Zhu et al. (2506.06975, v5 avr. 2026) | ICLR 2026 | Classe 1 réponse de l'API parmi 100 réponses du vrai modèle local, test d'uniformité des rangs. Exige le modèle local et ses logits. [lu] | 100 prompts x 1 réponse. 4 bits détecté, 8 bits : puissance presque nulle sur Gemma et Mistral | github.com/xyzhu123/RUT |
| AgentProv, Wang, Zhao, Backes, Boenisch, Dziedzic (2609.00052) | EMNLP 2026 | Lit l'outil choisi par le modèle (appel d'outil), pas le texte. MMD à noyau "égalité" + permutation. Les auteurs de la tâche. [lu] | 20 gabarits x 50 = 1000 requêtes ; 10 gabarits x 50 = 500 requêtes donnent 99,8 %. Faux positifs avec prompt système injecté : 7 % (MET 67 %, RUT 53 %). 8 bits accepté comme identique ; 4 bits : verdict mixte | "sera publié", pas de lien |
| Cai et al. (2504.04715) | 2025 | Compare tous les tests texte seul. Conclusion : insuffisants contre une quantification subtile ; classifieurs BERT à 50 % (hasard), le prompt "qui es-tu" échoue (<5 %). [lu] | 25 prompts x 10 pour MMD | n/a |
| B3IT "Border Inputs", Chauvin et al. (2602.11083) | 2026 | Prompts "à égalité" où deux tokens se disputent la 1re place : un changement minuscule des poids les fait basculer. Compare les ensembles de réponses possibles. Texte seul. [lu] | 5 prompts, 50 réf. + 3 requêtes de test ; détecte 1 seul pas de fine-tuning (AUC 0,87) ; basse température | github.com/timothee-chauvin/token-efficient-change-detection-llm-apis |
| DFA "relay audit" (2610.37086) | 2026 | 5 questions à choix fermé, chi-deux par question puis combinaison de Cauchy (une seule question très parlante suffit). Texte seul. [lu] | 5 x 50 = 250 réponses ; non testé sur quantif/LoRA | non fourni |
| IRIS (2607.20860) | 2026 | Demander un tirage au hasard (pile/face, chiffre, 5 bits) : les biais de tirage trahissent le backend. 179 variables par réponse. [lu] | AUROC 0,993 avec m=1 ; 14 paires "même modèle chez 2 fournisseurs" sur 15 jugées différentes (vraie quantif/noyaux). Signal maximal vers T=1, nul en décodage glouton | github.com/Photen/IRIS-audit |
| FLIPS (2606.03330) | 2026 | Séquence de 500 symboles "au hasard" sur 2 symboles ; tests NIST + XGBoost. Détecte température, prompt système, int4, fp8 : donc il VOIT nos déguisements. [lu] | 8 requêtes ; 96 % fermé, 90 % ouvert ; 99 % à T=0,4 contre 87 % à T=1,0 | github.com/GurvanR/FLIPS-LLM-Instance-Fingerprinting |
| One Token Is Enough (2607.10252) | 2026 | Distribution de la réponse d'un seul token à 40 mini-tâches ; divergence de Jensen-Shannon (écart entre deux répartitions). Texte seul. [lu] | 16 cases x 15 = 240 requêtes ; erreur 9,5 % ; quantif/LoRA non testés | Zenodo 21278793 |
| Forty Questions (2610.28281) | 2026 | Réponses en un mot, profil de fréquences par question, Bayes naïf (on multiplie les probabilités). [lu] | 40 réponses : 92,5 % parmi 94 modèles. Prompt système : -6 à -9 pts ; persona : -9 à -16 pts | github.com/tap2k/modelun |
| KBF (2605.29524) | 2026 | Questions chiffrées au bord de ce que le modèle sait, test binomial. Conçu invariant : 0/9 quantifications signalées, température 0 à 0,7 sans effet. [lu] | 109 à 681 sondes, envoyées par lots de 10 | github.com/Ooo0ption/KBF |
| API-TAIL (2610.36801) | 2026 | Surprise de chaque réponse sous le vrai modèle local, température alignée. Exige les poids locaux. Un prompt système injecté est signalé dans 79 à 99 % des audits : il le voit. [lu] | 200 réponses ; 0,33 à 1,0 de puissance à 10 % de substitution | non fourni |
| AdaptPrint (2608.22213) | 2026 | 3 stratégies de sondes (directe, suite, relance), similarité d'embeddings et n-grammes. [lu] | 2 requêtes par sonde ; LoRA : 8/12 en top-1 ; prompt système 81 % | non fourni |
| SetPrint (2611.12950) | 2026 | Réponses à des prompts perturbés, embeddings all-mpnet, produit scalaire des changements. [lu] | 200 réponses par modèle ; AUROC 0,78 | non fourni |
| Anosognosia (2610.06174) | 2026 | Le texte d'un modèle quantifié 4 bits est reconnaissable seulement par comparaison avec un texte 4 bits de référence (85 %). [lu] | seul : 40 % (hasard 33 %) | github.com/Omusubi0123/llm-anosognosia |
| RESF (aclanthology 2025.emnlp-main.247) | EMNLP 2025 | Empreintes optimisées pour être sensibles à LoRA, int4, int8 ; test en deux étages (support puis distribution) pour ne pas confondre avec la température. [résumé] : 98,8 % avec 5 empreintes. Exige l'accès local. | 5 empreintes | non vérifié |
| Stemma (2607.25880), BReF (2609.06330) | 2026 | Régions de décision / réponses A-B-C-D sous perturbations. BReF exige les 4 probabilités (non fournies) ; 4 bits 6/6. [résumé]/[lu] | grosses batteries | non fournis |

Ce que la littérature établit [lu] :
- Aucun article ne traite NOTRE problème exact : regrouper 33 endpoints sans référence, avec déguisements, en 100 réponses chacun, en séparant quantification et LoRA. Tous supposent un modèle de référence fiable et comparent un suspect à lui.
- 4 bits : détectable. 8 bits / fp8 : presque indétectable en texte seul (MET, RUT, AgentProv, KBF par construction). Il faut s'attendre à ne pas séparer ref-02 (fp8) de S.
- Les trois déguisements ne sont pas égaux. Le prompt système est le pire faux positif des tests texte (MET 67 %, RUT 53 %, API-TAIL 79-99 %). Le filigrane est aussi visible que la quantification pour MET (0,43 à 1,0). La température est le plus doux, sauf pour les tests "tirage au hasard" (IRIS, FLIPS) qui en dépendent.
- La littérature ne dit rien sur le décodage par faisceau (beam search = le modèle explore plusieurs suites et garde la plus probable, donc déterministe et répétitif). Zone vide : à mesurer nous-mêmes.

## 3. Méthodes classées de la meilleure à la plus simple

### Le levier que personne n'a utilisé tel quel : le budget compte des RÉPONSES, pas des tokens
Une réponse de 256 tokens coûte une requête, comme une réponse d'un token. On peut donc mettre 20 mini-questions dans UN prompt ("une réponse par ligne, 1 à 3 mots") : une requête donne 20 observations. Preuve partielle [lu] : KBF envoie ses sondes par lots de 10, FLIPS tire 500 symboles par réponse. 100 requêtes deviennent environ 1500 à 2000 réponses courtes par endpoint, contre 250 pour MET. [à tester] : les petits modèles tiennent-ils le format, et les réponses d'un même lot s'influencent-elles ? Mélanger l'ordre des questions à chaque répétition, identique pour tous les endpoints (comparaison appariée).

### M1 (recommandée) : empreinte par lots + poids appris sur ref + clustering cohérent
Prouvé (briques) : tirage au hasard (IRIS, FLIPS), réponses d'un mot (One Token, Forty Questions), questions chiffrées au bord du savoir (KBF), prompts à égalité (B3IT), chi-deux + Cauchy (DFA), permutation (MET, AgentProv).
Notre idée (assemblage, non publié) : tout est marqué [avis] ci-dessous.
1. Banque de questions (1 h) : 120 questions en 4 types. (a) 40 choix à deux options "près de l'égalité" (faits chiffrés ou comparaisons où un petit modèle hésite). (b) 30 questions chiffrées KBF (nombres rares). (c) 25 tirages au hasard (chiffre 1-10, lettre, mot). (d) 25 questions d'arithmétique à 3 chiffres. Chaque question demande un format fermé, dans le message utilisateur (le prompt système du fournisseur passe avant, donc notre consigne finale pèse plus).
2. Lots (30 min) : 6 lots de 20 questions. Chaque lot est interrogé 10 fois avec un ordre différent = 60 requêtes. 6 prompts x 2 répétitions de ces 60 servent aussi de test de déterminisme (voir section 4).
3. Statistique par paire (2 h) : pour chaque question, tableau des réponses (endpoint i contre j), chi-deux par question, puis combinaison de Cauchy (DFA) ; p-valeur par permutation des répétitions. Complément robuste à la température : comparer la réponse la plus fréquente par question (le classement des réponses est conservé par la température et le top-p, pas par un prompt système ni un filigrane) [avis].
4. Poids appris sur les ref (1 h) : sur les 9 ref, mesurer pour chaque type de question le rapport "écart entre S et T3/T4" sur "écart entre les 5 S (déguisements)". Garder les types à rapport élevé. Peu de données (36 paires) : donner des poids par type, pas par question [avis].
5. Clustering cohérent (1 h) : voir section 4.
Coût : 60 requêtes de base + 12 prompts longs (bloc C) + réserve. Personnes-heures : 8 à 10. Risque moyen : le format en lot dégrade un petit modèle ; l'effet des 8 bits/fp8 restera invisible.

### M2 : MET classique (Hamming) sur textes longs, en second avis
Étapes : 12 prompts ouverts x 5 répétitions, 128 tokens, comparer les 50 premiers tokens après découpage par un tokenizer local, permutation. 3 h. Budget : 60 requêtes, soit la zone où MET est faible (250 réponses nécessaires pour 77 %). Très sensible au prompt système et au filigrane [lu : 67 % de faux positifs]. À utiliser uniquement comme confirmation dans une famille déjà découpée, jamais seul.

### M3 : vraisemblance sous le modèle local (RUT / API-TAIL adaptés), le plus technique
Étapes : (1) identifier la famille d'un groupe d'endpoints (M1 blocs c + usage de tokens, section 5) ; (2) télécharger les poids candidats (l'énoncé cite OLMo ; HF allenai) ; (3) pour chaque réponse d'un endpoint, calculer la surprise moyenne par token sous le modèle local en bf16 ; (4) standardiser, aligner la température (API-TAIL) ; (5) comparer les profils entre endpoints du même groupe. 1 nœud A100 suffit. Réutilise les réponses du bloc C sans requête de plus. 10 à 14 h, risque élevé : on ne connaît pas les poids exacts ni les LoRA, le prompt système est visible (API-TAIL), le gain est surtout sur la quantification 4 bits.

### M4 : simulateur local de la tâche (notre idée, optionnel)
Sur les A100 : monter une copie de la tâche avec un modèle ouvert de la même taille probable : bf16, int8, nf4, fp8, LoRA entraîné sur peu de pas à plusieurs forces, avec filigrane KGW, prompt système, persona, T=1, glouton, faisceau. Génère des centaines de paires avec vérité, sans limite de requêtes. Sert à : trouver les prompts à égalité qui bougent le plus sous quantification et bruit de poids (B3IT), mesurer la puissance réelle à 100 réponses, régler le seuil. 4 à 6 h, et seulement si on connaît la famille. Les ref donnent déjà un banc d'essai gratuit mais avec 9 points seulement.

### M5 : classifieur de style (embeddings) pour la famille seule
Embedding moyen des réponses longues par endpoint, clustering. Vaut 10 % du score (famille) et sépare mal les variantes. 1 h, risque faible. À faire avec les réponses du bloc C.

## 4. Le meilleur, le plus dur, le plus technique, le plus rapide sûr
- Meilleur rapport score/risque : M1 (sans M3). Cible réaliste [avis] : famille proche de 1,0 ; dans la famille, 4 bits et LoRA fortes séparés, fp8/int8 et LoRA faibles confondus.
- Le plus dur à réussir : séparer fp8/int8 et LoRA 0,3 de la base avec 100 réponses. Les articles disent que c'est à la limite du possible [lu].
- Le plus technique : M3 + M4 (poids locaux, simulateur, alignement de température).
- Le plus rapide et sûr (4 à 5 h) : lots M1 sans poids appris : 60 requêtes de lots + 20 requêtes d'un prompt ouvert identique pour tous, taux d'accord exact par paire, famille par embeddings, clustering lien moyen, seuil réglé sur les ref. On livre une soumission valide dès que ce chemin tourne.

### Neutraliser le décodage (glouton, échantillonné, faisceau)
- Détecter : 6 prompts envoyés 2 fois à chaque endpoint (12 requêtes, comprises dans le bloc de base). Réponses identiques 6 fois sur 6 : endpoint déterministe (glouton ou faisceau). Écart de réponses chez des endpoints par ailleurs identiques : échantillonné. Attention : même un endpoint glouton diffère parfois à cause du calcul GPU (Cai : moins de 5 % d'accord entre fournisseurs).
- Faisceau : reconnaissable à des sorties répétitives, longues, et stables. Aucun article ; à mesurer en simulation M4.
- Répéter un prompt n'apporte rien sur un endpoint déterministe : mettre l'information dans la variété des questions, pas dans les répétitions.
- Statistique invariante : le classement des réponses fréquentes (la réponse la plus probable) ne change pas avec la température, le top-p ni le glouton. Comparer le "mode" par question, pas les fréquences, pour tout couple dont l'un est déterministe [avis].
- Astuce "égalité exacte" : deux endpoints déterministes de même modèle donnent la même réponse à une question fermée, sauf quand un prompt système ou un filigrane déplace un choix. Taux d'accord sur 120 questions : même modèle, 85 à 100 % ; modèles différents, nettement moins [avis, à tester sur ref-01 contre ref-03/05/06].

### Du test par paire au CSV (transitivité)
1. Famille : matrice de distance sur les profils (blocs c et d, embeddings, usage de tokens), clustering hiérarchique, coupe au plus grand saut de distance. Contrôle : chaque famille a au moins 2 endpoints ou on le justifie.
2. Dans chaque famille : p-valeur par paire, transformée en probabilité "même modèle" par régression logistique sur ref (et simulateur si on l'a).
3. Rendre la décision cohérente : A=B et B=C doivent donner A=C. Méthode : clustering par corrélation (on choisit la partition qui minimise le nombre de désaccords pondérés avec les probabilités de paires ; 33 nœuds, résolution exacte ou recherche locale en secondes). Alternative simple : lien moyen avec seuil.
4. Stabilité : rééchantillonner les questions 200 fois, garder la matrice de co-appartenance, couper à 0,5. Une paire instable devient un candidat pour la réserve de requêtes.
5. Partage de données : après regroupement, mettre en commun les réponses d'endpoints jugés identiques (E de la 1re passe) uniquement pour départager un tiers ; une fusion fausse contamine tout.
6. Réglage MCC : à seuil égal, une fausse paire "même" et une fausse paire "différente" coûtent presque autant quand les vraies paires "même" sont minoritaires ; régler le seuil sur les ref et le simulateur, pas sur le score public.

## 5. Plan du budget (100 réponses par endpoint) et protocole de pratique

### Découpage proposé (même liste de prompts pour tous les endpoints, tour par tour)
| Bloc | Requêtes | Contenu | Sert à |
|---|---|---|---|
| A. Lots fermés | 60 | 6 lots de 20 questions x 10 ordres (dont 6 prompts x 2 en déterminisme) | quantif 4 bits, LoRA forte, modèle de base, décodage |
| B. Prompts ouverts | 16 | 8 prompts identiques à tous, 2 réponses de 200 tokens ; 4 questions d'identité | famille, LoRA (style), vraisemblance locale (M3) |
| C. Tokenizer | 0 | lire `usage.prompt_tokens` / `completion_tokens` si le serveur les renvoie | famille, longueur du prompt système caché |
| D. Réserve | 24 | jamais dépensée avant l'analyse : pour les paires ambiguës, avec les questions les plus discriminantes | trancher |
Étape 1 : A + B = 76 sur chaque endpoint. Analyse. Étape 2 : D selon besoin. Toute requête est enregistrée telle quelle (answers.jsonl, comme `task_template.py`).
Débit : 3300 réponses, 4 en parallèle. Avec des réponses de 60 tokens, comptez 40 à 90 min ; avec 256 tokens, 2 à 3 h [avis, à chronométrer sur ref].

Test gratuit du côté tokens [à tester sur ref] : si la réponse JSON contient `usage`, alors (a) `prompt_tokens` moins le nombre de tokens de notre prompt = longueur du prompt système du fournisseur (AgentProv utilise cette fuite pour repérer les prompts cachés [lu]) ; (b) un prompt avec des caractères rares (emoji, mots de langues variées) donne un nombre de tokens propre à chaque tokenizer : empreinte de famille quasi gratuite [avis]. Tester aussi sur ref (requêtes rejetées = non comptées pour les ep) : l'acceptation de `tools` (canal AgentProv, invariant au prompt système), `logprobs`, `n`, un message final de rôle assistant (préremplissage). Si `logprobs` ou `tools` passent, M1 et M3 deviennent nettement plus forts.

### Protocole sur ref-*
1. Mesurer d'abord (30 min) : budget des ref, présence de `usage`, latence, format de sortie des lots (taux de réponses parsées par ligne), déterminisme de chaque ref.
2. Collecter large (gratuit si confirmé) : les 120 questions x 30 ordres sur les 9 ref, plus 8 prompts ouverts x 10.
3. Simuler le vrai budget : tirer au hasard 60 requêtes de bloc A + 16 de bloc B par ref, 200 fois (bootstrap). Calculer, pour chaque tirage, la partition et le MCC avec `score_practice.py`. Le chiffre à retenir est la distribution du MCC sur les 200 tirages, pas un seul tirage.
4. Critères d'acceptation avant de toucher un `ep-*` : (a) les 5 S (plain, prompt système, persona, filigrane, T=1,0) restent ensemble dans au moins 90 % des tirages ; (b) T4 (LoRA 1,5) et T3 (LoRA 0,7) séparés de S dans au moins 80 % ; (c) pas de fusion de T3 avec T4. T1 (fp8) et T2 (LoRA 0,3) : noter le résultat, sans exiger.
5. Garde-fou : 36 paires ref seulement, donc un seuil réglé là peut ne pas se transférer (le modèle ref est "séparé", petit, une seule famille). Réserver une validation "laisser un déguisement dehors" : régler sans ref-06 (filigrane), tester dessus, et inversement.
6. Geler : liste de questions, poids, seuil. Relire à deux, puis lancer l'étape 1 sur les `ep-*`. Annoncer le plan sur l'issue T2 avant (règle de l'équipe).
7. Après l'étape 1 : une soumission pour contrôle de format (une toutes les 30 min). Le score public n'est qu'un contrôle grossier.

## 6. Pièges
- Budget irréversible : 100 réponses par `ep`. Aucun prompt "à voir" sur un `ep`. Une requête invalide n'est pas comptée, mais un prompt qui rend une réponse inutilisable l'est.
- Le lot : si le modèle saute des lignes, le décalage fausse tout. Numéroter les questions ("1.", "2.") et parser par numéro.
- Effet d'ordre dans un lot : mêmes ordres pour tous les endpoints, sinon la comparaison appariée est faussée.
- Filigrane : sur les questions "à égalité" il peut forcer un choix (il favorise un token vert selon le token précédent). Les prompts les plus sensibles à la quantification sont aussi ceux où le filigrane et le prompt système pèsent le plus. D'où la mesure du rapport S/T sur ref. L'hypothèse de la 1re passe "réponses courtes = filigrane invisible" n'est pas fondée.
- Les petits modèles ref ne se comportent pas comme les modèles de `ep` : les poids appris sur ref (types de questions) se transfèrent, les questions précises non.
- Température 1,0 vs défaut : un tirage au hasard change de forme avec la température [lu : IRIS, FLIPS]. Préférer les questions de connaissance et d'arithmétique, et le mode des réponses.
- Fusionner fp8/int8 avec la base est le choix attendu ; ne séparer que sur preuve. Toute coupure sans preuve crée de fausses paires "différentes".
- Une seule famille sur ref : le seuil de la coupe famille/modèle n'est pas calibré. Mettre le seuil famille sur la distribution des distances entre S et S, et regarder les sauts à l'œil.
- Latence, pannes 502/503 : `task_template.py` réessaie, mais les 429 sans `Retry-After` signifient "budget épuisé" : arrêter, ne pas relancer en boucle.
- Ce qui n'a pas été lu en détail : RESF (PDF illisible par l'outil), le papier prefix tree, les annexes tronquées de MET, IRIS, AgentProv. Les chiffres [résumé] ne sont pas des preuves.

## Sources
- MET 2410.20247 ; RUT 2506.06975 ; AgentProv 2609.00052 (EMNLP 2026, aclanthology.org/2026.emnlp-main.638) ; Cai 2504.04715
- B3IT 2602.11083 ; DFA 2610.37086 ; IRIS 2607.20860 ; FLIPS 2606.03330 ; One Token 2607.10252 ; Forty Questions 2610.28281 ; KBF 2605.29524 ; API-TAIL 2610.36801
- AdaptPrint 2608.22213 ; SetPrint 2611.12950 ; Anosognosia 2610.06174 ; RESF aclanthology.org/2025.emnlp-main.247 ; Stemma 2607.25880 ; BReF 2609.06330
- Fichiers de départ : E:\EAC\reports\sources\Model_Equality_Testing.txt et \t2\
