# Tâche 2 "Model Equality" : état de l'art et plan (21 nov. 2026)

Décision débloquée : quelle méthode implémenter, et combien ça coûte en heures
par rapport aux tâches 1 (watermark) et 3 (unlearning).
Rien n'a été interrogé : aucun appel API, aucune requête dépensée.
Ce qui vient des articles est marqué [papier]. Le reste est mon raisonnement : [avis].

## 1. Ce qu'on nous demande (5 lignes)
1. 33 endpoints (ep-01..33), texte seulement. 100 réponses max par endpoint pour toute la compétition.
2. Il faut dire quels endpoints ont le même modèle de base (famille, 10 % du score).
3. Puis, dans chaque famille, lesquels servent exactement le même modèle (90 % du score).
4. Même modèle = même checkpoint + même quantification + même LoRA. Filigrane, prompt système et
   décodage ne changent pas l'identité : ce sont des pièges (ils changent le texte sans changer le modèle).
5. Note finale = MCC sur les paires. Une paire fausse dans un sens ou dans l'autre coûte.

Image simple : 33 cuisiniers. Certains ont la même recette, mais l'un a un tablier différent (filigrane),
un autre un chef qui lui souffle des consignes (prompt système). D'autres ont une recette à peine modifiée
(quantification, LoRA). On goûte 100 plats chacun et on doit regrouper les vraies recettes identiques.

## 2. État de l'art et ce que chaque test demande

| Source | Idée | Échantillons nécessaires | Verdict pour nous |
|---|---|---|---|
| MET, Gao et al. [arXiv 2410.20247](https://arxiv.org/abs/2410.20247) | Test à deux échantillons, MMD avec noyau de Hamming (compte les positions de tokens identiques, complétions complétées à longueur L). p-valeur par simulation (1000 tirages du modèle de référence) ou par permutation. | Environ 10 réponses par prompt, m=25 prompts, donc N=250 par modèle (L=50 tokens). Pouvoir médian 77 % sur plusieurs altérations. | Il faut 2,5 fois plus que notre budget. Int8 sur Llama-3.1 8B : pouvoir 0,08 seulement. En espace caractères, nf4/int8 passent de 0,38/0,30 (N=10m) à 1,00/0,96 (N=50m). |
| RUT, Zhu et al. [arXiv 2506.06975](https://arxiv.org/abs/2506.06975) | Pour chaque prompt, on prend 1 réponse de l'API et on la classe parmi 100 réponses locales du vrai modèle (score log-rang). Si c'est le même modèle, les rangs sont uniformes (test de Cramér-von Mises). | 100 prompts x 1 requête API, 30 tokens max, mais exige le modèle authentique en local (avec probabilités). | Cadre idéal si on connaît les poids de la famille. 4 bits détecté, 8 bits quasi indétectable par toutes les méthodes (AUC 0,05 à 0,13 sur Gemma, Llama, Mistral). |
| AgentProv, Wang et al. [arXiv 2609.00052](https://arxiv.org/abs/2609.00052) | MMD sur la distribution des appels d'outils. Les auteurs de la tâche. | Budget non précisé dans le résumé. | Sort du texte pur, inutilisable ici. Utile pour une idée : MET donne 67 % de faux positifs et RUT 53 % quand un prompt système est injecté, AgentProv 7 % [papier]. Le prompt système est donc LE piège connu. |
| LLMmap [arXiv 2407.15847](https://arxiv.org/abs/2407.15847) | 8 requêtes bien choisies, un classifieur sur les réponses. | 8 à 10 requêtes, mais il faut entraîner sur les modèles candidats. | Distingue des modèles différents, pas des quantifications. Étude indépendante : 72 à 74 % de précision [Invisible Traces, arXiv 2501.18712](https://arxiv.org/pdf/2501.18712). |
| Idiosyncrasies, Sun et al. [arXiv 2502.12150](https://arxiv.org/abs/2502.12150) | Classifieur fine-tuné sur du texte : 97 % pour reconnaître 5 modèles de chat. | Beaucoup de données d'entraînement. Avec 10 exemples, 40 % seulement. Les réglages de température sont presque indétectables (38 % pour 5 classes). | Bon pour la famille. Le fait que la température est dure à voir aide : le décodage est un piège peu gênant. |
| Cai et al., substitution de modèle [arXiv 2504.04715](https://arxiv.org/abs/2504.04715) | Audit de substitution. | n/a | Conclusion des auteurs : les tests statistiques sur le texte sont gourmands en requêtes et échouent sur les substitutions subtiles. À lire comme un avertissement. |
| LLMPrint [arXiv 2509.25448](https://arxiv.org/pdf/2509.25448v1), LeafBench [arXiv 2508.19843](https://arxiv.org/pdf/2508.19843v3), RAFP [Columbia](https://www.cs.columbia.edu/~junfeng/papers/rafp/) | Empreintes par prompts optimisés qui poussent vers des tokens précis. 700 variantes quantifiées ou fine-tunées testées. | Prompts optimisés avec accès au modèle local. | Piste "prompts qui amplifient la différence" : faisable seulement sur le modèle de référence local. Je n'ai pas lu le détail. |

À retenir [papier] : avec 100 réponses par endpoint, on est sous le budget de MET et à peu près au
budget de RUT en nombre de prompts. On détectera bien les gros écarts (autre modèle, LoRA fort,
4 bits) et mal les petits (8 bits, fp8, LoRA faible).

## 3. Approches classées

Les 9 endpoints ref (ref-01..09) sont gratuits et sans limite d'après le texte officiel : tout se règle dessus.
Leur réponse connue : 1 famille, 5 endpoints "S" (simple, prompt système, persona, filigrane, température 1.0),
T1 = fp8, T2 = LoRA 0,3, T3 = LoRA 0,7, T4 = LoRA 1,5. C'est exactement le banc d'essai qu'il nous faut.
Piège à vérifier : les 4 variantes ne sont pas toutes à la même distance de S, donc la force des LoRA
sur les vrais endpoints peut différer.

### Approche A (recommandée) : "empreinte de réponses courtes" + clustering
- Idée : beaucoup de questions à réponse très courte (1 à 5 tokens). On compare, endpoint par endpoint,
  la distribution des réponses. Une courte réponse dépend peu du style, donc peu du prompt système, du
  persona ou du filigrane. Le modèle (poids) la fait varier.
- Prompts : 50 à 60 prompts. Mélange de : choix forcés ("Réponds par A ou B uniquement"), arithmétique
  à 3-4 chiffres, faits obscurs (là où la quantification flippe la réponse), "Donne un nombre au hasard
  entre 1 et 100", "un animal au hasard". Contraindre le format ("une seule ligne, sans explication").
  Choisir de préférence des prompts "à la frontière" : ceux où la réponse du modèle est partagée entre
  deux options, car un petit décalage des poids bouge alors la fréquence. Ces prompts se trouvent
  gratuitement sur les ref et sur un modèle local (voir section 5).
- Statistique : par paire d'endpoints, MMD avec noyau "égalité exacte" sur la réponse normalisée
  (minuscules, sans espaces ni ponctuation), p-valeur par permutation. Ou distance de Jensen-Shannon
  sur les fréquences par prompt. Puis clustering hiérarchique (lien moyen) avec seuil calibré sur les ref.
- Budget (100 par endpoint) : 20 pour l'étape 1 (20 prompts variés, tous endpoints, famille +
  premiers clusters), 60 pour l'étape 2 (prompts sélectionnés sur les ref, 12 prompts x 5 répétitions
  pour les paires ambiguës), 20 de réserve jamais dépensée avant l'étape finale.
- Effort : 5 à 7 h (prompts 1 h, pipeline 2 h, calibrage ref 2 h, essais 1-2 h).
- Risque : moyen. Les réponses courtes sont faibles pour quantification légère. Température 1.0 étale
  les distributions : comparer plutôt la réponse modale ou normaliser.

### Approche B : MMD de Hamming sur textes longs (MET)
- Idée : le test officiel de la littérature, appliqué paire par paire.
- Prompts : 20 prompts ouverts (complétion de préfixes fixes, texte encyclopédique), 5 répétitions,
  150 à 256 tokens. Comparer les 50 premiers tokens (re-tokéniser le texte avec un tokenizer local).
- Statistique : MMD Hamming, p-valeur par permutation (pas de modèle de référence requis) [papier].
- Budget : 100 = 20 prompts x 5. Exactement la configuration où le papier a le moins de pouvoir.
- Effort : 3 à 4 h. Risque : élevé. Le prompt système et la température 1.0 déplacent fortement
  la distribution : MET donne 67 % de faux positifs avec un prompt système [papier, AgentProv].
  Fonctionne comme second avis, pas comme méthode principale.

### Approche C : rangs de log-vraisemblance sous un modèle local (RUT adapté)
- Idée : pour chaque réponse d'un endpoint, calculer sa vraisemblance moyenne sous le modèle de base local.
  Une LoRA forte ou une quantification sévère donne un score moyen différent. Un seul chiffre par réponse.
- Prompts : 100 prompts courts, 1 réponse chacun (pour 100 réponses par endpoint, comme RUT [papier]).
  Réponse de 30 tokens max pour éviter que le filigrane ne s'accumule.
- Statistique : par endpoint, distribution des log-rangs sous le modèle local. Test de Kolmogorov-Smirnov
  ou de Cramér-von Mises entre endpoints, ou position par rapport à l'uniforme.
- Budget : 100 prompts x 1 (aucune répétition). Peut réutiliser les réponses de l'approche A si elles
  sont assez longues : on récupère le texte gratuitement.
- Effort : 6 à 8 h (récupérer les poids, logprobs, calibrage). Risque : élevé si on ne connaît pas la
  famille et ses poids exacts. Très bon si la famille est identifiable (voir 5).

### Approche D : classifieur de style (embeddings) pour la famille
- Idée : embeddings des réponses à 10 prompts ouverts, moyenne par endpoint, clustering. Pour les familles
  seulement (10 % du score), pas pour les variantes. 
- Effort : 2 h. Risque faible pour la famille, nul pour le reste. À faire gratuitement avec les réponses de A.

### Approche E (en complément) : fusion progressive
- Dès qu'on a deux endpoints jugés identiques, on met leurs réponses en commun (200 au lieu de 100).
  Cela double la puissance pour les comparaisons suivantes. À faire de façon prudente : une fusion
  erronée contamine tout le reste.

Classement : A (noyau) + D (famille) + E, puis C si la famille est connue, B en second avis.
Pas besoin de tout faire : A + D font déjà un score honnête.

## 4. Famille facile, variantes difficiles, et les pièges de décodage

### Famille (10 %)
Presque gratuit. Tout classifieur ou même un clustering sur les premiers tokens de réponses à
"Qui es-tu ?" / "Complète cette phrase" sépare des familles différentes [papier : changement de
famille détecté avec un pouvoir proche de 100 %, MET]. Les paires de la même famille sont les
seules ambiguës.

### Variantes (90 %)
- LoRA forte (comme ref-07, force 1,5) : change le style et le contenu, visible dans les embeddings
  et les fréquences de réponses [avis, à vérifier sur ref].
- LoRA faible (ref-09, force 0,3) et fp8 (ref-02) : proches de S. Avec 100 réponses, à peine détectables
  d'après MET et RUT (8 bits quasi invisible) [papier]. On accepte de les rater, mais il faut éviter
  de les mélanger avec des disguises.
- Quantification : pas de truc magique en texte seul. Les arithmétiques à plusieurs chiffres et les
  faits rares flippent plus que le reste [avis].

### Les pièges qui font croire à une différence
- Filigrane : décale la fréquence de certains tokens ("liste verte"). Visible sur les longues réponses
  (surtout en espace token). Réponses courtes = filigrane quasi invisible. C'est un argument de plus pour A.
- Prompt système : change le ton, la longueur, ajoute une persona. Contre-mesure : prompts qui imposent
  un format de sortie, et normaliser (minuscules, suppression des préfixes, ne regarder que la réponse).
- Décodage / température : étale ou resserre les probabilités. Contre-mesure : compter les réponses
  modales ou utiliser un noyau tolérant. Note : le papier d'idiosyncrasies voit les températures
  très mal (38 % pour 5 classes), donc peu gênant.
- Test pratique : parmi les 5 endpoints ref de type S, vérifier que la méthode les regroupe bien avec
  les 4 disguises (prompt, persona, filigrane, température). Si elle les sépare, elle est inutilisable.

## 5. Faire tourner les modèles candidats sur les A100 ?

Oui, utile, mais seulement dans trois cas :
1. Identifier la famille exacte. Demander gratuitement aux ref "Qui es-tu ?" et regarder la taille et le style.
   La consigne parle d'OLMo : si les familles sont des OLMo, on peut les récupérer sur Hugging Face
   (https://huggingface.co/allenai).
2. Trouver les prompts sensibles. Sur un A100, sortir les probabilités du modèle de base et de ses versions
   quantifiées (bitsandbytes int8, nf4, fp8) : garder les prompts où la distribution bouge le plus.
   Coût : quelques heures de GPU, pas de contrainte de budget.
3. Calculer les rangs de l'approche C.

Limite : on ne connaît pas les LoRA des organisateurs, donc on ne peut pas les recopier. Le gain
est sur les prompts et la calibration, pas sur "reproduire" un endpoint. 1 nœud suffit largement.
Ne pas confondre : le local ne remplace pas les 100 réponses par endpoint, il sert à mieux les dépenser.

## 6. Pièges (budget irréversible)
- Les 100 réponses sont définitives : écrire la liste de prompts, la relire à deux, et la tester sur
  les ref (gratuits) avant d'en dépenser une seule sur ep-xx.
- Une requête invalide n'est pas comptée, mais un prompt qui donne une réponse inutilisable l'est :
  tester le format de sortie sur ref d'abord.
- Ne jamais dépenser les 100 d'un coup. Étape 1 : 20. Étape 2 : 60. Réserve : 20.
- Le budget est par endpoint : on peut dépenser plus sur les endpoints ambigus et moins sur les évidents.
- Sauvegarder toutes les réponses brutes (answers.jsonl, comme dans task_template.py). Ne rien rejouer.
- 4 requêtes en parallèle : 3300 réponses. Réponses courtes (max_tokens petit) = plus rapide.
  Avec 256 tokens, compter plusieurs heures. [avis]
- Soumission : une toutes les 30 min. Public/privé : le score public ne couvre qu'une partie des
  paires. Ne pas régler un seuil sur le public. Décider du seuil sur les ref, geler, puis soumettre.
- MCC : un nombre de modèles trop grand (tout séparé) donne beaucoup de fausses "différentes", un
  nombre trop petit beaucoup de faux "pareils". Le seuil du clustering est le paramètre le plus risqué.
- Piège du leading space (MET vs RUT) : un espace en tête de réponse a gonflé le MMD à 1,0 [papier] :
  normaliser les textes.
- Dépendance : il se peut que les 33 endpoints incluent plus d'une famille avec des tailles différentes.
  Les seuils calibrés sur ref (une seule petite famille) pourraient ne pas se transférer [avis].

## 7. Difficultés (sur 10)
- Technicité : 6/10. La stat est simple (MMD, permutation, clustering). La finesse est dans la
  conception des prompts et la gestion du faux positif.
- Temps : 6/10. Un pipeline utilisable en 5 à 7 h, mais l'amélioration marginale est longue.
  Maximum utile : 12 h sur A + C.
- Calcul : 3/10 pour A et D, 5/10 pour C ou si on veut faire du criblage quantifié local.
  Un nœud A100 suffit.

Comparaison avec les autres tâches (lu dans les énoncés, pas exécuté) :
- T1 (localiser un filigrane d'image) : entraînement de modèles de segmentation sur 3 méthodes.
  Plus lourd en code et en GPU. À mon avis : technicité 7, temps 8, calcul 7.
- T3 (désapprentissage d'un ViT sur 3500 images) : entraînement et mesure de la ressemblance à un
  modèle ré-entraîné. À mon avis : technicité 7, temps 7, calcul 6.
- T2 est la moins coûteuse en calcul. Plafond de score incertain : les variantes subtiles risquent
  d'être invisibles quoi qu'on fasse. On peut viser les gros écarts sans risque : un score de
  0,3 à 0,5 me paraît plausible, au-delà c'est de la chance [avis, non mesuré].

## Plan de départ (ce que je ferais dans l'ordre)
1. 30 min : demander aux ref de se présenter, identifier le modèle, écrire 60 prompts candidats.
2. 2 h : interroger les 9 ref gratuitement avec les 60 prompts x 10 répétitions. Mesurer
   quelle statistique regroupe les 5 S ensemble et sépare T1..T4. Choisir 12 à 20 prompts gagnants.
3. 30 min : relire, geler, dépenser 20 requêtes sur chaque ep (famille + clusters).
4. 1 h : analyse, première soumission pour calibrer. Ne pas surapprendre du public.
5. Étape 2 : dépenser 60 de plus sur les paires ambiguës. Puis soumission finale, réserve intacte.

## Sources
- MET : https://arxiv.org/abs/2410.20247
- RUT : https://arxiv.org/abs/2506.06975
- AgentProv : https://arxiv.org/abs/2609.00052
- LLMmap : https://arxiv.org/abs/2407.15847 ; évaluation indépendante : https://arxiv.org/pdf/2501.18712
- Idiosyncrasies : https://arxiv.org/abs/2502.12150
- Auditing model substitution : https://arxiv.org/abs/2504.04715
- LLMPrint : https://arxiv.org/pdf/2509.25448v1 ; LeafBench : https://arxiv.org/pdf/2508.19843v3 ; RAFP : https://www.cs.columbia.edu/~junfeng/papers/rafp/
- Doc officiel et fichiers de départ : E:\EAC\reports\sources\Model_Equality_Testing.txt et \t2\
- Données lues dans les pages web, pas dans les PDF complets : les chiffres de MET (page 1 sur 2 seulement)
  et de RUT sont à confirmer dans le texte complet avant de s'y fier.
