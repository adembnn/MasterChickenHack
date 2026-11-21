# T2 : empreinte de modèle + pipeline assisté par GPU (JURECA)

Rien n'a été appelé (aucun ep-*, aucun téléchargement). Marquage : [papier] = lu dans un résumé/page web,
[non vérifié] = je n'ai pas pu confirmer, [avis] = raisonnement ou estimation de ma part.
Complète reports/t2-model-equality-etat-de-lart.md (A/B/C/D/E) sans le répéter.

## 0. Idées nouvelles qui changent le plan (à lire en premier)

1. **Le budget est en RÉPONSES, pas en tokens.** 100 réponses x 256 tokens = 25 600 tokens par endpoint.
   Le plan A du rapport existant (réponses de 1 à 5 tokens) jette 98 % du signal gratuit. Toujours
   demander des réponses qui remplissent 256 tokens. Pour un test de vraisemblance, le signal croît
   avec le nombre de tokens, pas avec le nombre de requêtes. Entrée jusqu'à 1 024 tokens : on peut aussi
   donner un long préfixe pour placer le modèle dans un régime précis (continuation de texte).
2. **Le local sert de "scoreur de tokens", pas de générateur d'endpoint.** Chaque texte d'endpoint est noté
   sous K modèles candidats (base, instruct, quantifiés, LoRA à force variée). On obtient des vecteurs
   endpoint x candidat (section 3). C'est le RUT généralisé : RUT [papier arXiv 2506.06975] utilise 1 rang par
   réponse ; nous utilisons des centaines de rangs/logprobs par réponse (256 tokens), donc ~100x plus de signal par requête.
3. **Les "endpoints synthétiques" locaux** (section 4) donnent un jeu étiqueté illimité imitant exactement les
   déguisements ; on y entraîne un classifieur de paires et on calibre les seuils sans dépenser de requêtes.
4. **Les 9 ref sont gratuits et illimités** : ce sont les seules données réelles avec les LoRA et le fp8 des
   organisateurs. Les utiliser (a) pour identifier le modèle ref (OLMo ? "Qui es-tu ?", style), (b) pour valider que
   le pipeline synthétique reproduit les distances réelles (section 6), (c) pour calibrer la force LoRA équivalente.
5. **Sous-ensembles de tokens invariants** (section 5) : on peut calculer les features seulement sur les positions
   où le filigrane/température/prompt système ne changent presque rien.

## 1. Littérature utile (et ce qu'elle dit de nos 33 endpoints)

| Source | Contenu | Utilité pour nous |
|---|---|---|
| LLMmap, Pasquini et al., USENIX Sec. 2025, https://arxiv.org/abs/2407.15847 | 3 à 8 requêtes, classifieur (fermé 42 LLM >95 %, ouvert par contraste), robuste aux system prompts et à l'échantillonnage [papier]. Code : github.com/pasquini-dario/LLMmap [non vérifié, ne pas exécuter] | Bon pour la FAMILLE (10 %), pas pour quantif/LoRA. Éval. indépendante 72-74 % [papier 2501.18712]. Idée à reprendre : requêtes de type "bannière" ("quel modèle es-tu", requêtes de refus/alignement, questions de connaissances, requête mal formée) |
| Idiosyncrasies, Sun et al. https://arxiv.org/abs/2502.12150 | Classifieur de texte 97 % sur 5 modèles de chat ; très mauvais pour la température [papier d'après rapport existant] | Les mots/débuts de phrase signent la famille ; la température/échantillonnage est un déguisement faible |
| Cai et al., "Are You Getting What You Pay For?" https://arxiv.org/abs/2504.04715 | Un classifieur de texte entre pleine précision et FP8/INT8 est au niveau du hasard [papier]. Conclusion : le texte seul est inefficace contre la quantification légère | Plafond réaliste : fp8/int8 sera difficile ; il faut la vraisemblance sous le modèle local, pas un classifieur de style |
| RUT, Zhu et al. (ICLR 2026) https://arxiv.org/abs/2506.06975 | 1 appel par prompt, rangs sous modèle local, test d'uniformité ; bat MMD sur quantification, mais aucun test ne détecte le 8 bits sur Gemma/Mistral [papier]. Robuste au "rerouting" | Cadre de base ; nous l'étendons (token-level) |
| MET, Gao et al. https://arxiv.org/abs/2410.20247 | MMD Hamming ; 77,4 % de pouvoir médian avec ~10 échantillons x 20-25 prompts ; considère le filigrane comme une DISTORSION détectée, donc le filigrane est visible du test [papier] | Confirme que filigrane = nuisance qu'il faut modéliser. Aucun résultat sur la survie du filigrane à la quantification [non trouvé] |
| UTF (tokens sous-entraînés) https://arxiv.org/pdf/2410.12318 | Empreintes par tokens rares, mais le propriétaire doit insérer la paire (coopératif) [papier] | Non applicable tel quel ; mais idée "tokens rares = séparent les tokenizers/familles" |
| One Token Is Enough https://arxiv.org/abs/2607.10252 ; FLIPS https://arxiv.org/pdf/2606.03330 ; "Token Counts Are Not Model Lineage" https://arxiv.org/pdf/2608.29930 | Distribution de sortie sur prompts triviaux d'un mot comme empreinte (165 modèles) ; tailles de tokenization comme empreinte étroite [résumé web, non lu en détail, [non vérifié]] | Idée "réponse à 1 mot" = déjà dans plan A ; compter les tokens du prompt n'est pas possible ici (on ne voit pas usage) |
| LAPD (vraisemblance base vs aligné) https://arxiv.org/html/2604.16923 ; PhantomHunter https://arxiv.org/html/2506.15683v1 ; "When Detection Fails" https://arxiv.org/abs/2506.09975 | Le ratio de log-vraisemblance base/fine-tuné est bruité en zone haute entropie, on le standardise ; un fine-tune garde des listes de probabilités proches de sa base [papier, snippets] | Soutient notre statistique : LLR standardisé par token, pondéré par l'information |
| MarkLLM https://github.com/THU-BPM/MarkLLM (KGW, Unigram, ...) ; lm-watermarking (Kirchenbauer) | Implémentations open source de KGW et Unigram [papier + vérifié que le dépôt existe via la page] | Pour fabriquer les endpoints filigranés (section 4). Je n'ai pas vérifié l'API exacte : lire le code avant |
| OpenRouter blog sur fp8/fp4 https://openrouter.ai/blog/insights/llm-quantization | Pas de différence de précision mesurable sur benchmarks entre fp8/fp4/fp16 [papier/blog] | Confirme : le fp8 se voit en distribution, pas en précision de tâche |
| Forums/blogs d'astuces d'audit | Rien de fiable trouvé en recherche ; la littérature insiste que les requêtes "spéciales" peuvent être détectées et re-routées (non pertinent ici, les endpoints du hackathon sont statiques) | - |

Faits pratiques de la littérature à retenir : (i) LoRA/fine-tune garde la famille de probabilités de la base, donc
base et LoRA sont proches ET distinguables seulement avec beaucoup de tokens ; (ii) la quantification 4 bits est détectable,
8 bits/fp8 quasi pas à N faible ; (iii) le filigrane et le prompt système sont les gros confondeurs.

## 2. Ce que 25 600 tokens permettent (ordre de grandeur, [avis])

Pour une petite perturbation, par token : E[LLR] = KL, Var[LLR] ≈ 2·KL. Sur N tokens : z ≈ sqrt(N·KL/2).
N = 25 600 (100 réponses pleines, un seul endpoint), N = 51 200 si comparaison de deux endpoints (le test paire
est moins puissant que le test "vs candidat connu").

| Perturbation | KL/token typique par rapport à la base (ordre de grandeur, [non vérifié]) | z avec N=25 600 | Détectable ? |
|---|---|---|---|
| nf4 / int4 | 1e-2 à 5e-2 | 11 à 25 | Oui, facilement |
| LoRA force 1,5 | 1e-2 à 1e-1 (dépend de l'adaptateur) | >10 | Oui |
| LoRA force 0,7 | 3e-3 à 3e-2 | 6 à 20 | Oui |
| LoRA force 0,3 | 3e-4 à 5e-3 | 2 à 8 | Parfois |
| fp8 / int8 | 1e-4 à 2e-3 | 1,1 à 5 | Limite (si on a le candidat exact) |

Réserves : (a) ces z supposent qu'on connaît EXACTEMENT le candidat local et que ni température ni filigrane
ne s'ajoutent ; une température 1.0 vs défaut ou un filigrane peuvent produire un KL bien supérieur à celui du
fp8 (nuisance), d'où l'importance des sous-ensembles invariants et du modèle de nuisance (section 5) ; (b) les
tokens à faible entropie ne portent presque aucun signal : les "25 600 tokens" valent peut-être 5 000 utiles ;
(c) un simple "fp8 local par arrondi" peut différer de l'implémentation serveur (vLLM fp8 dynamique des
activations ?) : prévoir plusieurs variantes fp8 locales et prendre le max.

## 3. Pipeline de scoring "endpoint x candidat"

### 3.1 Candidats locaux (zoo)
Hypothèse : les familles sont OLMo (l'énoncé cite "OLMo base model" dans l'exemple). Tailles/versions connues :
OLMo 3 : 7B et 32B, base/Instruct(7B)/Think, plus 3.1 32B Instruct/Think, sortis nov. 2025
(https://huggingface.co/collections/allenai/olmo-3 [papier/web]) ; OLMo 2 : 1B/7B/13B/32B (0425 etc.) [non vérifié, de mémoire] ;
OLMo-1/1.7 7B ; OLMoE-1B-7B. Les ref servent "un petit modèle séparé" : probablement 1B (OLMo-2-1B Instruct ?) [avis].
Un A100 40 Go : 7B en bf16 = 15 Go OK ; 13B bf16 = 26 Go OK ; 32B bf16 = 64 Go NON (utiliser int8/nf4 ou 2 GPU en
`device_map=auto`, mais alors le "zoo" quantifié du 32B est lui-même une approximation).
Téléchargement : les noeuds de calcul JURECA n'ont typiquement pas d'internet [non vérifié] : télécharger les poids sur le
noeud de login dans le cache HF partagé de l'équipe (le script hackathon_setup.sh configure déjà des caches partagés HF/uv),
puis HF_HUB_OFFLINE=1 dans les jobs. Ne rien télécharger avant l'autorisation de l'équipe (règle de la tâche).

Pour chaque famille/taille F, candidats : F-base, F-instruct (+ SFT/DPO/RLVR si ce sont des variantes distinctes), puis variantes :
- fp8 (e4m3 poids, éventuellement activations), int8 (bitsandbytes LLM.int8 et RTN par canal), nf4 (bnb), GPTQ-4bit (gptqmodel/AutoGPTQ,
  ~30-60 min de calibration par 7B), AWQ-4bit (llm-compressor ou autoawq [non vérifié, projet peu maintenu]).
- LoRA : cf. 3.3.

### 3.2 Quantification locale sur A100 (sans FP8 matériel)
- A100 (Ampere) n'a pas de cœurs FP8 ; émuler par "fake quant" : poids arrondis en `torch.float8_e4m3fn` avec échelle par tenseur ou
  par canal puis recastés en bf16. Numériquement, c'est ce que fait un fp8 poids-seulement. Pour W8A8-fp8 (vLLM), ajouter un
  arrondi fp8 des activations par hook (échelle dynamique par token). Faire les DEUX variantes et noter le max.
- int8 : bnb `load_in_8bit` (LLM.int8, outliers en fp16) vs RTN symétrique par canal : deux candidats.
- nf4 : bnb `load_in_4bit`, `bnb_4bit_quant_type="nf4"`, avec/sans double quantification ; compute dtype bf16.
- Valider chaque candidat quantifié : la perplexité sur 20 prompts ne doit pas bouger de plus de ~1 % ; sinon bug.

### 3.3 LoRA locaux et mise à l'échelle de la force
- On n'a pas l'adaptateur des organisateurs. Deux options : (a) adaptateurs publics OLMo si existants [non vérifié] ;
  (b) entraîner nos propres LoRA (section 4.4) sur petits jeux : r=8-16, alpha=16-32, sur q,k,v,o,gate,up,down, lr 1e-4, 200-500 pas.
- Mise à l'échelle de la force s : delta_W = s · (alpha/r) · B·A. Dans PEFT, boucler sur les modules LoRA et multiplier
  `module.scaling["default"] *= s` ; ou, équivalent, multiplier tous les `lora_B.weight` par s. Fusionner (`merge_and_unload`) pour
  la génération rapide. Force 1,5 = extrapolation (bruit de style plus marqué, pas de garantie de stabilité).
  Générer s dans {0,2 ; 0,3 ; 0,5 ; 0,7 ; 1,0 ; 1,5}.
- Calibration de s par rapport aux ref : nos LoRA ne sont pas ceux des organisateurs ; la correspondance se fait au niveau de la
  statistique (KL/token vers la base) : on choisit s tel que notre KL moyen égale celui mesuré entre ref-09 / ref-04 / ref-07 et ref S
  (mesuré avec les vecteurs de la section 3.4). Ainsi nos synthétiques ont la "même difficulté".

### 3.4 Features par token et par endpoint
Entrée : 100 textes (prompt + réponse tokenisée avec le tokenizer candidat, chat template identique à celui des endpoints ; ne scorer
que les tokens de la réponse ; ignorer les 8-16 premiers tokens, car le prompt système/persona y laisse l'empreinte principale).
Pour chaque (endpoint e, candidat c) et chaque token t : lp (log-prob sous c), rang du token observé, H (entropie de c), p_top1, masque
"top1 observé".
Agrégats par (e, c) (le vecteur de features) :
1. mean lp, mean rang-log (RUT généralisé), fraction top-1 observé, fraction rang<=3, mean(lp + H) = surprise centrée (≈0 si le
   texte est échantillonné de c à T=1 sans déguisement ; c'est le "z-score d'entropie"), variance de (lp + H).
2. Les mêmes sous 3 à 5 classes d'entropie de c (H<0,3 ; 0,3-1 ; 1-2 ; >2 nats) : donne la signature de nuisance (température = pente
   par classe ; filigrane = biais token-spécifique).
3. LLR standardisé entre paires de candidats (ex. fp8 vs base ; LoRA0,5 vs base) par réponse, puis moyenne et écart-type sur 100 réponses.
4. Rang-PIT : pour chaque token, la fonction de répartition PIT randomisée sous c ; un modèle correct donne PIT uniforme ; statistiques KS/CvM
   sur l'ensemble et par classe d'entropie.
5. Features de famille/tokenizer : meilleur candidat par mean lp (argmax sur c), marge entre le meilleur et le deuxième, fraction des
   tokens hors-vocabulaire du tokenizer c (pièce non tokenisée proprement, i.e. rétro-tokenisation différente).
6. Longueur de réponse, taux de refus, taux de terminaison avant 256 (invariants faibles mais gratuits).
Sortie : matrice F[e, c, k]. Taille ~33 x (30 candidats) x ~40 = peu de Mo.
Pour la paire (i,j) : features = |F[i]-F[j]|, signe et normalisation par l'écart-type bootstrap (rééchantillonnage des 100 réponses),
aucune dépendance au candidat unique.

### 3.5 Identification de la famille avec ce vecteur
La famille = le candidat de plus haute log-vraisemblance (mean lp) ; entre tailles différentes d'OLMo la différence est énorme
(plusieurs dixièmes de nat/token), donc précision attendue sur la famille 98-100 % [avis]. Si aucun candidat n'est bon (mean lp très
inférieure, rang médian élevé pour tous), l'endpoint est un modèle non reproductible : bascule vers le plan de secours (section 8).

## 4. Fabriquer des endpoints synthétiques

Principe : pour chaque modèle de base B (par ex. OLMo-2-1B/7B, plusieurs checkpoints pour avoir plusieurs "familles"),
générer des endpoints avec les mêmes 5 degrés de liberté que l'énoncé : (modèle = checkpoint x quantif x LoRA) et (déguisement =
filigrane x prompt système x persona x décodage). Étiquette = identité du modèle ; le déguisement est la nuisance. On produit
des paires dont la vérité terrain est connue.

### 4.1 Déguisements (nuisances)
- Filigrane : KGW (hachage du token précédent, gamma 0,25, delta 1-3, h=1 ou 4) et Unigram (liste verte fixe, gamma 0,5, delta 1-2)
  via MarkLLM ou lm-watermarking (LogitsProcessor HF). Plusieurs clés. Environ 30 % des synthétiques filigranés.
  Variante réaliste : on ne sait pas quels (gamma, delta, h) utilisent les organisateurs ; échantillonner large et laisser le
  classifieur apprendre l'invariance.
- Prompt système : tirer dans une bibliothèque de 50-100 prompts (assistant générique, concis, "réponds en français", style
  entreprise, etc.) ; les ref-03 (system prompt) et ref-05 (persona) donnent des exemples réels de longueur/ton. Les réponses de ref-03/05
  à des prompts neutres révèlent ce que fait l'organisateur ; les imiter.
- Persona : "Tu es Capitaine Haddock/un juriste/..." (un prompt système de style).
- Échantillonnage : T dans {0,6 ; 0,8 ; 1,0 ; 1,2}, top-p {0,9 ; 0,95 ; 1}, top-k {0 ; 50}, min-p ; l'endpoint ref-08 "temperature 1.0"
  indique que le défaut est probablement plus bas (0,7 ?) [avis] ; mesurer sur ref (pente de lp+H par classe d'entropie).
- Décodage : également beam/greedy ? (greedy est inatteignable par l'API si la température est ignorée, mais le décodage "strategy" est
  une propriété serveur) ; inclure greedy et top-k à petite proportion.
### 4.2 Génération
vLLM (si installable ; V1 a changé l'API des processeurs de logits [non vérifié]) pour les cas sans filigrane (fusion des poids LoRA/quantif
déjà faite : un modèle chargé par variante) ; HF `generate` avec LogitsProcessor pour le filigrane. Batch 64-128, 256 tokens max, une
réponse par prompt (comme l'API). Débit : ordre de 2-5k tok/s par A100 en vLLM 7B, 1-2k tok/s HF [avis, non mesuré].
Prompts : le même ensemble de P prompts (P=100) qui sera utilisé en vrai, plus un second ensemble disjoint pour la validation.
### 4.3 Taille et coût
Un endpoint synthétique = 100 réponses x 256 tokens = 25 600 tokens. 1 000 endpoints = 25,6 M tokens : 25,6M/(4 GPU x 3k) = 35 min (vLLM)
à 1,8 h (HF). 3 000 endpoints = environ 2 à 5 h de noeud. Suffisant pour un classifieur de paires (millions de paires étiquetées). La plus
grosse partie du coût est le scoring (3.4) : 25,6M tokens x 30 candidats = 768 M tokens de passes avant ; à ~15k tok/s/GPU en 7B
= 14 h GPU, soit 3,5 h noeud. Réduire : 10 candidats utiles + sous-échantillonnage à 60 réponses : ~1 h noeud. [avis]
### 4.4 LoRA synthétiques
Entraîner K=8-16 adaptateurs LoRA distincts sur de petits jeux (1-3k exemples) : style (poésie, juridique, code, formel), instruction-following
(Alpaca-like), un jeu "bruit" (réponses d'un autre modèle). Chaque adaptateur x force s. Entraînement : 5-15 min par adaptateur
sur 1 A100 en 7B (r=16, bf16, 300 pas) ; 1B : 2-4 min. Comme les LoRA des organisateurs nous sont inconnus, faire varier la nature
des adaptateurs pour que le classifieur apprenne "décalage de poids" et non "style X".
Option intéressante : (a) mesurer sur ref-04/07/09 ce que fait l'adaptateur (style ? domaine ?) ; (b) si c'est un style clair, essayer
de DISTILLER un adaptateur proche : fine-tuner nos LoRA sur des milliers de réponses de ref-07 (force 1,5, texte gratuit illimité) comme jeu
d'entraînement, pour obtenir un adaptateur "voisin" des organisateurs ; il ne servira que si les endpoints scorés utilisent la MÊME recette
(inconnu : les ref sont un modèle différent, [avis] la recette est probablement la même ou voisine).
### 4.5 Étiquetage et jeux
Train : bases/checkpoints/LoRA/seeds A. Validation : combinaisons jamais vues (nouveaux seeds de LoRA, nouvelle clé de filigrane, nouveaux
prompts système, nouveau 8-bit). Test d'honneur : les 9 ref (vraies étiquettes). Règle go/no-go : AUC paire sur validation synthétique ET
exactitude des ref doivent concorder.

## 5. Ce qui reste invariant et ce qui ne l'est pas

| Facteur | Effet sur le texte | Invariant ? / contre-mesure |
|---|---|---|
| Filigrane KGW/Unigram | Biais sur les tokens "verts" (delta), surtout aux positions à entropie moyenne/haute | Presque nul sur tokens où p_top1 > 0,9 : utiliser la classe "H < 0,3" comme canal invariant ; modéliser un biais token-spécifique (Unigram : liste fixe estimable en agrégeant ~25k tokens : fréquence observée/attendue sous c ; KGW h=1 : biais par token précédent, estimable pour les bigrams fréquents) puis le soustraire avant LLR (idée du "watermark stealing", Jovanović et al. 2024 [non vérifié, de mémoire]) |
| Température/top-p | Élargit/resserre : pente de lp+H par classe d'entropie | Estimer T par endpoint par régression sur (rang, lp) sous la base candidate (mélange de l'auto-calibration), puis comparer à T estimé ; ne pas confondre avec quantification car l'effet est monotone en H |
| Prompt système / persona | Style, longueur, premiers tokens | Ignorer 8-16 premiers tokens, regarder le surcoût de lp en moyenne ; préférer des prompts utilisateur qui imposent un format ou une tâche (continuation, calculs), pour que le contexte système pèse peu |
| Quantification | Perturbation diffuse des logits, bruit symétrique ; la distribution reste "centrée" sur la base | Vue par la corrélation du logit-différence avec les logits "tail" ; signature quasi-invariante au système et au filigrane car additive |
| LoRA | Changement dirigé (direction de style/connaissance) | Visible par un LLR dirigé (sous-espace) : projeter (lp_c - lp_base) sur ses composantes principales apprises sur les synthétiques |
| Longueur de réponse | Dépend des déguisements | Seulement un trait faible |

Principe : combiner (i) canaux invariants (tokens bas-entropie, premiers tokens ignorés), (ii) retrait de nuisance estimé endpoint par
endpoint, (iii) un classifieur entraîné sur synthétiques qui voit les deux canaux et apprend l'invariance de façon empirique.

## 6. Quels prompts maximisent la séparation par réponse (budget en réponses)

Critères : (1) réponse qui remplit ~256 tokens (sinon tokens gaspillés) ; (2) entropie moyenne modérée (0,5-2 nats/token) : ni
déterministe (signal nul) ni très haute (variance énorme et le filigrane domine) ; (3) peu sensible au prompt système/persona :
tâches impératives et format imposé ; (4) sensibilité aux poids : connaissances de queue (faits rares), arithmétique multi-chiffres,
codes, listes ; (5) diversité de domaines pour exciter les différentes couches LoRA (style, code, math, multilingue, refus).
Familles de prompts (idées, à construire sur 100) :
- Continuation de texte encyclopédique rare (préfixe de 150-300 tokens, "continue sans commentaire") : 25 prompts. Bon pour LLR global.
- Listes numérotées longues (liste de 60 faits/pays/dates) : 15. Fort signal sur les connaissances, filigrane faible (tokens contraints).
- Arithmétique, tables de calcul, conversion (suite de 40 calculs) : 15. Sensible à la quantification (erreurs numériques).
- Code (fonction courte + tests) : 10. Basse entropie : canal invariant.
- Récit libre court avec contraintes ("écris 10 phrases commençant par 'Le'") : 15. Haute entropie : à garder peu.
- Questions de refus/alignement/banner ("Qui es-tu ? Qui t'a créé ?") : 10. Pour famille et LoRA (identité).
- Multilingue (fr, de, es) : 10. Les LoRA touchent souvent la langue.
Sélection par apprentissage : sur un grand pool (500-1 000 prompts) générer les réponses des synthétiques (budget local illimité), calculer pour
chaque prompt l'information de Fisher "same vs different" (gain en AUC du classifieur par prompt, ou statistique de séparation par
token), garder les 100 meilleurs (glouton, avec contrainte de diversité). Le pool peut être testé aussi sur les ref réels (gratuit) pour
vérifier que le classement synthétique prédit l'ordre réel. Les 100 prompts sont les mêmes pour les 33 endpoints (alignement des features) ;
la réserve doit être planifiée : 60 prompts-réponses d'abord (étape 1), 40 pour les paires ambiguës (le même prompt rejoué reste
valable : la réponse est échantillonnée, donc une 2e réponse au même prompt est un nouvel échantillon).

## 7. Précisions attendues, heures GPU, heures d'ingénierie [avis, non mesurées]

| Cible | Précision paire attendue | Remarque |
|---|---|---|
| Famille | 97-100 % ; MCC_famille 0,95-1,0 | Si famille = OLMo de tailles différentes |
| Même modèle vs LoRA fort/nf4/int4 | 90-98 % | |
| Même modèle vs LoRA 0,7 | 80-95 % | |
| Même modèle vs LoRA 0,3 | 55-80 % | |
| Même modèle vs fp8/int8 | 50-70 % | proche du hasard si l'implémentation serveur diffère de l'émulation locale |
| MCC_modèle global | 0,5 à 0,85 | selon la part de variantes subtiles |

Effet d'une erreur sur MCC (cas ref : 5 S + 4 variantes, 36 paires, 10 paires "mêmes") : fp8 fusionné avec S : MCC=0,73 ; fp8 et LoRA 0,3
fusionnés avec S : 0,52 ; tout correct : 1,0 (calcul direct à partir de la formule MCC). Les erreurs de fusion coûtent donc ~0,25 MCC chacune
dans une petite famille.
Heures GPU (noeud de 4 A100) : téléchargement + chargement 0,5 h ; quantifs et GPTQ/AWQ 1 à 2 h ; LoRA 1 h ; génération synthétique 2-5 h ; scoring 1-3 h ;
classifieur/boosting sur CPU. Total : 6-12 h noeud, pouvant être étalé ; un seul noeud suffit si on réduit le zoo.
Heures d'ingénierie (2 personnes) : 2 h infrastructure (venv, vLLM, téléchargements) ; 3 h générateur synthétique + filigrane ; 2 h scoring ; 2 h classifieur
et calibrage ; 1 h intégration avec le script de soumission. Total ~10 h-personne, soit 5 h à deux en parallèle.

## 8. Modes d'échec et plans de secours

1. **Endpoints non-OLMo ou modèles fermés/ inconnus** (aucun candidat local ne colle) : les vecteurs endpoint x candidat deviennent des
   "descripteurs" par proxy (un seul LM de référence pour tous, mean lp/rang/entropie par classe). Les distances inter-endpoint restent
   utilisables (analogue à MET sans modèle de référence), précision paire dégradée mais famille quasi intacte (descripteurs de style + banners
   + LLMmap-like). Signal : si le meilleur mean lp de l'endpoint est très inférieur à celui des ref sous leur vrai modèle, déclencher ce plan.
2. **Versions/tailles non exactes** (ex. OLMo-2-7B vs 7B-0424) : rechercher parmi plusieurs révisions ; un décalage de checkpoint est beaucoup plus
   grand que la quantification, donc visible et sera pris pour un autre modèle (juste, car checkpoints différents = modèles différents).
3. **Émulation quantifiée ≠ serveur** : prendre le max sur plusieurs implémentations ; entraîner le classifieur sur un mélange d'implémentations ;
   sinon retomber sur une règle "écart à la base" au lieu de "écart au candidat fp8 exact".
4. **LoRA non reproductibles** : on ne les connaît pas ; on les détecte comme "écart à la base" (signature de direction), et on groupe les endpoints par
   proximité mutuelle plutôt que par identification. Deux LoRA de forces différentes d'un même adaptateur seront proches mais séparables par la norme
   de l'écart (utilise la force comme coordonnée radiale).
5. **Filigrane qui annule le signal** (delta fort) : basculer sur le canal bas-entropie et sur la soustraction de biais token-spécifique ; si inefficace,
   augmenter le poids des prompts à tokens contraints (code, listes).
6. **Dérive ref -> scorés** : les seuils calibrés sur ref (petit modèle) ne se transfèrent pas à un 7B/32B ; recalibrer sur synthétiques OLMo de la même taille.
7. **Sur-ajustement du score public (30 %)** : ne pas régler sur le classement public ; seuil fixé sur synthétique + ref, puis gelé.
8. **Coût de fautes irréversibles** : requêtes en vrai seulement après validation complète sur ref ; sauvegarder toutes les réponses brutes (jsonl).
9. **Clustering** : MCC pénalise symétriquement ; utiliser une partition par clustering hiérarchique avec seuil choisi pour maximiser le MCC attendu sur
   synthétiques à taille de famille comparable (rapport nombre de modèles/famille inconnu : tester le seuil sous plusieurs hypothèses 2-10 modèles par famille).

## 9. Plan de la nuit (heures 0 à 18 ; H0 = samedi ~17 h, H18 = dimanche ~11 h, échéance dimanche 12 h)

| Heures | Tâche | Go / no-go |
|---|---|---|
| H0-1 | Installer environnement JURECA (script setup), vérifier GPU, pip/uv : transformers, peft, bitsandbytes, vllm (optionnel). Interroger les 9 ref gratuitement : "Qui es-tu ?" x 10, 30 prompts de test (sans ep-*). Identifier le modèle ref et la famille probable des ep. | GO si on identifie le modèle ref (mean lp élevé sous un candidat). NO-GO famille OLMo => plan 8.1 dès H1 |
| H1-3 | Télécharger les poids candidats (login node), établir le zoo : base, instruct, bnb nf4/int8, fake-fp8. Valider perplexité. Écrire le scoreur (3.4) et tester sur les réponses des ref. | GO si les ref S sont classés à ~100 % sous le même candidat et que les vecteurs de T1..T4 sont séparés de S par ordre (T4>T3>T2) |
| H3-6 | Générateur de synthétiques (filigrane, prompts système, T) ; 200 endpoints pilotes ; premier classifieur sur LLR/rang. | GO si AUC validation > 0,85 sur variantes "fortes" ; sinon simplifier (retirer features) |
| H6-9 | Entraîner LoRA (K=8), générer 1 000+ endpoints, sélection de prompts (section 6). Dans le même temps, geler la liste de 100 prompts (relue à deux). | GO si la liste de prompts est gelée avant toute dépense |
| H9-10 | Étape 1 sur ep-* : 40 réponses/endpoint, 33 endpoints (1 320 requêtes ; à 4 en parallèle). Famille ; premières classes. | GO si les familles se regroupent clairement ; sinon plan secours |
| H10-13 | Scoring sur 33 endpoints, entraînement final, calibrage seuil sur synthétique + ref ; première soumission (calibration, pas de réglage sur public). | GO si la validation synthétique et ref concordent à +/- 0,1 de MCC |
| H13-16 | Étape 2 : 40 réponses supplémentaires sur les paires ambiguës ; recalcul ; deuxième soumission. 20 réponses gardées en réserve. | Arrêter si MCC ne bouge pas de plus de 0,03 : le reste n'est que du bruit |
| H16-18 | Soumission finale, vérification du format CSV avec submission.py et score_practice.py sur ref ; ne rien changer après H17. | Aucune nouvelle idée après H16 |
Rappel : une soumission par 30 min ; ne jamais lancer de nouveau code non testé au dernier moment.
Partage avec T1/T3 : T2 est peu coûteux en GPU ; ne pas bloquer un noeud complet plus de 3 h d'affilée pendant que T1/T3 en ont besoin ;
prévoir 1 GPU par job (--gres=gpu:1) pour les scorings et quantifs, 4 GPU seulement pour la génération synthétique.

## 10. Squelette de code (à exécuter plus tard, rien lancé)
- score.py : charge un candidat (bf16/bnb/fake-fp8, + LoRA mis à l'échelle), applique le template de chat, calcule lp, rang, entropie par token de réponse ; écrit
  un .npz par (endpoint, candidat). Mettre `torch.no_grad`, batch triés par longueur, `attn_implementation="sdpa"`.
- synth.py : paramètres (base, quant, lora, s, wm, sys, persona, T, top_p) -> jsonl de 100 réponses ; `--seed` ; log JSON des étiquettes.
- feats.py : agrégats 3.4, bootstrap sur réponses ; pairs.py : features de paires, classifieur (LightGBM/sklearn HistGradientBoosting), validation par groupes (groupe =
  couple base/LoRA, pour éviter la fuite).
- cluster.py : matrice de probabilités paire -> partition (average linkage + correction de transitivité), sortie CSV `endpoint,family,model`.

## Sources principales
LLMmap https://arxiv.org/abs/2407.15847 ; RUT https://arxiv.org/abs/2506.06975 ; MET https://arxiv.org/abs/2410.20247 ; Cai et al. https://arxiv.org/abs/2504.04715 ;
Idiosyncrasies https://arxiv.org/abs/2502.12150 ; UTF https://arxiv.org/pdf/2410.12318 ; MarkLLM https://github.com/THU-BPM/MarkLLM (https://arxiv.org/abs/2405.10051) ;
OLMo 3 https://huggingface.co/collections/allenai/olmo-3 ; LAPD https://arxiv.org/html/2604.16923 ; PhantomHunter https://arxiv.org/html/2506.15683v1 ;
One Token Is Enough https://arxiv.org/abs/2607.10252 ; FLIPS https://arxiv.org/pdf/2606.03330 ; OpenRouter quantization https://openrouter.ai/blog/insights/llm-quantization.
Non lus en entier : tous les arXiv ci-dessus ont été vus via résumés/snippets ; les chiffres de la section 2 sont mes estimations.
