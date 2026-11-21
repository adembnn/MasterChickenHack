# T2 : audit impitoyable des hypothèses non vérifiées de la méthode v3

But de ce document : lister **toutes** les hypothèses, implicites ou explicites, sur lesquelles repose
`t2-method-v3.md` (et les rapports qu'elle synthétise : #10 #11 #12 #13, plus `t2-fingerprinting.md`,
`t2-lora-calibration.md`), et dire pour chacune : (1) le risque concret si elle est fausse, (2) si elle est
testable gratuitement (local / lecture attentive / web), (3) sinon, le coût minimal en requêtes `ref-*` pour
la vérifier. Rien n'a été appelé sur `ep-*` pour écrire ce rapport. Relecture ligne à ligne de
`reports/sources/Model_Equality_Testing.txt` (texte officiel intégral, ci-après « l'énoncé ») et des 9
rapports `reports/titouan/t2-*.md`.

Classement : risque × probabilité d'être fausse, les plus dangereuses en premier. « Risque » = perte de MCC
potentielle si l'hypothèse est fausse ET qu'on ne l'a pas détectée avant de dépenser le budget `ep-*`.

---

## Rang 1 — Le bruit de fond mesuré (0,30-0,34) suppose un décodage glouton déterministe côté serveur, jamais confirmé

**L'hypothèse.** Tout le calibrage de seuil (`tau = 0,43`, §12 de v3, mesuré dans #12) part d'un « bruit de
serving entre copies du même modèle » = 0,30-0,34, mesuré sur le jumeau **en décodage glouton** (greedy,
déterministe, le seul bruit venant de la taille de lot/dtype). Si les vrais `ep-*` échantillonnent
réellement (sampling avec température > 0), le bruit entre deux réponses d'un **même** endpoint à un
**même** prompt serait bien supérieur à 0,30-0,34, et notre seuil serait beaucoup trop bas : des copies
identiques seraient massivement isolées à tort (FP sur la famille ET sur le modèle).

**Ce que dit l'énoncé, exactement.** Ligne 109-111 : « *Sampling settings such as temperature and seed are
ignored* ». Ça dit que **nos** paramètres client (temperature, seed) n'ont aucun effet — pas que le serveur
décode glouton. Ligne 28-29 : « *A deployment may vary in its decoding watermark, ... decoding strategy...* »
— **la stratégie de décodage est explicitement listée comme une dimension qui peut varier entre
déploiements**, et ligne 34-35 confirme qu'une différence de stratégie de décodage « ne change pas
l'identité du modèle ». Autrement dit, l'énoncé dit lui-même que deux endpoints du même modèle peuvent avoir
des décodages différents (l'un glouton, l'autre échantillonné) — ce qui est une variante bien plus violente
que le « bruit de serving » lot/dtype qu'on a mesuré. **Rien dans l'énoncé ne dit que le décodage par défaut
est glouton.** Le jumeau n'a jamais testé « même modèle, un greedy et l'autre sampling T=0.7 » : la table de
risques de #13 ne couvre que le filigrane, pas un décodage réellement stochastique sans filigrane.

**Risque si faux.** Le plus grave de toute la méthode : si une partie des 33 `ep-*` échantillonne
réellement, `tau=0,43` sous-protège contre les faux positifs famille/modèle sur CES endpoints précis — ils
ressembleraient à des « modèles différents » alors qu'ils sont identiques. #12 lui-même note (§7) qu'un
bruit réel à 0,42 au lieu de 0,34 (+0,08 seulement) fait déjà chuter le MCC de 0,97 à 0,85. Un sampling réel
sans graine fixée peut facilement produire un taux de divergence > 0,6-0,8 à T élevée sur des prompts à
haute entropie (c'est exactement ce qui est observé pour LoRA 0,3 = 0,82 dans le même tableau) — soit un
chevauchement presque total avec les variantes qu'on veut détecter. **Ça peut rendre le signal principal de
la méthode inutilisable sur un sous-ensemble inconnu des 33 endpoints, sans qu'aucun signal d'alerte actuel
(§2 de `t2-plan-deploiement-reel.md`) ne le détecte spécifiquement** — les 4 signaux d'alerte existants
couvrent la famille, la marge candidat, le format, et la variance globale, mais pas « deux réponses au même
prompt sur le même endpoint sont-elles identiques ».

**Testable gratuitement ?** Oui, en partie, sur `ref-*`, avant toute requête `ep-*`, et c'est **le test le
moins cher de tout ce document** : envoyer le **même prompt deux fois** au même `ref-xx` et comparer les
deux réponses littéralement. Si elles diffèrent déjà (hors watermark — repérable par position de première
divergence très précoce selon la règle de #13), c'est la preuve directe d'un décodage non-glouton côté
serveur, et le calibrage doit changer de nature (il faudrait alors mesurer le bruit intra-endpoint, pas
seulement inter-endpoint).

**Coût minimal en requêtes `ref-*`.** 2 requêtes par `ref-xx` (même prompt répété), sur les 9 `ref-*` = 18
requêtes, gratuites et hors budget `ep-*`. Idéalement sur 2-3 prompts différents par ref pour ne pas
conclure sur un seul coup de dé = ~40-50 requêtes `ref-*`. C'est le test à faire en toute première priorité,
avant même l'étape 0 actuelle de v3 qui ne teste que les **paires** de `ref-*` entre eux, jamais la
répétition du **même** prompt sur le **même** endpoint.

---

## Rang 2 — Le filigrane réel est forcément de type KGW/Unigram à fenêtre 1

**L'hypothèse.** `t2-watermark-robustness.md` (#13), seule source mesurée sur la bascule filigrane,
reproduit uniquement KGW (Kirchenbauer et al., fenêtre de hachage h=1, gamma=0,25). `t2-method-v3.md` §1
étape 1 généralise la règle de bascule (« première divergence médiane ≤ 4 tokens → filigrane probable, pas
poids différents ») à **tout** filigrane sans le dire explicitement. Mais l'énoncé ne précise aucun schéma
(ligne 28 dit juste « decoding watermark »). D'autres familles de filigrane existent et n'ont pas la même
signature de première divergence : un filigrane à fenêtre large (h > 1, contexte de plusieurs tokens), un
filigrane par liste verte fixe de type Unigram pur (déjà cité comme alternative dans `t2-fingerprinting.md`
§4.1 mais jamais mesuré dans #13), ou un filigrane de type « distribution-préservant » / échantillonnage
biaisé par graine cachée (Aaronson/Kuditipudi, SynthID) ont des profils de divergence et de détectabilité
différents — certains ne divergent PAS dès le premier token (ils ne biaisent la distribution que
statistiquement sur de longues séquences), ce qui casserait silencieusement la règle de bascule « ≤ 4
tokens » de v3 : on classerait alors un filigrane plus discret comme un écart de poids réel.

**Risque si faux.** Direct sur Rang 1 bis : des endpoints filigranés avec un schéma à divergence tardive
seraient lus comme « poids différents » (faux négatif sur le pooling de famille/modèle, car on isolerait
des endpoints identiques). Le texte même de l'énoncé laisse entendre (ligne 17, « *decoding watermark* »,
une seule mention, aucun détail) que le hackathon pourrait avoir choisi n'importe quel schéma standard — y
compris un schéma qui n'a pas la signature « divergence quasi immédiate » mesurée pour KGW seul.

**Testable gratuitement ?** Partiellement : chercher dans la littérature (web) la liste des schémas de
filigrane *open-source les plus répandus en 2025-2026 pour du texte LLM* et vérifier si tous partagent la
propriété « biais appliqué dès le premier token généré » (c'est le cas pour KGW, Unigram, et la plupart des
filigranes à biais logit — mais pas pour les filigranes « distribution-préservant » comme Aaronson/Gumbel ou
SynthID, qui ne biaisent pas chaque token individuellement de la même façon). On peut déjà, sans requête,
étendre `watermark_robust.py` (#13) pour simuler un filigrane Unigram pur et un filigrane de type Gumbel/
Aaronson sur le jumeau, et vérifier si la règle « ≤ 4 tokens » tient — c'est un test gratuit, pas encore
fait (§: « Prochaine expérience » de #13 ne mentionne que température+fenêtre large, pas un autre type de
schéma).

**Coût minimal `ref-*` si besoin de confirmer en réel.** Aucun test direct ne peut confirmer le *type exact*
de filigrane sans la clé — mais on peut mesurer, sur `ref-*`, la position de première divergence pour
**toutes** les paires `ref` présumées identiques (pas seulement celles qui ont l'air d'avoir un filigrane
actif) : si certaines paires montrent une divergence *progressive* plutôt qu'immédiate tout en étant
pourtant déguisées (vérifiable via `practice_answers.csv`, qui donne la vraie réponse), c'est le signe qu'un
filigrane à signature différente de KGW est en jeu. Coût : 0 requête supplémentaire, c'est une relecture des
données déjà prévues en étape 0 de v3, juste avec un œil différent (comparer aux étiquettes connues au lieu
de supposer le type de filigrane).

---

## Rang 3 — Le budget `ref-*` est séparé du budget `ep-*` (le pool de 100 réponses)

**L'hypothèse.** `t2-fingerprinting.md` §0 affirme : « *Les 9 ref sont gratuits et illimités* » ;
`t2-plan-deploiement-reel.md` répète partout « gratuit, illimité » pour justifier de tout calibrer sur `ref`
avant de toucher `ep`. C'est la prémisse de **toute** la méthode (étape 0 de v3 entière, les 9 rapports
`t2-*.md`, le calendrier §5 de `t2-plan-deploiement-reel.md`).

**Ce que dit l'énoncé, exactement.** Section « API Access and Query Budget » (ligne 100-112) : « *Each team
may collect up to 100 responses per endpoint for the whole competition* ». Cette phrase arrive **avant** la
section « Reference / Practice endpoints » et ne nomme que les « 33 endpoints » du paragraphe précédent
(« *We provide 33 LLM deployments... ep-01 to ep-33* », ligne 23-24). La section Reference (ligne 114-124)
dit : « *score_practice.py scores your practice answer locally, as often as you like* » — ceci confirme
l'absence de limite pour le **scoring local**, pas pour les **requêtes réseau** vers `ref-*`. Nulle part le
texte ne dit explicitement « les requêtes vers `ref-*` ne comptent pas dans votre budget » ou « le budget de
100/endpoint s'applique aussi aux `ref-*` ». C'est une lacune réelle du texte, exactement comme signalé dans
la tâche confiée à cette session.

**Risque si faux.** Si `ref-*` partage en réalité un pool avec `ep-*` (par exemple si le compteur de
requêtes est global par équipe et non par endpoint, ou si un cap caché existe sur `ref-*` lui aussi), **toute
la stratégie actuelle est auto-destructrice** : v3 prévoit de dépenser des centaines de requêtes `ref-*`
(calibration, synthétiques comparés aux ref, mining de prompts validé sur ref, etc. — cf. §0 à chaque étape
0 de chaque rapport) avant de toucher `ep-*`, en présumant que ça ne coûte rien. Si c'est faux, le budget
réel sur `ep-*` serait déjà partiellement ou totalement épuisé avant la première requête `ep-*` volontaire —
un échec total et silencieux (aucun message d'erreur ne préviendrait avant la requête `ep-*` refusée).

**Testable gratuitement ?** Oui, en lisant le code fourni (`query_reference.py`, `task_template.py`, les
scripts huggingface du dépôt `SprintML/model_equality` et `SprintML/hackathon`) : ces scripts, une fois
téléchargés (étape déjà prévue, aucune requête réseau nécessaire pour juste les lire), révèlent très
probablement, dans leur code, si le compteur de requêtes est partagé ou non (recherche d'un champ `budget`,
`quota`, ou d'un endpoint séparé pour `ref` vs `ep` dans l'implémentation du client). `submissions-ops.md`
ligne 114 note déjà : « *Vérifier d'abord que le champ `budget` des `ref-*` est utilisable par la formule
`limit - used` du template* » — cette vérification est déjà identifiée comme nécessaire mais **rien dans les
rapports ne confirme qu'elle a été faite**. C'est une lecture de code, gratuite, zéro requête.

**Coût minimal en requêtes si la lecture de code ne tranche pas.** 1 seule requête `ref-01` suffit : envoyer
une requête, puis immédiatement consulter le champ retourné par l'API (`get_endpoint_info` ou équivalent du
`task_template.py`) pour voir si le compteur `used`/`budget` de `ep-01` (un endpoint qu'on n'a jamais
touché) a changé. Si non : les pools sont bien séparés. Coût : 1 requête `ref-01`, 0 requête `ep-*`.

---

## Rang 4 — La structure des 33 endpoints (nombre de familles, variantes par famille) ressemble à ce qu'on simule

**L'hypothèse.** Le MCC attendu (0,93-0,97, #12) est calculé sur 6 structures simulées (S1 à S6, voir
`t2-mcc-simulation.md` §2) choisies par les auteurs du rapport, pas par les organisateurs. Rien dans
l'énoncé ne donne le nombre de familles réelles ni la répartition modèle/variantes. `t2-method-v3.md` ne
teste **aucune** hypothèse sur la structure elle-même (nombre de familles 2 à 10, taille de chaque famille,
proportion d'endpoints « identiques » vs « variantes ») au-delà de ce qui est simulé.

**Ce que dit l'énoncé.** Ligne 22-26 : « *We provide 33 LLM deployments... Some endpoints serve the same
model under different deployment settings, others serve different models.* » Aucun chiffre. L'exemple donné
(ligne 42-45, « *two deployments of the same OLMo base model* ») ne dit même pas que le modèle de base réel
est OLMo pour les `ep-*` — c'est un exemple pédagogique générique, voir Rang 5.

**Risque si faux.** Modéré à élevé selon le cas. #12 §8 note déjà lui-même : « *Famille supposée juste (10 %
du score)... Une erreur de famille se propage ici en FP/FN non comptés* ». Si la vraie structure a beaucoup
plus de petites familles de taille 2-3 (au lieu des familles de 5-11 simulées), le coût d'un FP/FN change
fortement (une fusion fausse dans un cluster de taille 2 est bien moins pénalisante qu'une fusion fausse
dans un cluster de taille 11, mais l'inverse est vrai pour un FN) — la règle de prudence « isoler par défaut
» de l'étape §3.6 de `t2-plan-deploiement-reel.md` n'est optimale que sous certaines structures, pas toutes.
Si beaucoup de familles ont une seule paire de variantes extrêmement proches (ex. deux int8 différents de la
même base), le seuil tau=0,43 calibré sur les structures simulées pourrait rater des cas limites non
représentés dans S1-S6.

**Testable gratuitement ?** Partiellement, par calcul combinatoire pur (zéro requête) : 528 paires au total
sur 33 endpoints est cohérent avec C(33,2)=528 (confirmé, ligne 53 de l'énoncé). Mais la répartition réelle
famille/modèle n'est accessible qu'en observant les vraies réponses. On ne peut PAS la déduire du texte seul
— ceci est une vraie inconnue structurelle, pas une négligence de lecture.

**Coût minimal `ref-*`.** Zéro coût supplémentaire : les 9 `ref-*` SONT déjà un échantillon réduit de la
même structure de déguisements (« *the same kinds of disguises and traps* », ligne 118-119), avec les vraies
étiquettes dans `practice_answers.csv`. Il faut simplement généraliser la simulation de #12 à couvrir une
plage plus large de structures (y compris des familles de taille 2, des familles asymétriques avec 1 modèle
dominant + 1 variante isolée) **avant** de figer tau sur `ep-*`, en vérifiant que tau=0,43 reste dans la
zone plate (0,40-0,45) pour ces structures aussi. C'est un calcul, pas une requête.

---

## Rang 5 — L'identité réelle du modèle derrière les `ep-*` est OLMo (ou en tout cas un modèle que le jumeau local approxime bien)

**L'hypothèse.** `t2-fingerprinting.md` §3.1 : « *Hypothèse : les familles sont OLMo (l'énoncé cite "OLMo
base model" dans l'exemple)* », marqué `[avis]` par l'auteur lui-même. Toute la méthode NLL croisée (#13),
le zoo de candidats (§3 de `t2-fingerprinting.md`), et la calibration LoRA (#14) reposent sur OLMo-2-1B/7B
comme jumeau. C'est exactement l'hypothèse que la session précédente (`t2-identite-modele.md`, référencée
dans la tâche mais absente du dépôt à ce jour — aucun fichier `*identite*` trouvé, aucune branche
`t2/identite-modele` trouvée non plus) devait creuser.

**Ce que dit l'énoncé, exactement.** Ligne 42-45 : « *For example, two deployments of the same OLMo base
model...* ». Le mot « example » (« *Pour exemple* » en français) précède directement : c'est un exemple
pédagogique pour illustrer la définition d'identité de modèle (même checkpoint + même quantif + même LoRA =
même modèle), **pas une affirmation que les 33 `ep-*` réels sont basés sur OLMo**. Les auteurs du challenge
(Xun Wang, Bihe Zhao et al., ligne 3-5) ont publié un papier cité en référence, « AgentProv », mais aucun
luxe de détail sur le modèle choisi pour ce challenge précis dans le texte fourni.

**Risque si faux.** Très élevé *sur la partie modèle* (90 % du score), mais déjà anticipé et avec un plan de
secours chiffré : `t2-fingerprinting.md` §8.1 et `t2-plan-deploiement-reel.md` §4 donnent un plan de repli
(clustering générique sans jumeau) avec un MCC estimé dégradé mais non nul (0,45 au lieu de 0,73 en ordre de
grandeur, chiffres `[avis]` non mesurés). Le risque principal n'est donc pas « score nul » mais « perte de
temps GPU et humain à construire un zoo de candidats OLMo inutile si le vrai modèle est Llama/Qwen/Mistral/
Gemma », et surtout « fausse confiance dans la calibration LoRA/int8 » si on applique les seuils mesurés sur
OLMo-2-1B à un modèle de nature très différente (taille d'embedding, tokenizer, sensibilité à la
quantification — MET cité dans `t2-fingerprinting.md` note déjà un facteur ×100 de sensibilité entre
familles de modèles pour la même perturbation).

**Testable gratuitement ?** Oui, en grande partie, et c'est déjà en cours (plan §9 heure 0-1 de
`t2-fingerprinting.md`) : interroger `ref-*` avec des questions d'identité (« Qui es-tu ? », « Quel est ton
tokenizer ? », demander de répéter des tokens rares spécifiques à un tokenizer donné) et comparer la
vraisemblance sous plusieurs familles de modèles locales (pas seulement OLMo — il faut **élargir le zoo**
pour inclure au moins un représentant de chaque grande famille ouverte de taille comparable : Llama-3.2-1B/
3B, Qwen2.5-1.5B/3B, Gemma-2-2B, SmolLM2, Mistral-7B — actuellement `t2-fingerprinting.md` §3.1 ne mentionne
QUE OLMo dans le zoo prévu, ce qui biaise la détection vers OLMo même si c'est faux : un argmax sur un zoo
qui ne contient qu'OLMo trouvera toujours « OLMo » comme meilleur candidat, même si le vrai modèle est
ailleurs). C'est le point le plus urgent à corriger dans le plan actuel, et c'est gratuit : ça ne demande que
d'élargir la liste de candidats locaux téléchargés, pas de requêtes `ref-*` supplémentaires.

**Coût minimal `ref-*`.** Déjà couvert par la Ronde F (16 réponses/ref, déjà prévue gratuitement) — il n'y a
aucune raison de limiter le budget `ref-*` disponible puisque son statut « illimité » reste l'hypothèse
dominante tant que Rang 3 n'a pas inversé la conclusion. Donc coût additionnel réel : 0, à condition
d'élargir le zoo de candidats *avant* de lancer l'étape 0.

---

## Rang 6 — Les LoRA utilisent une mise à l'échelle alpha/r identique à celle supposée (`scaling = alpha/r`, multipliable par un facteur `s`)

**L'hypothèse.** `t2-fingerprinting.md` §3.3 et `t2-lora-calibration.md` §1 définissent la « force LoRA »
comme un facteur `s` multipliant `module.scaling["default"]` (= alpha/r dans l'implémentation standard
PEFT), calibré sur des LoRA **synthétiques entraînés par l'équipe**, pas sur les vrais adaptateurs des
organisateurs (jamais vus). La courbe `identical_frac(s)` de #14 est mesurée sur CES LoRA maison (r=8,
alpha=16, q/v_proj), avec l'hypothèse implicite que l'effet d'un « LoRA de force 0,3/0,7/1,0/1,5 » des
organisateurs (termes utilisés dans l'exemple `ref-09`/`ref-04`/`ref-07` de la tâche) correspond au même
mécanisme de mise à l'échelle.

**Ce que dit l'énoncé.** L'énoncé ne mentionne même pas le mot « LoRA » explicitement avec un paramètre de
force — ligne 28-29 dit juste « *or LoRA adapter* », ligne 37 dit « *a version with a different LoRA
adapter, is a different model* ». La notion de « force » (0.3/0.7/1.0/1.5) apparaît seulement dans les
rapports de l'équipe, probablement héritée des noms des `ref-*` eux-mêmes (ou d'une convention vue sur le
dépôt HuggingFace `SprintML/model_equality`, non vérifié dans ce rapport). **Rien ne garantit que les
organisateurs utilisent le même mécanisme `scaling *= s`** : un LoRA appliqué avec un rang différent (r=16
au lieu de r=8), des modules différents (tous les projecteurs au lieu de q/v seulement), ou un mécanisme de
mise à l'échelle différent (DoRA, rsLoRA qui change la normalisation en `1/sqrt(r)` au lieu de `1/r`) aurait
une courbe divergence-vs-force structurellement différente de celle mesurée dans #14.

**Risque si faux.** Moyen. La courbe `identical_frac(s)` de #14 sert surtout à qualifier la **force relative**
entre deux LoRA (pour les ranger l'un par rapport à l'autre), pas à identifier un seuil absolu critique pour
la décision binaire « même modèle ou non » (c'est tau=0,43 qui fait ce travail, calibré séparément sur le
bruit, pas sur la forme exacte de la courbe LoRA). Le risque principal est donc plus fin : si la vraie forme
de la courbe LoRA des organisateurs est très différente (ex. plus abrupte, ou avec un plancher différent de
0,68-0,76), l'estimateur inverse `ŝ = -s0·ln(frac/f0)` (§3 de #14) donnerait une force LoRA estimée fausse —
mais ceci n'affecte PAS directement le MCC, seulement l'interprétation qualitative, sauf si cette estimation
est utilisée pour fusionner à tort deux endpoints en pensant qu'ils ont « la même force donc c'est le même
LoRA » (deux adaptateurs LoRA différents entraînés à mettre la même force de divergence restent deux
**modèles différents** selon la règle de l'énoncé — ligne 37). Il faut vérifier que la méthode ne fait
JAMAIS cette inférence (identité de force ⇒ identité de LoRA) : à relire dans le pipeline de décision final.

**Testable gratuitement ?** Oui, partiellement : vérifier sur le dépôt `SprintML/model_equality` (déjà prévu
au téléchargement, lecture de code gratuite) si des détails sur la génération des `ref-*`/`ep-*` (scripts de
setup, README, noms de fichiers) révèlent le mécanisme exact utilisé. Sinon, mesurer sur le jumeau avec
PLUSIEURS mécanismes de mise à l'échelle (scaling classique vs rsLoRA vs DoRA) pour voir si la courbe change
de forme qualitativement (ceci n'a pas encore été fait dans #14, qui ne teste qu'un seul mécanisme).

**Coût minimal `ref-*`.** Aucun test direct sur `ref-*` ne peut révéler le mécanisme interne d'un LoRA en
boîte noire (texte seul). Le seul signal disponible est indirect : comparer la forme de la courbe
divergence-vs-réponse des `ref-*` connus comme LoRA (ref-04, ref-07, ref-09 selon `practice_answers.csv`) à
la famille de courbes obtenues avec différents mécanismes locaux, et vérifier laquelle colle le mieux
(classification de forme de courbe, pas de nouvelle requête : les réponses `ref-*` de la Ronde F suffisent).

---

## Rang 7 — La quantification réelle est l'une des trois testées (int8 / fp8 / nf4), pas GPTQ/AWQ/GGUF ou une variante serveur (vLLM W8A8 dynamique)

**L'hypothèse.** `t2-fingerprinting.md` §3.1 et #12/#13 ne testent que int8 (bnb LLM.int8), fp8_e4m3/e5m2
(émulé en "fake quant", l'A100 n'ayant pas de cœurs FP8 matériels — admis explicitement en §3.2 de
`t2-fingerprinting.md`), et nf4 (bnb 4-bit). GPTQ, AWQ et GGUF (Q4_K_M, Q5_K_M, etc.) sont mentionnés comme
candidats à construire en §3.1 mais **jamais mesurés** dans aucun rapport chiffré (#12, #13, #14) — seuls
int8/fp8/nf4 ont une courbe de divergence mesurée.

**Ce que dit l'énoncé.** Ligne 28-29 dit juste « *quantization configuration* », aucune précision sur le
format. L'énoncé ne favorise aucun format particulier.

**Risque si faux.** Potentiellement élevé mais partiellement amorti : GPTQ, AWQ et GGUF sont, comme nf4,
des quantifications assez agressives (4 bits en général) qui divergent probablement **plus** que le bruit
de fond — la table mesurée montre déjà que nf4 (4 bits) a identical_frac=0,983 (quasi toujours détecté). Un
format 4-bit différent (GPTQ-4bit, AWQ-4bit) perturbe les poids d'une manière différente (calibration par
activation vs RTN direct) mais avec un ordre de grandeur de perturbation comparable à nf4 — donc
**probablement** aussi facilement détecté, mais ceci reste une extrapolation non mesurée, pas un fait
chiffré. Le vrai risque est plus subtil : si le serveur utilise une quantification W8A8 dynamique (comme
vLLM fp8 avec échelle par token, déjà noté comme un biais possible en §3.2 de `t2-fingerprinting.md`
lui-même : « *Pour W8A8-fp8 (vLLM), ajouter un arrondi fp8 des activations par hook (échelle dynamique par
token). Faire les DEUX variantes et noter le max* » — note que cette double variante n'a en réalité **pas**
été mesurée séparément dans les tables chiffrées de #12/#13, qui ne donnent qu'un seul chiffre fp8 par
schéma, sans dire si c'est poids-seul ou poids+activations).

**Testable gratuitement ?** Oui, entièrement en local (zéro requête) : ajouter GPTQ-4bit/AWQ-4bit/GGUF
Q4_K_M au script de calibration existant (`twin/calibration.py`, déjà construit pour les LoRA, extensible
aux quantifs) et mesurer leur `identical_frac` exactement comme pour int8/fp8/nf4 dans #12. Coût : du temps
GPU (calibration GPTQ/AWQ ~30-60 min par modèle, déjà noté dans `t2-fingerprinting.md` §3.1), zéro requête
réseau. C'est une mesure manquante qu'il est facile et gratuit de combler avant de dépenser le budget réel.

**Coût minimal `ref-*`.** Zéro — cette hypothèse se teste entièrement en local. Les `ref-*` ne peuvent de
toute façon pas révéler le format exact de quantification (boîte noire texte seul), seulement si le niveau
de divergence mesuré colle à l'une des courbes locales (ce qui est déjà le plan actuel).

---

## Rang 8 — Le serveur n'ajoute aucune troncature, filtrage ou post-traitement supplémentaire au texte retourné

**L'hypothèse.** Implicite partout : toutes les statistiques (comparaison exacte, NLL croisée, position de
première divergence) supposent que le texte reçu est le texte brut généré par le modèle, token pour token,
sans filtrage de contenu, sans troncature anticipée (modération), sans normalisation (espaces, Unicode), et
sans reformatage (ex. ajout d'un préfixe "Voici la réponse :" par un wrapper serveur).

**Ce que dit l'énoncé.** Rien d'explicite sur un filtrage de contenu. Ligne 110-112 dit seulement : « *At
most 4 requests at a time per team... Invalid requests are rejected and not counted against your budget* »
— ceci parle du rejet de **requêtes invalides**, pas d'un filtrage du **contenu généré**. Aucune garantie
donnée sur la fidélité du texte retourné au texte brut du modèle.

**Risque si faux.** Modéré. `t2-plan-deploiement-reel.md` §2 anticipe déjà une partie de ce risque avec le
signal d'alerte n°3 (« taux de refus, taux de troncature avant 256 tokens, taux de réponse vide, préambule
inattendu ») — donc ce n'est pas un angle mort complet, mais le signal d'alerte compare au comportement sur
`ref-*`, qui pourrait avoir **le même** post-traitement serveur que `ep-*` (ce qui le rendrait invisible par
comparaison différentielle) ou un post-traitement **différent** (ce qui fausserait la comparaison en sens
inverse). Si un filtrage de sécurité/modération tronque spécifiquement certains contenus (ex. tout ce qui
ressemble à des informations personnelles, ou certains mots), cela introduirait un biais systématique
corrélé au contenu du prompt plutôt qu'à l'identité du modèle — un faux signal de "modèle différent" sur des
prompts qui déclenchent le filtre pour un endpoint mais pas pour l'endpoint voisin (même modèle, mais
filtrage asymétrique selon la configuration de sécurité du provider).

**Testable gratuitement ?** Partiellement. On ne peut pas lire le code serveur. Mais on peut, sur `ref-*`,
chercher spécifiquement des signatures de post-traitement : troncature systématique à une longueur ronde
suspecte (pas exactement 256 tokens mais un multiple ou une valeur fixe plus basse), caractères de
remplacement Unicode suspects, préfixes/suffixes constants à travers tous les prompts d'un même endpoint
indépendamment du contenu. C'est une analyse des données déjà collectées en Ronde F, zéro coût
supplémentaire.

**Coût minimal `ref-*`.** 0 requête additionnelle : à vérifier dans l'analyse de la Ronde F existante, en
ajoutant explicitement cette catégorie de vérification (elle n'est pas listée nommément dans les 4 signaux
d'alerte actuels, qui parlent de « taux de troncature » global mais pas de « troncature à une longueur
suspecte constante » ou de « caractères de remplacement »).

---

## Rang 9 — Le nombre de prompts/réponses « gratuits » restants sur `ref-*` n'est pas lui-même soumis à un throttling de débit qui ralentirait la calibration (contrainte temps, pas budget)

**L'hypothèse.** Implicite dans tous les calendriers (`t2-plan-deploiement-reel.md` §5) : on suppose qu'on
peut envoyer un grand nombre de requêtes `ref-*` (synthétiques comparés, mining de prompts, calibration de
force LoRA via `ref-04/07/09`) dans la fenêtre de temps disponible (H0-H3 du plan), sans contrainte de débit
autre que la limite de 4 requêtes simultanées par équipe (ligne 108 de l'énoncé, qui s'applique à **toutes**
les requêtes de l'équipe, `ref` et `ep` confondus si c'est un cap global réseau plutôt qu'un cap par type).

**Ce que dit l'énoncé.** Ligne 108 : « *At most 4 requests at a time per team.* » — cette ligne apparaît
dans la section « API Access and Query Budget » qui précède la section Reference, et rien n'indique si elle
s'applique séparément à `ref-*` et `ep-*` ou globalement à l'équipe sur l'ensemble des deux types
d'endpoints.

**Risque si faux.** Faible à modéré : un simple risque de temps perdu (files d'attente, limite de débit
partagée entre tous les usages `ref`+`ep` de l'équipe, et donc moins de parallélisme disponible pour la
vraie campagne `ep-*` si `ref-*` est lancé en même temps) plutôt qu'un risque de score. C'est surtout un
risque de planification/coordination d'équipe (qui utilise les 4 slots à quel moment), déjà en partie couvert
par la règle « un seul propriétaire T2, annonce sur le bus » de `t2-plan-deploiement-reel.md` §6.

**Testable gratuitement ?** Oui, trivialement, en envoyant 5 requêtes `ref-*` simultanées et en observant si
la 5e est rejetée alors qu'aucune requête `ep-*` n'est en cours — révèle si la limite de 4 est un cap par
type d'endpoint ou global équipe. Coût : quasi nul (quelques requêtes `ref-*`, qui sont déjà présumées
gratuites — mais voir Rang 3 si cette présomption tombe).

**Coût minimal `ref-*`.** 5 requêtes simultanées, une seule fois, pour observer le comportement du
rate-limiter. Négligeable dans tous les scénarios.

---

## Récapitulatif : tableau de priorité

| Rang | Hypothèse | Risque | Testable gratuitement | Coût réel minimal |
|---|---|---|---|---|
| 1 | Décodage serveur glouton (bruit 0,30-0,34 valide pour tous les ep) | **Très élevé**, invisible aux signaux d'alerte actuels | Oui (répéter un prompt) | ~40-50 req. `ref-*` |
| 2 | Filigrane = KGW/Unigram, divergence toujours immédiate | Élevé | Oui (simulation locale + relecture `practice_answers.csv`) | 0 req. supplémentaire |
| 3 | Budget `ref-*` séparé de `ep-*` | **Très élevé si faux** (auto-sabotage silencieux) | Oui (lecture de code) | 1 req. `ref-01` pour confirmer |
| 4 | Structure des familles proche des simulations S1-S6 | Modéré | Oui (calcul, généraliser la simulation) | 0 req. supplémentaire |
| 5 | Modèle réel = OLMo | Élevé sur la partie modèle, amorti par plan de secours | Oui (élargir le zoo local, interroger `ref-*`) | 0 req. supplémentaire (Ronde F déjà prévue) |
| 6 | Mécanisme LoRA alpha/r identique à celui supposé | Modéré | Partiellement (tester plusieurs mécanismes en local) | 0 req. direct possible |
| 7 | Quantification ∈ {int8, fp8, nf4} | Modéré, probablement amorti (formats 4-bit se ressemblent) | Oui (étendre la calibration locale) | 0 req. supplémentaire |
| 8 | Pas de troncature/filtrage serveur supplémentaire | Modéré | Partiellement (relire les données Ronde F) | 0 req. supplémentaire |
| 9 | Limite de 4 requêtes : par type ou globale équipe | Faible (planification, pas score) | Oui | 5 req. `ref-*` |

---

## Recommandation : les hypothèses CRITIQUES à vérifier avant tout budget `ep-*`, et comment, au moindre coût

Avant la première requête `ep-*`, dans cet ordre (chaque étape bloque la suivante si elle échoue) :

1. **Rang 3 (budget séparé).** Lire le code de `query_reference.py`/`task_template.py` téléchargé depuis
   `SprintML/hackathon` et `SprintML/model_equality` pour confirmer que le compteur `budget`/`used` est bien
   isolé par endpoint et que `ref-*` n'y touche pas. Si le code ne tranche pas : 1 requête `ref-01` +
   vérification que le compteur de `ep-01` n'a pas bougé. **Coût : 0-1 requête.** C'est la vérification la
   moins chère et celle dont l'échec serait le plus dévastateur (elle invaliderait rétroactivement toute
   la stratégie « ref illimité » déjà en cours d'exécution).

2. **Rang 1 (décodage glouton vs sampling réel).** Envoyer le même prompt deux fois à chacun des 9 `ref-*`
   (idéalement sur 2-3 prompts par ref pour robustesse). Comparer les deux réponses littéralement. Si
   identiques (hors signature filigrane connue) pour tous les `ref` : le bruit mesuré sur le jumeau est
   probablement transférable, la méthode v3 reste valide. Si certains diffèrent sans signature filigrane
   typique (divergence tardive, pas immédiate) : recalibrer tau en mesurant le bruit **intra**-endpoint
   avant de fixer un seuil **inter**-endpoint. **Coût : ~20-50 requêtes `ref-*`.**

3. **Rang 5 (identité du modèle, zoo élargi).** Avant de lancer l'étape 0 de v3, élargir le zoo de candidats
   locaux pour inclure au moins Llama-3.2-1B/3B, Qwen2.5-1.5B/3B, Gemma-2-2B en plus d'OLMo-2. Recalculer la
   vraisemblance des réponses déjà prévues en Ronde F (16 réponses/ref) sous ce zoo élargi, pas seulement
   sous OLMo. **Coût : 0 requête supplémentaire** (réutilise la Ronde F déjà budgétée), mais du temps GPU
   pour télécharger 3-4 modèles de plus (quelques Go chacun, faisable sur le nœud de login).

4. **Rang 2 (type de filigrane).** Étendre `watermark_robust.py` pour simuler, en plus de KGW, un schéma
   Unigram pur et un schéma distribution-préservant (Gumbel/Aaronson), et vérifier si la règle « divergence
   ≤ 4 tokens ⇒ filigrane » tient pour les trois. **Coût : 0 requête**, uniquement du calcul GPU local
   (quelques minutes, même ordre de grandeur que le job `t2wm` déjà exécuté en 162 s).

5. **Rang 7 (quantifications non testées).** Ajouter GPTQ-4bit, AWQ-4bit et au moins une variante GGUF
   Q4_K_M au script de calibration existant (`twin/calibration.py`), mesurer leur `identical_frac` comme
   pour int8/fp8/nf4. **Coût : 0 requête**, du temps GPU pour la calibration (30-60 min par format).

Les points 3, 4 et 5 ne coûtent **aucune requête** (local uniquement) et devraient être faits en parallèle
du reste du plan v3, sans retarder l'étape 0 réelle. Les points 1 et 2 nécessitent un petit nombre de
requêtes `ref-*` (~60-100 au total), négligeables si Rang 3 confirme que `ref-*` est bien hors budget — mais
c'est précisément *pourquoi* Rang 3 doit être vérifié en tout premier : s'il s'avère faux, les points 1 et 2
doivent être réduits au strict minimum (quelques requêtes, pas 60-100).

**En une phrase** : la méthode v3 est statistiquement solide sur ce qu'elle a mesuré, mais elle repose sur
trois piliers jamais vérifiés sur le vrai système — le pool de budget `ref`/`ep`, le caractère glouton du
décodage serveur, et l'identité OLMo du modèle — dont au moins le premier peut être vérifié avec zéro ou une
requête, et les deux autres avec moins de 100 requêtes `ref-*` au total, bien avant de risquer le moindre
`ep-*`.
