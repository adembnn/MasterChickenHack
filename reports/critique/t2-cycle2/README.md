# T2 · Cycle 2 : bloc 1 = 5 questions ouvertes longues × 5 tirages (25 réponses par ep, identiques pour les 33)

**Décision débloquée** : quoi dépenser sur les 75 réponses restantes pour permettre les tests entre guichets (étape D de
`reports/critique/t2-tour-severe.md`). Zéro requête. Jumeaux Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B (h200c).

## Livrable
`probes_v4.py` : copie de `tracks/t2-model-equality/probes_v4.py` (branche `t2/collecte-25`, md5 identique sur h200c et
dans la copie locale) où **seul le bloc 1 change** (emplacements 25 à 49, mêmes numéros de `slot`) ; les blocs 0, 2 et 3
sont identiques à l'original (vérifié par comparaison de listes). `plan()` renvoie maintenant le plan du cycle 2 ; l'ancien
reste appelable par `plan_v4_initial()`. Installation : `cp probes_v4.py tracks/t2-model-equality/probes_v4.py`, puis
`python3 collecte_reelle.py --targets ep --block 1` (compatible tel quel, `kind` = `open`).
Les 5 questions, chacune suivie de « Answer in about 250 words. », tirée 5 fois (tirage r en boucle externe : une
interruption laisse le même nombre de tirages à chaque question) :
| id | domaine | question |
|---|---|---|
| C101 | récit | Write a short story about two strangers stuck in a lift during a power cut. |
| C102 | science | Describe how a black hole forms and what happens near its event horizon. |
| C103 | liste | List and explain the main causes of the French Revolution. |
| C104 | conseil | How should a student prepare for a difficult exam in two weeks? |
| C105 | code | Describe how a REST API differs from a GraphQL API, with a short code example of each. |
Aucun préfixe imposé (le serveur ouvre un nouveau tour, voir `t2-prefill/README.md`), aucun `max_tokens=1`.

## Dry-run passé
`collecte_reelle.py --targets ep --block 1 --dry-run` sur une copie de la branche `t2/collecte-25` + ce `probes_v4.py` :
825 réponses simulées (33 × 25), 0 échec, 5 probe_id × 5 tirages par guichet, 25 `slot` distincts par guichet, aucun réseau.
Idem `--targets ref` (225). Les deux dry-runs écrivent dans un dossier de travail à part, pas dans `tracks/`.

## Comment les 5 questions ont été choisies (mesuré, 0 requête)
Banque de 20 candidates (5 domaines × 4). Pour chacune et pour chaque jumeau : 4 réponses de base tirées à T = 0,75 (256
tokens), relues par la base, par le fp8 e4m3 per-tensor (W8 et W8A8) et par la base sous 3 prompts système (confondants).
J = divergence de Jeffreys moyenne par token (`qscan.py`, `bank.py`). Un candidat par domaine, le meilleur sur les 3
jumeaux :
- J(W8A8) par token : récit 0,033-0,034 ; science (trou noir) 0,033 ; conseil (examen) 0,028 ; liste (Révolution) 0,028 ;
  code (REST/GraphQL) 0,026. Les exercices de code purs sont les plus bas (0,019 à 0,021) : un seul, avec un exemple.
- **Gain de la sélection modeste** : +10 % de J par rapport à la moyenne de la banque. Le vrai gain vient de la longueur
  (256 tokens contre ~150 en moyenne au cycle 1) et du fait que les 33 guichets reçoivent les mêmes prompts.
- LoRA : non mesurable sans l'adaptateur. Seul indice, sur les 9 ref réels (SmolLM2-360M, ~21 prompts, 1 à 9 par type) :
  l'écart de NLL de ref-07 aux copies est le plus fort sur les questions « Explain… » (+0,50 nat/token, n = 9) et les listes
  (+0,32), ref-04 aussi (+0,09 et +0,11). D'où 1 liste et 1 science dans le choix ; le récit et le code ne sont pas testés
  sur LoRA, ils servent de diversité.
- Écart prompt système / fp8 : sur Qwen et Llama le prompt Acme déplace la distribution autant que le fp8 (min J_fp8/J_Acme
  de 0,16 à 0,54 selon la question) : ne pas lire le fp8 sans retirer le prompt caché d'abord (étape B d'Adem).

## Gain de puissance attendu, 50 réponses cumulées contre 25 (`power.out`)
z_oracle = √(N·J)/2 : lecteur exact du fp8 contre la base, pas de déguisement, N = tokens, J = divergence par token.
C1 = les 23 réponses de lecture du bloc 0 (21 questions + 2 répétitions) ; C2 = les 25 nouvelles.
| famille | tokens/rép. C1 | J/token C1 | J/token C2 | z oracle : C1 / C2 seul / cumulé | gain cumulé | gain si 25 de plus du type C1 |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | 155 | 0,039 | 0,030 | 5,9 / 6,9 / 9,1 | ×1,54 | ×1,41 |
| Qwen2.5-1.5B | 144 | 0,032 | 0,030 | 5,1 / 6,9 / 8,6 | ×1,68 | ×1,41 |
| Llama-3.2-1B | 165 | 0,031 | 0,030 | 5,4 / 6,9 / 8,8 | ×1,62 | ×1,41 |
**Lecture** : à 50 réponses cumulées le z gagne ×1,5 à ×1,7 (contre ×1,41 pour 25 réponses de même nature). Ce sont des
bornes hautes : l'écart avec les z mesurés au bloc 0 (3,4 à 6,4 pour les fp8 trouvés) montre qu'un z réel vaut
environ la moitié à la totalité de l'oracle. Pas de gain de J par token (0,030 contre 0,031 à 0,039) : tout vient de la
longueur. Le test **entre guichets** (MMD, vraisemblance croisée) gagne en plus la comparabilité : 5 tirages par question
pour les 33 guichets, ce que le cycle 1 (21 questions distinctes × 1 tirage) ne permet pas.

## Réserves
1. z oracle = hypothèse fp8 exacte et aucun déguisement ; une LoRA, une persona ou un filigrane réduisent le z réel.
2. 5 questions × 5 tirages : seulement 5 contextes différents. Si une LoRA ne touche pas ces 5 domaines, elle passe
   inaperçue ; le cycle 1 en avait 21.
3. Les réponses de 256 tokens sont tronquées (consigne 250 mots ≈ 330 tokens) : on ne voit jamais la fin ; sans
   conséquence pour la relecture token par token.
4. Les scripts de lecture existants filtrent `kind in (reading, repeat)` : ajouter `open`.
[agent critique · T2]
