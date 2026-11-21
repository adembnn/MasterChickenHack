# T2 : la méthode généralise-t-elle à une famille autre qu'OLMo ?

Rôle : challenger final de la méthode v3 (`t2-method-v3.md`, tau=0,42-0,43) sur la question
« ce seuil marche-t-il peu importe le vrai modèle derrière les 33 endpoints, ou seulement si
c'est OLMo ? ». Aucune requête API hackathon. Calcul local (stdlib Python, CPU, aucune
contention GPU — la H200 partagée était à 100 % d'utilisation avec 5 jobs concurrents au
moment de la rédaction).

## 0. Verdict

**La méthode a besoin d'une calibration par famille (ou d'un équivalent adaptatif) — elle ne
généralise pas telle quelle.** Pas parce qu'on a mesuré un échec sur une autre famille (cette
mesure est en cours, voir §3), mais parce que **la seule donnée réelle dont on dispose montre
déjà que le bruit de fond n'est pas une constante universelle, même sur UN SEUL modèle** : il
varie de 0,30 à 0,34 juste en changeant la taille de lot de serving (`results_exp1_a100.json`,
jumeau OLMo-2-1B, n=300). Rien ne garantit que les 33 endpoints réels partagent la même taille
de lot, la même implémentation de quantification, ni la même famille. Un seuil figé à 0,43,
calibré sur un seul point de cet espace, est fragile par construction. La solution proposée (une
calibration **adaptative par auto-consistance**, §2) coûte 20 réponses sur le budget de 100 et
récupère la majorité de la perte dès que l'hétérogénéité inter-famille dépasse ~0,10-0,12 de
divergence (voir le croisement des courbes en §1).

## 1. Ce qu'on peut déjà démontrer sans attendre de nouvelles mesures

### 1.1 Le fait mesuré qui remet en cause la constante unique
`results_exp1_a100.json` (jumeau OLMo-2-1B, n=300, batch 1/8/32, dtype bf16/fp16) :

| condition | divergence (p = 1 - identical_frac) |
|---|---|
| bs=1 | 0,300 |
| bs=8 | 0,340 |
| bs=32 | 0,317 |
| dtype float16 | 0,333 |

Écart de 0,04 **sur le même modèle, les mêmes poids, juste le batching serveur**. tau=0,43 a été
choisi comme `max(bruit mesuré) + 2·SE` = 0,43 précisément pour absorber cette variance connue
*intra*-modèle. Mais rien n'indique que l'écart *inter*-famille (taille différente, tokenizer
différent, implémentation serveur différente, peut-être un moteur d'inférence différent) reste
dans la même fourchette de 0,04. C'est l'hypothèse non testée centrale.

### 1.2 Simulation de sensibilité (`tracks/t2-model-equality/family_robustness_sim.py`)
Script écrit pour ce rapport, stdlib only, 4000 tirages par point. Principe : au lieu de
supposer (comme `mcc_simulation.py` §7) un décalage *uniforme* du bruit sur tous les endpoints,
on tire **une famille à la fois** son propre bruit de fond autour de la moyenne OLMo
(0,323) ± une demi-largeur `d` (hétérogénéité inter-famille, inconnue — balayée), on applique
aux variantes altérées (int8/fp8/nf4/LoRA) le même facteur multiplicatif que celui mesuré sur
OLMo, et on compare 3 stratégies de décision :

- **tau fixe 0,43** (méthode v3 actuelle, un seul chiffre pour toutes les familles) ;
- **tau oracle par famille** (borne supérieure théorique — on connaît par magie le vrai bruit de
  chaque famille, sans aucune mesure) ;
- **tau adaptatif** : on dépense 20 réponses du budget de 100 pour rejouer le même prompt
  plusieurs fois sur un endpoint déjà suspecté « même modèle » (ou deux endpoints du même
  cluster présumé), on mesure le taux de divergence RÉEL de cette famille sur cet échantillon,
  et `tau_famille = p̂_bruit + 2·SE(n=20)`. Les 80 réponses restantes servent à la décision.

Résultat (MCC moyen, 4000 tirages, 6 familles de 4-7 endpoints, ~45% de variantes altérées) :

| demi-largeur du bruit inter-famille | tau fixe 0,43 | tau oracle/famille | tau adaptatif (20 réponses) |
|---|---|---|---|
| ±0,00 (toutes les familles = OLMo) | 0,975 | 0,965 | 0,861 |
| ±0,03 | 0,966 | 0,966 | 0,863 |
| ±0,06 | 0,940 | 0,965 | 0,863 |
| ±0,10 | 0,888 | 0,963 | 0,863 |
| ±0,15 | 0,809 | 0,959 | 0,859 |
| ±0,20 | 0,700 | 0,955 | 0,841 |

**Lecture** : si toutes les familles ont vraiment le bruit d'OLMo (±0,00), le seuil fixe est
meilleur (0,975 vs 0,861) — l'adaptatif paie un coût réel (bruit d'estimation sur seulement 20
réponses, budget en moins pour la décision). Mais dès que l'hétérogénéité inter-famille dépasse
**~0,10-0,12** (ce qui est plausible : on a déjà 0,04 par le seul batching sur UN modèle, et des
familles de taille/tokenizer différents peuvent plausiblement diverger plus), **les courbes se
croisent** : le seuil fixe s'effondre (0,70 à ±0,20) alors que l'adaptatif reste quasi stable
(0,841 à ±0,20) parce qu'il se recalibre sur le signal réel de chaque famille au lieu de supposer
qu'elle se comporte comme OLMo.

Coût du budget d'auto-consistance (à demi-largeur 0,15) :

| réponses dépensées en auto-consistance | MCC adaptatif | MCC fixe (référence) |
|---|---|---|
| 10 | 0,828 | 0,809 |
| 20 | 0,859 | 0,809 |
| 30 | 0,871 | 0,809 |

20 réponses suffisent déjà à dépasser le seuil fixe dans ce scénario ; 30 encore un peu mieux,
mais la marge décroît (loi des grands nombres : au-delà de ~20-30, le gain par réponse dépensée
en calibration devient inférieur au gain par réponse dépensée en décision).

**Limite explicite de cette simulation** : les profils de divergence pour int8/fp8/nf4/LoRA par
famille non-OLMo sont des **extrapolations** (même facteur multiplicatif qu'OLMo, appliqué au
bruit réel de la famille), pas des mesures. Ce n'est pas une prédiction de MCC réel : c'est une
démonstration que **si** l'hétérogénéité inter-famille existe (ce qui est probable, pas certain),
l'adaptatif la tolère et le fixe non. La vraie mesure arrive (§3).

## 2. Méthode adaptative proposée (prête à l'emploi dès que les clusters de famille sont connus)

1. **Étape F** (déjà dans la méthode v3) identifie les familles/clusters candidats par texte.
2. **Pour chaque cluster "même modèle" présumé** (au moins 2 endpoints), choisir 10 à 20 prompts
   déjà joués dans la vague F/A et **relire la même paire de réponses déjà obtenue** (ne coûte
   AUCUNE requête supplémentaire si on réutilise les prompts de F/A au lieu d'un budget séparé —
   version "gratuite" de l'idée) OU rejouer un sous-ensemble de prompts une 2e fois sur le MÊME
   endpoint pour mesurer directement le bruit de ce serveur précis (version "payante", 10-20
   réponses, recommandée si le budget le permet car elle mesure le bruit du VRAI serveur, pas
   seulement la variance entre deux endpoints présumés identiques, qui mélange bruit + éventuel
   écart de poids).
3. `tau_famille = p̂_bruit_mesuré + 2·SE(n_mesure)`, plutôt que le 0,43 universel.
4. Si `n_mesure` est trop petit pour une famille isolée (singleton, pas de paire à comparer) :
   retomber sur 0,43 comme valeur par défaut documentée — l'adaptatif ne remplace le fixe que
   quand il y a assez de signal pour mesurer, jamais une extrapolation sans donnée.
5. Coût total raisonnable sur 33 endpoints / ~6-8 familles présumées : 10-20 réponses par famille
   (pas par endpoint) si on mesure une seule paire représentative par famille et qu'on applique
   `tau_famille` à tous les membres du cluster.

## 3. État des données réelles au moment de la rédaction

Aucune mesure réelle de divergence (int8/fp8/nf4/LoRA) n'existe encore pour une famille autre
qu'OLMo-2-1B dans le dépôt. Ce qui existe :
- `fingerprint_bank.py` / `results_fingerprint_bank.json` : identification textuelle (auto-ID,
  cutoff, style) sur 9 familles (Qwen2.5 1.5B/7B, SmolLM2, OLMo-2-1B, Pythia, StableLM2,
  Phi-3.5-mini, Ministral-8B, Mistral-7B). Utile pour l'identité, pas pour le bruit/seuil.
- `family_detection.py` : séparation OLMo vs Qwen2.5 vs SmolLM2 par texte (0 % d'identique entre
  familles sur prompts ouverts). Répond « peut-on distinguer deux familles », pas « le seuil
  intra-famille généralise-t-il ».
- **En cours** (annoncé sur l'issue #4 pendant la rédaction de ce rapport, session « banque
  d'empreintes ») : `exp1_divergence_multi.py`, reprenant l'expérience de divergence d'OLMo
  (fake-quant fp8/int8, nf4 réel, LoRA 0,3/0,7/1,0/1,5) sur Qwen2.5-1.5B, Ministral-8B,
  Mistral-7B, Phi-3.5-mini, SmolLM2-1.7B — **exactement la mesure manquante pour trancher ce
  rapport avec des chiffres réels au lieu d'une simulation de sensibilité**. ~10-20 min/modèle,
  5 modèles : résultat attendu après ce rapport (limite d'usage de cette session atteinte avant
  la fin du job). La VM H200 partagée était à 100 % d'utilisation (5 jobs concurrents, dont ce
  job) : aucun calcul GPU supplémentaire n'a été lancé par cette session pour ne pas aggraver la
  contention.

**Action de suivi recommandée** (pour qui reprend ce rapport) : dès que
`exp1_divergence_multi.py` produit ses résultats, remplacer le §1.2 (simulation) par les vrais
taux de divergence par famille et refaire tourner `family_robustness_sim.py` avec les vraies
valeurs à la place du facteur multiplicatif extrapolé — la conclusion qualitative (fixe fragile,
adaptatif robuste) devrait tenir, mais le point de croisement exact (ici ~0,10-0,12) doit être
recalculé sur les vraies données.

## 4. Ce que ça change pour la méthode v3

- Ne pas figer tau=0,43 comme constante universelle pour les 33 endpoints si plusieurs familles
  sont identifiées (étape F) : calculer `tau_famille` par cluster dès que ≥2 endpoints d'un
  cluster permettent une mesure de bruit (§2).
- Budget : prévoir explicitement 10-20 réponses par famille présumée pour la calibration
  adaptative dans la vague B (déjà prévue pour les paires ambiguës dans `t2-method-v3.md` §1,
  étape 3) — ne pas ajouter un budget séparé, réutiliser la vague existante.
- Documenter noir sur blanc dans la soumission finale (README / commentaire issue #4) que
  tau=0,43 est une valeur par défaut pour les familles sans assez de signal pour une mesure
  directe, pas un chiffre validé universellement.
- **Ce rapport ne traite pas** le problème plus grave déjà signalé par Adem sur l'issue #4 (la
  v3 suppose des copies "même modèle" en clair, alors que le corrigé `practice_answers.csv`
  montre qu'elles sont toutes déguisées) — ce chantier est distinct (filigrane/persona/prompt
  système) et suivi par d'autres sessions (« robustesse aux déguisements », « test en aveugle »).
  Les deux problèmes se cumulent : un tau par famille mal calibré ET un déguisement non neutralisé
  sont deux façons indépendantes d'isoler une vraie paire "même modèle" à tort.
