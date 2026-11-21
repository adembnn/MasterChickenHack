# T2 : plan opérationnel pour dépenser le budget réel (100 réponses x 33 `ep-*`)

Décision débloquée : comment dépenser, en vrai, un budget qu'on ne peut jamais reconstituer. Ce rapport ne fait aucune hypothèse statistique nouvelle : il **synthétise et rend actionnable** ce que [`t2-statistics.md`](t2-statistics.md) (§4-7, maths et simulations) et [`t2-fingerprinting.md`](t2-fingerprinting.md) (§6-9, pipeline et plan de secours) ont déjà établi, et le croise avec les contraintes opérationnelles de [`submissions-ops.md`](submissions-ops.md) (délais, formats, risques). Rien n'a été appelé pour écrire ce document. Écrit pour qu'un non-statisticien de l'équipe puisse l'exécuter sans relire les 3 rapports sources.

Rappel des chiffres qui encadrent tout : 33 `ep-*`, 100 réponses chacun **pour toute la compétition**, 256 tokens max/réponse, 4 requêtes simultanées max (toute l'équipe), **une soumission CSV toutes les 30 min, erreur comprise**. Code stoppé dimanche 22/11 12:00 (`CLAUDE.md`). Les 9 `ref-*` sont gratuits et illimités : tout ce qui peut se décider sur `ref-*` doit l'être AVANT de toucher à un `ep-*`.

## 1. Tout d'un coup, ou en plusieurs vagues ?

**Réponse : en plusieurs vagues, mais pas pour "cribler" en ligne — pour se protéger et pour cibler.** La répartition déjà fixée dans `t2-statistics.md` §5 est :

| Vague | Réponses/endpoint | Total requêtes (33 ep) | Rôle |
|---|---|---|---|
| Ronde F (famille) | 16 | 528 | 16 prompts ouverts distincts, 1 réponse chacun. Sépare les familles (OLMo de tailles différentes a un écart de vraisemblance énorme : séparable en 5-10 réponses). Détecte aussi le nombre de familles réelles. |
| Ronde A (cœur) | 48 | 1 584 | 16 prompts choisis **hors ligne** (sur `ref-*` + jumeau local), x3 répétitions. Construit la matrice de distances complète et le premier dendrogramme. |
| Ronde B (adaptative) | jusqu'à 24, **seulement sur les endpoints dans une paire ambiguë** | ~300-800 (pas 33x24) | Les 4 prompts les plus discriminants de la Ronde A, rejoués. |
| Réserve | 12, jamais dépensée sans décision explicite | 0-400 | Vérification finale sur 2-3 cas vraiment indécis. |

Total maximum : 100/endpoint, jamais plus — mais en pratique beaucoup d'endpoints n'utiliseront pas la Ronde B ni la réserve, et c'est voulu : **une réponse non dépensée ne vaut rien pour le score, donc ne pas la dépenser n'est jamais une perte**, seulement le fait de la dépenser au mauvais endroit en est une.

### Pour "tout d'un coup" (100 d'un bloc)
- Maximise la puissance statistique *par endpoint* : la variance d'un contraste de vraisemblance décroît en 1/N, donc N=100 d'un coup est strictement le plus puissant *si on sait déjà quoi mesurer*.
- Aucun risque d'erreur de répartition ni de code de criblage buggé en pleine nuit.
- Une seule fenêtre d'exécution, plus simple à planifier et à surveiller (moins d'occasions de double-dépense par deux opérateurs, risque n°2 de `submissions-ops.md` §10).

### Contre "tout d'un coup"
- **Irréversible au carré** : si l'hypothèse de famille (OLMo, quelle taille) ou la liste de prompts gelée est fausse, ou si le format de réponse de l'API diffère de ce qu'on attend (troncature, préfixe), **100 % du budget est perdu sur un design cassé**, sans aucun moyen de corriger.
- Aucune marge pour réagir à une surprise (ex. le filigrane est bien plus fort que mesuré sur `ref-*`, ou le modèle réel n'est pas OLMo du tout).
- La littérature de simulation (`twostage.py` dans `t2-statistics.md` §5) montre que cribler *en ligne avec peu de réponses par endpoint* ne marche pas pour un signal subtil (le bruit binomial domine à n=2) — mais ce n'est PAS un argument pour dépenser tout d'un coup : c'est un argument pour faire la sélection de prompts **hors ligne** (déjà fait, Ronde A) et garder les vagues réelles seulement comme **filet de sécurité macro** (changement de famille, erreur de format) et comme **levier d'allocation** (qui reçoit la Ronde B), pas comme un outil de calibration fine en ligne.

### Le compromis retenu, en une phrase
On dépense 64/100 (F+A) partout, ce qui donne déjà la matrice de distances complète et une première soumission valide ; on garde 36/100 en réserve ciblée, dépensée seulement sur les ~20-40 % d'endpoints qui restent ambigus après F+A (§3). Le coût d'opportunité de cette prudence est faible : la Ronde F (16) à elle seule donne déjà la famille à ~98-100 % (`t2-fingerprinting.md` §7) ; le gros risque (modèle candidat totalement faux) se détecte avec 16-30 réponses, bien avant d'avoir englouti le budget.

## 2. Backtester une vague réelle contre le jumeau local (signaux d'alerte sur les 20-30 premières vraies réponses)

Le "backtest" se fait en deux temps : **avant** (sur `ref-*`, gratuit, déjà prévu en `t2-statistics.md` §4) on construit la distribution attendue sous H0 (même modèle + nuisance) et sous H1 (variante) ; **après** la première vraie vague (Ronde F, 16/endpoint, puis les premières réponses de la Ronde A), on compare les vraies données à ces distributions attendues. Les 20-30 premières réponses réelles par endpoint (= Ronde F + un peu de Ronde A, sans attendre la fin) suffisent pour ces contrôles parce qu'ils portent sur des écarts *grossiers*, pas sur la discrimination fine qui demande les 100.

Quatre signaux concrets, chacun avec un seuil numérique issu des rapports existants :

1. **Le modèle candidat local ne "colle" à aucun endpoint réel.** Mesure : `mean log-prob` du texte de l'endpoint sous le meilleur candidat local (argmax sur les ~30 candidats du zoo, `t2-fingerprinting.md` §3.4-3.5). Comparer à la même mesure sur `ref-*` sous SON vrai candidat identifié en étape 0. Seuil d'alerte : écart > 0,2-0,3 nat/token pour plus de la moitié des 33 endpoints. C'est le signal le plus grave : il dit que l'hypothèse de famille (OLMo, telle taille) est fausse pour les `ep-*` réels — direct vers le plan de secours (§4).
2. **Aucun candidat ne se détache clairement.** Mesure : marge entre le meilleur et le deuxième candidat en `mean lp`. Seuil d'alerte : marge < 0,05 nat/token pour la majorité des endpoints (comparer à la marge observée entre OLMo-2-1B et une autre taille sur `ref-*`, qui doit être nette, plusieurs dixièmes). Si la marge est petite partout, la famille elle-même est incertaine, pas seulement le modèle précis.
3. **Le format des réponses diffère de ce qu'on a calibré sur les prompts synthétiques/`ref-*`** : taux de refus, taux de troncature avant 256 tokens, taux de réponse vide, présence d'un préambule inattendu ("Bien sûr ! Voici..."). Seuil d'alerte : taux > 2x ce qu'on observe sur `ref-*` avec les mêmes prompts. Conséquence si déclenché : **s'arrêter avant la Ronde A**, corriger les prompts (ils gaspillent des tokens sur du texte inutile), ne jamais continuer à dépenser sur un gabarit cassé.
4. **La variance réelle est bien plus grande que la variance simulée.** Mesure : écart-type bootstrap (ré-échantillonnage des réponses déjà reçues, `t2-statistics.md` §3.4) comparé à l'écart-type attendu par la simulation (sim `calib.py`/`alloc.py`). Seuil d'alerte : rapport > 2x. Conséquence : le signal réel est plus bruité que prévu par le jumeau — il faut regrouper plus d'endpoints par "prototype de cluster" (la mise en commun §5 de `t2-statistics.md` réduit la variance de ~40 %) et prévoir de consommer la réserve plus largement, pas seulement sur 2-3 cas.

Test de cohérence interne supplémentaire, gratuit en information : la Ronde F révèle souvent le prompt système/persona (en demandant à l'endpoint de le répéter). Si deux endpoints partagent exactement le même prompt système extrait, mais que leurs features de vraisemblance diffèrent énormément (z-score > 5 dans les unités normalisées de `t2-statistics.md` §2.4), c'est un signal que la nuisance "prompt système" n'explique pas l'écart — donc que ces deux endpoints sont probablement des modèles différents, et pas un faux positif du test.

**Comment lire le verdict global en une phrase :** si les signaux 1-3 sont calmes, le jumeau est utilisable, on continue le plan nominal (§1, §3). Si le signal 1 ou 2 est déclenché, le plan de secours (§4) prend le relais **immédiatement**, avant d'avoir dépensé la Ronde A complète sur un design inutile.

## 3. Algorithme séquentiel : comment réallouer le budget restant

Point important à ne pas rater : le budget est **plafonné par endpoint** (100 chacun), pas un pot commun de 3 300 réponses. "Réallouer" ne veut donc pas dire déplacer des réponses d'un endpoint à un autre, mais décider **pour quels endpoints** on dépense au-delà de F+A (64), et jusqu'où.

Algorithme, à exécuter après chaque vague :

```
Entrée : matrice T_ef (distance normalisée) et P(même | T_ef) pour les 528 paires
         intra-famille (calibrée sur ref + synthétiques, t2-statistics.md §6.2 :
         p* ≈ 0,35-0,45, pas 0,5 ni une p-valeur à 0,05)

1. Après Ronde F+A (64 réponses dépensées, tous endpoints) :
   Pour chaque paire (e,f) de même famille présumée :
     - P >= 0,6  -> décision "même modèle", gelée
     - P <= 0,2  -> décision "modèles différents", gelée
     - sinon     -> "ambigu", ajouter e et f à E_ambig

2. Si E_ambig est vide :
   -> ARRÊT. Ne pas dépenser la Ronde B ni la réserve (ça ne changerait rien
      de mesurable). Geler la partition, passer à la soumission (§5).

3. Si E_ambig est non vide :
   Pour chaque endpoint de E_ambig, dépenser la Ronde B (jusqu'à 24 réponses,
   les 4 prompts de plus forte contribution à T_ef en Ronde A), en comparant
   non pas à l'autre endpoint mais au PROTOTYPE du cluster suspecté (somme
   des réponses des autres membres) : réduit la variance de ~40 % (t2-statistics
   §5), donc une paire "limite" à 64 réponses devient souvent décidable à 88.

4. Recalculer P(même | T_ef) avec Ronde B incluse :
     - P sort de [0,2 ; 0,6]  -> décision gelée, ARRÊT pour cette paire,
       le reliquat de budget (jusqu'à 12, la réserve) N'EST PAS dépensé.
     - P reste dans [0,2 ; 0,6] -> passer à l'étape 5.

5. Règle de priorité sur la réserve (12 restants) : ne JAMAIS la dépenser sur
   toutes les paires encore ambiguës d'un coup. Prioriser par coût d'erreur =
   taille du cluster concerné (table MCC de t2-fingerprinting.md §7 : fusionner
   un endpoint dans un cluster de taille k coûte ~k faux positifs, l'isoler
   coûte k-1 faux négatifs). Dépenser la réserve d'abord sur les paires qui
   affectent le plus grand cluster, ensuite sur les autres, jusqu'à épuisement
   du temps ou du budget.

6. Règle de défaut si toujours ambigu après 100/100 dépensées : ISOLER
   (ne pas fusionner). Raison (t2-statistics.md §6.3, règle de prudence c) :
   un singleton ne coûte aucun faux positif ; une fusion fausse dans un gros
   cluster coûte beaucoup. En cas de doute réel, le biais "séparer" est le
   moins cher en MCC attendu.

7. Règle d'arrêt globale (vaut pour tout le monde, pas seulement E_ambig) :
   si l'ajout de la Ronde B sur un échantillon pilote de paires change le MCC
   estimé par validation croisée de moins de 0,03, arrêter TOUTE dépense
   supplémentaire (y compris la réserve) : le reste n'est que du bruit
   (seuil déjà fixé dans le plan de la nuit de t2-fingerprinting.md §9).
```

**Cas "signal plus fort que prévu"** (le jumeau sous-estimait la séparabilité) : si après Ronde F+A toutes les paires sont déjà P>=0,9 ou P<=0,1 (`E_ambig` vide), c'est ce cas. Conséquence concrète et positive : ne pas dépenser Ronde B/réserve sur ce track, rediriger le temps GPU/personne vers T1/T3, et garder la réserve T2 comme pure assurance contre une erreur de format découverte tardivement (jamais zéro, cf. §4 de `submissions-ops.md`, risque n°1).

**Cas "signal plus faible que prévu"** (déjà couvert par le signal d'alerte n°4 du §2) : `E_ambig` est grand (>50 % des endpoints). Dans ce cas, la règle 5 ci-dessus (prioriser par taille de cluster) devient le levier principal : on ne peut pas tout vérifier avec 36 réponses/endpoint supplémentaires, donc on protège d'abord les décisions qui coûtent le plus cher en MCC si elles sont fausses — pas celles qu'on est juste curieux de résoudre.

## 4. Pire scénario : le jumeau ne ressemble pas du tout au vrai modèle

Scénario précis : le signal d'alerte n°1 ou n°2 du §2 se déclenche sur la majorité des 33 endpoints — aucun candidat OLMo local n'explique les réponses réelles (mauvaise famille, mauvaise taille, ou les organisateurs n'ont pas utilisé OLMo pour les `ep-*` malgré l'indice sur `ref-*`). C'est le mode d'échec n°1 déjà identifié dans `t2-fingerprinting.md` §8.

**Plan de repli minimal, qui garantit un score non nul :**

1. **Ne pas abandonner la famille (poids 0,1, presque gratuite même sans jumeau).** Les traits de style/bannière ("Qui es-tu ?", refus, longueur de réponse, tokenisation) séparent la famille même sans modèle de référence exact (LLMmap/Idiosyncrasies, 97-100 % rapportés dans `t2-fingerprinting.md` §1). Ça ne coûte que la Ronde F déjà dépensée (16/endpoint) : rien de supplémentaire n'est nécessaire pour ce point.
2. **Model-level sans jumeau : clustering générique, pas d'identification.** Utiliser un noyau texte générique (n-grammes de mots/caractères, ou distance delta sur réponses à choix forcé — le noyau le plus robuste aux nuisances d'après AgentProv, cf. `t2-statistics.md` §1 et §2.1) directement entre les 33 endpoints, SANS passer par un modèle local. Clustering hiérarchique à lien moyen, seuil calibré sur `ref-*` (même si le modèle `ref` diffère du vrai modèle `ep`, le RATIO séparation-intra/séparation-inter se transfère mieux que la valeur brute).
3. **Budget minimal requis : 16-30 réponses/endpoint** (la Ronde F, éventuellement complétée par le début de la Ronde A) — on n'a jamais besoin de la Ronde B/réserve dans ce scénario, puisque sans jumeau il n'y a pas de paires "presque tranchées" à affiner, seulement un classement de distances brutes.
4. **Ce que ça rapporte, en ordre de grandeur :** `t2-fingerprinting.md` §8.1 estime que la famille reste quasi intacte (MCC_famille ~0,9-1,0) et que la précision de paire se dégrade mais reste **au-dessus de zéro** par simple proximité mutuelle de style (descripteurs invariants). Estimation grossière : MCC_modèle 0,3-0,5 au lieu de 0,5-0,85 en régime nominal, donc score total de l'ordre de 0,1x0,95 + 0,9x0,4 ≈ 0,45 au lieu de 0,1+0,9x0,7 ≈ 0,73. Dégradé, mais loin du 0 d'une soumission "tout différent" (score garanti nul, confirmé dans `submissions-ops.md` §1.2).
5. **Règle d'exécution :** soumettre CE candidat de repli **dès qu'il est prêt**, même si on espère l'améliorer ensuite. `CLAUDE.md` et `submissions-ops.md` §0.4 confirment que seule la meilleure soumission compte pour le score final — soumettre tôt un plancher non nul n'a aucun coût, et protège contre toute panne, bug ou manque de temps en fin de nuit.

## 5. Calendrier heure par heure (aujourd'hui samedi 21/11 jusqu'à dimanche 22/11 12:00)

Ce calendrier spécialise, pour le budget réel T2 seulement, le plan déjà posé dans `t2-fingerprinting.md` §9 et `submissions-ops.md` §9 (qui couvrent les 3 tracks) : il ne les remplace pas, il précise QUAND les `ep-*` sont touchés et combien de créneaux de 30 min sont utilisés.

| Heure | Action T2 | `ep-*` touchés ? | Soumissions CSV |
|---|---|---|---|
| 16:00-18:00 | Étape 0 sur `ref-*` (gratuit, illimité) : identité du modèle ref, latence/débit à 4 parallèles, extraction prompt système. Construction du zoo de candidats locaux (quantifs, LoRA synthétiques). | Non | — |
| 18:00-21:00 | Génération de synthétiques, sélection des prompts par critère de Fisher (hors ligne), gel de la liste de 16+16 prompts (relue à deux, postée sur l'issue #4 — règle d'équipe, obligatoire avant tout `ep-*`). | Non | — |
| 21:00-21:30 | Annonce du plan figé sur l'issue T2 (#4). Dernière vérification du validateur CSV sur un fichier factice. | Non | Soumission de contrôle "tout différent / famille au hasard" possible ici pour valider le format serveur (score ~0 ou 0,1, sans risque) |
| 21:30-22:15 | **Ronde F** : 16 réponses/endpoint, 528 requêtes, 4 en parallèle (~20-45 min selon la latence mesurée en étape 0). | Oui, 1ère vague | — |
| 22:15-22:45 | Analyse immédiate : signaux d'alerte §2 (famille, marge, format, variance). **Décision go (plan nominal §3) / no-go (plan de secours §4).** | Non (analyse) | — |
| 22:45-23:45 | **Ronde A** (plan nominal) OU dépense équivalente ciblée sur le clustering générique (plan de secours) : 48 réponses/endpoint, 1 584 requêtes (~1-2 h selon latence à 256 tokens). | Oui, 2e vague | — |
| 23:45-00:15 | Calcul de T_ef, P(même\|T), identification de `E_ambig` (§3, étapes 1-2). Rédaction du premier CSV réel (33 lignes, validé par `validate_t2.py`). | Non (analyse) | **Soumission #1 ~00:15** (créneau 30 min lancé) |
| 00:15-01:00 | **Ronde B** sur `E_ambig` seulement (jusqu'à 24/endpoint concerné, pas 33x24). | Oui, 3e vague (ciblée) | — |
| 01:00-01:30 | Recalcul, décisions gelées ou réserve ciblée (§3 étapes 4-5). | Non (analyse) | **Soumission #2 à 01:00** si le CSV a changé (sinon attendre d'avoir un vrai changement, ne pas gaspiller un créneau) |
| 01:30-08:00 | **Relais de sommeil** (cf. `submissions-ops.md` §9 pour qui dort quand). GPU : calcul des features de vraisemblance en tâche de fond (ne consomme aucun `ep-*`). Dépense de la réserve UNIQUEMENT si §3 étape 5/6 le demande, et annoncée sur le bus avant. | Possible mais rare (réserve ciblée) | — (pas de nouveau CSV sauf changement réel) |
| 08:00-09:30 | Tout le monde éveillé. Décision finale par paire encore ambiguë (règle d'isolement par défaut, §3 étape 6). MBR / clustering de corrélation, comparaison des deux décodeurs (`t2-statistics.md` §6.3). | Non (fin de dépense) | — |
| 09:30-10:00 | **Répétition générale (R3)** : CSV candidat final, validé, `sha256` noté. | Non | **Soumission #3 ~09:30-10:00** (vraie candidate, pas un test) |
| 10:00-10:30 | Marge de vérification (score public = contrôle de format seulement, jamais de réglage dessus, règle `t2-statistics.md` §6.3d). | Non | — |
| 10:30 / 11:00 / 11:30 | **3 créneaux de secours réservés** (déjà la règle de `submissions-ops.md` §11) : chaque soumission ici ne change le CSV que si une vraie correction de format ou de clustering a été trouvée depuis 10:00 — ne jamais soumettre "pour soumettre". | Non | **Soumissions #4, #5, #6 si nécessaire** |
| 11:40 | **Dernier envoi possible.** Le scoring peut prendre jusqu'à ~500 s (`submissions-ops.md` §0.6) : aucun envoi après 11:40. | Non | — |
| 11:40-12:00 | Gel total. Vérifier le leaderboard, aucun process actif, `git status` propre sur la branche T2. | Non | — |

Budget de créneaux : 6 soumissions utilisées sur un maximum théorique d'environ 25 disponibles entre 00:15 et 12:00 (750 min / 30 min) — large marge pour une erreur de format en cours de route sans jamais manquer la fenêtre finale. Compte cohérent avec la réserve de "au moins 4 créneaux dans la dernière heure" déjà recommandée dans `submissions-ops.md` §7.

## 6. Ce qui reste vrai quel que soit le scénario

- Jamais deux opérateurs `ep-*` en même temps (limite de 4 requêtes simultanées partagée par toute l'équipe) ; un seul propriétaire T2, annonce sur le bus avant chaque vague (`CLAUDE.md` règle "ressource partagée").
- Chaque vague écrit son JSONL brut avant tout parsing (journal intent/result de `submissions-ops.md` §5) : si une analyse est fausse, on peut toujours la refaire sur les données déjà payées — ce qui ne peut jamais être refait, c'est la requête elle-même.
- Aucune décision de seuil ne se prend sur le score public (30 % des paires) : il sert à vérifier le format, pas à calibrer (`t2-statistics.md` §6.3d).
