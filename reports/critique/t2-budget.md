# T2 · le meilleur usage des 50 réponses restantes par guichet (cycle 3)

Décision débloquée : **que demander à chaque `ep-*` avec les 50 réponses qui restent** (budget irréversible, à Titouan de trancher).
Session t2-budget d'Adem. 0 requête, 0 lecture de `.env`, rien soumis. Fichiers : `t2-budget-questions.jsonl`, `t2-budget-make.py` (fabrique le fichier et vérifie le budget), `t2-budget-info.py`, `t2-budget-repeat.py`, `t2-budget-filigrane.py`, `t2-budget-longueur.py`.

## 1. État du cycle 3
Prévu, pas écrit. Aucun `probes_v4_c3.py` (dépôt, branches distantes, `E:\EAC-lab`). Le fil #4 ne le cite que deux fois comme intention (« prose longue et comparaisons, pas de listes » ; « on s'en sert pour le cycle 3 »). Aucune requête `ep-*` en cours : 50 sur 100 dépensées par guichet. On part donc de zéro.

## 2. Quelles questions ont porté le plus d'information ?
Critères simples et rejouables sur les réponses déjà payées (`t2-budget-info.py`) : longueur en mots, diversité lexicale (mots distincts sur les 60 premiers), distance de vocabulaire aux copies, début de réponse commun avec les copies. Pour chaque question : écart variantes − copies, divisé par le bruit des copies ; on retranche ce qu'un tirage au hasard des étiquettes donnerait (200 permutations).

**Validation sur les ref (corrigé connu : 02 fp8, 04 / 07 / 09 LoRA, contre 5 copies) : la longueur marche.** Les variantes écrivent plus long que les copies : compare +2,0 σ, explain +1,7, paragraphe de 80 mots +1,3, story +1,2, list +1,0, python +0,8, steps +0,5. Excès sur le hasard : compare +0,99, paragraphe +0,85, explain +0,75, list +0,12, steps +0,09. C'est le même ordre que la mesure par vraisemblance de la session critique (prose et comparaisons au-dessus, listes et pas-à-pas en bas).

**Sur les ep (variantes confirmées 17, 18, 22, 05, 15 contre copies confirmées 04, 14, 23, 21, 32, 07, plus 10, 13, 19 en Llama où le public n'a confirmé aucune copie) : aucun critère ne sépare, quel que soit le type.** Tous les excès sont entre −0,4 et +0,3 (seules exceptions : « extract » n = 3 et « rewrite » début commun n = 11, trop peu pour compter). Corrélation de rang entre le classement ep et le classement ref : **+0,10**, donc rien. Lecture : sur les ref la LoRA est entraînée fort et allonge les réponses ; sur les ep les variantes (fp8, LoRA faibles) ne laissent aucune trace de surface avec 1 réponse par question. **On ne peut pas classer les questions sur les ep par un critère de surface ; le classement qui tient reste celui par vraisemblance sur les ref** (prose > comparaison > pas-à-pas > liste, réécriture).

**Longueur des réponses ep (mots, 825 + 689 réponses)** : steps 190 (92 % au-dessus de 170 mots, donc coupées au plafond de 256 tokens), list 133 (32 % coupées), compare 133, python 129, explain 108, story 108, paragraphe de 80 mots 97, rewrite 65 (médiane 50), identité et extraction 30. Les questions en prose à 100 à 130 mots se terminent avant le plafond ; le pas-à-pas dépasse.

## 3. Filigrane par répétition : mesure faite
`t2-budget-filigrane.py` : sur les ref, 4 tirages de 21 questions. **ref-06 (filigrane) : 11 % des paires de réponses partagent plus de 30 mots de début, 9 questions sur 21 montrent au moins une paire à 20 mots ou plus. Les 8 autres ref : 0 paire à 30 mots sur 1 008, et 3 paires sur 126 pour ref-02 à 20 mots.** Le seuil « ≥ 30 mots communs » n'a donc aucune fausse alerte mesurée.
Sur les ep, la question P002 posée deux fois au bloc 0 : **ep-06 partage 60 mots, tous les autres au plus 15** (ep-25 15, ep-19 10, ep-21 9, ep-09 8). ep-06 (Qwen-1,5B) est donc le filigrane de sa famille. Par le gabarit « 3 simples par famille » (06 / 32 / 26 pour Qwen-1,5B), celui des deux autres familles est parmi **ep-17, 04, 25** (Qwen-3B) et **ep-13, 19, 10** (Llama) : ep-25 et ep-19 sont les plus hauts. Pourquoi ça compte : un filigrane est une copie, mais les lecteurs le voient comme « texte improbable » et le prennent pour une variante (ep-17 est précisément « variante confirmée » au public).

## 4. Plan des 50 réponses (chiffré)
Le budget est par guichet : on ne peut pas reporter des réponses d'un guichet à l'autre. Donc on les dépense là où elles décident quelque chose.

| Ligne | Qui | Nombre par guichet | Raison |
|---|---|---|---|
| **L1** prose longue, 30 questions distinctes (14 « Explain X… for a curious teenager », 8 « Compare A and B… », 8 « paragraphe de 80 mots… ») | 22 non tranchés ; 23 d'entre elles pour les 11 tranchés | 30 (23) | seul type validé (ref) ; réponses de 100 à 130 mots qui se terminent avant le plafond ; sujets nouveaux, aucun doublon avec les 46 déjà payées |
| **R1** répétition de P002 (« paragraphe vélo »), 2 tirages de plus | tous | 2 | passe de 2 à 4 tirages = 6 paires par guichet, détecte un filigrane à 11 % par paire |
| **R2** seconde question répétée ×4 (« un port au lever du jour ») | suspects 25, 13, 19, 10, 04, 17 | 4 | 6 paires de plus ; détection cumulée ≈ 75 % par suspect, 0 fausse alerte |
| **B** contexte long commun, conversation à 3 tours (600 tokens environ : une vue d'ensemble de l'histoire de la poste, puis 6 consignes de paragraphes différentes) | 22 non tranchés | 6 | compare les guichets sur le même long contexte ; dilue persona et prompt (non mesuré : **test ref d'abord**) |
| **D** extraction : 2 demandes directes + 1 « tour assistant » forcé | 22 non tranchés | 3 | seuls 02, 16, 20, 29, 22, 32 ont rendu un prompt au bloc 0 ; les autres répètent la question (ep-08 y compris, alors qu'il est classé « prompt ») (non mesuré : **test ref d'abord** pour D3) |
| **Réserve** | tous | non tranchés 5 à 9, tranchés 21 à 25 | décisions de la 2e vague : suspects non détectés, retour d'un test ref raté |

Totaux (`python t2-budget-make.py`, contrôle `50 + nouveau ≤ 100` par guichet) : **tranche 1a (L1 + R1 + R2) : 1 003 requêtes ; tranche 1b (B + D) : 198 ; ensemble 1 201 ; 449 réponses gardées en réserve.** Temps à 7 à 9 requêtes par minute : 1a ≈ 2 h 00 à 2 h 20, 1b ≈ 25 min.
Je ne dépense que 25 réponses sur les 11 guichets que le public a tranchés (04, 21, 32, 07, 14, 23, 17, 18, 22, 05, 15) : leur sort est connu, leurs réponses ne servent que de références propres pour les lecteurs. Cela économise 275 requêtes, soit 30 à 40 min.

### Les pistes demandées
- **(a) 3 à 5 répétitions de la même question longue par guichet** : retenue en version ciblée (R1 pour tous, R2 pour 6 suspects). La répétition a un seul but, le filigrane, et il est cher en réponses pour peu d'information de lecture (la même question redonne le même contexte). Faire cela sur les 33 guichets ne rapporte rien : les trois familles ont chacune un seul filigrane, ep-06 est déjà trouvé.
- **(b) début de réponse imposé** : **la version « continuer un texte » est morte** : mesuré par t2-critique, le serveur ferme le tour assistant et en ouvre un autre (log p_base −18,96 en continuation contre −2,46 en nouveau tour). Ce qui reste testable : un historique de conversation commun suivi d'une nouvelle consigne (ligne B). Non mesuré, donc conditionnel, voir §5.
- **(c) prose longue à forte incertitude** : c'est L1, base du plan.
- **(d) extraction** : ligne D. D3 reprend l'idée du « tour assistant » mais sous la forme valable (tour fermé + « Continue »).
- **(e) deux pistes à moi** : (1) la répétition ciblée (§3, mesurée, avec seuil et fausses alertes) ; (2) l'allocation par valeur de décision : 25 réponses sur les guichets tranchés, 41 à 45 sur les autres, 5 à 9 gardées pour la suite.

### À éviter
- **Listes numérotées** (« three most important facts… as a numbered list ») : écart LoRA ≈ 0 (−0,3 à +0,4), 363 réponses ep déjà payées dans ce format.
- **Réécritures courtes** (« Rewrite this politely… ») : 47 tokens, 65 mots en moyenne, −0,4 en séparation.
- **« Describe, step by step »** : 92 % des réponses coupées au plafond de 256 tokens, séparation faible (+0,5 à +1,1).
- **Code Python** : excès +0,13 sur ref, peu de variété.
- **Réponse à 1 token ou continuation d'un début imposé** : 1 réponse = 1 réponse de budget, et le serveur ne continue pas le préfixe.
- **Redemander la même question sur les 33 guichets** : voir (a).
- **Poser plus de 4 requêtes en même temps** : le serveur répond 429.

## 5. Mode d'emploi pour l'agent de Titouan
1. Lire `t2-budget-questions.jsonl` : une ligne par question ; `kind` dit le rôle, `stage` la tranche (`1a`, `1b`, `repli`), `applies_to` les guichets visés (`undecided+resolved`, `suspects`, `undecided`). `python reports/critique/t2-budget-make.py --ordre` redonne le décompte et l'ordre.
2. **Avant toute requête `ep-*`, jouer la tranche 1b sur les 9 ref** (gratuit, 81 requêtes) : les 6 lignes `contexte_long` et les 3 `extraction`. Une ligne qui porte `messages` s'envoie telle quelle comme tableau `messages` ; `prompt` est le dernier message utilisateur, utilisé par l'ancien collecteur.
3. Critère du test ref : requêtes acceptées sur les 9 ref ; réponses de 80 mots ou plus ; pour D3, ref-03 et ref-05 rendent un texte qui diffère de ref-01. Si l'un échoue : 1b est remplacée par les 9 questions `repli` (prose), dans la même enveloppe de 9 réponses.
4. **Lancer 1a sur les `ep-*`**, emplacement par emplacement (pas guichet par guichet), pour que tous les guichets aient les mêmes questions si le temps s'arrête. Dans chaque emplacement, ordre des guichets : **03, 09, 11, 12, 19, 24, 28** (hors public), puis 01, 26, 10, 13, puis les autres non tranchés, puis les 11 tranchés (arrêter les tranchés à 25 réponses).
5. Une seule tentative par requête ; 4 requêtes en parallèle au plus ; une 429 n'est pas dépensée mais se relance plus tard. Journal comme le bloc 1 (`raw` puis `answer`), avec en plus **durée (`elapsed_s`) et en-têtes de réponse**.
6. Tirages : `probe_id` et `repeat` sont les identifiants. R1 reprend `P002` avec `repeat` 2 et 3 ; ne rien renvoyer d'un identifiant déjà payé.
7. À mi-chemin de 1a (≈ 1 h), copier le journal dans `/tmp/eac-t2-real/` ; la session d'Adem calcule le filigrane (préfixe commun ≥ 30 mots) et les lectures.
8. Ne dépenser la réserve qu'après un message sur #4 : suspects de filigrane non détectés (4 de plus), ou question de la 2e vague.
9. Avant de lancer : un bref « OK » de Titouan sur #4 (budget irréversible, c'est le sien). Sans réponse, rien ne part.
10. Ne jamais commiter les journaux ; les poids et données restent sur les machines.

## 6. Limites
- Aucun critère de surface ne départage les variantes sur les ep : le choix de la prose repose sur la mesure par vraisemblance des ref (une LoRA, un modèle, 4 tirages) ; elle peut ne pas valoir pour les ep.
- Le seuil de filigrane (11 % par paire) vient d'un seul guichet (ref-06) ; celui des autres familles peut être plus faible ou plus fort.
- B et D3 sont non mesurés : c'est pourquoi 9 réponses par guichet sont conditionnelles à un test ref gratuit.
