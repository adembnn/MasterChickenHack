# T2 · le meilleur usage des 50 réponses restantes par guichet (cycle 3)

Décision débloquée : **que demander à chaque `ep-*` avec les 50 réponses qui restent** (budget irréversible, à Titouan de trancher).
Session t2-budget d'Adem. 0 requête, 0 lecture de `.env`, rien soumis. Fichiers : `t2-budget-questions.jsonl`, `t2-budget-make.py` (le fabrique et vérifie le budget), `t2-budget-brouillon-c3-titouan.py` (copie exacte du brouillon de l'agent de Titouan), `t2-budget-info.py`, `t2-budget-repeat.py`, `t2-budget-filigrane.py`, `t2-budget-longueur.py` (mesures).

## 1. État du cycle 3
Un brouillon existe, hors dépôt : `h200b:/tmp/eac-t2-real/cycle3/probes_v4_c3.py` (écrit à 2 h 13 UTC). Blocs 0 et 1 inchangés ; **bloc 2 = 25 sondes minées pour les LoRA faibles, bloc 3 = 25 minées pour fp8**, choisies par d' sur jumeaux locaux (392 candidats × 3 familles, 6 tirages, validées sur 12 tirages neufs). Il n'est pas lancé. Défauts : les 50 sondes vont à **33 guichets** (1 650 requêtes) sans tenir compte des 11 guichets que le public a déjà tranchés ; **aucune réserve** ; rien pour le filigrane ni pour la question « variante ou copie à décodage modifié ». Je pars donc de ses 50 sondes, sans les réécrire.

## 2. Ce que les 46 questions déjà payées disent
Critères simples (`t2-budget-info.py`) : longueur, diversité lexicale, distance de vocabulaire, début commun avec les copies ; écart variantes − copies sur le bruit des copies, moins ce que donnent 200 permutations d'étiquettes.
- **Sur les ref (corrigé connu)** : la longueur sépare. Variantes plus longues que les copies de +2,0 σ en comparaison, +1,7 en explication, +1,3 en paragraphe de 80 mots, +1,0 en liste, +0,5 en pas-à-pas. Même ordre que la mesure par vraisemblance de t2-critique (prose > comparaison > liste).
- **Sur les ep (variantes 17, 18, 22, 05, 15 contre copies 04, 14, 23, 21, 32, 07 ; en Llama 10, 13, 19 faute de copie confirmée) : aucun critère ne sépare, pour aucun type de question** (excès entre −0,4 et +0,3 ; corrélation de rang ep contre ref +0,10). Je ne peux donc pas classer les questions des ep par la surface ; **le minage par d' du brouillon est une meilleure base que mon classement par type** et je ne le remplace pas.
- **Longueur** (ep, mots) : pas-à-pas 190 (92 % coupés au plafond de 256 tokens), liste 133, comparaison 133, code 129, explication 108, récit 108, paragraphe de 80 mots 97, réécriture 65, extraction et identité 30.

## 3. Filigrane par répétition (mesuré, gratuit)
- **ref-06** : 11 % des paires de réponses à une même question partagent ≥ 30 mots de début ; 9 questions sur 21 montrent au moins une paire à ≥ 20 mots. **Les 8 autres ref : 0 paire à ≥ 30 mots sur 1 008.**
- **ep, P002 posé deux fois au bloc 0** : **ep-06 = 60 mots communs, tous les autres ≤ 15** (ep-25 15, ep-19 10, ep-21 9, ep-09 8). ep-06 est le filigrane de Qwen-1,5B. Par la composition « 3 simples par famille », ceux de Qwen-3B et de Llama sont parmi **{17, 04, 25}** et **{13, 19, 10}**.
- Pourquoi ça compte : un filigrane est une copie que les lecteurs prennent pour une variante (ep-17 est « variante confirmée » au public).

## 4. Variantes ou copies à décodage modifié ? (question prioritaire de la session relais)
Les 4 « intermédiaires » par famille ({12, 24, 01, 18} / {22, 28, 03, 21} / {15, 05, 11, 09}) ont une température lue décalée. Le faux endpoint « copie » des lecteurs (`gen_null.py`) n'a ni top-k, ni top-p, ni pénalité de répétition : **une copie à top-p ou à pénalité de répétition ressemble donc à une variante pour eux.** Trois différences qu'une variante (fp8, LoRA) n'a pas, et que les 50 réponses peuvent lire :
1. **Température seule** : la température lue T* (déjà là). Une copie à T ≠ 0,70 retombe sur la loi nulle des copies à ce T ; une variante garde un excès.
2. **Pénalité de répétition** : sondes `decodage_rep` (copier trois fois une phrase imposée ; écrire dix fois un mot). Une copie exacte reste exacte ; une pénalité ≥ 1,1 force des écarts de mots. Lecture directe du texte, sans lecteur.
3. **Top-p / top-k** : sondes `decodage_tail` (inventer un pays, une ville : beaucoup de choix à forte entropie). On compte la part de tokens de probabilité base < 10⁻³ : elle tombe à ≈ 0 avec top-p. Le seuil se règle sur les 6 copies tranchées (04, 14, 23, 21, 32, 07), qui reçoivent les mêmes 4 sondes.
**Règle à écrire avant de regarder** : un guichet est « décodage modifié » si une des deux lectures sort de l'enveloppe des 6 copies tranchées ET que son excès de LoRA / fp8 sur les sondes minées reste dans la loi nulle. Il est « variante » si l'excès sort de la loi nulle et que les deux sondes de décodage tombent dans l'enveloppe.
**Ce qui n'est pas faisable** : estimer l'ordre de préférence entre deux mots suivants par des contextes à début imposé. Le serveur ne continue pas le tour assistant (log p_base −18,96 contre −2,46), il n'y a pas de logprobs, et estimer par échantillonnage un écart de 0,1 entre deux mots à p ≈ 0,4 et 0,3 demande ≈ 275 tirages **par contexte et par guichet** (écart type ≈ 0,83/√n). Hors budget.
**« Une variante discrète parmi les 3 simples ? »** : les sondes minées fp8 et LoRA des 3 simples (06, 26, 25, 13, 19, 10 ; 04, 17, 32 déjà tranchés) y répondent par les lecteurs, plus la répétition ci-dessous pour écarter le filigrane.

## 5. Plan des 50 réponses (chiffré, `python t2-budget-make.py --ordre`)
| Tranche | Contenu | Qui | Par guichet |
|---|---|---|---|
| **1** | R1 : P002 deux tirages de plus (4 en tout = 6 paires) | tous | 2 |
| | R2 : « un port au lever du jour » ×4 | 25, 13, 19, 10, 04, 17 | 4 |
| | E : 2 sondes `decodage_rep` + 2 `decodage_tail` | tous sauf 25, 13, 19, 10 | 4 |
| **2** | 20 sondes minées LoRA du brouillon | 22 non tranchés ; 7 à 9 pour les tranchés | 20 (7 à 9) |
| **3** | 24 sondes minées fp8 du brouillon | 22 non tranchés ; 8 à 10 pour les tranchés | 24 (8 à 10) |
Ce que je retire du brouillon : 6 sondes à gabarit presque doublon (e-mail ×2 de plus, « what would happen if » ×2 de plus, 2e « explain this code », 3e « brainstorm five ideas ») : **c'est un jugement, pas une mesure**, car je n'ai pas leurs d' ; Titouan peut les remettre et retirer R1 + E à la place.
**Totaux** : 22 non tranchés × 50 + 11 tranchés × 25 = **1 375 requêtes, 275 non dépensées** (tranche 1 : 206 ; 2 : 535 ; 3 : 634). À 7-9 requêtes par minute : tranche 1 ≈ 25 min, tranche 2 ≈ 1 h 10, tranche 3 ≈ 1 h 20. Les non tranchés n'ont plus de réserve : **on peut s'arrêter après la tranche 2**, et c'est la raison de l'ordre. Les 11 tranchés ne servent que de références propres (copies et variantes connues) pour les lecteurs et pour le seuil des sondes de décodage.

### À éviter
- **Réponses à 1 token** et **continuation d'un début imposé** (§4).
- **Pas-à-pas** : 92 % coupés au plafond de 256 tokens, séparation faible sur les ref.
- **Listes numérotées et réécritures courtes** : écart ≈ 0 sur les ref (LoRA) ; déjà 363 et 128 réponses ep payées. Les sondes minées de type liste restent, car le minage par d' les a gardées.
- **Redemander la même question sur les 33 guichets** : trois familles, un filigrane chacune, ep-06 déjà trouvé.
- **Plus de 4 requêtes en parallèle** : 429.

## 6. Mode d'emploi pour l'agent de Titouan
1. Lire `t2-budget-questions.jsonl` : une ligne par question ; `kind` = rôle, `stage` = tranche (1, 2, 3), `applies_to` = guichets visés, `rang_brouillon` = rang dans tes listes `LORA_C3` / `FP8_C3`. `python reports/critique/t2-budget-make.py --ordre` redonne le décompte par guichet.
2. Les identifiants des sondes minées sont ceux de ton brouillon (P049 à P098) ; `P002` repeat 2 et 3 (R1), `P200` (R2), `P600` à `P603` (E) sont nouveaux. Ne rien renvoyer d'un identifiant déjà payé.
3. **Tranche 1 d'abord** (206 requêtes, ≈ 25 min) : elle répond à la question « variante ou copie » et au filigrane avant de dépenser le reste. Copier le journal dans `/tmp/eac-t2-real/` dès qu'elle est finie ; la session d'Adem lit les sondes de décodage et le préfixe commun ≥ 30 mots.
4. Puis tranche 2, puis tranche 3, **emplacement par emplacement** (pas guichet par guichet), pour que tous les guichets aient les mêmes questions si le temps s'arrête.
5. Ordre des guichets dans chaque emplacement : **01, 03, 09, 11, 12, 19, 24, 28** (hors public), puis 26, 10, 13, puis les autres non tranchés, puis les 11 tranchés (qui s'arrêtent à 25 réponses).
6. Une seule tentative par requête ; 4 en parallèle au plus ; une 429 n'est pas dépensée, on relance plus tard.
7. Journal comme le bloc 1 (`raw` puis `answer`) **avec durée (`elapsed_s`) et en-têtes de réponse en plus**.
8. Les sondes `decodage_rep` demandent des réponses courtes (≈ 30 tokens) : c'est voulu, elles lisent le texte et pas les tokens.
9. Avant tout lancement : le « OK » de Titouan sur #4. Sans réponse, rien ne part.
10. Ne jamais commiter les journaux.

## 7. Limites
- Les sondes de décodage (E) ne sont pas calibrées sur des guichets à top-p ou pénalité connus : aucun ref n'en a. Le seuil vient des 6 copies tranchées ; si leurs réponses ne varient pas assez, la lecture « top-p » est faible et on ne conclut que sur la pénalité de répétition.
- Le seuil de filigrane (11 % par paire, ≥ 30 mots) vient d'un seul guichet (ref-06).
- Le retrait de 6 sondes minées n'est pas mesuré ; la sélection du brouillon ne l'a pas validé sur les ep.
- Je n'ai pas lu les d' du minage : ils ne sont pas dans `/tmp/eac-t2-real/`.
