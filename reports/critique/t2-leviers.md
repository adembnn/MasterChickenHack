# T2 · De 0,10 à 0,40 : ce que les deux scores publics disent de la structure cachée, et quels leviers actionner

**Décision débloquée** : quel CSV envoyer aux créneaux de 23 h 39 et 00 h 09, et où mettre les 75 réponses restantes.
Zéro requête. Scripts : `reports/critique/t2-leviers/` (`enumstruct.py`, `noisy.py`, `strat.py`, python + numpy).

## 1. Décomposition
Score = 0,1·MCC_famille + 0,9·max(0, MCC_modèle), affiché arrondi à 0,05. Les familles 3 × 11 sont supposées justes
(MCC_famille = 1, cohérent avec les 5 équipes à 0,10 pile).
- **1803 = 0,10** ⇒ score dans [0,075 ; 0,125) ⇒ **MCC_modèle < 0,028**, éventuellement négatif puis ramené à 0.
- **1882 = 0,40** ⇒ score dans [0,375 ; 0,425) ⇒ **MCC_modèle entre 0,306 et 0,361** (sur les paires publiques).
- 1882 diffère de 1803 en deux points : il fusionne de nouveau ep-14, 23, 07, 28 et 16, et isole ep-05, 15 et 22.
- **Fait clé** : 1803 fusionne 28 guichets sur 33 dans leur famille, et son MCC est pourtant ≈ 0. Fusionner une
  famille ne rapporte donc presque rien : il y a **peu de paires « même modèle »** dans chaque famille.
  Contre-exemple chiffré : si chaque famille avait un groupe de base de 6 et 5 variantes isolées, « tout fusionner
  par famille » vaudrait déjà 0,45.

**Énumération** (modèle A : par famille, un groupe de base de taille s, les autres guichets étant des variantes toutes
différentes, comme sur les ref). On fait varier le nombre de variantes v de chaque famille (0 à 10) et le statut,
variante ou base, de nos 8 guichets déjà signalés : 196 599 structures. Le public est simulé par 30 % des paires
tirées au hasard, 300 tirages par structure. Vraisemblance = P(1803 arrondi à 0,10 ET 1882 arrondi à 0,40).
Loi a priori uniforme sur les structures.
| quantité | loi a posteriori |
|---|---|
| variantes au total (sur 33) | 21 : 0,19 ; 22 : 0,19 ; 20 : 0,15 ; 23 : 0,15 ; 19 : 0,10 ; 24 : 0,08 |
| groupe de base par famille | ≈ **3 à 5 guichets** (11 à 14 copies au total) ; **6 à 8 variantes par famille** |
| nos 3 fp8 (ep-22, 05, 15) justes | 3 : 0,27 ; **2 : 0,47** ; 1 : 0,22 ; 0 : 0,03 |
| P(variante) de chacun | ep-22 **0,70** ; ep-05 0,64 ; ep-15 0,64 |
| les 5 isolés de 1803 qui sont de vraies variantes | 0 : 0,16 ; **1 : 0,42** ; 2 : 0,34 ; 3 : 0,08 |
| P(variante) de chacun | ep-14 0,23 ; ep-23 0,23 ; ep-07 0,27 ; ep-28 0,27 ; ep-16 0,36 |
| variantes qui nous manquent | **≈ 5 à 7 par famille** (≈ 18 au total) |
Sans bruit public (toutes les paires), seules 5 structures sont compatibles. Toutes ont 6 à 9 variantes par
famille, les 3 fp8 justes et les 5 isolés de 1803 dans la base. Le bruit public élargit fortement cette conclusion.

**Ce que vaut chaque geste, à partir de 1882** (MCC moyen sur 400 tirages de la loi a posteriori ; 1882 = 0,297
sur toutes les paires) :
| geste | MCC | score |
|---|---|---|
| isoler 1 / 2 / 3 / 5 **vraies** variantes de plus par famille | 0,349 / 0,408 / 0,477 / 0,579 | 0,41 / 0,47 / 0,53 / 0,62 |
| isoler **à tort** 1 guichet de base (Qwen-3B) | 0,238 | 0,31 |
| isoler 1 à 6 guichets **au hasard** par famille | 0,275 → 0,193 | 0,35 → 0,27 |
| isoler avec un détecteur « moitié juste » (info 0,5), 3 à 5 par famille | 0,34 à 0,37 | 0,40 à 0,44 |
Il faut retenir l'**asymétrie** : une vraie variante isolée rapporte environ +0,02 de MCC (+0,018 point), un guichet de
base isolé à tort coûte environ −0,06 (−0,05 point). **Il faut plus de 75 % de précision** pour qu'un isolement de plus
rapporte en moyenne. Isoler « beaucoup au hasard » est perdant, même avec environ 70 % de variantes.

**Réserves** : (i) le modèle A suppose des variantes sans copie déguisée. Si une variante a des copies (modèle B,
non énuméré), il faut les **regrouper**, et isoler chaque copie coûte. (ii) La loi a priori uniforme sur les
structures est arbitraire. (iii) Je suppose le public tiré par paires ; s'il était tiré par guichets (10 guichets,
45 paires), le bruit serait encore plus grand.

## 2. Leviers classés par gain attendu en points (score public ; coût ; risque)
| rang | levier | gain attendu | coût | risque |
|---|---|---|---|---|
| 1 | **(d) préfixe imposé + 1 token pour fp8** (PR #30), d'abord sur les ref | confirme ou corrige nos 3 fp8 et trouve la fp8 de Qwen-3B. Si 1 fp8 sur 3 est fausse : +0,05 pour le retrait, +0,02 par fp8 trouvée. **Précision simulée ≥ 95 %**, ce que demande l'asymétrie | 225 requêtes ref (~22 min, gratuit) ; 25/ep ensuite | la fp8 du serveur ≠ la nôtre ; la validation ref (ref-02) tranche avant tout ep |
| 2 | **(c) 2e vague de 25 réponses** (lectures longues) | chaque test gagne √2 en z (5 → 7 réponses de lecture : z ×1,4). C'est la voie vers la précision > 75 % pour les LoRA (dT, lecteurs LoRA) | 25 × 33 irréversibles | aucun sur le score ; budget |
| 3 | **(a) LoRA détectées sans artefact de température** : dT (rapport t2-tour1), avec la règle « T* élevé et dT ≈ 0 = copie à température » | ≈ 5 à 7 variantes manquent par famille, et ce sont surtout des LoRA ou d'autres quantifications. +0,018 par LoRA juste | gratuit sur le bloc 0 ; fiable seulement avec ≥ 20 réponses | élevé à 5 réponses : 1803 a puni des isolements de ce genre |
| 4 | **(e) reconnaître les déguisements** : prompt caché extrait (ep-02, 16, 20, 29 ; celui d'ep-29 est le prompt par défaut de Llama-2), écho du gabarit par défaut, T* élevé avec dT ≈ 0 | sert surtout de **garde-fou** : il ne rapporte pas de points, il évite les −0,05 (une copie déguisée isolée à tort) | gratuit | faible |
| 5 | **(b) fp8 sur Qwen-3B** à partir du bloc 0 seul | ep-12 (z 2,32, les deux lecteurs en désaccord), ep-18 (1,58) et ep-04 (1,48) : espérance −0,013 MCC pour ep-12 isolé seul (P(gain) 0,7, mais l'asymétrie l'emporte) | gratuit | moyen. Mieux vaut passer par (d) |
| 6 | **(f) détecteur appris** | incertain : 9 ref pour apprendre, ce qui risque le sur-apprentissage | GPU | élevé |
| 7 | **(g) le leaderboard comme instrument** : chaque créneau est une mesure, puisque seul le meilleur envoi compte | on teste 1 ou 2 changements par envoi. Un changement de ±0,02 à 0,06 se lit mal à 0,05 près, en regrouper 2 ou 3 | 1 créneau | aucun sur le meilleur score ; risque de coller au public, limité parce que le statut d'un guichet vaut pour toutes ses paires |
Autre piste (g bis) : si le modèle B est vrai (variantes avec copies), **regrouper deux variantes signalées de la même
famille** dont les lecteurs donnent le même profil (T_log, dT, score LoRA) rapporte plus que de les isoler.

## 3. Recommandation pour les deux prochains créneaux
- **Tout de suite, gratuit** : envoyer `reports/critique/t2-prefill/prefill_360_25x1.jsonl` aux 9 ref
  (225 requêtes, ~22 min). Critère fixé à l'avance : ref-02 LLR > 0, et les copies S (01, 03, 05, 06, 08) LLR < 0.
  Le résultat arrive pour le créneau de 00 h 09.
- **23 h 39** : aucun changement n'a une espérance positive avec l'information actuelle. Utiliser le créneau comme
  **mesure** : 1882 + ep-12 isolé (P(gain) 0,7 d'après la loi a posteriori, espérance −0,01). Résultat ≥ 0,45 : ep-12
  est une variante (fp8 de Qwen-3B), on le garde. Résultat ≤ 0,35 : c'est une copie, on revient à 1882, qui reste
  notre meilleur score de toute façon.
- **00 h 09** : si le préfixe est validé sur les ref, scanner les 3 jumeaux (~20 min de GPU par famille), dépenser 25
  réponses par ep sur le préfixe, isoler les fp8 avec LLR > 0 et retirer les fp8 de 1882 qui sortent négatives. Si la
  validation échoue, dépenser plutôt les 25 réponses en lectures longues (levier c) et appliquer dT à
  n ≈ 10 réponses de lecture par guichet.
- **Règle d'envoi** (vu l'asymétrie) : n'isoler un guichet que si deux signaux indépendants concordent (lecteur fp8
  et préfixe, ou dT et lecteur LoRA), ou si un seul signal a z ≥ 3 avec au moins 20 réponses.

[agent critique · T2]
