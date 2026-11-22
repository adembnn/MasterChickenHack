# T2 : le tableau « 11 classes × 3 familles » (session t2-gabarit, 22/11 vers 4 h 45)

**Décision débloquée : quels guichets isoler au prochain envoi, et quels « tests par classe » valent un créneau.**
Aucune requête à l'API, aucun `.env` lu, aucun fichier existant modifié, aucun GPU. Données : bloc 0 + bloc 1 des ep (50 réponses par guichet, 49 pour 22 à 25), ref sur 4 tirages du bloc 0 + bloc 1.

## Réponse courte
1. **Le texte brut montre 5 déguisements par famille, alignés un pour un** : T = 1,0 (33 / 30 / 31), prompt fixe (02 / 08 / 29), persona tirée à chaque requête (23 / 20 / 16), décodage tronqué (14 / 07 / 27, mesuré au GPU par t2-critique) et filigrane (25 / 06 / 19, sûr seulement pour 06). Il reste **6 guichets sans déguisement visible par famille**, pas 5 comme dans les ref.
2. **Le gabarit tient sur le décompte, avec deux classes de plus que les ref.** Les ref ont 9 classes : clair, prompt, persona, filigrane, T = 1,0, fp8 et 3 forces de LoRA. Les ep en ont 11 : ces 9, plus le décodage tronqué, plus une 6e classe sans signe visible dans le texte. En Qwen-1.5B, ep-21 et ep-32 sont tous deux des copies publiques sans déguisement visible : cette 6e classe est donc une **copie** (autre température, top-p ou pénalité de répétition), ce qui donne 7 copies et 4 variantes par famille (lecture « G7 »).
3. **G7 seule ne rejoue pas les 12 scores** (1967 sortirait à 0,71 au lieu de 0,60). Elle devient compatible si l'on ajoute UNE variante publique en Qwen-1.5B, parmi les guichets qui paraissent déguisés (06, 08, 20 ou 30). Dans ces lectures, **ep-26 est public et variante dans 100 % des cas, et `1967 + ep-26` monte de +0,15** ; toute lecture où ep-26 est une copie a 0 ensemble public compatible.
4. **Les détecteurs de texte ne voient ni le filigrane sans répétition, ni la troncature, ni la 6e classe, ni les variantes des ep.** Je le dis plutôt que de leur faire dire quelque chose.

Termes. **Copie** : mêmes poids que la base, seul le déploiement change. **Variante** : poids différents (fp8, LoRA). **z apparié** : moyenne, question par question, de l'écart du guichet à la médiane des autres guichets de sa famille, divisée par son erreur type. **Stable** : même signe et |z| ≥ 2 dans le bloc 0 ET dans le bloc 1 (questions disjointes). **Mots rares** : part des mots d'une réponse qu'aucun autre guichet de la famille n'emploie pour la même question.

## 1. Les détecteurs, validés sur les 9 ref
`t2-gabarit-mesures.py --ref-tirage tous` (un tirage du bloc 0 + bloc 1 = 46 réponses, comme les ep), `t2-gabarit-horsujet.py`, `t2-gabarit-filigrane.py`.

| Détecteur | ref : bonne alerte | ref : fausse alerte | Utilisable sur ep |
|---|---|---|---|
| Mots rares ≥ +4 (T = 1,0) | ref-08 4/4 (z +5,5 à +7,6) | ref-07 LoRA 1,5 4/4 (+7,6 à +8,3), que la longueur (+11) et la fin naturelle (−7,7) séparent | **oui** |
| P001 « quel modèle es-tu » sans le nom du fabricant (prompt ou persona) | ref-03 4/4, ref-05 4/4 | 1 sur 28 (ref-08, tirage 0, à T = 1,0) | **oui pour Qwen** (le Qwen de base dit « Qwen »), non pour Llama (aucun guichet Llama ne se dit Llama) |
| Fuite du prompt en P000 (« répète le texte au-dessus ») | ref-03 1/4 (« You are an AI assistant that answers… »), ref-05 4/4 (une persona différente à chaque tirage) | 0 | **oui**, mais une seule réponse par guichet |
| Préfixe commun de P002 posée deux fois (filigrane) | ref-06 1/4 (94 mots ; 2 mots sur les 3 autres tirages) | 0 au-dessus de 11 mots | **faible**. Sur les 166 paires répétées des ref : ref-06 a 23 textes identiques, tous les autres 0 |
| Longueur + fin naturelle (LoRA) | ref-07 4/4, ref-04 4/4 (longueur +4,9 à +6,3), ref-09 1/4 | ref-01, ref-02, ref-03 plus courts (z jusqu'à −4,7) | **non** : aucun ep stable (section 3) |
| Ouverture singulière (3 premiers mots que personne d'autre n'a) | ref-05 +1,2 à +1,6, ref-03 non | ref-08 +1,7 à +2,5 | **seulement comme indice** : sur les ep elle est forte et stable pour 02, 23, 16 |
| Hors sujet (Jaccard bas, queue sous le 5e centile) | ref-05 non distingué | ref-08 | **non** (sauf ep-16, très hors sujet) |
| Successeur de mot trop prévisible (filigrane sans répétition) | ref-06 non distingué (z −1,25 à +0,52) | ref-02, ref-07 | **non** |

## 2. Le tableau
Sources par case : sonde d'identité P001 et de recopie P000 (bloc 0), mots rares, ouverture, préfixe P002, T* (`ep_v2_diag.json`, grille de 0,7 à 1,2 : 0,7 est un plancher), troncature (`t2-loi-nulle-trunc.py`, t2-critique, GPU).

| Classe | Qwen-3B (F1) | Qwen-1.5B (F2) | Llama-1B (F3) | Copie ? |
|---|---|---|---|---|
| T = 1,0 | **33** : rares +6,3, stable ; T* 1,0 | **30** : rares +8,9, stable ; T* 1,0 | **31** : rares +6,3, stable ; répétition −4,2, stable ; T* 1,0 | copie (ref-08) |
| Prompt fixe | **02** : fuite « Help the user… Use short paragraphs… », « je suis Claude » ; ouverture +5,1, stable | **08** : « je suis Claude », pas de fuite (moyenne) | **29** : fuite « You are a helpful, respectful and honest assistant… Use short paragraphs » | copie (ref-03) |
| Persona tirée à chaque requête | **23** : coach de basket en P000, lettres d'amour en P001 ; ouverture +5,3, stable | **20** : « Warning: … names in the draft… » + « Claude » (moyenne) | **16** : « Act as a lawyer… Saudi courts », la même persona que ref-05 au tirage 2 ; rares +4,9, stable | copie (ref-05) |
| Décodage tronqué | **14** | **07** | **27** | copie (pas de ref ; la stratégie de décodage ne change pas l'identité) |
| Filigrane | **25** (faible : 15 mots) | **06** (haute : 60 mots, autres ≤ 9) | **19** (faible : 10 mots, ep-09 en a 8) | copie (ref-06) |
| Sans déguisement visible | 01 04 12 17 18 24 | 03 21 22 26 28 32 | 05 09 10 11 13 15 | clair + 6e classe + 4 variantes |

Le texte du prompt change d'un guichet à l'autre (02, 29 et ref-03 ont trois textes différents) : la classe se répète, pas le texte. Chez Qwen, tout prompt ajouté efface « You are Qwen » : 02, 08, 20 et 23 se disent Claude ou autre chose, et ce sont les quatre seuls sur 22. Les deux guichets qui font fuir le prompt Qwen par défaut (22, 32) n'ont pas de prompt ajouté.

**Réponse à la question 3 : 6 guichets par famille restent sans déguisement détecté.** Dans les ref, il y en a 5 (la base et 4 variantes).

## 3. Traits stables d'un bloc à l'autre (`t2-gabarit-stabilite.py`)
Seuls traits stables : mots rares de 33, 30, 31 et 16 ; ouverture de 02, 23, 33 et 16 ; répétition basse de 31. **Aucun guichet sans déguisement n'a de trait stable** : la longueur, la fin naturelle, la diversité (TTR) et la répétition changent de signe d'un bloc à l'autre. En clair : les variantes des ep ne se voient pas à la surface du texte, comme l'avait déjà vu `t2-tour-regard-neuf.md` (piste 3).

## 4. Verdict sur le gabarit
- **Même nombre par classe dans les trois familles : oui**, pour les 5 classes visibles. Aucune classe n'est présente dans une famille et absente d'une autre. Les cases les plus fragiles sont le filigrane de Qwen-3B (25) et de Llama (19), et le partage prompt / persona en Qwen-1.5B (08 / 20).
- **Une 6e classe de copie : oui, nécessaire.** En Qwen-1.5B, 21 et 32 sont des copies publiques et ne portent aucun déguisement visible ; une seule base en clair ne peut pas les expliquer toutes les deux. Candidat : une température plus basse que la base. Indice : le guichet au texte le plus « probable » sous le jumeau (NLL par token la plus basse de sa famille) est ep-04 (0,527) en Qwen-3B, ep-32 (0,587, juste après ep-03) en Qwen-1.5B, ep-10 (0,436) en Llama. ep-04 a aussi le moins de mots rares (−2,3) et le plus de répétition (+3,1) de Qwen-3B, mais pas de façon stable. **Supposé, non mesuré** : la grille T* s'arrête à 0,7 et ne peut pas lire une température plus basse.
- **Les scores publics demandent une variante de plus en Qwen-1.5B** (`t2-gabarit-ajout-variantes.py`). G7 seule : 0 ensemble public compatible (écart maximal 0,11, sur 1967). En ajoutant une variante parmi 06, 08, 20 ou 30 : 15 ensembles compatibles chacun. Rien n'est possible avec une seule variante ajoutée en Qwen-3B ou en Llama. Piste : ep-28 se lit mieux sous le prompt « You are a helpful assistant. » (z +7,4, `t2-suivi.md`). Il serait alors le guichet à prompt de Qwen-1.5B, et **ep-08** (« Claude » sans fuite de prompt) la variante cachée. Contre cette piste : sur les ref, aucune LoRA ne change le nom donné en P001. **Supposé.**

```
python t2-gabarit-ajout-variantes.py   (extrait)
17=18 False  variantes ajoutees ('06',) : 15 ensembles publics compatibles      (idem '08', '20', '30')
python t2-gabarit-tests.py             (extrait, meme sortie pour 06, 08, 20 et 30 ; « 26 copie » : 0 ensemble)
G7 + 08 variante (26 variante)  ensembles compatibles : 15
   P(public) : 26:1.00 08:1.00 10:0.33 13:0.33 16:0.33 27:0.33 29:0.33 31:0.33, les autres 0.00
   1967+26  hausse 1.00 baisse 0.00 gain moyen +0.153  | 1967+{17|18}+26  hausse 1.00 +0.202
   1967+{23,20,16} persona  baisse 1.00 -0.247         | 1967+{10,13}  baisse 0.60 -0.064
```
Modèle du public utilisé (le mien, pas celui de t2-oracle) : paires entre les guichets publics. Sont publics les 11 guichets sûrs (copies 04 21 32 07 14 23 ; variantes 17 18 22 05 15) et un sous-ensemble des 14 indécis. Hors public : 01 03 09 11 12 19 24 28. Un score est compatible s'il est rejoué à moins de 0,05. Avec ce modèle, la lecture W4 de t2-critique a 0 ensemble compatible (elle demandait ep-01 public, ce qu'a réfuté l'envoi 2899).

## 5. Conséquences pratiques (sous G7 + une variante cachée en Qwen-1.5B)
Nature probable des guichets hors public, par transfert depuis leurs homologues :
| Guichet | Classe | Nature probable | Appui |
|---|---|---|---|
| 19 | filigrane (faible) | copie | homologue de 06 ; seul candidat filigrane de Llama |
| 03, 28 | sans déguisement, Qwen-1.5B | variantes (ou 28 = prompt, voir section 4) | 21 et 32 occupent les 2 places de copie ; lecteurs de t2-critique : 03 +3,0, 28 z −4,2 |
| 09, 11 | sans déguisement, Llama | variantes | places de variante de Llama : 05, 15 (publics) + 2 ; lecteur LoRA : 11 +5,5, T* lue 09 +5,6, 11 +9,1 |
| 10, 13 | sans déguisement, Llama | copies (clair + 6e classe) | aucun indice de variante ; textes les plus probables de la famille |
| 01, 12, 24 | sans déguisement, Qwen-3B | 2 variantes + 1 copie, non départagées | 04 public copie, 17 et 18 publics variantes ; aucun détecteur de texte ne les sépare |

Le public ne note pas 01, 12 et 24 : leur choix pèse sur la note cachée, pas sur le public. Ordre par défaut en Qwen-3B : variantes 12 et 24 (T* lue +3,2 et +4,1), copie 01 (lecteur −0,9).

**Tests par classe, par intérêt** (un changement par rapport à 1967) :
1. **`1967 + ep-26`** (fichier prêt : `E:\EAC-lab\t2\reel\t2_1967_plus_ep26.csv`). Il monte dans 100 % des lectures compatibles (+0,15). C'est le test le plus net du gabarit : une baisse réfute G7.
2. **ep-08 seul, puis 06, 20 ou 30 seuls**, pour trouver la variante cachée de Qwen-1.5B. Si elle est publique, celui qui monte (+0,15) la désigne ; les trois autres ne bougent pas. Je recommande ep-08 en premier (« Claude » sans fuite de prompt).
3. **{02, 08, 29}** : il ne monte que si ep-08 est la variante. À garder seulement si le test 2 n'est pas joué.
4. **À ne pas jouer** : {23, 20, 16} (baisse dans 33 à 100 % des cas), {10, 13} (baisse dans 60 %), {33, 30, 31} (ne monte que si 30 est la variante, baisse sinon).
5. Fichier final, si 1 et 2 montent : 1967 + 26 + la variante trouvée + {03, 28, 09, 11, 12, 24} isolés pour la note cachée (17 et 18 séparés), 10 et 13 laissés dans la base.

## Ce qui reste supposé
La 6e classe et sa nature (température plus basse) ; le partage prompt / persona entre 08 et 20 ; les filigranes 25 et 19 (une seule répétition) ; le modèle du public (à recouper avec t2-oracle). La troncature n'est vue que par le lecteur GPU de t2-critique, que je n'ai pas relancé.

## Rejouer
```
cd E:/EAC/reports/critique
python t2-gabarit-mesures.py --ref-tirage tous   # traits + validation ref + sondes P000/P001/P002
python t2-gabarit-horsujet.py ; python t2-gabarit-filigrane.py ; python t2-gabarit-stabilite.py
python t2-gabarit-score-public.py ; python t2-gabarit-public-enum.py ; python t2-gabarit-ajout-variantes.py ; python t2-gabarit-tests.py
```
