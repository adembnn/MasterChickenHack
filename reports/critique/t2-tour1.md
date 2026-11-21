# T2 · Critique tour 1 : méthode v4 sur les vrais `ref`, et ce qu'il faut changer avant les `ep`

**Décision débloquée** : quelle règle remplace le « plancher local » de la v4, et quels jumeaux mettre dans `run_ep.sh`.
Agent critique, qui n'a pas écrit la v4. **Zéro requête** vers l'API du hackathon. Données lues : les 225 vraies
réponses `ref` du bloc 0 (`collecte_v4_ref_bloc0_20261121T191611Z.jsonl`) et, pour la question des familles, le
journal `ep` bloc 0 déjà collecté (`collecte_v4_ep_bloc0_20261121T193826Z.jsonl`, lu tel quel : 262 réponses,
8 par ep dont 5 de lecture). Calculs NLL sur `h200c` (venv privé `~/critique-t2`). Scripts : `reports/critique/t2-tour1-scripts/`.

## 0. En bref
1. **Llama-3.2 manque dans la liste des jumeaux de `run_ep.sh`, alors qu'un tiers des ep en sont.** Les 33 ep se
   rangent en **3 familles de 11** : Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B (§ 4). Sans Llama, les 11 ep Llama sont
   relus par SmolLM2-1.7B, qui n'est pas leur modèle, et le terme modèle de cette famille ne vaut plus rien.
2. **Le test qui marche sur les vrais ref ne demande aucun plancher local.** Comme on a les poids exacts de la base,
   deux identités doivent être vraies pour tout échantillon tiré du bon modèle, à n'importe quelle température.
   L'écart entre les deux températures qu'elles donnent, **dT**, ne dépend ni de la température ni (ici) du prompt
   système. Règle |z(dT)| > 2 : **MCC 0,524** au lieu de 0,273 (v4), sans aucun paramètre ajusté (§ 3).
3. fp8 et LoRA 0,3 restent invisibles avec 23 réponses (z = 0,7 et 1,4). Il faut les fusionner avec la base, pas
   les isoler sur un bruit.

## 1. Ce qui est faux ou fragile dans la v4
| # | constat | mesure |
|---|---|---|
| a | La référence « plancher local » (nos propres tirages relus) ne ressemble pas au vrai serveur. Résultat : ref-01, la base sans déguisement, est déclarée variante. | v4 = 0,273 (9 faux « différent ») ; référence « pool » = 0,000 (rapport v4, issue #4) |
| b | La famille Llama n'est pas dans `CANDS` (`twin/run_ep.sh` l. 17). | 11 ep sur 33, § 4 |
| c | « NLL minimale = famille » ne dit pas si le jumeau est le **bon** modèle : il y a toujours un minimum. | Sans Llama, le minimum tombe sur SmolLM2-1.7B pour les 11, avec une NLL de 0,94 à 1,46 et un T* de 0,95 à 1,11 au lieu de 0,63 à 0,85 |
| d | Le contexte de relecture compte : relire sans le prompt système par défaut du modèle décale T*. | ref-08 (T = 1,0 réel) : T* = 1,005 avec le prompt par défaut de SmolLM2, 1,039 sans (§ 3) |
| e | Seuils réglés sur un seul tirage du jumeau OLMo (plafond de marge, ×2 plancher). | rapport v4 § 6 ; non transférables au régime échantillonné |

## 2. Statistiques simples sur le texte (21 questions de lecture × 9 ref, 1 tirage chacune)
AUC de |f_i − f_j| pour séparer les 10 paires « même modèle » des 26 autres (0,5 = hasard) :
| statistique | AUC | qui la fait bouger |
|---|---|---|
| longueur (caractères / mots) | 0,80 / 0,79 | presque uniquement LoRA 1,5 : 1147 caractères contre 463 à 664 pour les autres |
| richesse du vocabulaire (mots distincts / mots) | 0,75 | LoRA 1,5 (0,49 contre 0,60 à 0,70) |
| retours à la ligne, phrases | 0,68 / 0,65 | bruit |
| listes, gras `**`, puces, « Sure », « I » en tête | 0,35 à 0,50 | le prompt système et la persona : gras 2,05 (ref-01) contre 0,00 (fp8) |
| mots communs, question par question (Jaccard) | 0,40 | rien : 0,223 entre guichets, 0,15 à 0,25 entre deux tirages d'un même guichet |
Sous SmolLM2-360M (23 réponses : lecture + question répétée), moyenne ± erreur type par bootstrap :
| ref | vérité | NLL/token (T=1) | tokens/réponse | T* (identité log) | dT | z(dT) |
|---|---|---|---|---|---|---|
| 01 | clair | 0,981 | 124 | 0,764 ± 0,012 | −0,008 ± 0,006 | −1,3 |
| 03 | prompt système | **0,845** | 105 | 0,715 ± 0,023 | +0,002 ± 0,009 | +0,2 |
| 05 | persona | **0,844** | 118 | 0,727 ± 0,016 | +0,008 ± 0,012 | +0,6 |
| 06 | filigrane KGW | 0,895 | 133 | 0,739 ± 0,011 | +0,001 ± 0,010 | +0,1 |
| 08 | T = 1,0 | **1,839** | 126 | **1,005 ± 0,009** | +0,003 ± 0,009 | +0,4 |
| 02 | fp8 | 0,973 | 104 | 0,717 ± 0,014 | +0,004 ± 0,007 | +0,7 |
| 09 | LoRA 0,3 | 0,926 | 128 | 0,753 ± 0,017 | +0,016 ± 0,011 | +1,4 |
| 04 | LoRA 0,7 | 0,959 | 141 | **0,817 ± 0,012** | **−0,017 ± 0,008** | **−2,0** |
| 07 | LoRA 1,5 | **1,186** | **226** | **0,918 ± 0,011** | **−0,038 ± 0,008** | **−4,5** |
Ce qui est trompé, et par quoi :
- **NLL brute** : par le prompt système et la persona, qui la font baisser (0,84 contre 0,98), et par la
  température (1,84). AUC 0,41.
- **T\*** sépare LoRA 0,7 et 1,5 de la base (0,82 et 0,92 contre 0,72 à 0,76), mais aussi la copie à T = 1,0.
  AUC 0,49.
- **Le filigrane** ne se voit dans aucune statistique, ce qui est voulu, puisque c'est une copie.
- **dT** n'est trompé par aucun des 4 déguisements (|z| ≤ 0,6), ni par la température.

## 3. Le test absolu : deux identités, aucun plancher
Pour un token x tiré de p_T (le bon modèle à la température T) :
(i) E[−log p_T(x)] = H(p_T), et (ii) E[p_T(x)] = Σ p_T². On cherche T_log, la température qui annule (i), et T_col,
celle qui annule (ii). Pour le vrai modèle, quelle que soit la température de service, **T_log = T_col**. Une LoRA
déforme la forme de la distribution, pas seulement sa température, donc les deux T se séparent.
On pose dT = T_log − T_col, et z = dT divisé par son erreur type (bootstrap sur les réponses).
- **Règle** (fixée a priori à |z| > 2, chaque guichet jugé avec ses seules réponses et les poids publics) :
  partition {01, 02, 03, 05, 06, 08, 09} {04} {07}. `score_practice.py` : **MCC 0,524** (TP 10, FP 11, FN 0).
- **Robuste au contexte** : relu sans le prompt système par défaut, la partition et le MCC (0,524) ne changent pas
  (z : 04 −2,8 ; 07 −4,9 ; tous les autres |z| ≤ 1,9).
- **Fragile au nombre de réponses** (sous-échantillons de k réponses par guichet, 40 tirages) : MCC moyen 0,25 (k = 8),
  0,31 (k = 12), 0,37 (k = 16), 0,42 (k = 20). LoRA 0,7 n'est signalée que 15 à 40 % du temps. LoRA 1,5 l'est 100 %
  du temps dès k = 12. Les copies S sont signalées à tort 0 à 20 % du temps.
- **Validation « un guichet de côté »** : elle n'apprend rien ici, car la décision d'un guichet n'utilise ni les
  autres guichets ni les étiquettes. Le seul choix fait en voyant les données est **lequel** des 6 indicateurs essayés
  garder (T*, dT, ECE, taux d'erreurs sur les tokens à p > 0,9, longueur, NLL). C'est un risque de sur-apprentissage
  réel, avec seulement 2 variantes positives. D'où la prédiction pré-enregistrée de l'expérience 1.
- **Plafond atteignable** avec ce type de signal : séparer aussi fp8 et LoRA 0,3 demanderait un signal qu'on n'a pas.
  0,524 est donc le meilleur score réaliste sur la structure du corrigé, à moins qu'une piste nouvelle ne marche.

## 4. Familles des 33 ep, sans requête de plus
NLL/token de la variante et T* (identité log), pour les 5 lectures de chaque ep, sous 14 modèles candidats :
Qwen2.5 0,5/1,5/3/7B, SmolLM2 135M/360M/1,7B, SmolLM3-3B, OLMo-2 1B/7B, Phi-3.5-mini, Mistral-7B et Llama-3.2-1B
(miroir `unsloth`).
| famille (jumeau de NLL minimale) | ep | NLL sous ce jumeau | T* | 2e meilleur jumeau |
|---|---|---|---|---|
| Qwen2.5-3B | 01 02 04 12 14 17 18 23 24 25 33 | 0,61 à 1,24 | 0,63 à 1,04 | Phi-3.5 ou Qwen-1,5B, ≥ 1,02 |
| Qwen2.5-1.5B | 03 06 07 08 20 21 22 26 28 30 32 | 0,61 à 1,51 | 0,68 à 1,01 | ≥ 1,05 |
| Llama-3.2-1B | 05 09 10 11 13 15 16 19 27 29 31 | 0,50 à 0,97 | 0,63 à 1,02 | SmolLM2-1.7B, ≥ 0,94 |
- **Écart net** : le 2e meilleur candidat est toujours au moins 0,3 nat/token plus haut, sauf pour ep-30 et ep-33
  (NLL 1,51 et 1,24, T* ≈ 1,0 : probablement T = 1,0 ou une LoRA forte).
- **Contrôle absolu de « bon jumeau »** : sous le vrai modèle, T* reste proche de la température de service
  (≈ 0,7, comme sur les ref : 0,72 à 0,76). Sous un faux jumeau, T* monte à 0,95 ou plus. Sur les ref,
  SmolLM2-360M donne T* = 0,72 à 0,76, tous les autres 0,89 à 1,28.
- **Indice textuel gratuit** dans la sonde d'extraction P000 : ep-11, 13 et 27 recopient l'en-tête du gabarit
  Llama 3 (« Cutting Knowledge Date: December 2023 / Today Date … ») ; ep-22 et 32 recopient le prompt par défaut
  de Qwen (« You are Qwen, created by Alibaba Cloud… ») ; ref-06 et ref-09 celui de SmolLM2 (« You are a helpful AI
  assistant named SmolLM… »). C'est cohérent avec la NLL dans 5 cas sur 5. Le texte de la réponse « identité »
  (P001), en revanche, ment souvent : Claude, BERT, Meta AI, OpenAI.
- **Premiers signaux dT dans chaque famille** (5 réponses, erreurs types peu fiables, **à confirmer** avec les blocs
  1 à 3) : ep-14 (z = −23, T* 0,93) et ep-07 (z = −8, T* 0,87) ont la signature de ref-07 (LoRA forte). Viennent
  ensuite ep-22 (−3,8), ep-28 (−2,9), ep-01 (+2,7), ep-25 (+2,5) et ep-16 (+4,4). ep-16 a un prompt « avocat
  saoudien » extrait, et une relecture sans ce contexte peut fausser dT.

## 5. Pistes hors des réponses évidentes
- **dT** (§ 3) : un test de calibration qui ne dépend pas de la température, rendu possible parce qu'on a les poids
  exacts de la base. C'est ce qui manquait à la v4, et à toutes les statistiques de distance de l'issue #4
  (répartitions de réponses, AgentProv), que la température et le prompt système trompent autant qu'une LoRA.
- **Le gabarit de chat comme empreinte** : l'écho du prompt par défaut dit à la fois la famille et « pas de prompt
  système personnalisé ». À l'inverse, un prompt personnalisé extrait (ep-02, 16, 20, 29 ; celui d'ep-29 est le prompt
  par défaut de Llama-2) devient le contexte de relecture de ce guichet.
- **Rattacher une variante déguisée à sa sœur** : deux guichets signalés de la même famille, avec T_log et dT
  compatibles à 2 erreurs types près, sont probablement la même LoRA sous deux déguisements. Non mesuré.

## 6. Expériences suivantes, classées par gain attendu sur le score final
1. **Ajouter `meta-llama/Llama-3.2-1B-Instruct` (miroir `unsloth`, déjà en cache sur h200c) à `CANDS`, plus le
   contrôle T* ≤ 0,9 sous le jumeau retenu.** Gain : la famille Llama, soit un tiers des paires « même modèle »,
   passe d'une relecture par un faux modèle à une relecture exacte. Coût : 0 requête, environ 5 min de GPU. Le 3B est
   à télécharger avec `model.safetensors.index.json` (échec ici faute de ce fichier) ; T* sous le 1B (0,63 à 0,85)
   suggère que le 1B suffit.
2. **Remplacer la décision « plancher local » par la règle |z(dT)| > 2 sous le jumeau de la famille.** Sur les ref :
   0,273 → 0,524. Prédiction **pré-enregistrée** pour les blocs ref 1 à 3 (environ 96 réponses de lecture par
   guichet) : ref-04 z < −3 et ref-07 z < −6 ; ref-01, 03, 05, 06 et 08 |z| < 2 ; fp8 et LoRA 0,3 non garantis.
   Si ref-04 reste au-dessus de −2, la règle ne vaut pas plus que « isoler les LoRA fortes ».
3. **Relire chaque guichet sous le contexte de son prompt extrait** quand la sonde P000 rend un prompt personnalisé
   (ep-02, 16, 20, 29), puis recalculer dT. À valider d'abord sur ref-03 et ref-05 : leurs extractions semblent
   inventées, et on vérifiera que dT ne bouge pas.
4. **Dans chaque famille, fusionner les guichets signalés compatibles en (T_log, dT)** (test à 2 erreurs types), et
   laisser les non signalés dans le groupe de base. Simuler d'abord sur les ref en « déguisant » le signal : sans
   effet sur les ref, faute de variante en double.
5. **Ne rien dépenser pour fp8 et LoRA 0,3.** En projection (z proportionnel à √n, non mesuré), LoRA 0,3 passerait
   d'environ 1,4 à environ 2,9 avec 96 lectures, mais avec le signe opposé aux autres LoRA, donc sans règle a priori.
   Les fusionner avec la base, comme le font déjà AgentProv, MET et RUT pour la quantification 8 bits.

## 7. Reproduire
```bash
# h200c, venv privé (uv) : torch 2.15, transformers 5.24
python nll2.py ref0.jsonl HuggingFaceTB/SmolLM2-360M-Instruct ref2_360.json          # identités (i) et (ii), 13 T
python nll2.py ref0.jsonl HuggingFaceTB/SmolLM2-360M-Instruct ref2_360_empty.json empty
python nll.py  ep0.jsonl  <candidat> ep_<candidat>.json                               # 14 candidats, table § 4
# local
python calib.py ref2_360.json ; python rule.py ref2_360.json                          # § 3 (MCC 0,524, sous-échantillons)
python textstats.py <journal ref> ; python fam.py ref ; python fam.py ep              # § 2 et § 4
```
[agent critique · T2]
