# T2 : la méthode v4 marche-t-elle quand le modèle n'est pas OLMo ?

**Décision débloquée** : peut-on brancher la v4 (sondes à choix forcé + NLL croisée) sur les 33 `ep-*` sans savoir
quel modèle tourne derrière ? **Réponse courte : pas sa partie boîte noire.** Les sondes à choix forcé échouent, et
de la même façon, sur OLMo, Qwen2.5 et SmolLM2. La NLL croisée transfère parfaitement d'un modèle à l'autre avec une
règle unique (MCC 1,00 partout), mais elle demande les poids de l'endpoint. Sur la vraie API, elle n'est donc
utilisable que si on a identifié le modèle et qu'on en a une copie locale.

0 requête vers l'API du hackathon. Calcul sur h200b et h200c, 21/11, 18:30-19:25.

## 1. Ce qui a été testé

**La v4 n'était pas publiée** quand j'ai commencé (pas de branche `t2/method-v4`, pas de PR). J'ai donc construit une
reconstruction minimale : `tracks/t2-model-equality/twin/method_v4_multimodel.py`.
- Elle utilise les deux briques décrites sur l'issue #4 : sondes à choix forcé façon AgentProv, et NLL croisée
  contre fp8 et LoRA.
- La 1re v4 de la session sœur, trouvée sur h200b, lisait les endpoints sans leurs déguisements. Ça revenait à un
  oracle. Je le lui ai signalé, elle l'a jetée, et sa nouvelle version n'était pas encore poussée à la fin de ce
  travail. **Ce rapport ne mesure donc pas la v4 officielle**, mais les deux briques qu'elle combine.

**Monde synthétique.** Pour chaque modèle, 9 faux `ref` dans la structure exacte de `practice_answers.csv` :
| ref | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|
| nature | brut | fp8 e4m3 | prompt système | LoRA 0,7 | persona | filigrane KGW δ=6 | LoRA 1,5 | T=1,0 | LoRA 0,3 |
| modèle | S | T1 | S | T3 | S | S | T4 | S | T2 |

Il y a deux mondes :
- **propre** : les 5 copies S partagent le même objet de poids. Leur écart NLL vaut 0 par construction.
- **bruit de service** (`--serving_noise`) : les copies S sont servies par des instances distinctes (bf16, fp16,
  fp32). Ça donne un plancher de bruit numérique réaliste.

Modèles testés : OLMo-2-1B-Instruct (référence), Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct.

**Procédure de décision unique (PR #22), identique partout, sans dire quel modèle c'est :**
- *Sondes* : 100 questions sans bonne réponse (un nombre, une lettre, une couleur…). On ne compare que le mot
  extrait. Chaque endpoint rejoue ses 100 sondes (autre taille de lot, autres graines s'il échantillonne), ce qui
  donne son auto-désaccord `p`. Une paire est « même modèle » ssi `désaccord ≤ max(p_a, p_b) + 2·SE`.
- *NLL croisée* : sur 40 prompts ouverts, `gap = NLL_B(texte_A) − NLL_A(texte_A)`, dans les deux sens. Une paire est
  « même modèle » ssi `|t| ≤ 2`.
- *v4* : « même modèle » ssi les deux disent « même ».
- Partition, puis MCC calculé par `reports/sources/t2/score_practice.py`. C'est le seul code qui lit le corrigé.
- Aucun seuil appris. Une 1re version cherchait le meilleur seuil sur OLMo puis le figeait. Titouan l'a jugée
  biaisée et elle a été abandonnée.

## 2. Résultat : MCC par famille, même procédure partout

Chaque case donne union-find / lien complet. Le lien complet est la correction proposée au §4.

| procédure | OLMo | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|---|
| sondes seules (**compatible boîte noire**) | 0,00 / 0,27 | 0,00 / 0,27 | 0,00 / 0,27 |
| v4 = sondes ET NLL | 0,49 / 0,27 | 0,49 / 0,27 | 0,22 / 0,27 |
| NLL croisée seule (\|t\| ≤ 2, **demande les poids**) | 0,52 / **1,00** | 1,00 / **1,00** | 0,73 / **1,00** |

Les mondes propre et avec bruit donnent exactement les mêmes MCC sur les 3 modèles. Avec bruit, les copies S ont
|t| ≤ 1,7, sous le seuil de 2 : le bruit est réel mais ne fait basculer aucune décision.

Groupes obtenus avec la NLL seule + lien complet, sur les 3 modèles : `{01,03,05,06,08} {02} {04} {07} {09}`, soit la
vérité exacte. Avec les sondes seules en union-find : `{01,02,03,04,07,08,09} {05} {06}` sur les 3 modèles.

## 3. Pourquoi les sondes échouent (même cause sur les 3 familles)

Accord des réponses avec `ref-01` (100 sondes) :

| | fp8 | sys. prompt | LoRA 0,7 | persona | filigrane | LoRA 1,5 | T=1,0 | LoRA 0,3 |
|---|---|---|---|---|---|---|---|---|
| OLMo | 0,92 | 0,72 | 0,93 | **0,38** | **0,26** | 0,93 | 0,57 | 0,81 |
| Qwen | 0,89 | 0,69 | 0,85 | **0,59** | **0,33** | 0,78 | 0,82 | 0,78 |
| SmolLM2 | 0,94 | 0,51 | 0,86 | **0,40** | **0,36** | 0,85 | 0,68 | 0,90 |

1. **Le signal est inversé.** Les copies déguisées du même modèle (persona, filigrane) sont plus loin de `ref-01`
   que les modèles différents (fp8, LoRA). Une question sans bonne réponse a par construction un écart minuscule entre
   ses deux meilleurs choix.
   - Un filigrane δ=6 sur une réponse d'un seul token suffit à faire basculer le choix.
   - Une persona « enjouée » change la préférence (« Spring! » au lieu de « Winter »).
   - Ce sont donc les questions les plus sensibles aux déguisements, l'inverse de ce qu'on voulait.
2. **L'endpoint à température 1,0 se contredit** à 57 % (OLMo), 35 % (Qwen) et 48 % (SmolLM2), même avec la réponse
   majoritaire sur 5 tirages. Sa tolérance `p + 2·SE` devient énorme, il « ressemble » à tout le monde, et l'union-find
   enchaîne 7 endpoints dans un seul groupe.
3. **fp8 et LoRA restent proches de la base** (0,78-0,94) : la sonde est quasi aveugle à ces variantes, comme
   AgentProv l'annonçait pour le 8 bits.

C'est structurel, pas propre à OLMo : **la méthode « généralise » dans l'échec**. La combinaison v4 (sondes ET NLL)
hérite du défaut. Les sondes empêchent la NLL de regrouper persona et filigrane, d'où 0,22-0,49.

## 4. NLL croisée : ce qui transfère, et sa limite

- **La règle 2·SE transfère sans recalibrer.** t de Student contre `ref-01` :

  | | fp8 | LoRA 0,3 | LoRA 0,7 | LoRA 1,5 | copies S (avec bruit) |
  |---|---|---|---|---|---|
  | OLMo | 3,47 | 5,56 | 8,18 | 8,07 | 0,56-0,90 |
  | Qwen | 2,95 | 3,89 | 8,70 | 10,85 | −1,24-0,44 |
  | SmolLM2 | **1,06** | 4,19 | 9,00 | 7,12 | −0,40-1,69 |

- **Correction minimale, la même partout : lien complet au lieu de l'union-find.** Deux groupes ne fusionnent que si
  toutes leurs paires croisées disent « même ». En union-find, un fp8 ou une LoRA 0,3 correctement séparé de `ref-01`
  revient par une copie déguisée dont la paire n'est pas significative. Avec le lien complet, la NLL seule passe de
  0,52-0,73 à 1,00 sur les 3 familles. Ça ne sauve pas les sondes (0,27).
- **Fragilité sur SmolLM2.** fp8 contre `ref-01` n'est pas significatif (t = 1,06). Il n'est isolé que parce qu'une
  autre paire fp8/copie S l'est. Avec moins de prompts ou un autre tirage, fp8 serait fusionné sur ce modèle :
  1,00 est optimiste pour fp8. Ça rejoint le barème d'Adem : fusionner fp8 par défaut ne coûte que ~0,27 MCC.
- **Limite principale, qui conditionne tout.** Calculer `NLL_B(texte_A)` demande les poids de B. Sur la vraie API
  (texte seul, sans logprobs, cf. `t2-method-v3.md` étape 1), c'est impossible tel quel. La seule version réalisable :
  identifier la famille (étape F), télécharger ce modèle, et relire les textes des deux endpoints avec cette copie
  locale. C'est ce que fait la nouvelle v4 de la session sœur. Ce test montre que la statistique est bonne *si* on
  a la bonne copie locale. Il ne mesure pas ce qui se passe si la copie locale est le mauvais modèle ou une autre
  taille.

## 5. Recommandations

1. **Ne pas utiliser les sondes à choix forcé pour regrouper les copies déguisées.** Elles font l'inverse
   (MCC 0,00 en union-find sur les 3 familles). Elles pourraient servir à séparer des familles très différentes :
   non testé ici.
2. **Toute décision « même modèle » par la NLL passe par le lien complet**, pas par l'union-find : de 0 (Qwen, déjà
   à 1,00) à +0,48 MCC (OLMo), même règle pour tous les modèles.
3. **La branche NLL dépend de l'identification du modèle.** Prochaine mesure qui tranche : relire avec la *mauvaise*
   copie locale (textes Qwen relus par SmolLM2, et inversement) et regarder si la règle |t| ≤ 2 tient encore.
4. Pour la soumission plancher sans NLL : la partition « tout fusionné sauf les endpoints visiblement dégradés »
   (barème d'Adem, issue #4) reste la référence boîte noire. Ces sondes ne font pas mieux.

## 6. Limites

- Une seule graine, 100 sondes, 40 prompts NLL. Trois petits modèles (1-1,7 B). Phi-3.5, Mistral et Ministral non
  testés : faute de temps, après deux OOM sur des GPU partagées.
- Déguisements de ma reconstruction, pas forcément ceux des vrais endpoints : persona et prompt système maison,
  filigrane KGW δ=6, fp8 e4m3 simulé, LoRA r=8 entraînée à préfixer « Sure thing! ».
  - Ce préfixe est retiré avant l'extraction de la réponse aux sondes, sinon la LoRA serait détectée par
    l'artefact.
  - Cet entraînement donne une perte finale identique pour les 3 forces : l'échelle est appliquée après.
- Bruit de service approximé par des dtypes différents, pas par du vrai batching de serveur.

## Reproduire

```bash
# h200c, venv privé ~/eac-t2-multi/.venv (torch 2.15.0+cu132, transformers 5.24.0, peft 0.22.0)
python method_v4_multimodel.py --model qwen --calibrate --serving_noise --out adapt/noisy_qwen.json
python analyse_v4_adaptatif.py --dir results_v4_multimodele
```
Mesures brutes : `tracks/t2-model-equality/twin/results_v4_multimodele/`, avec les 36 paires, les écarts NLL par
prompt, les réponses aux sondes et leur rejeu.
