# T2 : mining de prompts « à quasi-égalité » pour séparer un modèle de ses variantes légères

**Décision que ce rapport débloque** : quels prompts envoyer aux 33 endpoints (100 réponses chacun, budget
irréversible) pour que deux endpoints qui portent le même modèle produisent le même texte, et que deux endpoints qui
diffèrent par une quantification ou une LoRA faible produisent un texte différent, même quand l'un des deux a un
system prompt ou une température ajoutés.

Tout est local (jumeau OLMo-2-0425-1B-Instruct sur un A100 de JURECA, jobs `t2mining` 15901690 et `t2minval` 15901773).
**Aucune requête sur l'API du hackathon.** Code : `tracks/t2-model-equality/twin/prompt_mining.py` et
`validate_prompts.py` ; sortie : `tracks/t2-model-equality/twin/discriminating_prompts.json` (100 prompts classés).

## 1. Idée

Une petite perturbation des poids (int8, fp8, LoRA à force 0,3) déplace peu les logits. En décodage glouton, le
texte ne change que si le token top-1 et le token top-2 étaient presque à égalité. Donc :
- un prompt où le modèle est **indécis au premier token** (marge de probabilité top1 - top2 faible) est un capteur
  sensible à toute perturbation des poids ;
- mais il est aussi sensible au **bruit numérique pur** (taille de lot, dtype), qui n'est pas un changement d'identité ;
- il faut donc **sélectionner** les prompts qui divergent sous int8/LoRA **et pas** sous le bruit, puis vérifier que
  ce signal **survit aux déguisements** (system prompt, température 0,7).

## 2. Méthode (4 étapes, 283 s sur un A100)

1. **Banque** : 1 719 prompts à réponse courte et forcée, générés par templates : primalité (« Yes or No »),
   comparaisons `a×b > c` serrées, vrai/faux, arithmétique exacte (« Reply with only the number »), divisions arrondies,
   modulo, complétions à un mot (contraire, « My favorite color is ___ »), parité, divisibilité.
2. **Pré-filtre** : une passe avant sous la base, marge top1-top2 au premier token généré ; on garde les **300 plus petites**
   (marge de 0,0000 à 0,184). Deux contrôles : les 300 **plus grandes** marges et **300 au hasard**.
3. **Validation gloutonne** (20 tokens) : base vs `noise` (même modèle bf16, lot 8 au lieu de 32), `dtype` (mêmes poids en
   fp16), `int8` (arrondi par canal), `lora03` (LoRA r=8 q/v entraînée à imiter la base avec un espace devant, force 0,3).
   Score glouton = int8 + lora03 - noise - dtype. On garde les 150 meilleurs.
4. **Déguisements** sur ces 150 : (a) system prompt générique ajouté, glouton ; (b) température 0,7, 5 tirages par prompt,
   **même graine pour la base et la variante** (ainsi une divergence vient d'une vraie différence de distribution, pas du
   hasard). Score final = glouton + 0,5·(score system) + 0,5·(score température). Les 100 meilleurs sont rangés.

## 3. Résultats

### 3.1 Le filtre par marge amplifie TOUT (signal et bruit) : taux de divergence gloutonne, 300 prompts par ligne

| Sous-ensemble | noise (lot) | dtype (fp16) | int8 | LoRA 0,3 |
|---|---|---|---|---|
| 300 plus petites marges | **10,0 %** | 8,3 % | **14,7 %** | **69,3 %** |
| 300 au hasard (contrôle) | 2,3 % | 1,7 % | 2,7 % | 45,7 % |
| 300 plus grandes marges (contrôle) | 0,0 % | 0,0 % | 0,0 % | 29,0 % |

Lecture : à longueur égale (20 tokens), les prompts indécis divergent 5,5x plus souvent sous int8 et 1,5x plus sous
LoRA 0,3 que des prompts au hasard, mais aussi 4x plus sous le simple bruit numérique. La marge seule ne suffit pas :
la sélectivité vient de l'étape 3 qui soustrait le bruit. À l'inverse, les prompts très confiants ne bougent jamais
(int8 0 %) : ce sont de bons **témoins négatifs** (ils doivent rester identiques entre endpoints de même modèle).

### 3.2 Les 100 finalistes (chiffres **biaisés par la sélection**, voir 3.3 pour les chiffres honnêtes)

| | noise | dtype | int8 | LoRA 0,3 |
|---|---|---|---|---|
| Glouton, 100 finalistes | 2 % | 4 % | 26 % | 95 % |
| Glouton, top 50 | 2 % | 2 % | **42 %** | 92 % |
| System prompt ajouté, 100 | 3 % | 2 % | 10 % | 43 % |
| System prompt ajouté, top 50 | 2 % | 2 % | 16 % | **76 %** |
| Température 0,7, 100 (taux sur 5 tirages) | - | 4 % | 12 % | 69 % |
| Température 0,7, top 50 | - | 3,6 % | 15 % | 64 % |

Composition des finalistes : arithmétique exacte 46, primalité 19, division arrondie 19, modulo 11, complétion 5.
Aucun prompt de comparaison, parité, divisibilité ou vrai/faux n'a survécu : la base y est trop confiante au premier token.
47 finalistes gardent un signal sous les deux déguisements à la fois ; 51 perdent tout signal sous system prompt
(en particulier les 19 prompts de primalité : LoRA 100 % en glouton, 0 % avec system prompt, 87 % à température 0,7).

### 3.3 Validation hors-échantillon (nouvelle LoRA, nouvelles graines, contrôle aléatoire)

Job `t2minval` 15901773 (379 s) : une **nouvelle LoRA** (graine 4242, autres prompts d'entraînement, force 0,3),
de **nouvelles graines** de température, et 100 prompts **au hasard** hors finalistes comme contrôle.

| Ensemble | Condition | noise | dtype | int8 | LoRA 0,3 (nouvelle) |
|---|---|---|---|---|---|
| 100 finalistes | glouton | 4 % | 4 % | 26 % | **81 %** |
| | system prompt | 3 % | 2 % | 11 % | 42 % |
| | température 0,7 | - | 4,8 % | 11,2 % | 57,2 % |
| top 50 | glouton | 4 % | 2 % | **42 %** | 80 % |
| | system prompt | 4 % | 2 % | 14 % | **66 %** |
| | température 0,7 | - | 5,2 % | 8,4 % | 52 % |
| 100 au hasard (contrôle) | glouton | 0 % | 2 % | 3 % | 40 % |
| | system prompt | 2 % | 3 % | 6 % | 16 % |
| | température 0,7 | - | 1,8 % | 3,4 % | 40,6 % |

Lecture :
- **LoRA : le gain tient sur une LoRA jamais vue** : 81 % contre 40 % au hasard en glouton (x2), 42-66 % contre 16 %
  avec system prompt (x2,6 à x4), 52-57 % contre 41 % à température 0,7 (gain faible : à T=0,7 même des prompts
  quelconques divergent souvent sur 20 tokens).
- **int8 : 26 % (top 50 : 42 %) contre 3 % au hasard**, soit x9 à x14. Attention, l'int8 est déterministe : ce n'est pas
  une nouvelle variante, seulement la confirmation que le chiffre est reproductible sur un autre nœud. Sous déguisement
  il retombe à 8-14 %, toujours 2 à 4x le bruit, mais le rapport signal/bruit devient mince (11 % vs 3-5 %).
- **Bruit** : 4 % de divergence entre deux lots sur les finalistes (2 % au premier passage) : le non-déterminisme
  numérique varie d'un nœud à l'autre ; le compter comme un plancher de ~2-5 %, jamais 0.

## 4. Ce que ça veut dire pour les 33 endpoints

1. **Le texte exact d'un 1B à 20 tokens ne voit presque pas l'int8** : au mieux ~40 % de divergence sur les meilleurs
   prompts, 2-3 % sur des prompts au hasard. Confirme Cai et al. (texte seul vs fp8/int8 ≈ hasard) et le rapport
   `t2-fingerprinting.md` : pour int8/fp8 il faudra la vraisemblance token par token, pas la comparaison de texte.
2. **La LoRA faible (0,3) se voit bien** sur ces prompts : 70 à 95 % de divergence en glouton, et le signal survit à
   la température 0,7 (50-70 %) et, pour la moitié des prompts, au system prompt (40-75 %). Validé sur une 2e LoRA (81 %).
3. **Le system prompt est le déguisement le plus dangereux** : il divise le signal LoRA par 2 et tue les prompts binaires.
   Pour comparer deux endpoints dont l'un a un system prompt, préférer les prompts arithmétiques « Reply with only the
   number » du top 50, ou les témoins négatifs (grandes marges) pour tester l'égalité.
4. **Le bruit de fond n'est pas nul** : 2-10 % de divergence entre deux lancements du MÊME modèle selon le lot/dtype.
   Deux endpoints du même modèle sur des serveurs différents divergeront donc parfois sur ces prompts. Il faut un
   seuil, pas une égalité stricte : avec les 100 prompts, attendre ~2-5 divergences pour « même modèle »,
   ~25 (top 50 : ~40 %) pour int8, ~80 pour LoRA 0,3 en glouton ; avec system prompt, ~3 / ~11 / ~42.
5. Les témoins négatifs (300 plus grandes marges, int8 0 %) servent à **détecter l'échantillonnage** : si deux appels
   du même endpoint diffèrent sur ces prompts, l'endpoint échantillonne (température > 0) et la comparaison de texte
   doit basculer sur des taux plutôt que sur l'identité.

## 5. Limites et suites

- Jumeau 1B, LoRA maison (q/v, r=8) entraînée sur des prompts des mêmes templates (in-distribution, comme dans
  l'exp1) : les taux LoRA sont optimistes ; la validation 3.3 change la LoRA mais pas les templates.
- Mesuré uniquement sur OLMo-2 1B. À rejouer sur le 7B si le zoo le permet (le script prend ~5 min pour 1B).
- Les 100 prompts sont des mesures de « quasi-égalité » pour CE modèle : la marge dépend du modèle. Si les endpoints
  sont une autre famille, relancer `prompt_mining.py` avec le bon `MODEL` (le pré-filtre coûte une passe avant).
- Suites utiles (5 min chacune) : (a) ajouter fp8 e4m3/e5m2 et nf4 au scoring ; (b) remplacer le score binaire
  « diverge/ne diverge pas » par la position de première divergence ; (c) pondérer les prompts par leur marge
  pour construire un test statistique (somme des divergences vs loi binomiale du bruit).
