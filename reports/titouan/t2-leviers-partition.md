# T2 · Trois leviers de partition (inventaire, vote de detecteurs, MBR) — testes avant d'etre appliques

**Resultat :** sur un banc CPU a verite connue (3 familles, 24 guichets), les trois leviers **ne passent pas**
les criteres fixes (MCC_modele >= 0,75, aucune copie isolee a tort, prior faux coutant <= 0,05). Appliques
quand meme aux 33 `ep-*` reels : un seul endpoint change de groupe entre les deux variantes (`ep-28`), et la
preuve est faible — a ne pas utiliser pour une soumission sans corroboration par une autre methode.

Code : `tracks/t2-model-equality/twin/build_world_cpu.py` (banc), `twin/partition_levers.py` (les 3 leviers +
score). Aucune requete envoyee vers l'API du hackathon par cette sous-session (CPU seul, comme demande).

## Le banc : 3 familles, verite connue, CPU

`HuggingFaceTB/SmolLM2-360M-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`, `Felladrin/Llama-160M-Chat-v1` (substitue
a TinyLlama-1.1B : son telechargement de 2 Go a echoue 3 fois sur la connexion disponible ; ce modele 160M,
meme architecture Llama, a le meme role pour le banc). Par famille, 8 guichets, 10 prompts partages :
- 5 **memes poids** (base), deguisements differents : brut, system prompt, persona, filigrane KGW (delta=4),
  echantillonnage T=0,9 — classe de 5 (l'inventaire reel attend 4 a 6 membres).
- 1 **fp8** (fausse quantification e4m3 par ligne, hors lm_head) — singleton.
- 2 **LoRA** (meme adaptateur entraine sans peft — module bas rang injecte sur q_proj/v_proj, entraine sur
  les propres completions du modele — forces 0,4 et 1,3) — singletons.

24 guichets, 276 paires, verite connue (famille, poids).

## Deux detecteurs independants (vote)

- **A — profil de NLL** : chaque guichet est relu, texte seul, par UN SEUL lecteur local (SmolLM2-360M,
  jamais partage) ; score(A,B) = correlation de Pearson entre les deux vecteurs de NLL par prompt. Des poids
  identiques devraient rater/reussir sur les memes prompts, quel que soit le deguisement.
- **B — recouvrement lexical** : Jaccard de 3-grammes de mots entre les reponses, moyenne sur les prompts.
  Independant de A (pas de relecture), complementaire quand une famille entiere est loin du lecteur.

Score d'ensemble = moyenne des deux scores centres-reduits (z-score).

## Levier 2 — vote + clustering de correlation + triangles incoherents

Clustering de correlation glouton (fusion des deux grappes au score moyen le plus eleve, tant qu'il est
positif) sur le score d'ensemble. Diagnostic des triangles : pour (A,B,C) avec 1 ou 2 aretes "meme" sur 3
(incoherent), le coupable probable est celui dont les deux aretes ont la plus faible magnitude.

**Sur le banc** (verite connue) : 1183 triangles incoherents sur 2024 (58 %, tres eleve). Clustering final :
**MCC = 0,083** (tp=15 fp=91 fn=15 tn=155), 4 guichets isoles a tort (`qwen-wm`, `smol-persona`, `tlla-t09`,
`tlla-wm` — des **deguisements de decodage**, pas des poids differents : le vote les sort a tort de leur
classe). Tres loin du seuil de 0,75.

## Levier 1 — decodage sous contrainte d'inventaire, prior volontairement faux

Bonus de prior de taille de grappe (pic a 1 et a la taille de la "grande classe" attendue) ajoute au score de
fusion, avec lambda=0,5 et la taille cible a 4, 5 (vraie) ou 6 (prior volontairement faux de +-1) :

| lambda | prior taille | MCC |
|---|---|---|
| 0 | — | 0,083 |
| 0,5 | 4 (faux -1) | 0,278 |
| 0,5 | 5 (correct) | 0,278 |
| 0,5 | 6 (faux +1) | 0,278 |

Le prior de taille **ameliore** le MCC (0,083 -> 0,278), mais **le prior correct et les deux priors
volontairement faux donnent exactement la MEME partition** : a cette magnitude (lambda=0,5, variance=3), le
bonus de forme ne fait pas basculer une seule fusion differemment entre les trois cibles. Le critere
« le prior faux ne doit pas coûter plus de 0,05 » est donc rempli (cout = 0), mais pour une raison qui ne
valide pas la methode : le signal brut domine totalement, le prior n'a pas assez de poids pour discriminer
entre 4/5/6, donc il ne peut pas non plus se tromper. Un lambda plus fort ferait sans doute bouger la
partition — mais rien dans ce test ne montre que le prior AIDE a corriger les vraies erreurs (tp reste a 15
dans les trois cas).

## Levier 3 — MBR, 200 rééchantillonnages bootstrap des prompts

MCC = 0,083 (identique au clustering sans prior — le tirage median retombe sur la meme partition). Stabilite
des paires (>= 80 % d'accord sur les 200 tirages) : **93 %** — tres stable, mais stable sur une partition
largement fausse. La stabilite mesure la reproductibilite de la methode, pas sa justesse ; les deux ne
doivent pas etre confondus.

## Pourquoi ca echoue : le probleme de puissance, encore

Repartition intra/inter-famille du clustering (levier 2) : intra-famille tp=15 fp=42 fn=15 tn=12 (MCC~0, sur
84 paires) ; inter-famille 49 paires FAUSSEMENT fusionnees sur 196 (des guichets de familles DIFFERENTES
rassembles). Avec seulement 10 prompts partages, le signal de 2 detecteurs bon marche (correlation de NLL
sous UN lecteur externe + recouvrement lexical) ne separe pas de maniere fiable les variantes de poids d'une
meme famille, et laisse meme fusionner des familles entierement differentes — exactement le probleme de
puissance statistique deja documente sur les vraies donnees `ref-*` (rapport `t2-nll-deguisements.md`,
MCC=0,000, N=21). Un detecteur plus fort (le t de Student apparie contre un atlas de nuisances, deja valide
en simulation a N=60, cf. ce meme rapport) ferait probablement mieux dans le vote, mais n'a pas ete integre
ici par manque de temps (echeance CPU de la tache).

## Applique quand meme aux 33 `ep-*` reels (bloc0, 20 prompts `reading` partages)

Un seul desaccord entre le clustering de correlation (deterministe) et la partition MBR (bootstrap-medoide,
150 tirages) : **`ep-28`**, isole seul par le clustering pur, rattache a la grande classe
(`ep-01,02,03,04,06,07,08,12,14,17,18,20,21,26,30,32,33`) par le consensus bootstrap. `ep-28` est aussi le
3e "coupable" le plus frequent du diagnostic de triangles (129/5456 triangles incoherents sur ses aretes,
derriere `ep-05` et `ep-21`).

**Mon evaluation : ce changement n'est PAS bien etaye.** Les deux signes (desaccord clustering/MBR, frequence
dans les triangles incoherents) pointent dans la meme direction, mais le meme pipeline echoue nettement le
test de validation ci-dessus (MCC=0,083 sur verite connue, bien au-dessous du seuil). Je ne recommande pas de
changer `ep-28` de groupe sur cette seule base — seulement de le signaler comme candidat a revoir avec une
methode plus forte (le t aparie contre atlas, ou le detecteur d'histogrammes deja a AUC 0,92 en simulation)
avant toute decision de soumission.

## Reproduire

```
python tracks/t2-model-equality/twin/build_world_cpu.py --out twin/world_cpu.jsonl --truth twin/world_cpu_truth.csv
python tracks/t2-model-equality/twin/partition_levers.py bench --jsonl twin/world_cpu.jsonl --truth twin/world_cpu_truth.csv
python tracks/t2-model-equality/twin/partition_levers.py apply_real --jsonl <collecte_v4_ep_bloc0_*.jsonl>
```
