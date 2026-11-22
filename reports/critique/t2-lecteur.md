# T2 : lecteur calé sur le réel (session t2-lecteur d'Adem, 22/11)

Décision débloquée : quels guichets hors public isoler dans le fichier final, et lesquels laisser dans la base.
Aucune requête à l'API, aucune soumission. Calculs sur h200d (`~/eac/t2/lecteur2/`), h200h étant devenu injoignable à 5 h 25 (ses deux rebonds h200f et h200g ne répondent plus).

## Mots utilisés
- **Jumeau** : les vrais poids du modèle de base (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B), chargés chez nous.
- **Vraisemblance** : probabilité que le jumeau donne au texte exact rendu par un guichet. Si le guichet sert le jumeau, ses textes sont « attendus » ; sinon, il reste un écart.
- **T\*** : la température qui rend les textes du guichet les plus vraisemblables sous le jumeau (température lue).
- **Faux guichet** : un guichet simulé par nous, qui est une copie exacte du jumeau. Il sert d'étalon : un vrai guichet ne compte comme décalé que s'il sort de la plage des faux.
- **z** : écart d'un guichet à la médiane des faux, en écarts types robustes. z = 3 veut dire « plus loin que presque tous les faux ».
- **Direction LoRA** : petite modification apprise sur les textes d'un guichet. Si elle rend aussi plus vraisemblables les textes d'un AUTRE guichet, les deux portent la même modification.
- **Loi nulle élargie** : faux guichets déguisés (température, top-k, top-p, min-p, pénalité de répétition, prompt système, persona, filigrane à liste verte), pour voir jusqu'où un déguisement seul fait monter les z.
- **AUC** : probabilité qu'une variante confirmée tirée au hasard ait un z plus haut qu'une copie confirmée tirée au hasard (1 = séparation parfaite, 0,5 = hasard).

## 1. Statistiques par guichet (lecteur existant, reproduit)
(à compléter)

## 2. Quantifications simulées
(à compléter)

## 3. Directions LoRA : la matrice des liens
(à compléter)

## 4. Loi nulle élargie : guichets déguisés
(à compléter)

## 5. Étalonnage sur les guichets confirmés
(à compléter)

## 6. P(variante) et fichier candidat
(à compléter)

## Scripts
`reports/critique/t2-lecteur-*.py` : `gen_disg2` (copies déguisées), `quantlib` (quantifications simulées), `fitdisg` (lecteur ajusté), `analyse` (z, AUC), `mkdgn`, `deguises` (classement des guichets déguisés). Lecteur existant : `t2-loi-nulle-gl_reader.py` et `gl_analyse.py` (grille T ramenée à 0,6-1,2 par pas de 0,1, 10 faux par température).
