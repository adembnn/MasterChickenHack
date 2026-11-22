# T3 · session t3-simulateur : bilan (21 nov. 2026, mission close)

Décision que ce rapport sert : quoi réutiliser du simulateur, et quoi ne plus croire.

## Ce qui reste valable
- **`score.py`** (H200 `~/eac/t3/t3-simulateur/`, copie `E:\EAC-lab\t3\t3-simulateur\`) : la formule officielle appliquée à des logits (sorties brutes du modèle avant probabilités). `python score.py --selftest` passe ses 6 tests (orig → oubli 0 ; un gold seul → 1 ; porte à 70 % ; rétention). Relu ligne à ligne par deux sessions indépendantes. Usage : `python score.py --labels L.npz --orig O.npz --gold G1.npz G2.npz --sub S.npz`.
- **Le vrai `model.pt` est entraîné de zéro** : similarité cosinus 0,00 avec 10 jeux de poids pré-entraînés timm ; coefficients LayerScale passés de 1e-4 à 0,01–0,08.
- **Réglage d'entraînement sur le vrai modèle** (4 derniers blocs + norme + tête, LayerScale gelé) : à lr 1e-4 et 600 pas, 84 % des images visées changent de classe ; à **lr 3e-4 et 1 500 pas, 99,5 %**, avec 0,0 % de changement sur les images non visées et une dérive KL(orig‖sub) de 0,023 sur des mélanges d'images.
- **Banc de notation** `methods.py` : note une méthode avec les trois facteurs séparés (S_loss, S_pred, S_ret) en 2 à 4 minutes, à condition d'avoir des golds fidèles.
- **Cible à température 2** : la racine carrée renormalisée de la cible à température 1 reproduit la moyenne des golds simulés (KL 0,011) ; le mélange 0,15·w de S1 la dégrade (0,068).

## Ce qui est réfuté
- **Hypothèse « 130 images par classe, 9 classes oubliées en entier » (monde B).** Elle venait du plafond à 130 des effectifs et rejouait deux scores réels (0,004 pour 0,0039 ; 0,015 pour 0,0114). Deux envois réels la contredisent : `Z9_b0` (les 9 classes rendues fausses) = 0,0 alors que le monde B donnait 0,19 à 0,25 à cette opération, et `G10_PG_T70` = 0,0208, donc la porte des 70 % n'explique pas ce 0. Le gold connaît ces classes.
- **Candidat `cand_sonde_v3_L10.pt`** : même opération que `Z9_b0`, non soumis, à ne pas soumettre.
- **Tous les chiffres du monde B** (gold à 62 % de précision, loss 3,73 ± 5,35, « 92 % du KL sur les images où le gold change de classe », sonde à 0,25, borne haute 0,51) décrivent un monde qui n'est pas le réel.
- **Monde A** (400 images par classe, golds partis de poids pré-entraînés) : non fidèle non plus. Son gold est trop facile (précision 96 %, loss 0,14) et il donne 0,000 aux deux envois réels rejoués.

## Ce qui n'a pas été fait
- Aucun gold entraîné de zéro : un seul orig de zéro existe (précision de validation 0,67, confiance 0,978 contre 0,9997 pour le vrai). Le monde fidèle demandé au départ n'existe donc pas.
- Pas de rejeu des dix envois réels, pas de recette gagnante. Le modèle calé par t3-verificateur sur les envois réels guide désormais les soumissions.

## Erreur de méthode à retenir
J'ai tenu pour confirmée une hypothèse sur deux points de calage seulement, alors qu'un gold « peu précis partout » les reproduisait aussi (objection du relecteur, `reports/critique/t3-simulateur-tour1.md`). La sonde réelle a tranché en une soumission.

## État des machines
H200 : tous mes processus arrêtés (0 restant, vérifié à 20 h 38 UTC). `h200e` : jamais utilisée. Jülich : jobs `t3sim1` (15902091) et `t3sim2` (15902139) peut-être encore en file, à annuler par l'orchestrateur (accès MFA refusé pour moi).
