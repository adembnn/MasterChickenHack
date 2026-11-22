# T2 · Stabilité du classement par bloc

Comparaison CPU sur les prédictions déjà produites par Titouan dans `/tmp/eac-t2-real/cycle2_pipeline/courbes.json` sur H200-2. Les neuf endpoints de référence incluent cinq copies et quatre variantes connues. Deux classifieurs existants, `r3_k` et `r6_noT_k25`, sont évalués sur les mêmes regroupements de blocs de 25 réponses. Aucun appel ou envoi sur le serveur de compétition.

Pour chaque regroupement, comparer le seuil de 0,5 sur les réponses mises en commun, les quatre meilleurs scores mis en commun, puis les quatre meilleurs rangs médians calculés bloc par bloc. Départager les égalités par le rang moyen. Le MCC porte sur les 36 paires de référence.

| Réponses | Modèle | Groupes | Seuil 0,5 | Top 4 scores mis en commun | Top 4 rangs médians |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | r3 | 6 | 0,595 | 0,538 | 0,908 |
| 50 | r6 | 6 | 0,778 | 0,908 | 0,908 |
| 75 | r3 | 4 | 0,604 | 1,000 | 1,000 |
| 75 | r6 | 4 | 0,734 | 0,862 | 1,000 |

À 50 réponses, le rang médian top 4 retrouve la LoRA 0,3 dans 5 regroupements sur 6 pour les deux modèles. Le MCC chute à **0,734** en top 3 et **0,721** en top 5. Les regroupements partagent des blocs. Les références avaient déjà servi au développement des lecteurs, et le nombre de variantes y est connu. Les MCC mesurent la stabilité sur ces références.

Pour les données de compétition, le rang par bloc peut corroborer les variantes faibles sur les sorties locales du cycle 2. Conserver les deux rangs individuels, leur étendue et l’accord avec les lecteurs de vraisemblance. La décision dépend de cet accord et de preuves indépendantes sur le nombre de variantes. Sur les guichets ambigus, la méthode est un moyen de garder trace des désaccords sans consommation de requêtes. [Branches et discussion T2](https://github.com/adembnn/EAC/issues/4)
