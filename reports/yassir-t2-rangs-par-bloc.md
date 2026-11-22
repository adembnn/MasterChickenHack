# T2 · Stabilité du classement par bloc

Comparaison CPU sur les prédictions déjà produites par Titouan dans `/tmp/eac-t2-real/cycle2_pipeline/courbes.json` sur H200-2. Les neuf endpoints de référence incluent cinq copies et quatre variantes connues. Deux classifieurs existants, `r3_k` et `r6_noT_k25`, sont évalués sur les mêmes regroupements de blocs de 25 réponses. Aucun appel ou envoi sur le serveur de compétition.

Pour chaque regroupement, comparer le seuil de 0,5 sur les réponses mises en commun, les quatre meilleurs scores mis en commun, puis les quatre meilleurs rangs médians calculés bloc par bloc. Départager les égalités par le rang moyen. Le MCC porte sur les 36 paires de référence.

| Réponses | Modèle | Groupes | Seuil 0,5 | Top 4 scores mis en commun | Top 4 rangs médians |
| ---: | --- | ---: | ---: | ---: | ---: |
| 50 | r3 | 6 | 0,595 | 0,538 | 0,908 |
| 50 | r6 | 6 | 0,778 | 0,908 | 0,908 |
| 75 | r3 | 4 | 0,604 | 1,000 | 1,000 |
| 75 | r6 | 4 | 0,734 | 0,862 | 1,000 |

À 50 réponses, le rang médian top 4 retrouve la LoRA 0,3 dans 5 regroupements sur 6 pour les deux modèles. Le MCC chute à **0,734** en top 3 et **0,721** en top 5. Les regroupements partagent des blocs. Les références avaient déjà servi au développement des lecteurs, et le nombre de variantes y est connu. Le calcul des scores bruts s'appuie sur les cinq copies de référence identifiées par le corrigé, utilisées comme ancres de similarité dans `c2courbes.py`. Les MCC mesurent donc la stabilité conditionnelle à ces ancres connues.

Pour transférer sur les données de compétition, il faut d'abord sélectionner les ancres sans le corrigé et valider cette sélection. Ensuite, le rang par bloc peut servir de contrôle sur les sorties du cycle 2. Conserver les deux rangs individuels, leur étendue et l’accord avec les lecteurs de vraisemblance. La décision dépend de cet accord et de preuves indépendantes sur le nombre de variantes. Sur les guichets ambigus, la méthode est un moyen de garder trace des désaccords sans consommation de requêtes. [Branches et discussion T2](https://github.com/adembnn/EAC/issues/4)

## Contrôle sans nombre imposé de variantes

J'ai testé une partition exacte des matrices de similarité du détecteur de Titouan, en maximisant la somme des arêtes internes centrées. Deux pondérations, probabilité brute ou log-odds, et quatre seuils, de 0,10 à 0,50, donnent huit variantes de décodage. Ce calcul impose la transitivité et laisse libre la taille de chaque groupe. Les étiquettes de référence servent uniquement au calcul du MCC de ce second contrôle.

Au premier tour sur références, la partition exacte maintient le MCC de **0,721** obtenu par lien complet. Sur les tours suivants, les configurations testées varient de **0,721 à 1,000**, alors que les trois partitions publiées par lien complet rendent **1,000**. En particulier, l'optimisation peut rattacher le fp8 à la base (tour 3) ou la LoRA 0,3 (tours 2 et 4). Cette méthode sert de test de sensibilité et n'améliore pas le décodeur publié.

Sur la matrice du bloc 0 des 11 endpoints Llama, `ep-15` est isolé dans **8/8** configurations. `ep-05` et `ep-19` sont isolés dans **7/8**, `ep-10` dans **6/8**, `ep-13` dans **4/8**. `ep-09` et `ep-11` restent dans la base dans **8/8**, en conflit avec le signal du lecteur directionnel sur bloc 0. Ces huit réglages exploitent la même matrice et mesurent la sensibilité au décodage. Les scores du bloc 1 et le lecteur directionnel font la différence entre hypothèses sur les poids. [Matrices et décodeur de Titouan](https://github.com/adembnn/EAC/tree/t2/detecteur-appris/tracks/t2-model-equality/detecteur)

## Contrôle des réponses du cycle 2

Le bloc 1 partagé contient **825 réponses de compétition** et **223 réponses de référence**. Le reçu **2707** maintient le score officiel à **0,60**. Les lecteurs d'Adem analysent les deux blocs à 50 réponses par guichet. [Dépôt du bloc 1](https://github.com/adembnn/EAC/issues/4#issuecomment-7072885065) · [Reçu 2707](https://github.com/adembnn/EAC/issues/4#issuecomment-7072934345)

Contrôle CPU sur **1 514 réponses ordinaires de compétition** et **412 réponses ordinaires de référence**. Centrer par question, puis mesurer la déviation de style avec neuf caractéristiques simples (longueur, mise en forme, pronoms, introduction). Les questions du bloc 1 sont distinctes. La séparation des variantes connues donne une AUC de **0,60** au bloc 0 et **0,65** au bloc 1 (test exact de permutation, **p = 0,278** au bloc 1). La corrélation des rangs entre blocs vaut **0,045** pour Llama, **0,091** pour Qwen-1.5B et **0,618** pour Qwen-3B.

Les profils de style d'`ep-16` et `ep-29` persistent entre blocs (cosinus **0,83** et **0,74**), cohérents avec les nuisances persona et prompt système déjà relevées. `ep-02` et `ep-23` figurent parmi les trois déviations les plus fortes pour Qwen-3B dans les deux blocs. Le contrôle ne fournit pas de décision sur l'égalité des poids. Aucun appel au serveur ni soumission pour ces contrôles.
