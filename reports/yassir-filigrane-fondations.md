# Filigrane : ce que la littérature change pour nos trois tracks (agent de Yassir)

**Décision débloquée : on sait où le filigrane survit et où il meurt, donc on sait quoi mesurer
dans T1 et quoi ne pas dépenser dans T2.** Fondé sur 24 papiers de Pierre Fernandez (Meta FAIR) et
12 sources qu'il cite, lus en texte intégral. Aucun chiffre de ce rapport n'est une estimation.

## Pourquoi Fernandez est là, et ce que ça dit du jury

Il a fait la keynote. Ses papiers sont dans l'état de l'art que le kit utilise (PixelSeal est le sien).
Les organisateurs sont du camp offensif, CISPA et SprintML. Lui est du camp défensif. Toute la
littérature qu'il cite mesure la robustesse à des transformations classiques, pas la résistance à un
attaquant qui a le gradient. C'est exactement la distinction qu'il pose lui-même, robustesse contre
sécurité, et c'est le vocabulaire à employer si un jury nous interroge dimanche.

## Les cinq chiffres qui changent nos choix

1. **VINE (ICLR 2025)** mesure les filigranes post-hoc de type latent à **1,76 % de TPR** à 0,1 % de
   faux positifs après régénération stochastique de l'image, pour 2,19 s par image. Autrement dit :
   dès qu'une zone est regénérée par un modèle, le filigrane d'origine disparaît. C'est le principe
   même de T1.
2. **InvisMark (Microsoft, WACV 2025)** transplante un résidu d'une image à une autre et récupère
   **97,6 % des bits sur une image jamais filigranée**, contre 67 % pour TrustMark. Un filigrane sans
   dépendance au contenu prouve la présence d'un marqueur, pas l'authenticité de l'image.
3. **TrustMark** s'entraîne sur un modèle de bruit à 18 transformations. C'est le banc sur lequel
   PixelSeal se compare, donc les zones qu'on nous demande de trouver sont celles que ces 18
   transformations préservent.
4. **WOUAF** chiffre le coût de l'attribution : 8,4 h pour retrouver un utilisateur parmi N, 1 minute
   pour un détecteur public, 1 seconde pour une extraction directe.
5. **Cayre, Fontaine, Furon (2005)** : la sécurité se mesure en nombre d'images nécessaires à
   l'attaque, pas en résistance à une attaque donnée. Aucun papier de Fernandez ne donne cette
   mesure, sauf le sien de 2024 sur les attaques de copie et de retrait.

## Ce que ça implique pour T1

Les trois méthodes du kit sont post-hoc et **globales** : elles décodent un message par image
entière. La consigne le dit elle-même, la question de la tâche est justement de savoir si on peut
localiser un filigrane global. Conséquence pratique : nos masques sont les zones où le décodeur
réussit encore à lire les bits du message, pas les zones où l'image a été modifiée.

L'ordre des retouches par difficulté attendue, du plus facile au plus dur :
1. **outpainting** : le filigrane est intact sur la zone d'origine, la zone ajoutée est propre. La
   frontière est franche, le décodage par fenêtre doit la trouver nette.
2. **retrait d'objet** : la zone retouchée est petite, souvent intérieure. Le risque est de la
   manquer, pas de la confondre.
3. **greffe d'objet** : le filigrane est présent sur l'objet collé seulement, donc une petite zone
   isolée au milieu d'une image propre. C'est le cas le plus dur au décodage par fenêtre, parce que
   la fenêtre contient beaucoup de pixels propres qui moyennent les logits vers 50 %.
4. **altération locale** : dépend de l'ampleur, c'est le cas où le segmentateur supervisé doit
   prendre le relais du décodeur.

D'où la stratégie en deux étages, et non un seul : décodage pour la sensibilité, segmentateur
entraîné sur les 510 masques connus pour la précision locale. La greffe d'objet est précisément le
cas où le segmentateur vaut mieux que le décodeur.

## Ce que ça implique pour T2

La détection de filigrane de texte est un signal d'identité utilisable : c'est le papier radioactive
(Fernandez 2024) et le papier de détection publique. Si deux endpoints servent le même modèle avec
des filigranes différents, un détecteur tranche. Mais TextSeal et SynthID-Text exigent des réponses
longues pour être fiables, et le budget est de 100 réponses par endpoint : il faut donc des prompts
qui produisent du texte long et répétitif, pas des réponses d'un mot.

Les 9 références couvrent déjà les six axes annoncés (base, quantification, LoRA, watermark, prompt
système, décodage), et `practice_answers.csv` donne la vérité. Régler la décision là-dessus avant de
toucher aux 33 endpoints reste la seule façon de ne pas brûler un budget irréversible.

## Ce que ça implique pour T3

Le papier radioactivity dit qu'un modèle qui a vu des données garde une signature détectable même
après désapprentissage partiel : c'est exactement ce que la composante de prédiction du score mesure,
image par image, contre des modèles gold réentraînés. Donc viser « identique au gold », pas « ne sait
plus rien ». Une montée de gradient nue rend le modèle inutilisable sur ces images et fait perdre les
deux termes.

## Les trois contradictions à connaître si le jury creuse

- VINE contredit directement les filigranes post-hoc de Fernandez : 1,76 % de TPR après régénération.
  Il l'admet dans sa keynote et renvoie vers la famille sémantique.
- InvisMark montre qu'un résidu se transplante : 97,6 % de bits sur une image jamais marquée.
- WOUAF contre l'extraction directe : 8,4 h contre 1 seconde. Le débat sur les détecteurs publics est
  un débat de coût, pas de faisabilité.

## Non vérifié

- Le code des trois baselines (`tracks/*/`) n'a jamais été exécuté sur JURECA ni sur GPU. Il est relu,
  linté, et la logique de T2 est testée sur des réponses synthétiques. À traiter comme point de
  départ.
- Aucune de nos méthodes n'a encore de score : la clé API du leaderboard manque à toute l'équipe.
