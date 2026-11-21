# Tâche 1 : ce que dit la communauté, comment combiner les méthodes, comment juger

Date : 21 nov. 2026. Débloque : quelles méthodes on code pour la tâche 1 (échéance dim. 22 nov. 12 h) et sur quelle règle on tranche entre elles.
Légende : [VU] = lu pendant cette passe (page ouverte, lien donné). [REPRIS] = lu dans le rapport précédent `t1-watermark-etat-de-lart.md`, pas relu ici. [IDÉE] = notre déduction, jamais mesurée.
Les méthodes A à G sont celles du rapport précédent (A boîte exacte, B corrélation avec l'embedder, C petit réseau, D fenêtres, E carte brute du décodeur, F détecteur de boîte TrustMark, G bases).

## 0. L'essentiel en 8 lignes

1. Il n'existe pas de consensus. Aucune source communautaire ne localise PixelSeal, TrustMark ou StegaStamp après coup (section 1).
2. Reddit est bloqué par l'outil Chrome ; X ne renvoie rien d'utile ; les dépôts GitHub des quatre filigranes ne contiennent aucune discussion de localisation (section 1).
3. Le cas le plus proche, publié par des équipes des mêmes organisateurs : la tâche « localiser un filigrane de texte » de la finale de juillet 2026. Les meilleurs ont utilisé : les clés exactes, un score comparable d'un document à l'autre, un lissage en segments (modèle de Markov caché), et une règle écrite avant de soumettre (section 1.3). On peut transposer ça à l'image.
4. A, E, F, B, C ne sont pas des rivales : ce sont des étages d'une même chaîne (section 3).
5. Chaîne recommandée : F/E donnent un départ, A fixe la boîte, E/B trouvent les trous, C fusionne et apprend les bords, puis lissage. Routage par filigrane (section 3.2).
6. Juger : validation croisée en 5 plis par filigrane, scores par type d'édition, comparaison appariée sur les mêmes images. Un écart de moins de 0,03 par filigrane est du bruit (section 4).
7. Alerte : la page du jeu de données Hugging Face indique « v2.1 » publiée la veille ; elle donne 25 Mo de limite de soumission, le sujet dit 20 Mo. Vérifier qu'on a la bonne version (section 1.4).
8. Ordre conseillé inchangé : G, expérience 0, A, puis C si A laisse des trous mal rendus.

## 1. Sources communautaires : ce qui a été cherché et trouvé

### 1.1 Ce qui n'a pas marché (dit sans détour)
- Reddit : la navigation vers reddit.com/search est refusée par l'outil (« This site is not allowed due to safety restrictions ») [VU]. Rien lu. Pas de contournement tenté.
- X, recherche « PixelSeal » : un seul résultat, une publicité qui dit qu'une API vidéo pose une marque PixelSeal invisible [VU, https://x.com/search?q=PixelSeal&f=live]. « "Pixel Seal" watermark », « TrustMark watermark », « CISPA hackathon watermark » : « No results » [VU]. « watermark localization » : seulement du bruit publicitaire [VU]. Je n'ai pas fait les autres requêtes (StegaStamp, videoseal pixelwise, SprintML…) : les cinq premières étaient vides, j'ai arrêté pour ne pas gaspiller.
- Profil de Pierre Fernandez (auteur de WAM et PixelSeal) : dernier message du 7 juil. sur un outil de détection de filigrane chez Meta, rien sur la localisation [VU, https://x.com/pierrefdz]. Le compte « TomSander_ » n'existe pas à cette adresse [VU].
- Hugging Face : modèles « watermark localization » = 0 résultat ; espaces = un seul, un retireur de filigrane SynthID, hors sujet [VU, https://huggingface.co/models?search=watermark%20localization].
- Zhihu, CSDN, Qiita : non faits (le reste a rendu quasiment rien, le temps était mieux employé sur les dépôts).
- Aucun dépôt GitHub public ne cite « Localize Image Watermark », PixelSeal+TrustMark+StegaStamp ensemble, ou la tâche 30 de SprintML pour des images [VU, `gh search repos`, recherche vide].

### 1.2 GitHub : les trois filigranes et WAM
- facebookresearch/videoseal (PixelSeal) : 50 tickets parcourus, aucun sur la localisation, le masque ou le rognage. Le plus proche : #94 « motif en blob basse fréquence visible sur les zones lisses sombres en haute résolution » [VU, https://github.com/facebookresearch/videoseal/issues/94]. Intérêt pour nous : le filigrane est visible/plus fort en zone lisse sombre, pas l'inverse de ce que dit notre rapport précédent (zones lisses = marge faible). [IDÉE] À mesurer : la marge par texture, dans les deux sens.
- adobe/trustmark : 17 tickets. Aucun sur la localisation. #27 (pourquoi limiter le ratio d'aspect en mode P, pourquoi ne pas redimensionner le résidu à la taille de l'image) est resté sans réponse [VU, https://github.com/adobe/trustmark/issues/27]. #37 : le décodeur Rust perd des filigranes quand le redimensionnement se fait sans anti-crénelage [VU, https://github.com/adobe/trustmark/issues/37] : le redimensionnement du rognage (A) doit imiter exactement celui du wrapper du kit.
- tancik/StegaStamp : les tickets #5, #20, #26 donnent la recette du détecteur d'origine : des StegaStamps de 400x400 px, tirés avec échelle aléatoire, collés sur des images haute résolution DIV2K ; le détecteur marche mieux avec un bord clair ; le décodeur était entraîné avec bords blancs, le détecteur sans [VU, https://github.com/tancik/StegaStamp/issues/5 et /20]. Le code d'entraînement du détecteur n'est pas publié (tickets #20, #42 ouverts). Intérêt : même recette que notre méthode C sur données synthétiques (collage au hasard), donc C a un précédent pour StegaStamp.
- facebookresearch/watermark-anything (WAM) : #5 (plusieurs messages) : Pierre Fernandez répond « faire passer 4 messages avec 4 masques » ; #7 : le regroupement DBSCAN bloque au-dessus d'environ 40 % de pixels [VU, https://github.com/facebookresearch/watermark-anything/issues/7]. Leçon : la localisation de WAM passe par un canal « filigrané ? » appris par pixel, pas par un simple décodage ; rien à reprendre tel quel.
- Aucun ticket ne dit qu'un détecteur de ces filigranes donne un résultat fiable sur un rognage partiel : c'est notre hypothèse A [IDÉE], toujours à confirmer par l'expérience 0.

### 1.3 Éditions précédentes du même organisateur (le plus utile)
Trois dépôts publics d'anciens participants, tous du championnat CISPA/SprintML :
- Vienne, 13-14 déc. 2025 : dataset inference, attribution de texte, retrait de filigrane image BitMark par inpainting + diffusion + flou [VU, https://github.com/florianzainzinger1/cispa-hackathon-championship-vienna]. Pas de localisation.
- Finale, Sarrebruck, 2-3 juil. 2026 : « Text Watermark Localization » (quatre schémas de texte), métrique TPR à 0,1 % de faux positifs [VU, https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026 et https://github.com/andresmartinezcarcel/cispa-cybersecurity-ai-hackathon-finale]. Scores publics : l'équipe 4e atteint 0,424 en rang 2 sur cette tâche ; l'équipe IPCVictory ≈ 0,127 [VU]. Un écart de 3 à 4 fois entre équipes : la méthode compte énormément, et la tâche image sera probablement aussi étalée.
- Ce que l'équipe 4e a fait, transposable [VU, notes dans https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026/tree/main/task1/reports] :
  1. Elle a les clés exactes de chaque schéma, donc ce n'est pas de la détection à l'aveugle mais de la « reconstruction de signal ». Pour nous : on a le message caché connu ; la marge de A est le même objet.
  2. Elle convertit chaque signal en p-valeur sous l'hypothèse « texte propre », parce que le score est pooled sur tous les tokens : des scores comparables d'un document à l'autre par construction. Pour nous : [IDÉE] convertir chaque carte (A, E, B) en z-score contre des messages faux tirés au hasard (le « témoin » de B), pour pouvoir les moyenner ou les comparer entre images.
  3. Elle lisse avec un modèle de Markov caché (HMM) puis un modèle à durée explicite (HSMM) : le HMM dit « dans un segment filigrané, il y a 95 % de chances que le token suivant le soit aussi ». Pour nous : [IDÉE] l'équivalent image est un modèle de champ de Markov / coupe de graphe sur la grille 128x128, qui lisse la carte de cellules.
  4. Elle a écrit une règle « ship gate » avant de soumettre : une variante ne part que si elle gagne sur les moitiés de validation et sur les partages inversés, pour ne pas s'adapter aux 30 % du classement public [VU]. Pour nous : section 4.4.
  5. Elle a mesuré une borne physique : un segment ne peut être trouvé que s'il apporte assez de preuve (≈ 12 nats, un « nat » = unité d'information) ; 37 % de la masse filigranée était hors d'atteinte quel que soit le modèle [VU, rapport 15_full_audit]. Pour nous : [IDÉE] certaines zones filigranées (petites, très éditées, lisses) auront une marge trop faible pour être trouvées par A ou E ; il faut mesurer ce plafond (section 4.3, « oracle ») avant de s'acharner sur C.
  6. Écart local/public observé chez eux : 0,438 en validation, 0,374 en public [VU, même rapport]. Ne pas s'attendre à ce que le public reflète le local.
  7. Une piste de « bord net » qu'ils croyaient valoir +0,068 s'est révélée un artefact d'oracle : ils l'ont mesurée avant de l'implémenter [VU]. À copier : avant de coder une amélioration, calculer son gain maximal avec les vrais masques (section 4.3).
- Autre leçon sur les filigranes image : dans « Erasing the Invisible » (NeurIPS 2024), une équipe a surécrit le filigrane StegaStamp en ré-encodant le message inversé avec l'encodeur d'origine, et obtenu 0,5 de détection [VU, https://github.com/leiluk1/erasing-the-invisible-beige-box]. Pour nous : les éditions « ajout d'objet » qui seraient faites par réencodage pourraient laisser un message inversé ou différent. [IDÉE] Tester sur les trous : la marge est-elle négative (message inversé) ou autour de 0 (bruit) ? Si négative, c'est un signal de plus.

### 1.4 La page du jeu de données (à lire avant tout)
- Page du dépôt Hugging Face : « Watermark localisation (images) v2.1: train, test, starter kit », publiée environ 14 h avant ma lecture ; train.zip 191 Mo, test.zip 379 Mo, starter_kit.zip 10 Ko [VU, https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/tree/main].
- Son README.txt dit 25 Mo maximum pour le fichier de soumission et masques de forme [1020,128,128] ; le sujet que nous avons dit 20 Mo [VU, README.txt du même dépôt]. Un tenseur booléen fait environ 17 Mo : sans importance, mais re-télécharger si l'équipe a pris une version plus ancienne (le test a pu changer).
- La section « Community » du dépôt est vide : personne n'a posé de question publique [VU].

## 2. Les sept méthodes, en une image chacune

| | Image mentale | Ce que ça sort | Aveugle à |
|---|---|---|---|
| A. Boîte exacte | On essaie des cadres sur l'image jusqu'à ce que le décodeur retrouve notre message : c'est le « mot de passe » qui dit où est la porte. | Une boîte (4 coordonnées), puis des trous si on creuse | Les zones non rectangulaires (transplantation) ; les images où le message ne se retrouve plus du tout |
| B. Corrélation embedder | On demande à l'embedder « où voudrais-tu mettre ton motif ? » et on regarde où l'image contient déjà ce motif. | Carte fine au pixel, bruitée | Zones lisses (peu de haute fréquence) ; dépend du contenu |
| C. Petit réseau | Un élève qui regarde l'image et toutes les cartes et apprend à dessiner le masque. | Un masque complet | Ce qu'il n'a pas vu à l'entraînement (écart synthétique/réel) |
| D. Fenêtres alignées | Une loupe qu'on promène et qui dit à chaque endroit « filigrane / pas filigrane ». | Carte grossière par cellule | Petits objets ; fenêtres trop petites pour porter 100 à 256 bits |
| E. Carte brute du décodeur | On lit les « votes » internes du décodeur case par case (8x8). | Carte de 8x8 cases floue | Détails fins ; StegaStamp (pas de votes par case dans le kit) |
| F. Détecteur de boîte TrustMark | Un détecteur d'objets entraîné par Adobe à repérer le rectangle filigrané. | Une boîte | Tout ce qui n'est pas TrustMark ; trous ; dépend d'un entraînement hors de notre distribution |
| G. Bases | Tout 0, tout 1, rectangle central, masque médian. | Un masque fixe | Tout : sert de plancher |

Une phrase à retenir : A et F donnent des boîtes, E et D des cartes grossières, B une carte fine, C un masque. Seul C produit directement la forme demandée. Les autres sont des entrées ou des vérifications de C.

## 3. Une chaîne, pas sept concurrentes

### 3.1 Qui sert à quoi (étages)
1. Départ : F (TrustMark), E (PixelSeal, TrustMark) ou D (si l'expérience 0 dit « image entière ») proposent une zone grossière. Rôle : réduire le nombre de boîtes à tester pour A.
2. Boîte : A raffine ce départ en boîte exacte et prouve statistiquement que la zone est filigranée.
3. Trous : E (cases du décodeur dans la boîte), B (carte fine) et la marge de fenêtres alignées trouvent où le message disparaît.
4. Fusion : C reçoit tout (image, carte de boîte, E, B) et apprend les bords et les formes ; il corrige aussi les erreurs de A.
5. Lissage et décision : coupe de graphe ou règles simples (retirer les petites composantes, remplir les petits trous, décider tout 0 / tout 1). Seuils réglés par pli.
G sert de plancher et d'essai de la chaîne de soumission. Tout est [IDÉE] : aucune source publiée ne décrit cette chaîne pour ces filigranes ; chaque étage a un précédent séparé (section 1.2 et rapport précédent).

### 3.2 Trois chaînes concrètes
Chaîne 1, « boîte puis trous » (sans entraînement, la plus rapide) : F ou E/D → A → E dans la boîte → post-traitement. Un seuil par filigrane. Gain attendu fort sur les éditions « agrandissement » (la boîte suffit) ; faible sur « ajout d'objet » fin.
Chaîne 2, « tout-en-un appris » : calcule A, E, B, D une fois par image, empile en canaux, C produit le masque ; A n'est qu'un canal. Gain attendu plus fort sur les trous irréguliers et la transplantation ; risque : écart entre données synthétiques et vraies retouches génératives.
Chaîne 3, « routage par filigrane et par confiance » (ma recommandation [IDÉE]) :
- TrustMark : F + A (boîte quasi compacte, ratio 2:1), puis trous par E, puis post-traitement ; C seulement si les trous coûtent.
- PixelSeal : A + E + B, puis C (le test de 256 bits est le plus net, c'est là que A est le plus fiable).
- StegaStamp : A + D + B (pas de votes par case dans le kit), puis C ; budget de candidats réduit (ONNX, plus lent).
- Par image : si la marge de A dépasse un seuil (message bien retrouvé), chaîne 1 ; sinon chaîne 2 (repli sur C). Le seuil se règle par filigrane sur les plis.

### 3.3 Comment combiner les cartes : trois options
- Empiler comme canaux d'entrée de C : le réseau apprend lui-même les poids. Avantage : rien à calibrer. Inconvénient : consomme les 170 images + données synthétiques ; nécessite la section 4 pour ne pas surapprendre.
- Moyenner des cartes : possible seulement si chaque carte est exprimée dans la même unité. Recette empruntée à l'équipe 4e : z-score contre le témoin des messages faux (section 1.3, point 2) [IDÉE]. Peu coûteux, sans entraînement, mais ignore les interactions.
- Choisir par image selon la confiance : une règle simple (marge de A, accord entre E et A) route vers la chaîne 1 ou 2. Peu de paramètres, donc peu de surapprentissage.
Ordre d'essai conseillé : choisir par image (3 paramètres) → moyenne de z-scores → empilement dans C. On ne passe à l'option suivante que si l'écart mesuré dépasse le seuil de la section 4.2.

## 4. Comment on juge « meilleure »

### 4.1 Protocole local exact (510 images train)
1. Données : 170 images par filigrane. Plis : 5 plis par filigrane, tirés au hasard avec graine fixe, stratifiés par type d'édition (étape 3). Même découpage pour toutes les méthodes (sinon comparaison faussée).
2. Score : `miou_image` et le score de `wmloc/utils.py` du kit, jamais une formule refaite à la main. Les masques sont d'abord passés par `downsample_mask` du kit pour que le calcul corresponde au serveur [VU, enoncé : « the starter kit contains the exact downsampling function »].
3. Type d'édition : le kit ne donne que `full_watermark`, le masque 512 et 128, le filigrane et le message [VU, README.txt du jeu de données]. Si aucune étiquette de type n'existe, la dériver du masque [IDÉE] : fraction de surface filigranée, nombre de composantes connexes, trous (agrandissement : un rectangle ; ajout/retrait : rectangle avec trous ; transplantation : forme d'objet sans rectangle) et `full_watermark`. Regarder 30 images pour vérifier la règle.
4. Rapports à produire pour chaque méthode : (a) le tableau global ; (b) le tableau par filigrane (3 lignes) ; (c) le tableau par type d'édition x filigrane ; (d) la distribution du score par image (beaucoup de 1,0 attendus, voir 4.2) ; (e) un fichier de 20 pires images à regarder.
5. Calibrage sans fuite : tout seuil, post-traitement, nombre de départs de A est réglé sur 4 plis et évalué sur le 5e (validation imbriquée). Règler et évaluer sur les mêmes images gonfle le score.

### 4.2 Quel écart compte, avec 170 images par filigrane
Calcul d'ordre de grandeur [IDÉE, hypothèse non mesurée : écart-type du score par image ≈ 0,25 ; à mesurer sur G dès la première heure] :
- Erreur de la moyenne : 0,25 / racine(170) ≈ 0,019 par filigrane ; 0,25 / racine(510) ≈ 0,011 pour l'ensemble.
- Comparaison de deux méthodes sur les mêmes images : si leurs scores par image sont corrélés à 0,7, l'écart-type de la différence ≈ 0,19, donc erreur ≈ 0,015 par filigrane, 0,008 global.
- Règle pratique : un écart inférieur à 0,03 par filigrane (ou 0,015 global) est du bruit ; entre 0,03 et 0,05 on demande une confirmation ; au-dessus de 0,05 on y croit. Remplacer ces chiffres par la mesure dès qu'on a les scores par image (bootstrap appariée : 1 000 tirages de 170 images avec remise, on lit l'intervalle à 95 % de la différence ; on exige qu'il exclue 0).
- Critère de robustesse en plus : la méthode doit gagner dans au moins 4 des 5 plis et ne pas perdre sur un type d'édition de plus de 0,05.
- Attention à la forme du score : la classe vide vaut 1. Une image « tout propre » bien prédite rapporte 1,0 ; une seule cellule fausse la ramène à 0,5 [rapport précédent, section 6]. La distribution est donc en bosses ; la moyenne dépend beaucoup de quelques décisions « tout 0 / tout 1 ». Les rapporter à part.

### 4.3 Ablations (chaque ligne = un tableau de 4.1)
Colonnes : global, par filigrane, par type d'édition.
1. G (plancher) ; 2. E seul ; 3. F seul (TrustMark) ; 4. A seul (boîte rectangle, sans trous) ; 5. A + trous par E ; 6. A + trous par E + B ; 7. C seul sur image + E ; 8. A + C (A en canal) ; 9. tout + lissage ; 10. routage par confiance.
Deux lignes « oracle » indispensables [IDÉE, calquées sur ce qu'a fait l'équipe 4e, section 1.3] :
- Oracle de boîte : donner à la chaîne la vraie boîte (calculée depuis le masque). Mesure le score maximal d'une chaîne « boîte puis trous » : si l'oracle plafonne à 0,85, ne cherchons pas mieux que ça avec A.
- Oracle de trous : vrai masque sauf un bruit simulé de 1 cellule de bord. Mesure le plafond de la métrique à cause de la résolution.
Si une amélioration n'ajoute rien à l'oracle correspondant, ne pas la coder.

### 4.4 Ne pas s'adapter au classement public (30 % du test, ≈ 306 images)
- Erreur du public : 0,25 / racine(306) ≈ 0,014 : deux soumissions à 0,02 d'écart sur le public ne se distinguent pas. L'équipe 4e a mesuré 0,438 en local contre 0,374 en public sur son propre modèle [VU, 15_full_audit].
- Règle écrite maintenant, avant la première soumission (copiée de la « ship gate » de l'équipe 4e) [IDÉE de transposition] :
  1. On choisit la méthode et ses seuils sur le score local en 5 plis seul.
  2. Une soumission n'est lancée que si le score local dépasse celui de la précédente de plus que le seuil de 4.2.
  3. Le public ne sert qu'à détecter une panne (soumission invalide, inversion 0/1, résolution) : on exige simplement qu'il soit dans l'erreur de ± 0,03 du local. S'il est bien plus bas, chercher la cause (écart de distribution test/train) avant de changer de méthode.
  4. On ne règle jamais un seuil sur le public. Le classement ne garde que le meilleur de l'équipe [VU, sujet], donc une soumission « pour tester » ne coûte rien en score mais ne doit pas guider le choix.
  5. Garder une soumission « bord de sécurité » prête (G ou la chaîne 1) en cas de panne de dernière minute.
- Le classement privé = 70 % restants : c'est lui qui compte. Le choix final entre A seul, C seul, A + C se fait sur 510 images locales (erreur ≈ 0,011), plus fiable que sur les 306 du public.

## 5. Y a-t-il un consensus ?

Non. Ce que j'ai vérifié :
- Aucune page lue (GitHub, Hugging Face, X, profils des auteurs) ne décrit comment localiser PixelSeal, TrustMark ou StegaStamp en post-hoc [VU, section 1].
- La littérature citée dans le rapport précédent donne des briques sans recette commune : WAM entraîne le filigrane lui-même à segmenter, MaskMark et ReliaGuard ajoutent un réseau de localisation, TrustMark livre un détecteur de boîte, StegaStamp avait un détecteur de collage appris [REPRIS ; WAM : https://arxiv.org/abs/2411.07231 ; TrustMark : https://arxiv.org/abs/2311.18297 ; StegaStamp : https://arxiv.org/abs/1904.05343 ; MaskMark : https://arxiv.org/abs/2504.12739].
- Le seul terrain d'expérience publié dans le même cadre (même organisateur, tâche de texte) montre que les gagnants ont convergé sur « signal exact + score calibré + lissage en segments + garde-fou de soumission », pas sur un modèle appris massif. C'est une analogie, pas une preuve pour des images [VU + IDÉE].
- Ce qui nous distingue : nous avons les trois décodeurs, les trois embedders et le message exact. Peu de gens publient avec ça. Il y a donc peu de chances qu'une solution « standard » existe à copier : on peut s'attendre à un classement très étalé (voir 1.3, 0,127 contre 0,424 sur la tâche texte).

## 6. Recommandation et plan en 6 lignes

1. Dès maintenant : télécharger la version v2.1 du jeu, vérifier les hashes si l'équipe en avait une autre (1.4).
2. 0 h 30 : G + expérience 0 (rapport précédent) + mesure de l'écart-type par image (4.2).
3. 3 h : chaîne 1 sur PixelSeal et TrustMark (A + E + F) avec ablations 1 à 6 et les deux oracles.
4. 6 h : A sur StegaStamp ; lancer les données synthétiques pour C ; mesurer l'oracle de boîte.
5. 14 h : C par filigrane, plis, test de l'option « moyenne de z-scores » et du routage par confiance.
6. 17 h : choix final par la règle 4.4, une ou deux soumissions.

Option la plus forte écartée : affiner le décodeur lui-même comme segmenteur à la WAM (plus fort en théorie, mais 10 à 14 h par filigrane et StegaStamp en ONNX non entraînable ; voir rapport précédent).

## 7. Liens (tous ouverts pendant cette passe sauf mention)
- https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026 (finale 2026, texte, rang 4)
- https://github.com/andresmartinezcarcel/cispa-cybersecurity-ai-hackathon-finale (finale 2026, IPCVictory)
- https://github.com/florianzainzinger1/cispa-hackathon-championship-vienna (Vienne déc. 2025)
- https://github.com/leiluk1/erasing-the-invisible-beige-box (NeurIPS 2024, surécriture StegaStamp)
- https://github.com/facebookresearch/videoseal/issues/94 ; https://github.com/adobe/trustmark/issues/27 ; https://github.com/adobe/trustmark/issues/37 ; https://github.com/tancik/StegaStamp/issues/5 ; https://github.com/tancik/StegaStamp/issues/20 ; https://github.com/facebookresearch/watermark-anything/issues/7
- https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/tree/main (v2.1, README.txt)
- https://x.com/search?q=PixelSeal&f=live ; https://x.com/pierrefdz
Aucune soumission lancée, aucun appel payant, aucun appel à l'API du hackathon. Trois onglets Chrome ouverts par moi ont été fermés ; les autres onglets n'ont pas été touchés.
